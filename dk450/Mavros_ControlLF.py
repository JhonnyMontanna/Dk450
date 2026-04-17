#!/usr/bin/env python3
"""
leader_follower_node.py — PID líder-seguidor con prealimentación
=================================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.
Migración de MavlinkControlLF2.py.

Ley de control (planteamiento matemático):
    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L − v_S)
    FF    = ψ̇_L · Rz(π/2) · d(t)
    d(t)  = [d·cos(ψ_L+α), d·sin(ψ_L+α), Δz]   (offset polar rotante)
    r_cmd = Kp_ψ·e_ψ + Ki_ψ·∫e_ψ + Kd_ψ·(ψ̇_L − ψ̇_S)

Parámetros ROS2:
  leader_ns       : namespace del líder           (default: 'uav1')
  follower_ns     : namespace del seguidor        (default: 'uav2')
  offset_d        : distancia de separación [m]   (default: 2.0)
  offset_alpha    : ángulo de formación [rad]     (default: π = detrás)
  offset_dz       : diferencia de altitud [m]     (default: 0.0)
  kp / ki / kd    : ganancias PID posición        (default: 0.5/0.05/0.2)
  kp_yaw/ki_yaw/kd_yaw: ganancias PID yaw        (default: 0.8/0.0/0.1)
  v_max           : velocidad máxima cmd [m/s]    (default: 3.0)
  yaw_rate_max    : yaw_rate máximo [rad/s]       (default: 1.0)
  integral_limit  : anti-windup posición [m·s]    (default: 2.0)
  integral_yaw_lim: anti-windup yaw [rad·s]       (default: 1.0)
  rate            : frecuencia de control [Hz]    (default: 20)

Tópicos:
  Sub: /<leader_ns>/mavros/local_position/pose            (PoseStamped)
  Sub: /<leader_ns>/mavros/local_position/velocity_local  (TwistStamped)
  Sub: /<follower_ns>/mavros/local_position/pose          (PoseStamped)
  Sub: /<follower_ns>/mavros/local_position/velocity_local(TwistStamped)
  Pub: /<follower_ns>/mavros/setpoint_velocity/cmd_vel    (TwistStamped)
"""

import math
import csv
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped

# ─────────────────────────────────────────────────────────────────────────────
# QoS
# ─────────────────────────────────────────────────────────────────────────────
_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (inline)
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_yaw(q) -> float:
    """Extrae yaw (rad) de quaternion ROS. ENU: yaw=0 → este."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )

def _wrap(theta: float) -> float:
    """Proyecta θ a (−π, π]."""
    return math.atan2(math.sin(theta), math.cos(theta))

def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))

def _make_twist(node, vx, vy, vz, yr) -> TwistStamped:
    msg = TwistStamped()
    msg.header.stamp    = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.twist.linear.x  = float(vx)
    msg.twist.linear.y  = float(vy)
    msg.twist.linear.z  = float(vz)
    msg.twist.angular.z = float(yr)
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Nodo
# ─────────────────────────────────────────────────────────────────────────────
class LeaderFollowerNode(Node):

    def __init__(self):
        super().__init__('leader_follower_node')

        self.declare_parameter('leader_ns',        'uav1')
        self.declare_parameter('follower_ns',      'uav2')
        self.declare_parameter('offset_d',          2.0)
        self.declare_parameter('offset_alpha',      math.pi)
        self.declare_parameter('offset_dz',         0.0)
        self.declare_parameter('kp',                0.5)
        self.declare_parameter('ki',                0.05)
        self.declare_parameter('kd',                0.2)
        self.declare_parameter('kp_yaw',            0.8)
        self.declare_parameter('ki_yaw',            0.0)
        self.declare_parameter('kd_yaw',            0.1)
        self.declare_parameter('v_max',             3.0)
        self.declare_parameter('yaw_rate_max',      1.0)
        self.declare_parameter('integral_limit',    2.0)
        self.declare_parameter('integral_yaw_lim',  1.0)
        self.declare_parameter('rate',             20)

        L_ns       = self.get_parameter('leader_ns').value
        F_ns       = self.get_parameter('follower_ns').value
        self.d     = self.get_parameter('offset_d').value
        self.alpha = self.get_parameter('offset_alpha').value
        self.dz    = self.get_parameter('offset_dz').value
        self.kp    = self.get_parameter('kp').value
        self.ki    = self.get_parameter('ki').value
        self.kd    = self.get_parameter('kd').value
        self.kp_y  = self.get_parameter('kp_yaw').value
        self.ki_y  = self.get_parameter('ki_yaw').value
        self.kd_y  = self.get_parameter('kd_yaw').value
        self.vmax  = self.get_parameter('v_max').value
        self.yrmax = self.get_parameter('yaw_rate_max').value
        self.ilim  = self.get_parameter('integral_limit').value
        self.iylim = self.get_parameter('integral_yaw_lim').value
        rate       = self.get_parameter('rate').value
        self.dt    = 1.0 / rate

        # Estado de ambos drones (sin lock: executor single-thread serializa callbacks)
        self._L = {k: None for k in ('x','y','z','vx','vy','vz','yaw','yaw_rate')}
        self._S = {k: None for k in ('x','y','z','vx','vy','vz','yaw','yaw_rate')}

        # Memoria PID
        self._integ     = [0.0, 0.0, 0.0]
        self._integ_yaw = 0.0

        # ── Suscripciones líder ───────────────────────────────────────────────
        self.create_subscription(
            PoseStamped,
            f'/{L_ns}/local_position/pose',
            self._cb_lpose, _STATE_QOS
        )
        self.create_subscription(
            TwistStamped,
            f'/{L_ns}/local_position/velocity_local',
            self._cb_lvel, _STATE_QOS
        )

        # ── Suscripciones seguidor ────────────────────────────────────────────
        self.create_subscription(
            PoseStamped,
            f'/{F_ns}/local_position/pose',
            self._cb_spose, _STATE_QOS
        )
        self.create_subscription(
            TwistStamped,
            f'/{F_ns}/local_position/velocity_local',
            self._cb_svel, _STATE_QOS
        )

        # ── Publicador ────────────────────────────────────────────────────────
        self._pub = self.create_publisher(
            TwistStamped,
            f'/{F_ns}/setpoint_velocity/cmd_vel',
            _MAVROS_QOS
        )

        # ── CSV ───────────────────────────────────────────────────────────────
        fname = f'lf_{int(time.time())}.csv'
        self._csv_f = open(fname, 'w', newline='')
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow([
            't',
            'lx','ly','lz','lvx','lvy','lvz','l_yaw','l_yawrate',
            'sx','sy','sz','svx','svy','svz','s_yaw','s_yawrate',
            'xd','yd','zd','ex','ey','ez','e_yaw',
            'ff_x','ff_y',
            'vx_p','vy_p','vz_p',
            'vx_i','vy_i','vz_i',
            'vx_d','vy_d','vz_d',
            'vx_cmd','vy_cmd','vz_cmd',
        ])
        self._t0 = self.get_clock().now()
        self.get_logger().info(f'CSV: {fname}')

        # ── Timer de control ──────────────────────────────────────────────────
        self.create_timer(self.dt, self._loop)
        self.get_logger().info(
            f'LeaderFollowerNode: líder={L_ns} seguidor={F_ns}\n'
            f'  d={self.d}m α={math.degrees(self.alpha):.1f}° Δz={self.dz}m\n'
            f'  Kp={self.kp} Ki={self.ki} Kd={self.kd}\n'
            f'  Kp_ψ={self.kp_y} Ki_ψ={self.ki_y} Kd_ψ={self.kd_y}'
        )

    # ── Callbacks líder ───────────────────────────────────────────────────────
    def _cb_lpose(self, msg: PoseStamped):
        p = msg.pose.position
        self._L['x']   = p.x
        self._L['y']   = p.y
        self._L['z']   = p.z
        self._L['yaw'] = _quat_to_yaw(msg.pose.orientation)

    def _cb_lvel(self, msg: TwistStamped):
        v = msg.twist.linear
        self._L['vx']       = v.x
        self._L['vy']       = v.y
        self._L['vz']       = v.z
        self._L['yaw_rate'] = msg.twist.angular.z

    # ── Callbacks seguidor ────────────────────────────────────────────────────
    def _cb_spose(self, msg: PoseStamped):
        p = msg.pose.position
        self._S['x']   = p.x
        self._S['y']   = p.y
        self._S['z']   = p.z
        self._S['yaw'] = _quat_to_yaw(msg.pose.orientation)

    def _cb_svel(self, msg: TwistStamped):
        v = msg.twist.linear
        self._S['vx']       = v.x
        self._S['vy']       = v.y
        self._S['vz']       = v.z
        self._S['yaw_rate'] = msg.twist.angular.z

    def _ready(self):
        keys = ('x','y','z','vx','vy','vz','yaw','yaw_rate')
        return (all(self._L[k] is not None for k in keys) and
                all(self._S[k] is not None for k in keys))

    # ── Ley de control ────────────────────────────────────────────────────────
    def _loop(self):
        if not self._ready():
            return

        L, S = self._L, self._S

        # 1. Offset polar rotante d(t)
        angle = L['yaw'] + self.alpha
        dx = self.d * math.cos(angle)
        dy = self.d * math.sin(angle)
        dz = self.dz

        # 2. Posición deseada
        xd = L['x'] + dx
        yd = L['y'] + dy
        zd = L['z'] + dz

        # 3. Error de posición
        ex = xd - S['x']
        ey = yd - S['y']
        ez = zd - S['z']

        # 4. Integrador con anti-windup
        self._integ[0] = _clamp(self._integ[0] + ex * self.dt, self.ilim)
        self._integ[1] = _clamp(self._integ[1] + ey * self.dt, self.ilim)
        self._integ[2] = _clamp(self._integ[2] + ez * self.dt, self.ilim)

        # 5. Diferencia de velocidades (acción derivativa sin ruido numérico)
        dv_x = L['vx'] - S['vx']
        dv_y = L['vy'] - S['vy']
        dv_z = L['vz'] - S['vz']

        # 6. Prealimentación: ψ̇_L · Rz(π/2) · d
        ff_x = L['yaw_rate'] * (-dy)
        ff_y = L['yaw_rate'] * ( dx)

        # 7. Ley PID posición
        vx = ff_x + self.kp*ex + self.ki*self._integ[0] + self.kd*dv_x
        vy = ff_y + self.kp*ey + self.ki*self._integ[1] + self.kd*dv_y
        vz =        self.kp*ez + self.ki*self._integ[2] + self.kd*dv_z

        # Clamp velocidad horizontal
        v_h = math.hypot(vx, vy)
        if v_h > self.vmax:
            vx *= self.vmax / v_h
            vy *= self.vmax / v_h
        vz = _clamp(vz, self.vmax)

        # 8. PID yaw
        e_yaw = _wrap(L['yaw'] - S['yaw'])
        self._integ_yaw = _clamp(
            self._integ_yaw + e_yaw * self.dt, self.iylim
        )
        dyaw = L['yaw_rate'] - S['yaw_rate']
        yr_cmd = _clamp(
            self.kp_y * e_yaw + self.ki_y * self._integ_yaw + self.kd_y * dyaw,
            self.yrmax
        )

        # 9. Publicar (ENU: vx=este, vy=norte, vz=arriba)
        self._pub.publish(_make_twist(self, vx, vy, vz, yr_cmd))

        # 10. CSV
        t_s = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        self._csv_w.writerow([
            f'{t_s:.4f}',
            L['x'], L['y'], L['z'], L['vx'], L['vy'], L['vz'],
            L['yaw'], L['yaw_rate'],
            S['x'], S['y'], S['z'], S['vx'], S['vy'], S['vz'],
            S['yaw'], S['yaw_rate'],
            f'{xd:.4f}', f'{yd:.4f}', f'{zd:.4f}',
            f'{ex:.4f}', f'{ey:.4f}', f'{ez:.4f}', f'{e_yaw:.4f}',
            f'{ff_x:.4f}', f'{ff_y:.4f}',
            f'{self.kp*ex:.4f}', f'{self.kp*ey:.4f}', f'{self.kp*ez:.4f}',
            f'{self.ki*self._integ[0]:.4f}', f'{self.ki*self._integ[1]:.4f}',
            f'{self.ki*self._integ[2]:.4f}',
            f'{self.kd*dv_x:.4f}', f'{self.kd*dv_y:.4f}', f'{self.kd*dv_z:.4f}',
            f'{vx:.4f}', f'{vy:.4f}', f'{vz:.4f}',
        ])

    def destroy_node(self):
        self._pub.publish(_make_twist(self, 0.0, 0.0, 0.0, 0.0))
        self._csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LeaderFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupción por usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
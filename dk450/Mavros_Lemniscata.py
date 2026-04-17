#!/usr/bin/env python3
"""
lemniscate_node.py — Vuelo en lemniscata de Bernoulli
======================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.
Migración de MavlinkLemniscata.py.

El primer setpoint es EXACTAMENTE la posición actual del dron (sin salto).

Modos de inicio:
  'center' → dron en el cruce central (θ₀ = π/2, pos_relativa = 0,0)
  'tip'    → dron en el extremo del lóbulo derecho (θ₀ = 0, pos_relativa = a,0)

Parámetros ROS2:
  uav_ns        : namespace del dron              (default: 'uav1')
  axis_a        : semieje X — ancho lóbulo [m]   (default: 4.0)
  axis_b        : semieje Y — alto lóbulo [m]    (default: 2.0)
  angular_speed : velocidad angular [rad/s]       (default: 0.3)
  start_mode    : 'center' | 'tip'               (default: 'center')
  rate          : frecuencia de control [Hz]      (default: 50)
  conv_radius   : radio convergencia fase cola [m](default: 0.15)
  conv_speed    : velocidad umbral cola [m/s]     (default: 0.10)
  conv_hold     : hold en zona [s]                (default: 1.0)
  conv_timeout  : timeout fase cola [s]           (default: 15.0)

Tópicos:
  Sub: /<uav_ns>/mavros/local_position/pose           (PoseStamped)
  Sub: /<uav_ns>/mavros/local_position/velocity_local (TwistStamped)
  Pub: /<uav_ns>/mavros/setpoint_position/local       (PoseStamped)
"""

import math
import csv
import time
import threading
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
def _make_pose(node, x: float, y: float, z: float, yaw: float) -> PoseStamped:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    msg = PoseStamped()
    msg.header.stamp    = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = float(sy)
    msg.pose.orientation.w = float(cy)
    return msg

# ── Parametrización de la lemniscata ─────────────────────────────────────────
def _lem_pos(a, b, t):
    s, c = math.sin(t), math.cos(t)
    d = 1.0 + s * s
    return a * c / d, b * s * c / d

def _lem_vel(a, b, omega, t):
    s, c = math.sin(t), math.cos(t)
    d = 1.0 + s * s
    dx_dt = a * (-s * d - c * 2.0 * s * c) / (d * d)
    dy_dt = b * ((c*c - s*s) * d - s * c * 2.0 * s * c) / (d * d)
    return dx_dt * omega, dy_dt * omega

def _lem_yaw(a, b, omega, t):
    vx, vy = _lem_vel(a, b, omega, t)
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return math.atan2(vy, vx)


# ─────────────────────────────────────────────────────────────────────────────
# Nodo
# ─────────────────────────────────────────────────────────────────────────────
class LemniscateNode(Node):

    def __init__(self):
        super().__init__('lemniscate_node')

        self.declare_parameter('uav_ns',       'uav1')
        self.declare_parameter('axis_a',        4.0)
        self.declare_parameter('axis_b',        2.0)
        self.declare_parameter('angular_speed', 0.3)
        self.declare_parameter('start_mode',   'center')
        self.declare_parameter('rate',          50)
        self.declare_parameter('conv_radius',   0.15)
        self.declare_parameter('conv_speed',    0.10)
        self.declare_parameter('conv_hold',     1.0)
        self.declare_parameter('conv_timeout', 15.0)

        ns            = self.get_parameter('uav_ns').value
        self.a        = self.get_parameter('axis_a').value
        self.b        = self.get_parameter('axis_b').value
        self.omega    = self.get_parameter('angular_speed').value
        self.mode     = self.get_parameter('start_mode').value
        rate          = self.get_parameter('rate').value
        self.conv_r   = self.get_parameter('conv_radius').value
        self.conv_v   = self.get_parameter('conv_speed').value
        self.conv_hold= self.get_parameter('conv_hold').value
        self.conv_tout= self.get_parameter('conv_timeout').value
        self.dt       = 1.0 / rate

        self._lock     = threading.Lock()
        self._pos      = None
        self._vel      = None
        self._home_evt = threading.Event()
        self._home     = None

        self.create_subscription(
            PoseStamped,
            f'/{ns}/local_position/pose',
            self._cb_pose, _STATE_QOS
        )
        self.create_subscription(
            TwistStamped,
            f'/{ns}/local_position/velocity_local',
            self._cb_vel, _STATE_QOS
        )
        self._pub = self.create_publisher(
            PoseStamped,
            f'/{ns}/setpoint_position/local',
            _MAVROS_QOS
        )

        fname = f'lemniscate_{int(time.time())}.csv'
        self._csv_f = open(fname, 'w', newline='')
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow([
            't', 'x_sp', 'y_sp', 'x', 'y',
            'theta', 'vx_des', 'vy_des', 'err_pos'
        ])
        self._t0 = None
        self.get_logger().info(f'CSV: {fname}')

        threading.Thread(target=self._flight,
                         daemon=True, name='lem-flight').start()
        self.get_logger().info(
            f'LemniscateNode ns={ns} a={self.a} b={self.b} '
            f'ω={self.omega} modo={self.mode}'
        )

    def _cb_pose(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            self._pos = (p.x, p.y, p.z)
            if self._home is None:
                self._home = (p.x, p.y, p.z)
                self._home_evt.set()

    def _cb_vel(self, msg: TwistStamped):
        with self._lock:
            v = msg.twist.linear
            self._vel = (v.x, v.y, v.z)

    def _get(self):
        with self._lock:
            return self._pos, self._vel, self._home

    def _flight(self):
        self.get_logger().info('Esperando posición inicial...')
        self._home_evt.wait()
        _, _, home = self._get()
        x0, y0, z0 = home

        # Centro geométrico según modo de inicio
        if self.mode == 'center':
            cx, cy  = x0, y0
            t0_p    = math.pi / 2   # lem_pos(a,b,π/2) = (0,0) ✓
        else:
            cx, cy  = x0 - self.a, y0
            t0_p    = 0.0           # lem_pos(a,b,0) = (a,0) ✓

        lx0, ly0 = _lem_pos(self.a, self.b, t0_p)
        err0 = math.hypot(cx + lx0 - x0, cy + ly0 - y0)
        self.get_logger().info(
            f'Centro figura: ({cx:.3f},{cy:.3f}) '
            f'error inicial={err0:.4f}m'
        )

        duration = 2 * math.pi / self.omega
        steps    = int(duration / self.dt)
        self._t0 = time.monotonic()
        self.get_logger().info(
            f'Iniciando lemniscata: {steps} pasos @ {1/self.dt:.0f}Hz T={duration:.1f}s'
        )

        next_t = time.monotonic()
        x_sp = y_sp = 0.0
        yaw = 0.0

        for i in range(steps):
            theta       = t0_p + self.omega * i * self.dt
            lx, ly      = _lem_pos(self.a, self.b, theta)
            x_sp        = cx + lx
            y_sp        = cy + ly
            yaw         = _lem_yaw(self.a, self.b, self.omega, theta)
            vx_d, vy_d  = _lem_vel(self.a, self.b, self.omega, theta)

            self._pub.publish(_make_pose(self, x_sp, y_sp, z0, yaw))

            pos, _, _ = self._get()
            if pos:
                err = math.hypot(pos[0] - x_sp, pos[1] - y_sp)
                t_now = time.monotonic() - self._t0
                self._csv_w.writerow([
                    f'{t_now:.4f}',
                    f'{x_sp:.4f}', f'{y_sp:.4f}',
                    f'{pos[0]:.4f}', f'{pos[1]:.4f}',
                    f'{theta:.4f}', f'{vx_d:.4f}', f'{vy_d:.4f}',
                    f'{err:.4f}'
                ])

            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

        # ── Fase de cola ──────────────────────────────────────────────────────
        self.get_logger().info(
            f'Convergiendo al punto final (radio={self.conv_r}m)...'
        )
        t_tail = time.monotonic()
        zone_t = None

        while True:
            now = time.monotonic()
            if now - t_tail > self.conv_tout:
                self.get_logger().warn('Timeout de convergencia.')
                break

            self._pub.publish(_make_pose(self, x_sp, y_sp, z0, yaw))
            pos, vel, _ = self._get()

            if pos:
                dist  = math.hypot(pos[0] - x_sp, pos[1] - y_sp)
                speed = math.hypot(*(vel or (0.0, 0.0, 0.0))[:2])
                if dist < self.conv_r and speed < self.conv_v:
                    zone_t = zone_t or now
                    if now - zone_t >= self.conv_hold:
                        self.get_logger().info(
                            f'Convergencia: dist={dist:.3f}m vel={speed:.3f}m/s'
                        )
                        break
                else:
                    zone_t = None

            time.sleep(self.dt)

        self.get_logger().info('Registro completo.')

    def destroy_node(self):
        self._csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LemniscateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
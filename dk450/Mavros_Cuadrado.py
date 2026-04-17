#!/usr/bin/env python3
"""
waypoints_node.py — Seguidor de waypoints en lazo cerrado
==========================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.
Migración de MavlinkCuadrado.py.

Parámetros ROS2:
  uav_ns           : namespace del dron             (default: 'uav1')
  advance_mode     : 'convergence' | 'timer'        (default: 'convergence')
  waypoint_interval: segundos por WP en modo timer  (default: 5.0)
  conv_radius      : radio de convergencia [m]      (default: 0.30)
  conv_speed       : velocidad umbral [m/s]         (default: 0.15)
  conv_hold        : hold en zona [s]               (default: 1.0)
  conv_timeout     : timeout por WP [s]             (default: 20.0)
  rate             : frecuencia de control [Hz]     (default: 20)
  yaw              : yaw fijo [rad], NaN=ignorar    (default: nan)

Editar DEFAULT_WAYPOINTS para cambiar la ruta (coordenadas relativas al despegue, ENU).

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
# Waypoints por defecto: cuadrado 5×5 a 3 m de altitud (ENU, relativos al home)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_WAYPOINTS = [
    (0.0, 0.0, 3.0),   # hover inicial
    (5.0, 0.0, 3.0),
    (5.0, 5.0, 3.0),
    (0.0, 5.0, 3.0),
    (0.0, 0.0, 3.0),   # regreso
]

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
# Helper
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


# ─────────────────────────────────────────────────────────────────────────────
# Nodo
# ─────────────────────────────────────────────────────────────────────────────
class WaypointsNode(Node):

    def __init__(self):
        super().__init__('waypoints_node')

        self.declare_parameter('uav_ns',            'uav1')
        self.declare_parameter('advance_mode',      'convergence')
        self.declare_parameter('waypoint_interval',  5.0)
        self.declare_parameter('conv_radius',        0.30)
        self.declare_parameter('conv_speed',         0.15)
        self.declare_parameter('conv_hold',          1.0)
        self.declare_parameter('conv_timeout',      20.0)
        self.declare_parameter('rate',              20)
        self.declare_parameter('yaw',              float('nan'))

        ns               = self.get_parameter('uav_ns').value
        self.adv_mode    = self.get_parameter('advance_mode').value
        self.wp_interval = self.get_parameter('waypoint_interval').value
        self.conv_r      = self.get_parameter('conv_radius').value
        self.conv_v      = self.get_parameter('conv_speed').value
        self.conv_hold   = self.get_parameter('conv_hold').value
        self.conv_tout   = self.get_parameter('conv_timeout').value
        rate             = self.get_parameter('rate').value
        yaw_p            = self.get_parameter('yaw').value
        self.yaw_fixed   = None if math.isnan(yaw_p) else yaw_p
        self.dt          = 1.0 / rate

        self._lock       = threading.Lock()
        self._pos        = None
        self._vel        = None
        self._home       = None
        self._home_evt   = threading.Event()

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

        fname = f'waypoints_{int(time.time())}.csv'
        self._csv_f = open(fname, 'w', newline='')
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow(['t', 'wp', 'x_sp', 'y_sp', 'z_sp',
                               'x', 'y', 'z', 'dist', 'speed'])
        self._t0 = None
        self.get_logger().info(f'CSV: {fname}')

        threading.Thread(target=self._sequence,
                         daemon=True, name='wp-seq').start()
        self.get_logger().info(
            f'WaypointsNode ns={ns} modo={self.adv_mode} '
            f'{len(DEFAULT_WAYPOINTS)} waypoints'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
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

    # ── Publicar waypoint ─────────────────────────────────────────────────────
    def _pub_wp(self, x, y, z):
        yaw = self.yaw_fixed if self.yaw_fixed is not None else 0.0
        self._pub.publish(_make_pose(self, x, y, z, yaw))

    # ── Log CSV ───────────────────────────────────────────────────────────────
    def _log(self, idx, tx, ty, tz, pos, vel):
        if self._t0 is None or pos is None:
            return
        x, y, z = pos
        vx, vy, vz = vel or (0.0, 0.0, 0.0)
        dist  = math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2)
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        self._csv_w.writerow([
            f'{time.monotonic()-self._t0:.4f}', idx,
            f'{tx:.4f}', f'{ty:.4f}', f'{tz:.4f}',
            f'{x:.4f}', f'{y:.4f}', f'{z:.4f}',
            f'{dist:.4f}', f'{speed:.4f}'
        ])

    # ── Secuencia principal (hilo separado) ───────────────────────────────────
    def _sequence(self):
        self.get_logger().info('Esperando posición inicial...')
        self._home_evt.wait()
        _, _, home = self._get()
        hx, hy, hz = home
        wps = [(hx+dx, hy+dy, hz+dz) for dx, dy, dz in DEFAULT_WAYPOINTS]
        self._t0 = time.monotonic()
        self.get_logger().info(f'Home=({hx:.2f},{hy:.2f},{hz:.2f})')

        for idx, (tx, ty, tz) in enumerate(wps):
            self.get_logger().info(
                f'[{idx+1}/{len(wps)}] → ({tx:.2f},{ty:.2f},{tz:.2f})'
            )
            if self.adv_mode == 'timer':
                self._exec_timer(idx, tx, ty, tz)
            else:
                self._exec_conv(idx, tx, ty, tz)

        self.get_logger().info('Secuencia completada.')

    def _exec_timer(self, idx, tx, ty, tz):
        t0     = time.monotonic()
        next_t = t0
        while time.monotonic() - t0 < self.wp_interval:
            self._pub_wp(tx, ty, tz)
            pos, vel, _ = self._get()
            self._log(idx, tx, ty, tz, pos, vel)
            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)
        self.get_logger().info(f'  WP{idx+1} avanzado por tiempo')

    def _exec_conv(self, idx, tx, ty, tz):
        t0     = time.monotonic()
        zone_t = None
        next_t = t0
        while True:
            now = time.monotonic()
            if now - t0 > self.conv_tout:
                self.get_logger().warn(f'  WP{idx+1} timeout — avanzando')
                break
            self._pub_wp(tx, ty, tz)
            pos, vel, _ = self._get()
            self._log(idx, tx, ty, tz, pos, vel)
            if pos:
                x, y, z = pos
                vx, vy, vz = vel or (0.0, 0.0, 0.0)
                dist  = math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2)
                speed = math.sqrt(vx**2 + vy**2 + vz**2)
                if dist < self.conv_r and speed < self.conv_v:
                    zone_t = zone_t or now
                    if now - zone_t >= self.conv_hold:
                        self.get_logger().info(
                            f'  WP{idx+1} OK dist={dist:.3f}m '
                            f'vel={speed:.3f}m/s en {now-t0:.1f}s'
                        )
                        break
                else:
                    zone_t = None
            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

    def destroy_node(self):
        self._csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
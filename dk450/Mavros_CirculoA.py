#!/usr/bin/env python3
"""
circle_node.py — Vuelo circular con path following en lazo cerrado
==================================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.

Parámetros ROS2:
  uav_ns        : namespace del dron              (default: 'uav1')
  radius        : radio del círculo [m]           (default: 4.0)
  angular_speed : velocidad angular [rad/s]       (default: 0.5)
  rate          : frecuencia de control [Hz]      (default: 50)
  lookahead_time: anticipación del setpoint [s]   (default: 1.5)
  kp_radial     : ganancia corrección radial      (default: 0.5)

Tópicos:
  Sub: /<uav_ns>/mavros/local_position/pose           (PoseStamped)
  Sub: /<uav_ns>/mavros/local_position/velocity_local (TwistStamped)
  Pub: /<uav_ns>/mavros/setpoint_position/local       (PoseStamped)

Frame: ENU (x=este, y=norte, z=arriba) — estándar MAVROS2/ROS2.
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
# Helpers (inline — sin imports externos)
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_yaw(q) -> float:
    """Extrae yaw (rad) de un quaternion ROS. ENU: yaw=0 → este."""
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def _make_pose(node, x: float, y: float, z: float, yaw: float) -> PoseStamped:
    """Construye un PoseStamped en frame 'map' (ENU)."""
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
class CircleNode(Node):

    def __init__(self):
        super().__init__('circle_node')

        self.declare_parameter('uav_ns',         'uav1')
        self.declare_parameter('radius',          4.0)
        self.declare_parameter('angular_speed',   0.5)
        self.declare_parameter('rate',            50)
        self.declare_parameter('lookahead_time',  1.5)
        self.declare_parameter('kp_radial',       0.5)

        ns         = self.get_parameter('uav_ns').value
        self.R     = self.get_parameter('radius').value
        self.omega = self.get_parameter('angular_speed').value
        rate       = self.get_parameter('rate').value
        self.la    = self.omega * self.get_parameter('lookahead_time').value
        self.kp_r  = self.get_parameter('kp_radial').value
        self.dt    = 1.0 / rate

        self._lock  = threading.Lock()
        self._pos   = None
        self._cx = self._cy = self._cz = None

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

        fname = f'circle_{int(time.time())}.csv'
        self._csv_f = open(fname, 'w', newline='')
        self._csv_w = csv.writer(self._csv_f)
        self._csv_w.writerow(['t', 'x_sp', 'y_sp', 'x', 'y', 'theta', 'err_r'])
        self._t0 = self.get_clock().now()
        self.get_logger().info(f'CSV: {fname}')

        self.create_timer(self.dt, self._loop)
        self.get_logger().info(
            f'CircleNode ns={ns} R={self.R}m ω={self.omega}rad/s '
            f'lookahead={math.degrees(self.la):.1f}° Kp_r={self.kp_r}'
        )

    def _cb_pose(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            self._pos = (p.x, p.y, p.z)

    def _cb_vel(self, msg: TwistStamped):
        pass   # disponible si se necesita en futuras versiones

    def _loop(self):
        with self._lock:
            pos = self._pos
        if pos is None:
            return

        x, y, z = pos

        # Fijar centro la primera vez (dron ya en el perímetro: cx = x − R)
        if self._cx is None:
            self._cx, self._cy, self._cz = x - self.R, y, z
            self.get_logger().info(
                f'Centro círculo: ({self._cx:.2f}, {self._cy:.2f}, {self._cz:.2f})'
            )

        cx, cy, cz = self._cx, self._cy, self._cz
        dx, dy = x - cx, y - cy
        r      = math.hypot(dx, dy)
        theta  = math.atan2(dy, dx)
        err_r  = r - self.R

        theta_sp = theta + self.la
        x_sp = cx + self.R * math.cos(theta_sp)
        y_sp = cy + self.R * math.sin(theta_sp)

        # Corrección radial
        if r > 1e-3:
            x_sp -= self.kp_r * err_r * (dx / r)
            y_sp -= self.kp_r * err_r * (dy / r)

        self._pub.publish(_make_pose(self, x_sp, y_sp, cz, theta_sp + math.pi / 2))

        t_s = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        self._csv_w.writerow([
            f'{t_s:.4f}', f'{x_sp:.4f}', f'{y_sp:.4f}',
            f'{x:.4f}', f'{y:.4f}', f'{theta:.4f}', f'{err_r:.4f}'
        ])

    def destroy_node(self):
        self._csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CircleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupción por usuario.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
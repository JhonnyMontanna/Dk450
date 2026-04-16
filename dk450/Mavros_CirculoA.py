#!/usr/bin/env python3
"""
circle_node.py — Vuelo circular con path following en lazo cerrado
==================================================================
ROS2 Humble + MAVROS2

Parámetros ROS2 (ros2 run ... --ros-args -p radius:=4.0 ...):
  uav_ns        : namespace del dron (default: 'uav1')
  radius        : radio del círculo [m]             (default: 4.0)
  angular_speed : velocidad angular [rad/s]         (default: 0.5)
  rate          : frecuencia de control [Hz]        (default: 50)
  lookahead_time: anticipación del setpoint [s]     (default: 1.5)
  kp_radial     : ganancia corrección radial        (default: 0.5)

Tópicos:
  Sub: /<uav_ns>/mavros/local_position/pose          (PoseStamped)
  Sub: /<uav_ns>/mavros/local_position/velocity_local (TwistStamped)
  Pub: /<uav_ns>/mavros/setpoint_position/local      (PoseStamped)
"""

import math
import csv
import time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped

from mavros2_utils import (
    MAVROS_QOS, STATE_QOS,
    quat_to_yaw, make_pose_stamped, wrap, clamp
)


class CircleNode(Node):

    def __init__(self):
        super().__init__('circle_node')

        # ── Parámetros ────────────────────────────────────────────────────────
        self.declare_parameter('uav_ns',         'uav1')
        self.declare_parameter('radius',          4.0)
        self.declare_parameter('angular_speed',   0.5)
        self.declare_parameter('rate',            50)
        self.declare_parameter('lookahead_time',  1.5)
        self.declare_parameter('kp_radial',       0.5)
        self.declare_parameter('v_max',           5.0)
        self.declare_parameter('conv_radius',     0.15)
        self.declare_parameter('conv_speed',      0.10)

        ns  = self.get_parameter('uav_ns').value
        self.R     = self.get_parameter('radius').value
        self.omega = self.get_parameter('angular_speed').value
        rate       = self.get_parameter('rate').value
        lt         = self.get_parameter('lookahead_time').value
        self.kp_r  = self.get_parameter('kp_radial').value
        self.v_max = self.get_parameter('v_max').value
        self.conv_r = self.get_parameter('conv_radius').value
        self.conv_v = self.get_parameter('conv_speed').value

        self.lookahead = self.omega * lt
        self.dt = 1.0 / rate

        # ── Estado del dron ───────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._pos  = None   # (x, y, z)
        self._vel  = None   # (vx, vy, vz)
        self._yaw  = None

        # Centro del círculo (se calcula al recibir la primera posición)
        self._cx = self._cy = self._cz = None
        self._initialized = False

        # ── Suscripciones ─────────────────────────────────────────────────────
        self.create_subscription(
            PoseStamped,
            f'/{ns}/mavros/local_position/pose',
            self._cb_pose, STATE_QOS
        )
        self.create_subscription(
            TwistStamped,
            f'/{ns}/mavros/local_position/velocity_local',
            self._cb_vel, STATE_QOS
        )

        # ── Publicador ────────────────────────────────────────────────────────
        self._pub = self.create_publisher(
            PoseStamped,
            f'/{ns}/mavros/setpoint_position/local',
            MAVROS_QOS
        )

        # ── CSV ───────────────────────────────────────────────────────────────
        fname = f'circle_{int(time.time())}.csv'
        self._csv_file   = open(fname, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            't', 'x_sp', 'y_sp', 'x', 'y',
            'theta', 'err_r', 'vx_ff', 'vy_ff'
        ])
        self._t0 = self.get_clock().now()
        self.get_logger().info(f'CSV: {fname}')

        # ── Timer de control ──────────────────────────────────────────────────
        self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f'CircleNode: ns={ns}, R={self.R} m, ω={self.omega} rad/s, '
            f'lookahead={math.degrees(self.lookahead):.1f}°, Kp_r={self.kp_r}'
        )

    # ── Callbacks de estado ───────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            self._pos = (p.x, p.y, p.z)
            self._yaw = quat_to_yaw(msg.pose.orientation)

    def _cb_vel(self, msg: TwistStamped):
        with self._lock:
            v = msg.twist.linear
            self._vel = (v.x, v.y, v.z)

    # ── Núcleo del controlador (path following + corrección radial) ───────────
    def _control_loop(self):
        with self._lock:
            pos = self._pos
            yaw = self._yaw

        if pos is None or yaw is None:
            return

        x, y, z = pos

        # Inicializar centro la primera vez
        if not self._initialized:
            # El dron ya está sobre el perímetro: centro desplazado RADIUS al oeste
            # En ENU: x=este → restar RADIUS en x coloca el centro al oeste del dron
            self._cx = x - self.R
            self._cy = y
            self._cz = z
            self._initialized = True
            self.get_logger().info(
                f'Centro círculo: cx={self._cx:.2f}, cy={self._cy:.2f}, '
                f'cz={self._cz:.2f} — dron en perímetro (err≈0)'
            )

        cx, cy, cz = self._cx, self._cy, self._cz

        # ── Path following con proyección (ec. del planteamiento) ─────────────
        dx = x - cx
        dy = y - cy
        r       = math.hypot(dx, dy)
        theta   = math.atan2(dy, dx)
        err_r   = r - self.R

        # Setpoint: punto del círculo en (theta + lookahead)
        theta_sp = theta + self.lookahead
        x_sp = cx + self.R * math.cos(theta_sp)
        y_sp = cy + self.R * math.sin(theta_sp)

        # Corrección radial incrustada en el setpoint
        if r > 1e-3:
            x_sp -= self.kp_r * err_r * (dx / r)
            y_sp -= self.kp_r * err_r * (dy / r)

        # Yaw tangencial al círculo
        yaw_sp = theta_sp + math.pi / 2

        # Velocidad feedforward (para CSV, no se envía — MAVROS la calcula internamente)
        vx_ff = -self.R * self.omega * math.sin(theta_sp)
        vy_ff =  self.R * self.omega * math.cos(theta_sp)

        # Clamp de posición del setpoint (seguridad: no más de 2·R del centro)
        dist_sp = math.hypot(x_sp - cx, y_sp - cy)
        if dist_sp > self.R * 2:
            ratio = self.R * 2 / dist_sp
            x_sp = cx + (x_sp - cx) * ratio
            y_sp = cy + (y_sp - cy) * ratio

        # ── Publicar ──────────────────────────────────────────────────────────
        self._pub.publish(make_pose_stamped(self, x_sp, y_sp, cz, yaw_sp))

        # ── CSV ───────────────────────────────────────────────────────────────
        t_s = (self.get_clock().now() - self._t0).nanoseconds * 1e-9
        self._csv_writer.writerow([
            f'{t_s:.4f}', f'{x_sp:.4f}', f'{y_sp:.4f}',
            f'{x:.4f}', f'{y:.4f}',
            f'{theta:.4f}', f'{err_r:.4f}',
            f'{vx_ff:.4f}', f'{vy_ff:.4f}'
        ])

    def destroy_node(self):
        self._csv_file.close()
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
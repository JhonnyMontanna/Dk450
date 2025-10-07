#!/usr/bin/env python3
"""
Controlador de trayectoria circular usando MAVROS
=================================================

• Publica velocidades lineales XY en `/mavros/setpoint_velocity/cmd_vel_unstamped`.
• Obtiene la posición NED (ros-flight usa ENU) de `/mavros/local_position/pose`.
• Cambia a modo OFFBOARD y arma vía servicios MAVROS.
• Guarda y grafica la trayectoria real vs deseada al final.

Requisitos:
  sudo apt install ros-$ROS_DISTRO-tf-transformations python3-matplotlib python3-pandas

Lanza antes MAVROS conectado al FCU y asegúrate de que `/mavros/setpoint_velocity/cmd_vel_unstamped` esté permitido (param `sysid` etc.).
"""

import math
import time
from threading import Event
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.srv import CommandBool, SetMode

import pandas as pd
import matplotlib.pyplot as plt

# ------------- Parámetros de trayectoria y control -------------
RADIUS = 10.0                # m
ANGULAR_SPEED = 0.2          # rad/s  (≈ 32 s por vuelta)
Kp = 0.5                     # ganancia proporcional
RATE = 20                    # Hz
DT = 1.0 / RATE

# ---------- Topic names ----------
POSE_TOPIC = '/mavros/local_position/pose'
VEL_TOPIC  = '/mavros/setpoint_velocity/cmd_vel_unstamped'


class CircleController(Node):
    def __init__(self):
        super().__init__('circle_controller')

        # QoS mejor esfuerzo para pose; fiable para velocidades
        qos_pose = QoSProfile(depth=10,
                              reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST)
        qos_vel  = QoSProfile(depth=10,
                              reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)

        # Suscriptor a la posición local (PoseStamped)
        self.pose_sub = self.create_subscription(
            PoseStamped, POSE_TOPIC, self.pose_cb, qos_pose)

        # Publicador de velocidades (Twist)
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, qos_vel)

        # Servicios de armar y set_mode
        self.arm_cli  = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_cli = self.create_client(SetMode, '/mavros/set_mode')

        # Esperamos disponibilidad de servicios
        self.get_logger().info('⏳ Esperando servicios MAVROS…')
        self.arm_cli.wait_for_service()
        self.mode_cli.wait_for_service()
        self.get_logger().info('✅ Servicios listos')

        # Datos recientes de pose
        self.last_pose = None
        self.pose_event = Event()

        # Buffers para logging
        self.t_buf, self.xd_buf, self.yd_buf, self.x_buf, self.y_buf = [], [], [], [], []

        # Comienza temporizador principal
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.timer = self.create_timer(DT, self.control_loop)

        # Secuencia de habilitación OFFBOARD y armado
        self.set_offboard_and_arm()

    # ---------- Callbacks ----------
    def pose_cb(self, msg: PoseStamped):
        self.last_pose = msg
        self.pose_event.set()

    # ---------- Servicios ----------
    def set_offboard_and_arm(self):
        # MAVROS exige publicar algunos setpoints antes de conmutar a OFFBOARD
        self.get_logger().info('🔄 Publicando setpoints en vacío…')
        twist = Twist()
        for _ in range(100):
            self.vel_pub.publish(twist)
            time.sleep(0.01)

        # Conmutar a OFFBOARD
        req_mode = SetMode.Request()
        req_mode.custom_mode = 'OFFBOARD'
        fut = self.mode_cli.call_async(req_mode)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result() and fut.result().mode_sent:
            self.get_logger().info('✅ OFFBOARD set')
        else:
            self.get_logger().error('❌ No se pudo poner en OFFBOARD')

        # Armar
        req_arm = CommandBool.Request()
        req_arm.value = True
        fut2 = self.arm_cli.call_async(req_arm)
        rclpy.spin_until_future_complete(self, fut2, timeout_sec=5.0)
        if fut2.result() and fut2.result().success:
            self.get_logger().info('🟢 Drone armado')
        else:
            self.get_logger().error('❌ No se pudo armar')

        # Esperar la primera pose
        self.get_logger().info('⌛ Esperando primera pose…')
        if not self.pose_event.wait(timeout=5.0):
            self.get_logger().error('No se recibió pose; saliendo.')
            rclpy.shutdown()

    # ---------- Control Loop ----------
    def control_loop(self):
        t_now = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        theta = ANGULAR_SPEED * t_now
        x_d = RADIUS * math.cos(theta)
        y_d = RADIUS * math.sin(theta)

        if self.last_pose is None:
            return  # aún no tenemos posición

        # Pose actual ENU (NED con signos?) – suponemos ENU
        x = self.last_pose.pose.position.x
        y = self.last_pose.pose.position.y

        vx = Kp * (x_d - x)
        vy = Kp * (y_d - y)

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = 0.0
        self.vel_pub.publish(twist)

        # Guardar log
        self.t_buf.append(t_now)
        self.xd_buf.append(x_d)
        self.yd_buf.append(y_d)
        self.x_buf.append(x)
        self.y_buf.append(y)

        # Detener después de una vuelta completa
        if t_now >= (2 * math.pi / ANGULAR_SPEED):
            self.get_logger().info('Trayectoria completa; deteniendo…')
            self.timer.cancel()
            self.vel_pub.publish(Twist())  # paro
            self.land_and_finish()

    def land_and_finish(self):
        # Cambiar a LAND mediante servicio set_mode
        req = SetMode.Request()
        req.custom_mode = 'AUTO.LAND'
        self.mode_cli.call_async(req)
        self.get_logger().info('🛬 LAND enviado')
        time.sleep(3)
        self.plot_trajectory()
        rclpy.shutdown()

    # ---------- Plot utils ----------
    def plot_trajectory(self):
        df = pd.DataFrame({
            'time': self.t_buf,
            'x_d': self.xd_buf,
            'y_d': self.yd_buf,
            'x':   self.x_buf,
            'y':   self.y_buf
        })
        self.get_logger().info(f'\n{df.head()}')
        plt.figure()
        plt.plot(df.x_d, df.y_d, label='Deseada')
        plt.plot(df.x, df.y, label='Real')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Trayectoria')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()


def main():
    rclpy.init()
    node = CircleController()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

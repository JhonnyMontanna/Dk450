#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class MavrosCircleROS2(Node):
    def __init__(self):
        super().__init__('mavros_circle_ros2')

        # Parámetros
        self.declare_parameter('drone_ns', 'uav1')
        self.declare_parameter('radius', 2.0)           # m
        self.declare_parameter('angular_speed', 1.0)    # rad/s
        self.declare_parameter('rate', 50)              # Hz
        self.declare_parameter('loops', 1)              # cuántos círculos completos

        ns   = self.get_parameter('drone_ns').value
        self.R   = float(self.get_parameter('radius').value)
        self.omega = float(self.get_parameter('angular_speed').value)
        self.rate  = int(self.get_parameter('rate').value)
        self.loops = int(self.get_parameter('loops').value)

        assert self.R > 0.0, "radius debe ser > 0"
        assert self.omega > 0.0, "angular_speed debe ser > 0"
        assert self.rate > 0, "rate debe ser > 0"
        assert self.loops >= 1, "loops debe ser >= 1"

        self.v = self.R * self.omega
        self.dt = 1.0 / self.rate

        # Publicador MAVROS (velocidades en ENU)
        topic = f'/{ns}/setpoint_velocity/cmd_vel_unstamped'
        self.pub = self.create_publisher(Twist, topic, 10)

        self.get_logger().info(
            f"🌀 Círculo para [{ns}]: R={self.R} m, ω={self.omega} rad/s, "
            f"v={self.v:.3f} m/s, rate={self.rate} Hz, loops={self.loops}"
        )

    def run(self):
        """
        Ejecuta 'loops' círculos completos publicando velocidades tangenciales.
        Bucle síncrono con pasos exactos; al final, detiene y cierra limpio.
        """
        steps_per_loop = max(1, round((2.0 * math.pi / self.omega) / self.dt))
        total_steps = steps_per_loop * self.loops

        self.get_logger().info(
            f"⏱️ Pasos por círculo: {steps_per_loop} (dt={self.dt:.4f}s) → total pasos={total_steps}"
        )

        t0 = time.perf_counter()
        for i in range(total_steps):
            # Tiempo “ideal” del paso i para mitigar drift de sleep()
            t_target = t0 + i * self.dt
            now = time.perf_counter()
            if t_target > now:
                time.sleep(t_target - now)

            # Fase actual (usar i*dt exacto, no el tiempo real para cerrar bien el 2π)
            t = i * self.dt
            theta = self.omega * t

            # Velocidad tangencial para describir un círculo en el plano XY (ENU)
            vx =  self.v * math.cos(theta + math.pi/2.0)
            vy =  self.v * math.sin(theta + math.pi/2.0)

            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = vy
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            self.pub.publish(msg)

        # Parada suave: enviar varias veces cero
        self.get_logger().info("⏹️ Círculo(s) completado(s): publicando velocidad cero…")
        zero = Twist()
        for _ in range(int(self.rate * 0.2)):  # 0.2 s de zeros
            self.pub.publish(zero)
            time.sleep(self.dt)

        self.get_logger().info("✅ Terminado. Cerrando nodo y ROS 2…")

def main(args=None):
    rclpy.init(args=args)
    node = MavrosCircleROS2()
    try:
        node.run()  # bucle síncrono (sin spin)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupción por teclado — deteniendo.")
        # publicar unos zeros por seguridad
        zero = Twist()
        for _ in range(10):
            node.pub.publish(zero)
            time.sleep(0.02)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
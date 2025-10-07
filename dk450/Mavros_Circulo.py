#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped

class MavrosCircleClosedLoop(Node):
    def __init__(self):
        super().__init__('mavros_circle_closed_loop')

        # Parámetros base (los tuyos más algunos para el controlador)
        self.declare_parameter('drone_ns', 'uav1')
        self.declare_parameter('radius', 2.0)           # m
        self.declare_parameter('angular_speed', 1.0)    # rad/s (ω)
        self.declare_parameter('rate', 50)              # Hz (control rate)
        self.declare_parameter('loops', 1)              # vueltas completas
        self.declare_parameter('Kp', 1.8)               # P para controlador de posición
        self.declare_parameter('Kd', 0.3)               # D para controlador de posición
        self.declare_parameter('max_speed', 3.0)        # límite en m/s para vx,vy
        self.declare_parameter('center_tolerance', 0.2) # m, tolerancia para "volver al centro"
        self.declare_parameter('pose_topic', 'mavros/local_position/pose') # relativo al namespace

        ns       = self.get_parameter('drone_ns').value
        self.R   = float(self.get_parameter('radius').value)
        self.omega = float(self.get_parameter('angular_speed').value)
        self.rate  = int(self.get_parameter('rate').value)
        self.loops = int(self.get_parameter('loops').value)

        self.Kp = float(self.get_parameter('Kp').value)
        self.Kd = float(self.get_parameter('Kd').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.center_tolerance = float(self.get_parameter('center_tolerance').value)
        pose_topic = f'/{ns}/{self.get_parameter("pose_topic").value}'

        assert self.R > 0.0, "radius debe ser > 0"
        assert self.omega > 0.0, "angular_speed debe ser > 0"
        assert self.rate > 0, "rate debe ser > 0"
        assert self.loops >= 1, "loops debe ser >= 1"

        # feedforward tangential speed (magnitud)
        self.v_ff = self.R * self.omega

        # Otras variables internas
        self.dt = 1.0 / self.rate
        self.theta = 0.0                # fase actual del setpoint sobre el círculo
        self.total_theta = 0.0          # para contar vueltas
        self.center = None              # (x,y,z) fijado en la primera pose recibida
        self.pose = None                # pose actual (x,y,z)
        self.last_error = (0.0, 0.0)    # para D
        self.finished_circle = False
        self.returning_to_center = False

        # Publicador de velocidad (mavros setpoint topic)
        vel_topic = f'/{ns}/setpoint_velocity/cmd_vel_unstamped'
        self.pub = self.create_publisher(Twist, vel_topic, 10)

        # Subscriber a la pose local (PoseStamped). Ajusta el topic si tu stack usa otro.
        self.sub_pose = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, 10
        )

        # Timer: controlador cerrado en rate
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"🌀 Nodo círculo cerrado [{ns}]: R={self.R} m, ω={self.omega} rad/s, "
            f"v_ff={self.v_ff:.3f} m/s, rate={self.rate} Hz, loops={self.loops}, "
            f"Kp={self.Kp}, Kd={self.Kd}, max_speed={self.max_speed}"
        )

        # Timeout: si no llega pose en X segundos avisamos y salimos limpio
        self.pose_wait_start = time.time()
        self.pose_wait_timeout = 5.0  # segundos

    def pose_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.pose = (x, y, z)

        # Si aún no tenemos centro, fijarlo en la primera pose recibida
        if self.center is None:
            self.center = (x, y, z)
            self.get_logger().info(f"📍 Centro fijado en la posición inicial: x={x:.3f}, y={y:.3f}, z={z:.3f}")

    def control_loop(self):
        # Si no hay pose aún, chequear timeout
        if self.pose is None:
            if time.time() - self.pose_wait_start > self.pose_wait_timeout:
                self.get_logger().error("⏱️ No se recibió pose en el tiempo límite. Publicando zeros y apagando timer.")
                self.publish_zero_times(10)
                rclpy.shutdown()
            return

        # Si no hay centro (no se ejecuta por seguridad), esperar
        if self.center is None:
            return

        # Si ya completó el círculo y regresó al centro, detener todo
        if self.finished_circle and self.returning_to_center is False:
            # activar fase de regreso al centro
            self.get_logger().info("🔁 Círculos completados: iniciando regreso al centro.")
            self.returning_to_center = True

        # Si estamos en fase de trazar el círculo
        if not self.returning_to_center:
            # avanzar theta según ω*dt (evita drift con timestamp real)
            dtheta = self.omega * self.dt
            self.theta += dtheta
            self.total_theta += dtheta

            # calcular setpoint en el perímetro
            cx, cy, cz = self.center
            target_x = cx + self.R * math.cos(self.theta)
            target_y = cy + self.R * math.sin(self.theta)
            target_z = cz  # mantener altura constante (puedes cambiar)

            # feedforward tangential velocity (derivada del parámetro)
            # derivada de [R cos, R sin] = R * ω * [-sin, cos]
            v_ff_x = - self.R * self.omega * math.sin(self.theta)
            v_ff_y =   self.R * self.omega * math.cos(self.theta)

            # error de posición
            cur_x, cur_y, cur_z = self.pose
            err_x = target_x - cur_x
            err_y = target_y - cur_y

            # derivada (D)
            derr_x = (err_x - self.last_error[0]) / self.dt
            derr_y = (err_y - self.last_error[1]) / self.dt

            # PD correction
            corr_x = self.Kp * err_x + self.Kd * derr_x
            corr_y = self.Kp * err_y + self.Kd * derr_y

            # comando de velocidad = feedforward + corrección
            cmd_x = v_ff_x + corr_x
            cmd_y = v_ff_y + corr_y

            # limitar magnitud
            mag = math.hypot(cmd_x, cmd_y)
            if mag > self.max_speed:
                scale = self.max_speed / mag
                cmd_x *= scale
                cmd_y *= scale

            # publicar
            msg = Twist()
            msg.linear.x = float(cmd_x)
            msg.linear.y = float(cmd_y)
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            self.pub.publish(msg)

            # guardar error para D
            self.last_error = (err_x, err_y)

            # detectar vuelta completa
            if self.total_theta >= 2.0 * math.pi * self.loops:
                self.get_logger().info("✅ Objetivo angular logrado (vueltas completadas). Parando perímetro.")
                self.finished_circle = True
                # parar momentáneamente feedforward y dejar que el controlador finalice
                self.theta = 0.0
                self.total_theta = 0.0
                # publicar unos zeros cortos para marcar transición
                self.publish_zero_times(int(self.rate * 0.05))
        else:
            # Fase de regresar al centro: controlador proporcional simple (con saturación)
            cur_x, cur_y, cur_z = self.pose
            cx, cy, cz = self.center
            err_x = cx - cur_x
            err_y = cy - cur_y
            dist_to_center = math.hypot(err_x, err_y)

            if dist_to_center <= self.center_tolerance:
                self.get_logger().info(f"🏁 Centro alcanzado (dist={dist_to_center:.3f} m). Publicando zeros y cerrando.")
                self.publish_zero_times(int(self.rate * 0.2))
                # shutdown ROS de forma ordenada
                self.destroy_node()
                rclpy.shutdown()
                return

            # controlador simple P para volver al centro
            cmd_x = self.Kp * err_x
            cmd_y = self.Kp * err_y
            mag = math.hypot(cmd_x, cmd_y)
            if mag > self.max_speed:
                scale = self.max_speed / mag
                cmd_x *= scale
                cmd_y *= scale

            msg = Twist()
            msg.linear.x = float(cmd_x)
            msg.linear.y = float(cmd_y)
            msg.linear.z = 0.0
            msg.angular.z = 0.0
            self.pub.publish(msg)

    def publish_zero_times(self, n):
        zero = Twist()
        for _ in range(n):
            self.pub.publish(zero)
            time.sleep(self.dt)

def main(args=None):
    rclpy.init(args=args)
    node = MavrosCircleClosedLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupción por teclado — deteniendo y publicando zeros.")
        node.publish_zero_times(10)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()

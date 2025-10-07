#!/usr/bin/env python3

import rclpy, math, tf_transformations, sys, termios, tty, select
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import SetParametersResult
from tf2_ros import TransformBroadcaster

class ControlDrone(Node):
    def __init__(self):
        super().__init__("ControlDrone")
        self.get_logger().info("Control drone Started")

        self.odom_sub = self.create_subscription(Odometry, "/r1/odom", self.odom_cbk, 1)
        self.cmd_vel_pub = self.create_publisher(Twist, "/r1/cmd_vel", 1)
        self.error_pub = self.create_publisher(Twist, "/drone/errors", 10)
        self.leader_sub = self.create_subscription(PoseStamped, "/leader/setpoint", self.leader_cbk, 10)

        self.timer = self.create_timer(0.1, self.ControlDrone_cbk)
        self.state = "stop"
        self.xyz, self.rpy, self.err_r = None, None, 10000
        self.t0 = self.get_clock().now().nanoseconds / 1e9

        self.leader_pose = None

        self.declare_parameter("k_gain", 0.4)
        self.k = self.get_parameter("k_gain").value
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.tf_b = TransformBroadcaster(self)
        self.tf_msg = TransformStamped()
        self.tf_msg.header.frame_id = "r1/odom"
        self.tf_msg.child_frame_id = "r1/base_footprint"

        self.drone_path_msg = Path()
        self.traje_path_msg = Path()
        self.drone_path_msg.header.frame_id = self.tf_msg.header.frame_id
        self.traje_path_msg.header.frame_id = self.tf_msg.header.frame_id
        self.pub_drone_path = self.create_publisher(Path, "r1/drone_path",1)
        self.pub_traje_path = self.create_publisher(Path, "r1/traje_path",1)

    def parameters_callback(self, params):
        for param in params:
            if param.name == "k_gain":
                self.k = param.value
                self.get_logger().info(f"Ganancia actualizada: {self.k}")
        return SetParametersResult(successful=True)

    def getKey(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ""
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def moveDrone(self, vx, vy, vz, wz):
        cmd_vel = Twist()
        cmd_vel.linear.x = vx
        cmd_vel.linear.y = vy
        cmd_vel.linear.z = vz
        cmd_vel.angular.z = wz
        self.cmd_vel_pub.publish(cmd_vel)

    def leader_cbk(self, msg):
        self.leader_pose = msg.pose.position

    def controlDrone(self):
        if self.leader_pose is None or self.xyz is None:
            return

        # Usar exclusivamente la posición del líder como setpoint
        Xd = self.leader_pose.x
        Yd = self.leader_pose.y
        Zd = self.leader_pose.z

        # Cálculo de errores
        err_x = Xd - self.xyz[0]
        err_y = Yd - self.xyz[1]
        err_z = Zd - self.xyz[2]
        err_yaw = 0.0  # No usamos orientación del líder

        # Ley de control proporcional
        Ux = self.k * err_x
        Uy = self.k * err_y

        # Transformación a sistema local del dron
        Vx = Ux * math.cos(self.yaw) + Uy * math.sin(self.yaw)
        Vy = -Ux * math.sin(self.yaw) + Uy * math.cos(self.yaw)
        Vz = self.k * err_z
        Wz = 0.0

        self.moveDrone(Vx, Vy, Vz, Wz)

        # Publicar errores
        error_msg = Twist()
        error_msg.linear.x = err_x
        error_msg.linear.y = err_y
        error_msg.linear.z = err_z
        self.error_pub.publish(error_msg)

        # Publicar trayectoria del líder como referencia
        traje_pose = PoseStamped()
        traje_pose.pose.position = self.leader_pose
        self.traje_path_msg.poses.append(traje_pose)

        self.pub_drone_path.publish(self.drone_path_msg)
        self.pub_traje_path.publish(self.traje_path_msg)

    def ControlDrone_cbk(self):
        key = self.getKey().lower()
        if key == "q":
            sys.exit(0)

        if key == "c":
            del self.drone_path_msg.poses[:]
            del self.traje_path_msg.poses[:]
            self.get_logger().info("Paths limpiados")

        # Máquina de estados
        if self.state == "stop" and key == "t":
            self.state = "takeOff"
            print("Modo: Despegue")
        elif self.state == "takeOff" and self.xyz and self.xyz[2] > 1.1:
            self.state = "hover"
            print("Modo: Hover")
        elif self.state == "hover" and key == "f":
            self.state = "follow"
            print("Modo: Siguiendo al líder")
        elif key == "l":
            self.state = "landing"
            print("Modo: Aterrizaje")
        elif self.state == "landing" and self.xyz and self.xyz[2] < 0.2:
            self.state = "stop"
            print("Modo: Detenido")

        # Comportamiento según estado
        if self.state == "stop":
            self.moveDrone(0.0, 0.0, 0.0, 0.0)
        elif self.state == "takeOff":
            self.moveDrone(0.0, 0.0, 0.3, 0.0)
        elif self.state == "hover":
            self.moveDrone(0.0, 0.0, 0.0, 0.0)
        elif self.state == "landing":
            self.moveDrone(0.0, 0.0, -0.3, 0.0)
        elif self.state == "follow":
            self.controlDrone()

    def odom_cbk(self, msg):
        self.xyz = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        r, p, y = tf_transformations.euler_from_quaternion(q)
        self.yaw = y
        self.rpy = [math.degrees(r), math.degrees(p), math.degrees(y)]

        self.tf_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_msg.transform.translation.x = self.xyz[0]
        self.tf_msg.transform.translation.y = self.xyz[1]
        self.tf_msg.transform.translation.z = self.xyz[2]
        self.tf_msg.transform.rotation = msg.pose.pose.orientation
        self.tf_b.sendTransform(self.tf_msg)

        drone_pose = PoseStamped()
        drone_pose.pose.position = msg.pose.pose.position
        self.drone_path_msg.poses.append(drone_pose)

def main(arg=None):
    rclpy.init(args=arg)
    nodeh = ControlDrone()
    try:
        rclpy.spin(nodeh)
    except Exception as error:
        print(error)
    except KeyboardInterrupt:
        print("Node terminated by user")

if __name__ == "__main__":
    main()

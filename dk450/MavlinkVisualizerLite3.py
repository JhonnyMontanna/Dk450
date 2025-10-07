#!/usr/bin/env python3
import rclpy
import threading
import math
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Quaternion, PointStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

class InteractiveDroneControl(Node):
    def __init__(self):
        super().__init__('interactive_drone_control')

        # RTK
        self.lat0 = math.radians(19.5951756)
        self.lon0 = math.radians(-99.2279035)
        self.h0 = 2334.38

        # 19.5951756 -99.2279035 2334.38

        #calibracion
        self.lat_ref = math.radians(19.5951686) 
        self.lon_ref = math.radians(-99.2279168)
        self.h_ref = self.h0

        # 19.5951686 -99.2279168 2334.38
        # 19.59517814 -99.22787466 2339

        # Coordenada esperada del punto de calibración en el sistema local (por defecto: (-1,1))
        self.expected_local_x = 1
        self.expected_local_y = 1

        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(self.lat0, self.lon0, self.h0)
        self.X_ref, self.Y_ref, self.Z_ref = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, self.h_ref)

        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        self.theta = self.compute_rotation_angle_definitiva()

        self.odom_pub = self.create_publisher(Odometry, "/r1/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.coord_pub = self.create_publisher(String, "/r1/coordinates", 10)
        self.position_pub = self.create_publisher(PointStamped, "/r1/position", 10)
        self.marker_pub = self.create_publisher(Marker, "/r1/marker_text", 10)

        self.quat_msg = Quaternion(w=1.0)
        self.timer = self.create_timer(0.1, self.publish_data)
        self.running = True
        self.x = self.y = self.z = 0.0

        self.input_thread = threading.Thread(target=self.read_input)
        self.input_thread.start()
        self.get_logger().info("Sistema listo - Ingrese lat lon alt o 'exit'")

    def read_input(self):
        while self.running:
            try:
                user_input = input("Coordenadas (lat lon alt) o 'exit': ")
                if user_input.lower() == 'exit':
                    self.running = False
                    break

                coords = list(map(float, user_input.strip().split()))
                if len(coords) == 3:
                    lat_rad = math.radians(coords[0])
                    lon_rad = math.radians(coords[1])
                    alt = coords[2]
                    X, Y, Z = self.geodetic_to_ecef(lat_rad, lon_rad, alt)
                    dX, dY, dZ = X - self.X0, Y - self.Y0, Z - self.Z0
                    enu = self.R_enu @ np.array([[dX], [dY], [dZ]])
                    enu_corrected = self.rotate_enu_definitiva(enu)

                    self.x = enu_corrected[0, 0]
                    self.y = enu_corrected[1, 0]
                    self.z = enu[2, 0]  # Z sin rotar
                    self.get_logger().info(f"Posición ENU corregida: X={self.x:.2f}, Y={self.y:.2f}, Z={self.z:.2f}")
                else:
                    print("Formato inválido. Use: lat lon alt")
            except Exception as e:
                print(f"Error: {e}. Intente nuevamente.")

    def publish_data(self):
        if not self.running:
            self.shutdown()
            return

        timestamp = self.get_clock().now().to_msg()

        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = "r1/odom"
        odom.child_frame_id = "r1/drone"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation = self.quat_msg
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = timestamp
        tf.header.frame_id = "r1/odom"
        tf.child_frame_id = "r1/drone"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = self.z
        tf.transform.rotation = self.quat_msg
        self.tf_broadcaster.sendTransform(tf)

        marker = Marker()
        marker.header.stamp = timestamp
        marker.header.frame_id = "r1/drone"
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.z = 0.5
        marker.scale.z = 0.2
        marker.color.r = marker.color.g = marker.color.b = marker.color.a = 1.0
        marker.text = f"X={self.x:.2f}, Y={self.y:.2f}, Z={self.z:.2f}"
        self.marker_pub.publish(marker)

    def geodetic_to_ecef(self, lat, lon, h):
        a = 6378137.0
        e2 = 0.00669437999014
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        X = (N + h) * math.cos(lat) * math.cos(lon)
        Y = (N + h) * math.cos(lat) * math.sin(lon)
        Z = (N * (1 - e2) + h) * math.sin(lat)
        return X, Y, Z

    def get_rotation_matrix(self, lat, lon):
        return np.array([
            [-math.sin(lon),               math.cos(lon),              0],
            [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
            [math.cos(lat)*math.cos(lon),  math.cos(lat)*math.sin(lon),  math.sin(lat)]
        ])

    def compute_rotation_angle_definitiva(self):
        dXr = self.X_ref - self.X0
        dYr = self.Y_ref - self.Y0
        dZr = self.Z_ref - self.Z0
        enu_ref = self.R_enu @ np.array([[dXr], [dYr], [dZr]])
        E_ref, N_ref = enu_ref[0, 0], enu_ref[1, 0]
        theta_expected = math.atan2(self.expected_local_y, self.expected_local_x)
        theta_measured = math.atan2(N_ref, E_ref)
        theta = theta_expected - theta_measured
        return theta

    def rotate_enu_definitiva(self, enu):
        E, N = enu[0, 0], enu[1, 0]
        R_corr = np.array([
            [math.cos(self.theta), -math.sin(self.theta)],
            [math.sin(self.theta),  math.cos(self.theta)]
        ])
        rotated = R_corr @ np.array([[E], [N]])
        return np.array([[rotated[0, 0]], [rotated[1, 0]], [enu[2, 0]]])

    def shutdown(self):
        self.get_logger().info("Apagando sistema...")
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveDroneControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()

if __name__ == '__main__':
    main()

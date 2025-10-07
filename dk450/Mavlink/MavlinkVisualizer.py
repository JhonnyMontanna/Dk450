#!/usr/bin/env python3
import rclpy
import math
import tf_transformations
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from pymavlink import mavutil
from tf2_ros import TransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA

# --------------------- WGS-84 --------------------- #
WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014e-3

class MavlinkVisualizer(Node):
    def __init__(self):
        super().__init__('mavlink_visualizer')

        # ------------- PARÁMETROS RTK ------------- #
        # Origen RTK (antena base)
        self.lat0 = math.radians(19.5951349)
        self.lon0 = math.radians(-99.2278921)
        self.h0   = 2333.35

        # Punto de calibración (geodésico)
        self.lat_ref = math.radians(19.5951348)
        self.lon_ref = math.radians(-99.2279023)
        self.h_ref   = self.h0

        # Vector esperado en sistema local
        self.expected_local_x = 1.0
        self.expected_local_y = 0.0

        # Fuentes de altitud: 'lidar', 'computed', 'ned'
        self.altitude_mode = 'lidar'  # Cambiar segun necesidad

        # Precompute ECEF & RTK rotation
        self.X0, self.Y0, self.Z0     = self.geodetic_to_ecef(math.degrees(self.lat0), math.degrees(self.lon0), self.h0)
        self.X_ref, self.Y_ref, self.Z_ref = self.geodetic_to_ecef(math.degrees(self.lat_ref), math.degrees(self.lon_ref), self.h_ref)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)
        self.theta = self.compute_rotation_angle()

        # LIDAR placeholder (metros)
        self.lidar_z = 0.0

        # --------- ROS 2 pubs & TF --------- #
        qos_be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom_pub   = self.create_publisher(Odometry, '/gps/odom', qos_be)
        self.path_pub   = self.create_publisher(Path,     '/gps/drone_path', qos_rel)
        self.marker_pub = self.create_publisher(Marker,   '/gps/drone_marker', qos_rel)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.drone_path = Path()
        self.drone_path.header.frame_id = 'rtk_odom'

        # TF frames
        self.tf_link = TransformStamped()
        self.tf_link.header.frame_id = 'rtk_odom'
        self.tf_link.child_frame_id = 'base_link'
        self.tf_foot = TransformStamped()
        self.tf_foot.header.frame_id = 'base_link'
        self.tf_foot.child_frame_id = 'base_footprint'

        self.quat_msg = Quaternion()

        # MAVLink setup
        self.mavlink_connect()
        self.set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 200000)
        self.set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,            200000)
        self.set_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR,     200000)

        # Poll
        self.create_timer(0.02, self.read_mavlink)

    def compute_rotation_angle(self):
        dx = self.X_ref - self.X0
        dy = self.Y_ref - self.Y0
        dz = self.Z_ref - self.Z0
        enu_ref = self.R_enu.dot([dx, dy, dz])
        east, north = enu_ref[0], enu_ref[1]
        theta_expected = math.atan2(self.expected_local_y, self.expected_local_x)
        theta_measured = math.atan2(north, east)
        return theta_expected - theta_measured

    def geodetic_to_ecef(self, lat, lon, alt):
        # lat, lon en grados
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
        x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
        y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
        z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_r)
        return x, y, z

    def get_rotation_matrix(self, lat_r, lon_r):
        return np.array([
            [-math.sin(lon_r),               math.cos(lon_r),               0],
            [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r),  math.cos(lat_r)],
            [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)]
        ])

    def mavlink_connect(self):
        self.master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
        self.master.wait_heartbeat(timeout=5)
        self.get_logger().info('MAVLink conectado')

    def set_message_interval(self, msg_id, interval_us):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, msg_id, interval_us, 0, 0, 0, 0, 0
        )

    def read_mavlink(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                return
            m = msg.get_type()
            if m == 'GLOBAL_POSITION_INT':
                self.handle_position(msg)
            elif m == 'ATTITUDE':
                self.handle_attitude(msg)
            elif m == 'DISTANCE_SENSOR':
                raw = getattr(msg, 'current_distance', getattr(msg, 'distance', 0))
                # msg.distance en cm a m
                self.lidar_z = float(raw) / 100.0

    def handle_position(self, msg):
        # decode
        lat = msg.lat * 1e-7
        lon = msg.lon * 1e-7
        alt_gps = msg.alt * 1e-3
        alt_ned = msg.relative_alt * 1e-3
        # compute XY from GPS altitude
        x_e, y_e, z_e = self.geodetic_to_ecef(lat, lon, alt_gps)
        d = np.array([x_e - self.X0, y_e - self.Y0, z_e - self.Z0])
        enu = self.R_enu.dot(d)
        xy = np.array([enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta),
                       enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)])
        xr, yr = xy[0], xy[1]
        # select Z
        if self.altitude_mode == 'lidar':
            zr = self.lidar_z
        elif self.altitude_mode == 'computed':
            zr = float(enu[2])
        elif self.altitude_mode == 'ned':
            zr = alt_ned
        else:
            zr = alt_gps

        ts = self.get_clock().now().to_msg()
        # Log and print to terminal
        self.get_logger().info(
            f"Lat:{lat:.7f} Lon:{lon:.7f} X_RTK:{xr:.2f} Y_RTK:{yr:.2f} Z:{zr:.2f}"
        )
        print(f"Lat={lat:.7f}, Lon={lon:.7f}, x={xr:.2f}, y={yr:.2f}, z={zr:.2f}")

        # Odometry
        self.tf_link.header.stamp = ts
        self.tf_link.transform.translation.x = float(xr)
        self.tf_link.transform.translation.y = float(yr)
        self.tf_link.transform.translation.z = float(zr)
        self.tf_link.transform.rotation = self.quat_msg
        self.tf_broadcaster.sendTransform(self.tf_link)

        self.tf_foot.header.stamp = ts
        self.tf_foot.transform.translation.z = float(-zr)
        self.tf_foot.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(self.tf_foot)

                # Update Path
        pose = PoseStamped()
        pose.header.stamp = ts
        pose.header.frame_id = 'rtk_odom'
        pose.pose.position.x = float(xr)
        pose.pose.position.y = float(yr)
        pose.pose.position.z = float(zr)
        self.drone_path.header.stamp = ts
        self.drone_path.poses.append(pose)
        self.path_pub.publish(self.drone_path)

        # Publish Marker Text
        marker = Marker()
        marker.header.stamp = ts
        marker.header.frame_id = 'rtk_odom'
        marker.ns = 'drone'
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(xr)
        marker.pose.position.y = float(yr)
        marker.pose.position.z = float(zr) + 1.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5
        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.text = f"x={xr:.2f}, y={yr:.2f}, z={zr:.2f}"
        self.marker_pub.publish(marker)

    def handle_attitude(self, msg):
        q = tf_transformations.quaternion_from_euler(msg.roll, msg.pitch, msg.yaw)
        self.quat_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def main():
    rclpy.init()
    node = MavlinkVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

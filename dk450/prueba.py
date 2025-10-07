#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, Imu, Range
from std_msgs.msg import Float64, ColorRGBA
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

class MultiDroneVisualizer(Node):
    def __init__(self):
        super().__init__('multi_drone_visualizer')
        # ---------------- Parameters ---------------- #
        self.declare_parameter('drone_ns', 'uav1')
        # RTK origin & calibration
        # Declare RTK origin and calibration parameters
        self.declare_parameter('origin_lat', 19.5942341)
        self.declare_parameter('origin_lon', -99.2280871)
        self.declare_parameter('origin_alt', 2329.0)
        self.declare_parameter('calib_lat', 19.5942429)
        self.declare_parameter('calib_lon', -99.2280774)
        self.declare_parameter('calib_alt', 2329.0)
        # Altitude source
        self.declare_parameter('altitude_mode', 'ned')    # 'lidar' | 'computed' | 'ned'
        # Calibration mode
        self.declare_parameter('calib_mode', 'angle')       # 'enu' | 'angle' | 'pair'
        self.declare_parameter('calib_ang', 180.0)          # degrees, CCW
        self.declare_parameter('expected_local_x', 1.0)     # used if calib_mode=='pair'
        self.declare_parameter('expected_local_y', 1.0)

        # === Covarianza visible (3σ + tamaño del dron) ===
        self.declare_parameter('cov_mode', 'fix')           # 'fix' | 'float' | 'gps'
        self.declare_parameter('confidence_k', 3.0)         # 3σ fijo
        self.declare_parameter('include_drone_size', True)
        self.declare_parameter('drone_radius_xy', 0.55)     # m
        self.declare_parameter('drone_half_height_z', 0.20) # m
        self.declare_parameter('show_ellipse_marker', True)

        # σ por modo (ajústalo si quieres)
        self.declare_parameter('sigma_fix_x', 0.016)
        self.declare_parameter('sigma_fix_y', 0.014)
        self.declare_parameter('sigma_fix_z', 0.050)
        self.declare_parameter('sigma_float_x', 0.080)
        self.declare_parameter('sigma_float_y', 0.070)
        self.declare_parameter('sigma_float_z', 0.250)
        self.declare_parameter('sigma_gps_x', 0.240)
        self.declare_parameter('sigma_gps_y', 0.210)
        self.declare_parameter('sigma_gps_z', 0.750)

        # Var. orientación (si no modelas, grandes)
        self.declare_parameter('var_roll', 1e6)
        self.declare_parameter('var_pitch', 1e6)
        self.declare_parameter('var_yaw', 1e6)

        # Read parameters
        self.ns   = self.get_parameter('drone_ns').get_parameter_value().string_value
        o_lat     = self.get_parameter('origin_lat').get_parameter_value().double_value
        o_lon     = self.get_parameter('origin_lon').get_parameter_value().double_value
        self.h0   = self.get_parameter('origin_alt').get_parameter_value().double_value
        c_lat     = self.get_parameter('calib_lat').get_parameter_value().double_value
        c_lon     = self.get_parameter('calib_lon').get_parameter_value().double_value
        self.h_ref= self.get_parameter('calib_alt').get_parameter_value().double_value
        self.altitude_mode = self.get_parameter('altitude_mode').get_parameter_value().string_value
        self.calib_mode    = self.get_parameter('calib_mode').get_parameter_value().string_value
        self.calib_ang     = self.get_parameter('calib_ang').get_parameter_value().double_value
        self.expected_local_x = self.get_parameter('expected_local_x').get_parameter_value().double_value
        self.expected_local_y = self.get_parameter('expected_local_y').get_parameter_value().double_value

        if self.altitude_mode not in ('lidar','computed','ned'):
            self.get_logger().warn(f"altitude_mode '{self.altitude_mode}' no válido, usando 'ned'")
            self.altitude_mode = 'ned'
        if self.calib_mode not in ('enu','angle','pair'):
            self.get_logger().warn(f"calib_mode '{self.calib_mode}' no válido, usando 'pair'")
            self.calib_mode = 'pair'

        # Convert to radians
        self.lat0    = math.radians(o_lat)
        self.lon0    = math.radians(o_lon)
        self.lat_ref = math.radians(c_lat)
        self.lon_ref = math.radians(c_lon)

        # Precompute frames: ECEF & ENU
        self.X0, self.Y0, self.Z0   = self.geodetic_to_ecef(self.lat0,    self.lon0,    self.h0)
        self.Xr, self.Yr, self.Zr   = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, self.h_ref)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        # Heading correction
        self.theta = self.compute_theta()

        # Sensor placeholders
        self.latest_quat = Quaternion()
        self.lidar_z     = 0.0
        self.rel_alt     = 0.0

        # QoS
        qos_be  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers (Odometry ahora RELIABLE)
        self.tf_br      = TransformBroadcaster(self)
        self.odom_pub   = self.create_publisher(Odometry, f'/{self.ns}/gps/odom', qos_rel)
        self.path_pub   = self.create_publisher(Path,     f'/{self.ns}/gps/drone_path', qos_rel)
        self.marker_pub = self.create_publisher(Marker,   f'/{self.ns}/gps/drone_marker', qos_rel)

        # Path
        self.drone_path = Path()
        self.drone_path.header.frame_id = 'rtk_odom'

        # Subs
        self.create_subscription(NavSatFix, f'/{self.ns}/global_position/global', self.cb_navsat, qos_be)
        self.create_subscription(Float64,   f'/{self.ns}/global_position/rel_alt', self.cb_rel_alt, qos_be)
        self.create_subscription(Imu,       f'/{self.ns}/imu/data',                 self.cb_imu,     qos_be)
        self.create_subscription(Range,     f'/{self.ns}/rangefinder/rangefinder',  self.cb_lidar,   qos_be)

        # Log config
        self.get_logger().info(
            f"NS={self.ns} | Origen: lat={o_lat:.7f}, lon={o_lon:.7f}, alt={self.h0:.2f} | "
            f"Calib: lat={c_lat:.7f}, lon={c_lon:.7f}, alt={self.h_ref:.2f} | "
            f"calib_mode={self.calib_mode}, theta={math.degrees(self.theta):.2f}° | alt_mode={self.altitude_mode}"
        )

        # Debug ticker para no saturar logs
        self._last_debug_log = self.get_clock().now()

    # ---------------- Cov utils ---------------- #
    def _get_sigmas_for_mode(self):
        mode = self.get_parameter('cov_mode').get_parameter_value().string_value
        try:
            sx = self.get_parameter(f'sigma_{mode}_x').get_parameter_value().double_value
            sy = self.get_parameter(f'sigma_{mode}_y').get_parameter_value().double_value
            sz = self.get_parameter(f'sigma_{mode}_z').get_parameter_value().double_value
        except Exception:
            self.get_logger().warn(f"cov_mode '{mode}' inválido. Usando 'fix'.")
            sx = self.get_parameter('sigma_fix_x').get_parameter_value().double_value
            sy = self.get_parameter('sigma_fix_y').get_parameter_value().double_value
            sz = self.get_parameter('sigma_fix_z').get_parameter_value().double_value
        return sx, sy, sz

    def compute_axes_and_cov(self):
        """Devuelve (ax, ay, az, cov6x6_rowmajor) para RViz."""
        k = self.get_parameter('confidence_k').get_parameter_value().double_value  # 3.0
        include_size = self.get_parameter('include_drone_size').get_parameter_value().bool_value
        r_xy = self.get_parameter('drone_radius_xy').get_parameter_value().double_value
        r_z  = self.get_parameter('drone_half_height_z').get_parameter_value().double_value
        sx, sy, sz = self._get_sigmas_for_mode()

        ax = math.sqrt((k*sx)**2 + (r_xy**2 if include_size else 0.0))
        ay = math.sqrt((k*sy)**2 + (r_xy**2 if include_size else 0.0))
        az = math.sqrt((k*sz)**2 + (r_z**2  if include_size else 0.0))

        vroll  = self.get_parameter('var_roll').get_parameter_value().double_value
        vpitch = self.get_parameter('var_pitch').get_parameter_value().double_value
        vyaw   = self.get_parameter('var_yaw').get_parameter_value().double_value

        cov = [0.0]*36
        cov[0]  = ax*ax
        cov[7]  = ay*ay
        cov[14] = az*az
        cov[21] = vroll
        cov[28] = vpitch
        cov[35] = vyaw
        return ax, ay, az, cov

    # ---------------- Callbacks ---------------- #
    def cb_navsat(self, msg: NavSatFix):
        # Geodetic -> ENU
        lat_r, lon_r, alt_gps = math.radians(msg.latitude), math.radians(msg.longitude), msg.altitude
        Xe, Ye, Ze = self.geodetic_to_ecef(lat_r, lon_r, alt_gps)
        d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
        enu = self.R_enu.dot(d)

        # Rotate to RTK local
        xr = enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta)
        yr = enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)

        # Altitude selection
        if   self.altitude_mode == 'lidar':    zr = self.lidar_z
        elif self.altitude_mode == 'computed': zr = float(enu[2])
        else:                                  zr = self.rel_alt

        stamp = self.get_clock().now().to_msg()

        # TF: base_link at UAV position
        tf_link = TransformStamped()
        tf_link.header.stamp = stamp
        tf_link.header.frame_id    = 'rtk_odom'
        tf_link.child_frame_id     = f'{self.ns}_base_link'
        tf_link.transform.translation.x = float(xr)
        tf_link.transform.translation.y = float(yr)
        tf_link.transform.translation.z = float(zr)
        tf_link.transform.rotation      = self.latest_quat
        self.tf_br.sendTransform(tf_link)

        # TF: base_footprint at ground (0)
        tf_foot = TransformStamped()
        tf_foot.header.stamp = stamp
        tf_foot.header.frame_id    = 'rtk_odom'
        tf_foot.child_frame_id     = f'{self.ns}_base_footprint'
        tf_foot.transform.translation.x = float(xr)
        tf_foot.transform.translation.y = float(yr)
        tf_foot.transform.translation.z = 0.0
        tf_foot.transform.rotation.w    = 1.0
        self.tf_br.sendTransform(tf_foot)

        # Odometry
        ax, ay, az, cov = self.compute_axes_and_cov()

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'rtk_odom'
        odom.child_frame_id  = f'{self.ns}_base_link'
        odom.pose.pose.position.x = float(xr)
        odom.pose.pose.position.y = float(yr)
        odom.pose.pose.position.z = float(zr)
        odom.pose.pose.orientation = self.latest_quat
        odom.pose.covariance = cov
        self.odom_pub.publish(odom)

        # Debug log (cada ~2 s)
        now = self.get_clock().now()
        if (now - self._last_debug_log).nanoseconds > 2_000_000_000:
            self.get_logger().info(f"[{self.ns}] Var(x,y,z)=({cov[0]:.3f}, {cov[7]:.3f}, {cov[14]:.3f})  -> a=(%.3f, %.3f, %.3f)m" % (ax, ay, az))
            self._last_debug_log = now

        # Path
        if self.drone_path.header.frame_id != 'rtk_odom':
            self.drone_path.header.frame_id = 'rtk_odom'
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose   = odom.pose.pose
        self.drone_path.header.stamp = stamp
        self.drone_path.poses.append(pose)
        self.path_pub.publish(self.drone_path)

        # Marker de texto
        text = Marker()
        text.header    = odom.header
        text.ns        = f'{self.ns}_text'
        text.id        = 0
        text.type      = Marker.TEXT_VIEW_FACING
        text.action    = Marker.ADD
        text.pose.position.x = float(xr)
        text.pose.position.y = float(yr)
        text.pose.position.z = float(zr) + 1.0
        text.pose.orientation.w = 1.0
        text.scale.z   = 0.5
        text.color     = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text      = f"{self.ns}: x={xr:.1f}, y={yr:.1f}, z={zr:.1f}"
        self.marker_pub.publish(text)

        # Marker elipsoide (plan B siempre visible)
        if self.get_parameter('show_ellipse_marker').get_parameter_value().bool_value:
            ell = Marker()
            ell.header = odom.header
            ell.ns = f'{self.ns}_ellipse'
            ell.id = 1
            ell.type = Marker.SPHERE
            ell.action = Marker.ADD
            ell.pose.position.x = float(xr)
            ell.pose.position.y = float(yr)
            ell.pose.position.z = float(zr)
            ell.pose.orientation.w = 1.0
            # scale son DIÁMETROS (2*a_i)
            ell.scale.x = 2.0 * ax
            ell.scale.y = 2.0 * ay
            ell.scale.z = 2.0 * az
            mode = self.get_parameter('cov_mode').get_parameter_value().string_value
            if mode == 'fix':
                ell.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.25)
            elif mode == 'float':
                ell.color = ColorRGBA(r=1.0, g=0.65, b=0.0, a=0.25)
            else:
                ell.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.25)
            self.marker_pub.publish(ell)

    def cb_rel_alt(self, msg: Float64):
        self.rel_alt = msg.data

    def cb_imu(self, msg: Imu):
        self.latest_quat = msg.orientation

    def cb_lidar(self, msg: Range):
        self.lidar_z = float(msg.range)

    # ---------------- Math ---------------- #
    def compute_theta(self):
        mode = self.calib_mode
        if mode == 'enu':
            return 0.0
        elif mode == 'angle':
            return math.radians(self.calib_ang)
        else:  # 'pair'
            dx, dy, dz = self.Xr - self.X0, self.Yr - self.Y0, self.Zr - self.Z0
            ref = self.R_enu.dot([dx, dy, dz])
            east, north = float(ref[0]), float(ref[1])
            theta_measured = math.atan2(north, east)
            theta_expected = math.atan2(self.expected_local_y, self.expected_local_x)
            return theta_expected - theta_measured

    def geodetic_to_ecef(self, lat_r, lon_r, alt):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
        x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
        y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
        z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_r)
        return x, y, z

    def get_rotation_matrix(self, lat_r, lon_r):
        return np.array([
            [-math.sin(lon_r),                 math.cos(lon_r),                 0],
            [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r), math.cos(lat_r)],
            [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)]
        ])


def main():
    rclpy.init()
    node = MultiDroneVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

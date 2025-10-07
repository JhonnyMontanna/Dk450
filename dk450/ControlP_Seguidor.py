#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix, Range
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped

# Constantes WGS84
WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014e-3

class AbsoluteProportionalFollower(Node):
    """
    Seguidor proporcional en coordenadas RTK-ENU + altura RAW:
    - X, Y: suscribe NavSatFix del líder y seguidor, transforma a ENU-RTK.
    - Z: suscribe rel_alt (Float64) del líder y seguidor, usa lectura directa.
    - Calcula control P con offsets en x,y,z.
    - Publica velocidades en cmd_vel_unstamped (Twist) y cmd_vel (TwistStamped).
    Asume OFFBOARD y armado previos.
    """
    def __init__(self):
        super().__init__('absolute_proportional_follower')
        # Parámetros
        p = self.declare_parameter
        p('drone_ns',    'uav2'); p('leader_ns', 'uav1')
        p('origin_lat',   19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat',    19.5942429); p('calib_lon', -99.2280774); p('calib_alt', 2329.0)
        p('offset_x',     1.0); p('offset_y', 0.0); p('offset_z', 0.5)
        p('kp',           0.5); p('rate',      20)

        # Leer parámetros
        self.ns        = self.get_parameter('drone_ns').value
        self.leader_ns = self.get_parameter('leader_ns').value
        origin_lat     = self.get_parameter('origin_lat').value
        origin_lon     = self.get_parameter('origin_lon').value
        origin_alt     = self.get_parameter('origin_alt').value
        calib_lat      = self.get_parameter('calib_lat').value
        calib_lon      = self.get_parameter('calib_lon').value
        calib_alt      = self.get_parameter('calib_alt').value
        self.offset    = (
            self.get_parameter('offset_x').value,
            self.get_parameter('offset_y').value,
            self.get_parameter('offset_z').value
        )
        self.kp        = self.get_parameter('kp').value
        rate           = self.get_parameter('rate').value
        self.dt        = 1.0 / float(rate)

        # RTK: origen y calibración
        lat0    = math.radians(origin_lat)
        lon0    = math.radians(origin_lon)
        lat_ref = math.radians(calib_lat)
        lon_ref = math.radians(calib_lon)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(lat0, lon0, origin_alt)
        Xr, Yr, Zr              = self.geodetic_to_ecef(lat_ref, lon_ref, calib_alt)
        self.R_enu  = self.get_rotation_matrix(lat0, lon0)
        self.theta  = self.compute_calibration_angle(Xr, Yr, Zr)

        # Estados
        self.leader_enu      = None
        self.follower_enu    = None
        self.leader_rel_alt  = 0.0
        self.follower_rel_alt= 0.0
        self.leader_ready    = False
        self.follower_ready  = False
        self.leader_alt_ready   = False
        self.follower_alt_ready = False

        # QoS
        gps_qos  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST)
        alt_qos  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST)

        # Subscripciones ENU
        self.create_subscription(
            NavSatFix,
            f'/{self.leader_ns}/global_position/global',
            self.cb_leader_gps,
            gps_qos)
        self.create_subscription(
            NavSatFix,
            f'/{self.ns}/global_position/global',
            self.cb_follower_gps,
            gps_qos)
        # Subscripciones ALT
        self.create_subscription(
            Float64,
            f'/{self.leader_ns}/global_position/rel_alt',
            self.cb_leader_alt,
            alt_qos)
        self.create_subscription(
            Float64,
            f'/{self.ns}/global_position/rel_alt',
            self.cb_follower_alt,
            alt_qos)

        # Publishers de velocidad
        self.pub_twist = self.create_publisher(
            Twist,        f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts    = self.create_publisher(
            TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel',     10)

        # Timer de control
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"[INIT] follower=/{self.ns}, leader=/{self.leader_ns}, "
            f"offset={self.offset}, kp={self.kp}, rate={rate}Hz"
        )

    def cb_leader_gps(self, msg: NavSatFix):
        enu = self.to_rtk_enu(msg)
        self.leader_enu = enu
        if not self.leader_ready:
            self.get_logger().info(f"[LEADER ENU] {enu}")
        self.leader_ready = True

    def cb_follower_gps(self, msg: NavSatFix):
        enu = self.to_rtk_enu(msg)
        self.follower_enu = enu
        if not self.follower_ready:
            self.get_logger().info(f"[FOLLOWER ENU] {enu}")
        self.follower_ready = True

    def cb_leader_alt(self, msg: Float64):
        self.leader_rel_alt = msg.data
        self.leader_alt_ready = True

    def cb_follower_alt(self, msg: Float64):
        self.follower_rel_alt = msg.data
        self.follower_alt_ready = True

    def to_rtk_enu(self, msg: NavSatFix):
        lat = math.radians(msg.latitude)
        lon = math.radians(msg.longitude)
        alt = msg.altitude
        X, Y, Z = self.geodetic_to_ecef(lat, lon, alt)
        d = np.array([X-self.X0, Y-self.Y0, Z-self.Z0])
        enu = self.R_enu.dot(d)
        # calibración XY
        x =  enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta)
        y =  enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)
        # enu[2] disponible pero no usado
        return [x, y]

    def control_loop(self):
        # Esperar readiness completo
        if not (self.leader_ready and self.follower_ready \
                and self.leader_alt_ready and self.follower_alt_ready):
            return

        # Deseado XY + offset
        xd = self.leader_enu[0] + self.offset[0]
        yd = self.leader_enu[1] + self.offset[1]
        # Deseado Z + offset
        zd = self.leader_rel_alt + self.offset[2]

        # Error
        ex = xd - self.follower_enu[0]
        ey = yd - self.follower_enu[1]
        ez = zd - self.follower_rel_alt

        # Control P
        vx = self.kp * ex
        vy = self.kp * ey
        vz = self.kp * ez

        # Debug
        self.get_logger().info(
            f"[DEBUG] ex={ex:.2f}, ey={ey:.2f}, ez={ez:.2f} | "
            f"vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}"
        )

        # Publish velocities
        cmd = Twist(); cmd.linear.x=vx; cmd.linear.y=vy; cmd.linear.z=vz; cmd.angular.z=0.0
        self.pub_twist.publish(cmd)
        ts = TwistStamped(); ts.header.stamp=self.get_clock().now().to_msg(); ts.twist=cmd
        self.pub_ts.publish(ts)

    def compute_calibration_angle(self, Xr, Yr, Zr):
        ref = self.R_enu.dot([Xr-self.X0, Yr-self.Y0, Zr-self.Z0])
        theta_exp = math.atan2(1.0,1.0)
        theta_meas= math.atan2(ref[1], ref[0])
        return theta_exp - theta_meas

    @staticmethod
    def geodetic_to_ecef(lat, lon, alt):
        N = WGS84_A/math.sqrt(1-WGS84_E2*math.sin(lat)**2)
        return ((N+alt)*math.cos(lat)*math.cos(lon),
                (N+alt)*math.cos(lat)*math.sin(lon),
                (N*(1-WGS84_E2)+alt)*math.sin(lat))

    @staticmethod
    def get_rotation_matrix(lat, lon):
        return np.array([
            [-math.sin(lon),                math.cos(lon),               0],
            [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
            [ math.cos(lat)*math.cos(lon),  math.cos(lat)*math.sin(lon), math.sin(lat)]
        ])


def main():
    rclpy.init()
    node = AbsoluteProportionalFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import NavSatFix

# Usar las mismas constantes WGS84 que tu visualizador (orden de magnitud correcto)
WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

class CircleRTKAccurate(Node):
    def __init__(self):
        super().__init__('circle_rtk_accurate')

        p = self.declare_parameter
        # Dinámica / misión
        p('drone_ns', 'uav1')
        p('radius', 2.0)
        p('angular_speed', 0.8)
        p('rate', 20)
        p('loops', 1)
        p('kp', 0.8)
        p('max_speed', 3.0)
        p('center_tolerance', 0.2)

        # RTK origin & calibration (usar los tuyos)
        p('origin_lat', 19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat', 19.5942429);  p('calib_lon', -99.2280774);  p('calib_alt', 2329.0)
        p('calib_mode', 'pair')   # 'enu'|'angle'|'pair'
        p('calib_ang', 180.0)     # deg si mode=='angle'
        p('expected_local_x', 1.0); p('expected_local_y', 1.0)

        # leer params
        self.ns = self.get_parameter('drone_ns').value
        self.R  = float(self.get_parameter('radius').value)
        self.omega = float(self.get_parameter('angular_speed').value)
        self.rate  = int(self.get_parameter('rate').value)
        self.loops = int(self.get_parameter('loops').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.center_tol = float(self.get_parameter('center_tolerance').value)
        self.calib_mode = self.get_parameter('calib_mode').value
        self.calib_ang  = float(self.get_parameter('calib_ang').value)
        self.expected_local_x = float(self.get_parameter('expected_local_x').value)
        self.expected_local_y = float(self.get_parameter('expected_local_y').value)

        # RTK origin & calib
        o_lat = self.get_parameter('origin_lat').value
        o_lon = self.get_parameter('origin_lon').value
        o_alt = self.get_parameter('origin_alt').value
        c_lat = self.get_parameter('calib_lat').value
        c_lon = self.get_parameter('calib_lon').value
        c_alt = self.get_parameter('calib_alt').value

        # convert to radians where required
        self.lat0 = math.radians(o_lat); self.lon0 = math.radians(o_lon)
        self.lat_ref = math.radians(c_lat); self.lon_ref = math.radians(c_lon)

        # ECEF & ENU precalc (definitivo)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(self.lat0, self.lon0, o_alt)
        self.Xr, self.Yr, self.Zr = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, c_alt)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        # compute theta using same logic que tu visualizer
        self.calib_mode = self.calib_mode if self.calib_mode in ('enu','angle','pair') else 'pair'
        self.theta = self.compute_theta()

        # estado
        self.center = None       # [x,y] ENU local (rotado) fijado en primera lectura
        self.current = None      # [x,y]
        self.theta_phase = 0.0
        self.total_theta = 0.0
        self.returning = False

        self.dt = 1.0 / float(self.rate)

        # qos
        gps_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # topics (igual que tu visualizer)
        pose_topic = f'/{self.ns}/global_position/global'
        self.create_subscription(NavSatFix, pose_topic, self.cb_navsat, gps_qos)

        # publishers de velocidad (match a tu stack)
        self.pub_twist = self.create_publisher(Twist, f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel', 10)

        # timer
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"[INIT] {self.ns} R={self.R} ω={self.omega} theta_cal={math.degrees(self.theta):.2f}°")

    # --- Callbacks ---
    def cb_navsat(self, msg: NavSatFix):
        # convert geodetic -> ECEF -> ENU -> rotate by theta (RTK local)
        lat_r = math.radians(msg.latitude); lon_r = math.radians(msg.longitude); alt = msg.altitude
        Xe, Ye, Ze = self.geodetic_to_ecef(lat_r, lon_r, alt)
        d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
        enu = self.R_enu.dot(d)
        xr = enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta)
        yr = enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)

        self.current = [float(xr), float(yr)]
        if self.center is None:
            self.center = [float(xr), float(yr)]
            self.get_logger().info(f"Centro fijado en ENU: {self.center}")

    # --- Control loop ---
    def control_loop(self):
        if self.current is None or self.center is None:
            return

        if not self.returning:
            dtheta = self.omega * self.dt
            self.theta_phase += dtheta
            self.total_theta += dtheta

            cx, cy = self.center
            tx = cx + self.R * math.cos(self.theta_phase)
            ty = cy + self.R * math.sin(self.theta_phase)

            # feedforward tangential:
            vff_x = - self.R * self.omega * math.sin(self.theta_phase)
            vff_y =   self.R * self.omega * math.cos(self.theta_phase)

            # error radial
            curx, cury = self.current
            errx = tx - curx
            erry = ty - cury

            corr_x = self.kp * errx
            corr_y = self.kp * erry

            cmd_x = vff_x + corr_x
            cmd_y = vff_y + corr_y

            # saturación
            mag = math.hypot(cmd_x, cmd_y)
            if mag > self.max_speed:
                s = self.max_speed / mag
                cmd_x *= s; cmd_y *= s

            # publicar (Twist y TwistStamped)
            t = Twist(); t.linear.x=cmd_x; t.linear.y=cmd_y; t.linear.z=0.0; t.angular.z=0.0
            self.pub_twist.publish(t)
            ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = t
            self.pub_ts.publish(ts)

            if self.total_theta >= 2.0*math.pi*self.loops:
                self.get_logger().info("Vueltas completadas — iniciando retorno.")
                self.returning = True
                self.total_theta = 0.0
                self.publish_zeroes(int(self.rate * 0.05))
        else:
            # regresar al centro
            cx, cy = self.center
            curx, cury = self.current
            errx = cx - curx; erry = cy - cury
            dist = math.hypot(errx, erry)
            if dist <= self.center_tol:
                self.get_logger().info(f"Centro alcanzado (dist={dist:.3f}). Publicando zeros.")
                self.publish_zeroes(int(self.rate * 0.2))
                return
            cmd_x = self.kp * errx; cmd_y = self.kp * erry
            mag = math.hypot(cmd_x, cmd_y)
            if mag > self.max_speed:
                s = self.max_speed / mag
                cmd_x *= s; cmd_y *= s
            t = Twist(); t.linear.x=cmd_x; t.linear.y=cmd_y; t.linear.z=0.0; t.angular.z=0.0
            self.pub_twist.publish(t)
            ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = t
            self.pub_ts.publish(ts)

    def publish_zeroes(self, n):
        zero = Twist()
        for _ in range(n):
            self.pub_twist.publish(zero)
            ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = zero
            self.pub_ts.publish(ts)
            time.sleep(self.dt)

    # --- utilidades (idénticas a tu visualizer) ---
    def compute_theta(self):
        mode = self.calib_mode
        if mode == 'enu':
            return 0.0
        elif mode == 'angle':
            return math.radians(self.calib_ang)
        else:  # pair
            dx, dy, dz = self.Xr - self.X0, self.Yr - self.Y0, self.Zr - self.Z0
            ref = self.R_enu.dot([dx, dy, dz])
            east, north = float(ref[0]), float(ref[1])
            theta_measured = math.atan2(north, east)
            theta_expected = math.atan2(self.expected_local_y, self.expected_local_x)
            return theta_expected - theta_measured

    @staticmethod
    def geodetic_to_ecef(lat_r, lon_r, alt):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
        x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
        y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
        z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_r)
        return x, y, z

    @staticmethod
    def get_rotation_matrix(lat_r, lon_r):
        return np.array([
            [-math.sin(lon_r),                 math.cos(lon_r),                 0],
            [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r), math.cos(lat_r)],
            [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)]
        ])

def main():
    rclpy.init()
    node = CircleRTKAccurate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import NavSatFix

WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

class CircleRTKSequence(Node):
    def __init__(self):
        super().__init__('circle_rtk_sequence')

        p = self.declare_parameter
        # misión
        p('drone_ns', 'uav1')
        p('radius', 2.0)
        p('angular_speed', 0.8)
        p('rate', 20)
        p('loops', 1)
        p('kp', 0.8)               # controlador radial durante círculo
        p('goto_kp', 0.8)         # P para goto (lineal)
        p('max_speed', 2.0)
        p('goto_tol', 0.2)
        p('center_tol', 0.2)

        # RTK origin & calib (mantén los tuyos)
        p('origin_lat', 19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat', 19.5942429);  p('calib_lon', -99.2280774);  p('calib_alt', 2329.0)
        p('calib_mode', 'pair')
        p('calib_ang', 180.0)
        p('expected_local_x', 1.0); p('expected_local_y', 1.0)

        # leer params
        self.ns = self.get_parameter('drone_ns').value
        self.R  = float(self.get_parameter('radius').value)
        self.omega = float(self.get_parameter('angular_speed').value)
        self.rate = int(self.get_parameter('rate').value)
        self.loops = int(self.get_parameter('loops').value)
        self.kp = float(self.get_parameter('kp').value)
        self.goto_kp = float(self.get_parameter('goto_kp').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.goto_tol = float(self.get_parameter('goto_tol').value)
        self.center_tol = float(self.get_parameter('center_tol').value)

        # RTK
        o_lat = self.get_parameter('origin_lat').value
        o_lon = self.get_parameter('origin_lon').value
        o_alt = self.get_parameter('origin_alt').value
        c_lat = self.get_parameter('calib_lat').value
        c_lon = self.get_parameter('calib_lon').value
        c_alt = self.get_parameter('calib_alt').value
        self.calib_mode = self.get_parameter('calib_mode').value
        self.calib_ang = float(self.get_parameter('calib_ang').value)
        self.expected_local_x = float(self.get_parameter('expected_local_x').value)
        self.expected_local_y = float(self.get_parameter('expected_local_y').value)

        self.lat0 = math.radians(o_lat); self.lon0 = math.radians(o_lon)
        self.lat_ref = math.radians(c_lat); self.lon_ref = math.radians(c_lon)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(self.lat0, self.lon0, o_alt)
        self.Xr, self.Yr, self.Zr = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, c_alt)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        self.calib_mode = self.calib_mode if self.calib_mode in ('enu','angle','pair') else 'pair'
        self.theta_cal = self.compute_theta()

        # estado
        self.home = None          # home ENU [x,y]
        self.current = None       # current ENU [x,y]
        self.start_point = None   # punto inicial del perímetro [x,y]
        self.state = 'wait_home'  # 'wait_home'|'goto_start'|'circle'|'return_home'|'done'
        self.theta_phase = 0.0
        self.total_theta = 0.0

        self.dt = 1.0 / float(self.rate)

        gps_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_topic = f'/{self.ns}/global_position/global'
        self.create_subscription(NavSatFix, pose_topic, self.cb_navsat, gps_qos)

        self.pub_twist = self.create_publisher(Twist, f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel', 10)

        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(f"[INIT] {self.ns} R={self.R} ω={self.omega} state={self.state}")

    # ---------- Callbacks ----------
    def cb_navsat(self, msg: NavSatFix):
        lat_r = math.radians(msg.latitude); lon_r = math.radians(msg.longitude); alt = msg.altitude
        Xe, Ye, Ze = self.geodetic_to_ecef(lat_r, lon_r, alt)
        d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
        enu = self.R_enu.dot(d)
        xr = enu[0]*math.cos(self.theta_cal) - enu[1]*math.sin(self.theta_cal)
        yr = enu[0]*math.sin(self.theta_cal) + enu[1]*math.cos(self.theta_cal)
        self.current = [float(xr), float(yr)]
        if self.home is None:
            # fijar home en la primera lectura
            self.home = [float(xr), float(yr)]
            # definir start point en (0, R) relativo a home: interpretamos (0,R) como offset en Y local
            self.start_point = [self.home[0] + 0.0, self.home[1] + self.R]
            self.get_logger().info(f"Home fijado: {self.home} | Start_point (0,R): {self.start_point}")
            # cambiar inmediatamente a goto_start
            self.state = 'goto_start'

    # ---------- Controladores ----------
    def publish_cmd(self, vx, vy):
        # saturación
        mag = math.hypot(vx, vy)
        if mag > self.max_speed:
            s = self.max_speed / mag
            vx *= s; vy *= s
        t = Twist(); t.linear.x = float(vx); t.linear.y = float(vy); t.linear.z = 0.0; t.angular.z = 0.0
        self.pub_twist.publish(t)
        ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = t
        self.pub_ts.publish(ts)

    def control_loop(self):
        if self.current is None or self.home is None:
            return

        if self.state == 'goto_start':
            # control P simple hacia start_point
            tx, ty = self.start_point
            cx, cy = self.current
            ex, ey = tx - cx, ty - cy
            dist = math.hypot(ex, ey)
            vx = self.goto_kp * ex
            vy = self.goto_kp * ey
            self.publish_cmd(vx, vy)
            # condición de llegada
            if dist <= self.goto_tol:
                self.get_logger().info(f"Llegó a start_point (dist={dist:.3f}). Iniciando círculo.")
                # fijar fase para comenzar en (0,R) con θ = π/2
                self.theta_phase = math.pi/2.0
                self.total_theta = 0.0
                self.state = 'circle'
                # publicar unos zeros cortos antes de empezar para estabilizar
                self.publish_zeroes(int(self.rate * 0.05))
            return

        if self.state == 'circle':
            # avanzar fase
            dtheta = self.omega * self.dt
            self.theta_phase += dtheta
            self.total_theta += dtheta

            cx, cy = self.home  # círculo centrado en home
            tx = cx + self.R * math.cos(self.theta_phase)
            ty = cy + self.R * math.sin(self.theta_phase)

            # feedforward tangential
            vff_x = - self.R * self.omega * math.sin(self.theta_phase)
            vff_y =   self.R * self.omega * math.cos(self.theta_phase)

            curx, cury = self.current
            errx, erry = tx - curx, ty - cury

            corr_x = self.kp * errx
            corr_y = self.kp * erry

            cmd_x = vff_x + corr_x
            cmd_y = vff_y + corr_y

            self.publish_cmd(cmd_x, cmd_y)

            if self.total_theta >= 2.0 * math.pi * self.loops:
                self.get_logger().info("Vueltas completadas. Iniciando retorno a home.")
                self.state = 'return_home'
                self.publish_zeroes(int(self.rate * 0.05))
            return

        if self.state == 'return_home':
            # P hacia home
            tx, ty = self.home
            cx, cy = self.current
            ex, ey = tx - cx, ty - cy
            dist = math.hypot(ex, ey)
            vx = self.goto_kp * ex
            vy = self.goto_kp * ey
            self.publish_cmd(vx, vy)
            if dist <= self.center_tol:
                self.get_logger().info(f"Home alcanzado (dist={dist:.3f}). Parando.")
                self.publish_zeroes(int(self.rate * 0.2))
                self.state = 'done'
            return

        if self.state == 'done':
            # ya detenido; publicar zeros por seguridad
            self.publish_cmd(0.0, 0.0)
            return

    def publish_zeroes(self, n):
        zero = Twist()
        for _ in range(n):
            self.pub_twist.publish(zero)
            ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = zero
            self.pub_ts.publish(ts)
            time.sleep(self.dt)

    # --- utilidades geodésicas (idénticas) ---
    def compute_theta(self):
        mode = self.calib_mode
        if mode == 'enu':
            return 0.0
        elif mode == 'angle':
            return math.radians(self.calib_ang)
        else:
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
    node = CircleRTKSequence()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

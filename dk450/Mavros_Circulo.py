#!/usr/bin/env python3
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import NavSatFix

# Constantes WGS84 (asegúrate que coinciden con tus otros nodos)
WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

class SimpleCircleSinglePoint(Node):
    """
    Traza solo un círculo centrado en 'center' calculado para que
    la POSICIÓN INICIAL del dron esté sobre el perímetro (sin gotos).
    - center = current - R * [cos(theta0), sin(theta0)]
    - se fija theta = theta0 y se avanza omega * dt hasta completar 2π*loops.
    - al finalizar publica zeros y para. No vuelve al centro.
    """
    def __init__(self):
        super().__init__('simple_circle_single_point')

        p = self.declare_parameter
        p('drone_ns', 'uav1')
        p('radius', 2.0)
        p('angular_speed', 0.6)   # rad/s
        p('rate', 20)             # Hz del controlador
        p('loops', 1)
        p('kp', 0.6)              # corrección radial P (pequeña)
        p('max_speed', 2.0)
        p('start_angle_deg', 0.0) # ángulo θ0 en grados: 0 => dron en +X relativo al centro

        # RTK origin & calib params (ajusta a los tuyos)
        p('origin_lat', 19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat', 19.5942429);  p('calib_lon', -99.2280774);  p('calib_alt', 2329.0)
        p('calib_mode', 'pair')
        p('calib_ang', 180.0)
        p('expected_local_x', 1.0); p('expected_local_y', 1.0)

        # leer params
        self.ns = self.get_parameter('drone_ns').value
        self.R = float(self.get_parameter('radius').value)
        self.omega = float(self.get_parameter('angular_speed').value)
        self.rate = int(self.get_parameter('rate').value)
        self.loops = int(self.get_parameter('loops').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.start_angle_deg = float(self.get_parameter('start_angle_deg').value)

        # RTK origin/calib
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

        # convertir a radianes para ECEF/ENU
        self.lat0 = math.radians(o_lat); self.lon0 = math.radians(o_lon)
        self.lat_ref = math.radians(c_lat); self.lon_ref = math.radians(c_lon)

        # ECEF & ENU precalc
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(self.lat0, self.lon0, o_alt)
        self.Xr, self.Yr, self.Zr = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, c_alt)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        # theta de calibración
        self.calib_mode = self.calib_mode if self.calib_mode in ('enu','angle','pair') else 'pair'
        self.theta_cal = self.compute_theta()

        # estados
        self.current = None     # posición ENU actual [x,y]
        self.center = None      # centro calculado para que current esté en el perímetro
        self.theta_phase = None # fase actual, inicializada a theta0
        self.total_theta = 0.0
        self.dt = 1.0 / float(self.rate)
        self.started = False
        self.finished = False

        # QoS y topics
        gps_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_topic = f'/{self.ns}/global_position/global'
        self.create_subscription(NavSatFix, pose_topic, self.cb_navsat, gps_qos)

        self.pub_twist = self.create_publisher(Twist, f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel', 10)

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"[INIT] {self.ns} R={self.R} ω={self.omega} start_ang={self.start_angle_deg}° rate={self.rate}Hz")

    # ---------------- callbacks ----------------
    def cb_navsat(self, msg: NavSatFix):
        lat_r = math.radians(msg.latitude); lon_r = math.radians(msg.longitude); alt = msg.altitude
        Xe, Ye, Ze = self.geodetic_to_ecef(lat_r, lon_r, alt)
        d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
        enu = self.R_enu.dot(d)
        xr = enu[0]*math.cos(self.theta_cal) - enu[1]*math.sin(self.theta_cal)
        yr = enu[0]*math.sin(self.theta_cal) + enu[1]*math.cos(self.theta_cal)
        self.current = [float(xr), float(yr)]

        # en la primera lectura definimos center de forma que current esté en el perímetro
        if not self.started and self.current is not None:
            theta0 = math.radians(self.start_angle_deg)
            # center = current - R * [cos(theta0), sin(theta0)]
            cx = self.current[0] - self.R * math.cos(theta0)
            cy = self.current[1] - self.R * math.sin(theta0)
            self.center = [float(cx), float(cy)]
            self.theta_phase = theta0
            self.total_theta = 0.0
            self.started = True
            self.get_logger().info(f"Inicio: current={self.current} | center calculado={self.center} | theta0={math.degrees(theta0):.1f}°")

    # ---------------- control loop ----------------
    def control_loop(self):
        if not self.started or self.finished:
            return
        if self.current is None or self.center is None:
            return

        # avanzar fase
        dtheta = self.omega * self.dt
        self.theta_phase += dtheta
        self.total_theta += dtheta

        cx, cy = self.center
        tx = cx + self.R * math.cos(self.theta_phase)
        ty = cy + self.R * math.sin(self.theta_phase)

        # feedforward tangential (v = R * ω)
        vff_x = - self.R * self.omega * math.sin(self.theta_phase)
        vff_y =   self.R * self.omega * math.cos(self.theta_phase)

        curx, cury = self.current
        errx, erry = tx - curx, ty - cury

        # corrección radial P
        corr_x = self.kp * errx
        corr_y = self.kp * erry

        cmd_x = vff_x + corr_x
        cmd_y = vff_y + corr_y

        # saturación
        mag = math.hypot(cmd_x, cmd_y)
        if mag > self.max_speed and mag > 0.0:
            s = self.max_speed / mag
            cmd_x *= s; cmd_y *= s

        # publicar
        t = Twist(); t.linear.x = float(cmd_x); t.linear.y = float(cmd_y); t.linear.z = 0.0; t.angular.z = 0.0
        self.pub_twist.publish(t)
        ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = t
        self.pub_ts.publish(ts)

        # condición de finalización: completó vueltas exactas → parar en el punto inicial
        if self.total_theta >= 2.0 * math.pi * self.loops:
            # publicar zeros suficientes para estabilizar en el punto final (debería coincidir con start)
            nzeros = int(self.rate * 0.5)  # 0.5 s de zeros
            zero = Twist()
            for _ in range(nzeros):
                self.pub_twist.publish(zero)
                ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = zero
                self.pub_ts.publish(ts)
                time.sleep(self.dt)
            self.get_logger().info("✅ Vueltas completadas — detenido en punto inicial (misma ubicación de inicio).")
            self.finished = True
            return

    # ---------------- utilidades geodésicas ----------------
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
    node = SimpleCircleSinglePoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

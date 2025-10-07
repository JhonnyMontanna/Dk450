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

def angle_wrap(a):
    """envuelve ángulo a (-pi, pi]"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

class CircleRTKPhaseLock(Node):
    """
    Circle follower con sincronización de fase (phase-lock) y compensación de delay.
    Basado en tu CircleRTKDirect, añade:
     - k_sync: ganancia de sincronización de fase
     - command_delay: retardo estimado (s) para compensar
     - max_accel: límite simple de aceleración (m/s^2) para suavizar comandos
    """
    def __init__(self):
        super().__init__('circle_rtk_phase_lock')

        p = self.declare_parameter
        # misión básica
        p('drone_ns', 'uav1')
        p('radius', 2.0)
        p('angular_speed', 1.2)   # prueba altas velocidades
        p('rate', 50)             # subir rate ayuda
        p('loops', 1)
        p('kp', 0.9)
        p('max_speed', 4.0)
        p('stop_zero_time', 0.5)

        # sincronización y compensación
        p('k_sync', 1.0)          # ganancia de phase-lock (0.2..2.0 típico)
        p('command_delay', 0.06)  # segundos (latencia estimada desde input->act)
        p('max_accel', 1.5)       # m/s^2, límite de aceleración de los comandos lineales

        # RTK / calib (tus valores)
        p('origin_lat', 19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat', 19.5942429);  p('calib_lon', -99.2280774); p('calib_alt', 2329.0)
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
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.stop_zero_time = float(self.get_parameter('stop_zero_time').value)

        self.k_sync = float(self.get_parameter('k_sync').value)
        self.command_delay = float(self.get_parameter('command_delay').value)
        self.max_accel = float(self.get_parameter('max_accel').value)

        # RTK precalc
        o_lat = self.get_parameter('origin_lat').value
        o_lon = self.get_parameter('origin_lon').value
        o_alt = self.get_parameter('origin_alt').value
        c_lat = self.get_parameter('calib_lat').value
        c_lon = self.get_parameter('calib_lon').value
        c_alt = self.get_parameter('calib_alt').value

        self.lat0 = math.radians(o_lat); self.lon0 = math.radians(o_lon)
        self.lat_ref = math.radians(c_lat); self.lon_ref = math.radians(c_lon)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(self.lat0, self.lon0, o_alt)
        self.Xr, self.Yr, self.Zr = self.geodetic_to_ecef(self.lat_ref, self.lon_ref, c_alt)
        self.R_enu = self.get_rotation_matrix(self.lat0, self.lon0)

        self.calib_mode = self.get_parameter('calib_mode').value
        self.calib_ang = float(self.get_parameter('calib_ang').value)
        self.expected_local_x = float(self.get_parameter('expected_local_x').value)
        self.expected_local_y = float(self.get_parameter('expected_local_y').value)
        self.calib_mode = self.calib_mode if self.calib_mode in ('enu','angle','pair') else 'pair'
        self.theta_cal = self.compute_theta()

        # estados
        self.current = None
        self.center = None
        self.theta_phase = 0.0
        self.total_theta = 0.0
        self.state = 'wait_pose'  # 'wait_pose'|'running'|'done'

        # para limitación de aceleración: guardar última velocidad publicada
        self.last_cmd_vx = 0.0
        self.last_cmd_vy = 0.0

        self.dt = 1.0 / float(self.rate)

        gps_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        pose_topic = f'/{self.ns}/global_position/global'
        self.create_subscription(NavSatFix, pose_topic, self.cb_navsat, gps_qos)

        self.pub_twist = self.create_publisher(Twist, f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel', 10)

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"[INIT] {self.ns} R={self.R} ω={self.omega} rate={self.rate}Hz k_sync={self.k_sync} delay={self.command_delay}s")

    def cb_navsat(self, msg: NavSatFix):
        lat_r = math.radians(msg.latitude); lon_r = math.radians(msg.longitude); alt = msg.altitude
        Xe, Ye, Ze = self.geodetic_to_ecef(lat_r, lon_r, alt)
        d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
        enu = self.R_enu.dot(d)
        xr = enu[0]*math.cos(self.theta_cal) - enu[1]*math.sin(self.theta_cal)
        yr = enu[0]*math.sin(self.theta_cal) + enu[1]*math.cos(self.theta_cal)
        self.current = [float(xr), float(yr)]

        if self.center is None:
            # fijar centro = current - (0,R) para que current==(0,R)
            self.center = [self.current[0], self.current[1] - self.R]
            self.theta_phase = math.pi/2.0
            self.total_theta = 0.0
            self.state = 'running'
            self.get_logger().info(f"Pose inicial. Centro={self.center}, theta_cal={math.degrees(self.theta_cal):.2f}°")

    def control_loop(self):
        if self.state == 'wait_pose' or self.current is None:
            return

        if self.state == 'running':
            cx, cy = self.center
            curx, cury = self.current

            # --- estimación de fase real desde posición ---
            theta_meas = math.atan2(cury - cy, curx - cx)   # en (-pi, pi]
            # unwrapping: queremos error continuo respecto a theta_phase
            # angle_error = smallest difference (theta_meas - theta_phase)
            angle_error = angle_wrap(theta_meas - self.theta_phase)

            # --- advance desired phase with feedforward + phase-lock correction + delay comp ---
            # feedforward increment
            feed_dtheta = self.omega * self.dt
            # phase-lock term
            sync_term = self.k_sync * angle_error
            # apply delay compensation by advancing expected phase by omega*command_delay
            delay_comp = self.omega * self.command_delay
            # update
            self.theta_phase += feed_dtheta + sync_term + delay_comp * (1.0 / max(1, int(self.rate * self.command_delay))) 
            # note: delay_comp scaled mildly so it doesn't blow up; main correction is sync_term

            # keep total_theta for loop counting (integrate feed_dtheta only to count real rotations)
            self.total_theta += feed_dtheta

            # target point on perimeter (use theta_phase as desired)
            tx = cx + self.R * math.cos(self.theta_phase)
            ty = cy + self.R * math.sin(self.theta_phase)

            # feedforward tangential velocity (R*omega * [-sin, cos])
            vff_x = - self.R * self.omega * math.sin(self.theta_phase)
            vff_y =   self.R * self.omega * math.cos(self.theta_phase)

            # radial error
            errx = tx - curx
            erry = ty - cury

            corr_x = self.kp * errx
            corr_y = self.kp * erry

            cmd_x = vff_x + corr_x
            cmd_y = vff_y + corr_y

            # --- aceleración limitada (simple) ---
            # compute desired acceleration (approx)
            desired_ax = (cmd_x - self.last_cmd_vx) / self.dt
            desired_ay = (cmd_y - self.last_cmd_vy) / self.dt
            a_norm = math.hypot(desired_ax, desired_ay)
            if a_norm > self.max_accel:
                # scale accel
                scale = self.max_accel / a_norm
                cmd_x = self.last_cmd_vx + desired_ax * scale * self.dt
                cmd_y = self.last_cmd_vy + desired_ay * scale * self.dt

            # saturación en magnitud
            mag = math.hypot(cmd_x, cmd_y)
            if mag > self.max_speed and mag > 0.0:
                s = self.max_speed / mag
                cmd_x *= s; cmd_y *= s

            # publicar
            self.publish_cmd(cmd_x, cmd_y)

            # guardar último comando
            self.last_cmd_vx = cmd_x
            self.last_cmd_vy = cmd_y

            # condición de finalización (completó vueltas)
            if self.total_theta >= 2.0 * math.pi * self.loops:
                self.get_logger().info("Vueltas completadas. Publicando zeros y deteniendo.")
                # publicar zeros un tiempo para parar
                n = int(self.rate * self.stop_zero_time)
                zero = Twist()
                for _ in range(max(1, n)):
                    self.pub_twist.publish(zero)
                    ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = zero
                    self.pub_ts.publish(ts)
                    time.sleep(self.dt)
                self.state = 'done'
            return

        if self.state == 'done':
            # zeros periódicos
            zero = Twist()
            self.pub_twist.publish(zero)
            ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = zero
            self.pub_ts.publish(ts)
            return

    def publish_cmd(self, vx, vy):
        t = Twist(); t.linear.x = float(vx); t.linear.y = float(vy); t.linear.z = 0.0; t.angular.z = 0.0
        self.pub_twist.publish(t)
        ts = TwistStamped(); ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = t
        self.pub_ts.publish(ts)

    # --- utilidades geo idénticas ---
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
    node = CircleRTKPhaseLock()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()

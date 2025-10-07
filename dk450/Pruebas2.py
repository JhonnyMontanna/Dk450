#!/usr/bin/env python3
import math
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
import matplotlib.pyplot as plt

# ----------  WGS-84  ----------
WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014e-3

def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_pi(a): return (a + math.pi) % (2*math.pi) - math.pi

def hdg_deg_to_yaw_enu_rad(hdg_deg: float) -> float:
    # 0°=Norte, 90°=Este (CW) -> ENU yaw (0=Este, CCW+)
    return wrap_pi(math.radians(90.0 - hdg_deg))

def yaw_from_quat(x, y, z, w) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return wrap_pi(math.atan2(siny_cosp, cosy_cosp))


class RelativePIDFollower(Node):
    """
    Seguidor PID con offset relativo:
      - Offset en coordenadas CILÍNDRICAS: r, theta, z (nuevo)
        * 'body': theta=0 hacia x_forward del líder, CCW hacia y_left
        * 'world': theta=0 hacia +X ENU (Este), CCW hacia +Y (Norte)
      - Soporte retro-compatibilidad con modo CARTESIANO (offset_x,y,z)
      - Control PID en x,y,z y yaw
    """

    def __init__(self):
        super().__init__('relative_pid_follower')

        # ---------- Parámetros ----------
        p = self.declare_parameter
        p('drone_ns',    'uav2'); p('leader_ns',    'uav1')

        # RTK base y punto de calibración
        p('origin_lat',  19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat',   19.5942429); p('calib_lon', -99.2280774); p('calib_alt', 2329.0)

        # ---- Offset: modo y frame ----
        p('offset_mode',  'cylindrical')      # 'cylindrical' | 'cartesian'
        p('offset_frame', 'body')             # 'body' | 'world'

        # Cilíndrico (nuevo)
        p('offset_r',           1.0)          # metros
        p('offset_theta_deg',   0.0)          # grados
        p('offset_z',           0.5)          # metros

        # Cartesiano (retro-compat)
        p('offset_x',    1.0); p('offset_y',  0.0); p('offset_z_cart', 0.5)

        # PID lineal
        p('pid_kp',      0.6); p('pid_ki',    0.0); p('pid_kd',  0.12)
        # PID angular
        p('kp_yaw',      1.2); p('ki_yaw',    0.0); p('kd_yaw',  0.05)
        p('yaw_offset_deg', 0.0)
        # Límites y frecuencia
        p('vmax_xy',     2.0); p('vmax_z',    1.0); p('wmax',    1.0)
        p('rate',        20)
        # Fuente de yaw
        p('yaw_source', 'auto')               # 'compass' | 'pose' | 'auto'

        # ---------- Lectura ----------
        self.ns        = self.get_parameter('drone_ns').value
        self.leader_ns = self.get_parameter('leader_ns').value

        origin_lat = self.get_parameter('origin_lat').value
        origin_lon = self.get_parameter('origin_lon').value
        origin_alt = self.get_parameter('origin_alt').value
        calib_lat  = self.get_parameter('calib_lat').value
        calib_lon  = self.get_parameter('calib_lon').value
        calib_alt  = self.get_parameter('calib_alt').value

        self.offset_mode  = self.get_parameter('offset_mode').value.lower()
        self.offset_frame = self.get_parameter('offset_frame').value.lower()

        # Cilíndrico
        self.off_r     = float(self.get_parameter('offset_r').value)
        self.off_th    = math.radians(float(self.get_parameter('offset_theta_deg').value))
        self.off_z     = float(self.get_parameter('offset_z').value)
        # Cartesiano (por compatibilidad)
        self.off_x     = float(self.get_parameter('offset_x').value)
        self.off_y     = float(self.get_parameter('offset_y').value)
        self.off_z_cart= float(self.get_parameter('offset_z_cart').value)

        # PID
        self.kp  = float(self.get_parameter('pid_kp').value)
        self.ki  = float(self.get_parameter('pid_ki').value)
        self.kd  = float(self.get_parameter('pid_kd').value)

        self.kp_yaw = float(self.get_parameter('kp_yaw').value)
        self.ki_yaw = float(self.get_parameter('ki_yaw').value)
        self.kd_yaw = float(self.get_parameter('kd_yaw').value)
        self.yaw_offset = math.radians(float(self.get_parameter('yaw_offset_deg').value))

        self.vmax_xy = float(self.get_parameter('vmax_xy').value)
        self.vmax_z  = float(self.get_parameter('vmax_z').value)
        self.wmax    = float(self.get_parameter('wmax').value)
        rate         = float(self.get_parameter('rate').value)
        self.dt      = 1.0 / max(1.0, rate)

        self.yaw_source = self.get_parameter('yaw_source').value.lower()

        # ---------- RTK & calibración ----------
        lat0 = math.radians(origin_lat); lon0 = math.radians(origin_lon)
        lat_ref = math.radians(calib_lat); lon_ref = math.radians(calib_lon)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(lat0, lon0, origin_alt)
        Xr, Yr, Zr = self.geodetic_to_ecef(lat_ref, lon_ref, calib_alt)
        self.R_enu = self.get_rotation_matrix(lat0, lon0)
        self.theta = self.compute_calibration_angle(Xr, Yr, Zr)

        # ---------- Estado ----------
        self.leader_xy = np.zeros(2)
        self.follow_xy = np.zeros(2)
        self.leader_alt = 0.0
        self.follow_alt = 0.0
        self.leader_yaw = 0.0
        self.follow_yaw = 0.0

        self.leader_gps_ready = False
        self.follow_gps_ready = False
        self.leader_alt_ready = False
        self.follow_alt_ready = False
        self.leader_yaw_ready = False
        self.follow_yaw_ready = False

        self.have_leader_compass = False
        self.have_follow_compass = False
        self.have_leader_pose    = False
        self.have_follow_pose    = False

        self.prev_error_xyz = np.zeros(3)
        self.int_xyz = np.zeros(3)
        self.prev_eyaw = 0.0
        self.int_yaw = 0.0

        # ---------- Plot ----------
        plt.ion()
        self.fig, self.axes = plt.subplots(4, 1, figsize=(8, 8))
        self.ax_x, self.ax_y, self.ax_z, self.ax_yaw = self.axes
        self.times = []
        self.xd_hist, self.x_hist = [], []
        self.yd_hist, self.y_hist = [], []
        self.zd_hist, self.z_hist = [], []
        self.psid_hist, self.psi_hist = [], []

        (self.l_xd,) = self.ax_x.plot([], [], label='Xd')
        (self.l_x,)  = self.ax_x.plot([], [], label='X'); self.ax_x.set_ylabel('X (m)'); self.ax_x.legend()
        (self.l_yd,) = self.ax_y.plot([], [], label='Yd')
        (self.l_y,)  = self.ax_y.plot([], [], label='Y'); self.ax_y.set_ylabel('Y (m)'); self.ax_y.legend()
        (self.l_zd,) = self.ax_z.plot([], [], label='Zd')
        (self.l_z,)  = self.ax_z.plot([], [], label='Z'); self.ax_z.set_ylabel('Z (m)'); self.ax_z.legend()
        (self.l_psid,) = self.ax_yaw.plot([], [], label='ψd')
        (self.l_psi,)  = self.ax_yaw.plot([], [], label='ψ'); self.ax_yaw.set_ylabel('yaw (rad)'); self.ax_yaw.set_xlabel('t (s)'); self.ax_yaw.legend()

        self.t0 = time.time()

        # ---------- QoS ----------
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # ---------- Subscripciones ----------
        self.create_subscription(NavSatFix, f'/{self.leader_ns}/global_position/global', self.cb_leader_gps, sensor_qos)
        self.create_subscription(NavSatFix, f'/{self.ns}/global_position/global',        self.cb_follow_gps, sensor_qos)
        self.create_subscription(Float64,   f'/{self.leader_ns}/global_position/rel_alt', self.cb_leader_alt, sensor_qos)
        self.create_subscription(Float64,   f'/{self.ns}/global_position/rel_alt',        self.cb_follow_alt, sensor_qos)

        if self.yaw_source in ('compass', 'auto'):
            self.create_subscription(Float64, f'/{self.leader_ns}/global_position/compass_hdg', self.cb_leader_hdg, sensor_qos)
            self.create_subscription(Float64, f'/{self.ns}/global_position/compass_hdg',        self.cb_follow_hdg, sensor_qos)
        if self.yaw_source in ('pose', 'auto'):
            self.create_subscription(PoseStamped, f'/{self.leader_ns}/local_position/pose', self.cb_leader_pose, sensor_qos)
            self.create_subscription(PoseStamped, f'/{self.ns}/local_position/pose',        self.cb_follow_pose, sensor_qos)

        # ---------- Publicadores ----------
        self.pub_twist = self.create_publisher(Twist,        f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts    = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel',    10)

        self.get_logger().info(
            f"[INIT] follower=/{self.ns} following=/{self.leader_ns}, mode={self.offset_mode}, "
            f"frame={self.offset_frame}, r={self.off_r}, theta_deg={math.degrees(self.off_th):.1f}, "
            f"z={self.off_z}, rate={1.0/self.dt:.1f}Hz"
        )

        self.create_timer(self.dt, self.control_loop)

    # --------- Callbacks ---------
    def cb_leader_gps(self, msg: NavSatFix):
        self.leader_xy = np.array(self.to_rtk_enu_xy(msg)); self.leader_gps_ready = True

    def cb_follow_gps(self, msg: NavSatFix):
        self.follow_xy = np.array(self.to_rtk_enu_xy(msg)); self.follow_gps_ready = True

    def cb_leader_alt(self, msg: Float64):
        self.leader_alt = float(msg.data); self.leader_alt_ready = True

    def cb_follow_alt(self, msg: Float64):
        self.follow_alt = float(msg.data); self.follow_alt_ready = True

    def cb_leader_hdg(self, msg: Float64):
        self.leader_yaw = hdg_deg_to_yaw_enu_rad(float(msg.data))
        self.leader_yaw_ready = True; self.have_leader_compass = True

    def cb_follow_hdg(self, msg: Float64):
        self.follow_yaw = hdg_deg_to_yaw_enu_rad(float(msg.data))
        self.follow_yaw_ready = True; self.have_follow_compass = True

    def cb_leader_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.leader_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if self.yaw_source == 'pose' or (self.yaw_source == 'auto' and not self.have_leader_compass):
            self.leader_yaw_ready = True
        self.have_leader_pose = True

    def cb_follow_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        self.follow_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if self.yaw_source == 'pose' or (self.yaw_source == 'auto' and not self.have_follow_compass):
            self.follow_yaw_ready = True
        self.have_follow_pose = True

    # --------- Conversiones ---------
    def to_rtk_enu_xy(self, msg: NavSatFix):
        lat = math.radians(msg.latitude); lon = math.radians(msg.longitude); alt = msg.altitude
        X, Y, Z = self.geodetic_to_ecef(lat, lon, alt)
        d = np.array([X-self.X0, Y-self.Y0, Z-self.Z0])
        enu = self.R_enu.dot(d)
        x = enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta)
        y = enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)
        return [x, y]

    def compute_calibration_angle(self, Xr, Yr, Zr):
        ref = self.R_enu.dot([Xr-self.X0, Yr-self.Y0, Zr-self.Z0])
        return math.atan2(1.0,1.0) - math.atan2(ref[1], ref[0])

    @staticmethod
    def geodetic_to_ecef(lat, lon, alt):
        N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
        return ((N+alt)*math.cos(lat)*math.cos(lon),
                (N+alt)*math.cos(lat)*math.sin(lon),
                (N*(1-WGS84_E2)+alt)*math.sin(lat))

    @staticmethod
    def get_rotation_matrix(lat, lon):
        return np.array([
            [-math.sin(lon),                 math.cos(lon),               0],
            [-math.sin(lat)*math.cos(lon),  -math.sin(lat)*math.sin(lon), math.cos(lat)],
            [ math.cos(lat)*math.cos(lon),   math.cos(lat)*math.sin(lon), math.sin(lat)]
        ])

    # --------- Control principal ---------
    def control_loop(self):
        # Ready para yaw en 'auto'
        if self.yaw_source == 'auto':
            if self.have_leader_compass or self.have_leader_pose: self.leader_yaw_ready = True
            if self.have_follow_compass or self.have_follow_pose: self.follow_yaw_ready = True

        if not (self.leader_gps_ready and self.follow_gps_ready and
                self.leader_alt_ready and self.follow_alt_ready and
                self.leader_yaw_ready and self.follow_yaw_ready):
            return

        # ===== Cómputo de offset (cilíndrico/cartesiano) =====
        if self.offset_mode == 'cylindrical':
            # Base (no rotada): vector en el frame indicado por offset_frame
            # En 'body': theta=0 -> x_forward del líder; CCW hacia y_left
            # En 'world': theta=0 -> +X ENU; CCW hacia +Y ENU
            ox_body = self.off_r * math.cos(self.off_th)
            oy_body = self.off_r * math.sin(self.off_th)
            oz      = self.off_z

            if self.offset_frame == 'body':
                # Rotar por yaw del líder hacia ENU
                c = math.cos(self.leader_yaw); s = math.sin(self.leader_yaw)
                offset_xy_world = np.array([ c*ox_body - s*oy_body,
                                             s*ox_body + c*oy_body ])
            else:  # world
                offset_xy_world = np.array([ox_body, oy_body])

        else:
            # Modo cartesiano (retro-compat)
            oz = self.off_z_cart
            if self.offset_frame == 'body':
                c = math.cos(self.leader_yaw); s = math.sin(self.leader_yaw)
                offset_xy_world = np.array([ c*self.off_x - s*self.off_y,
                                             s*self.off_x + c*self.off_y ])
            else:
                offset_xy_world = np.array([self.off_x, self.off_y])
        # =====================================================

        # Setpoints en ENU
        xd = self.leader_xy[0] + offset_xy_world[0]
        yd = self.leader_xy[1] + offset_xy_world[1]
        zd = self.leader_alt    + oz
        psid = wrap_pi(self.leader_yaw + self.yaw_offset)

        # Errores
        ex, ey = xd - self.follow_xy[0], yd - self.follow_xy[1]
        ez = zd - self.follow_alt
        e_yaw = wrap_pi(psid - self.follow_yaw)

        # PID lineal
        e_xyz = np.array([ex, ey, ez])
        self.int_xyz += e_xyz * self.dt
        dedt_xyz = (e_xyz - self.prev_error_xyz) / self.dt
        self.prev_error_xyz = e_xyz.copy()

        vx = self.kp*ex + self.ki*self.int_xyz[0] + self.kd*dedt_xyz[0]
        vy = self.kp*ey + self.ki*self.int_xyz[1] + self.kd*dedt_xyz[1]
        vz = self.kp*ez + self.ki*self.int_xyz[2] + self.kd*dedt_xyz[2]

        # Saturación y anti-windup simple
        vxy = math.hypot(vx, vy)
        if vxy > self.vmax_xy:
            scale = self.vmax_xy / max(1e-6, vxy); vx *= scale; vy *= scale
        vz = clamp(vz, -self.vmax_z, self.vmax_z)
        if self.ki > 0.0:
            ax = vx - (self.kp*ex + self.kd*dedt_xyz[0])
            ay = vy - (self.kp*ey + self.kd*dedt_xyz[1])
            az = vz - (self.kp*ez + self.kd*dedt_xyz[2])
            self.int_xyz = np.array([ax, ay, az]) / self.ki

        # PID angular
        self.int_yaw += e_yaw * self.dt
        dedt_yaw = (e_yaw - self.prev_eyaw) / self.dt
        self.prev_eyaw = e_yaw
        wz = self.kp_yaw*e_yaw + self.ki_yaw*self.int_yaw + self.kd_yaw*dedt_yaw
        wz = clamp(wz, -self.wmax, self.wmax)
        if self.ki_yaw > 0.0:
            self.int_yaw = (wz - (self.kp_yaw*e_yaw + self.kd_yaw*dedt_yaw)) / self.ki_yaw

        # Comandos
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = vx, vy, vz
        cmd.angular.z = wz
        self.pub_twist.publish(cmd)

        ts = TwistStamped()
        ts.header.stamp = self.get_clock().now().to_msg(); ts.twist = cmd
        self.pub_ts.publish(ts)

        # Plot
        t = time.time() - self.t0
        self.times.append(t)
        self.xd_hist.append(xd); self.x_hist.append(self.follow_xy[0])
        self.yd_hist.append(yd); self.y_hist.append(self.follow_xy[1])
        self.zd_hist.append(zd); self.z_hist.append(self.follow_alt)
        self.psid_hist.append(psid); self.psi_hist.append(self.follow_yaw)

        self.l_xd.set_data(self.times, self.xd_hist); self.l_x.set_data(self.times, self.x_hist)
        self.l_yd.set_data(self.times, self.yd_hist); self.l_y.set_data(self.times, self.y_hist)
        self.l_zd.set_data(self.times, self.zd_hist); self.l_z.set_data(self.times, self.z_hist)
        self.l_psid.set_data(self.times, self.psid_hist); self.l_psi.set_data(self.times, self.psi_hist)
        for ax in self.axes:
            ax.relim(); ax.autoscale_view()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = RelativePIDFollower()
    rclpy.spin(node)
    node.destroy_node()
    plt.ioff(); plt.show()
    rclpy.shutdown()


# ---- Utilidades geométricas ----
def geodetic_to_ecef(lat, lon, alt):
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
    return ((N+alt)*math.cos(lat)*math.cos(lon),
            (N+alt)*math.cos(lat)*math.sin(lon),
            (N*(1-WGS84_E2)+alt)*math.sin(lat))

def get_rotation_matrix(lat, lon):
    return np.array([
        [-math.sin(lon),                 math.cos(lon),               0],
        [-math.sin(lat)*math.cos(lon),  -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [ math.cos(lat)*math.cos(lon),   math.cos(lat)*math.sin(lon), math.sin(lat)]
    ])

# Bind estático a la clase (para mantener el mismo estilo que tu código previo)
RelativePIDFollower.geodetic_to_ecef = staticmethod(geodetic_to_ecef)
RelativePIDFollower.get_rotation_matrix = staticmethod(get_rotation_matrix)

if __name__ == '__main__':
    main()

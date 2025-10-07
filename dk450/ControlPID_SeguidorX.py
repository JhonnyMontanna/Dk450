#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped
import matplotlib.pyplot as plt
import time

# Constantes WGS84
WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014e-3

class AbsolutePIDFollower(Node):
    """
    Seguidor PID en coordenadas RTK-ENU con control en X, Y y Z:
    - X, Y: transforma NavSatFix a ENU-RTK.
    - Z: usa altitud relativa (rel_alt).
    - Control PID con offsets configurables.
    - Visualización de setpoints vs reales en tiempo real.
    """
    def __init__(self):
        super().__init__('absolute_pid_follower')
        # Declarar parámetros
        p = self.declare_parameter
        p('drone_ns',    'uav2'); p('leader_ns',   'uav1')
        p('origin_lat',  19.5942341); p('origin_lon', -99.2280871); p('origin_alt', 2329.0)
        p('calib_lat',   19.5942429); p('calib_lon', -99.2280774); p('calib_alt', 2329.0)
        p('offset_x',    1.0); p('offset_y',  0.0); p('offset_z', 0.5)
        p('pid_kp',      0.5); p('pid_ki',    0.0); p('pid_kd',  0.1)
        p('rate',        20)

        # Leer parámetros
        self.ns           = self.get_parameter('drone_ns').value
        self.leader_ns    = self.get_parameter('leader_ns').value
        origin_lat        = self.get_parameter('origin_lat').value
        origin_lon        = self.get_parameter('origin_lon').value
        origin_alt        = self.get_parameter('origin_alt').value
        calib_lat         = self.get_parameter('calib_lat').value
        calib_lon         = self.get_parameter('calib_lon').value
        calib_alt         = self.get_parameter('calib_alt').value
        self.offset       = (
            self.get_parameter('offset_x').value,
            self.get_parameter('offset_y').value,
            self.get_parameter('offset_z').value
        )
        # PID gains
        self.kp           = self.get_parameter('pid_kp').value
        self.ki           = self.get_parameter('pid_ki').value
        self.kd           = self.get_parameter('pid_kd').value
        rate               = self.get_parameter('rate').value
        self.dt           = 1.0 / float(rate)

        # RTK: origen y calibración
        lat0 = math.radians(origin_lat); lon0 = math.radians(origin_lon)
        lat_ref = math.radians(calib_lat); lon_ref = math.radians(calib_lon)
        self.X0, self.Y0, self.Z0 = self.geodetic_to_ecef(lat0, lon0, origin_alt)
        Xr, Yr, Zr = self.geodetic_to_ecef(lat_ref, lon_ref, calib_alt)
        self.R_enu = self.get_rotation_matrix(lat0, lon0)
        self.theta = self.compute_calibration_angle(Xr, Yr, Zr)

        # Estado interno
        self.leader_enu       = [0.0, 0.0]
        self.follower_enu     = [0.0, 0.0]
        self.leader_alt       = 0.0
        self.follower_alt     = 0.0
        self.leader_gps_ready = False
        self.follower_gps_ready = False
        self.leader_alt_ready   = False
        self.follower_alt_ready = False
        self.prev_error       = [0.0, 0.0, 0.0]
        self.integral         = [0.0, 0.0, 0.0]

        # Datos de trazado
        self.times    = []
        self.xd_list  = []
        self.x_list   = []
        self.yd_list  = []
        self.y_list   = []
        self.zd_list  = []
        self.z_list   = []
        self.start_time = time.time()

        # Configurar gráfico interactivo
        plt.ion()
        self.fig, (self.ax_x, self.ax_y, self.ax_z) = plt.subplots(3,1, figsize=(8,6))
        self.line_xd, = self.ax_x.plot([], [], label='X setpoint')
        self.line_x,  = self.ax_x.plot([], [], label='X real')
        self.ax_x.legend(); self.ax_x.set_ylabel('X (m)')
        self.line_yd, = self.ax_y.plot([], [], label='Y setpoint')
        self.line_y,  = self.ax_y.plot([], [], label='Y real')
        self.ax_y.legend(); self.ax_y.set_ylabel('Y (m)')
        self.line_zd, = self.ax_z.plot([], [], label='Z setpoint')
        self.line_z,  = self.ax_z.plot([], [], label='Z real')
        self.ax_z.legend(); self.ax_z.set_ylabel('Z (m)'); self.ax_z.set_xlabel('Time (s)')

        # QoS
        gps_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        alt_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        # Subscripciones
        self.create_subscription(NavSatFix, f'/{self.leader_ns}/global_position/global', self.cb_leader_gps, gps_qos)
        self.create_subscription(NavSatFix, f'/{self.ns}/global_position/global',        self.cb_follower_gps, gps_qos)
        self.create_subscription(Float64,  f'/{self.leader_ns}/global_position/rel_alt',   self.cb_leader_alt, alt_qos)
        self.create_subscription(Float64,  f'/{self.ns}/global_position/rel_alt',          self.cb_follower_alt, alt_qos)

        # Publicadores de velocidad
        self.pub_twist = self.create_publisher(Twist,        f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped', 10)
        self.pub_ts    = self.create_publisher(TwistStamped, f'/{self.ns}/mavros/setpoint_velocity/cmd_vel',       10)

        # Timer de control
        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(f"[INIT] PID follower=/{self.ns} following=/{self.leader_ns}, offset={self.offset}, kp={self.kp}, ki={self.ki}, kd={self.kd}, rate={rate}Hz")

    def cb_leader_gps(self, msg: NavSatFix):
        enu = self.to_rtk_enu(msg)
        self.leader_enu = enu
        self.leader_gps_ready = True

    def cb_follower_gps(self, msg: NavSatFix):
        enu = self.to_rtk_enu(msg)
        self.follower_enu = enu
        self.follower_gps_ready = True

    def cb_leader_alt(self, msg: Float64):
        self.leader_alt = msg.data
        self.leader_alt_ready = True

    def cb_follower_alt(self, msg: Float64):
        self.follower_alt = msg.data
        self.follower_alt_ready = True

    def to_rtk_enu(self, msg: NavSatFix):
        lat = math.radians(msg.latitude); lon = math.radians(msg.longitude); alt = msg.altitude
        X, Y, Z = self.geodetic_to_ecef(lat, lon, alt)
        d = np.array([X-self.X0, Y-self.Y0, Z-self.Z0])
        enu = self.R_enu.dot(d)
        x = enu[0]*math.cos(self.theta) - enu[1]*math.sin(self.theta)
        y = enu[0]*math.sin(self.theta) + enu[1]*math.cos(self.theta)
        return [x, y]

    def control_loop(self):
        # Asegurar lecturas completas
        if not (self.leader_gps_ready and self.follower_gps_ready and self.leader_alt_ready and self.follower_alt_ready):
            return

        # Setpoints con offsets
        xd = self.leader_enu[0] + self.offset[0]
        yd = self.leader_enu[1] + self.offset[1]
        zd = self.leader_alt    + self.offset[2]

        # Errores
        error = [xd - self.follower_enu[0], yd - self.follower_enu[1], zd - self.follower_alt]

        # Integral y derivada
        for i in range(3):
            self.integral[i] += error[i] * self.dt
        derivative = [(error[i] - self.prev_error[i]) / self.dt for i in range(3)]
        self.prev_error = error.copy()

        # Salida PID
        vx = self.kp*error[0] + self.ki*self.integral[0] + self.kd*derivative[0]
        vy = self.kp*error[1] + self.ki*self.integral[1] + self.kd*derivative[1]
        vz = self.kp*error[2] + self.ki*self.integral[2] + self.kd*derivative[2]

        # Publicar velocidades
        cmd = Twist(); cmd.linear.x=vx; cmd.linear.y=vy; cmd.linear.z=vz; cmd.angular.z=0.0
        self.pub_twist.publish(cmd)
        ts = TwistStamped(); ts.header.stamp=self.get_clock().now().to_msg(); ts.twist=cmd
        self.pub_ts.publish(ts)

        # Actualizar gráficas
        t = time.time() - self.start_time
        self.times.append(t)
        self.xd_list.append(xd); self.x_list.append(self.follower_enu[0])
        self.yd_list.append(yd); self.y_list.append(self.follower_enu[1])
        self.zd_list.append(zd); self.z_list.append(self.follower_alt)

        # Refrescar plots
        self.line_xd.set_data(self.times, self.xd_list)
        self.line_x .set_data(self.times, self.x_list)
        self.line_yd.set_data(self.times, self.yd_list)
        self.line_y .set_data(self.times, self.y_list)
        self.line_zd.set_data(self.times, self.zd_list)
        self.line_z .set_data(self.times, self.z_list)
        for ax, data_lists in zip((self.ax_x, self.ax_y, self.ax_z),
                                  ((self.xd_list+self.x_list), (self.yd_list+self.y_list), (self.zd_list+self.z_list))):
            ax.relim(); ax.autoscale_view()
        plt.pause(0.001)

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


def main():
    rclpy.init()
    node = AbsolutePIDFollower()
    rclpy.spin(node)
    node.destroy_node()
    plt.ioff(); plt.show()
    rclpy.shutdown()

if __name__=='__main__':
    main()

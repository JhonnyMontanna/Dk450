#!/usr/bin/env python3
"""
PID Follower with RTK-ENU and real-time visualization plus CSV logging:
- X,Y from NavSatFix → ENU-RTK
- Z from rel_alt (Float64)
- PID control on X, Y, Z with configurable gains and offsets
- Optional real-time matplotlib plots
- CSV logging of leader, setpoint, follower positions and errors
"""
import math
import csv
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, TwistStamped

# Constantes WGS84
WGS84_A  = 6378137.0
WGS84_E2 = 0.00669437999014e-3

class AbsolutePIDFollower(Node):
    def __init__(self):
        super().__init__('absolute_pid_follower')
        # Parámetros ROS
        p=self.declare_parameter
        p('drone_ns','uav2'); p('leader_ns','uav1')
        p('origin_lat',19.5942341); p('origin_lon',-99.2280871); p('origin_alt',2329.0)
        p('calib_lat',19.5942429); p('calib_lon',-99.2280774); p('calib_alt',2329.0)
        p('offset_x',1.0); p('offset_y',0.0); p('offset_z',1.5)
        p('pid_kp',0.5); p('pid_ki',0.0); p('pid_kd',0.0)
        p('rate',20); p('enable_plot',True)

        # Leer parámetros
        self.ns         = self.get_parameter('drone_ns').value
        self.leader_ns  = self.get_parameter('leader_ns').value
        origin_lat      = self.get_parameter('origin_lat').value
        origin_lon      = self.get_parameter('origin_lon').value
        origin_alt      = self.get_parameter('origin_alt').value
        calib_lat       = self.get_parameter('calib_lat').value
        calib_lon       = self.get_parameter('calib_lon').value
        calib_alt       = self.get_parameter('calib_alt').value
        self.offset     = (
            self.get_parameter('offset_x').value,
            self.get_parameter('offset_y').value,
            self.get_parameter('offset_z').value
        )
        self.kp         = self.get_parameter('pid_kp').value
        self.ki         = self.get_parameter('pid_ki').value
        self.kd         = self.get_parameter('pid_kd').value
        rate            = self.get_parameter('rate').value
        self.dt         = 1.0/ rate
        self.enable_plot= self.get_parameter('enable_plot').value

        # Configuración RTK-ENU
        lat0 = math.radians(origin_lat); lon0 = math.radians(origin_lon)
        lat_ref = math.radians(calib_lat); lon_ref = math.radians(calib_lon)
        self.X0,self.Y0,self.Z0 = self.geodetic_to_ecef(lat0,lon0,origin_alt)
        Xr,Yr,Zr = self.geodetic_to_ecef(lat_ref,lon_ref,calib_alt)
        self.R_enu = self.get_rotation_matrix(lat0,lon0)
        self.theta = self.compute_calibration_angle(Xr,Yr,Zr)

        # Estado
        self.leader_enu = [0.0,0.0]; self.follower_enu = [0.0,0.0]
        self.leader_alt = 0.0; self.follower_alt = 0.0
        self.ready_gps = False; self.ready_alt = False
        self.prev_error = [0.0,0.0,0.0]
        self.integral = [0.0,0.0,0.0]

        # CSV data
        self.csv_rows = []
        header = ['time','lx','ly','lz','sx','sy','sz','fx','fy','fz','ex','ey','ez']
        self.csv_rows.append(header)
        self.start_time = time.time()

        # Configurar grafico
        if self.enable_plot:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.times=[]; self.xd_list=[]; self.x_list=[]
            self.yd_list=[]; self.y_list=[]; self.zd_list=[]; self.z_list=[]
            plt.ion()
            self.fig,(self.ax_x,self.ax_y,self.ax_z)=plt.subplots(3,1,figsize=(8,6))
            self.line_xd,=self.ax_x.plot([],[],label='X set')
            self.line_x, =self.ax_x.plot([],[],label='X real')
            self.ax_x.legend(); self.ax_x.set_ylabel('X (m)')
            self.line_yd,=self.ax_y.plot([],[],label='Y set')
            self.line_y, =self.ax_y.plot([],[],label='Y real')
            self.ax_y.legend(); self.ax_y.set_ylabel('Y (m)')
            self.line_zd,=self.ax_z.plot([],[],label='Z set')
            self.line_z, =self.ax_z.plot([],[],label='Z real')
            self.ax_z.legend(); self.ax_z.set_ylabel('Z (m)'); self.ax_z.set_xlabel('Time (s)')

        # QoS y subscripciones
        gps_qos = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
        alt_qos = QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(NavSatFix,f'/{self.leader_ns}/global_position/global',self.cb_leader_gps,gps_qos)
        self.create_subscription(NavSatFix,f'/{self.ns}/global_position/global',self.cb_follower_gps,gps_qos)
        self.create_subscription(Float64,f'/{self.leader_ns}/global_position/rel_alt',self.cb_leader_alt,alt_qos)
        self.create_subscription(Float64,f'/{self.ns}/global_position/rel_alt',self.cb_follower_alt,alt_qos)

        # Publicadores vel
        self.pub_twist = self.create_publisher(Twist,f'/{self.ns}/setpoint_velocity/cmd_vel_unstamped',10)
        self.pub_ts    = self.create_publisher(TwistStamped,f'/{self.ns}/mavros/setpoint_velocity/cmd_vel',10)

        # Timer control
        self.create_timer(self.dt,self.control_loop)
        self.get_logger().info(f"[INIT] PID follower=/{self.ns} → /{self.leader_ns}, offset={self.offset}, kp={self.kp}, ki={self.ki}, kd={self.kd}, rate={rate}Hz, plot={self.enable_plot}")

    def cb_leader_gps(self,msg): self.leader_enu=self.to_rtk_enu(msg); self.ready_gps=True
    def cb_follower_gps(self,msg): self.follower_enu=self.to_rtk_enu(msg); self.ready_gps=True
    def cb_leader_alt(self,msg): self.leader_alt=msg.data; self.ready_alt=True
    def cb_follower_alt(self,msg): self.follower_alt=msg.data; self.ready_alt=True

    def to_rtk_enu(self,msg):
        lat,lon,alt=math.radians(msg.latitude),math.radians(msg.longitude),msg.altitude
        X,Y,Z=self.geodetic_to_ecef(lat,lon,alt)
        d=np.array([X-self.X0,Y-self.Y0,Z-self.Z0]); enu=self.R_enu.dot(d)
        x=enu[0]*math.cos(self.theta)-enu[1]*math.sin(self.theta)
        y=enu[0]*math.sin(self.theta)+enu[1]*math.cos(self.theta)
        return [x,y]

    def control_loop(self):
        if not(self.ready_gps and self.ready_alt): return
        # setpoints
        xd,yd,zd = self.leader_enu[0]+self.offset[0], self.leader_enu[1]+self.offset[1], self.leader_alt+self.offset[2]
        fx,fy,fz = self.follower_enu[0], self.follower_enu[1], self.follower_alt
        # errors
        ex,ey,ez = xd-fx, yd-fy, zd-fz
        # PID
        err=[ex,ey,ez]
        for i in range(3): self.integral[i]+=err[i]*self.dt
        deriv=[(err[i]-self.prev_error[i])/self.dt for i in range(3)]; self.prev_error=err.copy()
        vx=self.kp*ex+self.ki*self.integral[0]+self.kd*deriv[0]
        vy=self.kp*ey+self.ki*self.integral[1]+self.kd*deriv[1]
        vz=self.kp*ez+self.ki*self.integral[2]+self.kd*deriv[2]
        # publicar
        cmd=Twist(); cmd.linear.x=vx; cmd.linear.y=vy; cmd.linear.z=vz; cmd.angular.z=0.0
        self.pub_twist.publish(cmd)
        ts=TwistStamped(); ts.header.stamp=self.get_clock().now().to_msg(); ts.twist=cmd
        self.pub_ts.publish(ts)
        # CSV log
        t=time.time()-self.start_time
        row=[t, self.leader_enu[0], self.leader_enu[1], self.leader_alt, xd, yd, zd, fx, fy, fz, ex, ey, ez]
        self.csv_rows.append(row)
        # plots
        if self.enable_plot:
            self.times.append(t); self.xd_list.append(xd); self.x_list.append(fx)
            self.yd_list.append(yd); self.y_list.append(fy); self.zd_list.append(zd); self.z_list.append(fz)
            self.line_xd.set_data(self.times,self.xd_list); self.line_x.set_data(self.times,self.x_list)
            self.line_yd.set_data(self.times,self.yd_list); self.line_y.set_data(self.times,self.y_list)
            self.line_zd.set_data(self.times,self.zd_list); self.line_z.set_data(self.times,self.z_list)
            for ax,data in zip((self.ax_x,self.ax_y,self.ax_z),
                               ((self.xd_list+self.x_list),(self.yd_list+self.y_list),(self.zd_list+self.z_list))):
                ax.relim(); ax.autoscale_view()
            self.plt.pause(0.001)

    def compute_calibration_angle(self,Xr,Yr,Zr):
        ref=self.R_enu.dot([Xr-self.X0,Yr-self.Y0,Zr-self.Z0])
        return math.atan2(1.0,1.0)-math.atan2(ref[1],ref[0])

    @staticmethod
    def geodetic_to_ecef(lat,lon,alt):
        N= WGS84_A/math.sqrt(1-WGS84_E2*math.sin(lat)**2)
        return ((N+alt)*math.cos(lat)*math.cos(lon),
                (N+alt)*math.cos(lat)*math.sin(lon),
                (N*(1-WGS84_E2)+alt)*math.sin(lat))

    @staticmethod
    def get_rotation_matrix(lat,lon):
        return np.array([[-math.sin(lon),math.cos(lon),0],
                         [-math.sin(lat)*math.cos(lon),-math.sin(lat)*math.sin(lon),math.cos(lat)],
                         [math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)]])

    def save_csv(self):
        fname=f'pid_log_{int(self.start_time)}.csv'
        with open(fname,'w',newline='') as f:
            writer=csv.writer(f)
            writer.writerows(self.csv_rows)
        self.get_logger().info(f"CSV guardado en {fname}")

    def destroy_node(self):
        # guardar CSV antes de destruir
        self.save_csv()
        super().destroy_node()


def main():
    rclpy.init()
    node=AbsolutePIDFollower()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if node.enable_plot:
            import matplotlib.pyplot as plt; plt.ioff(); plt.show()
        rclpy.shutdown()

if __name__=='__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
from pymavlink import mavutil
import math

class MavlinkBridge(Node):
    def __init__(self):
        super().__init__('mavlink_bridge')
        
        # Publicadores
        self.gps_pub = self.create_publisher(Vector3, '/gps_position', 10)
        self.attitude_pub = self.create_publisher(Vector3, '/drone_attitude', 10)
        self.sonar_pub = self.create_publisher(Float32, '/sonar_altitude', 10)
        
        # Conexión MAVLink
        self.master = mavutil.mavlink_connection('udpin:0.0.0.0:14552')
        #self.master = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
        self.master.wait_heartbeat()
        self.get_logger().info("Conexión MAVLink establecida")
        
        # Variables reutilizables
        self.gps_msg = Vector3()
        self.attitude_msg = Vector3()
        self.sonar_msg = Float32()
        
        # Hilo de lectura rápida
        self.read_thread = self.create_timer(0.001, self.read_mavlink)

    def read_mavlink(self):
        try:
            while (msg := self.master.recv_match(blocking=False)) is not None:
                self.process_message(msg)
        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}", throttle_duration_sec=5)

    def process_message(self, msg):
        msg_type = msg.get_type()
        
        if msg_type == 'GLOBAL_POSITION_INT':
            self.gps_msg.x = msg.lat / 1e7  # Latitud
            self.gps_msg.y = msg.lon / 1e7  # Longitud
            self.gps_msg.z = msg.alt / 1e3  # Altitud
            
            
            print(f"\nLatitud: {self.gps_msg.x:.6f}°")
            print(f"Longitud: {self.gps_msg.y:.6f}°")
            
            self.gps_pub.publish(self.gps_msg)
            
        elif msg_type == 'RANGEFINDER':
            self.sonar_msg.data = msg.distance
            self.sonar_pub.publish(self.sonar_msg)
            print(f"\nLatitud: {self.sonar_msg:.6f}°")
            
        elif msg_type == 'ATTITUDE':
            self.attitude_msg.x = math.degrees(msg.roll)
            self.attitude_msg.y = math.degrees(msg.pitch)
            self.attitude_msg.z = math.degrees(msg.yaw)
            self.attitude_pub.publish(self.attitude_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MavlinkBridge()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Apagando nodo...")
    finally:
        node.master.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
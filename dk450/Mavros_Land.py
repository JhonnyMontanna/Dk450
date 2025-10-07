#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandTOL
import time

class GuidedLand(Node):
    def __init__(self):
        super().__init__('guided_land')

        # Parámetros:
        #  - drone_ns: namespace del dron (uav1, uav2…)
        #  - land_alt: altitud de aterrizaje (normalmente 0.0)
        self.declare_parameter('drone_ns',  'uav1')
        self.declare_parameter('land_alt',  0.0)

        ns  = self.get_parameter('drone_ns').get_parameter_value().string_value
        alt = self.get_parameter('land_alt').get_parameter_value().double_value

        # Cliente al servicio de aterrizaje:
        #   /<ns>/cmd/land
        self.land_cli = self.create_client(CommandTOL, f'/{ns}/cmd/land')

        self.get_logger().info(f'Esperando servicio /{ns}/cmd/land …')
        self.land_cli.wait_for_service()

        # Enviar comando de land
        self.call_land(alt)

        self.get_logger().info('¡Comando de aterrizaje enviado!')
        self.destroy_node()
        rclpy.shutdown()

    def call_land(self, altitude: float):
        req = CommandTOL.Request()
        req.latitude   = 0.0   # no usado si es takeoff/land local
        req.longitude  = 0.0
        req.altitude   = altitude
        req.min_pitch  = 0.0
        req.yaw        = 0.0

        fut = self.land_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)

        if fut.result() and fut.result().success:
            self.get_logger().info(f'Aterrizando a {altitude} m ✔')
        else:
            self.get_logger().error('Error en comando LAND ✖')


def main():
    rclpy.init()
    GuidedLand()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
import time

class GuidedTakeoff(Node):
    def __init__(self):
        super().__init__('guided_takeoff')

        # Parámetros:
        #  - drone_ns: namespace del dron (uav1, uav2…)
        #  - takeoff_alt: altura de despegue en metros
        self.declare_parameter('drone_ns',    'uav2')
        self.declare_parameter('takeoff_alt', 1.0)

        ns  = self.get_parameter('drone_ns').get_parameter_value().string_value
        alt = self.get_parameter('takeoff_alt').get_parameter_value().double_value

        # Clientes a los servicios reales que tienes listados:
        self.mode_cli    = self.create_client(SetMode,    f'/{ns}/set_mode')
        self.arm_cli     = self.create_client(CommandBool, f'/{ns}/cmd/arming')
        self.takeoff_cli = self.create_client(CommandTOL,  f'/{ns}/cmd/takeoff')

        self.get_logger().info(f'Esperando servicios en /{ns}/…')
        self.mode_cli.wait_for_service()
        self.arm_cli.wait_for_service()
        self.takeoff_cli.wait_for_service()

        # 1) Cambiar a GUIDED
        self.call_set_mode('GUIDED')
        time.sleep(1.0)

        # 2) Armar
        self.call_arm(True)
        time.sleep(1.0)

        # 3) Despegar a la altitud indicada
        self.call_takeoff(alt)

        self.get_logger().info('¡Comandos enviados!')
        self.destroy_node()
        rclpy.shutdown()

    def call_set_mode(self, mode: str):
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self.mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = fut.result() and fut.result().mode_sent
        self.get_logger().info(f'SetMode {mode}: {"✔" if ok else "✖"}')

    def call_arm(self, arm: bool):
        req = CommandBool.Request()
        req.value = arm
        fut = self.arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = fut.result() and fut.result().success
        self.get_logger().info(f'Arming {"✔" if ok else "✖"}')

    def call_takeoff(self, altitude: float):
        req = CommandTOL.Request()
        req.latitude  = 0.0    # no usado si es takeoff local
        req.longitude = 0.0
        req.altitude  = altitude
        req.min_pitch = 0.0
        req.yaw       = 0.0
        fut = self.takeoff_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = fut.result() and fut.result().success
        self.get_logger().info(f'Takeoff {altitude} m: {"✔" if ok else "✖"}')

def main():
    rclpy.init()
    GuidedTakeoff()

if __name__ == '__main__':
    main()

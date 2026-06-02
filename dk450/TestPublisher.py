"""
Corre esto EN TU MÁQUINA con ROS 2 activo:
  python3 /tmp/bench_publish.py

Mide cuánto tarda cada publish() y sendTransform() para saber
si el cuello de botella es ROS middleware o el código Python.


Benchmarking 500 iteraciones (2x odom + 2x marker + 2x TF por iter)...

Resultados:
  Tiempo total:        0.159 s para 500 iters
  Tiempo por muestra:  0.317 ms
  Throughput máximo:   3150 muestras/s

Implicaciones:
  A 10 Hz (log típico): speed máximo alcanzable = 315x
  A 50 Hz (log rápido): speed máximo alcanzable = 63x

"""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class BenchNode(Node):
    def __init__(self):
        super().__init__("bench_publish")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.odom_pub   = self.create_publisher(Odometry, "/bench/odom",   qos)
        self.marker_pub = self.create_publisher(Marker,   "/bench/marker", qos)
        self.tf_br      = TransformBroadcaster(self)

    def bench(self, n=500):
        from builtin_interfaces.msg import Time as RosTime
        stamp = RosTime(); stamp.sec = 1; stamp.nanosec = 0

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp; odom.header.frame_id = "rtk_odom"
        odom.pose.pose.position.x = 1.0
        odom.pose.pose.orientation.w = 1.0

        # Marker
        mk = Marker()
        mk.header = odom.header; mk.type = Marker.SPHERE; mk.action = Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.scale.x = mk.scale.y = mk.scale.z = 0.5
        mk.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.5)

        # TF
        tf = TransformStamped()
        tf.header.stamp = stamp; tf.header.frame_id = "rtk_odom"
        tf.child_frame_id = "bench_link"
        tf.transform.rotation.w = 1.0

        print(f"Benchmarking {n} iteraciones (2x odom + 2x marker + 2x TF por iter)...")
        t0 = time.perf_counter()
        for _ in range(n):
            self.odom_pub.publish(odom)
            self.odom_pub.publish(odom)
            self.marker_pub.publish(mk)
            self.marker_pub.publish(mk)
            self.tf_br.sendTransform(tf)
            self.tf_br.sendTransform(tf)
        total = time.perf_counter() - t0

        per_iter_ms = (total / n) * 1000
        max_speed   = 1000 / per_iter_ms  # samples/s máximos

        print(f"\nResultados:")
        print(f"  Tiempo total:        {total:.3f} s para {n} iters")
        print(f"  Tiempo por muestra:  {per_iter_ms:.3f} ms")
        print(f"  Throughput máximo:   {max_speed:.0f} muestras/s")
        print(f"\nImplicaciones:")
        print(f"  A 10 Hz (log típico): speed máximo alcanzable = {max_speed/10:.0f}x")
        print(f"  A 50 Hz (log rápido): speed máximo alcanzable = {max_speed/50:.0f}x")

rclpy.init()
node = BenchNode()
node.bench()
node.destroy_node()
rclpy.shutdown()
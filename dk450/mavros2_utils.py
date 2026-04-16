"""
mavros2_utils.py — helpers compartidos para todos los nodos ROS2/MAVROS2
=========================================================================
Tópicos MAVROS2 (Humble) usados en todos los nodos:

  Entrada (suscripción):
    /uavN/mavros/local_position/pose         → PoseStamped  (posición + orientación)
    /uavN/mavros/local_position/velocity_local → TwistStamped (velocidades locales)

  Salida (publicación):
    /uavN/mavros/setpoint_position/local     → PoseStamped  (setpoint de posición)
    /uavN/mavros/setpoint_velocity/cmd_vel   → TwistStamped (setpoint de velocidad)

Convenciones de frame:
  MAVROS publica LOCAL_POSITION_NED en frame ENU (ROS standard):
    x = este, y = norte, z = arriba
  Los scripts originales usaban NED (x=norte, y=este, z=abajo).
  Esta utilidad convierte internamente para mantener la lógica de control
  en ENU (que es lo que llega de MAVROS), sin confusión de frames.
"""

import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ── QoS estándar para MAVROS2 ─────────────────────────────────────────────────
# setpoint_position y setpoint_velocity requieren BEST_EFFORT + VOLATILE
MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# Tópicos de estado (pose, velocidad) también BEST_EFFORT
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


def quat_to_yaw(q) -> float:
    """
    Extrae el yaw (rad) de un quaternion geometry_msgs/Quaternion.
    Convención ROS/ENU: yaw=0 → este, positivo = antihorario desde arriba.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: float):
    """
    Crea un quaternion geometry_msgs-compatible a partir de yaw (rad).
    Retorna (x, y, z, w) como tupla — asignar manualmente al mensaje.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return 0.0, 0.0, sy, cy    # (x, y, z, w)


def wrap(theta: float) -> float:
    """Proyecta θ al intervalo (-π, π] — equivalente a atan2(sin θ, cos θ)."""
    return math.atan2(math.sin(theta), math.cos(theta))


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def make_pose_stamped(node, x: float, y: float, z: float, yaw: float):
    """
    Construye un PoseStamped listo para publicar como setpoint de posición.
    Frame: map (ENU). x=este, y=norte, z=arriba.
    """
    from geometry_msgs.msg import PoseStamped
    msg = PoseStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    qx, qy, qz, qw = yaw_to_quat(yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def make_twist_stamped(node, vx: float, vy: float, vz: float,
                       yaw_rate: float = 0.0):
    """
    Construye un TwistStamped listo para publicar como setpoint de velocidad.
    Velocidades en frame ENU. yaw_rate en rad/s.
    """
    from geometry_msgs.msg import TwistStamped
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.twist.linear.x  = vx
    msg.twist.linear.y  = vy
    msg.twist.linear.z  = vz
    msg.twist.angular.z = yaw_rate
    return msg

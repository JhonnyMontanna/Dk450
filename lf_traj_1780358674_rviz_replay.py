#!/usr/bin/env python3
"""
Replay RViz2 — generado desde lf_traj_1780358674.csv
Lee el CSV de trayectoria completa (despegue + prepos + círculo)
y publica los datos en tópicos ROS2 para visualización.

Tópicos publicados:
  /leader/path          — trayectoria completa del líder
  /follower/path        — trayectoria completa del seguidor
  /leader/pose_current  — pose actual del líder (para flecha de orientación)
  /follower/pose_current— pose actual del seguidor
  /formation/marker     — línea L→S y texto de fase

Uso:
  ros2 run ... python3 lf_traj_1780358674_rviz_replay.py
  (o simplemente: python3 lf_traj_1780358674_rviz_replay.py)
"""
import csv, time, math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as RosTime

CSV_FILE = "lf_traj_1780358674.csv"
FRAME_ID = "map"
SPEED    = 1.0   # 1.0 = tiempo real, 2.0 = doble velocidad

# Colores por fase para el marcador de línea L-S
PHASE_COLORS = {
    "init":      (0.6, 0.6, 0.6),   # gris
    "takeoff_L": (0.2, 0.8, 0.2),   # verde
    "takeoff_S": (0.2, 0.8, 0.6),   # verde-azul
    "prepos":    (1.0, 0.8, 0.0),   # amarillo
    "circle":    (1.0, 0.4, 0.0),   # naranja
    "return":    (0.4, 0.4, 1.0),   # azul claro
}

def _stamp(t_unix):
    sec = int(t_unix)
    ns  = int((t_unix - sec) * 1e9)
    ts  = RosTime(); ts.sec = sec; ts.nanosec = ns
    return ts

def _pose(x, y, z, yaw, stamp):
    ps = PoseStamped()
    ps.header.frame_id = FRAME_ID
    ps.header.stamp    = stamp
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    cy = math.cos(float(yaw) / 2)
    sy = math.sin(float(yaw) / 2)
    ps.pose.orientation.w = cy
    ps.pose.orientation.z = sy
    return ps

class ReplayNode(Node):
    def __init__(self, rows):
        super().__init__("lf_traj_replay")
        self.rows = rows

        # Trayectorias acumuladas
        self.pub_L  = self.create_publisher(Path, "/leader/path",   10)
        self.pub_S  = self.create_publisher(Path, "/follower/path", 10)

        # Pose actual (para flechas de orientación en RViz2)
        self.pub_Lc = self.create_publisher(PoseStamped, "/leader/pose_current",   10)
        self.pub_Sc = self.create_publisher(PoseStamped, "/follower/pose_current", 10)

        # Marcadores (línea L-S y texto de fase)
        self.pub_mk = self.create_publisher(MarkerArray, "/formation/marker", 10)

        self.path_L = Path(); self.path_L.header.frame_id = FRAME_ID
        self.path_S = Path(); self.path_S.header.frame_id = FRAME_ID

        self.idx    = 0
        self.t0_ros = None
        self.t0_dat = None

        self.create_timer(0.05, self._tick)   # 20 Hz de replay

    def _tick(self):
        if self.idx >= len(self.rows):
            return

        row   = self.rows[self.idx]
        t_rel = float(row["t"])

        # Sincronizar tiempo
        if self.t0_ros is None:
            self.t0_ros = time.monotonic()
            self.t0_dat = t_rel

        elapsed = (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat
        if t_rel > elapsed:
            return   # aún no es el momento de publicar esta muestra

        stamp = _stamp(float(row["timestamp_unix"]))
        phase = row.get("phase", "circle")

        lx, ly, lz = row["lx"], row["ly"], row["lz"]
        sx, sy, sz = row["sx"], row["sy"], row["sz"]
        l_yaw      = float(row["l_yaw"])
        s_yaw      = float(row["s_yaw"])

        # Acumular trayectorias
        self.path_L.poses.append(_pose(lx, ly, lz, l_yaw, stamp))
        self.path_S.poses.append(_pose(sx, sy, sz, s_yaw, stamp))
        self.path_L.header.stamp = stamp
        self.path_S.header.stamp = stamp

        self.pub_L.publish(self.path_L)
        self.pub_S.publish(self.path_S)

        # Pose actual
        self.pub_Lc.publish(_pose(lx, ly, lz, l_yaw, stamp))
        self.pub_Sc.publish(_pose(sx, sy, sz, s_yaw, stamp))

        # Marcadores
        color = PHASE_COLORS.get(phase, (0.5, 0.5, 0.5))
        ma    = MarkerArray()

        # Línea L → S
        mk_line = Marker()
        mk_line.header.frame_id = FRAME_ID
        mk_line.header.stamp    = stamp
        mk_line.ns   = "formation_line"
        mk_line.id   = 0
        mk_line.type = Marker.LINE_LIST
        mk_line.action = Marker.ADD
        mk_line.scale.x  = 0.06
        mk_line.color.r  = color[0]; mk_line.color.g = color[1]
        mk_line.color.b  = color[2]; mk_line.color.a = 0.9
        p1 = Point(); p1.x = float(lx); p1.y = float(ly); p1.z = float(lz)
        p2 = Point(); p2.x = float(sx); p2.y = float(sy); p2.z = float(sz)
        mk_line.points = [p1, p2]
        ma.markers.append(mk_line)

        # Texto de fase (sobre el líder)
        mk_txt = Marker()
        mk_txt.header.frame_id = FRAME_ID
        mk_txt.header.stamp    = stamp
        mk_txt.ns   = "phase_label"
        mk_txt.id   = 1
        mk_txt.type = Marker.TEXT_VIEW_FACING
        mk_txt.action = Marker.ADD
        mk_txt.pose.position.x = float(lx)
        mk_txt.pose.position.y = float(ly)
        mk_txt.pose.position.z = float(lz) + 0.5
        mk_txt.scale.z  = 0.3
        mk_txt.color.r  = 1.0; mk_txt.color.g = 1.0
        mk_txt.color.b  = 1.0; mk_txt.color.a = 0.9
        mk_txt.text = phase
        ma.markers.append(mk_txt)

        self.pub_mk.publish(ma)
        self.idx += 1


def main():
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV vacío."); return

    dur = float(rows[-1]["t"]) - float(rows[0]["t"])
    print(f"Cargadas {len(rows)} muestras. Duración: {dur:.1f}s")
    print(f"Fases registradas: {set(r['phase'] for r in rows)}")
    print(f"Velocidad replay: {SPEED}x")
    print("Tópicos: /leader/path  /follower/path  /formation/marker  ...")

    rclpy.init()
    node = ReplayNode(rows)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()

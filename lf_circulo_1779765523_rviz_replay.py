#!/usr/bin/env python3
"""
Replay RViz2 — generado desde lf_circulo_1779765523.csv
==========================================
Publica:
  /leader/path          nav_msgs/Path
  /follower/path        nav_msgs/Path
  /follower/setpoint    nav_msgs/Path
  /formation/marker     visualization_msgs/MarkerArray (linea L->S)

Uso:
  # Terminal 1: frame estático
  ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom

  # Terminal 2: replay
  python3 lf_circulo_1779765523_rviz_replay.py

  # Terminal 3: RViz2
  rviz2
  Añadir:  Path (/leader/path)
           Path (/follower/path)
           Path (/follower/setpoint)
           MarkerArray (/formation/marker)
  Fixed Frame: map
"""
import csv, time, math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as RosTime

CSV_FILE = "lf_circulo_1779765523.csv"
FRAME_ID = "map"
SPEED    = 1.0   # 1.0 = tiempo real, 2.0 = doble velocidad

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
    cy, sy = math.cos(float(yaw)/2), math.sin(float(yaw)/2)
    ps.pose.orientation.w = cy
    ps.pose.orientation.z = sy
    return ps

class ReplayNode(Node):
    def __init__(self, rows):
        super().__init__("lf_replay")
        self.rows = rows
        self.pub_L  = self.create_publisher(Path, "/leader/path",       10)
        self.pub_S  = self.create_publisher(Path, "/follower/path",     10)
        self.pub_Sd = self.create_publisher(Path, "/follower/setpoint", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/formation/marker", 10)
        self.path_L  = Path(); self.path_L.header.frame_id  = FRAME_ID
        self.path_S  = Path(); self.path_S.header.frame_id  = FRAME_ID
        self.path_Sd = Path(); self.path_Sd.header.frame_id = FRAME_ID
        self.idx    = 0
        self.t0_ros = None
        self.t0_dat = None
        self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.idx >= len(self.rows):
            return
        row   = self.rows[self.idx]
        t_rel = float(row["t"])
        if self.t0_ros is None:
            self.t0_ros = time.monotonic()
            self.t0_dat = t_rel
        if t_rel > (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat:
            return

        stamp = _stamp(float(row["timestamp_unix"]))
        lx, ly, lz = row["lx"], row["ly"], row["lz"]
        sx, sy, sz = row["sx"], row["sy"], row["sz"]
        xd, yd, zd = row["xd"], row["yd"], row["zd"]

        self.path_L.poses.append(_pose(lx, ly, lz, row["l_yaw"], stamp))
        self.path_S.poses.append(_pose(sx, sy, sz, row["s_yaw"], stamp))
        self.path_Sd.poses.append(_pose(xd, yd, zd, row["s_yaw"], stamp))

        for p in (self.path_L, self.path_S, self.path_Sd):
            p.header.stamp = stamp

        self.pub_L.publish(self.path_L)
        self.pub_S.publish(self.path_S)
        self.pub_Sd.publish(self.path_Sd)

        # Línea de formación
        mk = Marker()
        mk.header.frame_id = FRAME_ID; mk.header.stamp = stamp
        mk.ns = "formation"; mk.id = 0
        mk.type = Marker.LINE_LIST; mk.action = Marker.ADD
        mk.scale.x = 0.06
        mk.color.r = 1.0; mk.color.g = 0.4; mk.color.b = 0.0; mk.color.a = 0.9
        p1 = Point(); p1.x=float(lx); p1.y=float(ly); p1.z=float(lz)
        p2 = Point(); p2.x=float(sx); p2.y=float(sy); p2.z=float(sz)
        mk.points = [p1, p2]
        ma = MarkerArray(); ma.markers = [mk]
        self.pub_mk.publish(ma)

        self.idx += 1

def main():
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Cargadas {len(rows)} muestras de {CSV_FILE}")
    print(f"Duración: {float(rows[-1]['t']):.1f} s  |  Velocidad replay: {SPEED}x")
    rclpy.init()
    node = ReplayNode(rows)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()

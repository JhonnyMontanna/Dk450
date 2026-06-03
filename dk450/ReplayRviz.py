#!/usr/bin/env python3
"""
log_replay_rviz.py
==================
Reproduce un log CSV en RViz en tiempo real simulado.

Arquitectura:
  · MultiThreadedExecutor de ROS en hilo propio  → publishers/TF sin bloqueo
  · Hilo de replay con time.perf_counter/sleep    → timing preciso sin GIL

Drones: uav1 (Líder) · uav2 (Seguidor)

Topics por dron:
  /{ns}/gps/odom            nav_msgs/Odometry
  /{ns}/gps/drone_path      nav_msgs/Path
  /{ns}/gps/drone_marker    visualization_msgs/Marker
  TF: rtk_odom → {ns}_base_link / {ns}_base_footprint

USO:
  python log_replay_rviz.py --master vuelo1.csv
  python log_replay_rviz.py --master vuelo1.csv --speed 2.0 --loop
  python log_replay_rviz.py --master vuelo1.csv --speed 10.0 --no-path
"""

import sys
import math
import time
import threading
import argparse

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as RosTime


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CSV_FILE = "master_telemetry.csv"

PHASE_TIMES = {
    "TAKEOFF":     (0.0,    35.0),
    "POSITIONING": (35.0,   75.0),
    "TRAJECTORY":  (80.0,  125.0),
    "LANDING":     (150.0,  None),
}

R_EARTH   = 6_371_000.0
RADIUS    = 4.0

CENTER_OFFSET_X          = 0.0
CENTER_OFFSET_Y          = 0.0
CENTER_OFFSET_Z_LEADER   = 0.0
CENTER_OFFSET_Z_FOLLOWER = 0.0

CIRCLE_TRIM_START = 0.10
CIRCLE_TRIM_END   = 0.05

ELLIPSE_R_XY = 0.60
ELLIPSE_R_Z  = 0.20


# =============================================================================
# TRANSFORMACIÓN ENU
# =============================================================================

def gps_to_enu(lat, lon, ref_lat, ref_lon):
    x = math.radians(lat - ref_lat) * R_EARTH
    y = math.radians(lon - ref_lon) * R_EARTH * math.cos(math.radians(ref_lat))
    return x, y


def fit_circle(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    x, y = x[v], y[v]
    if len(x) < 4:
        return 0.0, 0.0, RADIUS, np.nan
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    res, _, _, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy = res[0], res[1]
    r   = math.sqrt(max(res[2] + cx**2 + cy**2, 0.0))
    rms = math.sqrt(np.mean((np.sqrt((x-cx)**2 + (y-cy)**2) - r)**2))
    return cx, cy, r, rms


def load_and_transform(path):
    print(f"\n[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python").sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas · {len(df.columns)} columnas")

    mask = df["L_lat"].notna() & df["L_lon"].notna()
    if not mask.any():
        raise ValueError("Sin GPS del líder en el CSV.")
    ref_lat = float(df[mask].iloc[0]["L_lat"])
    ref_lon = float(df[mask].iloc[0]["L_lon"])
    print(f"[ENU] Origen → lat={ref_lat:.8f}  lon={ref_lon:.8f}")

    for pfx in ("L", "S"):
        has = df[f"{pfx}_lat"].notna() & df[f"{pfx}_lon"].notna()
        ex = np.full(len(df), np.nan)
        ey = np.full(len(df), np.nan)
        if has.any():
            vals = df.loc[has, [f"{pfx}_lat", f"{pfx}_lon"]].apply(
                lambda r: gps_to_enu(r[f"{pfx}_lat"], r[f"{pfx}_lon"], ref_lat, ref_lon), axis=1)
            ex[has.values] = [v[0] for v in vals]
            ey[has.values] = [v[1] for v in vals]
        df[f"{pfx}_ex"], df[f"{pfx}_ey"] = ex, ey

    for pfx in ("L", "S"):
        off = CENTER_OFFSET_Z_LEADER if pfx == "L" else CENTER_OFFSET_Z_FOLLOWER
        if f"{pfx}_alt" in df.columns and df[f"{pfx}_alt"].notna().any():
            av = df[f"{pfx}_alt"].values.astype(float)
            df[f"{pfx}_cz"] = av - np.nanmin(av[:min(20, len(av))]) + off
        elif f"{pfx}_z" in df.columns and df[f"{pfx}_z"].notna().any():
            df[f"{pfx}_cz"] = -df[f"{pfx}_z"].values.astype(float) + off
        else:
            df[f"{pfx}_cz"] = np.nan

    # Ajuste círculo sobre TRAJECTORY
    t_arr = df["t"].values
    phases = np.full(len(t_arr), "UNKNOWN", dtype=object)
    t_max = t_arr[-1]
    for name, iv in PHASE_TIMES.items():
        if iv is None: continue
        t0, t1 = iv[0], (t_max if iv[1] is None else iv[1])
        phases[(t_arr >= t0) & (t_arr <= t1)] = name
    df["phase"] = phases

    traj = phases == "TRAJECTORY"
    if traj.sum() > 10:
        idx_t = np.where(traj)[0]
        nt = len(idx_t)
        fm = np.zeros(len(df), dtype=bool)
        fm[idx_t[int(nt*CIRCLE_TRIM_START)] : idx_t[max(0, int(nt*(1-CIRCLE_TRIM_END))-1)]+1] = True
        fm &= traj
        cx, cy, r, rms = fit_circle(df.loc[fm, "L_ex"].values, df.loc[fm, "L_ey"].values)
        print(f"[Círculo] ({cx:.3f}, {cy:.3f})  R={r:.3f} m  RMS={rms:.4f} m")
    else:
        cx, cy = 0.0, 0.0
        print("[Círculo] Sin datos TRAJECTORY suficientes.")

    ox, oy = cx + CENTER_OFFSET_X, cy + CENTER_OFFSET_Y
    for pfx in ("L", "S"):
        df[f"{pfx}_rx"] = df[f"{pfx}_ex"] - ox
        df[f"{pfx}_ry"] = df[f"{pfx}_ey"] - oy

    # Pre-convertir a numpy para acceso rápido en el loop
    cols = ["t", "phase",
            "L_rx", "L_ry", "L_cz", "S_rx", "S_ry", "S_cz",
            "L_qx", "L_qy", "L_qz", "L_qw",
            "S_qx", "S_qy", "S_qz", "S_qw"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    print(f"[Replay] {len(df)} muestras listas.\n")
    return df


# =============================================================================
# NODO ROS 2
# =============================================================================

def _ros_stamp(t: float) -> RosTime:
    msg = RosTime()
    msg.sec     = int(t)
    msg.nanosec = int((t % 1) * 1e9)
    return msg


class LogReplayNode(Node):

    def __init__(self, df: pd.DataFrame, speed: float, loop: bool, publish_path: bool):
        super().__init__("log_replay_rviz")
        self.speed        = speed
        self.loop         = loop
        self.publish_path = publish_path
        self.path_decim   = 10  # publicar path 1 de cada N muestras
        self._frame_count = 0

        self._last_phase = {"uav1": None, "uav2": None}

        # Pre-extraer arrays numpy para acceso O(1) sin iloc
        self.t_arr  = df["t"].values.astype(float)
        self.n      = len(self.t_arr)
        self.phases = df["phase"].values

        def _arr(col):
            s = df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
            v = s.values.astype(float)
            v[np.isnan(v)] = 0.0
            return v

        self.lx = _arr("L_rx"); self.ly = _arr("L_ry"); self.lz = _arr("L_cz")
        self.sx = _arr("S_rx"); self.sy = _arr("S_ry"); self.sz = _arr("S_cz")

        # Cuaterniones (si no existen quedan en 0 → w se fuerza a 1 abajo)
        self.lqx = _arr("L_qx"); self.lqy = _arr("L_qy")
        self.lqz = _arr("L_qz"); self.lqw = _arr("L_qw")
        self.sqx = _arr("S_qx"); self.sqy = _arr("S_qy")
        self.sqz = _arr("S_qz"); self.sqw = _arr("S_qw")

        # Si las columnas de cuaternión no existían en el CSV, forzar w=1
        if df["L_qw"].isna().all(): self.lqw[:] = 1.0
        if df["S_qw"].isna().all(): self.sqw[:] = 1.0

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.tf_br = TransformBroadcaster(self)

        self.pubs = {}
        for ns, ec in [
            ("uav1", ColorRGBA(r=0.0, g=0.85, b=0.2,  a=0.30)),
            ("uav2", ColorRGBA(r=1.0, g=0.50, b=0.0,  a=0.30)),
        ]:
            self.pubs[ns] = {
                "odom":      self.create_publisher(Odometry, f"/{ns}/gps/odom",         qos),
                "path":      self.create_publisher(Path,     f"/{ns}/gps/drone_path",   qos),
                "marker":    self.create_publisher(Marker,   f"/{ns}/gps/drone_marker", qos),
                "ell_color": ec,
                "path_msg":  self._mk_path(),
            }

        self._stop = threading.Event()
        # Hilo de replay: SOLO duerme y llama publish, no toca el executor
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def _mk_path(self):
        p = Path(); p.header.frame_id = "rtk_odom"; return p

    # ── publicación de un frame ───────────────────────────────────────────────


    def _pub_frame(self, ns, x, y, z, qx, qy, qz, qw, stamp, idx=0):
        p = self.pubs[ns]

        q = Quaternion(); q.x=qx; q.y=qy; q.z=qz; q.w=qw

        # TF base_link
        tf = TransformStamped()
        tf.header.stamp = stamp; tf.header.frame_id = "rtk_odom"
        tf.child_frame_id = f"{ns}_base_link"
        tf.transform.translation.x=x; tf.transform.translation.y=y; tf.transform.translation.z=z
        tf.transform.rotation = q
        self.tf_br.sendTransform(tf)

        # TF base_footprint
        tf2 = TransformStamped()
        tf2.header.stamp = stamp; tf2.header.frame_id = "rtk_odom"
        tf2.child_frame_id = f"{ns}_base_footprint"
        tf2.transform.translation.x=x; tf2.transform.translation.y=y; tf2.transform.translation.z=0.0
        tf2.transform.rotation.w = 1.0
        self.tf_br.sendTransform(tf2)

        # Odometry
        odom = Odometry()
        odom.header.stamp=stamp; odom.header.frame_id="rtk_odom"
        odom.child_frame_id=f"{ns}_base_link"
        odom.pose.pose.position.x=x; odom.pose.pose.position.y=y; odom.pose.pose.position.z=z
        odom.pose.pose.orientation=q
        p["odom"].publish(odom)

        # Path
        # Path — solo 1 de cada path_decim muestras para evitar serialización cuadrática
      #  self._frame_count += 1

    

        if self.publish_path:
            current_phase = self.phases[idx]
            if current_phase != self._last_phase[ns]:
                # Fase nueva: borrar path anterior con DELETE y reiniciar
                del_path = self._mk_path()
                del_path.header.stamp = stamp
                p["path"].publish(del_path)
                p["path_msg"] = self._mk_path()
                self._last_phase[ns] = current_phase

            if idx % self.path_decim == 0:
                ps = PoseStamped(); ps.header=odom.header; ps.pose=odom.pose.pose
                pm = p["path_msg"]; pm.header.stamp=stamp; pm.poses.append(ps)
                if len(pm.poses) > 500:
                    pm.poses = pm.poses[-500:]
                p["path"].publish(pm)

        # Marker texto
        mk = Marker(); mk.header=odom.header
        mk.ns=f"{ns}_text"; mk.id=0; mk.type=Marker.TEXT_VIEW_FACING; mk.action=Marker.ADD
        mk.pose.position.x=x; mk.pose.position.y=y; mk.pose.position.z=z+0.8
        mk.pose.orientation.w=1.0; mk.scale.z=0.4
        mk.color=ColorRGBA(r=1.0,g=1.0,b=1.0,a=1.0)
        mk.text=f"{ns}: x={x:.2f} y={y:.2f} z={z:.2f}"
        p["marker"].publish(mk)

        # Marker esfera
        el = Marker(); el.header=odom.header
        el.ns=f"{ns}_ellipse"; el.id=1; el.type=Marker.SPHERE; el.action=Marker.ADD
        el.pose.position.x=x; el.pose.position.y=y; el.pose.position.z=z
        el.pose.orientation.w=1.0
        el.scale.x=2*ELLIPSE_R_XY; el.scale.y=2*ELLIPSE_R_XY; el.scale.z=2*ELLIPSE_R_Z
        el.color=p["ell_color"]
        p["marker"].publish(el)

    # ── hilo de replay ────────────────────────────────────────────────────────

    def _replay_loop(self):
        t_arr = self.t_arr
        n     = self.n

        print(f"\n[DEBUG] speed={self.speed}  n={self.n}  "
              f"duracion_log={t_arr[-1]-t_arr[0]:.1f}s  "
              f"duracion_esperada={( t_arr[-1]-t_arr[0])/self.speed:.1f}s\n")

        while not self._stop.is_set():
            log_t0   = t_arr[0]
            wall_t0  = time.perf_counter()

            for i in range(n):
                if self._stop.is_set():
                    return

                log_t    = t_arr[i]
                deadline = wall_t0 + (log_t - log_t0) / self.speed
                remaining = deadline - time.perf_counter()
                if remaining > 1e-4:          # solo dormir si vale la pena
                    time.sleep(remaining)

                stamp = self.get_clock().now().to_msg()
           

                self._pub_frame("uav1",
                                self.lx[i], self.ly[i], self.lz[i],
                                self.lqx[i], self.lqy[i], self.lqz[i], self.lqw[i],
                                stamp, idx=i)
                self._pub_frame("uav2",
                                self.sx[i], self.sy[i], self.sz[i],
                                self.sqx[i], self.sqy[i], self.sqz[i], self.sqw[i],
                                stamp, idx=i)
                self._frame_count += 1  # ← agregar aquí


                if i % 50 == 0:
                    wall_elapsed = time.perf_counter() - wall_t0
                    esperado     = (log_t - log_t0) / self.speed
                    self.get_logger().info(
                        f"[Replay] log={log_t:.1f}s  "
                        f"wall={wall_elapsed:.2f}s  esperado={esperado:.2f}s  "
                        f"ratio={wall_elapsed/(esperado+1e-9):.2f}x  "
                        f"speed_param={self.speed}"
                    )

            if not self.loop:
                self.get_logger().info("[Replay] Fin. Ctrl+C para salir.")
                return

            self.get_logger().info("[Replay] Reiniciando (--loop)…")
            for ns in self.pubs:
                self.pubs[ns]["path_msg"] = self._mk_path()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master",  default=CSV_FILE)
    ap.add_argument("--speed",   type=float, default=1.0,
                    help="Factor velocidad (1.0=tiempo real, 10.0=x10)")
    ap.add_argument("--loop",    action="store_true")
    ap.add_argument("--no-path", action="store_true")
    ap.add_argument("--path-decim", type=int, default=10,
                    help="Publicar 1 de cada N poses en el Path (default: 10)")
    ap.add_argument("--ox",  type=float, default=None)
    ap.add_argument("--oy",  type=float, default=None)
    ap.add_argument("--ozl", type=float, default=None)
    ap.add_argument("--ozs", type=float, default=None)
    args = ap.parse_args()

    global CENTER_OFFSET_X, CENTER_OFFSET_Y
    global CENTER_OFFSET_Z_LEADER, CENTER_OFFSET_Z_FOLLOWER
    if args.ox  is not None: CENTER_OFFSET_X          = args.ox
    if args.oy  is not None: CENTER_OFFSET_Y          = args.oy
    if args.ozl is not None: CENTER_OFFSET_Z_LEADER   = args.ozl
    if args.ozs is not None: CENTER_OFFSET_Z_FOLLOWER = args.ozs

    try:
        df = load_and_transform(args.master)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró '{args.master}'"); sys.exit(1)

    rclpy.init()
    node = LogReplayNode(df, speed=args.speed, loop=args.loop,
                         publish_path=not args.no_path)
    node.path_decim = args.path_decim

    # MultiThreadedExecutor: el executor corre en su propio hilo del OS,
    # completamente separado del hilo de replay → sin contención de GIL
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("\n✅ Replay terminado.")


if __name__ == "__main__":
    main()
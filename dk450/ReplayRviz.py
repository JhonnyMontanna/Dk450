#!/usr/bin/env python3
"""
log_replay_rviz.py
==================
Reproduce un log CSV (master_telemetry.csv) en RViz en tiempo real simulado.

Combina:
  · La lógica de transformación ENU + offsets de ReplayTotal.py
  · Los publishers/TF/Markers de MavrosVisualizer.py

Drones publicados:
  · uav1  — Líder
  · uav2  — Seguidor

Topics publicados por dron:
  /{ns}/gps/odom            nav_msgs/Odometry
  /{ns}/gps/drone_path      nav_msgs/Path
  /{ns}/gps/drone_marker    visualization_msgs/Marker (texto + esfera)
  TF:  rtk_odom → {ns}_base_link
       rtk_odom → {ns}_base_footprint

USO:
  python log_replay_rviz.py
  python log_replay_rviz.py --master vuelo1.csv
  python log_replay_rviz.py --master vuelo1.csv --speed 2.0 --loop
  python log_replay_rviz.py --master vuelo1.csv --speed 0.5

PARÁMETROS:
  --master   CSV de telemetría        (default: master_telemetry.csv)
  --speed    Factor de velocidad      (default: 1.0 = tiempo real, 2.0 = doble)
  --loop     Repetir al terminar
  --no-path  No publicar Path (logs muy largos)
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
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as RosTime


# =============================================================================
# CONFIGURACIÓN — ajustar igual que en ReplayTotal.py
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
OFFSET_D  = 2.0
OFFSET_DZ = 1.0

CENTER_OFFSET_X          = 0.0
CENTER_OFFSET_Y          = 0.0
CENTER_OFFSET_Z_LEADER   = 0.0
CENTER_OFFSET_Z_FOLLOWER = 0.0

CIRCLE_TRIM_START = 0.10
CIRCLE_TRIM_END   = 0.05

ELLIPSE_SCALE_XY = 0.60   # radio visual dron XY [m]
ELLIPSE_SCALE_Z  = 0.20   # radio visual dron Z  [m]


# =============================================================================
# TRANSFORMACIÓN ENU (de ReplayTotal.py)
# =============================================================================

def gps_to_enu(lat, lon, ref_lat, ref_lon):
    ref_lat_r = math.radians(ref_lat)
    x = math.radians(lat - ref_lat) * R_EARTH
    y = math.radians(lon - ref_lon) * R_EARTH * math.cos(ref_lat_r)
    return x, y


def fit_circle(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return 0.0, 0.0, RADIUS, np.nan
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = res[0], res[1]
    r = math.sqrt(max(res[2] + cx**2 + cy**2, 0.0))
    resid = np.sqrt((x - cx)**2 + (y - cy)**2) - r
    rms = math.sqrt(np.mean(resid**2))
    return cx, cy, r, rms


def assign_phases(t_arr, phase_times):
    phases = np.full(len(t_arr), "UNKNOWN", dtype=object)
    t_max = t_arr[-1]
    for name, interval in phase_times.items():
        if interval is None:
            continue
        t0, t1 = interval
        t1 = t_max if t1 is None else t1
        phases[(t_arr >= t0) & (t_arr <= t1)] = name
    return phases


def load_and_transform(path):
    print(f"\n[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas · {len(df.columns)} columnas")

    mask_gps = df["L_lat"].notna() & df["L_lon"].notna()
    if not mask_gps.any():
        raise ValueError("No hay GPS del líder (L_lat/L_lon) en el CSV.")
    first   = df[mask_gps].iloc[0]
    ref_lat = float(first["L_lat"])
    ref_lon = float(first["L_lon"])
    print(f"[ENU] Origen → lat={ref_lat:.8f}  lon={ref_lon:.8f}")

    for pfx in ("L", "S"):
        clat, clon = f"{pfx}_lat", f"{pfx}_lon"
        has = df[clat].notna() & df[clon].notna()
        ex = np.full(len(df), np.nan)
        ey = np.full(len(df), np.nan)
        if has.any():
            vals = df.loc[has, [clat, clon]].apply(
                lambda r: gps_to_enu(r[clat], r[clon], ref_lat, ref_lon), axis=1)
            ex[has.values] = [v[0] for v in vals]
            ey[has.values] = [v[1] for v in vals]
        df[f"{pfx}_ex"] = ex
        df[f"{pfx}_ey"] = ey

    for pfx in ("L", "S"):
        alt_col = f"{pfx}_alt"
        z_col   = f"{pfx}_z"
        cz_col  = f"{pfx}_cz"
        off = CENTER_OFFSET_Z_LEADER if pfx == "L" else CENTER_OFFSET_Z_FOLLOWER
        if alt_col in df.columns and df[alt_col].notna().any():
            alt_vals   = df[alt_col].values.astype(float)
            alt_ground = np.nanmin(alt_vals[:min(20, len(alt_vals))])
            df[cz_col] = alt_vals - alt_ground + off
        elif z_col in df.columns and df[z_col].notna().any():
            df[cz_col] = -df[z_col].values.astype(float) + off
        else:
            df[cz_col] = np.nan

    t_arr  = df["t"].values
    phases = assign_phases(t_arr, PHASE_TIMES)
    df["phase"] = phases

    traj_mask = phases == "TRAJECTORY"
    if traj_mask.sum() > 10:
        idx_traj = np.where(traj_mask)[0]
        n_t  = len(idx_traj)
        i0_t = idx_traj[int(n_t * CIRCLE_TRIM_START)]
        i1_t = idx_traj[max(0, int(n_t * (1 - CIRCLE_TRIM_END)) - 1)]
        fit_mask = np.zeros(len(df), dtype=bool)
        fit_mask[i0_t:i1_t+1] = True
        fit_mask &= traj_mask
        cx_fit, cy_fit, r_fit, rms_fit = fit_circle(
            df.loc[fit_mask, "L_ex"].values,
            df.loc[fit_mask, "L_ey"].values)
        print(f"[Círculo] Centro ENU: ({cx_fit:.3f}, {cy_fit:.3f})  "
              f"R={r_fit:.3f} m  RMS={rms_fit:.4f} m")
    else:
        cx_fit, cy_fit, r_fit, rms_fit = 0.0, 0.0, RADIUS, np.nan

    ox = cx_fit + CENTER_OFFSET_X
    oy = cy_fit + CENTER_OFFSET_Y
    df["L_rx"] = df["L_ex"] - ox
    df["L_ry"] = df["L_ey"] - oy
    df["S_rx"] = df["S_ex"] - ox
    df["S_ry"] = df["S_ey"] - oy

    print(f"[Replay] {len(df)} muestras listas.\n")
    return df


# =============================================================================
# NODO ROS 2
# =============================================================================

def secs_to_ros_time(t: float) -> RosTime:
    msg = RosTime()
    msg.sec     = int(t)
    msg.nanosec = int((t - int(t)) * 1e9)
    return msg


class LogReplayNode(Node):

    def __init__(self, df: pd.DataFrame, speed: float, loop: bool, publish_path: bool):
        super().__init__("log_replay_rviz")

        self.df           = df
        self.speed        = speed
        self.loop         = loop
        self.publish_path = publish_path

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.tf_br = TransformBroadcaster(self)

        self.pubs = {}
        for ns, ellipse_color in [
            ("uav1", ColorRGBA(r=0.0,  g=0.85, b=0.2,  a=0.30)),
            ("uav2", ColorRGBA(r=1.0,  g=0.50, b=0.0,  a=0.30)),
        ]:
            self.pubs[ns] = {
                "odom":     self.create_publisher(Odometry, f"/{ns}/gps/odom",         qos),
                "path":     self.create_publisher(Path,     f"/{ns}/gps/drone_path",   qos),
                "marker":   self.create_publisher(Marker,   f"/{ns}/gps/drone_marker", qos),
                "ell_color": ellipse_color,
                "path_msg":  self._empty_path(),
            }

        # Hilo de reproducción independiente del executor de ROS
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _empty_path(self):
        p = Path()
        p.header.frame_id = "rtk_odom"
        return p

    def _publish_drone(self, ns: str, x: float, y: float, z: float,
                       quat: Quaternion, stamp: RosTime):
        p = self.pubs[ns]

        # TF base_link
        tf = TransformStamped()
        tf.header.stamp        = stamp
        tf.header.frame_id     = "rtk_odom"
        tf.child_frame_id      = f"{ns}_base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation      = quat
        self.tf_br.sendTransform(tf)

        # TF base_footprint
        tf2 = TransformStamped()
        tf2.header.stamp        = stamp
        tf2.header.frame_id     = "rtk_odom"
        tf2.child_frame_id      = f"{ns}_base_footprint"
        tf2.transform.translation.x = x
        tf2.transform.translation.y = y
        tf2.transform.translation.z = 0.0
        tf2.transform.rotation.w    = 1.0
        self.tf_br.sendTransform(tf2)

        # Odometry
        odom = Odometry()
        odom.header.stamp       = stamp
        odom.header.frame_id    = "rtk_odom"
        odom.child_frame_id     = f"{ns}_base_link"
        odom.pose.pose.position.x   = x
        odom.pose.pose.position.y   = y
        odom.pose.pose.position.z   = z
        odom.pose.pose.orientation  = quat
        p["odom"].publish(odom)

        # Path
        if self.publish_path:
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose   = odom.pose.pose
            p["path_msg"].header.stamp = stamp
            p["path_msg"].poses.append(ps)
            p["path"].publish(p["path_msg"])

        # Marker texto
        text = Marker()
        text.header = odom.header
        text.ns     = f"{ns}_text"
        text.id     = 0
        text.type   = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x  = x
        text.pose.position.y  = y
        text.pose.position.z  = z + 0.8
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text    = f"{ns}: x={x:.2f}  y={y:.2f}  z={z:.2f}"
        p["marker"].publish(text)

        # Marker esfera
        ell = Marker()
        ell.header = odom.header
        ell.ns     = f"{ns}_ellipse"
        ell.id     = 1
        ell.type   = Marker.SPHERE
        ell.action = Marker.ADD
        ell.pose.position.x  = x
        ell.pose.position.y  = y
        ell.pose.position.z  = z
        ell.pose.orientation.w = 1.0
        ell.scale.x = 2.0 * ELLIPSE_SCALE_XY
        ell.scale.y = 2.0 * ELLIPSE_SCALE_XY
        ell.scale.z = 2.0 * ELLIPSE_SCALE_Z
        ell.color   = p["ell_color"]
        p["marker"].publish(ell)

    # ── hilo de reproducción ──────────────────────────────────────────────────

    def _replay_loop(self):
        df     = self.df
        t_arr  = df["t"].values
        n      = len(df)

        def _get_quat(row, pfx):
            q = Quaternion()
            q.w = 1.0
            for field, col in [("x", f"{pfx}_qx"), ("y", f"{pfx}_qy"),
                                ("z", f"{pfx}_qz"), ("w", f"{pfx}_qw")]:
                if col in df.columns:
                    v = row[col]
                    if not (isinstance(v, float) and math.isnan(v)):
                        setattr(q, field, float(v))
            return q

        def _safe(v):
            return float(v) if not (isinstance(v, float) and math.isnan(v)) else 0.0

        while not self._stop_event.is_set():
            # ── Origen de tiempo para esta pasada ────────────────────────────
            wall_origin  = time.perf_counter()
            log_t_origin = float(t_arr[0])

            for idx in range(n):
                if self._stop_event.is_set():
                    return

                log_t    = float(t_arr[idx])
                # Momento en que este sample debería publicarse
                deadline = wall_origin + (log_t - log_t_origin) / self.speed

                # Dormir exactamente hasta el deadline
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)

                row = df.iloc[idx]
                lx  = _safe(row["L_rx"])
                ly  = _safe(row["L_ry"])
                lz  = _safe(row["L_cz"])
                sx  = _safe(row["S_rx"])
                sy  = _safe(row["S_ry"])
                sz  = _safe(row["S_cz"])

                stamp = secs_to_ros_time(log_t)
                self._publish_drone("uav1", lx, ly, lz, _get_quat(row, "L"), stamp)
                self._publish_drone("uav2", sx, sy, sz, _get_quat(row, "S"), stamp)

                if idx % 200 == 0:
                    elapsed_wall = time.perf_counter() - wall_origin
                    elapsed_log  = log_t - log_t_origin
                    self.get_logger().info(
                        f"[Replay] log_t={log_t:.1f}s  "
                        f"wall={elapsed_wall:.1f}s  "
                        f"fase={row.get('phase','?')}  "
                        f"L=({lx:.2f},{ly:.2f},{lz:.2f})  "
                        f"S=({sx:.2f},{sy:.2f},{sz:.2f})"
                    )

            # Fin de pasada
            if not self.loop:
                self.get_logger().info("[Replay] Fin del log. Ctrl+C para salir.")
                return

            self.get_logger().info("[Replay] Reiniciando (--loop)…")
            for ns in self.pubs:
                self.pubs[ns]["path_msg"] = self._empty_path()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master",  default=CSV_FILE,
                    help=f"CSV de telemetría (default: {CSV_FILE})")
    ap.add_argument("--speed",   type=float, default=1.0,
                    help="Factor de velocidad (1.0=tiempo real, 2.0=doble, 0.5=mitad)")
    ap.add_argument("--loop",    action="store_true",
                    help="Repetir el log al terminar")
    ap.add_argument("--no-path", action="store_true",
                    help="No publicar Path")
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
        print(f"\n[ERROR] No se encontró '{args.master}'\n"
              f"        Usa --master /ruta/al/archivo.csv")
        sys.exit(1)

    rclpy.init()
    node = LogReplayNode(df, speed=args.speed, loop=args.loop,
                         publish_path=not args.no_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\n✅ Replay terminado.")


if __name__ == "__main__":
    main()
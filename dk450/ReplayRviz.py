#!/usr/bin/env python3
"""
log_replay_rviz.py
==================
Reproduce un log CSV (master_telemetry.csv) en RViz en "tiempo real" simulado.

Combina:
  · La lógica de transformación ENU + offsets de ReplayTotal.py
  · Los publishers/TF/Markers de MavrosVisualizer.py

Drones publicados:
  · uav1  — Líder   (azul en RViz)
  · uav2  — Seguidor (rojo en RViz)

Topics publicados por dron (mismo esquema que MavrosVisualizer):
  /{ns}/gps/odom            nav_msgs/Odometry
  /{ns}/gps/drone_path      nav_msgs/Path
  /{ns}/gps/drone_marker    visualization_msgs/Marker  (texto + esfera)
  TF:  rtk_odom → {ns}_base_link
       rtk_odom → {ns}_base_footprint

USO:
  python log_replay_rviz.py
  python log_replay_rviz.py --master vuelo1.csv
  python log_replay_rviz.py --master vuelo1.csv --speed 2.0 --loop
  python log_replay_rviz.py --master vuelo1.csv --speed 0  # paso a paso

PARÁMETROS CLAVE:
  --master   CSV de telemetría  (default: master_telemetry.csv)
  --speed    Factor de velocidad de reproducción  (default: 1.0)
             0 = paso a paso (pulsa Enter para avanzar)
  --loop     Repetir indefinidamente al terminar
  --no-path  No publicar los Path (útil si el log es muy largo)
"""

import sys
import math
import time
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
# ── CONFIGURACIÓN (igual que ReplayTotal.py) ──────────────────────────────────
# =============================================================================

CSV_FILE = "master_telemetry.csv"

# Fases de vuelo  ← ajustar igual que en ReplayTotal
PHASE_TIMES = {
    "TAKEOFF":     (0.0,    35.0),
    "POSITIONING": (35.0,   75.0),
    "TRAJECTORY":  (80.0,  125.0),
    "LANDING":     (150.0,  None),
}

R_EARTH  = 6_371_000.0
RADIUS   = 4.0
OFFSET_D = 2.0
OFFSET_DZ = 1.0

CENTER_OFFSET_X          =  0.0
CENTER_OFFSET_Y          =  0.0
CENTER_OFFSET_Z_LEADER   =  0.0
CENTER_OFFSET_Z_FOLLOWER =  0.0

CIRCLE_TRIM_START = 0.10
CIRCLE_TRIM_END   = 0.05


# =============================================================================
# ── UTILIDADES (de ReplayTotal.py) ────────────────────────────────────────────
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
        mask = (t_arr >= t0) & (t_arr <= t1)
        phases[mask] = name
    return phases


def phase_segments(phases, label):
    segs, n, i = [], len(phases), 0
    while i < n:
        if phases[i] == label:
            j = i
            while j < n and phases[j] == label:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def load_and_transform(path):
    print(f"\n[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas · {len(df.columns)} columnas")

    # Origen ENU
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

    # Ajuste círculo
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
        print(f"[Círculo] Centro ENU: ({cx_fit:.3f}, {cy_fit:.3f})  R={r_fit:.3f} m  RMS={rms_fit:.4f} m")
    else:
        cx_fit, cy_fit, r_fit, rms_fit = 0.0, 0.0, RADIUS, np.nan

    ox = cx_fit + CENTER_OFFSET_X
    oy = cy_fit + CENTER_OFFSET_Y
    df["L_rx"] = df["L_ex"] - ox
    df["L_ry"] = df["L_ey"] - oy
    df["S_rx"] = df["S_ex"] - ox
    df["S_ry"] = df["S_ey"] - oy

    print(f"[Replay] {len(df)} muestras listas para reproducir.\n")
    return df


# =============================================================================
# ── NODO ROS 2 ────────────────────────────────────────────────────────────────
# =============================================================================

# Colores RGBA para los drones
COLOR_LEADER   = ColorRGBA(r=0.13, g=0.39, b=0.75, a=1.0)   # azul
COLOR_FOLLOWER = ColorRGBA(r=0.78, g=0.16, b=0.16, a=1.0)   # rojo

# Color elipsoide (sin modo GPS en replay, usamos verde fijo)
COLOR_ELLIPSE = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.25)

ELLIPSE_SCALE_XY = 0.60   # radio visual del dron en XY [m]
ELLIPSE_SCALE_Z  = 0.20   # radio visual en Z [m]


def float_to_ros_time(t_float):
    """Convierte segundos float a builtin_interfaces/Time."""
    sec     = int(t_float)
    nanosec = int((t_float - sec) * 1e9)
    msg = RosTime()
    msg.sec     = sec
    msg.nanosec = nanosec
    return msg


class LogReplayNode(Node):
    def __init__(self, df: pd.DataFrame, speed: float, loop: bool, publish_path: bool):
        super().__init__("log_replay_rviz")

        self.df           = df
        self.speed        = speed          # factor de velocidad (0 = paso a paso)
        self.loop         = loop
        self.publish_path = publish_path

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.tf_br = TransformBroadcaster(self)

        # Publishers para cada dron
        self.pubs = {}
        for ns, color in [("uav1", COLOR_LEADER), ("uav2", COLOR_FOLLOWER)]:
            self.pubs[ns] = {
                "odom":   self.create_publisher(Odometry, f"/{ns}/gps/odom",        qos),
                "path":   self.create_publisher(Path,     f"/{ns}/gps/drone_path",  qos),
                "marker": self.create_publisher(Marker,   f"/{ns}/gps/drone_marker", qos),
                "color":  color,
                "path_msg": self._empty_path(),
            }

        self.idx = 0
        self.t_arr = df["t"].values

        # Timer de reproducción (se ajusta dinámicamente según dt real del log)
        # Usamos un timer rápido y controlamos el tiempo internamente
        self._replay_active = True
        self._last_wall     = None
        self._last_log_t    = None

        # Disparar la reproducción en un timer de 10 ms
        self.timer = self.create_timer(0.01, self._step_cb)

        self.get_logger().info(
            f"LogReplayNode listo | {len(df)} muestras | "
            f"speed={speed if speed > 0 else 'paso-a-paso'} | loop={loop}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty_path(self):
        p = Path()
        p.header.frame_id = "rtk_odom"
        return p

    def _ros_now_from_log_t(self, log_t: float):
        """Usa el tiempo del log como stamp (para que RViz muestre el tiempo real del vuelo)."""
        return float_to_ros_time(log_t)

    def _publish_drone(self, ns: str, x: float, y: float, z: float,
                       quat: Quaternion, stamp: RosTime):
        pubs  = self.pubs[ns]
        color = pubs["color"]

        # ── TF base_link ──────────────────────────────────────────────────────
        tf = TransformStamped()
        tf.header.stamp          = stamp
        tf.header.frame_id       = "rtk_odom"
        tf.child_frame_id        = f"{ns}_base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation       = quat
        self.tf_br.sendTransform(tf)

        # ── TF base_footprint ─────────────────────────────────────────────────
        tf2 = TransformStamped()
        tf2.header.stamp          = stamp
        tf2.header.frame_id       = "rtk_odom"
        tf2.child_frame_id        = f"{ns}_base_footprint"
        tf2.transform.translation.x = x
        tf2.transform.translation.y = y
        tf2.transform.translation.z = 0.0
        tf2.transform.rotation.w    = 1.0
        self.tf_br.sendTransform(tf2)

        # ── Odometry ──────────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = "rtk_odom"
        odom.child_frame_id  = f"{ns}_base_link"
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = z
        odom.pose.pose.orientation   = quat
        pubs["odom"].publish(odom)

        # ── Path ──────────────────────────────────────────────────────────────
        if self.publish_path:
            ps = PoseStamped()
            ps.header = odom.header
            ps.pose   = odom.pose.pose
            pubs["path_msg"].header.stamp = stamp
            pubs["path_msg"].poses.append(ps)
            pubs["path"].publish(pubs["path_msg"])

        # ── Marker texto ──────────────────────────────────────────────────────
        text = Marker()
        text.header    = odom.header
        text.ns        = f"{ns}_text"
        text.id        = 0
        text.type      = Marker.TEXT_VIEW_FACING
        text.action    = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = z + 0.8
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text    = f"{ns}: x={x:.2f}  y={y:.2f}  z={z:.2f}"
        pubs["marker"].publish(text)

        # ── Marker esfera (elipsoide de posición) ─────────────────────────────
        ell = Marker()
        ell.header = odom.header
        ell.ns     = f"{ns}_ellipse"
        ell.id     = 1
        ell.type   = Marker.SPHERE
        ell.action = Marker.ADD
        ell.pose.position.x = x
        ell.pose.position.y = y
        ell.pose.position.z = z
        ell.pose.orientation.w = 1.0
        ell.scale.x = 2.0 * ELLIPSE_SCALE_XY
        ell.scale.y = 2.0 * ELLIPSE_SCALE_XY
        ell.scale.z = 2.0 * ELLIPSE_SCALE_Z
        ell.color   = COLOR_ELLIPSE
        # Tint según dron: líder verde, seguidor naranja
        if ns == "uav1":
            ell.color = ColorRGBA(r=0.0, g=0.8, b=0.2, a=0.30)
        else:
            ell.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.30)
        pubs["marker"].publish(ell)

    # ── Timer callback ────────────────────────────────────────────────────────

    def _step_cb(self):
        if not self._replay_active:
            return

        df   = self.df
        idx  = self.idx
        n    = len(df)

        if idx >= n:
            if self.loop:
                self.get_logger().info("[Replay] Reiniciando (--loop)…")
                self.idx = 0
                for ns in self.pubs:
                    self.pubs[ns]["path_msg"] = self._empty_path()
                self._last_wall  = None
                self._last_log_t = None
                return
            else:
                self.get_logger().info("[Replay] Fin del log. Ctrl+C para salir.")
                self._replay_active = False
                self.timer.cancel()
                return

        log_t = float(self.t_arr[idx])
        wall  = time.monotonic()

        # Control de tiempo real simulado
        if self.speed > 0:
            if self._last_wall is not None:
                dt_log  = log_t - self._last_log_t          # dt en el log
                dt_wall = wall  - self._last_wall            # dt real transcurrido
                dt_needed = dt_log / self.speed              # dt que debería haber pasado

                if dt_wall < dt_needed:
                    return   # aún no es el momento de publicar este sample
        # speed == 0: paso a paso — se publica y luego espera el Enter en el main

        row = df.iloc[idx]

        # Extraer posiciones ya transformadas (L_rx, L_ry, L_cz, S_rx, S_ry, S_cz)
        lx = float(row["L_rx"]) if not np.isnan(row["L_rx"]) else 0.0
        ly = float(row["L_ry"]) if not np.isnan(row["L_ry"]) else 0.0
        lz = float(row["L_cz"]) if not np.isnan(row["L_cz"]) else 0.0
        sx = float(row["S_rx"]) if not np.isnan(row["S_rx"]) else 0.0
        sy = float(row["S_ry"]) if not np.isnan(row["S_ry"]) else 0.0
        sz = float(row["S_cz"]) if not np.isnan(row["S_cz"]) else 0.0

        # Orientación: usar columnas de cuaternión si existen, si no identidad
        def get_quat(pfx):
            q = Quaternion()
            q.w = 1.0
            for field, col in [("x", f"{pfx}_qx"), ("y", f"{pfx}_qy"),
                                ("z", f"{pfx}_qz"), ("w", f"{pfx}_qw")]:
                if col in df.columns and not np.isnan(row[col]):
                    setattr(q, field, float(row[col]))
            return q

        stamp = self._ros_now_from_log_t(log_t)

        self._publish_drone("uav1", lx, ly, lz, get_quat("L"), stamp)
        self._publish_drone("uav2", sx, sy, sz, get_quat("S"), stamp)

        self._last_wall  = wall
        self._last_log_t = log_t
        self.idx += 1

        # Log de progreso cada 10 s simulados
        if idx % 200 == 0:
            self.get_logger().info(
                f"[Replay] t={log_t:.1f} s  idx={idx}/{n}  "
                f"fase={row.get('phase','?')}  "
                f"L=({lx:.2f},{ly:.2f},{lz:.2f})  S=({sx:.2f},{sy:.2f},{sz:.2f})"
            )


# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master",   default=CSV_FILE,
                    help=f"CSV de telemetría (default: {CSV_FILE})")
    ap.add_argument("--speed",    type=float, default=1.0,
                    help="Factor de velocidad (1.0 = tiempo real, 2.0 = doble, 0 = paso a paso)")
    ap.add_argument("--loop",     action="store_true",
                    help="Repetir el log al terminar")
    ap.add_argument("--no-path",  action="store_true",
                    help="No publicar Path (útil para logs muy largos)")
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

    if args.speed == 0:
        # Modo paso a paso interactivo
        node = LogReplayNode(df, speed=0, loop=args.loop,
                             publish_path=not args.no_path)
        print("\n[Paso-a-paso] Pulsa Enter para publicar cada muestra. Ctrl+C para salir.\n")
        try:
            while rclpy.ok() and node._replay_active:
                input()        # espera Enter
                node._step_cb()
                rclpy.spin_once(node, timeout_sec=0.01)
        except KeyboardInterrupt:
            pass
    else:
        node = LogReplayNode(df, speed=args.speed, loop=args.loop,
                             publish_path=not args.no_path)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass

    node.destroy_node()
    rclpy.shutdown()
    print("\n✅ Replay terminado.")


if __name__ == "__main__":
    main()
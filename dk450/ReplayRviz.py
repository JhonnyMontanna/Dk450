#!/usr/bin/env python3
"""
rviz_replay_v2.py
=================
Reproduce en RViz2 la trayectoria líder-seguidor desde master_telemetry.csv.
Basado en MavrosVisualizer.py — usa el mismo pipeline GPS→ECEF→ENU→RTK local.

COORDENADAS:
  El mismo cálculo que MultiDroneVisualizer:
    GPS (lat,lon,alt) → ECEF → ENU → rotación theta → frame rtk_odom
  Así la trayectoria replay se alinea perfectamente con el visualizador en vivo.

DIFERENCIACIÓN VISUAL (RViz2 no soporta líneas punteadas):
  Drones  → color fijo   (líder: azul  #1565C0 | seguidor: rojo #C62828)
  Fases   → grosor + alpha de línea:
              TAKEOFF      0.03 m  alpha 0.35
              POSITIONING  0.05 m  alpha 0.65
              TRAJECTORY   0.10 m  alpha 1.00
              LANDING      0.03 m  alpha 0.45

MARCADORES:
  ★ Esfera grande  — inicio físico de cada dron
  ● Esfera media   — condición inicial de TRAJECTORY
  ■ Cubo aplanado  — setpoint de preposicionamiento
  Círculos teóricos como LINE_STRIP

TÓPICOS:
  /replay/trajectory       MarkerArray — trayectoria acumulada animada
  /replay/current_pos      MarkerArray — esfera posición actual
  /replay/static_markers   MarkerArray — marcadores fijos (una vez)
  /replay/phase_label      MarkerArray — texto fase + tiempo

FRAME: rtk_odom  (mismo que MavrosVisualizer)

USO:
  python3 rviz_replay_v2.py --master master_telemetry.csv --speed 1.0

  Parámetros opcionales (mismos que MavrosVisualizer):
    --origin_lat  --origin_lon  --origin_alt
    --calib_ang
    --speed   (factor de velocidad, default 1.0)
"""

import sys
import math
import time
import argparse
import threading

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

# =============================================================================
#  CONFIGURACIÓN  ← EDITAR
# =============================================================================

CSV_FILE       = "master_telemetry.csv"
PLAYBACK_SPEED = 1.0      # 1.0=tiempo real | 2.0=doble velocidad
PUBLISH_RATE   = 20       # Hz

# ── Origen RTK (igual que MavrosVisualizer) ───────────────────────────────────
ORIGIN_LAT = 19.5942341
ORIGIN_LON = -99.2280871
ORIGIN_ALT = 2329.0

# ── Calibración angular ───────────────────────────────────────────────────────
# Ángulo de rotación CCW en grados (calib_mode='angle' de MavrosVisualizer)
# Ajustar para que el norte apunte correctamente en RViz
CALIB_ANG_DEG = 180.0

# ── Fases de vuelo (segundos) — AJUSTAR con los tiempos reales ───────────────
PHASE_TIMES = {
    "TAKEOFF":     (0.0,    15.0),
    "POSITIONING": (15.0,   45.0),
    "TRAJECTORY":  (45.0,  145.0),
    "LANDING":     (145.0,  None),   # None = hasta el final
}

# ── Parámetros del experimento ────────────────────────────────────────────────
RADIUS    = 4.0    # radio teórico círculo líder [m]
OFFSET_D  = 2.0    # separación teórica líder-seguidor [m]
OFFSET_DZ = 1.0    # diferencia de altitud objetivo [m]

# ── Setpoints de preposicionamiento en coordenadas RTK local ─────────────────
# Poner None si no aplica
SP_PREPOS = {
    "L": {"x": -4.0, "y": 0.0, "z": 4.0},
    "S": {"x": -6.0, "y": 0.0, "z": 4.0},
}

# =============================================================================
# FIN DE CONFIGURACIÓN
# =============================================================================

# WGS84
WGS84_A  = 6_378_137.0
WGS84_E2 = 6.69437999014e-3

# Colores dron (RGBA 0-1)
COLOR_LEADER   = (0.15, 0.40, 0.85, 1.0)
COLOR_FOLLOWER = (0.85, 0.15, 0.15, 1.0)

# Grosor [m] y alpha por fase
PHASE_VIZ = {
    "TAKEOFF":     (0.030, 0.35),
    "POSITIONING": (0.050, 0.65),
    "TRAJECTORY":  (0.100, 1.00),
    "LANDING":     (0.030, 0.45),
    "UNKNOWN":     (0.015, 0.20),
}

PHASE_LABEL = {
    "TAKEOFF":     "DESPEGUE",
    "POSITIONING": "POSICIONAMIENTO",
    "TRAJECTORY":  "TRAYECTORIA / CONTROL",
    "LANDING":     "ATERRIZAJE",
    "UNKNOWN":     "",
}

SPHERE_CURRENT = 0.25
SPHERE_START   = 0.35
SPHERE_CI      = 0.22
CUBE_SP        = 0.20


# =============================================================================
# PIPELINE DE COORDENADAS (igual que MavrosVisualizer)
# =============================================================================

def geodetic_to_ecef(lat_r, lon_r, alt):
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat_r)**2)
    x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
    y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
    z = (N * (1.0 - WGS84_E2) + alt) * math.sin(lat_r)
    return x, y, z


def get_rotation_matrix(lat_r, lon_r):
    return np.array([
        [-math.sin(lon_r),
          math.cos(lon_r),
          0.0],
        [-math.sin(lat_r) * math.cos(lon_r),
         -math.sin(lat_r) * math.sin(lon_r),
          math.cos(lat_r)],
        [ math.cos(lat_r) * math.cos(lon_r),
          math.cos(lat_r) * math.sin(lon_r),
          math.sin(lat_r)],
    ])


def gps_to_rtk(lat_deg, lon_deg, alt,
               X0, Y0, Z0, R_enu, theta):
    """
    Convierte GPS (grados, metros) al frame RTK local.
    Pipeline idéntico a cb_navsat() en MavrosVisualizer.
    Devuelve (xr, yr, zr).
    """
    lat_r = math.radians(lat_deg)
    lon_r = math.radians(lon_deg)
    Xe, Ye, Ze = geodetic_to_ecef(lat_r, lon_r, alt)
    d   = np.array([Xe - X0, Ye - Y0, Ze - Z0])
    enu = R_enu.dot(d)
    xr  = enu[0] * math.cos(theta) - enu[1] * math.sin(theta)
    yr  = enu[0] * math.sin(theta) + enu[1] * math.cos(theta)
    zr  = float(enu[2])
    return xr, yr, zr


# =============================================================================
# UTILIDADES DE DATOS
# =============================================================================

def assign_phases(t_arr, phase_times):
    phases = np.full(len(t_arr), "UNKNOWN", dtype=object)
    t_max  = float(t_arr[-1])
    for name, interval in phase_times.items():
        if interval is None:
            continue
        t0, t1 = interval
        t1 = t_max if t1 is None else float(t1)
        phases[(t_arr >= t0) & (t_arr <= t1)] = name
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


def load_data(path, X0, Y0, Z0, R_enu, theta):
    """
    Carga el CSV y convierte GPS→RTK local usando el mismo pipeline
    que MavrosVisualizer. Altitud: usa L_alt/S_alt (GPS barométrico)
    descontando la altitud de suelo para obtener altura relativa.
    """
    print(f"[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas · {len(df.columns)} columnas")

    t_arr  = df["t"].values
    phases = assign_phases(t_arr, PHASE_TIMES)
    df["phase"] = phases

    # ── Convertir GPS → RTK local por dron ───────────────────────────────────
    for pfx in ("L", "S"):
        has = df[f"{pfx}_lat"].notna() & df[f"{pfx}_lon"].notna()

        # Altitud: preferir _alt (GPS absoluto), caer a -_z (NED)
        alt_col = f"{pfx}_alt"
        z_col   = f"{pfx}_z"
        if alt_col in df.columns and df[alt_col].notna().any():
            alt_vals = df[alt_col].values.astype(float)
        elif z_col in df.columns and df[z_col].notna().any():
            alt_vals = -df[z_col].values.astype(float) + ORIGIN_ALT
        else:
            alt_vals = np.full(len(df), ORIGIN_ALT)

        xr = np.full(len(df), np.nan)
        yr = np.full(len(df), np.nan)
        zr = np.full(len(df), np.nan)

        for i in np.where(has.values)[0]:
            lat = float(df.at[i, f"{pfx}_lat"])
            lon = float(df.at[i, f"{pfx}_lon"])
            alt = float(alt_vals[i])
            if math.isfinite(lat) and math.isfinite(lon) and math.isfinite(alt):
                xr[i], yr[i], zr[i] = gps_to_rtk(lat, lon, alt,
                                                   X0, Y0, Z0, R_enu, theta)

        # Altitud relativa al suelo (restar mínimo inicial)
        valid_z = zr[np.isfinite(zr)]
        if len(valid_z):
            z_ground = np.min(valid_z[:max(1, min(20, len(valid_z)))])
            zr = zr - z_ground

        df[f"{pfx}_rx"] = xr   # norte en RTK local
        df[f"{pfx}_ry"] = yr   # este  en RTK local
        df[f"{pfx}_rz"] = zr   # altura sobre suelo

    # Resumen de fases
    print("[Fases]")
    for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING", "UNKNOWN"]:
        segs = phase_segments(phases, ph)
        if not segs:
            continue
        total = sum(t_arr[min(i1-1, len(t_arr)-1)] - t_arr[i0] for i0, i1 in segs)
        t0s = t_arr[segs[0][0]]
        t1s = t_arr[min(segs[-1][1]-1, len(t_arr)-1)]
        print(f"  {ph:<14s}: {total:6.1f} s  (t={t0s:.1f}–{t1s:.1f} s)")

    return df


# =============================================================================
# UTILIDADES DE MARKERS
# =============================================================================

FRAME_ID = "rtk_odom"   # mismo que MavrosVisualizer


def _stamp(node, m):
    m.header.stamp = node.get_clock().now().to_msg()
    return m


def _lifetime_inf():
    d = Duration(); d.sec = 0; d.nanosec = 0
    return d


def _rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _pt(xr, yr, zr):
    """
    Convierte coordenadas RTK local a Point de RViz.
    En rtk_odom: x=este (yr), y=norte (xr), z=arriba (zr)
    """
    p = Point()
    p.x = float(yr)    # este
    p.y = float(xr)    # norte
    p.z = float(zr)
    return p


def make_line_list(mid, ns, color_rgba, scale):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid
    m.type   = Marker.LINE_LIST
    m.action = Marker.ADD
    m.scale.x = float(scale)
    m.color   = _rgba(*color_rgba)
    m.lifetime = _lifetime_inf()
    m.pose.orientation.w = 1.0
    return m


def make_sphere(mid, ns, color_rgba, scale, xr, yr, zr):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid
    m.type   = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    m.color   = _rgba(*color_rgba)
    m.lifetime = _lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position = _pt(xr, yr, zr)
    return m


def make_cube(mid, ns, color_rgba, sx, sy, sz, xr, yr, zr):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid
    m.type   = Marker.CUBE
    m.action = Marker.ADD
    m.scale.x, m.scale.y, m.scale.z = float(sx), float(sy), float(sz)
    m.color   = _rgba(*color_rgba)
    m.lifetime = _lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position = _pt(xr, yr, zr)
    return m


def make_text(mid, ns, text, xr, yr, zr, scale=0.45):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid
    m.type   = Marker.TEXT_VIEW_FACING
    m.action = Marker.ADD
    m.text   = text
    m.scale.z = float(scale)
    m.color   = _rgba(1.0, 1.0, 1.0, 0.95)
    m.lifetime = _lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position = _pt(xr, yr, zr)
    return m


def make_circle_strip(mid, ns, color_rgba, scale, radius, z_h, n=180):
    """Círculo teórico como LINE_STRIP."""
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns, m.id = ns, mid
    m.type   = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = float(scale)
    m.color   = _rgba(*color_rgba)
    m.lifetime = _lifetime_inf()
    m.pose.orientation.w = 1.0
    theta = np.linspace(0.0, 2.0 * math.pi, n)
    for th in theta:
        # radio en plano XY del frame RTK (norte=x, este=y)
        xr_c = radius * math.cos(th)
        yr_c = radius * math.sin(th)
        m.points.append(_pt(xr_c, yr_c, z_h))
    return m


# =============================================================================
# NODO ROS2
# =============================================================================

class ReplayNode(Node):

    def __init__(self, df, speed):
        super().__init__("rviz_replay")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub_traj    = self.create_publisher(MarkerArray, "/replay/trajectory",      qos)
        self._pub_current = self.create_publisher(MarkerArray, "/replay/current_pos",     qos)
        self._pub_static  = self.create_publisher(MarkerArray, "/replay/static_markers",  qos)
        self._pub_label   = self.create_publisher(MarkerArray, "/replay/phase_label",     qos)

        self._df     = df
        self._speed  = speed
        self._t_arr  = df["t"].values
        self._phases = df["phase"].values

        # Pre-construir segmentos para el loop animado
        self._segs = self._build_segments()

        # Publicar estáticos una vez al arrancar
        self._publish_static()

        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

        dur = self._t_arr[-1] - self._t_arr[0]
        self.get_logger().info(
            f"Replay iniciado | speed={speed}x | duración={dur:.1f} s | "
            f"frame={FRAME_ID}"
        )
        self.get_logger().info(
            "Tópicos:\n"
            "  /replay/trajectory      — trayectoria acumulada\n"
            "  /replay/current_pos     — posición actual animada\n"
            "  /replay/static_markers  — inicio ★  CI ●  SP ■\n"
            "  /replay/phase_label     — fase y tiempo"
        )

    # ── Pre-compómputo de segmentos ───────────────────────────────────────────

    def _build_segments(self):
        """
        Lista de (i_start, i_end, pfx, phase) con límites en cambios de fase.
        """
        df = self._df; phases = self._phases; n = len(df)
        segs = []
        for pfx in ("L", "S"):
            rx = df[f"{pfx}_rx"].values
            ry = df[f"{pfx}_ry"].values
            rz = df[f"{pfx}_rz"].values
            i = 0
            while i < n - 1:
                ph = phases[i]; j = i + 1
                while (j < n and phases[j] == ph
                       and np.isfinite(rx[j]) and np.isfinite(ry[j])):
                    j += 1
                valid = np.isfinite(rx[i:j]) & np.isfinite(ry[i:j])
                if valid.sum() >= 2:
                    segs.append((i, j, pfx, ph))
                i = j
        return segs

    # ── Marcadores estáticos ──────────────────────────────────────────────────

    def _publish_static(self):
        df     = self._df
        phases = self._phases
        arr    = MarkerArray()
        mid    = 0

        def add(m):
            m.header.stamp = self.get_clock().now().to_msg()
            arr.markers.append(m)
            return m

        # Altura media de la fase TRAJECTORY (para los círculos teóricos)
        traj_mask = phases == "TRAJECTORY"
        z_traj_L = float(np.nanmean(df.loc[traj_mask, "L_rz"])) if traj_mask.any() else 4.0
        z_traj_S = float(np.nanmean(df.loc[traj_mask, "S_rz"])) if traj_mask.any() else 5.0

        # Círculos teóricos
        add(make_circle_strip(mid, "theory_leader",
                              (0.70, 0.70, 0.70, 0.40), 0.025, RADIUS, z_traj_L))
        mid += 1
        add(make_circle_strip(mid, "theory_follower",
                              (0.90, 0.55, 0.55, 0.38), 0.025, RADIUS + OFFSET_D, z_traj_S))
        mid += 1

        COLOR = {"L": COLOR_LEADER, "S": COLOR_FOLLOWER}
        LABEL = {"L": "Líder", "S": "Seguidor"}

        for pfx in ("L", "S"):
            rx = df[f"{pfx}_rx"].values
            ry = df[f"{pfx}_ry"].values
            rz = df[f"{pfx}_rz"].values
            color = COLOR[pfx]
            label = LABEL[pfx]

            # Primer punto válido → inicio ★
            valid_idx = np.where(np.isfinite(rx) & np.isfinite(ry))[0]
            if len(valid_idx):
                i0 = valid_idx[0]
                z0 = rz[i0] if np.isfinite(rz[i0]) else 0.0
                add(make_sphere(mid, f"start_{pfx}", color, SPHERE_START,
                                rx[i0], ry[i0], z0))
                mid += 1
                add(make_text(mid, f"start_lbl_{pfx}",
                              f"★ Inicio {label}",
                              rx[i0], ry[i0], z0 + 0.6))
                mid += 1

            # Condición inicial TRAJECTORY → CI ●
            traj_idx = np.where(phases == "TRAJECTORY")[0]
            if len(traj_idx):
                i_ci = traj_idx[0]
                if np.isfinite(rx[i_ci]) and np.isfinite(ry[i_ci]):
                    z_ci = rz[i_ci] if np.isfinite(rz[i_ci]) else 0.0
                    add(make_sphere(mid, f"ci_{pfx}",
                                   (*color[:3], 0.85), SPHERE_CI,
                                   rx[i_ci], ry[i_ci], z_ci))
                    mid += 1
                    add(make_text(mid, f"ci_lbl_{pfx}",
                                  f"● CI {label}",
                                  rx[i_ci], ry[i_ci], z_ci + 0.5, 0.32))
                    mid += 1

            # SP de preposicionamiento → cubo ■
            sp = SP_PREPOS.get(pfx)
            if sp is not None:
                add(make_cube(mid, f"sp_{pfx}",
                              (*color[:3], 0.70),
                              CUBE_SP, CUBE_SP, CUBE_SP * 0.3,
                              sp["x"], sp["y"], sp["z"]))
                mid += 1
                add(make_text(mid, f"sp_lbl_{pfx}",
                              f"■ SP {label}",
                              sp["x"], sp["y"], sp["z"] + 0.45, 0.30))
                mid += 1

        self._pub_static.publish(arr)
        self.get_logger().info(f"Marcadores estáticos: {mid} publicados")

    # ── Loop animado ──────────────────────────────────────────────────────────

    def _replay_loop(self):
        df      = self._df
        t_arr   = self._t_arr
        phases  = self._phases
        speed   = self._speed
        dt_pub  = 1.0 / PUBLISH_RATE

        t_start_data = float(t_arr[0])
        t_end_data   = float(t_arr[-1])
        t_wall_start = time.monotonic()

        COLOR = {"L": COLOR_LEADER, "S": COLOR_FOLLOWER}

        while not self._stop.is_set():
            t_wall = time.monotonic()
            t_data = t_start_data + (t_wall - t_wall_start) * speed
            t_data = min(t_data, t_end_data)

            i_now = int(np.searchsorted(t_arr, t_data, side="right")) - 1
            i_now = max(0, min(i_now, len(t_arr) - 1))

            stamp = self.get_clock().now().to_msg()

            # ── Trayectoria acumulada ─────────────────────────────────────────
            traj_arr = MarkerArray()
            mid = 0

            for pfx in ("L", "S"):
                color_base = COLOR[pfx]
                rx = df[f"{pfx}_rx"].values
                ry = df[f"{pfx}_ry"].values
                rz = df[f"{pfx}_rz"].values

                for (i0, i1, seg_pfx, ph) in self._segs:
                    if seg_pfx != pfx or i0 > i_now:
                        continue
                    i1_clip = min(i1, i_now + 1)
                    scale, alpha = PHASE_VIZ.get(ph, (0.015, 0.20))

                    m = make_line_list(mid, f"traj_{pfx}_{ph}",
                                      (*color_base[:3], alpha), scale)
                    m.header.stamp = stamp

                    for k in range(i0, i1_clip - 1):
                        if (np.isfinite(rx[k])   and np.isfinite(ry[k])
                                and np.isfinite(rz[k])
                                and np.isfinite(rx[k+1]) and np.isfinite(ry[k+1])
                                and np.isfinite(rz[k+1])):
                            m.points.append(_pt(rx[k],   ry[k],   rz[k]))
                            m.points.append(_pt(rx[k+1], ry[k+1], rz[k+1]))

                    if m.points:
                        traj_arr.markers.append(m)
                    mid += 1

            self._pub_traj.publish(traj_arr)

            # ── Posición actual (esferas animadas) ────────────────────────────
            curr_arr = MarkerArray()
            cid = 0
            for pfx in ("L", "S"):
                rx = df[f"{pfx}_rx"].values
                ry = df[f"{pfx}_ry"].values
                rz = df[f"{pfx}_rz"].values
                color = COLOR[pfx]

                x_c = rx[i_now] if np.isfinite(rx[i_now]) else 0.0
                y_c = ry[i_now] if np.isfinite(ry[i_now]) else 0.0
                z_c = rz[i_now] if np.isfinite(rz[i_now]) else 0.0

                m_c = make_sphere(cid, f"current_{pfx}",
                                  (*color[:3], 0.95), SPHERE_CURRENT,
                                  x_c, y_c, z_c)
                m_c.header.stamp = stamp
                curr_arr.markers.append(m_c)
                cid += 1

            self._pub_current.publish(curr_arr)

            # ── Etiqueta fase + tiempo ────────────────────────────────────────
            ph_now  = phases[i_now]
            lbl_str = PHASE_LABEL.get(ph_now, ph_now)
            t_rel   = t_data - t_start_data

            rx_L = df["L_rx"].values; ry_L = df["L_ry"].values
            rz_L = df["L_rz"].values
            x_lbl = rx_L[i_now] if np.isfinite(rx_L[i_now]) else 0.0
            y_lbl = ry_L[i_now] if np.isfinite(ry_L[i_now]) else 0.0
            z_lbl = (rz_L[i_now] if np.isfinite(rz_L[i_now]) else 0.0) + 1.3

            lbl_arr = MarkerArray()
            m_lbl = make_text(0, "phase_label",
                              f"{lbl_str}\nt = {t_rel:.1f} s",
                              x_lbl, y_lbl, z_lbl, scale=0.45)
            m_lbl.header.stamp = stamp
            lbl_arr.markers.append(m_lbl)
            self._pub_label.publish(lbl_arr)

            # ── Sleep exacto ──────────────────────────────────────────────────
            sleep_t = dt_pub - (time.monotonic() - t_wall)
            if sleep_t > 0:
                time.sleep(sleep_t)

            if t_data >= t_end_data:
                self.get_logger().info("Reproducción completada — trayectoria estática activa.")
                break

    def stop(self):
        self._stop.set()


# =============================================================================
# MAIN
# =============================================================================

def main():
    global ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT, CALIB_ANG_DEG, PLAYBACK_SPEED

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master",     default=CSV_FILE,
                    help=f"CSV de telemetría (default: {CSV_FILE})")
    ap.add_argument("--speed",      type=float, default=PLAYBACK_SPEED,
                    help="Factor de velocidad (default: %(default)s)")
    ap.add_argument("--origin_lat", type=float, default=ORIGIN_LAT,
                    help="Latitud origen RTK (default: %(default)s)")
    ap.add_argument("--origin_lon", type=float, default=ORIGIN_LON,
                    help="Longitud origen RTK (default: %(default)s)")
    ap.add_argument("--origin_alt", type=float, default=ORIGIN_ALT,
                    help="Altitud origen RTK m (default: %(default)s)")
    ap.add_argument("--calib_ang",  type=float, default=CALIB_ANG_DEG,
                    help="Ángulo calibración CCW deg (default: %(default)s)")
    args = ap.parse_args()

    ORIGIN_LAT    = args.origin_lat
    ORIGIN_LON    = args.origin_lon
    ORIGIN_ALT    = args.origin_alt
    CALIB_ANG_DEG = args.calib_ang
    PLAYBACK_SPEED = args.speed

    # ── Pre-cálculo del origen (igual que MultiDroneVisualizer.__init__) ──────
    lat0_r = math.radians(ORIGIN_LAT)
    lon0_r = math.radians(ORIGIN_LON)
    X0, Y0, Z0 = geodetic_to_ecef(lat0_r, lon0_r, ORIGIN_ALT)
    R_enu  = get_rotation_matrix(lat0_r, lon0_r)
    theta  = math.radians(CALIB_ANG_DEG)

    print(f"\n[Config] Origen RTK: lat={ORIGIN_LAT:.7f}  lon={ORIGIN_LON:.7f}  "
          f"alt={ORIGIN_ALT:.1f} m")
    print(f"[Config] calib_ang={CALIB_ANG_DEG:.1f}°  speed={PLAYBACK_SPEED}x")

    try:
        df = load_data(args.master, X0, Y0, Z0, R_enu, theta)
    except FileNotFoundError:
        print(f"\n[ERROR] No se encontró '{args.master}'")
        print("        Usa --master /ruta/al/archivo.csv")
        sys.exit(1)
    except KeyError as e:
        print(f"\n[ERROR] Columna no encontrada en el CSV: {e}")
        sys.exit(1)

    rclpy.init()
    node = ReplayNode(df, args.speed)

    print("\n══════════════════════════════════════════════════════")
    print("  En RViz2 (Fixed Frame: rtk_odom):")
    print("    Add → By topic → /replay/trajectory      → MarkerArray")
    print("    Add → By topic → /replay/current_pos     → MarkerArray")
    print("    Add → By topic → /replay/static_markers  → MarkerArray")
    print("    Add → By topic → /replay/phase_label     → MarkerArray")
    print("══════════════════════════════════════════════════════\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[INFO] Detenido por usuario.")
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
rviz_replay.py
==============
Reproduce en RViz2 la trayectoria líder-seguidor desde master_telemetry.csv.

REPRODUCCIÓN ANIMADA:
  - Avanza a velocidad real (o con factor de escala PLAYBACK_SPEED).
  - Publica la trayectoria acumulada hasta el instante t actual.
  - Una esfera "posición actual" se mueve en tiempo real para cada dron.
  - Al terminar, la trayectoria completa queda estática en RViz.

DIFERENCIACIÓN VISUAL:
  Drones   → color fijo  (líder: azul, seguidor: rojo)
  Fases    → grosor + alpha de la línea:
               TAKEOFF      delgado  0.03 m  alpha 0.35
               POSITIONING  medio    0.05 m  alpha 0.60
               TRAJECTORY   grueso   0.10 m  alpha 1.00
               LANDING      delgado  0.03 m  alpha 0.40
  Marcadores especiales (esferas):
               ★ Inicio físico de cada dron
               ● Condición inicial de TRAJECTORY
               ◇ Setpoint de preposicionamiento

TÓPICOS PUBLICADOS (todos en /replay/...):
  /replay/leader/trajectory        LINE_LIST por segmentos (acumulado)
  /replay/follower/trajectory      LINE_LIST por segmentos (acumulado)
  /replay/leader/current_pos       SPHERE — posición actual animada
  /replay/follower/current_pos     SPHERE — posición actual animada
  /replay/leader/markers           SPHERE_LIST — inicio, CI, SP
  /replay/follower/markers         SPHERE_LIST — inicio, CI, SP
  /replay/phase_label              TEXT_VIEW_FACING — fase activa actual

FRAME: map  (asegúrate de tener fixed frame = map en RViz2)

USO:
  # Terminal 1
  ros2 run rviz2 rviz2

  # Terminal 2
  python3 rviz_replay.py
  python3 rviz_replay.py --master mi_log.csv --speed 2.0

DEPENDENCIAS:
  pip install pandas numpy
  ROS2 Humble + rclpy + visualization_msgs + geometry_msgs
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
PLAYBACK_SPEED = 1.0      # 1.0 = tiempo real, 2.0 = doble velocidad, 0.5 = mitad
FRAME_ID       = "map"
PUBLISH_RATE   = 20       # Hz — frecuencia de publicación del loop animado

# Parámetros del experimento (para referencias teóricas)
RADIUS    = 4.0
OFFSET_D  = 2.0
OFFSET_DZ = 1.0

# Tiempos de fase (segundos) ← AJUSTAR con los tiempos reales del vuelo
PHASE_TIMES = {
    "TAKEOFF":     (0.0,    15.0),
    "POSITIONING": (15.0,   45.0),
    "TRAJECTORY":  (45.0,  145.0),
    "LANDING":     (145.0,  None),   # None = hasta el final
}

# Setpoints de preposicionamiento en ENU local (antes de recentrar)
SP_PREPOS = {
    "L": {"x": -4.0, "y": 0.0, "z": 4.0},
    "S": {"x": -6.0, "y": 0.0, "z": 4.0},
}

# Correcciones de recentrado (mismas que reconstruir_trayectoria_v3.py)
CENTER_OFFSET_X          =  0.0
CENTER_OFFSET_Y          =  0.0
CENTER_OFFSET_Z_LEADER   =  0.0
CENTER_OFFSET_Z_FOLLOWER =  0.0

# Fracción a recortar para el ajuste del círculo
CIRCLE_TRIM_START = 0.10
CIRCLE_TRIM_END   = 0.05

# =============================================================================
# FIN DE CONFIGURACIÓN
# =============================================================================

R_EARTH = 6_371_000.0

# ── Colores de dron ───────────────────────────────────────────────────────────
COLOR_LEADER   = (0.15, 0.40, 0.85, 1.0)   # RGBA azul
COLOR_FOLLOWER = (0.85, 0.15, 0.15, 1.0)   # RGBA rojo
COLOR_THEORY   = (0.75, 0.75, 0.75, 0.50)  # gris semitransparente

# ── Grosor y alpha por fase ────────────────────────────────────────────────────
#   { "FASE": (scale_m, alpha) }
PHASE_VIZ = {
    "TAKEOFF":     (0.030, 0.35),
    "POSITIONING": (0.050, 0.65),
    "TRAJECTORY":  (0.100, 1.00),
    "LANDING":     (0.030, 0.45),
    "UNKNOWN":     (0.020, 0.20),
}

# ── Tamaños de marcadores esféricos ──────────────────────────────────────────
SPHERE_CURRENT  = 0.25   # m — posición actual (animada)
SPHERE_START    = 0.30   # m — inicio físico ★
SPHERE_CI       = 0.22   # m — condición inicial ●
SPHERE_SP       = 0.20   # m — setpoint preposicionamiento ◇ (usamos esfera achatada)


# =============================================================================
# UTILIDADES DE DATOS
# =============================================================================

def gps_to_enu(lat, lon, ref_lat, ref_lon):
    ref_lat_r = math.radians(ref_lat)
    x = math.radians(lat - ref_lat) * R_EARTH
    y = math.radians(lon - ref_lon) * R_EARTH * math.cos(ref_lat_r)
    return x, y   # x=norte, y=este


def fit_circle(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return 0.0, 0.0, RADIUS
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = res[0], res[1]
    r = math.sqrt(max(res[2] + cx**2 + cy**2, 0.0))
    return cx, cy, r


def assign_phases(t_arr, phase_times):
    phases = np.full(len(t_arr), "UNKNOWN", dtype=object)
    t_max  = t_arr[-1]
    for name, interval in phase_times.items():
        if interval is None:
            continue
        t0, t1 = interval
        t1 = t_max if t1 is None else t1
        phases[(t_arr >= t0) & (t_arr <= t1)] = name
    return phases


def load_data(path):
    print(f"[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas")

    # Origen ENU
    mask = df["L_lat"].notna() & df["L_lon"].notna()
    first = df[mask].iloc[0]
    ref_lat, ref_lon = float(first["L_lat"]), float(first["L_lon"])

    # ENU por dron
    for pfx in ("L", "S"):
        has = df[f"{pfx}_lat"].notna() & df[f"{pfx}_lon"].notna()
        ex  = np.full(len(df), np.nan)
        ey  = np.full(len(df), np.nan)
        if has.any():
            vals = df.loc[has, [f"{pfx}_lat", f"{pfx}_lon"]].apply(
                lambda r: gps_to_enu(r[f"{pfx}_lat"], r[f"{pfx}_lon"],
                                     ref_lat, ref_lon), axis=1)
            ex[has.values] = [v[0] for v in vals]
            ey[has.values] = [v[1] for v in vals]
        df[f"{pfx}_ex"] = ex
        df[f"{pfx}_ey"] = ey

    # Altitud
    for pfx in ("L", "S"):
        off = CENTER_OFFSET_Z_LEADER if pfx == "L" else CENTER_OFFSET_Z_FOLLOWER
        alt_col = f"{pfx}_alt"
        z_col   = f"{pfx}_z"
        if alt_col in df.columns and df[alt_col].notna().any():
            alt_v  = df[alt_col].values.astype(float)
            ground = np.nanmin(alt_v[:min(20, len(alt_v))])
            df[f"{pfx}_cz"] = alt_v - ground + off
        elif z_col in df.columns:
            df[f"{pfx}_cz"] = -df[z_col].values.astype(float) + off
        else:
            df[f"{pfx}_cz"] = np.nan

    # Fases
    t_arr  = df["t"].values
    phases = assign_phases(t_arr, PHASE_TIMES)
    df["phase"] = phases

    # Ajuste de círculo (solo TRAJECTORY)
    traj_mask = phases == "TRAJECTORY"
    if traj_mask.sum() > 10:
        idx = np.where(traj_mask)[0]
        n_t = len(idx)
        i0  = idx[int(n_t * CIRCLE_TRIM_START)]
        i1  = idx[max(0, int(n_t * (1 - CIRCLE_TRIM_END)) - 1)]
        fit_m = np.zeros(len(df), dtype=bool)
        fit_m[i0:i1+1] = True
        fit_m &= traj_mask
        cx_fit, cy_fit, r_fit = fit_circle(
            df.loc[fit_m, "L_ex"].values,
            df.loc[fit_m, "L_ey"].values)
        print(f"[Círculo] Centro=({cx_fit:.3f}, {cy_fit:.3f})  R={r_fit:.3f} m")
    else:
        cx_fit, cy_fit, r_fit = 0.0, 0.0, RADIUS

    ox = cx_fit + CENTER_OFFSET_X
    oy = cy_fit + CENTER_OFFSET_Y
    df["L_rx"] = df["L_ex"] - ox
    df["L_ry"] = df["L_ey"] - oy
    df["S_rx"] = df["S_ex"] - ox
    df["S_ry"] = df["S_ey"] - oy

    # Ajustar SP_PREPOS con el mismo offset
    sp_centered = {}
    for k, v in SP_PREPOS.items():
        sp_centered[k] = {
            "x": v["x"] - ox,   # norte
            "y": v["y"] - oy,   # este
            "z": v["z"],
        }

    meta = {
        "r_fit": r_fit, "ox": ox, "oy": oy,
        "ref_lat": ref_lat, "ref_lon": ref_lon,
        "sp_centered": sp_centered,
    }
    return df, meta


# =============================================================================
# UTILIDADES DE MARKERS
# =============================================================================

def rgba(r, g, b, a=1.0):
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def pt(x, y, z):
    """Crea Point con convención RViz: x=este, y=norte, z=arriba."""
    p = Point()
    p.x, p.y, p.z = float(y), float(x), float(z)   # RViz ENU: x=este, y=norte
    return p


def lifetime_inf():
    d = Duration()
    d.sec = 0; d.nanosec = 0   # 0 = permanente
    return d


def make_line_list_marker(marker_id, ns, color_rgba, scale):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns              = ns
    m.id              = marker_id
    m.type            = Marker.LINE_LIST
    m.action          = Marker.ADD
    m.scale.x         = float(scale)
    m.color           = rgba(*color_rgba)
    m.lifetime        = lifetime_inf()
    m.pose.orientation.w = 1.0
    return m


def make_sphere_marker(marker_id, ns, color_rgba, scale, x, y, z):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns              = ns
    m.id              = marker_id
    m.type            = Marker.SPHERE
    m.action          = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = float(scale)
    m.color           = rgba(*color_rgba)
    m.lifetime        = lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position   = pt(x, y, z)
    return m


def make_text_marker(marker_id, ns, text, x, y, z, scale=0.4):
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns              = ns
    m.id              = marker_id
    m.type            = Marker.TEXT_VIEW_FACING
    m.action          = Marker.ADD
    m.text            = text
    m.scale.z         = float(scale)
    m.color           = rgba(1.0, 1.0, 1.0, 0.95)
    m.lifetime        = lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position   = pt(x, y, z)
    return m


def make_cylinder_marker(marker_id, ns, color_rgba, radius, height, x, y, z):
    """Círculo teórico como cilindro muy aplanado."""
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns              = ns
    m.id              = marker_id
    m.type            = Marker.CYLINDER
    m.action          = Marker.ADD
    m.scale.x = m.scale.y = float(radius * 2)
    m.scale.z = float(height)
    m.color           = rgba(*color_rgba)
    m.lifetime        = lifetime_inf()
    m.pose.orientation.w = 1.0
    m.pose.position   = pt(x, y, z)
    return m


def theory_circle_points(radius, n=120):
    """Devuelve lista de puntos ENU para un círculo centrado en el origen."""
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    return list(zip(np.zeros(n), radius * np.sin(theta), radius * np.cos(theta)))
    # x=norte=0 → círculo en plano YZ de RViz... mejor:
    # En ENU: norte=X, este=Y → círculo en plano XY
    # rx = R*cos(theta), ry = R*sin(theta)


def make_circle_line_strip(marker_id, ns, color_rgba, scale, radius, z_height, n=200):
    """Círculo teórico como LINE_LIST (pares de puntos consecutivos)."""
    m = Marker()
    m.header.frame_id = FRAME_ID
    m.ns   = ns
    m.id   = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = float(scale)
    m.color   = rgba(*color_rgba)
    m.lifetime = lifetime_inf()
    m.pose.orientation.w = 1.0
    theta = np.linspace(0, 2*np.pi, n)
    for th in theta:
        rx = radius * math.cos(th)   # norte
        ry = radius * math.sin(th)   # este
        m.points.append(pt(rx, ry, z_height))
    return m


# =============================================================================
# NODO ROS2
# =============================================================================

class ReplayNode(Node):

    def __init__(self, df, meta, speed):
        super().__init__("rviz_replay")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_traj    = self.create_publisher(MarkerArray,
                            "/replay/trajectory",    qos)
        self._pub_current = self.create_publisher(MarkerArray,
                            "/replay/current_pos",   qos)
        self._pub_static  = self.create_publisher(MarkerArray,
                            "/replay/static_markers", qos)
        self._pub_label   = self.create_publisher(MarkerArray,
                            "/replay/phase_label",   qos)

        self._df    = df
        self._meta  = meta
        self._speed = speed

        self._t_arr  = df["t"].values
        self._phases = df["phase"].values

        # Pre-computar segmentos de trayectoria como pares de puntos LINE_LIST
        # Un segmento = dos puntos consecutivos con mismo estilo
        self._segments = self._build_segments()

        # Publicar marcadores estáticos una vez
        self._publish_static()

        # Iniciar hilo de reproducción
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"Reproducción iniciada | speed={speed}x | "
            f"duración={self._t_arr[-1]-self._t_arr[0]:.1f} s"
        )

    # ── Pre-cómputo de segmentos ──────────────────────────────────────────────

    def _build_segments(self):
        """
        Construye lista de segmentos para LINE_LIST.
        Cada segmento: (i_start, i_end, pfx, phase)
        Con límites en cambios de fase.
        """
        df     = self._df
        phases = self._phases
        segs   = []
        n      = len(df)

        for pfx in ("L", "S"):
            rx = df[f"{pfx}_rx"].values
            ry = df[f"{pfx}_ry"].values
            cz = df[f"{pfx}_cz"].values

            i = 0
            while i < n - 1:
                ph = phases[i]
                j  = i + 1
                # Avanzar mientras misma fase y datos válidos
                while (j < n and phases[j] == ph
                       and np.isfinite(rx[j]) and np.isfinite(ry[j])
                       and np.isfinite(cz[j])):
                    j += 1
                # Solo guardar si hay al menos 2 puntos válidos
                valid = (np.isfinite(rx[i:j]) & np.isfinite(ry[i:j])
                         & np.isfinite(cz[i:j]))
                if valid.sum() >= 2:
                    segs.append((i, j, pfx, ph))
                i = j

        return segs

    # ── Marcadores estáticos (inicio, CI, SP, círculos teóricos) ─────────────

    def _publish_static(self):
        df   = self._df
        meta = self._meta
        arr  = MarkerArray()
        mid  = 0

        def stamp(m):
            m.header.stamp = self.get_clock().now().to_msg()
            return m

        # Círculo teórico líder
        r_fit = meta["r_fit"]
        z_traj = np.nanmean(df.loc[df["phase"]=="TRAJECTORY", "L_cz"].values) \
                 if (df["phase"]=="TRAJECTORY").any() else 4.0
        arr.markers.append(stamp(make_circle_line_strip(
            mid, "theory", (0.75, 0.75, 0.75, 0.45), 0.03, r_fit, z_traj)))
        mid += 1
        # Círculo teórico seguidor
        arr.markers.append(stamp(make_circle_line_strip(
            mid, "theory", (0.90, 0.50, 0.50, 0.40), 0.03, r_fit + OFFSET_D, z_traj + OFFSET_DZ)))
        mid += 1

        # Inicio físico ★ y CI ● por dron
        COLOR = {"L": COLOR_LEADER, "S": COLOR_FOLLOWER}
        for pfx in ("L", "S"):
            rx = df[f"{pfx}_rx"].values
            ry = df[f"{pfx}_ry"].values
            cz = df[f"{pfx}_cz"].values
            color = COLOR[pfx]

            # Primer punto válido = inicio físico
            valid_idx = np.where(np.isfinite(rx) & np.isfinite(ry))[0]
            if len(valid_idx):
                i0 = valid_idx[0]
                arr.markers.append(stamp(make_sphere_marker(
                    mid, f"start_{pfx}", color, SPHERE_START,
                    rx[i0], ry[i0], cz[i0] if np.isfinite(cz[i0]) else 0.0)))
                mid += 1
                label = "★ Líder" if pfx == "L" else "★ Seguidor"
                arr.markers.append(stamp(make_text_marker(
                    mid, f"start_label_{pfx}", label,
                    rx[i0], ry[i0], (cz[i0] if np.isfinite(cz[i0]) else 0.0) + 0.5)))
                mid += 1

            # Condición inicial de TRAJECTORY
            traj_idx = np.where(self._phases == "TRAJECTORY")[0]
            if len(traj_idx):
                i_ci = traj_idx[0]
                if np.isfinite(rx[i_ci]) and np.isfinite(ry[i_ci]):
                    arr.markers.append(stamp(make_sphere_marker(
                        mid, f"ci_{pfx}", (*color[:3], 0.85), SPHERE_CI,
                        rx[i_ci], ry[i_ci],
                        cz[i_ci] if np.isfinite(cz[i_ci]) else 0.0)))
                    mid += 1
                    label = "● CI Líder" if pfx == "L" else "● CI Seguidor"
                    arr.markers.append(stamp(make_text_marker(
                        mid, f"ci_label_{pfx}", label,
                        rx[i_ci], ry[i_ci],
                        (cz[i_ci] if np.isfinite(cz[i_ci]) else 0.0) + 0.5, 0.30)))
                    mid += 1

            # SP de preposicionamiento ◇
            sp = meta["sp_centered"].get(pfx)
            if sp:
                # Cubo aplanado para simular ◇
                m_sp = Marker()
                m_sp.header.frame_id = FRAME_ID
                m_sp.header.stamp    = self.get_clock().now().to_msg()
                m_sp.ns     = f"sp_{pfx}"
                m_sp.id     = mid
                m_sp.type   = Marker.CUBE
                m_sp.action = Marker.ADD
                m_sp.scale.x = m_sp.scale.y = SPHERE_SP
                m_sp.scale.z = SPHERE_SP * 0.3
                m_sp.color  = rgba(*color[:3], 0.70)
                m_sp.lifetime = lifetime_inf()
                m_sp.pose.orientation.w = 1.0
                m_sp.pose.position = pt(sp["x"], sp["y"], sp["z"])
                arr.markers.append(m_sp)
                mid += 1
                label = "◇ SP Líder" if pfx == "L" else "◇ SP Seguidor"
                arr.markers.append(stamp(make_text_marker(
                    mid, f"sp_label_{pfx}", label,
                    sp["x"], sp["y"], sp["z"] + 0.4, 0.28)))
                mid += 1

        self._pub_static.publish(arr)
        self.get_logger().info(f"Marcadores estáticos publicados ({mid} markers)")

    # ── Loop de reproducción ──────────────────────────────────────────────────

    def _replay_loop(self):
        df      = self._df
        t_arr   = self._t_arr
        phases  = self._phases
        speed   = self._speed
        dt_pub  = 1.0 / PUBLISH_RATE

        t_data_start = t_arr[0]
        t_data_end   = t_arr[-1]
        t_wall_start = time.monotonic()

        COLOR = {"L": COLOR_LEADER, "S": COLOR_FOLLOWER}
        LABEL_LONG = {
            "TAKEOFF":     "DESPEGUE",
            "POSITIONING": "POSICIONAMIENTO",
            "TRAJECTORY":  "TRAYECTORIA",
            "LANDING":     "ATERRIZAJE",
            "UNKNOWN":     "",
        }

        last_traj_idx = {"L": 0, "S": 0}  # índice hasta donde se publicó trayectoria

        while not self._stop.is_set():
            t_wall_now  = time.monotonic()
            t_elapsed   = (t_wall_now - t_wall_start) * speed
            t_current   = t_data_start + t_elapsed

            if t_current > t_data_end:
                t_current = t_data_end   # congelar al final

            # Índice actual en el CSV
            i_now = int(np.searchsorted(t_arr, t_current, side="right")) - 1
            i_now = max(0, min(i_now, len(t_arr) - 1))

            stamp = self.get_clock().now().to_msg()

            # ── Trayectoria acumulada ─────────────────────────────────────────
            traj_arr = MarkerArray()
            mid      = 0

            for pfx in ("L", "S"):
                color_base = COLOR[pfx]
                rx = df[f"{pfx}_rx"].values
                ry = df[f"{pfx}_ry"].values
                cz = df[f"{pfx}_cz"].values

                for (i0, i1, seg_pfx, ph) in self._segments:
                    if seg_pfx != pfx:
                        continue
                    if i0 > i_now:
                        break   # aún no llega el tiempo a este segmento

                    i1_clip = min(i1, i_now + 1)
                    scale, alpha = PHASE_VIZ.get(ph, (0.02, 0.20))
                    color = (*color_base[:3], alpha)

                    m = make_line_list_marker(mid, f"traj_{pfx}_{ph}", color, scale)
                    m.header.stamp = stamp

                    # LINE_LIST: pares de puntos (p0,p1), (p1,p2) ...
                    for k in range(i0, i1_clip - 1):
                        if (np.isfinite(rx[k]) and np.isfinite(ry[k])
                                and np.isfinite(cz[k])
                                and np.isfinite(rx[k+1]) and np.isfinite(ry[k+1])
                                and np.isfinite(cz[k+1])):
                            m.points.append(pt(rx[k],   ry[k],   cz[k]))
                            m.points.append(pt(rx[k+1], ry[k+1], cz[k+1]))
                    if m.points:
                        traj_arr.markers.append(m)
                    mid += 1

            self._pub_traj.publish(traj_arr)

            # ── Posición actual (esferas animadas) ───────────────────────────
            curr_arr = MarkerArray()
            cid = 0
            for pfx in ("L", "S"):
                rx = df[f"{pfx}_rx"].values
                ry = df[f"{pfx}_ry"].values
                cz = df[f"{pfx}_cz"].values
                color_base = COLOR[pfx]

                x_now = rx[i_now] if np.isfinite(rx[i_now]) else 0.0
                y_now = ry[i_now] if np.isfinite(ry[i_now]) else 0.0
                z_now = cz[i_now] if np.isfinite(cz[i_now]) else 0.0

                m_cur = make_sphere_marker(
                    cid, f"current_{pfx}",
                    (*color_base[:3], 0.95),
                    SPHERE_CURRENT, x_now, y_now, z_now)
                m_cur.header.stamp = stamp
                curr_arr.markers.append(m_cur)
                cid += 1

            self._pub_current.publish(curr_arr)

            # ── Etiqueta de fase actual ───────────────────────────────────────
            ph_now    = phases[i_now]
            label_str = LABEL_LONG.get(ph_now, ph_now)
            t_rel     = t_current - t_data_start

            # Posición del texto: sobre el líder actual
            rx_L = df["L_rx"].values
            ry_L = df["L_ry"].values
            cz_L = df["L_cz"].values
            x_lbl = rx_L[i_now] if np.isfinite(rx_L[i_now]) else 0.0
            y_lbl = ry_L[i_now] if np.isfinite(ry_L[i_now]) else 0.0
            z_lbl = (cz_L[i_now] if np.isfinite(cz_L[i_now]) else 0.0) + 1.2

            label_arr = MarkerArray()
            m_lbl = make_text_marker(
                0, "phase_label",
                f"{label_str}\nt = {t_rel:.1f} s",
                x_lbl, y_lbl, z_lbl, scale=0.45)
            m_lbl.header.stamp = stamp
            label_arr.markers.append(m_lbl)
            self._pub_label.publish(label_arr)

            # ── Dormir hasta siguiente publicación ────────────────────────────
            elapsed_loop = time.monotonic() - t_wall_now
            sleep_t = dt_pub - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

            if t_current >= t_data_end:
                self.get_logger().info("Reproducción completada. Trayectoria estática activa.")
                break

    def stop(self):
        self._stop.set()


# =============================================================================
# MAIN
# =============================================================================

def main():
    global FRAME_ID, PLAYBACK_SPEED

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=CSV_FILE,
                    help=f"CSV de telemetría (default: {CSV_FILE})")
    ap.add_argument("--speed",  type=float, default=PLAYBACK_SPEED,
                    help=f"Factor de velocidad (default: {PLAYBACK_SPEED})")
    ap.add_argument("--frame",  default=FRAME_ID,
                    help=f"Frame de RViz2 (default: {FRAME_ID})")
    args = ap.parse_args()

    FRAME_ID       = args.frame
    PLAYBACK_SPEED = args.speed

    try:
        df, meta = load_data(args.master)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró '{args.master}'")
        sys.exit(1)

    rclpy.init()
    node = ReplayNode(df, meta, args.speed)

    print("\n══════════════════════════════════════════════════")
    print("  RViz2 Replay — tópicos publicados:")
    print("  /replay/trajectory      — trayectoria acumulada")
    print("  /replay/current_pos     — posición actual (animada)")
    print("  /replay/static_markers  — inicio ★, CI ●, SP ◇")
    print("  /replay/phase_label     — fase y tiempo actual")
    print()
    print("  En RViz2 agrega:")
    print("    Add → By topic → /replay/trajectory      → MarkerArray")
    print("    Add → By topic → /replay/current_pos     → MarkerArray")
    print("    Add → By topic → /replay/static_markers  → MarkerArray")
    print("    Add → By topic → /replay/phase_label     → MarkerArray")
    print("  Fixed Frame: map")
    print("══════════════════════════════════════════════════\n")

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
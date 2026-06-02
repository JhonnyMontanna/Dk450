#!/usr/bin/env python3
"""
reconstruir_trayectoria_v3.py
==============================
Reconstruye trayectoria ENU líder-seguidor desde master_telemetry.csv.

CONFIGURACIÓN DE FASES (editar esta sección):
    Definir los tiempos de inicio y fin de cada fase del vuelo.
    Todo lo que esté fuera de los rangos definidos queda como UNKNOWN.

COLUMNAS ESPERADAS DEL CSV:
    t, L_lat, L_lon, L_alt, L_x, L_y, L_z, L_gvx, L_gvy, L_gvz
    S_lat, S_lon, S_alt, S_x, S_y, S_z, S_gvx, S_gvy, S_gvz
    dist_xy, dist_3d, dz   (opcionales, se recalculan si faltan)

USO:
    python reconstruir_trayectoria_v3.py
    python reconstruir_trayectoria_v3.py --master mi_log.csv
"""

import sys
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


# -----------------------------------------------------------------------------
# ARCHIVO DE ENTRADA
# -----------------------------------------------------------------------------
CSV_FILE = "master_telemetry.csv"

# -----------------------------------------------------------------------------
# FASES DE VUELO  ← EDITAR ESTOS TIEMPOS (en segundos)
#
# Formato de cada fase:
#   "NOMBRE": (t_inicio, t_fin)
#
# Reglas:
#   · Los rangos NO tienen que ser contiguos (los huecos quedan como UNKNOWN)
#   · Pueden solaparse (la última que aparece en el dict gana)
#   · Si no usas alguna fase, ponla como None:  "LANDING": None
#   · t_fin = None significa "hasta el final del vuelo"
# -----------------------------------------------------------------------------
PHASE_TIMES = {
    "TAKEOFF":     (0.0,    35.0),   # ← ajustar
    "POSITIONING": (35.0,   75.0),   # ← ajustar
    "TRAJECTORY":  (80.0,  125.0),   # ← ajustar
    "LANDING":     (150.0,  None),   # ← ajustar  (None = hasta el final)
}

# -----------------------------------------------------------------------------
# PARÁMETROS DEL EXPERIMENTO
# -----------------------------------------------------------------------------
R_EARTH  = 6_371_000.0
RADIUS   = 4.0     # radio teórico del círculo del líder [m]
OFFSET_D = 2.0     # separación teórica líder-seguidor [m]
OFFSET_DZ = 1.0    # diferencia de altitud objetivo [m]

# Setpoints de preposicionamiento (en coordenadas locales NED, frame del dron)
# Se usan para marcar el marcador ◇ en la vista XY.
# Poner None si no se usa preposicionamiento.
SP_PREPOS = {
    "L": {"x": -4.0, "y": 0.0, "z": 4.0},   # líder   ← ajustar o poner None
    "S": {"x": -6.0, "y": 0.0, "z": 4.0},   # seguidor ← ajustar o poner None
}

# -----------------------------------------------------------------------------
# RECENTRADO
# Offset manual adicional sobre el centro ajustado automáticamente por
# el ajuste de círculo por mínimos cuadrados.
# Útil para alinear la figura con el origen deseado.
# -----------------------------------------------------------------------------
CENTER_OFFSET_X          =  0.0   # [m] — positivo = mover trayectoria al norte
CENTER_OFFSET_Y          =  0.0   # [m] — positivo = mover trayectoria al este
CENTER_OFFSET_Z_LEADER   =  0.0   # [m] — corrección altitud líder
CENTER_OFFSET_Z_FOLLOWER =  0.0   # [m] — corrección altitud seguidor

# Fracción de la fase TRAJECTORY a recortar al inicio/fin para el ajuste
# del círculo (elimina transitorios). 0.0 = sin recorte.
CIRCLE_TRIM_START = 0.10
CIRCLE_TRIM_END   = 0.05

# =============================================================================
# FIN DE CONFIGURACIÓN — no es necesario editar debajo de esta línea
# =============================================================================

# -----------------------------------------------------------------------------
# Colores de drones
# -----------------------------------------------------------------------------
C_L = "#1565C0"   # azul  — líder
C_S = "#C62828"   # rojo  — seguidor

# Estilo de línea por fase: (linestyle, linewidth, alpha, label_largo)
PHASE_STYLE = {
    "TAKEOFF":     (":",              1.3, 0.60, "Despegue"),
    "POSITIONING": ("--",             1.6, 0.80, "Posicionamiento"),
    "TRAJECTORY":  ("-",              2.2, 0.95, "Trayectoria / Control"),
    "LANDING":     ((0, (4, 1, 1, 1)),1.3, 0.60, "Aterrizaje"),
    "UNKNOWN":     ("-",              0.7, 0.25, "Sin clasificar"),
}

# Color de banda de fondo en gráficas t vs variable
PHASE_BG = {
    "TAKEOFF":     "#E3F2FD",
    "POSITIONING": "#FFF8E1",
    "TRAJECTORY":  "#E8F5E9",
    "LANDING":     "#FCE4EC",
    "UNKNOWN":     "#F5F5F5",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def gps_to_enu(lat, lon, ref_lat, ref_lon):
    """Convierte GPS a ENU local (x=norte, y=este) en metros."""
    ref_lat_r = math.radians(ref_lat)
    x = math.radians(lat - ref_lat) * R_EARTH
    y = math.radians(lon - ref_lon) * R_EARTH * math.cos(ref_lat_r)
    return x, y


def fit_circle(x, y):
    """Ajuste de círculo por mínimos cuadrados. Devuelve cx, cy, r, rms."""
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
    """
    Asigna una etiqueta de fase a cada muestra de tiempo según PHASE_TIMES.
    Los huecos entre fases quedan como 'UNKNOWN'.
    """
    phases = np.full(len(t_arr), "UNKNOWN", dtype=object)
    t_max  = t_arr[-1]
    for name, interval in phase_times.items():
        if interval is None:
            continue
        t0, t1 = interval
        t1 = t_max if t1 is None else t1
        mask = (t_arr >= t0) & (t_arr <= t1)
        phases[mask] = name
    return phases


def phase_segments(phases, label):
    """Devuelve lista de (i_start, i_end) donde phases == label."""
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


# =============================================================================
# CARGA Y TRANSFORMACIÓN
# =============================================================================

def load_and_transform(path):
    print(f"\n[IO] Cargando {path} …")
    df = pd.read_csv(path, sep=None, engine="python")   # acepta coma o tabulador
    df = df.sort_values("t").reset_index(drop=True)
    print(f"     {len(df)} filas · {len(df.columns)} columnas")
    print(f"     Columnas: {list(df.columns)[:12]} …")

    # ── Origen ENU: primer GPS válido del líder ───────────────────────────────
    mask_gps = df["L_lat"].notna() & df["L_lon"].notna()
    if not mask_gps.any():
        raise ValueError("No hay GPS del líder (L_lat / L_lon) en el CSV.")
    first    = df[mask_gps].iloc[0]
    ref_lat  = float(first["L_lat"])
    ref_lon  = float(first["L_lon"])
    print(f"[ENU] Origen → lat={ref_lat:.8f}  lon={ref_lon:.8f}")

    # ── Convertir GPS → ENU para ambos drones ────────────────────────────────
    for pfx in ("L", "S"):
        clat, clon = f"{pfx}_lat", f"{pfx}_lon"
        has = df[clat].notna() & df[clon].notna()
        ex  = np.full(len(df), np.nan)
        ey  = np.full(len(df), np.nan)
        if has.any():
            vals = df.loc[has, [clat, clon]].apply(
                lambda r: gps_to_enu(r[clat], r[clon], ref_lat, ref_lon), axis=1)
            ex[has.values] = [v[0] for v in vals]
            ey[has.values] = [v[1] for v in vals]
        df[f"{pfx}_ex"] = ex   # norte [m]
        df[f"{pfx}_ey"] = ey   # este  [m]

    # ── Altitud: usar L_alt/S_alt (GPS barométrico) si existe,
    #    si no usar -L_z (NED local, positivo arriba) ──────────────────────────
    for pfx in ("L", "S"):
        alt_col  = f"{pfx}_alt"
        z_col    = f"{pfx}_z"
        cz_col   = f"{pfx}_cz"
        off = CENTER_OFFSET_Z_LEADER if pfx == "L" else CENTER_OFFSET_Z_FOLLOWER
        if alt_col in df.columns and df[alt_col].notna().any():
            # Restar altitud de despegue para tener altura sobre suelo
            alt_vals   = df[alt_col].values.astype(float)
            alt_ground = np.nanmin(alt_vals[:min(20, len(alt_vals))])
            df[cz_col] = alt_vals - alt_ground + off
        elif z_col in df.columns and df[z_col].notna().any():
            df[cz_col] = -df[z_col].values.astype(float) + off
        else:
            df[cz_col] = np.nan

    # ── Asignar fases ─────────────────────────────────────────────────────────
    t_arr = df["t"].values
    phases = assign_phases(t_arr, PHASE_TIMES)
    df["phase"] = phases
    print("[Fases] Distribución:")
    for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING", "UNKNOWN"]:
        segs  = phase_segments(phases, ph)
        if not segs:
            continue
        total = sum(t_arr[min(i1-1, len(t_arr)-1)] - t_arr[i0] for i0, i1 in segs)
        t0s   = t_arr[segs[0][0]]
        t1s   = t_arr[min(segs[-1][1]-1, len(t_arr)-1)]
        print(f"  {ph:<14s}: {total:6.1f} s  (t={t0s:.1f} – {t1s:.1f} s)")

    # ── Ajuste de círculo sobre la fase TRAJECTORY ────────────────────────────
    traj_mask = phases == "TRAJECTORY"
    if traj_mask.sum() > 10:
        idx_traj = np.where(traj_mask)[0]
        n_t  = len(idx_traj)
        i0_t = idx_traj[int(n_t * CIRCLE_TRIM_START)]
        i1_t = idx_traj[max(0, int(n_t * (1 - CIRCLE_TRIM_END)) - 1)]
        fit_mask = np.zeros(len(df), dtype=bool)
        fit_mask[i0_t:i1_t+1] = True
        fit_mask &= traj_mask
        x_fit = df.loc[fit_mask, "L_ex"].values
        y_fit = df.loc[fit_mask, "L_ey"].values
        cx_fit, cy_fit, r_fit, rms_fit = fit_circle(x_fit, y_fit)
        print(f"[Círculo] Centro ENU bruto: ({cx_fit:.3f}, {cy_fit:.3f})"
              f"  R={r_fit:.3f} m  RMS={rms_fit:.4f} m")
    else:
        cx_fit, cy_fit, r_fit, rms_fit = 0.0, 0.0, RADIUS, np.nan
        print("[Círculo] Sin datos de TRAJECTORY suficientes para ajuste.")

    # ── Recentrar: mover el centro del círculo al origen ─────────────────────
    ox = cx_fit + CENTER_OFFSET_X
    oy = cy_fit + CENTER_OFFSET_Y
    df["L_rx"] = df["L_ex"] - ox   # posición recentrada norte
    df["L_ry"] = df["L_ey"] - oy   # posición recentrada este
    df["S_rx"] = df["S_ex"] - ox
    df["S_ry"] = df["S_ey"] - oy

    # ── Separación y dz (recalcular con datos recentrados) ────────────────────
    df["sep_xy"] = np.sqrt((df["L_rx"] - df["S_rx"])**2 +
                           (df["L_ry"] - df["S_ry"])**2)
    df["sep_3d"] = np.sqrt((df["L_rx"] - df["S_rx"])**2 +
                           (df["L_ry"] - df["S_ry"])**2 +
                           (df["L_cz"] - df["S_cz"])**2)
    df["dz_calc"] = df["L_cz"] - df["S_cz"]

    # Guardar metadata
    df.attrs.update({
        "ref_lat": ref_lat, "ref_lon": ref_lon,
        "ox": ox, "oy": oy,
        "cx_fit": cx_fit, "cy_fit": cy_fit,
        "r_fit":  r_fit,  "rms_fit": rms_fit,
    })
    return df


# =============================================================================
# HELPERS GRÁFICOS
# =============================================================================

def add_phase_bands(ax, t_arr, phases):
    """Añade bandas de color de fondo por fase."""
    for ph, color in PHASE_BG.items():
        for (i0, i1) in phase_segments(phases, ph):
            ax.axvspan(t_arr[i0], t_arr[min(i1-1, len(t_arr)-1)],
                       alpha=0.18, color=color, linewidth=0, zorder=0)


def phase_line_legend():
    """Devuelve handles de leyenda para los estilos de línea de fase."""
    items = []
    for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING"]:
        ls, lw, alpha, label = PHASE_STYLE[ph]
        items.append(mlines.Line2D([], [], color="#555", ls=ls, lw=lw,
                                   alpha=max(alpha, 0.6), label=label))
    return items


# =============================================================================
# FIGURA 1 — PLANO XY CON FASES
# =============================================================================

def plot_xy(df):
    phases = df["phase"].values
    t_arr  = df["t"].values
    r_fit  = df.attrs["r_fit"]

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.canvas.manager.set_window_title("Figura 1 — Plano XY por fases")
    ax.set_title("Trayectoria XY — líder y seguidor\n(marco ENU, centro del círculo en origen)",
                 fontsize=11, pad=12)

    # ── Referencias teóricas ──────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 500)
    ax.plot(r_fit * np.sin(theta), r_fit * np.cos(theta),
            color="#BDBDBD", lw=0.9, ls=":", zorder=1,
            label=f"Círculo teórico líder  (R={r_fit:.2f} m)")
    ax.plot((r_fit + OFFSET_D) * np.sin(theta), (r_fit + OFFSET_D) * np.cos(theta),
            color="#EF9A9A", lw=0.9, ls=":", zorder=1,
            label=f"Círculo teórico seguidor  (R={r_fit+OFFSET_D:.2f} m)")

    # ── Trayectorias segmentadas por fase ─────────────────────────────────────
    for pfx, color, drone_label in [("L", C_L, "Líder"), ("S", C_S, "Seguidor")]:
        rx = df[f"{pfx}_rx"].values
        ry = df[f"{pfx}_ry"].values

        for ph in ["UNKNOWN", "TAKEOFF", "POSITIONING", "LANDING", "TRAJECTORY"]:
            ls, lw, alpha, _ = PHASE_STYLE[ph]
            zorder = 4 if ph == "TRAJECTORY" else 2
            for (i0, i1) in phase_segments(phases, ph):
                seg_x = rx[i0:i1]
                seg_y = ry[i0:i1]
                valid  = np.isfinite(seg_x) & np.isfinite(seg_y)
                if valid.sum() < 2:
                    continue
                # Eje Y = este, eje X = norte  →  plot(este, norte)
                ax.plot(seg_y[valid], seg_x[valid],
                        color=color, ls=ls, lw=lw, alpha=alpha,
                        zorder=zorder, solid_capstyle="round", dash_capstyle="round")

    # ── Marcadores de inicio (★) ──────────────────────────────────────────────
    for pfx, color, drone_label in [("L", C_L, "Líder"), ("S", C_S, "Seguidor")]:
        rx = df[f"{pfx}_rx"].values
        ry = df[f"{pfx}_ry"].values
        valid_idx = np.where(np.isfinite(rx) & np.isfinite(ry))[0]
        if len(valid_idx) == 0:
            continue
        i0 = valid_idx[0]
        ax.scatter(ry[i0], rx[i0], marker="*", s=300, color=color,
                   edgecolors="white", linewidths=0.8, zorder=10,
                   label=f"★ Inicio {drone_label}  ({rx[i0]:.1f}, {ry[i0]:.1f})")
        ax.annotate(f"  Inicio\n  {drone_label}",
                    xy=(ry[i0], rx[i0]), fontsize=8, color=color,
                    va="center", xytext=(ry[i0]+0.3, rx[i0]+0.2))

    # ── Condición inicial de TRAJECTORY (●) ──────────────────────────────────
    for pfx, color, drone_label in [("L", C_L, "Líder"), ("S", C_S, "Seguidor")]:
        rx = df[f"{pfx}_rx"].values
        ry = df[f"{pfx}_ry"].values
        traj_idx = np.where(df["phase"].values == "TRAJECTORY")[0]
        if len(traj_idx) == 0:
            continue
        i_ci = traj_idx[0]
        if not (np.isfinite(rx[i_ci]) and np.isfinite(ry[i_ci])):
            continue
        ax.scatter(ry[i_ci], rx[i_ci], marker="o", s=130, color=color,
                   edgecolors="white", linewidths=1.5, zorder=9,
                   label=f"● Cond. inicial trayectoria {drone_label}")
        ax.annotate(f"  CI {drone_label}",
                    xy=(ry[i_ci], rx[i_ci]), fontsize=8, color=color,
                    va="top", xytext=(ry[i_ci]+0.2, rx[i_ci]-0.4))

    # ── Setpoints de preposicionamiento (◇) ──────────────────────────────────
    # DESACTIVADO: Los SP no están bien representados por ahora
    # ox = df.attrs["ox"]
    # oy = df.attrs["oy"]
    # for pfx, color, drone_label in [("L", C_L, "Líder"), ("S", C_S, "Seguidor")]:
    #     sp = SP_PREPOS.get(pfx)
    #     if sp is None:
    #         continue
    #     # SP está en NED local: x=norte, y=este
    #     # Recentrar igual que las trayectorias
    #     sp_rx = sp["x"] - ox
    #     sp_ry = sp["y"] - oy
    #     ax.scatter(sp_ry, sp_rx, marker="D", s=80, color=color,
    #                facecolors="none", linewidths=1.8, zorder=8,
    #                label=f"◇ SP preposicionamiento {drone_label}")
    #     ax.annotate(f"  SP {drone_label}",
    #                 xy=(sp_ry, sp_rx), fontsize=7.5, color=color,
    #                 alpha=0.85, xytext=(sp_ry+0.2, sp_rx+0.2))









    # ── Origen recentrado ─────────────────────────────────────────────────────
    ax.scatter(0, 0, marker="+", s=150, color="#616161", linewidths=1.8,
               zorder=6, label="+ Centro círculo (origen)")

    # ── Ejes y estética ───────────────────────────────────────────────────────
    ax.set_xlabel("Este [m]",  fontsize=11)
    ax.set_ylabel("Norte [m]", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.28, linestyle=":")
    ax.axhline(0, color="#BDBDBD", lw=0.5)
    ax.axvline(0, color="#BDBDBD", lw=0.5)

    # ── Leyendas separadas ────────────────────────────────────────────────────
    leg1 = ax.legend(handles=phase_line_legend(),
                     title="Estilo de línea por fase", title_fontsize=8.5,
                     loc="upper left", fontsize=8.5,
                     framealpha=0.93, edgecolor="#CCCCCC")
    ax.add_artist(leg1)

    handles2, labels2 = ax.get_legend_handles_labels()
    ax.legend(handles=handles2, labels=labels2,
              title="Drones y referencias", title_fontsize=8.5,
              loc="upper right", fontsize=8.5,
              framealpha=0.93, edgecolor="#CCCCCC", ncol=1)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURA 2 — X, Y, Z vs TIEMPO
# =============================================================================

def plot_xyz_tiempo(df):
    phases = df["phase"].values
    t_arr  = df["t"].values

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.canvas.manager.set_window_title("Figura 2 — X, Y, Z vs Tiempo")
    fig.suptitle("Posición X (norte), Y (este), Z (altitud) vs tiempo", fontsize=11)

    config = [
        ("Norte X [m]",   "L_rx",  "S_rx"),
        ("Este  Y [m]",   "L_ry",  "S_ry"),
        ("Altitud Z [m]", "L_cz",  "S_cz"),
    ]

    for ax, (ylabel, l_col, s_col) in zip(axes, config):
        add_phase_bands(ax, t_arr, phases)
        for col, color, label in [(l_col, C_L, "Líder"), (s_col, C_S, "Seguidor")]:
            if col in df.columns:
                seg = df[["t", col]].dropna()
                ax.plot(seg["t"], seg[col], color=color, lw=1.5, alpha=0.9, label=label)
        ax.axhline(0, color="k", lw=0.4, ls=":", alpha=0.35)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Etiquetas de fase en el eje superior
    ax_top  = axes[0]
    ylim    = ax_top.get_ylim()
    y_label = ylim[1] - (ylim[1] - ylim[0]) * 0.02
    seen    = set()
    for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING"]:
        _, _, _, label_long = PHASE_STYLE[ph]
        for (i0, i1) in phase_segments(phases, ph):
            t_mid = (t_arr[i0] + t_arr[min(i1-1, len(t_arr)-1)]) / 2
            if ph not in seen:
                ax_top.text(t_mid, y_label, label_long[:4].upper(),
                            ha="center", va="top", fontsize=7,
                            color="#666", style="italic")
                seen.add(ph)

    axes[-1].set_xlabel("Tiempo (s)")

    # Parches de fase en la leyenda del primer eje
    patches = [mpatches.Patch(color=PHASE_BG[ph], alpha=0.55,
                              label=PHASE_STYLE[ph][3])
               for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING"]]
    h0, l0 = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=h0 + patches, labels=l0 + [PHASE_STYLE[ph][3]
                   for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING"]],
                   fontsize=7.5, loc="upper right", ncol=2)

    plt.tight_layout()
    return fig


# =============================================================================
# FIGURA 3 — ΔZ Y SEPARACIÓN vs TIEMPO
# =============================================================================

def plot_dz_separacion(df):
    phases = df["phase"].values
    t_arr  = df["t"].values

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.canvas.manager.set_window_title("Figura 3 — ΔZ y Separación vs Tiempo")
    fig.suptitle("Diferencia de altitud (ΔZ) y separación líder-seguidor", fontsize=11)

    ax1 = axes[0]
    add_phase_bands(ax1, t_arr, phases)
    dz_col = "dz_calc" if "dz_calc" in df.columns else "dz"
    seg = df[["t", dz_col]].dropna()
    ax1.plot(seg["t"], seg[dz_col], lw=1.8, color="#F57C00", label="ΔZ real")
    ax1.axhline(OFFSET_DZ, color="#888", lw=1.0, ls="--", alpha=0.7,
                label=f"Objetivo ΔZ = {OFFSET_DZ} m")
    ax1.axhline(0, color="k", lw=0.4, ls=":", alpha=0.35)
    ax1.set_ylabel("ΔZ = Z_líder − Z_seguidor (m)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    ax2 = axes[1]
    add_phase_bands(ax2, t_arr, phases)
    seg2 = df[["t", "sep_xy", "sep_3d"]].dropna()
    ax2.plot(seg2["t"], seg2["sep_xy"], lw=1.8, color="#388E3C",
             label="Separación horizontal XY")
    ax2.plot(seg2["t"], seg2["sep_3d"], lw=1.5, color="#66BB6A",
             ls="--", alpha=0.8, label="Separación 3D")
    ax2.axhline(OFFSET_D, color="#888", lw=1.0, ls="--", alpha=0.7,
                label=f"Objetivo d = {OFFSET_D} m")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Separación (m)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig


# =============================================================================
# RESUMEN CONSOLA
# =============================================================================

def print_summary(df):
    t_arr  = df["t"].values
    phases = df["phase"].values

    print("\n" + "="*65)
    print("  RESUMEN DEL VUELO")
    print("="*65)
    print(f"  Duración total   : {t_arr[-1] - t_arr[0]:.1f} s")
    print(f"  Origen ENU       : lat={df.attrs['ref_lat']:.8f}  "
          f"lon={df.attrs['ref_lon']:.8f}")
    print(f"  Centro círculo   : ENU=({df.attrs['cx_fit']:.3f}, {df.attrs['cy_fit']:.3f}) m")
    print(f"  Radio ajustado   : {df.attrs['r_fit']:.3f} m  "
          f"(teórico {RADIUS:.1f} m)  RMS={df.attrs['rms_fit']:.4f} m")

    for pfx, label in [("L", "Líder"), ("S", "Seguidor")]:
        alt = df[f"{pfx}_cz"].dropna()
        if len(alt):
            print(f"  Alt. máx {label:<10s}: {alt.max():.2f} m")

    sep = df["sep_xy"].dropna()
    if len(sep):
        traj_mask = phases == "TRAJECTORY"
        sep_traj = df.loc[traj_mask, "sep_xy"].dropna()
        print(f"  Sep XY (total)   : media={sep.mean():.2f}  "
              f"max={sep.max():.2f}  min={sep.min():.2f} m")
        if len(sep_traj):
            print(f"  Sep XY (TRAJ)    : media={sep_traj.mean():.2f}  "
                  f"max={sep_traj.max():.2f}  min={sep_traj.min():.2f} m")

    print()
    print("  FASES:")
    for ph in ["TAKEOFF", "POSITIONING", "TRAJECTORY", "LANDING", "UNKNOWN"]:
        segs = phase_segments(phases, ph)
        if not segs:
            continue
        total = sum(t_arr[min(i1-1, len(t_arr)-1)] - t_arr[i0] for i0, i1 in segs)
        t0s   = t_arr[segs[0][0]]
        t1s   = t_arr[min(segs[-1][1]-1, len(t_arr)-1)]
        print(f"  {ph:<14s}: {total:6.1f} s  "
              f"(t = {t0s:.1f} – {t1s:.1f} s)")
    print("="*65 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=CSV_FILE,
                    help=f"CSV de telemetría (default: {CSV_FILE})")
    ap.add_argument("--no-dz",   action="store_true",
                    help="Omitir figura ΔZ y separación")
    ap.add_argument("--ox",  type=float, default=None,
                    help="Sobrescribir CENTER_OFFSET_X [m]")
    ap.add_argument("--oy",  type=float, default=None,
                    help="Sobrescribir CENTER_OFFSET_Y [m]")
    ap.add_argument("--ozl", type=float, default=None,
                    help="Sobrescribir CENTER_OFFSET_Z_LEADER [m]")
    ap.add_argument("--ozs", type=float, default=None,
                    help="Sobrescribir CENTER_OFFSET_Z_FOLLOWER [m]")
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
        print(f"\n[ERROR] No se encontró '{args.master}'")
        print("        Usa  --master /ruta/al/archivo.csv")
        sys.exit(1)

    print_summary(df)

    print("[Plot] Generando figuras…")
    plot_xy(df)
    print("  ✔ Figura 1 — Plano XY con fases")

    plot_xyz_tiempo(df)
    print("  ✔ Figura 2 — X, Y, Z vs tiempo")

    if not args.no_dz:
        plot_dz_separacion(df)
        print("  ✔ Figura 3 — ΔZ y separación")

    print("\nCierra las ventanas para terminar.\n")
    plt.show()
    print("✅ Listo.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
plot_lf_results.py — Visualización post-vuelo Líder-Seguidor
=============================================================
Replica exactamente las 7 figuras de MavlinkControlLFx4.py
a partir de un archivo CSV de log.

VERSIÓN MEJORADA:
    - Agrega cálculo de métricas integrales (IAE)
    - Compatible con formato de tabla ancho (sin columnas derivadas)
    - Calcula automáticamente dist_xy, err_xy, etc. si no existen

Uso:
    python plot_lf_results.py

Configuración:
    - Modificar la variable ARCHIVO_CSV con el nombre de tu archivo
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL USUARIO - ¡MODIFICA AQUÍ!
# ═══════════════════════════════════════════════════════════════════════════════

# ── Nombre del archivo CSV (MODIFICA ESTA LÍNEA CON TU ARCHIVO) ────────────────
ARCHIVO_CSV = "lf_circulo_real2.csv"  # <--- Cambia aquí el nombre de tu archivo

# ── Origen del sistema de coordenadas (centro del círculo o punto de referencia) ──
ORIGEN_X = 0   # Coordenada X del origen (metros)
ORIGEN_Y = 0   # Coordenada Y del origen (metros)

# ── Opciones de salida ─────────────────────────────────────────────────────────
SAVE_FIGURES = False    # True = guardar PNGs | False = mostrar en pantalla
OUTPUT_DIR = "."        # Directorio donde guardar las figuras

# ── Opciones adicionales ───────────────────────────────────────────────────────
SHOW_IAE_METRICS = True       # Mostrar métricas IAE en consola
EXPORT_METRICS_CSV = True     # Exportar métricas a archivo CSV

# ── Estilo de visualización ────────────────────────────────────────────────────
MISSION_PLANNER_STYLE = True  # True = X=Este, Y=Norte | False = X=Norte, Y=Este

# ═══════════════════════════════════════════════════════════════════════════════

# ── Paleta de colores ─────────────────────────────────────────────────────────
C_refL = np.array([0.40, 0.65, 1.00])   # azul claro  — setpoint líder
C_refS = np.array([1.00, 0.60, 0.40])   # naranja claro — setpoint seguidor
C_L    = np.array([0.10, 0.35, 0.75])   # azul oscuro — líder real
C_S    = np.array([0.80, 0.15, 0.10])   # rojo        — seguidor real
C_dist = np.array([0.25, 0.75, 0.45])   # verde       — distancia
C_form = np.array([0.50, 0.50, 0.50])   # gris        — vector formación
LW, LWS = 2.0, 1.2

SC_DRONE = 0.40
SC_FRAME = SC_DRONE * 2.0
RADIUS = 4.0  # Radio del círculo (para la trayectoria teórica)

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES PARA MÉTRICAS IAE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iae(t, error_signal):
    """Calcula IAE = ∫|e| dt"""
    if len(error_signal) == 0:
        return 0.0
    
    # Asegurar misma longitud
    if len(error_signal) > len(t):
        error_signal = error_signal[:len(t)]
    elif len(error_signal) < len(t):
        t = t[:len(error_signal)]
    
    abs_err = np.abs(error_signal)
    iae = integrate.trapezoid(abs_err, t)
    return iae


def compute_all_iae(t, df, variables):
    """Calcula IAE para múltiples variables"""
    metrics = {}
    
    for col_name, display_name, units in variables:
        if col_name in df.columns:
            error = df[col_name].astype(float).to_numpy()
            if len(error) > len(t):
                error = error[:len(t)]
            elif len(error) < len(t):
                t_adj = t[:len(error)]
                iae = compute_iae(t_adj, error)
            else:
                iae = compute_iae(t, error)
            
            metrics[display_name] = {
                'IAE': iae,
                'units': units,
                'col_name': col_name
            }
    
    return metrics


def print_iae_metrics(metrics, duration):
    """Imprime métricas IAE"""
    if not metrics:
        return
    
    print('\n' + '='*60)
    print('  MÉTRICAS IAE (Integral Absolute Error)')
    print('='*60)
    print(f'  Duración: {duration:.2f} s')
    print('─'*60)
    print(f'  {"Variable":<25} {"IAE":>12} {"Unidad":>10}')
    print(f'  {"":25} {"[·s]":>12} {"":10}')
    print('─'*60)
    
    for var_name, data in metrics.items():
        iae = data['IAE']
        units = data.get('units', '')
        print(f'  {var_name:<25} {iae:>12.4f}   {units}·s')
    
    print('─'*60)
    print('='*60 + '\n')


def export_iae_csv(metrics, base_filename):
    """Exporta IAE a CSV"""
    if not metrics:
        return
    
    metrics_fname = f'{base_filename}_IAE_metrics.csv'
    rows = [{'variable': var_name, 'units': data['units'], 'IAE': data['IAE']} 
            for var_name, data in metrics.items()]
    
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_fname, index=False)
    print(f"📊 IAE metrics saved: {metrics_fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE DIBUJO
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_np(arr):
    return np.arctan2(np.sin(arr), np.cos(arr))


def _draw_drone_frame(ax, x, y, yaw, scale=SC_DRONE):
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=4, headlength=3.5, headaxislength=3, alpha=0.85)
    
    if MISSION_PLANNER_STYLE:
        ax.quiver(x, y, scale * np.sin(yaw), scale * np.cos(yaw),
                  color=[0.85, 0.10, 0.10], **kw)
        ax.quiver(x, y, scale * np.sin(yaw + np.pi/2), scale * np.cos(yaw + np.pi/2),
                  color=[0.10, 0.65, 0.10], **kw)
    else:
        ax.quiver(x, y, scale * np.cos(yaw), scale * np.sin(yaw),
                  color=[0.85, 0.10, 0.10], **kw)
        ax.quiver(x, y, scale * np.cos(yaw + np.pi/2), scale * np.sin(yaw + np.pi/2),
                  color=[0.10, 0.65, 0.10], **kw)


def _draw_coordinate_frame(ax, x, y, scale=SC_FRAME):
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.007, headwidth=4, headlength=4, headaxislength=3.5, alpha=0.9)
    
    if MISSION_PLANNER_STYLE:
        ax.quiver(x, y, scale, 0, color=[0.10, 0.65, 0.10], **kw)
        ax.quiver(x, y, 0, scale, color=[0.85, 0.10, 0.10], **kw)
        ax.plot(x, y, 'ok', ms=6, zorder=5)
        ax.text(x + scale + 0.12, y - 0.10, 'x (Este)', fontsize=10, fontweight='bold', va='top')
        ax.text(x - 0.10, y + scale + 0.10, 'y (Norte)', fontsize=10, fontweight='bold', ha='right')
    else:
        ax.quiver(x, y, scale, 0, color=[0.85, 0.10, 0.10], **kw)
        ax.quiver(x, y, 0, scale, color=[0.10, 0.65, 0.10], **kw)
        ax.plot(x, y, 'ok', ms=6, zorder=5)
        ax.text(x + scale + 0.12, y - 0.10, 'Norte', fontsize=10, fontweight='bold', va='top')
        ax.text(x - 0.10, y + scale + 0.10, 'Este', fontsize=10, fontweight='bold', ha='right')


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results():
    csv_path = os.path.join(OUTPUT_DIR, ARCHIVO_CSV) if OUTPUT_DIR != "." else ARCHIVO_CSV
    
    if not os.path.isfile(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    print(f"\n📂 Loading: {ARCHIVO_CSV}")
    print(f"   Columns found: {list(df.columns)}")

    # Función segura para obtener columnas
    def get_col(col, default=None):
        if col in df.columns:
            return df[col].astype(float).to_numpy()
        return default

    # ── Tiempo ────────────────────────────────────────────────────────────────
    t = get_col('time')
    if t is None:
        t = get_col('t')
    if t is None:
        t = np.arange(len(df)) * 0.05
        print(f"  ⚠️  No 'time' column, using index * 0.05s")
    
    # ── Líder ─────────────────────────────────────────────────────────────────
    lx = get_col('lx')
    ly = get_col('ly')
    lz = get_col('lz')
    psiL = get_col('l_yaw')
    
    # Si no hay datos de líder, usar ceros
    if lx is None:
        lx = np.zeros(len(t))
        ly = np.zeros(len(t))
        lz = np.zeros(len(t))
        psiL = np.zeros(len(t))
        print("  ⚠️  No leader position data, using zeros")
    
    # ── Seguidor ─────────────────────────────────────────────────────────────
    sx = get_col('sx')
    sy = get_col('sy')
    sz = get_col('sz')
    psiS = get_col('s_yaw')
    
    if sx is None:
        sx = np.zeros(len(t))
        sy = np.zeros(len(t))
        sz = np.zeros(len(t))
        psiS = np.zeros(len(t))
        print("  ⚠️  No follower position data, using zeros")
    
    # ── Setpoints ────────────────────────────────────────────────────────────
    xd = get_col('xd')
    yd = get_col('yd')
    zd = get_col('zd')
    lxsp = get_col('lx_sp')
    lysp = get_col('ly_sp')
    
    if xd is None:
        xd = lx if lx is not None else np.zeros(len(t))
        yd = ly if ly is not None else np.zeros(len(t))
        zd = lz if lz is not None else np.zeros(len(t))
    
    if lxsp is None:
        lxsp = lx
        lysp = ly
    
    # ── Errores (calcular si no existen) ─────────────────────────────────────
    ex = get_col('ex')
    ey = get_col('ey')
    ez = get_col('ez')
    e_yaw = get_col('e_yaw')
    
    if ex is None:
        ex = xd - sx
        ey = yd - sy
        ez = zd - sz
        print("  ℹ️  Computing errors from setpoints")
    
    if e_yaw is None:
        e_yaw = _wrap_np(psiL - psiS) if len(psiL) == len(psiS) else np.zeros(len(t))
    
    # ── Distancias (calcular si no existen) ──────────────────────────────────
    dist_xy = get_col('dist_xy')
    if dist_xy is None:
        dist_xy = np.hypot(ex, ey) if ex is not None else np.zeros(len(t))
        print("  ℹ️  Computing dist_xy from errors")
    
    dist_z = get_col('dist_z')
    if dist_z is None:
        dist_z = np.abs(ez) if ez is not None else np.zeros(len(t))
    
    err_xy = np.hypot(ex, ey) if ex is not None else np.zeros(len(t))
    
    # ── PID components (opcional) ────────────────────────────────────────────
    ff_x = get_col('ff_x')
    ff_y = get_col('ff_y')
    vx_p = get_col('vx_p')
    vy_p = get_col('vy_p')
    vx_d = get_col('vx_d')
    vy_d = get_col('vy_d')
    vx_cmd = get_col('vx_cmd')
    vy_cmd = get_col('vy_cmd')
    
    # Valores por defecto si no existen
    if ff_x is None: ff_x = np.zeros(len(t))
    if ff_y is None: ff_y = np.zeros(len(t))
    if vx_p is None: vx_p = np.zeros(len(t))
    if vy_p is None: vy_p = np.zeros(len(t))
    if vx_d is None: vx_d = np.zeros(len(t))
    if vy_d is None: vy_d = np.zeros(len(t))
    if vx_cmd is None: vx_cmd = np.zeros(len(t))
    if vy_cmd is None: vy_cmd = np.zeros(len(t))
    
    # ── Altitud deseada ──────────────────────────────────────────────────────
    lz_sp = get_col('lz_sp')
    if lz_sp is None:
        lz_sp = np.median(lz) * np.ones_like(lz)
    
    # ── Aplicar desplazamiento del origen ────────────────────────────────────
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        lx = lx - ORIGEN_X
        ly = ly - ORIGEN_Y
        sx = sx - ORIGEN_X
        sy = sy - ORIGEN_Y
        lxsp = lxsp - ORIGEN_X
        lysp = lysp - ORIGEN_Y
        xd = xd - ORIGEN_X
        yd = yd - ORIGEN_Y
    
    # ── Valores deseados de formación ────────────────────────────────────────
    desired_dist_xy = np.median(dist_xy[100:200]) if len(dist_xy) > 200 else np.mean(dist_xy)
    desired_dist_z = np.median(dist_z[100:200]) if len(dist_z) > 200 else np.mean(dist_z)
    
    n = len(t)
    duration = t[-1] - t[0] if len(t) > 1 else 0
    
    # ── Errores del líder ────────────────────────────────────────────────────
    ex_L = lxsp - lx
    ey_L = lysp - ly
    ez_L = lz_sp - lz
    theta_sp = np.arctan2(lysp, lxsp)
    psiL_sp = _wrap_np(theta_sp + math.pi/2)
    epsi_L = _wrap_np(psiL_sp - psiL)
    
    # ── Métricas RMS ─────────────────────────────────────────────────────────
    def rms(v):
        return np.sqrt(np.mean(v**2)) if len(v) > 0 else 0.0
    
    print('\n' + '='*55)
    print(f'  FILE: {ARCHIVO_CSV}')
    print(f'  Samples: {n}   Duration: {duration:.1f} s')
    print('='*55)
    print('  FLIGHT METRICS (RMS)')
    print('─'*55)
    print('  FOLLOWER ERRORS (relative to setpoint):')
    print(f'    RMS error x  : {rms(ex):.4f} m')
    print(f'    RMS error y  : {rms(ey):.4f} m')
    print(f'    RMS error z  : {rms(ez):.4f} m')
    print(f'    RMS error ψ  : {rms(e_yaw):.4f} rad')
    print(f'    RMS error xy : {rms(err_xy):.4f} m')
    print('─'*55)
    print('  LEADER ERRORS (relative to setpoint):')
    print(f'    RMS error x  : {rms(ex_L):.4f} m')
    print(f'    RMS error y  : {rms(ey_L):.4f} m')
    print(f'    RMS error z  : {rms(ez_L):.4f} m')
    print(f'    RMS error ψ  : {rms(epsi_L):.4f} rad')
    print('─'*55)
    print('  FORMATION DISTANCE L-S:')
    print(f'    Mean distance XY : {np.mean(dist_xy):.4f} m')
    print(f'    Max distance XY  : {np.max(dist_xy):.4f} m')
    print(f'    Min distance XY  : {np.min(dist_xy):.4f} m')
    print(f'    Formation RMS error: {rms(dist_xy - desired_dist_xy):.4f} m')
    print('─'*55)
    print(f'  Mean Z distance L-S: {np.mean(dist_z):.4f} m')
    print('='*55 + '\n')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # IAE METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    if SHOW_IAE_METRICS:
        # Calcular IAE para cada error
        iae_x = compute_iae(t, np.abs(ex))
        iae_y = compute_iae(t, np.abs(ey))
        iae_z = compute_iae(t, np.abs(ez))
        iae_yaw = compute_iae(t, np.abs(e_yaw))
        iae_xy = compute_iae(t, err_xy)
        
        # Error de formación
        dist_err = np.abs(dist_xy - desired_dist_xy)
        iae_form = compute_iae(t, dist_err)
        
        print('='*60)
        print('  IAE METRICS (Integral Absolute Error)')
        print('='*60)
        print(f'  Duration: {duration:.2f} s')
        print('─'*60)
        print(f'  {"Variable":<25} {"IAE":>12} {"Unidad":>10}')
        print(f'  {"":25} {"[·s]":>12} {"":10}')
        print('─'*60)
        print(f'  {"e_x (follower)":<25} {iae_x:>12.4f}   m·s')
        print(f'  {"e_y (follower)":<25} {iae_y:>12.4f}   m·s')
        print(f'  {"e_z (follower)":<25} {iae_z:>12.4f}   m·s')
        print(f'  {"e_ψ (follower)":<25} {iae_yaw:>12.4f}   rad·s')
        print(f'  {"error_xy (follower)":<25} {iae_xy:>12.4f}   m·s')
        print(f'  {"formation error":<25} {iae_form:>12.4f}   m·s')
        print('─'*60)
        print('='*60 + '\n')
        
        # Exportar a CSV
        if EXPORT_METRICS_CSV:
            base_name = os.path.splitext(ARCHIVO_CSV)[0]
            metrics = {
                'e_x (follower)': {'IAE': iae_x, 'units': 'm'},
                'e_y (follower)': {'IAE': iae_y, 'units': 'm'},
                'e_z (follower)': {'IAE': iae_z, 'units': 'm'},
                'e_ψ (follower)': {'IAE': iae_yaw, 'units': 'rad'},
                'error_xy (follower)': {'IAE': iae_xy, 'units': 'm'},
                'formation error': {'IAE': iae_form, 'units': 'm'}
            }
            export_iae_csv(metrics, base_name)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    plt.ioff()
    base = os.path.splitext(ARCHIVO_CSV)[0]
    figs = []
    
    # Transform coordinates for Mission Planner style
    if MISSION_PLANNER_STYLE:
        lx_plot, ly_plot = ly, lx
        sx_plot, sy_plot = sy, sx
        xd_plot, yd_plot = yd, xd
        lxsp_plot, lysp_plot = lysp, lxsp
        th = np.linspace(0, 2 * math.pi, 400)
        circ_x = RADIUS * np.sin(th)
        circ_y = RADIUS * np.cos(th)
        x_label, y_label = 'x [m]', 'y [m]'
        title_suffix = ' (Mission Planner style)'
    else:
        lx_plot, ly_plot = lx, ly
        sx_plot, sy_plot = sx, sy
        xd_plot, yd_plot = xd, yd
        lxsp_plot, lysp_plot = lxsp, lysp
        th = np.linspace(0, 2 * math.pi, 400)
        circ_x = RADIUS * np.cos(th)
        circ_y = RADIUS * np.sin(th)
        x_label, y_label = 'Norte [m]', 'Este [m]'
        title_suffix = ''
    
    # Select indices for plots
    n_frames = min(6, n)
    idx_fr = np.round(np.linspace(0, n-1, n_frames)).astype(int) if n > 1 else [0]
    n_d = min(20, n)
    idx_d = np.round(np.linspace(0, n-1, n_d+2)).astype(int)[1:-1] if n > 2 else [0]
    
    # ── Figure 1: XY Trajectory ──────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.plot(lxsp_plot, lysp_plot, '--', color=C_refL, lw=LWS, label='Setpoint Leader', alpha=0.8)
    ax.plot(xd_plot, yd_plot, '--', color=C_refS, lw=LWS, label='Setpoint Follower', alpha=0.8)
    ax.plot(lx_plot, ly_plot, '-', color=C_L, lw=LW, label='Leader real')
    ax.plot(sx_plot, sy_plot, '-', color=C_S, lw=LW, label='Follower real')
    ax.plot(circ_x, circ_y, ':', color=[0.6,0.6,0.6], lw=1.0, label='Theoretical')
    
    if len(lx_plot) > 0:
        ax.plot(lx_plot[0], ly_plot[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Leader start')
    if len(sx_plot) > 0:
        ax.plot(sx_plot[0], sy_plot[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Follower start')
    
    ax.plot(0, 0, '+k', ms=10, mew=2, label='Origin')
    
    for i in idx_d:
        if i < len(lx_plot) and i < len(sx_plot):
            ax.quiver(lx_plot[i], ly_plot[i], 
                      sx_plot[i] - lx_plot[i], sy_plot[i] - ly_plot[i],
                      angles='xy', scale_units='xy', scale=1,
                      color=C_form, width=0.003, headwidth=3.5,
                      headlength=3.5, headaxislength=3, alpha=0.65)
    
    for i in idx_fr:
        if i < len(lx_plot):
            _draw_drone_frame(ax, lx_plot[i], ly_plot[i], psiL[i] if i < len(psiL) else 0)
        if i < len(sx_plot):
            _draw_drone_frame(ax, sx_plot[i], sy_plot[i], psiS[i] if i < len(psiS) else 0)
    
    _draw_coordinate_frame(ax, 0, 0)
    ax.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label='Forward')
    ax.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label='Lateral')
    ax.plot([], [], '-', color=C_form, lw=1.5, label='Vector d')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f'XY Trajectory - Leader & Follower{title_suffix}', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig1.tight_layout()
    figs.append((fig1, f'{base}_fig1_XY_trajectory'))
    
    # ── Figure 2: Leader trajectory ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax2.plot(lxsp_plot, lysp_plot, '--', color=C_refL, lw=LWS, label='Setpoint Leader')
    ax2.plot(lx_plot, ly_plot, '-', color=C_L, lw=LW, label='Leader real')
    if len(lx_plot) > 0:
        ax2.plot(lx_plot[0], ly_plot[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Start')
    ax2.plot(0, 0, '+k', ms=10, mew=2, label='Origin')
    for i in idx_fr[:4]:
        if i < len(lx_plot):
            _draw_drone_frame(ax2, lx_plot[i], ly_plot[i], psiL[i] if i < len(psiL) else 0)
    _draw_coordinate_frame(ax2, 0, 0)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title(f'Leader: Setpoint vs Real Trajectory{title_suffix}', fontsize=12)
    ax2.legend(loc='best', fontsize=8)
    fig2.tight_layout()
    figs.append((fig2, f'{base}_fig2_leader_trajectory'))
    
    # ── Figure 3: Follower trajectory ────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax3.plot(xd_plot, yd_plot, '--', color=C_refS, lw=LWS, label='Setpoint Follower')
    ax3.plot(sx_plot, sy_plot, '-', color=C_S, lw=LW, label='Follower real')
    if len(sx_plot) > 0:
        ax3.plot(sx_plot[0], sy_plot[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Start')
    ax3.plot(0, 0, '+k', ms=10, mew=2, label='Origin')
    for i in idx_fr[:4]:
        if i < len(sx_plot):
            _draw_drone_frame(ax3, sx_plot[i], sy_plot[i], psiS[i] if i < len(psiS) else 0)
    _draw_coordinate_frame(ax3, 0, 0)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    ax3.set_title(f'Follower: Setpoint vs Real Trajectory{title_suffix}', fontsize=12)
    ax3.legend(loc='best', fontsize=8)
    fig3.tight_layout()
    figs.append((fig3, f'{base}_fig3_follower_trajectory'))
    
    # ── Figure 4: Position & orientation vs time ─────────────────────────────
    fig4, axs4 = plt.subplots(4, 1, figsize=(11, 9), sharex=True, facecolor='white')
    fig4.suptitle('Position & Orientation vs Time', fontsize=13, fontweight='bold')
    axs4[0].plot(t, lxsp, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axs4[0].plot(t, xd, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[0].plot(t, lx, '-', color=C_L, lw=LW, label='x Leader')
    axs4[0].plot(t, sx, '-', color=C_S, lw=LW, label='x Follower')
    axs4[0].set_ylabel('x [m]')
    axs4[0].grid(True, alpha=0.3)
    axs4[0].legend(loc='best', ncol=2, fontsize=8)
    axs4[1].plot(t, lysp, '--', color=C_refL, lw=LWS)
    axs4[1].plot(t, yd, '--', color=C_refS, lw=LWS)
    axs4[1].plot(t, ly, '-', color=C_L, lw=LW)
    axs4[1].plot(t, sy, '-', color=C_S, lw=LW)
    axs4[1].set_ylabel('y [m]')
    axs4[1].grid(True, alpha=0.3)
    axs4[2].plot(t, lz_sp, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axs4[2].plot(t, zd, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[2].plot(t, lz, '-', color=C_L, lw=LW, label='z Leader')
    axs4[2].plot(t, sz, '-', color=C_S, lw=LW, label='z Follower')
    axs4[2].set_ylabel('z [m]')
    axs4[2].grid(True, alpha=0.3)
    axs4[2].legend(loc='best', ncol=2, fontsize=8)
    axs4[3].plot(t, np.degrees(psiL_sp), '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[3].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW, label='ψ Leader')
    axs4[3].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW, label='ψ Follower')
    axs4[3].set_ylabel('ψ [°]')
    axs4[3].set_xlabel('Time [s]')
    axs4[3].grid(True, alpha=0.3)
    axs4[3].legend(loc='best', ncol=2, fontsize=8)
    fig4.tight_layout()
    figs.append((fig4, f'{base}_fig4_position_vs_time'))
    
    # ── Figure 5: Follower errors ────────────────────────────────────────────
    fig5, axs5 = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig5.suptitle('Follower Errors relative to setpoint', fontsize=13, fontweight='bold')
    for ax5, data, lbl in zip(axs5,
            [ex, ey, ez, np.degrees(e_yaw)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        ax5.plot(t, data, '-', color=C_S, lw=LW)
        ax5.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax5.set_ylabel(lbl)
        ax5.grid(True, alpha=0.3)
    axs5[-1].set_xlabel('Time [s]')
    fig5.tight_layout()
    figs.append((fig5, f'{base}_fig5_follower_errors'))
    
    # ── Figure 5b: Leader errors ─────────────────────────────────────────────
    fig5b, axs5b = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig5b.suptitle('Leader Errors relative to setpoint', fontsize=13, fontweight='bold')
    for ax5b, data, lbl in zip(axs5b,
            [ex_L, ey_L, ez_L, np.degrees(epsi_L)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        ax5b.plot(t, data, '-', color=C_L, lw=LW)
        ax5b.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax5b.set_ylabel(lbl)
        ax5b.grid(True, alpha=0.3)
    axs5b[-1].set_xlabel('Time [s]')
    fig5b.tight_layout()
    figs.append((fig5b, f'{base}_fig5b_leader_errors'))
    
    # ── Figure 6: Formation distance ─────────────────────────────────────────
    fig6, axs6 = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig6.suptitle('Formation Distance L-S and Yaw', fontsize=13, fontweight='bold')
    axs6[0].plot(t, dist_xy, '-', color=C_dist, lw=LW, label='XY distance real')
    axs6[0].axhline(desired_dist_xy, color='r', ls='--', lw=1.5,
                    label=f'Target d ≈ {desired_dist_xy:.2f} m')
    axs6[0].set_ylabel('XY dist [m]')
    axs6[0].grid(True, alpha=0.3)
    axs6[0].legend(loc='best', fontsize=9)
    axs6[1].plot(t, dist_z, '-', color=[0.55, 0.25, 0.65], lw=LW, label='Z distance real')
    axs6[1].axhline(desired_dist_z, color='r', ls='--', lw=1.5,
                    label=f'Target Δz ≈ {desired_dist_z:.2f} m')
    axs6[1].set_ylabel('Z dist [m]')
    axs6[1].grid(True, alpha=0.3)
    axs6[1].legend(loc='best', fontsize=9)
    axs6[2].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW, label='ψ Leader')
    axs6[2].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW, label='ψ Follower')
    axs6[2].set_ylabel('ψ [°]')
    axs6[2].set_xlabel('Time [s]')
    axs6[2].grid(True, alpha=0.3)
    axs6[2].legend(loc='best', fontsize=9)
    fig6.tight_layout()
    figs.append((fig6, f'{base}_fig6_formation_distance'))
    
    # ── Figure 7: PID breakdown ──────────────────────────────────────────────
    fig7, axs7 = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor='white')
    fig7.suptitle('PID Controller + Feed-forward Breakdown (Follower)',
                  fontsize=13, fontweight='bold')
    for ax7, ff, pp, dd, cmd, lb in zip(
            axs7,
            [ff_x, ff_y], [vx_p, vy_p], [vx_d, vy_d], [vx_cmd, vy_cmd],
            ['X Axis [m/s]', 'Y Axis [m/s]']):
        ax7.plot(t, ff, lw=LWS, label='Feed-forward', color=[0.85, 0.55, 0.10])
        ax7.plot(t, pp, lw=LWS, label='Proportional', color=C_L)
        ax7.plot(t, dd, lw=LWS, label='Derivative', color=C_dist)
        ax7.plot(t, cmd, 'k-', lw=LW, alpha=0.85, label='Cmd total')
        ax7.axhline(0, color='gray', ls='--', lw=0.6)
        ax7.set_ylabel(lb)
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
    axs7[-1].set_xlabel('Time [s]')
    fig7.tight_layout()
    figs.append((fig7, f'{base}_fig7_PID'))
    
    # ── Save or show ─────────────────────────────────────────────────────────
    if SAVE_FIGURES:
        out_dir = OUTPUT_DIR if OUTPUT_DIR != "." else os.path.dirname(os.path.abspath(csv_path)) or "."
        for fig, name in figs:
            path = os.path.join(out_dir, name + '.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  💾 Saved: {path}")
        print(f"\n✅ {len(figs)} figures saved to {out_dir}")
    else:
        print(f"  Showing {len(figs)} figures. Close windows to finish.\n")
        plt.show()


if __name__ == '__main__':
    plot_results()
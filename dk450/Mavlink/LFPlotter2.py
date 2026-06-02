#!/usr/bin/env python3
"""
plot_lf_results.py — Visualización post-vuelo Líder-Seguidor
=============================================================
Replica exactamente las 7 figuras de MavlinkControlLFx4.py
a partir de un archivo CSV de log.

VERSIÓN MEJORADA:
    - Agrega cálculo de métricas integrales (IAE, ISE, ITAE)
    - Agrega análisis de estadísticas de error
    - Agrega gráfico de métricas integrales acumuladas

Uso:
    python plot_lf_results.py

Configuración:
    - Modificar la variable ARCHIVO_CSV con el nombre de tu archivo
    - Opcionalmente ajustar ORIGEN_X, ORIGEN_Y para el centro del sistema de referencia
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
ARCHIVO_CSV = "lf_log_SITL1.csv"  # <--- Cambia aquí el nombre de tu archivo

# ── Origen del sistema de coordenadas (centro del círculo o punto de referencia) ──
# Si no quieres desplazar, déjalo en (0, 0)
ORIGEN_X = 0   # Coordenada X del origen (metros)
ORIGEN_Y = 0   # Coordenada Y del origen (metros)

# ── Opciones de salida ─────────────────────────────────────────────────────────
SAVE_FIGURES = False    # True = guardar PNGs | False = mostrar en pantalla
OUTPUT_DIR = "."        # Directorio donde guardar las figuras (punto = actual)

# ── Opciones adicionales ───────────────────────────────────────────────────────
SHOW_INTEGRAL_METRICS = True      # Mostrar métricas integrales en consola
PLOT_INTEGRAL_METRICS = True      # Graficar evolución de métricas integrales
EXPORT_METRICS_CSV = True         # Exportar métricas a archivo CSV

# ═══════════════════════════════════════════════════════════════════════════════

# ── Paleta de colores (igual que el script original) ──────────────────────────
C_refL = np.array([0.40, 0.65, 1.00])   # azul claro  — setpoint líder
C_refS = np.array([1.00, 0.60, 0.40])   # naranja claro — setpoint seguidor
C_L    = np.array([0.10, 0.35, 0.75])   # azul oscuro — líder real
C_S    = np.array([0.80, 0.15, 0.10])   # rojo        — seguidor real
C_dist = np.array([0.25, 0.75, 0.45])   # verde       — distancia
C_form = np.array([0.50, 0.50, 0.50])   # gris        — vector formación
C_int  = np.array([0.60, 0.30, 0.80])   # morado      — métricas integrales
LW,  LWS = 2.0, 1.2

SC_DRONE = 0.40
SC_FRAME = SC_DRONE * 2.0

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES PARA MÉTRICAS INTEGRALES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_integral_metrics(t, error_signal, error_name=""):
    """
    Calcula métricas integrales para una señal de error.
    
    Args:
        t: array de tiempos
        error_signal: array de errores
        error_name: nombre del error (para imprimir)
    
    Returns:
        dict con IAE, ISE, ITAE
    """
    dt = np.diff(t)
    
    # Asegurar que t y error tienen la misma longitud
    if len(error_signal) > len(t):
        error_signal = error_signal[:len(t)]
    elif len(error_signal) < len(t):
        t = t[:len(error_signal)]
    
    abs_err = np.abs(error_signal)
    sq_err = error_signal ** 2
    
    # IAE = ∫|e| dt
    iae = integrate.trapezoid(abs_err, t)
    
    # ISE = ∫ e² dt
    ise = integrate.trapezoid(sq_err, t)
    
    # ITAE = ∫ t·|e| dt
    t_abs_err = t * abs_err
    itae = integrate.trapezoid(t_abs_err, t)
    
    return {
        'IAE': iae,
        'ISE': ise,
        'ITAE': itae,
        'max_error': np.max(abs_err),
        'mean_error': np.mean(abs_err),
        'std_error': np.std(error_signal),
        'rms_error': np.sqrt(np.mean(sq_err))
    }


def compute_all_integral_metrics(t, df, variables):
    """
    Calcula métricas integrales para múltiples variables.
    
    Args:
        t: array de tiempos
        df: DataFrame con los datos
        variables: lista de tuplas (nombre_columna, nombre_para_mostrar, unidades)
    
    Returns:
        dict con métricas para cada variable
    """
    metrics = {}
    
    for col_name, display_name, units in variables:
        if col_name in df.columns:
            error = df[col_name].astype(float).to_numpy()
            # Recortar al tamaño de t
            if len(error) > len(t):
                error = error[:len(t)]
            metrics[display_name] = compute_integral_metrics(t, error, display_name)
            metrics[display_name]['units'] = units
            metrics[display_name]['col_name'] = col_name
        else:
            print(f"  ⚠️  Columna '{col_name}' no encontrada en el CSV")
    
    return metrics


def print_integral_metrics(metrics, duration):
    """
    Imprime las métricas integrales de forma formateada.
    """
    if not metrics:
        return
    
    print('\n' + '='*75)
    print('  MÉTRICAS INTEGRALES DE ERROR (IAE, ISE, ITAE)')
    print('='*75)
    print(f'  Duración del segmento: {duration:.2f} s')
    print('─'*75)
    print(f'  {"Variable":<20} {"IAE":>12} {"ISE":>12} {"ITAE":>12} {"RMS":>10}')
    print(f'  {"":20} {"[m·s]":>12} {"[m²·s]":>12} {"[m·s²]":>12} {"[m]":>10}')
    print('─'*75)
    
    for var_name, data in metrics.items():
        if var_name == 'duration':
            continue
        units = data.get('units', '')
        unit_str = f'[{units}·s]' if units else ''
        print(f'  {var_name:<20} {data["IAE"]:>12.4f} {data["ISE"]:>12.4f} '
              f'{data["ITAE"]:>12.4f} {data["rms_error"]:>10.4f}')
    
    print('─'*75)
    print('\n  Interpretación:')
    print('  • IAE (Integral Absolute Error)  → sensibilidad a errores sostenidos')
    print('  • ISE (Integral Square Error)    → penaliza errores grandes')
    print('  • ITAE (Integral Time Absolute Error) → enfatiza errores al final')
    print('  • RMS (Root Mean Square)         → error cuadrático medio')
    print('='*75 + '\n')


def export_metrics_csv(metrics, base_filename):
    """
    Exporta las métricas integrales a un archivo CSV.
    """
    if not metrics:
        return
    
    metrics_fname = f'{base_filename}_integral_metrics.csv'
    
    # Preparar datos para CSV
    rows = []
    for var_name, data in metrics.items():
        if var_name == 'duration':
            continue
        row = {
            'variable': var_name,
            'units': data.get('units', ''),
            'IAE': data['IAE'],
            'ISE': data['ISE'],
            'ITAE': data['ITAE'],
            'RMS_error': data['rms_error'],
            'max_error': data['max_error'],
            'mean_error': data['mean_error'],
            'std_error': data['std_error']
        }
        rows.append(row)
    
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(metrics_fname, index=False)
    print(f"📊 Métricas integrales guardadas en: {metrics_fname}")


def plot_cumulative_metrics(t, df, variables, base_filename):
    """
    Grafica la evolución acumulada de las métricas IAE, ISE, ITAE.
    """
    if not PLOT_INTEGRAL_METRICS:
        return
    
    n_vars = len([v for v in variables if v[0] in df.columns])
    if n_vars == 0:
        return
    
    # Calcular métricas acumuladas
    dt = np.diff(t)
    if len(dt) == 0:
        return
    
    # Asegurar que todos los arrays tengan la misma longitud
    t_plot = t.copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='white')
    fig.suptitle('Métricas Integrales Acumuladas (IAE, ISE, ITAE)', fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))
    color_idx = 0
    
    for col_name, display_name, units in variables:
        if col_name not in df.columns:
            continue
        
        error = df[col_name].astype(float).to_numpy()
        if len(error) > len(t_plot):
            error = error[:len(t_plot)]
        elif len(error) < len(t_plot):
            t_plot = t_plot[:len(error)]
        
        abs_err = np.abs(error)
        sq_err = error ** 2
        
        # Calcular integrales acumuladas
        iae_cum = integrate.cumulative_trapezoid(abs_err, t_plot, initial=0)
        ise_cum = integrate.cumulative_trapezoid(sq_err, t_plot, initial=0)
        
        t_abs_err = t_plot * abs_err
        itae_cum = integrate.cumulative_trapezoid(t_abs_err, t_plot, initial=0)
        
        axes[0].plot(t_plot, iae_cum, '-', color=colors[color_idx], lw=2, 
                    label=f'{display_name}')
        axes[1].plot(t_plot, ise_cum, '-', color=colors[color_idx], lw=2,
                    label=f'{display_name}')
        axes[2].plot(t_plot, itae_cum, '-', color=colors[color_idx], lw=2,
                    label=f'{display_name}')
        
        color_idx += 1
    
    axes[0].set_ylabel('IAE [m·s]')
    axes[0].set_title('Integral del Error Absoluto')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=8)
    
    axes[1].set_ylabel('ISE [m²·s]')
    axes[1].set_title('Integral del Error Cuadrático')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=8)
    
    axes[2].set_xlabel('Tiempo [s]')
    axes[2].set_ylabel('ITAE [m·s²]')
    axes[2].set_title('Integral del Error Absoluto Ponderado por Tiempo')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=8)
    
    fig.tight_layout()
    
    # Guardar o mostrar
    if SAVE_FIGURES:
        out_dir = OUTPUT_DIR if OUTPUT_DIR != "." else os.path.dirname(os.path.abspath(csv_path)) or "."
        path = os.path.join(out_dir, f'{base_filename}_fig_integral_metrics.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  💾 Guardado: {path}")
    else:
        plt.show()
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE DIBUJO (sin cambios)
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_np(arr):
    return np.arctan2(np.sin(arr), np.cos(arr))

def _draw_drone_frame(ax, x, y, yaw, scale=SC_DRONE):
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=4, headlength=3.5, headaxislength=3, alpha=0.85)
    ax.quiver(x, y,  scale*np.cos(yaw),           scale*np.sin(yaw),
              color=[0.85, 0.10, 0.10], **kw)
    ax.quiver(x, y,  scale*np.cos(yaw+np.pi/2),   scale*np.sin(yaw+np.pi/2),
              color=[0.10, 0.65, 0.10], **kw)

def _draw_coordinate_frame(ax, x, y, scale=SC_FRAME):
    """Dibuja el marco de coordenadas de referencia en la posición (x, y)"""
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.007, headwidth=4, headlength=4, headaxislength=3.5, alpha=0.9)
    ax.quiver(x, y, scale, 0, color=[0.85, 0.10, 0.10], **kw)
    ax.quiver(x, y, 0, scale, color=[0.10, 0.65, 0.10], **kw)
    ax.plot(x, y, 'ok', ms=6, zorder=5)
    ax.text(x + scale + 0.12, y - 0.10, r'$\hat{x}$', fontsize=10, fontweight='bold', va='top')
    ax.text(x - 0.10, y + scale + 0.10, r'$\hat{y}$', fontsize=10, fontweight='bold', ha='right')


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results():
    csv_path = os.path.join(OUTPUT_DIR, ARCHIVO_CSV) if OUTPUT_DIR != "." else ARCHIVO_CSV
    
    if not os.path.isfile(csv_path):
        print(f"❌ No se encontró el archivo: {csv_path}")
        print(f"   Verifica que el archivo '{ARCHIVO_CSV}' exista en el directorio actual")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if len(df) == 0:
        print("⚠️  El archivo CSV está vacío.")
        return

    def arr(col):
        return df[col].astype(float).to_numpy()

    t          = arr('t')
    lx, ly, lz = arr('lx'), arr('ly'), arr('lz')
    sx, sy, sz = arr('sx'), arr('sy'), arr('sz')
    xd, yd, zd = arr('xd'), arr('yd'), arr('zd')
    lxsp, lysp = arr('lx_sp'), arr('ly_sp')
    ex, ey, ez = arr('ex'), arr('ey'), arr('ez')
    e_yaw      = arr('e_yaw')
    dist_xy    = arr('dist_xy')
    dist_z     = arr('dist_z')
    psiL       = arr('l_yaw')
    psiS       = arr('s_yaw')
    ff_x, ff_y = arr('ff_x'), arr('ff_y')
    vx_p, vy_p = arr('vx_p'), arr('vy_p')
    vx_d, vy_d = arr('vx_d'), arr('vy_d')
    vx_cmd, vy_cmd = arr('vx_cmd'), arr('vy_cmd')
    
    # ── Calcular setpoint Z del líder (si no existe en el CSV) ─────────────────
    if 'lz_sp' in df.columns:
        lz_sp = arr('lz_sp')
    else:
        lz_sp = np.median(lz) * np.ones_like(lz)
        print(f"  ℹ️  Columna 'lz_sp' no encontrada. Usando mediana de lz = {lz_sp[0]:.2f} m como setpoint")
    
    # Aplicar desplazamiento del origen si es necesario
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        print(f"  📍 Aplicando desplazamiento del origen: centro en ({ORIGEN_X}, {ORIGEN_Y})")
        lx = lx - ORIGEN_X
        ly = ly - ORIGEN_Y
        sx = sx - ORIGEN_X
        sy = sy - ORIGEN_Y
        lxsp = lxsp - ORIGEN_X
        lysp = lysp - ORIGEN_Y
        xd = xd - ORIGEN_X
        yd = yd - ORIGEN_Y

    # Obtener los valores deseados de formación desde los datos
    desired_dist_xy = np.median(dist_xy[100:200]) if len(dist_xy) > 200 else np.mean(dist_xy)
    desired_dist_z = np.median(dist_z[100:200]) if len(dist_z) > 200 else np.mean(dist_z)
    
    n       = len(t)
    err_xy  = np.hypot(ex, ey)
    psidesS = psiL.copy()
    duration = t[-1] - t[0]

    # ── Métricas en consola (RMS existente) ────────────────────────────────────
    def rms(v): return np.sqrt(np.mean(v**2)) if len(v) > 0 else 0.0

    # Calcular errores del líder
    ex_L = lxsp - lx
    ey_L = lysp - ly
    ez_L = lz_sp - lz
    theta_sp = np.arctan2(lysp, lxsp)
    psiL_sp = _wrap_np(theta_sp + math.pi/2)
    epsi_L = _wrap_np(psiL_sp - psiL)

    print('\n' + '='*55)
    print(f'  ARCHIVO : {ARCHIVO_CSV}')
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        print(f'  ORIGEN  : ({ORIGEN_X}, {ORIGEN_Y}) m')
    print(f'  Muestras: {n}   Duración: {duration:.1f} s')
    print('='*55)
    print('  MÉTRICAS DE VUELO (RMS)')
    print('─'*55)
    print('  ERRORES DEL SEGUIDOR (respecto a su setpoint):')
    print(f'    RMS error x  : {rms(ex):.4f} m')
    print(f'    RMS error y  : {rms(ey):.4f} m')
    print(f'    RMS error z  : {rms(ez):.4f} m')
    print(f'    RMS error ψ  : {rms(e_yaw):.4f} rad')
    print(f'    RMS error xy : {rms(err_xy):.4f} m')
    print('─'*55)
    print('  ERRORES DEL LÍDER (respecto a su setpoint):')
    print(f'    RMS error x  : {rms(ex_L):.4f} m')
    print(f'    RMS error y  : {rms(ey_L):.4f} m')
    print(f'    RMS error z  : {rms(ez_L):.4f} m')
    print(f'    RMS error ψ  : {rms(epsi_L):.4f} rad')
    print('─'*55)
    print('  DISTANCIA DE FORMACIÓN L-S:')
    print(f'    Distancia media   XY : {np.mean(dist_xy):.4f} m')
    print(f'    Distancia deseada XY : {desired_dist_xy:.4f} m (estimada)')
    print(f'    Distancia máx     XY : {np.max(dist_xy):.4f} m')
    print(f'    Distancia mín     XY : {np.min(dist_xy):.4f} m')
    print(f'    Error formación RMS  : {rms(dist_xy - desired_dist_xy):.4f} m')
    print('─'*55)
    print(f'  Distancia Z media L-S : {np.mean(dist_z):.4f} m')
    print(f'  Distancia Z deseada   : {desired_dist_z:.4f} m (estimada)')
    print('='*55 + '\n')

    # ═══════════════════════════════════════════════════════════════════════════
    # NUEVO: CÁLCULO DE MÉTRICAS INTEGRALES (IAE, ISE, ITAE)
    # ═══════════════════════════════════════════════════════════════════════════
    
    if SHOW_INTEGRAL_METRICS:
        # Definir las variables a analizar
        variables_integral = [
            ('ex', 'e_x (seguidor)', 'm'),
            ('ey', 'e_y (seguidor)', 'm'),
            ('ez', 'e_z (seguidor)', 'm'),
            ('e_yaw', 'e_ψ (seguidor)', 'rad'),
            ('err_xy', 'error_xy (seguidor)', 'm'),
        ]
        
        # Calcular métricas integrales
        integral_metrics = compute_all_integral_metrics(t, df, variables_integral)
        
        # Agregar error de formación (distancia L-S)
        if 'dist_xy' in df.columns:
            dist_err = np.abs(dist_xy - desired_dist_xy)
            t_adj = t[:len(dist_err)]
            integral_metrics['formación XY'] = compute_integral_metrics(t_adj, dist_err, 'formación XY')
            integral_metrics['formación XY']['units'] = 'm'
        
        # Imprimir métricas
        print_integral_metrics(integral_metrics, duration)
        
        # Exportar a CSV
        if EXPORT_METRICS_CSV:
            base_name = os.path.splitext(ARCHIVO_CSV)[0]
            export_metrics_csv(integral_metrics, base_name)
        
        # Graficar métricas acumuladas
        if PLOT_INTEGRAL_METRICS:
            base_name = os.path.splitext(ARCHIVO_CSV)[0]
            plot_cumulative_metrics(t, df, variables_integral, base_name)

    # ═══════════════════════════════════════════════════════════════════════════
    # GRÁFICAS ORIGINALES (sin cambios)
    # ═══════════════════════════════════════════════════════════════════════════

    plt.ioff()
    base = os.path.splitext(ARCHIVO_CSV)[0]
    figs = []

    n_frames = 6
    idx_fr = np.round(np.linspace(0, n-1, n_frames)).astype(int)
    n_d    = 20
    idx_d  = np.round(np.linspace(0, n-1, n_d+2)).astype(int)[1:-1]

    # ── Fig 1 — Trayectoria XY completa con frames y vector de formación ──────
    fig1, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.plot(lxsp, lysp, '--', color=C_refL, lw=LWS, label='Setpoint Líder',    alpha=0.8)
    ax.plot(xd,   yd,   '--', color=C_refS, lw=LWS, label='Setpoint Seguidor', alpha=0.8)
    ax.plot(lx,   ly,   '-',  color=C_L,    lw=LW,  label='Líder real')
    ax.plot(sx,   sy,   '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax.plot(lx[0], ly[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Inicio Líder')
    ax.plot(sx[0], sy[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Inicio Seguidor')
    
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        ax.plot(0, 0, '+k', ms=10, mew=2, label=f'Origen ({ORIGEN_X}, {ORIGEN_Y})')
    else:
        ax.plot(0, 0, '+k', ms=10, mew=2, label='Origen (0,0)')
    
    for i in idx_d:
        ax.quiver(lx[i], ly[i], sx[i]-lx[i], sy[i]-ly[i],
                  angles='xy', scale_units='xy', scale=1,
                  color=C_form, width=0.003, headwidth=3.5,
                  headlength=3.5, headaxislength=3, alpha=0.65)
    for i in idx_fr:
        _draw_drone_frame(ax, lx[i], ly[i], psiL[i])
        _draw_drone_frame(ax, sx[i], sy[i], psiS[i])
    
    _draw_coordinate_frame(ax, 0, 0)
    
    ax.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}$ (frente)')
    ax.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}$ (lateral)')
    ax.plot([], [], '-', color=C_form, lw=1.5,        label=r'Vector $\mathbf{d}$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(r'$x$ [m]', fontsize=11)
    ax.set_ylabel(r'$y$ [m]', fontsize=11)
    ax.set_title('Trayectoria XY — Líder y Seguidor (Datos reales del vuelo)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig1.tight_layout()
    figs.append((fig1, f'{base}_fig1_trayectoria_XY'))

    # ── Fig 2 — Trayectoria del Líder (setpoint vs real) ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax2.plot(lxsp, lysp, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax2.plot(lx,   ly,   '-',  color=C_L,    lw=LW,  label='Líder real')
    ax2.plot(lx[0], ly[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Inicio')
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        ax2.plot(0, 0, '+k', ms=10, mew=2, label=f'Origen ({ORIGEN_X}, {ORIGEN_Y})')
    else:
        ax2.plot(0, 0, '+k', ms=10, mew=2, label='Origen (0,0)')
    for i in idx_fr[:4]:
        _draw_drone_frame(ax2, lx[i], ly[i], psiL[i])
    _draw_coordinate_frame(ax2, 0, 0)
    ax2.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}$')
    ax2.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}$')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    ax2.set_title('Líder: Setpoint vs Trayectoria Real', fontsize=12)
    ax2.legend(loc='best', fontsize=8)
    fig2.tight_layout()
    figs.append((fig2, f'{base}_fig2_lider_sp_vs_real'))

    # ── Fig 3 — Trayectoria del Seguidor (setpoint vs real) ──────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax3.plot(xd, yd, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax3.plot(sx, sy, '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax3.plot(sx[0], sy[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Inicio')
    if ORIGEN_X != 0.0 or ORIGEN_Y != 0.0:
        ax3.plot(0, 0, '+k', ms=10, mew=2, label=f'Origen ({ORIGEN_X}, {ORIGEN_Y})')
    else:
        ax3.plot(0, 0, '+k', ms=10, mew=2, label='Origen (0,0)')
    for i in idx_fr[:4]:
        _draw_drone_frame(ax3, sx[i], sy[i], psiS[i])
    _draw_coordinate_frame(ax3, 0, 0)
    ax3.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}$')
    ax3.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}$')
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]')
    ax3.set_title('Seguidor: Setpoint vs Trayectoria Real', fontsize=12)
    ax3.legend(loc='best', fontsize=8)
    fig3.tight_layout()
    figs.append((fig3, f'{base}_fig3_seguidor_sp_vs_real'))

    # ── Fig 4 — Posición y orientación vs tiempo ──────────────────────────────
    fig4, axs4 = plt.subplots(4, 1, figsize=(11, 9), sharex=True, facecolor='white')
    fig4.suptitle('Posición y orientación vs Tiempo', fontsize=13, fontweight='bold')
    axs4[0].plot(t, lxsp, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axs4[0].plot(t, xd,   '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[0].plot(t, lx,   '-',  color=C_L,    lw=LW,  label='x Líder')
    axs4[0].plot(t, sx,   '-',  color=C_S,    lw=LW,  label='x Seguidor')
    axs4[0].set_ylabel('x [m]'); axs4[0].grid(True, alpha=0.3)
    axs4[0].legend(loc='best', ncol=2, fontsize=8); axs4[0].set_title('Posición X')
    axs4[1].plot(t, lysp, '--', color=C_refL, lw=LWS)
    axs4[1].plot(t, yd,   '--', color=C_refS, lw=LWS)
    axs4[1].plot(t, ly,   '-',  color=C_L,    lw=LW)
    axs4[1].plot(t, sy,   '-',  color=C_S,    lw=LW)
    axs4[1].set_ylabel('y [m]'); axs4[1].grid(True, alpha=0.3); axs4[1].set_title('Posición Y')
    axs4[2].plot(t, lz_sp, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axs4[2].plot(t, zd,    '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[2].plot(t, lz,    '-',  color=C_L,    lw=LW,  label='z Líder')
    axs4[2].plot(t, sz,    '-',  color=C_S,    lw=LW,  label='z Seguidor')
    axs4[2].set_ylabel('z [m]'); axs4[2].grid(True, alpha=0.3)
    axs4[2].legend(loc='best', ncol=2, fontsize=8); axs4[2].set_title('Altitud')
    axs4[3].plot(t, np.degrees(psidesS), '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[3].plot(t, np.degrees(psiL),    '-',  color=C_L,    lw=LW,  label='ψ Líder')
    axs4[3].plot(t, np.degrees(psiS),    '-',  color=C_S,    lw=LW,  label='ψ Seguidor')
    axs4[3].set_ylabel('ψ [°]'); axs4[3].set_xlabel('Tiempo [s]')
    axs4[3].grid(True, alpha=0.3)
    axs4[3].legend(loc='best', ncol=2, fontsize=8); axs4[3].set_title('Yaw')
    fig4.tight_layout()
    figs.append((fig4, f'{base}_fig4_posicion_vs_tiempo'))

    # ── Fig 5 — Errores del seguidor ──────────────────────────────────────────
    fig5, axs5 = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig5.suptitle('Errores del Seguidor respecto a su setpoint', fontsize=13, fontweight='bold')
    for ax5, data, lbl in zip(axs5,
            [ex, ey, ez, np.degrees(e_yaw)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        ax5.plot(t, data, '-', color=C_S, lw=LW)
        ax5.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax5.set_ylabel(lbl); ax5.grid(True, alpha=0.3)
    axs5[-1].set_xlabel('Tiempo [s]')
    fig5.tight_layout()
    figs.append((fig5, f'{base}_fig5_errores_seguidor'))

    # ── Fig 5b — Errores del líder respecto a su setpoint ────────────────────
    fig5b, axs5b = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig5b.suptitle('Errores del Líder respecto a su setpoint', fontsize=13, fontweight='bold')
    for ax5b, data, lbl in zip(axs5b,
            [ex_L, ey_L, ez_L, np.degrees(epsi_L)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        ax5b.plot(t, data, '-', color=C_L, lw=LW)
        ax5b.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax5b.set_ylabel(lbl); ax5b.grid(True, alpha=0.3)
    axs5b[-1].set_xlabel('Tiempo [s]')
    fig5b.tight_layout()
    figs.append((fig5b, f'{base}_fig5b_errores_lider'))

    # ── Fig 6 — Distancia L-S y yaw comparado ────────────────────────────────
    fig6, axs6 = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig6.suptitle('Distancia de Formación L-S y Yaw', fontsize=13, fontweight='bold')
    axs6[0].plot(t, dist_xy, '-', color=C_dist, lw=LW, label='Distancia XY real')
    axs6[0].axhline(desired_dist_xy, color='r', ls='--', lw=1.5,
                    label=f'Objetivo d ≈ {desired_dist_xy:.2f} m')
    axs6[0].set_ylabel('Dist XY [m]'); axs6[0].grid(True, alpha=0.3)
    axs6[0].legend(loc='best', fontsize=9); axs6[0].set_title('Distancia Horizontal L-S')
    axs6[1].plot(t, dist_z, '-', color=[0.55, 0.25, 0.65], lw=LW, label='Distancia Z real')
    axs6[1].axhline(desired_dist_z, color='r', ls='--', lw=1.5,
                    label=f'Objetivo Δz ≈ {desired_dist_z:.2f} m')
    axs6[1].set_ylabel('Dist Z [m]'); axs6[1].grid(True, alpha=0.3)
    axs6[1].legend(loc='best', fontsize=9); axs6[1].set_title('Distancia Vertical L-S')
    axs6[2].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW, label='ψ Líder')
    axs6[2].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW, label='ψ Seguidor')
    axs6[2].set_ylabel('ψ [°]'); axs6[2].set_xlabel('Tiempo [s]')
    axs6[2].grid(True, alpha=0.3)
    axs6[2].legend(loc='best', fontsize=9); axs6[2].set_title('Comparación de Yaw')
    fig6.tight_layout()
    figs.append((fig6, f'{base}_fig6_distancia_formacion'))

    # ── Fig 7 — Desglose PID ──────────────────────────────────────────────────
    fig7, axs7 = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor='white')
    fig7.suptitle('Desglose del Controlador PID + Feed-forward del Seguidor',
                  fontsize=13, fontweight='bold')
    for ax7, ff, pp, dd, cmd, lb in zip(
            axs7,
            [ff_x, ff_y], [vx_p, vy_p], [vx_d, vy_d], [vx_cmd, vy_cmd],
            ['Eje X [m/s]', 'Eje Y [m/s]']):
        ax7.plot(t, ff,  lw=LWS, label='Feed-forward', color=[0.85, 0.55, 0.10])
        ax7.plot(t, pp,  lw=LWS, label='Proporcional', color=C_L)
        ax7.plot(t, dd,  lw=LWS, label='Derivativo',   color=C_dist)
        ax7.plot(t, cmd, 'k-',   lw=LW, alpha=0.85,    label='Cmd total')
        ax7.axhline(0, color='gray', ls='--', lw=0.6)
        ax7.set_ylabel(lb); ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)
    axs7[-1].set_xlabel('Tiempo [s]')
    fig7.tight_layout()
    figs.append((fig7, f'{base}_fig7_PID'))

    # ── Guardar o mostrar ─────────────────────────────────────────────────────
    if SAVE_FIGURES:
        out_dir = OUTPUT_DIR if OUTPUT_DIR != "." else os.path.dirname(os.path.abspath(csv_path)) or "."
        for fig, name in figs:
            path = os.path.join(out_dir, name + '.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  💾 Guardado: {path}")
        print(f"\n✅ {len(figs)} figuras guardadas en {out_dir}")
    else:
        print(f"  Mostrando {len(figs)} figuras. Cierra las ventanas para terminar.\n")
        plt.show()

# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_results()
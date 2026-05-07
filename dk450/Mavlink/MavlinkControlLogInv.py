#!/usr/bin/env python3
"""
MavlinkControlLFLog.py — Controlador líder-seguidor con líder virtual desde CSV
================================================================================
Lee el CSV generado por MavlinkCirculoLog.py e interpola la trayectoria del
líder en tiempo real. Aplica la misma ley de control de MavlinkControlLF.py con
gráficas estilo MATLAB que incluyen frames de drones y sistema de coordenadas RTK.
"""

import math
import time
import csv
import threading
import matplotlib.pyplot as plt
import numpy as np
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
CONN           = 'udp:127.0.0.1:14552'
FOLLOWER_SYSID  = 1
FOLLOWER_COMPID = 1

# ── Offset polar ──────────────────────────────────────────────────────────────
OFFSET_D     = 1.0          # metros de separación horizontal
OFFSET_ALPHA = -math.pi/2   # ángulo respecto al eje del líder [rad]
OFFSET_DZ    = 1.0          # diferencia de altitud [m]

# ── Ganancias PID ─────────────────────────────────────────────────────────────
KP   = 1.0
KI   = 0.00
KD   = 0.0
KP_YAW = 2.0
KI_YAW = 0.0
KD_YAW = 0.0

# ── Límites ───────────────────────────────────────────────────────────────────
INTEGRAL_LIMIT     = 2.0
INTEGRAL_YAW_LIMIT = 1.0
V_MAX              = 3.0
YAW_RATE_MAX       = 1.0
RATE               = 20
DT                 = 1.0 / RATE
SPEED_FACTOR       = 1.0

# ── Archivo de log del líder ──────────────────────────────────────────────────
LEADER_CSV = 'circulo_log_1778123967.csv'

# ── Pre-posicionamiento ───────────────────────────────────────────────────────
PRE_POSITION        = True
PRE_CONV_RADIUS     = 0.20
PRE_CONV_Z          = 0.15
PRE_CONV_SPEED      = 0.10
PRE_CONV_HOLD       = 1.5
PRE_CONV_TIMEOUT    = 30.0
PRE_RATE            = 10

# ── Configuración de inversión de ejes ────────────────────────────────────────
INVERTIR_EJE_X = True   # True = invertir (derecha ↔ izquierda)
INVERTIR_EJE_Y = True   # True = invertir (arriba ↔ abajo)

# ── Parámetros de visualización (igual que MATLAB) ────────────────────────────
SC_DRONE = 0.50      # Escala para frames de drones
FW_ARROW = 2.8       # Ancho de flecha para ejes
SC_RTK   = SC_DRONE * 2.0   # Escala para ejes RTK (más grande)
FW_RTK   = FW_ARROW + 0.8   # Ancho de flecha para ejes RTK

# ── CSV de salida ─────────────────────────────────────────────────────────────
CSV_HEADER = [
    'time', 'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    'lx_sp', 'ly_sp', 'lz_sp', 'lex', 'ley',
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    'xd', 'yd', 'zd', 'ex', 'ey', 'ez', 'e_yaw',
    'ff_x', 'ff_y', 'ff_z', 'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i', 'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd', 'yaw_rate_cmd',
]

# ──────────────────────────────────────────────────────────────────────────────
# CARGA E INTERPOLACIÓN DEL CSV
# ──────────────────────────────────────────────────────────────────────────────
def load_leader_csv(path: str) -> tuple:
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        has_setpoint = all(c in reader.fieldnames for c in ('x_sp', 'y_sp', 'z_sp', 'ex_L', 'ey_L'))
        
        for row in reader:
            d = {k: float(v) for k, v in row.items()}
            if not has_setpoint:
                d['x_sp'], d['y_sp'], d['z_sp'] = d['x'], d['y'], d['z']
                d['ex_L'], d['ey_L'] = 0.0, 0.0
            rows.append(d)
    
    print(f'📂 CSV cargado: {len(rows)} muestras, duración={rows[-1]["t"]:.1f} s')
    return rows, has_setpoint

def lerp(a, b, t): return a + (b - a) * t

def lerp_yaw(y0, y1, t):
    diff = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
    return y0 + t * diff

def interpolate_leader(data: list, t_csv: float, idx: int) -> tuple:
    if t_csv <= data[0]['t']: return dict(data[0]), 0
    if t_csv >= data[-1]['t']: return dict(data[-1]), len(data) - 2
    
    while idx < len(data) - 2 and data[idx + 1]['t'] < t_csv:
        idx += 1
    
    r0, r1 = data[idx], data[idx + 1]
    dt_seg = r1['t'] - r0['t']
    alpha = (t_csv - r0['t']) / dt_seg if dt_seg > 1e-9 else 0.0
    
    return {
        't': t_csv, 'x': lerp(r0['x'], r1['x'], alpha),
        'y': lerp(r0['y'], r1['y'], alpha), 'z': lerp(r0['z'], r1['z'], alpha),
        'vx': lerp(r0['vx'], r1['vx'], alpha), 'vy': lerp(r0['vy'], r1['vy'], alpha),
        'vz': lerp(r0['vz'], r1['vz'], alpha),
        'yaw': lerp_yaw(r0['yaw'], r1['yaw'], alpha),
        'yaw_rate': lerp(r0['yaw_rate'], r1['yaw_rate'], alpha),
        'x_sp': lerp(r0['x_sp'], r1['x_sp'], alpha),
        'y_sp': lerp(r0['y_sp'], r1['y_sp'], alpha),
        'z_sp': lerp(r0['z_sp'], r1['z_sp'], alpha),
        'ex_L': lerp(r0['ex_L'], r1['ex_L'], alpha),
        'ey_L': lerp(r0['ey_L'], r1['ey_L'], alpha),
    }, idx

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DEL SEGUIDOR
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_follower = dict(x=None, y=None, z=None, vx=None, vy=None, vz=None, yaw=None, yaw_rate=None)

def _mavlink_reader(master, stop_event):
    while not stop_event.is_set():
        msg = master.recv_match(type=['LOCAL_POSITION_NED', 'ATTITUDE'], blocking=True, timeout=0.1)
        if msg is None or msg.get_srcSystem() != FOLLOWER_SYSID: continue
        with _state_lock:
            if msg.get_type() == 'LOCAL_POSITION_NED':
                _follower.update({'x': msg.x, 'y': msg.y, 'z': -msg.z, 'vx': msg.vx, 'vy': msg.vy, 'vz': -msg.vz})
            elif msg.get_type() == 'ATTITUDE':
                _follower.update({'yaw': msg.yaw, 'yaw_rate': msg.yawspeed})

def get_follower():
    with _state_lock: return dict(_follower)

def follower_ready(s): return all(v is not None for v in s.values())

# ──────────────────────────────────────────────────────────────────────────────
# MATEMÁTICAS
# ──────────────────────────────────────────────────────────────────────────────
def wrap(theta): return math.atan2(math.sin(theta), math.cos(theta))
def clamp(val, limit): return max(-limit, min(limit, val))

def compute_offset(psi_L):
    angle = psi_L + OFFSET_ALPHA
    return (OFFSET_D * math.cos(angle), OFFSET_D * math.sin(angle), OFFSET_DZ)

def compute_offset_dot(yaw_rate_L, dx, dy):
    return (yaw_rate_L * (-dy), yaw_rate_L * dx, 0.0)

class PIDState:
    def __init__(self):
        self.integral = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0

def compute_control(L: dict, S: dict, pid: PIDState) -> tuple:
    dx, dy, dz = compute_offset(L['yaw'])
    xd, yd, zd = L['x'] + dx, L['y'] + dy, L['z'] + dz
    ex, ey, ez = xd - S['x'], yd - S['y'], zd - S['z']
    
    pid.integral = [clamp(pid.integral[0] + ex * DT, INTEGRAL_LIMIT),
                    clamp(pid.integral[1] + ey * DT, INTEGRAL_LIMIT),
                    clamp(pid.integral[2] + ez * DT, INTEGRAL_LIMIT)]
    
    dv_x, dv_y, dv_z = L['vx'] - S['vx'], L['vy'] - S['vy'], L['vz'] - S['vz']
    ff_x, ff_y, ff_z = compute_offset_dot(L['yaw_rate'], dx, dy)
    
    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z
    
    v_horiz = math.hypot(vx, vy)
    if v_horiz > V_MAX: vx, vy = vx * V_MAX / v_horiz, vy * V_MAX / v_horiz
    vz, vz_ned = clamp(vz, V_MAX), -vz
    
    e_yaw = wrap(L['yaw'] - S['yaw'])
    pid.integral_yaw = clamp(pid.integral_yaw + e_yaw * DT, INTEGRAL_YAW_LIMIT)
    dyaw = L['yaw_rate'] - S['yaw_rate']
    yaw_rate_cmd = clamp(KP_YAW * e_yaw + KI_YAW * pid.integral_yaw + KD_YAW * dyaw, YAW_RATE_MAX)
    
    c = dict(xd=xd, yd=yd, zd=zd, ex=ex, ey=ey, ez=ez, e_yaw=e_yaw,
             ff_x=ff_x, ff_y=ff_y, ff_z=ff_z,
             vx_p=KP*ex, vy_p=KP*ey, vz_p=KP*ez,
             vx_i=KI*pid.integral[0], vy_i=KI*pid.integral[1], vz_i=KI*pid.integral[2],
             vx_d=KD*dv_x, vy_d=KD*dv_y, vz_d=KD*dv_z, vx=vx, vy=vy, vz=vz)
    return vx, vy, vz_ned, yaw_rate_cmd, c

# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE COMANDOS MAVLINK
# ──────────────────────────────────────────────────────────────────────────────
TYPE_MASK_VEL_YAWRATE = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                         mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE)

TYPE_MASK_POS_YAW = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                     mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

def send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(0, FOLLOWER_SYSID, FOLLOWER_COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_VEL_YAWRATE,
        0, 0, 0, vx, vy, vz_ned, 0, 0, 0, 0, yaw_rate)

def send_position_yaw(master, x, y, z_ned, yaw):
    master.mav.set_position_target_local_ned_send(0, FOLLOWER_SYSID, FOLLOWER_COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, TYPE_MASK_POS_YAW,
        x, y, z_ned, 0, 0, 0, 0, 0, 0, yaw, 0)

def goto_initial_position(master, leader_data: list):
    L0 = leader_data[0]
    dx, dy, dz = compute_offset(L0['yaw'])
    x_target, y_target, z_target = L0['x'] + dx, L0['y'] + dy, L0['z'] + dz
    z_ned, yaw_target = -z_target, L0['yaw']
    
    print(f'\n📌 Pre-posicionamiento: objetivo=({x_target:.2f}, {y_target:.2f}, {z_target:.2f})')
    input('▶️ Presiona ENTER para mover el drone al punto inicial...\n')
    
    dt_pre, t_start, t_in_zone = 1.0/PRE_RATE, time.monotonic(), None
    
    while True:
        elapsed = time.monotonic() - t_start
        send_position_yaw(master, x_target, y_target, z_ned, yaw_target)
        
        S = get_follower()
        if follower_ready(S):
            dist_xy = math.hypot(S['x'] - x_target, S['y'] - y_target)
            dist_z, speed = abs(S['z'] - z_target), math.hypot(S['vx'], S['vy'])
            
            if (dist_xy < PRE_CONV_RADIUS and dist_z < PRE_CONV_Z and speed < PRE_CONV_SPEED):
                if t_in_zone is None: t_in_zone = time.monotonic()
                elif time.monotonic() - t_in_zone >= PRE_CONV_HOLD:
                    print(f'✅ Pre-posicionamiento completado')
                    return True
            else: t_in_zone = None
        
        if elapsed >= PRE_CONV_TIMEOUT:
            print(f'⚠️ Timeout de pre-posicionamiento')
            return False
        time.sleep(dt_pre)

# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE VISUALIZACIÓN (ESTILO MATLAB)
# ──────────────────────────────────────────────────────────────────────────────
C_refL = np.array([0.40, 0.65, 1.00])   # azul claro - setpoint líder
C_refS = np.array([1.00, 0.60, 0.40])   # naranja - setpoint seguidor
C_L    = np.array([0.10, 0.35, 0.75])   # azul oscuro - líder real
C_S    = np.array([0.80, 0.15, 0.10])   # rojo - seguidor real
C_form = np.array([0.50, 0.50, 0.50])   # gris - vectores d
C_dist = np.array([0.25, 0.75, 0.45])   # verde - distancia
LW, LWS = 2.0, 1.2

def _wrap_np(arr): return np.arctan2(np.sin(arr), np.cos(arr))

def transform_for_mirror(x, y):
    x_factor = -1 if INVERTIR_EJE_X else 1
    y_factor = -1 if INVERTIR_EJE_Y else 1
    return x * x_factor, y * y_factor

def draw_drone_frame(ax, x, y, yaw, color_x=[0.85, 0.10, 0.10], color_y=[0.10, 0.65, 0.10], scale=SC_DRONE, lw=FW_ARROW):
    """Dibuja los ejes del drone (rojo para X, verde para Y)"""
    ax.quiver(x, y, scale*np.cos(yaw), scale*np.sin(yaw), angles='xy', scale_units='xy', scale=1,
              color=color_x, width=0.015, headwidth=5, headlength=5, headaxislength=4.5, alpha=0.9)
    ax.quiver(x, y, scale*np.cos(yaw+np.pi/2), scale*np.sin(yaw+np.pi/2), angles='xy', scale_units='xy', scale=1,
              color=color_y, width=0.015, headwidth=5, headlength=5, headaxislength=4.5, alpha=0.9)

def draw_rtk_frame(ax):
    """Dibuja el sistema de coordenadas RTK en el origen (estilo MATLAB)"""
    # Eje X (rojo)
    ax.quiver(0, 0, SC_RTK, 0, angles='xy', scale_units='xy', scale=1,
              color=[0.85, 0.10, 0.10], width=0.020, headwidth=6, headlength=6, alpha=0.9)
    # Eje Y (verde)
    ax.quiver(0, 0, 0, SC_RTK, angles='xy', scale_units='xy', scale=1,
              color=[0.10, 0.65, 0.10], width=0.020, headwidth=6, headlength=6, alpha=0.9)
    # Punto de origen
    ax.plot(0, 0, 'ok', markersize=8, markeredgecolor='k', markerfacecolor='k')
    # Etiquetas
    ax.text(SC_RTK+0.15, 0.05, r'$\hat{x}_{RTK}$', fontsize=12, fontweight='bold')
    ax.text(-0.05, SC_RTK+0.15, r'$\hat{y}_{RTK}$', fontsize=12, fontweight='bold', ha='center')

def plot_results(buf):
    """Genera las 7 figuras estilo MATLAB con frames de drones y sistema RTK"""
    t = np.array(buf['t'])
    xL, yL, zL, psiL = np.array(buf['xl']), np.array(buf['yl']), np.array(buf['zl']), np.array(buf['psiL'])
    xS, yS, zS, psiS = np.array(buf['xs']), np.array(buf['ys']), np.array(buf['zs']), np.array(buf['psiS'])
    xdesL, ydesL, zdesL = np.array(buf['xdesL']), np.array(buf['ydesL']), np.array(buf['zdesL'])
    xdesS, ydesS, zdesS = np.array(buf['xd']), np.array(buf['yd']), np.array(buf['zd'])
    
    psidesS = psiL.copy()
    n = len(t)
    
    # Errores
    ex_L, ey_L, ez_L = np.array(buf['exL']), np.array(buf['eyL']), np.zeros(n)
    epsi_L = np.degrees(_wrap_np(psidesS - psiL))
    ex_S, ey_S, ez_S = np.array(buf['ex']), np.array(buf['ey']), np.array(buf['ez'])
    epsi_S = np.degrees(_wrap_np(psidesS - psiS))
    dist = np.hypot(xL - xS, yL - yS)
    
    # Métricas
    def rms(v): return np.sqrt(np.mean(v**2))
    print('\n============= MÉTRICAS =============')
    print('             Líder       Seguidor')
    print(f'RMS error x: {rms(ex_L):.4f} m   {rms(ex_S):.4f} m')
    print(f'RMS error y: {rms(ey_L):.4f} m   {rms(ey_S):.4f} m')
    print(f'RMS error z: {rms(ez_L):.4f} m   {rms(ez_S):.4f} m')
    print(f'RMS error ψ: {rms(np.radians(epsi_L)):.4f} rad  {rms(np.radians(epsi_S)):.4f} rad')
    print('------------------------------------')
    print(f'Distancia deseada L-S:  {OFFSET_D:.4f} m')
    print(f'Distancia media   L-S:  {np.mean(dist):.4f} m')
    print(f'Distancia máx     L-S:  {np.max(dist):.4f} m')
    print(f'Distancia mín     L-S:  {np.min(dist):.4f} m')
    print(f'Error formación RMS:    {rms(dist - OFFSET_D):.4f} m')
    print('====================================\n')
    
    # ==================== FIGURA 1 - Trayectoria XY completa ====================
    fig1, ax = plt.subplots(figsize=(7.5, 7.5), facecolor='white')
    fig1.suptitle('Trayectoria XY con marcos de referencia y vector de formacion', fontsize=12)
    
    # Aplicar inversión de ejes
    xdesL_t, ydesL_t = transform_for_mirror(xdesL, ydesL)
    xdesS_t, ydesS_t = transform_for_mirror(xdesS, ydesS)
    xL_t, yL_t = transform_for_mirror(xL, yL)
    xS_t, yS_t = transform_for_mirror(xS, yS)
    xL0_t, yL0_t = transform_for_mirror(np.array([xL[0]]), np.array([yL[0]]))
    xS0_t, yS0_t = transform_for_mirror(np.array([xS[0]]), np.array([yS[0]]))
    
    # Setpoints y trayectorias
    ax.plot(xdesL_t, ydesL_t, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax.plot(xdesS_t, ydesS_t, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax.plot(xL_t, yL_t, '-', color=C_L, lw=LW, label='Líder real')
    ax.plot(xS_t, yS_t, '-', color=C_S, lw=LW, label='Seguidor real')
    ax.plot(xL0_t, yL0_t, 'o', ms=10, mfc=C_L, mec='k', label='Inicio L')
    ax.plot(xS0_t, yS0_t, 'o', ms=10, mfc=C_S, mec='k', label='Inicio S')
    
    # Vectores d (líneas de formación)
    n_lin = 14
    idxs = np.round(np.linspace(0, n-1, n_lin+2)).astype(int)[1:-1]
    for i in idxs:
        xi_t, yi_t = transform_for_mirror(np.array([xL[i]]), np.array([yL[i]]))
        xj_t, yj_t = transform_for_mirror(np.array([xS[i]]), np.array([yS[i]]))
        ax.plot([xi_t[0], xj_t[0]], [yi_t[0], yj_t[0]], '-', color=C_form, lw=1.2, alpha=0.7)
    
    # Frames de los drones en múltiples instantes
    n_frames = 5
    idx_frames = np.round(np.linspace(0, n-1, n_frames+1)).astype(int)[:-1]
    for idx in idx_frames:
        # Transformar coordenadas para este frame
        xLk_t, yLk_t = transform_for_mirror(np.array([xL[idx]]), np.array([yL[idx]]))
        xSk_t, ySk_t = transform_for_mirror(np.array([xS[idx]]), np.array([yS[idx]]))
        psiLk, psiSk = psiL[idx], psiS[idx]
        
        # Nota: Los ángulos de yaw NO se invierten porque son orientaciones relativas
        draw_drone_frame(ax, xLk_t[0], yLk_t[0], psiLk, scale=SC_DRONE, lw=FW_ARROW)
        draw_drone_frame(ax, xSk_t[0], ySk_t[0], psiSk, scale=SC_DRONE, lw=FW_ARROW)
    
    # Sistema de coordenadas RTK en el origen
    draw_rtk_frame(ax)
    
    # Entradas de leyenda para los ejes
    ax.plot([], [], '-', color=[0.85, 0.10, 0.10], lw=2.5, label=r'$\hat{x}_D$')
    ax.plot([], [], '-', color=[0.10, 0.65, 0.10], lw=2.5, label=r'$\hat{y}_D$')
    ax.plot([], [], '-', color=[0.45, 0.45, 0.45], lw=2.0, label='Vector $\\mathbf{d}$')
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_{RTK}$ [m]', fontsize=11)
    ax.set_ylabel('$y_{RTK}$ [m]', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    fig1.tight_layout()
    
    # ==================== FIGURA 2 - Líder: setpoint vs real ====================
    fig2, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.plot(xdesL, ydesL, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax.plot(xL, yL, '-', color=C_L, lw=LW, label='Líder real')
    ax.plot(xL[0], yL[0], 'o', ms=10, mfc=C_L, mec='k', label='Inicio')
    draw_rtk_frame(ax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Líder: Setpoint vs Trayectoria Real')
    ax.legend(loc='best')
    fig2.tight_layout()
    
    # ==================== FIGURA 3 - Seguidor: setpoint vs real ==================
    fig3, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.plot(xdesS, ydesS, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax.plot(xS, yS, '-', color=C_S, lw=LW, label='Seguidor real')
    ax.plot(xS[0], yS[0], 'o', ms=10, mfc=C_S, mec='k', label='Inicio')
    draw_rtk_frame(ax)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Seguidor: Setpoint vs Trayectoria Real')
    ax.legend(loc='best')
    fig3.tight_layout()
    
    # ==================== FIGURA 4 - Estados vs Tiempo ===========================
    fig4, axes = plt.subplots(4, 1, figsize=(9.5, 7.5), sharex=True, facecolor='white')
    fig4.suptitle('Posición y orientación vs Tiempo', fontsize=12)
    
    axes[0].plot(t, xdesL, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axes[0].plot(t, xdesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axes[0].plot(t, xL, '-', color=C_L, lw=LW, label='x Líder')
    axes[0].plot(t, xS, '-', color=C_S, lw=LW, label='x Seguidor')
    axes[0].set_ylabel('x [m]'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', ncol=2, fontsize=8)
    axes[0].set_title('Posición X vs Tiempo')
    
    axes[1].plot(t, ydesL, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axes[1].plot(t, ydesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axes[1].plot(t, yL, '-', color=C_L, lw=LW, label='y Líder')
    axes[1].plot(t, yS, '-', color=C_S, lw=LW, label='y Seguidor')
    axes[1].set_ylabel('y [m]'); axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', ncol=2, fontsize=8)
    axes[1].set_title('Posición Y vs Tiempo')
    
    axes[2].plot(t, zdesL, '--', color=C_refL, lw=LWS, label='Setpoint L')
    axes[2].plot(t, zdesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axes[2].plot(t, zL, '-', color=C_L, lw=LW, label='z Líder')
    axes[2].plot(t, zS, '-', color=C_S, lw=LW, label='z Seguidor')
    axes[2].set_ylabel('z [m]'); axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', ncol=2, fontsize=8)
    axes[2].set_title('Posición Z (Altitud) vs Tiempo')
    
    axes[3].plot(t, np.degrees(psidesS), '--', color=C_refS, lw=LWS, label='Setpoint S')
    axes[3].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW, label='ψ Líder')
    axes[3].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW, label='ψ Seguidor')
    axes[3].set_ylabel('ψ [°]'); axes[3].set_xlabel('Tiempo [s]'); axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='best', ncol=2, fontsize=8)
    axes[3].set_title('Yaw vs Tiempo')
    
    fig4.tight_layout()
    
    # ==================== FIGURA 5 - Errores del líder ===========================
    fig5, axes = plt.subplots(4, 1, figsize=(9.5, 7), sharex=True, facecolor='white')
    fig5.suptitle('Errores del Líder respecto a su setpoint', fontsize=12)
    for ax, data, lbl in zip(axes, [ex_L, ey_L, ez_L, epsi_L],
                              ['$e_x$ [m]', '$e_y$ [m]', '$e_z$ [m]', '$e_\\psi$ [°]']):
        ax.plot(t, data, '-', color=C_L, lw=LW)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
    axes[3].set_xlabel('Tiempo [s]')
    fig5.tight_layout()
    
    # ==================== FIGURA 6 - Errores del seguidor ========================
    fig6, axes = plt.subplots(4, 1, figsize=(9.5, 7), sharex=True, facecolor='white')
    fig6.suptitle('Errores del Seguidor respecto a su setpoint', fontsize=12)
    for ax, data, lbl in zip(axes, [ex_S, ey_S, ez_S, epsi_S],
                              ['$e_x$ [m]', '$e_y$ [m]', '$e_z$ [m]', '$e_\\psi$ [°]']):
        ax.plot(t, data, '-', color=C_S, lw=LW)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
    axes[3].set_xlabel('Tiempo [s]')
    fig6.tight_layout()
    
    # ==================== FIGURA 7 - Distancia de formación ======================
    fig7, ax = plt.subplots(figsize=(9.5, 3.8), facecolor='white')
    ax.plot(t, dist, '-', color=C_dist, lw=LW, label='Distancia real L-S')
    ax.axhline(OFFSET_D, color=[0.0, 0.5, 0.2], ls='--', lw=1.2, label=f'd = {OFFSET_D:.1f} m')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Distancia [m]')
    ax.set_title('Distancia entre Líder y Seguidor vs Tiempo')
    ax.grid(True, alpha=0.3); ax.legend(loc='best')
    fig7.tight_layout()
    
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f'\n🔄 Configuración de inversión de ejes: X={INVERTIR_EJE_X}, Y={INVERTIR_EJE_Y}')
    
    leader_data, _ = load_leader_csv(LEADER_CSV)
    duration = leader_data[-1]['t']
    
    print(f'🔗 Conectando a {CONN} ...')
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    
    stop_reader = threading.Event()
    reader_thread = threading.Thread(target=_mavlink_reader, args=(master, stop_reader), daemon=True)
    reader_thread.start()
    
    print('⏳ Esperando telemetría...', end='', flush=True)
    while not follower_ready(get_follower()): time.sleep(0.02)
    print(' listo.')
    
    if PRE_POSITION:
        goto_initial_position(master, leader_data)
        input('\n▶️ Iniciar controlador? Presiona ENTER...\n')
    else:
        input('\n▶️ Drone listo? Presiona ENTER...\n')
    
    out_csv_path = f'lf_log_{int(time.time())}.csv'
    out_csv_f = open(out_csv_path, 'w', newline='')
    out_csv_w = csv.writer(out_csv_f)
    out_csv_w.writerow(CSV_HEADER)
    print(f'💾 Log: {out_csv_path}')
    
    plot_buf = dict(t=[], xl=[], yl=[], zl=[], psiL=[], xs=[], ys=[], zs=[], psiS=[],
                    xdesL=[], ydesL=[], zdesL=[], exL=[], eyL=[], xd=[], yd=[], zd=[], ex=[], ey=[], ez=[])
    
    pid, idx_hint, t_start, t_log = PIDState(), 0, time.monotonic(), 0.0
    
    print('🚀 Control iniciado. Ctrl+C para detener.\n')
    
    try:
        next_t = time.monotonic()
        while True:
            elapsed = (time.monotonic() - t_start) * SPEED_FACTOR
            L, idx_hint = interpolate_leader(leader_data, elapsed, idx_hint)
            S = get_follower()
            
            if follower_ready(S):
                vx, vy, vz_ned, yaw_rate_cmd, c = compute_control(L, S, pid)
                send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate_cmd)
                t_log += DT
                
                out_csv_w.writerow([f'{t_log:.4f}', L['x'], L['y'], L['z'], L['vx'], L['vy'], L['vz'],
                                    L['yaw'], L['yaw_rate'], L['x_sp'], L['y_sp'], L['z_sp'], L['ex_L'], L['ey_L'],
                                    S['x'], S['y'], S['z'], S['vx'], S['vy'], S['vz'], S['yaw'], S['yaw_rate'],
                                    c['xd'], c['yd'], c['zd'], c['ex'], c['ey'], c['ez'], c['e_yaw'],
                                    c['ff_x'], c['ff_y'], c['ff_z'], c['vx_p'], c['vy_p'], c['vz_p'],
                                    c['vx_i'], c['vy_i'], c['vz_i'], c['vx_d'], c['vy_d'], c['vz_d'],
                                    c['vx'], c['vy'], c['vz'], yaw_rate_cmd])
                
                for key, val in [('t', t_log), ('xl', L['x']), ('yl', L['y']), ('zl', L['z']), ('psiL', L['yaw']),
                                 ('xs', S['x']), ('ys', S['y']), ('zs', S['z']), ('psiS', S['yaw']),
                                 ('xdesL', L['x_sp']), ('ydesL', L['y_sp']), ('zdesL', L['z_sp']),
                                 ('exL', L['ex_L']), ('eyL', L['ey_L']),
                                 ('xd', c['xd']), ('yd', c['yd']), ('zd', c['zd']),
                                 ('ex', c['ex']), ('ey', c['ey']), ('ez', c['ez'])]:
                    plot_buf[key].append(val)
                
                if int(t_log) % 5 == 0 and int(t_log - DT) % 5 != 0:
                    print(f't={t_log:.1f}s | err_xy={math.hypot(c["ex"], c["ey"]):.3f}m')
                
                if elapsed >= duration: break
            
            next_t += DT
            time.sleep(max(0, next_t - time.monotonic()))
    
    except KeyboardInterrupt:
        print('\n⏹️ Interrupción.')
    finally:
        send_velocity_yawrate(master, 0, 0, 0, 0)
        time.sleep(0.2)
        stop_reader.set()
        reader_thread.join(timeout=2)
        out_csv_f.close()
        master.close()
        print(f'💾 Log: {out_csv_path}')
        plot_results(plot_buf)

if __name__ == '__main__':
    main()
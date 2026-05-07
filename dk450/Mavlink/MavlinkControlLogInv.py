#!/usr/bin/env python3
"""
MavlinkControlLFLog.py — Controlador líder-seguidor con líder virtual desde CSV
================================================================================
Lee el CSV generado por MavlinkCirculoLog.py e interpola la trayectoria del
líder en tiempo real. Aplica la misma ley de control de MavlinkControlLF.py:

    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L - v_S)

donde el offset es polar rotante:
    d(t) = [D·cos(ψ_L + α),  D·sin(ψ_L + α),  Δz]
    ḋ(t) = ψ̇_L · Rz(π/2) · d    ← término de prealimentación

El drone físico sigue una trayectoria similar al líder pero desplazada según
(OFFSET_D, OFFSET_ALPHA, OFFSET_DZ). Esto simula el escenario líder-seguidor
usando un solo drone real + el log del líder.

OFFSET_ALPHA = π    → el seguidor va DETRÁS del líder
OFFSET_ALPHA = π/2  → a la izquierda
OFFSET_ALPHA = 0    → delante (formación de avanzada)
etc.

Uso:
  python3 MavlinkControlLFLog.py circulo_log_1777151606.csv
"""

import math
import time
import csv
import threading
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
CONN           = 'udp:127.0.0.1:14552'
FOLLOWER_SYSID  = 1
FOLLOWER_COMPID = 1

# ── Offset polar (ec. offset_polar) ──────────────────────────────────────────
# d(t) = [D·cos(ψ_L + α),  D·sin(ψ_L + α),  Δz]
# El vector rota solidariamente con el yaw del líder.
OFFSET_D     = 1.0          # metros de separación horizontal
OFFSET_ALPHA = -math.pi/2     # ángulo respecto al eje del líder [rad]
                             #   π   = detrás   (formación clásica)
                             #   π/2 = izquierda
                             #   0   = delante
OFFSET_DZ    = 1.0          # diferencia de altitud [m] (positivo = más alto)

# ── Ganancias PID posición ────────────────────────────────────────────────────
KP   = 1.0
KI   = 0.00
KD   = 0.0

# ── Ganancias PID yaw ─────────────────────────────────────────────────────────
KP_YAW = 2.0
KI_YAW = 0.0
KD_YAW = 0.0

# ── Anti-windup ───────────────────────────────────────────────────────────────
INTEGRAL_LIMIT     = 2.0    # m·s por eje
INTEGRAL_YAW_LIMIT = 1.0    # rad·s

# ── Límites de velocidad ──────────────────────────────────────────────────────
V_MAX        = 3.0          # m/s
YAW_RATE_MAX = 1.0          # rad/s

# ── Frecuencia de control ─────────────────────────────────────────────────────
RATE = 20                   # Hz
DT   = 1.0 / RATE

# ── Replay ────────────────────────────────────────────────────────────────────
SPEED_FACTOR = 1.0          # < 1.0 = más lento, > 1.0 = más rápido

# ── Archivo de log del líder (generado por MavlinkCirculoLog.py) ──────────────
LEADER_CSV = 'circulo_log_1778128773.csv' \
''   # ← CAMBIAR AQUÍ

# ── Pre-posicionamiento (ir al primer setpoint antes de iniciar el control) ───
#
#   PRE_POSITION = True   → el drone se desplaza al primer punto de la
#                           trayectoria (posición del líder t=0 + offset)
#                           usando control de POSICIÓN de ArduPilot, espera
#                           a que converja y luego pide un segundo ENTER para
#                           arrancar el controlador de velocidad.
#
#   PRE_POSITION = False  → comportamiento original: un solo ENTER y arranca
#                           el controlador desde donde esté el drone.
#
PRE_POSITION        = True   # activar / desactivar pre-posicionamiento

# Radio de convergencia para considerar que el drone llegó al punto inicial [m]
PRE_CONV_RADIUS     = 0.20   # m  (horizontal)
PRE_CONV_Z          = 0.15   # m  (vertical)
# Velocidad máxima para considerar que el drone está quieto [m/s]
PRE_CONV_SPEED      = 0.10   # m/s
# Segundos consecutivos dentro del criterio para confirmar convergencia
PRE_CONV_HOLD       = 1.5    # s
# Tiempo máximo esperando la convergencia antes de advertir y continuar [s]
PRE_CONV_TIMEOUT    = 30.0   # s
# Frecuencia de envío del setpoint de posición durante el pre-posicionamiento
PRE_RATE            = 10     # Hz

# ── Máscaras MAVLink ──────────────────────────────────────────────────────────
TYPE_MASK_VEL_YAWRATE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

# Máscara para control de POSICIÓN + YAW (pre-posicionamiento)
TYPE_MASK_POS_YAW = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# ── CONFIGURACIÓN DE INVERSIÓN DE EJES PARA LA FIGURA 1 ──────────────────────
# Activar/desactivar la inversión de cada eje independientemente
INVERTIR_EJE_X = True   # True = invertir (derecha ↔ izquierda)
INVERTIR_EJE_Y = True   # True = invertir (arriba ↔ abajo)
# Nota: La transformación aplicada es:
#   x_transformado = x * (-1 if INVERTIR_EJE_X else 1)
#   y_transformado = y * (-1 if INVERTIR_EJE_Y else 1)

# ── CSV de salida ─────────────────────────────────────────────────────────────
CSV_HEADER = [
    'time',
    'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    'xd', 'yd', 'zd',
    'ex', 'ey', 'ez', 'e_yaw',
    'ff_x', 'ff_y', 'ff_z',
    'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i',
    'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd',
    'yaw_rate_cmd',
]


# ──────────────────────────────────────────────────────────────────────────────
# CARGA E INTERPOLACIÓN DEL CSV (líder virtual)
# ──────────────────────────────────────────────────────────────────────────────
def load_leader_csv(path: str) -> list:
    """
    Carga el CSV y devuelve lista de dicts.
    Siempre garantiza las claves x_sp, y_sp, z_sp, ex_L, ey_L:
      - Si el CSV las tiene, se usan directamente.
      - Si no (formato antiguo), se genera fallback: setpoint = posición real, error = 0.
    """
    _SP_COLS = ('x_sp', 'y_sp', 'z_sp', 'ex_L', 'ey_L')
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        has_sp = all(c in (reader.fieldnames or []) for c in _SP_COLS)
        if not has_sp:
            print('⚠️  CSV sin columnas de setpoint (x_sp, y_sp, z_sp, ex_L, ey_L).')
            print('   Usando posición real del líder como fallback del setpoint.')
        for row in reader:
            d = {k: float(v) for k, v in row.items()}
            if not has_sp:
                d['x_sp'] = d['x']
                d['y_sp'] = d['y']
                d['z_sp'] = d['z']
                d['ex_L'] = 0.0
                d['ey_L'] = 0.0
            rows.append(d)
    if not rows:
        raise ValueError(f'CSV vacío: {path}')
    print(f'📂 CSV cargado: {len(rows)} muestras, '
          f'duración={rows[-1]["t"]:.1f} s  '
          f'[setpoint: {"✓" if has_sp else "fallback"}]')
    return rows


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_yaw(y0, y1, t):
    diff = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
    return y0 + t * diff


def interpolate_leader(data: list, t_csv: float, idx: int) -> tuple:
    """
    Interpola el estado del líder para el tiempo t_csv.
    Devuelve (dict_estado, nuevo_idx).
    Búsqueda incremental — O(1) amortizado.
    """
    if t_csv <= data[0]['t']:
        return dict(data[0]), 0
    if t_csv >= data[-1]['t']:
        return dict(data[-1]), len(data) - 2

    while idx < len(data) - 2 and data[idx + 1]['t'] < t_csv:
        idx += 1

    r0, r1 = data[idx], data[idx + 1]
    dt_seg = r1['t'] - r0['t']
    alpha  = (t_csv - r0['t']) / dt_seg if dt_seg > 1e-9 else 0.0

    return {
        't':        t_csv,
        'x':        lerp(r0['x'],        r1['x'],        alpha),
        'y':        lerp(r0['y'],        r1['y'],        alpha),
        'z':        lerp(r0['z'],        r1['z'],        alpha),
        'vx':       lerp(r0['vx'],       r1['vx'],       alpha),
        'vy':       lerp(r0['vy'],       r1['vy'],       alpha),
        'vz':       lerp(r0['vz'],       r1['vz'],       alpha),
        'yaw':      lerp_yaw(r0['yaw'],  r1['yaw'],      alpha),
        'yaw_rate': lerp(r0['yaw_rate'], r1['yaw_rate'], alpha),
        'x_sp':     lerp(r0['x_sp'],     r1['x_sp'],     alpha),
        'y_sp':     lerp(r0['y_sp'],     r1['y_sp'],     alpha),
        'z_sp':     lerp(r0['z_sp'],     r1['z_sp'],     alpha),
        'ex_L':     lerp(r0['ex_L'],     r1['ex_L'],     alpha),
        'ey_L':     lerp(r0['ey_L'],     r1['ey_L'],     alpha),
    }, idx


# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DEL SEGUIDOR — hilo lector MAVLink
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_follower = dict(x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 yaw=None, yaw_rate=None)


def _mavlink_reader(master, stop_event):
    """Lee LOCAL_POSITION_NED y ATTITUDE del drone seguidor."""
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        if msg.get_srcSystem() != FOLLOWER_SYSID:
            continue

        mtype = msg.get_type()
        with _state_lock:
            if mtype == 'LOCAL_POSITION_NED':
                _follower['x']  = msg.x
                _follower['y']  = msg.y
                _follower['z']  = -msg.z
                _follower['vx'] = msg.vx
                _follower['vy'] = msg.vy
                _follower['vz'] = -msg.vz
            elif mtype == 'ATTITUDE':
                _follower['yaw']      = msg.yaw
                _follower['yaw_rate'] = msg.yawspeed


def get_follower():
    with _state_lock:
        return dict(_follower)


def follower_ready(s):
    return all(v is not None for v in s.values())


# ──────────────────────────────────────────────────────────────────────────────
# MATEMÁTICAS: OFFSET POLAR + PREALIMENTACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_offset(psi_L):
    """
    d(t) = [D·cos(ψ_L + α),  D·sin(ψ_L + α),  Δz]
    El offset rota con el yaw del líder.
    """
    angle = psi_L + OFFSET_ALPHA
    return (OFFSET_D * math.cos(angle),
            OFFSET_D * math.sin(angle),
            OFFSET_DZ)


def compute_offset_dot(yaw_rate_L, dx, dy):
    """
    ḋ = ψ̇_L · Rz(π/2) · d = ψ̇_L · [-dy, dx, 0]
    Término de prealimentación que compensa el giro del líder.
    """
    return (yaw_rate_L * (-dy),
            yaw_rate_L * ( dx),
            0.0)


def clamp(val, limit):
    return max(-limit, min(limit, val))


# ──────────────────────────────────────────────────────────────────────────────
# LEY DE CONTROL
# ──────────────────────────────────────────────────────────────────────────────
class PIDState:
    def __init__(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0


def compute_control(L: dict, S: dict, pid: PIDState) -> tuple:
    """
    Calcula v_cmd y yaw_rate_cmd según la ley de control PID con prealimentación.

    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L - v_S)

    Retorna: (vx, vy, vz_ned, yaw_rate_cmd, components_dict)
    """
    # 1. Offset polar rotante
    dx, dy, dz = compute_offset(L['yaw'])

    # 2. Posición deseada del seguidor
    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz

    # 3. Error de posición
    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']

    # 4. Integrador con anti-windup
    pid.integral[0] = clamp(pid.integral[0] + ex * DT, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * DT, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * DT, INTEGRAL_LIMIT)

    # 5. Derivativa: diferencia de velocidades (sin ruido de diferenciación numérica)
    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    # 6. Prealimentación (feedforward)
    ff_x, ff_y, ff_z = compute_offset_dot(L['yaw_rate'], dx, dy)

    # 7. Ley de control posición
    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z   # positivo arriba

    # Clamp horizontal
    v_horiz = math.hypot(vx, vy)
    if v_horiz > V_MAX:
        vx *= V_MAX / v_horiz
        vy *= V_MAX / v_horiz
    vz     = clamp(vz, V_MAX)
    vz_ned = -vz   # convertir a NED

    # 8. Control de yaw
    e_yaw = wrap(L['yaw'] - S['yaw'])
    pid.integral_yaw = clamp(pid.integral_yaw + e_yaw * DT, INTEGRAL_YAW_LIMIT)
    dyaw   = L['yaw_rate'] - S['yaw_rate']
    yaw_rate_cmd = clamp(
        KP_YAW * e_yaw + KI_YAW * pid.integral_yaw + KD_YAW * dyaw,
        YAW_RATE_MAX
    )

    c = dict(
        xd=xd, yd=yd, zd=zd,
        ex=ex, ey=ey, ez=ez, e_yaw=e_yaw,
        ff_x=ff_x, ff_y=ff_y, ff_z=ff_z,
        vx_p=KP*ex,           vy_p=KP*ey,           vz_p=KP*ez,
        vx_i=KI*pid.integral[0], vy_i=KI*pid.integral[1], vz_i=KI*pid.integral[2],
        vx_d=KD*dv_x,         vy_d=KD*dv_y,         vz_d=KD*dv_z,
        vx=vx, vy=vy, vz=vz,
    )
    return vx, vy, vz_ned, yaw_rate_cmd, c


# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE COMANDOS
# ──────────────────────────────────────────────────────────────────────────────
def send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0, FOLLOWER_SYSID, FOLLOWER_COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz_ned,
        0, 0, 0,
        0, yaw_rate
    )


def send_position_yaw(master, x, y, z_ned, yaw):
    """Envía setpoint de POSICIÓN + YAW en marco LOCAL_NED."""
    master.mav.set_position_target_local_ned_send(
        0, FOLLOWER_SYSID, FOLLOWER_COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z_ned,
        0, 0, 0,
        0, 0, 0,
        yaw, 0
    )


def goto_initial_position(master, leader_data: list):
    """
    Mueve el drone seguidor al primer setpoint de la trayectoria
    usando control de posición de ArduPilot.

    El punto objetivo se calcula igual que compute_control() en t=0:
      posición_deseada = L(t=0).pos + offset( L(t=0).yaw )

    Esto garantiza que cuando el controlador de velocidad arranque en t=0,
    el error inicial sea prácticamente cero.

    Solo se ejecuta si PRE_POSITION = True.
    Retorna True si convergió, False si se agotó el timeout.
    """
    # ── Punto inicial: idéntico a lo que compute_control() calculará en t=0 ──
    # Se usa la primera muestra del CSV (estado real del líder en t=0).
    L0 = leader_data[0]

    dx, dy, dz = compute_offset(L0['yaw'])   # misma función que usa el control
    x_target   = L0['x'] + dx
    y_target   = L0['y'] + dy
    z_target   = L0['z'] + dz     # altitud positiva (ENU)
    z_ned      = -z_target         # NED para MAVLink
    yaw_target = L0['yaw']        # el seguidor alinea su yaw con el líder en t=0

    print(f'\n📌 Pre-posicionamiento activado.')
    print(f'   Líder t=0 : x={L0["x"]:.2f}  y={L0["y"]:.2f}  '
          f'z={L0["z"]:.2f}  yaw={math.degrees(L0["yaw"]):.1f}°')
    print(f'   Offset    : dx={dx:.2f}  dy={dy:.2f}  dz={dz:.2f}  '
          f'(D={OFFSET_D} m, α={math.degrees(OFFSET_ALPHA):.0f}°)')
    print(f'   Objetivo  : x={x_target:.2f}  y={y_target:.2f}  '
          f'z={z_target:.2f}  yaw={math.degrees(yaw_target):.1f}°')
    print(f'   Criterio  : dist_xy<{PRE_CONV_RADIUS} m | '
          f'dist_z<{PRE_CONV_Z} m | vel<{PRE_CONV_SPEED} m/s | '
          f'hold {PRE_CONV_HOLD} s')
    input('▶️  Presiona ENTER para mover el drone al punto inicial...\n')

    dt_pre    = 1.0 / PRE_RATE
    t_start   = time.monotonic()
    t_in_zone = None
    converged = False

    while True:
        now    = time.monotonic()
        elapsed = now - t_start

        # Enviar setpoint de posición repetidamente (ArduPilot lo requiere)
        send_position_yaw(master, x_target, y_target, z_ned, yaw_target)

        S = get_follower()
        if follower_ready(S):
            dist_xy = math.hypot(S['x'] - x_target, S['y'] - y_target)
            dist_z  = abs(S['z'] - z_target)
            speed   = math.hypot(S['vx'], S['vy'])

            # Progreso por consola cada 2 s
            if int(elapsed) % 2 == 0 and int(elapsed - dt_pre) % 2 != 0:
                print(f'   t={elapsed:.1f}s | '
                      f'dist_xy={dist_xy:.2f} m  dist_z={dist_z:.2f} m  '
                      f'vel={speed:.2f} m/s')

            if dist_xy < PRE_CONV_RADIUS and dist_z < PRE_CONV_Z \
                    and speed < PRE_CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= PRE_CONV_HOLD:
                    print(f'✅ Pre-posicionamiento completo: '
                          f'dist_xy={dist_xy:.3f} m  '
                          f'dist_z={dist_z:.3f} m  vel={speed:.3f} m/s')
                    converged = True
                    break
            else:
                t_in_zone = None

        if elapsed >= PRE_CONV_TIMEOUT:
            print(f'⚠️  Timeout de pre-posicionamiento ({PRE_CONV_TIMEOUT:.0f} s). '
                  f'El drone no convergió completamente.')
            break

        time.sleep(dt_pre)

    return converged


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS — estructura idéntica al script MATLAB de Simulink
# ──────────────────────────────────────────────────────────────────────────────

# Paleta idéntica al MATLAB
C_refL = np.array([0.40, 0.65, 1.00])   # azul claro  - setpoint líder
C_refS = np.array([1.00, 0.60, 0.40])   # naranja claro - setpoint seguidor
C_L    = np.array([0.10, 0.35, 0.75])   # azul oscuro  - líder real
C_S    = np.array([0.80, 0.15, 0.10])   # rojo oscuro  - seguidor real
C_form = np.array([0.50, 0.50, 0.50])   # gris - líneas de formación / vectores d
C_dist = np.array([0.25, 0.75, 0.45])   # verde - distancia real
LW  = 2.0
LWS = 1.2

# Parámetros de frames
SC_DRONE = 0.25      # longitud de los ejes del frame de cada drone
SC_RTK   = SC_DRONE * 2.0   # longitud del frame RTK (más grande)
FW_ARROW = 1.5       # linewidth flechas de drones
FW_RTK   = FW_ARROW + 0.8   # linewidth flechas RTK


def _wrap_np(arr):
    """Vectorized wrap to (-π, π]."""
    return np.arctan2(np.sin(arr), np.cos(arr))


def transform_for_mirror(x, y):
    """Aplica la transformación de espejo según las flags de configuración."""
    x_factor = -1 if INVERTIR_EJE_X else 1
    y_factor = -1 if INVERTIR_EJE_Y else 1
    return x * x_factor, y * y_factor


def _draw_drone_frame(ax, x, y, yaw):
    """
    Dibuja los dos ejes del frame de orientación de un drone en (x, y).
    Rojo = x̂_D (frente del drone), Verde = ŷ_D (lateral).
    Los componentes del vector YA están en coordenadas del plot (post-espejo),
    así que se pasan directamente a quiver sin transformar el ángulo.
    """
    sx = -1 if INVERTIR_EJE_X else 1
    sy = -1 if INVERTIR_EJE_Y else 1
    # Eje x̂ (frente)
    ax.quiver(x, y, sx * SC_DRONE * np.cos(yaw), sy * SC_DRONE * np.sin(yaw),
              angles='xy', scale_units='xy', scale=1,
              color=[0.85, 0.10, 0.10], width=0.007,
              headwidth=4, headlength=4, headaxislength=3.5)
    # Eje ŷ (lateral)
    ax.quiver(x, y, sx * SC_DRONE * np.cos(yaw + np.pi/2), sy * SC_DRONE * np.sin(yaw + np.pi/2),
              angles='xy', scale_units='xy', scale=1,
              color=[0.10, 0.65, 0.10], width=0.007,
              headwidth=4, headlength=4, headaxislength=3.5)


def _draw_rtk_frame(ax):
    """
    Dibuja el sistema de coordenadas RTK en el origen (0, 0).
    Los ejes apuntan en la dirección física positiva, ajustados al espejo activo.
    """
    xf = -1 if INVERTIR_EJE_X else 1
    yf = -1 if INVERTIR_EJE_Y else 1
    ax.quiver(0, 0, SC_RTK * xf, 0, angles='xy', scale_units='xy', scale=1,
              color=[0.85, 0.10, 0.10], width=0.010,
              headwidth=5, headlength=5, headaxislength=4.5)
    ax.quiver(0, 0, 0, SC_RTK * yf, angles='xy', scale_units='xy', scale=1,
              color=[0.10, 0.65, 0.10], width=0.010,
              headwidth=5, headlength=5, headaxislength=4.5)
    ax.plot(0, 0, 'ok', ms=6, mfc='k', mec='k')
    ax.text(SC_RTK * xf + 0.06 * xf, 0.02 * yf, r'$\hat{x}_{RTK}$',
            fontsize=11, fontweight='bold', color=[0.85, 0.10, 0.10])
    ax.text(0.02 * xf, SC_RTK * yf + 0.07 * yf, r'$\hat{y}_{RTK}$',
            fontsize=11, fontweight='bold', color=[0.10, 0.65, 0.10], ha='center')


def plot_results(buf):
    """
    Genera las 7 figuras idénticas al script MATLAB de Simulink.

    Errores del líder: leídos de buf['exL'] / buf['eyL'], que vienen
    directamente del CSV de MavlinkCirculoLog (x_sp - x_real grabado
    muestra a muestra). NO se recalculan como xdesL - xL, ya que xdesL
    en el buf es la posición real del líder usada como proxy, no el
    setpoint geométrico exacto del círculo.

    Fig 1 — Trayectoria XY completa con:
             · inversión de ejes configurable (INVERTIR_EJE_X/Y)
             · líneas de formación L→S distribuidas (estilo MATLAB)
             · frames de orientación de ambos drones (quiver, 6 instantes)
             · frame RTK en el origen
    Fig 2 — Líder: setpoint vs real + frame RTK
    Fig 3 — Seguidor: setpoint vs real + frame RTK
    Fig 4 — Estados x, y, z, ψ vs tiempo (4 subplots)
    Fig 5 — Errores del líder (4 subplots)
    Fig 6 — Errores del seguidor (4 subplots)
    Fig 7 — Distancia de formación vs tiempo
    """
    t = np.array(buf['t'])

    xL   = np.array(buf['xl']);   yL  = np.array(buf['yl'])
    zL   = np.array(buf['zl']);   psiL = np.array(buf['psiL'])
    xS   = np.array(buf['xs']);   yS  = np.array(buf['ys'])
    zS   = np.array(buf['zs']);   psiS = np.array(buf['psiS'])

    xdesL  = np.array(buf['xdesL']); ydesL = np.array(buf['ydesL'])
    zdesL  = np.array(buf.get('zdesL', buf['zl']))  # fallback: altitud líder
    xdesS  = np.array(buf['xd']);    ydesS = np.array(buf['yd'])
    zdesS  = np.array(buf['zd'])
    psidesS = psiL.copy()

    n = len(t)

    # ── Errores ───────────────────────────────────────────────────────────────
    # Líder: leídos del CSV del círculo — son x_sp−x calculados muestra a muestra
    # con el setpoint geométrico exacto del arco circular, no xdesL−xL.
    ex_L   = np.array(buf['exL'])   # x_sp − x_real  (grabado en MavlinkCirculoLog)
    ey_L   = np.array(buf['eyL'])   # y_sp − y_real
    ez_L   = np.zeros(n)            # 0 por construcción (altitud constante)
    epsi_L = np.degrees(_wrap_np(psidesS - psiL))

    # Seguidor: calculados en tiempo real por compute_control
    ex_S   = np.array(buf['ex'])
    ey_S   = np.array(buf['ey'])
    ez_S   = np.array(buf['ez'])
    epsi_S = np.degrees(_wrap_np(psidesS - psiS))

    # ── Distancia real L-S ────────────────────────────────────────────────────
    dist = np.hypot(xL - xS, yL - yS)

    # ── Métricas (idénticas al fprintf de MATLAB) ─────────────────────────────
    def rms(v):
        return np.sqrt(np.mean(v**2))

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

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 1 — Trayectoria XY: espejo + vectores d + frames drones + RTK
    # ─────────────────────────────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # Pre-calcular todos los arrays transformados
    xdesL_t, ydesL_t = transform_for_mirror(xdesL, ydesL)
    xdesS_t, ydesS_t = transform_for_mirror(xdesS, ydesS)
    xL_t, yL_t       = transform_for_mirror(xL, yL)
    xS_t, yS_t       = transform_for_mirror(xS, yS)
    xL0_t, yL0_t     = transform_for_mirror(np.array([xL[0]]), np.array([yL[0]]))
    xS0_t, yS0_t     = transform_for_mirror(np.array([xS[0]]), np.array([yS[0]]))

    # Setpoints y trayectorias reales
    ax.plot(xdesL_t, ydesL_t, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax.plot(xdesS_t, ydesS_t, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax.plot(xL_t, yL_t, '-',  color=C_L,    lw=LW,  label='Líder real')
    ax.plot(xS_t, yS_t, '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax.plot(xL0_t, yL0_t, 'o', ms=10, mfc=C_L, mec='k', label='Inicio L')
    ax.plot(xS0_t, yS0_t, 'o', ms=10, mfc=C_S, mec='k', label='Inicio S')

    # Vectores d: flechas L→S con cabeza (igual que MATLAB quiver)
    n_d  = 24
    idxd = np.round(np.linspace(0, n - 1, n_d + 2)).astype(int)[1:-1]
    for i in idxd:
        ax.quiver(xL_t[i], yL_t[i],
                  xS_t[i] - xL_t[i], yS_t[i] - yL_t[i],
                  angles='xy', scale_units='xy', scale=1,
                  color=C_form, width=0.003, headwidth=3,
                  headlength=4, headaxislength=3.5, alpha=0.80)

    # Frames de orientación de ambos drones (hasta 6 instantes)
    idx_fr = np.unique(np.round(np.linspace(0, n - 1, min(6, n))).astype(int))
    for i in idx_fr:
        _draw_drone_frame(ax, xL_t[i], yL_t[i], psiL[i])
        _draw_drone_frame(ax, xS_t[i], yS_t[i], psiS[i])

    # Frame RTK en el origen
    _draw_rtk_frame(ax)

    # Entradas de leyenda para ejes y vector d
    ax.plot([], [], '-', color=[0.85, 0.10, 0.10], lw=2.5, label=r'$\hat{x}_D$')
    ax.plot([], [], '-', color=[0.10, 0.65, 0.10], lw=2.5, label=r'$\hat{y}_D$')
    ax.plot([], [], '-', color=C_form, lw=2.0, label=r'Vector $\mathbf{d}$')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x_{RTK}$ [m]', fontsize=11)
    ax.set_ylabel(r'$y_{RTK}$ [m]', fontsize=11)
    ax.set_title('Trayectoria XY con marcos de referencia y vector de formación')
    ax.legend(loc='best', fontsize=9)
    fig1.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 2 — Líder: setpoint vs real + espejo + frame RTK
    # ─────────────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    xdesL_t, ydesL_t = transform_for_mirror(xdesL, ydesL)
    xL_t,    yL_t    = transform_for_mirror(xL, yL)
    xL0_t,   yL0_t   = transform_for_mirror(np.array([xL[0]]), np.array([yL[0]]))
    ax.plot(xdesL_t, ydesL_t, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax.plot(xL_t,    yL_t,    '-',  color=C_L,    lw=LW,  label='Líder real')
    ax.plot(xL0_t,   yL0_t,   'o',  ms=10, mfc=C_L, mec='k', label='Inicio')
    # Frames de orientación del líder (hasta 6 instantes)
    idx_fr2 = np.unique(np.round(np.linspace(0, n - 1, min(6, n))).astype(int))
    for i in idx_fr2:
        _draw_drone_frame(ax, xL_t[i], yL_t[i], psiL[i])
    ax.plot([], [], '-', color=[0.85, 0.10, 0.10], lw=2.5, label=r'$\hat{x}_D$')
    ax.plot([], [], '-', color=[0.10, 0.65, 0.10], lw=2.5, label=r'$\hat{y}_D$')
    _draw_rtk_frame(ax)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x_{RTK}$ [m]'); ax.set_ylabel(r'$y_{RTK}$ [m]')
    ax.set_title('Líder: Setpoint vs Trayectoria Real')
    ax.legend(loc='best')
    fig2.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 3 — Seguidor: setpoint vs real + espejo + frame RTK
    # ─────────────────────────────────────────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    xdesS_t, ydesS_t = transform_for_mirror(xdesS, ydesS)
    xS_t,    yS_t    = transform_for_mirror(xS, yS)
    xS0_t,   yS0_t   = transform_for_mirror(np.array([xS[0]]), np.array([yS[0]]))
    ax.plot(xdesS_t, ydesS_t, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax.plot(xS_t,    yS_t,    '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax.plot(xS0_t,   yS0_t,   'o',  ms=10, mfc=C_S, mec='k', label='Inicio')
    # Frames de orientación del seguidor (hasta 6 instantes)
    idx_fr3 = np.unique(np.round(np.linspace(0, n - 1, min(6, n))).astype(int))
    for i in idx_fr3:
        _draw_drone_frame(ax, xS_t[i], yS_t[i], psiS[i])
    ax.plot([], [], '-', color=[0.85, 0.10, 0.10], lw=2.5, label=r'$\hat{x}_D$')
    ax.plot([], [], '-', color=[0.10, 0.65, 0.10], lw=2.5, label=r'$\hat{y}_D$')
    _draw_rtk_frame(ax)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$x_{RTK}$ [m]'); ax.set_ylabel(r'$y_{RTK}$ [m]')
    ax.set_title('Seguidor: Setpoint vs Trayectoria Real')
    ax.legend(loc='best')
    fig3.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 4 — Estados x, y, z, ψ vs tiempo (4 subplots)
    # ─────────────────────────────────────────────────────────────────────────
    fig4, axes = plt.subplots(4, 1, figsize=(9.5, 7.5), sharex=True, facecolor='white')
    fig4.suptitle('Posición y orientación vs Tiempo', fontsize=12)

    ax = axes[0]
    ax.plot(t, xdesL, '--', color=C_refL, lw=LWS, label='Setpoint L')
    ax.plot(t, xdesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    ax.plot(t, xL,    '-',  color=C_L,   lw=LW,  label='x Líder')
    ax.plot(t, xS,    '-',  color=C_S,   lw=LW,  label='x Seguidor')
    ax.set_ylabel('x [m]'); ax.grid(True)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.set_title('Posición X vs Tiempo')

    ax = axes[1]
    ax.plot(t, ydesL, '--', color=C_refL, lw=LWS, label='Setpoint L')
    ax.plot(t, ydesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    ax.plot(t, yL,    '-',  color=C_L,   lw=LW,  label='y Líder')
    ax.plot(t, yS,    '-',  color=C_S,   lw=LW,  label='y Seguidor')
    ax.set_ylabel('y [m]'); ax.grid(True)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.set_title('Posición Y vs Tiempo')

    ax = axes[2]
    ax.plot(t, zdesS, '--', color=C_refS, lw=LWS, label='Setpoint S')
    ax.plot(t, zL,    '-',  color=C_L,   lw=LW,  label='z Líder')
    ax.plot(t, zS,    '-',  color=C_S,   lw=LW,  label='z Seguidor')
    ax.set_ylabel('z [m]'); ax.grid(True)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.set_title('Posición Z (Altitud) vs Tiempo')

    ax = axes[3]
    ax.plot(t, np.degrees(psidesS), '--', color=C_refS, lw=LWS, label='Setpoint S')
    ax.plot(t, np.degrees(psiL),    '-',  color=C_L,   lw=LW,  label='ψ Líder')
    ax.plot(t, np.degrees(psiS),    '-',  color=C_S,   lw=LW,  label='ψ Seguidor')
    ax.set_ylabel('ψ [°]'); ax.set_xlabel('Tiempo [s]'); ax.grid(True)
    ax.legend(loc='best', ncol=2, fontsize=8)
    ax.set_title('Yaw vs Tiempo')

    fig4.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 5 — Errores del líder (4 subplots)
    # ─────────────────────────────────────────────────────────────────────────
    fig5, axes = plt.subplots(4, 1, figsize=(9.5, 7), sharex=True, facecolor='white')
    fig5.suptitle('Errores del Líder respecto a su setpoint', fontsize=12)

    for ax, data, lbl in zip(axes,
                              [ex_L, ey_L, ez_L, epsi_L],
                              ['eₓ [m]', 'e_y [m]', 'e_z [m]', 'e_ψ [°]']):
        ax.plot(t, data, '-', color=C_L, lw=LW)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lbl); ax.grid(True)

    axes[3].set_xlabel('Tiempo [s]')
    fig5.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 6 — Errores del seguidor (4 subplots)
    # ─────────────────────────────────────────────────────────────────────────
    fig6, axes = plt.subplots(4, 1, figsize=(9.5, 7), sharex=True, facecolor='white')
    fig6.suptitle('Errores del Seguidor respecto a su setpoint', fontsize=12)

    for ax, data, lbl in zip(axes,
                              [ex_S, ey_S, ez_S, epsi_S],
                              ['eₓ [m]', 'e_y [m]', 'e_z [m]', 'e_ψ [°]']):
        ax.plot(t, data, '-', color=C_S, lw=LW)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lbl); ax.grid(True)

    axes[3].set_xlabel('Tiempo [s]')
    fig6.tight_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # FIG 7 — Distancia de formación vs tiempo
    # ─────────────────────────────────────────────────────────────────────────
    fig7, ax = plt.subplots(figsize=(9.5, 3.8), facecolor='white')
    ax.plot(t, dist, '-', color=C_dist, lw=LW, label='Distancia real L-S')
    ax.axhline(OFFSET_D, color=np.array([0.0, 0.5, 0.2]), ls='--', lw=1.2,
               label=f'd = {OFFSET_D:.1f} m')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Distancia [m]')
    ax.set_title('Distancia entre Líder y Seguidor vs Tiempo')
    ax.grid(True); ax.legend(loc='best')
    fig7.tight_layout()

    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    csv_path = LEADER_CSV
    speed    = SPEED_FACTOR

    # Mostrar configuración de inversión de ejes
    print(f'\n🔄 Configuración de inversión de ejes para Figura 1:')
    print(f'   INVERTIR_EJE_X = {INVERTIR_EJE_X}  {"(invertido)" if INVERTIR_EJE_X else "(normal)"}')
    print(f'   INVERTIR_EJE_Y = {INVERTIR_EJE_Y}  {"(invertido)" if INVERTIR_EJE_Y else "(normal)"}')
    if INVERTIR_EJE_X or INVERTIR_EJE_Y:
        print(f'   → Aplicando transformación: x → {"-x" if INVERTIR_EJE_X else "x"}, y → {"-y" if INVERTIR_EJE_Y else "y"}')

    # Cargar trayectoria del líder virtual
    leader_data = load_leader_csv(csv_path)
    duration    = leader_data[-1]['t']

    # Conectar
    print(f'🔗 Conectando a {CONN} ...')
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f'✅ Heartbeat: SYS={master.target_system} COMP={master.target_component}')

    # Hilo lector del seguidor
    stop_reader = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader, args=(master, stop_reader),
        daemon=True, name='mavlink-reader'
    )
    reader_thread.start()

    # Esperar telemetría completa del seguidor
    print('⏳ Esperando telemetría del seguidor...', end='', flush=True)
    while not follower_ready(get_follower()):
        time.sleep(0.02)
    print(' listo.')

    S0 = get_follower()
    print(f'📍 Seguidor inicial: x={S0["x"]:.2f}, y={S0["y"]:.2f}, '
          f'z={S0["z"]:.2f}, yaw={math.degrees(S0["yaw"]):.1f}°')
    print(f'⚙️  Offset: D={OFFSET_D} m, α={math.degrees(OFFSET_ALPHA):.0f}°, '
          f'Δz={OFFSET_DZ} m')
    print(f'⚙️  PID pos: Kp={KP}, Ki={KI}, Kd={KD}')
    print(f'⚙️  PID yaw: Kp={KP_YAW}, Ki={KI_YAW}, Kd={KD_YAW}')
    print(f'⚙️  speed_factor={speed}x | duración líder={duration:.1f} s')
    print(f'⚙️  Pre-posicionamiento: {"ACTIVADO" if PRE_POSITION else "DESACTIVADO"}')

    # ── Pre-posicionamiento (opcional) ────────────────────────────────────────
    if PRE_POSITION:
        goto_initial_position(master, leader_data)
        input('\n▶️  ¿Iniciar el controlador de seguimiento? Presiona ENTER...\n')
    else:
        input('\n▶️  Drone listo? Presiona ENTER para iniciar el seguimiento...\n')

    # CSV de salida
    out_csv_path = f'lf_log_{int(time.time())}.csv'
    out_csv_f    = open(out_csv_path, 'w', newline='')
    out_csv_w    = csv.writer(out_csv_f)
    out_csv_w.writerow(CSV_HEADER)
    print(f'💾 Log de seguimiento en: {out_csv_path}')

    # ── Buffers para gráfica — estructura extendida para coincidir con MATLAB ──
    plot_buf = dict(
        t=[],
        # Líder real
        xl=[], yl=[], zl=[], psiL=[],
        # Seguidor real
        xs=[], ys=[], zs=[], psiS=[],
        # Setpoints del líder (del CSV del círculo: x_sp, y_sp, z_sp)
        xdesL=[], ydesL=[], zdesL=[],
        # Errores del líder (del CSV del círculo: ex_L, ey_L)
        exL=[], eyL=[],
        # Setpoints seguidor (calculados por compute_control)
        xd=[], yd=[], zd=[],
        # Errores seguidor
        ex=[], ey=[], ez=[],
    )

    pid      = PIDState()
    idx_hint = 0
    t_start  = time.monotonic()
    t_log    = 0.0

    print('🚀 Control iniciado. Ctrl+C para detener.\n')

    try:
        next_t = time.monotonic()
        while True:
            now     = time.monotonic()
            elapsed = (now - t_start) * speed

            # Obtener estado del líder virtual interpolado
            L, idx_hint = interpolate_leader(leader_data, elapsed, idx_hint)

            # Obtener estado del seguidor real
            S = get_follower()

            if follower_ready(S):
                vx, vy, vz_ned, yaw_rate_cmd, c = compute_control(L, S, pid)
                send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate_cmd)

                t_log += DT

                # Log CSV
                out_csv_w.writerow([
                    f'{t_log:.4f}',
                    L['x'], L['y'], L['z'], L['vx'], L['vy'], L['vz'],
                    L['yaw'], L['yaw_rate'],
                    S['x'], S['y'], S['z'], S['vx'], S['vy'], S['vz'],
                    S['yaw'], S['yaw_rate'],
                    c['xd'], c['yd'], c['zd'],
                    c['ex'], c['ey'], c['ez'], c['e_yaw'],
                    c['ff_x'], c['ff_y'], c['ff_z'],
                    c['vx_p'], c['vy_p'], c['vz_p'],
                    c['vx_i'], c['vy_i'], c['vz_i'],
                    c['vx_d'], c['vy_d'], c['vz_d'],
                    c['vx'], c['vy'], c['vz'],
                    yaw_rate_cmd,
                ])

                # ── Buffers gráfica ──
                plot_buf['t'].append(t_log)
                # Líder real
                plot_buf['xl'].append(L['x'])
                plot_buf['yl'].append(L['y'])
                plot_buf['zl'].append(L['z'])
                plot_buf['psiL'].append(L['yaw'])
                # Seguidor real
                plot_buf['xs'].append(S['x'])
                plot_buf['ys'].append(S['y'])
                plot_buf['zs'].append(S['z'])
                plot_buf['psiS'].append(S['yaw'])
                # Setpoints del líder (del CSV del círculo)
                plot_buf['xdesL'].append(L['x_sp'])
                plot_buf['ydesL'].append(L['y_sp'])
                plot_buf['zdesL'].append(L['z_sp'])
                # Errores del líder (del CSV del círculo)
                plot_buf['exL'].append(L['ex_L'])
                plot_buf['eyL'].append(L['ey_L'])
                # Setpoints seguidor
                plot_buf['xd'].append(c['xd'])
                plot_buf['yd'].append(c['yd'])
                plot_buf['zd'].append(c['zd'])
                # Errores seguidor
                plot_buf['ex'].append(c['ex'])
                plot_buf['ey'].append(c['ey'])
                plot_buf['ez'].append(c['ez'])

                # Imprimir progreso cada 5 s
                if int(t_log) % 5 == 0 and int(t_log - DT) % 5 != 0:
                    err = math.hypot(c['ex'], c['ey'])
                    print(f't={t_log:.1f}s | líder=({L["x"]:.2f},{L["y"]:.2f}) | '
                          f'seg=({S["x"]:.2f},{S["y"]:.2f}) | '
                          f'|e_xy|={err:.3f} m | '
                          f'e_yaw={math.degrees(c["e_yaw"]):.1f}°')

                # Fin de trayectoria del líder
                if elapsed >= duration:
                    print(f'\n✅ Trayectoria del líder completada ({duration:.1f} s).')
                    break

            # Timing absoluto
            next_t += DT
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\n⏹️  Interrupción por usuario.')

    finally:
        # Detener el drone
        send_velocity_yawrate(master, 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.2)

        stop_reader.set()
        reader_thread.join(timeout=2.0)

        out_csv_f.flush()
        out_csv_f.close()
        master.close()

        print(f'💾 Log guardado: {out_csv_path}')
        print('🔌 Conexión cerrada.')

        plot_results(plot_buf)


if __name__ == '__main__':
    main()
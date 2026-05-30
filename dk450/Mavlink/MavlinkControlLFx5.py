#!/usr/bin/env python3
"""
LF_Circulo_RT_v2.py — Líder-Seguidor en Tiempo Real con Círculo
================================================================
Cambios respecto a v1:
  - Etapa de despegue para cada dron (con confirmación ENTER)
  - Log de TRAYECTORIA separado (`lf_traj_*.csv`): registra posición/yaw
    de AMBOS drones desde el inicio del programa (despegue, prepos, círculo).
    Usado para el replay de RViz2.
  - Log de CONTROL (`lf_circulo_*.csv`): igual que antes, solo se llena
    cuando el PID está activo. No se modificó ninguna columna ni lógica.
  - El replay de RViz2 se genera a partir del log de trayectoria.

Secuencia:
  1. Conectar y fijar origen ENU
  2. Esperar telemetría de ambos drones
  3. Despegue del LÍDER    (ENTER para confirmar)
  4. Despegue del SEGUIDOR (ENTER para confirmar)
  5. Preposicionamiento
  6. Círculo + PID seguidor
  7. Regreso (opcional)
  8. Aterrizaje del LÍDER    (ENTER para confirmar)
  9. Aterrizaje del SEGUIDOR (ENTER para confirmar)
 10. Guardar CSVs y generar replay RViz2
"""

import math
import time
import csv
import json
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
from pymavlink import mavutil

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE CORRIDA
# ══════════════════════════════════════════════════════════════════════════════
SKIP_PREPOS  = False
SKIP_RETURN  = False
RESET_ORIGIN = False

LIVE_PLOT_ENABLED   = True
LIVE_PLOT_TRAJ_ONLY = True
LIVE_PLOT_RATE      = 4        # Hz de refresco de la ventana en vivo

# ══════════════════════════════════════════════════════════════════════════════
# CONEXIONES Y SYSIDS
# ══════════════════════════════════════════════════════════════════════════════
LEADER_CONN    = 'udp:127.0.0.1:14552'
FOLLOWER_CONN  = 'udp:127.0.0.1:14553'

R_EARTH = 6371000.0

LEADER_SYSID    = 1
LEADER_COMPID   = 0
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# ══════════════════════════════════════════════════════════════════════════════
# DESPEGUE
# ══════════════════════════════════════════════════════════════════════════════
TAKEOFF_ALT_LEADER   = 2.0   # metros AGL
TAKEOFF_ALT_FOLLOWER = 2.0
TAKEOFF_TIMEOUT      = 30.0  # segundos máximos esperando altitud
TAKEOFF_ALT_TOL      = 0.3   # tolerancia [m] para considerar "alcanzó altitud"

# ══════════════════════════════════════════════════════════════════════════════
# FRAME ENU COMÚN
# ══════════════════════════════════════════════════════════════════════════════
_enu_origin = {'lat': None, 'lon': None, 'alt': None}
ENU_ORIGIN_FILE = 'enu_origin.json'
_leader_ned_home = {'x': None, 'y': None}


def set_enu_origin(lat_deg, lon_deg, alt_m):
    _enu_origin['lat'] = lat_deg
    _enu_origin['lon'] = lon_deg
    _enu_origin['alt'] = alt_m
    print(f"\n[ENU] Origen fijado:")
    print(f"      lat={lat_deg:.8f}  lon={lon_deg:.8f}  alt={alt_m:.2f} m")


def save_enu_origin():
    with open(ENU_ORIGIN_FILE, 'w') as f:
        json.dump({'lat': _enu_origin['lat'],
                   'lon': _enu_origin['lon'],
                   'alt': _enu_origin['alt']}, f, indent=2)
    print(f"[ENU] Origen guardado en {ENU_ORIGIN_FILE}")


def load_enu_origin():
    if not os.path.exists(ENU_ORIGIN_FILE):
        return False
    with open(ENU_ORIGIN_FILE) as f:
        data = json.load(f)
    set_enu_origin(data['lat'], data['lon'], data['alt'])
    print(f"[ENU] Origen CARGADO desde {ENU_ORIGIN_FILE}")
    return True


def gps_to_enu(lat_deg, lon_deg, alt_m):
    ref_lat = math.radians(_enu_origin['lat'])
    dlat    = math.radians(lat_deg - _enu_origin['lat'])
    dlon    = math.radians(lon_deg - _enu_origin['lon'])
    x = dlat * R_EARTH
    y = dlon * R_EARTH * math.cos(ref_lat)
    z = alt_m - _enu_origin['alt']
    return x, y, z


def capture_leader_ned_home(master_leader):
    print("[ENU] Capturando home NED del líder...", end='', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15.0:
        msg = master_leader.recv_match(
            type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
        if msg:
            with _lock:
                lat = _leader_state.get('lat')
            if lat and _enu_origin['lat']:
                enu_x, enu_y, _ = gps_to_enu(lat, _leader_state.get('lon', 0), 0)
                _leader_ned_home['x'] = enu_x - msg.x
                _leader_ned_home['y'] = enu_y - msg.y
                print(f" ✅  offset=({_leader_ned_home['x']:.2f}, {_leader_ned_home['y']:.2f})")
                return True
    print(" ❌ timeout")
    return False

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DEL CÍRCULO Y PID
# ══════════════════════════════════════════════════════════════════════════════
RADIUS        = 4.0
ANGULAR_SPEED = 0.2
LEADER_RATE   = 50

CONV_RADIUS  = 0.15
CONV_SPEED   = 0.10
CONV_HOLD    = 1.0
CONV_TIMEOUT = 15.0

PREPOS_Z_LEADER   = 3.0
PREPOS_Z_FOLLOWER = 3.0
PREPOS_TIMEOUT    = 40.0
PREPOS_CONV_R     = 0.30
PREPOS_CONV_V     = 0.20
PREPOS_CONV_HOLD  = 2.0

RETURN_ENABLED = True

RETURN_TIMEOUT = 40.0

FOLLOWER_RATE = 20

OFFSET_D     = 2.0
OFFSET_ALPHA = -math.pi / 2
OFFSET_DZ    = 1.0

KP, KI, KD              = 0.5, 0.0, 0.0
KP_YAW, KI_YAW, KD_YAW = 0.8, 0.0, 0.0

INTEGRAL_LIMIT     = 2.0
INTEGRAL_YAW_LIMIT = 1.0
V_MAX              = 2.0
V_MAX_Z            = 1.0
YAW_RATE_MAX       = 1.0

# ══════════════════════════════════════════════════════════════════════════════
# MÁSCARAS MAVLINK
# ══════════════════════════════════════════════════════════════════════════════
TYPE_MASK_POS_YAW = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)
TYPE_MASK_VEL_YAWRATE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO COMPARTIDO
# ══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()


def _empty_state():
    return dict(x=None, y=None, z=None,
                vx=None, vy=None, vz=None,
                yaw=None, yaw_rate=None,
                lat=None, lon=None, alt=None)


_leader_state   = _empty_state()
_follower_state = _empty_state()

# ── Log de CONTROL (solo durante PID, igual que v1) ──────────────────────────
_log_lock = threading.Lock()
_LOG_COLS = [
    'timestamp_unix', 't', 'leader_phase',
    'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    'lx_sp', 'ly_sp',
    'l_lat', 'l_lon', 'l_alt',
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    's_lat', 's_lon', 's_alt',
    'xd', 'yd', 'zd',
    'ex', 'ey', 'ez',
    'err_xy', 'err_z',
    'e_yaw', 'dist_xy', 'dist_z',
    'ff_x', 'ff_y',
    'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i',
    'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd', 'yaw_rate_cmd',
]
_log = {k: [] for k in _LOG_COLS}

# ── Log de TRAYECTORIA (desde el inicio del programa, para RViz2) ─────────────
# Columnas: tiempo, fase, posición y yaw de ambos drones
_TRAJ_COLS = [
    'timestamp_unix', 't', 'phase',
    'lx', 'ly', 'lz', 'l_yaw',
    'sx', 'sy', 'sz', 's_yaw',
]
_traj_log      = {k: [] for k in _TRAJ_COLS}
_traj_lock     = threading.Lock()
_traj_timer    = None   # threading.Timer periódico
_traj_t0       = None   # tiempo de referencia del programa completo
_traj_phase    = 'init' # fase actual: init, takeoff_L, takeoff_S, prepos, circle, return, land_L, land_S

_circle_done  = threading.Event()
_stop_all     = threading.Event()
_leader_phase = 'circle'   # para el log de control

# ══════════════════════════════════════════════════════════════════════════════
# LOG DE TRAYECTORIA — grabación periódica desde inicio
# ══════════════════════════════════════════════════════════════════════════════
def _traj_tick():
    """Se llama periódicamente (10 Hz) para registrar posición de ambos drones."""
    global _traj_timer
    if _stop_all.is_set():
        return

    ts_unix = time.time()
    t_rel   = time.monotonic() - _traj_t0

    with _lock:
        L = dict(_leader_state)
        S = dict(_follower_state)
        phase = _traj_phase

    # Solo registrar si tenemos al menos posición x,y,z de ambos
    lx  = L.get('x')  or 0.0
    ly  = L.get('y')  or 0.0
    lz  = L.get('z')  or 0.0
    lyw = L.get('yaw') or 0.0
    sx  = S.get('x')  or 0.0
    sy  = S.get('y')  or 0.0
    sz  = S.get('z')  or 0.0
    syw = S.get('yaw') or 0.0

    row = {
        'timestamp_unix': f'{ts_unix:.6f}',
        't':     f'{t_rel:.4f}',
        'phase': phase,
        'lx':    f'{lx:.4f}',  'ly':    f'{ly:.4f}',  'lz':    f'{lz:.4f}',
        'l_yaw': f'{lyw:.4f}',
        'sx':    f'{sx:.4f}',  'sy':    f'{sy:.4f}',  'sz':    f'{sz:.4f}',
        's_yaw': f'{syw:.4f}',
    }
    with _traj_lock:
        for k in _TRAJ_COLS:
            _traj_log[k].append(row[k])

    # Re-programar el siguiente tick
    _traj_timer = threading.Timer(0.1, _traj_tick)
    _traj_timer.daemon = True
    _traj_timer.start()


def start_traj_log(t0):
    """Arranca el log de trayectoria desde el inicio del programa."""
    global _traj_t0, _traj_timer
    _traj_t0 = t0
    _traj_timer = threading.Timer(0.0, _traj_tick)
    _traj_timer.daemon = True
    _traj_timer.start()


def set_traj_phase(phase: str):
    global _traj_phase
    with _lock:
        _traj_phase = phase


def stop_traj_log():
    global _traj_timer
    if _traj_timer is not None:
        _traj_timer.cancel()


def save_traj_csv() -> str:
    fname = f'lf_traj_{int(time.time())}.csv'
    with _traj_lock:
        rows = list(zip(*[_traj_log[k] for k in _TRAJ_COLS]))
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_TRAJ_COLS)
        w.writerows(rows)
    print(f"📍 Trayectoria CSV: {fname}  ({len(rows)} muestras)")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# LECTORES MAVLINK
# ══════════════════════════════════════════════════════════════════════════════
def _reader(master, state_dict, stop_event):
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['GLOBAL_POSITION_INT', 'LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1)
        if msg is None:
            continue
        mtype = msg.get_type()
        with _lock:
            if mtype == 'GLOBAL_POSITION_INT':
                lat = msg.lat * 1e-7
                lon = msg.lon * 1e-7
                state_dict['lat'] = lat
                state_dict['lon'] = lon
                if _enu_origin['lat'] is not None:
                    x, y, _ = gps_to_enu(lat, lon, 0.0)
                    state_dict['x'] = x
                    state_dict['y'] = y
                state_dict['vx'] = msg.vx * 0.01
                state_dict['vy'] = msg.vy * 0.01
            elif mtype == 'LOCAL_POSITION_NED':
                state_dict['z']   = -msg.z
                state_dict['vz']  = -msg.vz
                state_dict['alt'] = -msg.z
            elif mtype == 'ATTITUDE':
                state_dict['yaw']      = msg.yaw
                state_dict['yaw_rate'] = msg.yawspeed


def get_leader():
    with _lock: return dict(_leader_state)


def get_follower():
    with _lock: return dict(_follower_state)


def state_ready(s):
    return all(s.get(k) is not None
               for k in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'yaw_rate'])

# ══════════════════════════════════════════════════════════════════════════════
# DESPEGUE
# ══════════════════════════════════════════════════════════════════════════════
def _set_mode_guided(master, sysid, compid, label=''):
    """Envía MAV_CMD_DO_SET_MODE para GUIDED (mode=4 en ArduCopter)."""
    master.mav.command_long_send(
        sysid, compid,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0
    )
    time.sleep(1.0)
    print(f"  [{label}] Modo GUIDED enviado.")


def _arm(master, sysid, compid, label=''):
    """Arma los motores y espera confirmación."""
    master.mav.command_long_send(
        sysid, compid,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
    print(f"  [{label}] Armando motores...", end='', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 8.0:
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1.0)
        if msg and (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            print(" ✅ armado")
            return True
        print('.', end='', flush=True)
    print(" ⚠️  no se confirmó armado")
    return False


def _send_takeoff(master, sysid, compid, altitude, label=''):
    """Envía MAV_CMD_NAV_TAKEOFF."""
    master.mav.command_long_send(
        sysid, compid,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0,
        0, 0, float(altitude)
    )
    print(f"  [{label}] 🚀 Despegando a {altitude} m AGL...")


def _wait_altitude(get_state_fn, target_alt, tol, timeout, label=''):
    """Espera hasta que el dron alcance target_alt ± tol."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = get_state_fn()
        z = s.get('z')
        if z is not None and abs(z - target_alt) < tol:
            print(f"  [{label}] ✅ Altitud alcanzada: {z:.2f} m")
            return True
        time.sleep(0.3)
    print(f"  [{label}] ⚠️  Timeout esperando altitud")
    return False


def do_takeoff(master, sysid, compid, get_state_fn, alt, label):
    """
    Secuencia completa de despegue para un dron:
      GUIDED → ARM → TAKEOFF → esperar altitud.
    """
    print(f"\n{'─'*60}")
    print(f"  DESPEGUE — {label}")
    print(f"{'─'*60}")
    _set_mode_guided(master, sysid, compid, label)
    ok = _arm(master, sysid, compid, label)
    if not ok:
        print(f"  [{label}] ⚠️  Continuando igualmente (simulador puede no confirmar).")
    _send_takeoff(master, sysid, compid, alt, label)
    _wait_altitude(get_state_fn, alt, TAKEOFF_ALT_TOL, TAKEOFF_TIMEOUT, label)

def do_land(master, sysid, compid, get_state_fn, label):
    """
    Envía MAV_CMD_NAV_LAND y espera hasta que el dron toca tierra (z < 0.15 m).
    """
    print(f"\n{'─'*60}")
    print(f"  ATERRIZAJE — {label}")
    print(f"{'─'*60}")
    master.mav.command_long_send(
        sysid, compid,
        mavutil.mavlink.MAV_CMD_NAV_LAND, 0,
        0, 0, 0, 0,
        0, 0, 0
    )
    print(f"  [{label}] 🛬 Aterrizando...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 40.0:
        s = get_state_fn()
        z = s.get('z')
        if z is not None and z < 0.15:
            print(f"  [{label}] ✅ En tierra (z={z:.2f} m)")
            return True
        time.sleep(0.5)
    print(f"  [{label}] ⚠️  Timeout esperando aterrizaje")
    return False


# ══════════════════════════════════════════════════════════════════════════════
# ENVÍO DE COMANDOS (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def send_pos_enu(master, sysid, compid, enu_x, enu_y, z_agl, yaw, drone_state):
    delta_x = enu_x - drone_state['x']
    delta_y = enu_y - drone_state['y']
    delta_z_ned = -(z_agl - drone_state['z'])
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
        TYPE_MASK_POS_YAW,
        delta_x, delta_y, delta_z_ned,
        0, 0, 0, 0, 0, 0, yaw, 0)


def send_pos_enu_abs(master, sysid, compid, enu_x, enu_y, z_agl, yaw):
    x_ned = enu_x - _leader_ned_home['x']
    y_ned = enu_y - _leader_ned_home['y']
    z_ned = -z_agl
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x_ned, y_ned, z_ned,
        0, 0, 0, 0, 0, 0, yaw, 0)


def send_vel_yawrate(master, sysid, compid, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0, vx, vy, vz_ned, 0, 0, 0, 0, yaw_rate)

# ══════════════════════════════════════════════════════════════════════════════
# PID (igual que v1, sin cambios)
# ══════════════════════════════════════════════════════════════════════════════
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def clamp(val, limit):
    return max(-limit, min(limit, val))


def compute_offset(psi_L):
    angle = psi_L + OFFSET_ALPHA
    return OFFSET_D * math.cos(angle), OFFSET_D * math.sin(angle), OFFSET_DZ


def compute_ff(yaw_rate_L, dx, dy):
    return yaw_rate_L * (-dy), yaw_rate_L * dx


class PIDState:
    def __init__(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0


def compute_control(L, S, pid, dt):
    dx, dy, dz = compute_offset(L['yaw'])
    xd = L['x'] + dx;  yd = L['y'] + dy;  zd = L['z'] + dz
    ex = xd - S['x'];  ey = yd - S['y'];  ez = zd - S['z']

    pid.integral[0] = clamp(pid.integral[0] + ex * dt, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * dt, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * dt, INTEGRAL_LIMIT)

    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    ff_x, ff_y = compute_ff(L['yaw_rate'], dx, dy)

    px, py, pz    = KP * ex,              KP * ey,              KP * ez
    ix, iy, iz    = KI * pid.integral[0], KI * pid.integral[1], KI * pid.integral[2]
    dx_, dy_, dz_ = KD * dv_x,           KD * dv_y,            KD * dv_z

    vx = ff_x + px + ix + dx_
    vy = ff_y + py + iy + dy_
    vz =        pz + iz + dz_

    v_h = math.hypot(vx, vy)
    if v_h > V_MAX:
        vx *= V_MAX / v_h
        vy *= V_MAX / v_h
    vz = clamp(vz, V_MAX_Z)
    vz_ned = -vz

    e_yaw = wrap(L['yaw'] - S['yaw'])
    pid.integral_yaw = clamp(pid.integral_yaw + e_yaw * dt, INTEGRAL_YAW_LIMIT)
    dyaw = L['yaw_rate'] - S['yaw_rate']
    yaw_rate_cmd = clamp(
        KP_YAW * e_yaw + KI_YAW * pid.integral_yaw + KD_YAW * dyaw,
        YAW_RATE_MAX)

    return dict(
        vx=vx, vy=vy, vz_ned=vz_ned, yaw_rate_cmd=yaw_rate_cmd,
        xd=xd, yd=yd, zd=zd,
        ex=ex, ey=ey, ez=ez,
        ff_x=ff_x, ff_y=ff_y,
        px=px, py=py, pz=pz,
        ix=ix, iy=iy, iz=iz,
        dx=dx_, dy=dy_, dz=dz_,
        e_yaw=e_yaw,
    )

# ══════════════════════════════════════════════════════════════════════════════
# UTILIDAD — WAYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def _fly_to_wp(master, sysid, compid, tx, ty, tz_agl, get_state_fn,
               label='', conv_r=None, conv_v=None, conv_hold=None,
               timeout=None, rate=20):
    conv_r    = conv_r    or PREPOS_CONV_R
    conv_v    = conv_v    or PREPOS_CONV_V
    conv_hold = conv_hold or PREPOS_CONV_HOLD
    timeout   = timeout   or PREPOS_TIMEOUT
    dt        = 1.0 / rate
    t0        = time.monotonic()
    t_in_zone = None
    next_t    = t0

    while not _stop_all.is_set():
        now = time.monotonic()
        if now - t0 > timeout:
            print(f"  [{label}] ⚠️  Timeout convergencia.")
            return False
        s = get_state_fn()
        if not state_ready(s):
            time.sleep(dt); continue
        send_pos_enu(master, sysid, compid, tx, ty, tz_agl, yaw=0.0, drone_state=s)
        dist   = math.hypot(s['x'] - tx, s['y'] - ty)
        dist_z = abs(s['z'] - tz_agl)
        speed  = math.hypot(s['vx'], s['vy'])
        if dist < conv_r and dist_z < conv_r * 2 and speed < conv_v:
            if t_in_zone is None:
                t_in_zone = now
            elif now - t_in_zone >= conv_hold:
                print(f"  [{label}] ✅ Convergido en {now-t0:.1f}s  "
                      f"dist_xy={dist:.3f}m  dist_z={dist_z:.3f}m")
                return True
        else:
            t_in_zone = None
        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0:
            time.sleep(sl)
    return False

# ══════════════════════════════════════════════════════════════════════════════
# PREPOSICIONAMIENTO (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def preposition(master_leader, master_follower):
    theta0      = math.pi
    yaw_inicial = theta0 + math.pi / 2

    L_x = -RADIUS;  L_y = 0.0;  L_z = PREPOS_Z_LEADER
    dx_off, dy_off, _ = compute_offset(yaw_inicial)
    S_x = L_x + dx_off;  S_y = L_y + dy_off;  S_z = PREPOS_Z_FOLLOWER

    print(f"\n{'─'*65}")
    print(f"  PREPOSICIONAMIENTO")
    print(f"  Centro círculo: (0.0, 0.0) ENU")
    print(f"  yaw_inicial={math.degrees(yaw_inicial):.1f}°")
    print(f"  Líder    → ENU ({L_x:.2f}, {L_y:.2f})  z={L_z:.1f}m AGL")
    print(f"  Seguidor → ENU ({S_x:.2f}, {S_y:.2f})  z={S_z:.1f}m AGL")
    print(f"{'─'*65}")

    set_traj_phase('prepos')

    print(f"\n  Turno 1/2 — SEGUIDOR moviéndose...")
    ok_s = _fly_to_wp(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                      S_x, S_y, S_z, get_follower, label='SEGUIDOR prepos')
    if not ok_s:
        print("  ⚠️  Seguidor no convergió.")
    input("  ENTER para mover el LÍDER...\n")

    print(f"  Turno 2/2 — LÍDER moviéndose...")
    dt        = 1.0 / 20
    t0        = time.monotonic()
    t_in_zone = None

    while not _stop_all.is_set():
        if time.monotonic() - t0 > PREPOS_TIMEOUT:
            print("  [LIDER prepos] ⚠️  Timeout."); break
        L = get_leader()
        if not state_ready(L):
            time.sleep(dt); continue
        send_pos_enu(master_leader, LEADER_SYSID, LEADER_COMPID,
                     L_x, L_y, L_z, yaw_inicial, drone_state=L)
        dist   = math.hypot(L['x'] - L_x, L['y'] - L_y)
        dist_z = abs(L['z'] - L_z)
        speed  = math.hypot(L['vx'], L['vy'])
        if dist < PREPOS_CONV_R and dist_z < PREPOS_CONV_R * 2 and speed < PREPOS_CONV_V:
            if t_in_zone is None:
                t_in_zone = time.monotonic()
            elif time.monotonic() - t_in_zone >= PREPOS_CONV_HOLD:
                print(f"  [LIDER prepos] ✅ Convergido en {time.monotonic()-t0:.1f}s")
                break
        else:
            t_in_zone = None
        time.sleep(dt)

    L = get_leader(); S = get_follower()
    if state_ready(L) and state_ready(S):
        sep   = math.hypot(L['x'] - S['x'], L['y'] - S['y'])
        sep_z = abs(L['z'] - S['z'])
        print(f"\n  Preposicionamiento completado.")
        print(f"  Sep L-S: {sep:.2f}m horiz  {sep_z:.2f}m vert  "
              f"(objetivo d={OFFSET_D}m dz={OFFSET_DZ}m)")
    return L_x, L_y, L_z, S_x, S_y, S_z

# ══════════════════════════════════════════════════════════════════════════════
# REGRESO (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def return_to_start(master_leader, master_follower, L_x, L_y, L_z, S_x, S_y, S_z):
    print(f"\n{'─'*65}")
    print(f"  REGRESO AL ORIGEN"); print(f"{'─'*65}")
    set_traj_phase('return')
    print(f"  Turno 1/2 — LÍDER regresando...")
    _fly_to_wp(master_leader, LEADER_SYSID, LEADER_COMPID,
               L_x, L_y, L_z, get_leader, label='LIDER regreso', timeout=RETURN_TIMEOUT)
    print(f"  Turno 2/2 — SEGUIDOR regresando...")
    _fly_to_wp(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
               S_x, S_y, S_z, get_follower, label='SEGUIDOR regreso', timeout=RETURN_TIMEOUT)
    print(f"  Regreso completado.")

# ══════════════════════════════════════════════════════════════════════════════
# HILO A — CÍRCULO DEL LÍDER (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def _thread_circle(master_leader, leader_sp_q):
    global _leader_phase
    dt       = 1.0 / LEADER_RATE
    duration = 2 * math.pi / ANGULAR_SPEED
    steps    = int(duration / dt)

    while True:
        if state_ready(get_leader()): break
        time.sleep(0.05)

    cx, cy = 0.0, 0.0
    theta0 = math.pi

    print(f"\n[LIDER] Centro círculo: ({cx:.2f}, {cy:.2f}) ENU")
    print(f"[LIDER] R={RADIUS}m  ω={ANGULAR_SPEED}rad/s  T={duration:.1f}s")

    next_t = time.monotonic()
    _leader_phase = 'circle'
    set_traj_phase('circle')
    x_sp = cx + RADIUS * math.cos(theta0)
    y_sp = cy + RADIUS * math.sin(theta0)
    yaw  = theta0 + math.pi / 2

    for i in range(steps):
        if _stop_all.is_set(): return
        t_s   = i * dt
        theta = theta0 + ANGULAR_SPEED * t_s
        x_sp  = cx + RADIUS * math.cos(theta)
        y_sp  = cy + RADIUS * math.sin(theta)
        yaw   = theta + math.pi / 2
        send_pos_enu_abs(master_leader, LEADER_SYSID, LEADER_COMPID,
                         x_sp, y_sp, PREPOS_Z_LEADER, yaw)
        with _lock:
            leader_sp_q['x_sp'] = x_sp
            leader_sp_q['y_sp'] = y_sp
        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    x_final, y_final = x_sp, y_sp
    _leader_phase = 'tail'
    t_tail    = time.monotonic()
    t_in_zone = None
    print("[LÍDER] Fase cola...")

    while not _stop_all.is_set():
        if time.monotonic() - t_tail > CONV_TIMEOUT:
            print("[LÍDER] ⚠️  Timeout cola."); break
        L = get_leader()
        if state_ready(L):
            dist  = math.hypot(L['x'] - x_final, L['y'] - y_final)
            speed = math.hypot(L['vx'], L['vy'])
            send_pos_enu_abs(master_leader, LEADER_SYSID, LEADER_COMPID,
                             x_final, y_final, PREPOS_Z_LEADER, yaw)
            with _lock:
                leader_sp_q['x_sp'] = x_final
                leader_sp_q['y_sp'] = y_final
            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None: t_in_zone = time.monotonic()
                elif time.monotonic() - t_in_zone >= CONV_HOLD:
                    print(f"[LÍDER] ✅ Convergido dist={dist:.3f}m"); break
            else:
                t_in_zone = None
        time.sleep(dt)

    _circle_done.set()
    print("[LÍDER] 🏁 Trayectoria finalizada.")

# ══════════════════════════════════════════════════════════════════════════════
# HILO B — PID DEL SEGUIDOR (igual que v1, log de control intacto)
# ══════════════════════════════════════════════════════════════════════════════
def _thread_pid(master_follower, start_time_ref, leader_sp_q):
    """
    Log de control: idéntico a v1.
    Solo se llena cuando el PID está activo (durante el círculo).
    """
    dt  = 1.0 / FOLLOWER_RATE
    pid = PIDState()

    print("[SEGUIDOR] Esperando telemetría...", end='', flush=True)
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if _stop_all.is_set(): return
        time.sleep(0.05)
    print(" ✅")

    next_t = time.monotonic()

    while not _stop_all.is_set():
        L = get_leader()
        S = get_follower()

        if state_ready(L) and state_ready(S):
            c = compute_control(L, S, pid, dt)
            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                             c['vx'], c['vy'], c['vz_ned'], c['yaw_rate_cmd'])

            t_el    = time.monotonic() - start_time_ref
            ts_unix = time.time()
            dist_xy = math.hypot(L['x'] - S['x'], L['y'] - S['y'])
            dist_z  = abs(L['z'] - S['z'])
            err_xy  = math.hypot(c['ex'], c['ey'])

            with _lock:
                lxsp = leader_sp_q.get('x_sp', L['x'])
                lysp = leader_sp_q.get('y_sp', L['y'])

            with _log_lock:
                row = {
                    'timestamp_unix': f'{ts_unix:.6f}',
                    't':              f'{t_el:.4f}',
                    'leader_phase':    _leader_phase,
                    'lx':  f'{L["x"]:.4f}', 'ly': f'{L["y"]:.4f}', 'lz': f'{L["z"]:.4f}',
                    'lvx': f'{L["vx"]:.4f}','lvy':f'{L["vy"]:.4f}','lvz':f'{L["vz"]:.4f}',
                    'l_yaw':     f'{L["yaw"]:.4f}',
                    'l_yawrate': f'{L["yaw_rate"]:.4f}',
                    'lx_sp': f'{lxsp:.4f}', 'ly_sp': f'{lysp:.4f}',
                    'l_lat': f'{L.get("lat",0):.8f}',
                    'l_lon': f'{L.get("lon",0):.8f}',
                    'l_alt': f'{L.get("alt",0):.3f}',
                    'sx':  f'{S["x"]:.4f}', 'sy': f'{S["y"]:.4f}', 'sz': f'{S["z"]:.4f}',
                    'svx': f'{S["vx"]:.4f}','svy':f'{S["vy"]:.4f}','svz':f'{S["vz"]:.4f}',
                    's_yaw':     f'{S["yaw"]:.4f}',
                    's_yawrate': f'{S["yaw_rate"]:.4f}',
                    's_lat': f'{S.get("lat",0):.8f}',
                    's_lon': f'{S.get("lon",0):.8f}',
                    's_alt': f'{S.get("alt",0):.3f}',
                    'xd': f'{c["xd"]:.4f}', 'yd': f'{c["yd"]:.4f}', 'zd': f'{c["zd"]:.4f}',
                    'ex': f'{c["ex"]:.4f}', 'ey': f'{c["ey"]:.4f}', 'ez': f'{c["ez"]:.4f}',
                    'err_xy': f'{err_xy:.4f}', 'err_z': f'{abs(c["ez"]):.4f}',
                    'e_yaw':   f'{c["e_yaw"]:.4f}',
                    'dist_xy': f'{dist_xy:.4f}', 'dist_z': f'{dist_z:.4f}',
                    'ff_x': f'{c["ff_x"]:.4f}', 'ff_y': f'{c["ff_y"]:.4f}',
                    'vx_p': f'{c["px"]:.4f}', 'vy_p': f'{c["py"]:.4f}', 'vz_p': f'{c["pz"]:.4f}',
                    'vx_i': f'{c["ix"]:.4f}', 'vy_i': f'{c["iy"]:.4f}', 'vz_i': f'{c["iz"]:.4f}',
                    'vx_d': f'{c["dx"]:.4f}', 'vy_d': f'{c["dy"]:.4f}', 'vz_d': f'{c["dz"]:.4f}',
                    'vx_cmd': f'{c["vx"]:.4f}', 'vy_cmd': f'{c["vy"]:.4f}',
                    'vz_cmd': f'{-c["vz_ned"]:.4f}',
                    'yaw_rate_cmd': f'{c["yaw_rate_cmd"]:.4f}',
                }
                for k in _LOG_COLS:
                    _log[k].append(row[k])

            if int(t_el) % 5 == 0 and int(t_el - dt) % 5 != 0:
                print(f"  t={t_el:6.1f}s | err_xy={err_xy:.3f}m  dist={dist_xy:.2f}m"
                      f"  e_yaw={math.degrees(c['e_yaw']):.1f}°"
                      f"  cmd=({c['vx']:.2f},{c['vy']:.2f})")

        if _circle_done.is_set(): break
        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0, 0, 0, 0)
    print("[SEGUIDOR] ⏹️  Control detenido.")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV DE CONTROL (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def save_control_csv() -> str:
    fname = f'lf_circulo_{int(time.time())}.csv'
    with _log_lock:
        rows = list(zip(*[_log[k] for k in _LOG_COLS]))
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_LOG_COLS)
        w.writerows(rows)
    print(f"📄 Control CSV: {fname}  ({len(rows)} muestras, {len(_LOG_COLS)} columnas)")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# REPLAY RVIZ2 — generado a partir del log de TRAYECTORIA (no del de control)
# ══════════════════════════════════════════════════════════════════════════════
def generate_rviz_replay(traj_csv_fname: str) -> str:
    """
    Genera un script de replay RViz2 a partir del CSV de trayectoria completa.
    El CSV de trayectoria tiene datos desde el inicio (despegue, prepos, círculo).
    Columnas usadas: timestamp_unix, t, phase, lx, ly, lz, l_yaw, sx, sy, sz, s_yaw
    """
    replay_fname = traj_csv_fname.replace('.csv', '_rviz_replay.py')
    code = f'''#!/usr/bin/env python3
"""
Replay RViz2 — generado desde {traj_csv_fname}
Lee el CSV de trayectoria completa (despegue + prepos + círculo)
y publica los datos en tópicos ROS2 para visualización.

Tópicos publicados:
  /leader/path          — trayectoria completa del líder
  /follower/path        — trayectoria completa del seguidor
  /leader/pose_current  — pose actual del líder (para flecha de orientación)
  /follower/pose_current— pose actual del seguidor
  /formation/marker     — línea L→S y texto de fase

Uso:
  ros2 run ... python3 {replay_fname}
  (o simplemente: python3 {replay_fname})
"""
import csv, time, math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as RosTime

CSV_FILE = "{traj_csv_fname}"
FRAME_ID = "map"
SPEED    = 1.0   # 1.0 = tiempo real, 2.0 = doble velocidad

# Colores por fase para el marcador de línea L-S
PHASE_COLORS = {{
    "init":      (0.6, 0.6, 0.6),   # gris
    "takeoff_L": (0.2, 0.8, 0.2),   # verde
    "takeoff_S": (0.2, 0.8, 0.6),   # verde-azul
    "prepos":    (1.0, 0.8, 0.0),   # amarillo
    "circle":    (1.0, 0.4, 0.0),   # naranja
    "return":    (0.4, 0.4, 1.0),   # azul claro
}}

def _stamp(t_unix):
    sec = int(t_unix)
    ns  = int((t_unix - sec) * 1e9)
    ts  = RosTime(); ts.sec = sec; ts.nanosec = ns
    return ts

def _pose(x, y, z, yaw, stamp):
    ps = PoseStamped()
    ps.header.frame_id = FRAME_ID
    ps.header.stamp    = stamp
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    cy = math.cos(float(yaw) / 2)
    sy = math.sin(float(yaw) / 2)
    ps.pose.orientation.w = cy
    ps.pose.orientation.z = sy
    return ps

class ReplayNode(Node):
    def __init__(self, rows):
        super().__init__("lf_traj_replay")
        self.rows = rows

        # Trayectorias acumuladas
        self.pub_L  = self.create_publisher(Path, "/leader/path",   10)
        self.pub_S  = self.create_publisher(Path, "/follower/path", 10)

        # Pose actual (para flechas de orientación en RViz2)
        self.pub_Lc = self.create_publisher(PoseStamped, "/leader/pose_current",   10)
        self.pub_Sc = self.create_publisher(PoseStamped, "/follower/pose_current", 10)

        # Marcadores (línea L-S y texto de fase)
        self.pub_mk = self.create_publisher(MarkerArray, "/formation/marker", 10)

        self.path_L = Path(); self.path_L.header.frame_id = FRAME_ID
        self.path_S = Path(); self.path_S.header.frame_id = FRAME_ID

        self.idx    = 0
        self.t0_ros = None
        self.t0_dat = None

        self.create_timer(0.05, self._tick)   # 20 Hz de replay

    def _tick(self):
        if self.idx >= len(self.rows):
            return

        row   = self.rows[self.idx]
        t_rel = float(row["t"])

        # Sincronizar tiempo
        if self.t0_ros is None:
            self.t0_ros = time.monotonic()
            self.t0_dat = t_rel

        elapsed = (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat
        if t_rel > elapsed:
            return   # aún no es el momento de publicar esta muestra

        stamp = _stamp(float(row["timestamp_unix"]))
        phase = row.get("phase", "circle")

        lx, ly, lz = row["lx"], row["ly"], row["lz"]
        sx, sy, sz = row["sx"], row["sy"], row["sz"]
        l_yaw      = float(row["l_yaw"])
        s_yaw      = float(row["s_yaw"])

        # Acumular trayectorias
        self.path_L.poses.append(_pose(lx, ly, lz, l_yaw, stamp))
        self.path_S.poses.append(_pose(sx, sy, sz, s_yaw, stamp))
        self.path_L.header.stamp = stamp
        self.path_S.header.stamp = stamp

        self.pub_L.publish(self.path_L)
        self.pub_S.publish(self.path_S)

        # Pose actual
        self.pub_Lc.publish(_pose(lx, ly, lz, l_yaw, stamp))
        self.pub_Sc.publish(_pose(sx, sy, sz, s_yaw, stamp))

        # Marcadores
        color = PHASE_COLORS.get(phase, (0.5, 0.5, 0.5))
        ma    = MarkerArray()

        # Línea L → S
        mk_line = Marker()
        mk_line.header.frame_id = FRAME_ID
        mk_line.header.stamp    = stamp
        mk_line.ns   = "formation_line"
        mk_line.id   = 0
        mk_line.type = Marker.LINE_LIST
        mk_line.action = Marker.ADD
        mk_line.scale.x  = 0.06
        mk_line.color.r  = color[0]; mk_line.color.g = color[1]
        mk_line.color.b  = color[2]; mk_line.color.a = 0.9
        p1 = Point(); p1.x = float(lx); p1.y = float(ly); p1.z = float(lz)
        p2 = Point(); p2.x = float(sx); p2.y = float(sy); p2.z = float(sz)
        mk_line.points = [p1, p2]
        ma.markers.append(mk_line)

        # Texto de fase (sobre el líder)
        mk_txt = Marker()
        mk_txt.header.frame_id = FRAME_ID
        mk_txt.header.stamp    = stamp
        mk_txt.ns   = "phase_label"
        mk_txt.id   = 1
        mk_txt.type = Marker.TEXT_VIEW_FACING
        mk_txt.action = Marker.ADD
        mk_txt.pose.position.x = float(lx)
        mk_txt.pose.position.y = float(ly)
        mk_txt.pose.position.z = float(lz) + 0.5
        mk_txt.scale.z  = 0.3
        mk_txt.color.r  = 1.0; mk_txt.color.g = 1.0
        mk_txt.color.b  = 1.0; mk_txt.color.a = 0.9
        mk_txt.text = phase
        ma.markers.append(mk_txt)

        self.pub_mk.publish(ma)
        self.idx += 1


def main():
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV vacío."); return

    dur = float(rows[-1]["t"]) - float(rows[0]["t"])
    print(f"Cargadas {{len(rows)}} muestras. Duración: {{dur:.1f}}s")
    print(f"Fases registradas: {{set(r['phase'] for r in rows)}}")
    print(f"Velocidad replay: {{SPEED}}x")
    print("Tópicos: /leader/path  /follower/path  /formation/marker  ...")

    rclpy.init()
    node = ReplayNode(rows)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
'''
    with open(replay_fname, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"🤖 Replay RViz2: {replay_fname}")
    return replay_fname

# ══════════════════════════════════════════════════════════════════════════════
# LIVE PLOT (igual que v1)
# ══════════════════════════════════════════════════════════════════════════════
def _init_live_plot_traj():
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Trayectoria XY — Tiempo Real', fontsize=12)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.4)
    ln_th, = ax.plot([], [], 'k:', lw=1,   label='Teórico')
    ln_L,  = ax.plot([], [], 'g-', lw=2,   label='Líder')
    ln_sp, = ax.plot([], [], 'r--',lw=1.2, label='Setpoint S', alpha=0.7)
    ln_S,  = ax.plot([], [], 'b-', lw=2,   label='Seguidor')
    ax.legend(fontsize=8, loc='upper right')
    lines = dict(xy_th=ln_th, xy_L=ln_L, xy_sp=ln_sp, xy_S=ln_S)
    axes  = dict(xy=ax)
    return fig, axes, lines


def _init_live_plot_full():
    plt.ion()
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Control Líder-Seguidor — Tiempo Real', fontsize=12)
    gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.35)
    ax_xy  = fig.add_subplot(gs[:, 0])
    ax_x   = fig.add_subplot(gs[0, 1])
    ax_y   = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_z   = fig.add_subplot(gs[2, 1], sharex=ax_x)
    ax_err = fig.add_subplot(gs[3, 1], sharex=ax_x)
    ax_px  = fig.add_subplot(gs[4, 1], sharex=ax_x)
    ax_py  = fig.add_subplot(gs[5, 1], sharex=ax_x)

    ax_xy.set_title('Trayectoria XY')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.4)
    ln_xy_th, = ax_xy.plot([], [], 'k:', lw=1,   label='Teórico')
    ln_xy_L,  = ax_xy.plot([], [], 'g-', lw=2,   label='Líder')
    ln_xy_sp, = ax_xy.plot([], [], 'r--',lw=1.2, label='Setpoint S', alpha=0.7)
    ln_xy_S,  = ax_xy.plot([], [], 'b-', lw=2,   label='Seguidor')
    ax_xy.legend(fontsize=8, loc='upper right')

    ax_x.set_title('X vs t'); ax_x.set_ylabel('X [m]'); ax_x.grid(True, alpha=0.4)
    ln_x_l,  = ax_x.plot([], [], 'g-',  lw=1.5, label='Líder')
    ln_x_sp, = ax_x.plot([], [], 'r--', lw=1.2, label='Setpoint S')
    ln_x_s,  = ax_x.plot([], [], 'b-',  lw=1.5, label='Seguidor')
    ax_x.legend(fontsize=7)

    ax_y.set_title('Y vs t'); ax_y.set_ylabel('Y [m]'); ax_y.grid(True, alpha=0.4)
    ln_y_l,  = ax_y.plot([], [], 'g-',  lw=1.5)
    ln_y_sp, = ax_y.plot([], [], 'r--', lw=1.2)
    ln_y_s,  = ax_y.plot([], [], 'b-',  lw=1.5)

    ax_z.set_title('Z vs t'); ax_z.set_ylabel('Z [m]'); ax_z.grid(True, alpha=0.4)
    ln_z_l,  = ax_z.plot([], [], 'g-',  lw=1.5)
    ln_z_sp, = ax_z.plot([], [], 'r--', lw=1.2)
    ln_z_s,  = ax_z.plot([], [], 'b-',  lw=1.5)

    ax_err.set_title('Errores'); ax_err.set_ylabel('Error [m]')
    ax_err.axhline(0, color='k', ls='--', lw=0.7); ax_err.grid(True, alpha=0.4)
    ln_ex, = ax_err.plot([], [], lw=1.2, label='ex', color='tab:red')
    ln_ey, = ax_err.plot([], [], lw=1.2, label='ey', color='tab:blue')
    ln_ez, = ax_err.plot([], [], lw=1.2, label='ez', color='tab:green')
    ax_err.legend(fontsize=7)

    ax_px.set_title('PID Eje X'); ax_px.set_ylabel('m/s')
    ax_px.axhline(0, color='gray', ls='--', lw=0.6); ax_px.grid(True, alpha=0.4)
    ln_ffx, = ax_px.plot([], [], lw=1.2, label='FF',  color='tab:orange')
    ln_px,  = ax_px.plot([], [], lw=1.2, label='P',   color='tab:blue')
    ln_dx,  = ax_px.plot([], [], lw=1.2, label='D',   color='tab:green')
    ln_cx,  = ax_px.plot([], [], 'k-',   lw=1.8, label='Cmd', alpha=0.8)
    ax_px.legend(fontsize=7)

    ax_py.set_title('PID Eje Y'); ax_py.set_ylabel('m/s'); ax_py.set_xlabel('Tiempo [s]')
    ax_py.axhline(0, color='gray', ls='--', lw=0.6); ax_py.grid(True, alpha=0.4)
    ln_ffy, = ax_py.plot([], [], lw=1.2, color='tab:orange')
    ln_py,  = ax_py.plot([], [], lw=1.2, color='tab:blue')
    ln_dy,  = ax_py.plot([], [], lw=1.2, color='tab:green')
    ln_cy,  = ax_py.plot([], [], 'k-',   lw=1.8, alpha=0.8)

    lines = dict(
        xy_th=ln_xy_th, xy_L=ln_xy_L, xy_sp=ln_xy_sp, xy_S=ln_xy_S,
        x_l=ln_x_l, x_sp=ln_x_sp, x_s=ln_x_s,
        y_l=ln_y_l, y_sp=ln_y_sp, y_s=ln_y_s,
        z_l=ln_z_l, z_sp=ln_z_sp, z_s=ln_z_s,
        ex=ln_ex, ey=ln_ey, ez=ln_ez,
        ffx=ln_ffx, px=ln_px, dx=ln_dx, cx=ln_cx,
        ffy=ln_ffy, py=ln_py, dy=ln_dy, cy=ln_cy,
    )
    axes = dict(xy=ax_xy, x=ax_x, y=ax_y, z=ax_z, err=ax_err, px=ax_px, py=ax_py)
    return fig, axes, lines


def _update_live_plot(fig, axes, lines, traj_only=False):
    """Actualiza el live plot usando el log de CONTROL (cuando esté activo)."""
    with _log_lock:
        if len(_log['t']) < 2: return
        t    = np.array([float(v) for v in _log['t']])
        lx   = np.array([float(v) for v in _log['lx']])
        ly   = np.array([float(v) for v in _log['ly']])
        lz   = np.array([float(v) for v in _log['lz']])
        sx   = np.array([float(v) for v in _log['sx']])
        sy   = np.array([float(v) for v in _log['sy']])
        sz   = np.array([float(v) for v in _log['sz']])
        xd   = np.array([float(v) for v in _log['xd']])
        yd   = np.array([float(v) for v in _log['yd']])
        zd   = np.array([float(v) for v in _log['zd']])
        if not traj_only:
            ex   = np.array([float(v) for v in _log['ex']])
            ey   = np.array([float(v) for v in _log['ey']])
            ez   = np.array([float(v) for v in _log['ez']])
            ff_x = np.array([float(v) for v in _log['ff_x']])
            ff_y = np.array([float(v) for v in _log['ff_y']])
            vx_p = np.array([float(v) for v in _log['vx_p']])
            vy_p = np.array([float(v) for v in _log['vy_p']])
            vx_d = np.array([float(v) for v in _log['vx_d']])
            vy_d = np.array([float(v) for v in _log['vy_d']])
            vx_c = np.array([float(v) for v in _log['vx_cmd']])
            vy_c = np.array([float(v) for v in _log['vy_cmd']])

    th = np.linspace(0, 2 * math.pi, 300)
    lines['xy_th'].set_data(RADIUS * np.sin(th), RADIUS * np.cos(th))
    lines['xy_L'].set_data(ly, lx)
    lines['xy_sp'].set_data(yd, xd)
    lines['xy_S'].set_data(sy, sx)

    if not traj_only:
        for ln_l, ln_sp, ln_s, dl, dd, ds in [
            (lines['x_l'], lines['x_sp'], lines['x_s'], lx, xd, sx),
            (lines['y_l'], lines['y_sp'], lines['y_s'], ly, yd, sy),
            (lines['z_l'], lines['z_sp'], lines['z_s'], lz, zd, sz),
        ]:
            ln_l.set_data(t, dl); ln_sp.set_data(t, dd); ln_s.set_data(t, ds)
        lines['ex'].set_data(t, ex); lines['ey'].set_data(t, ey); lines['ez'].set_data(t, ez)
        lines['ffx'].set_data(t, ff_x); lines['px'].set_data(t, vx_p)
        lines['dx'].set_data(t, vx_d);  lines['cx'].set_data(t, vx_c)
        lines['ffy'].set_data(t, ff_y); lines['py'].set_data(t, vy_p)
        lines['dy'].set_data(t, vy_d);  lines['cy'].set_data(t, vy_c)

    for ax in axes.values():
        ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS POST-VUELO (igual que v1, sin cambios)
# ══════════════════════════════════════════════════════════════════════════════
C_refL = np.array([0.40, 0.65, 1.00])
C_refS = np.array([1.00, 0.60, 0.40])
C_L    = np.array([0.10, 0.35, 0.75])
C_S    = np.array([0.80, 0.15, 0.10])
C_dist = np.array([0.25, 0.75, 0.45])
C_form = np.array([0.50, 0.50, 0.50])
LW, LWS = 2.0, 1.2
SC_DRONE = 0.40
SC_RTK   = SC_DRONE * 2.0


def _wrap_np(arr):
    return np.arctan2(np.sin(arr), np.cos(arr))


# yaw MAVLink: 0=Norte, π/2=Este (sentido horario desde arriba)
# En el plot: horizontal=Este=sin(yaw_mav), vertical=Norte=cos(yaw_mav)

def _draw_drone_frame(ax, x_plot, y_plot, yaw_mav, scale=SC_DRONE, lw=1.2):
    """
    x_plot, y_plot : coordenadas en el plot (Este, Norte)
    yaw_mav        : yaw MAVLink (0=Norte, crece Este, radianes)
    """
    # Frente del drone en coordenadas del plot
    dx_front = scale * np.sin(yaw_mav)   # componente Este (horizontal)
    dy_front = scale * np.cos(yaw_mav)   # componente Norte (vertical)
    # Lateral izquierdo = frente rotado +90° en ENU
    dx_left  = scale * np.sin(yaw_mav + np.pi/2)
    dy_left  = scale * np.cos(yaw_mav + np.pi/2)

    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=4, headlength=3.5, headaxislength=3, alpha=0.85)
    ax.quiver(x_plot, y_plot, dx_front, dy_front,
              color=[0.85, 0.10, 0.10], **kw)   # rojo = frente
    ax.quiver(x_plot, y_plot, dx_left,  dy_left,
              color=[0.10, 0.65, 0.10], **kw)   # verde = lateral izq

def _draw_rtk_frame(ax, scale=SC_RTK):
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.007, headwidth=4, headlength=4, headaxislength=3.5, alpha=0.9)
    ax.quiver(0, 0, scale, 0, color=[0.85, 0.10, 0.10], **kw)
    ax.quiver(0, 0, 0, scale, color=[0.10, 0.65, 0.10], **kw)
    ax.plot(0, 0, 'ok', ms=6, zorder=5)
    ax.text(scale+0.12, -0.10, r'$\hat{x}_{ENU}$', fontsize=10, fontweight='bold', va='top')
    ax.text(-0.10, scale+0.10, r'$\hat{y}_{ENU}$', fontsize=10, fontweight='bold', ha='right')


def plot_results():
    """Gráficas post-vuelo usando el log de CONTROL (igual que v1)."""
    def arr(key): return np.array([float(v) for v in _log[key]])

    with _log_lock:
        t              = arr('t')
        lx, ly, lz     = arr('lx'), arr('ly'), arr('lz')
        sx, sy, sz      = arr('sx'), arr('sy'), arr('sz')
        xd, yd, zd      = arr('xd'), arr('yd'), arr('zd')
        lxsp, lysp      = arr('lx_sp'), arr('ly_sp')
        ex, ey, ez      = arr('ex'), arr('ey'), arr('ez')
        e_yaw           = arr('e_yaw')
        dist_xy         = arr('dist_xy')
        dist_z          = arr('dist_z')
        psiL, psiS      = arr('l_yaw'), arr('s_yaw')
        ff_x, ff_y      = arr('ff_x'), arr('ff_y')
        vx_p, vy_p      = arr('vx_p'), arr('vy_p')
        vx_d, vy_d      = arr('vx_d'), arr('vy_d')
        vx_cmd, vy_cmd  = arr('vx_cmd'), arr('vy_cmd')

    if len(t) == 0:
        print("⚠️  Sin datos de control para graficar."); return

    n        = len(t)
    psidesS  = psiL.copy()
    err_xy   = np.hypot(ex, ey)

    def rms(v): return np.sqrt(np.mean(v**2)) if len(v) > 0 else 0.0

    print('\n' + '='*50)
    print('  MÉTRICAS DE VUELO (fase círculo)')
    print('='*50)
    print(f'  RMS error x  : {rms(ex):.4f} m')
    print(f'  RMS error y  : {rms(ey):.4f} m')
    print(f'  RMS error z  : {rms(ez):.4f} m')
    print(f'  RMS error ψ  : {rms(e_yaw):.4f} rad')
    print(f'  RMS error xy : {rms(err_xy):.4f} m')
    print(f'  Dist media L-S: {np.mean(dist_xy):.4f} m  (objetivo={OFFSET_D} m)')
    print(f'  Error formación RMS: {rms(dist_xy - OFFSET_D):.4f} m')
    print('='*50 + '\n')

    ###

    # ── Después de las métricas existentes, antes de plt.ioff() ──────────────────

    # Paso de tiempo (no uniforme en general, usamos diferencias)
    dt_arr = np.diff(t, prepend=t[0])   # Δt en cada muestra



    # IAE por eje
    iae_x   = np.trapezoid(np.abs(ex),  t)
    iae_y   = np.trapezoid(np.abs(ey),  t)
    iae_z   = np.trapezoid(np.abs(ez),  t)
    iae_xy  = np.trapezoid(np.hypot(ex, ey), t)
    iae_yaw = np.trapezoid(np.abs(e_yaw), t)



    # IAE acumulado en el tiempo (para graficar evolución)
    iae_x_cum  = np.cumsum(np.abs(ex)  * dt_arr)
    iae_y_cum  = np.cumsum(np.abs(ey)  * dt_arr)
    iae_z_cum  = np.cumsum(np.abs(ez)  * dt_arr)
    iae_xy_cum = np.cumsum(np.hypot(ex, ey) * dt_arr)

    # IAE de error de formación (distancia real vs deseada)
    iae_form     = np.trapezoid(np.abs(dist_xy - OFFSET_D), t)
    iae_form_cum = np.cumsum(np.abs(dist_xy - OFFSET_D) * dt_arr)

    # Imprimir
    print(f'  IAE error x        : {iae_x:.4f} m·s')
    print(f'  IAE error y        : {iae_y:.4f} m·s')
    print(f'  IAE error z        : {iae_z:.4f} m·s')
    print(f'  IAE error XY plano : {iae_xy:.4f} m·s')
    print(f'  IAE error formación: {iae_form:.4f} m·s')
    print(f'  IAE error yaw      : {iae_yaw:.4f} rad·s')


    # ── Fig N — IAE acumulado ────────────────────────────────────────────────────
    figN, axsN = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor='white')
    figN.suptitle('IAE Acumulado del Seguidor — fase círculo',
                fontsize=13, fontweight='bold')

    axsN[0].plot(t, iae_x_cum,  '-', color=C_S,    lw=LW, label=f'IAE x  = {iae_x:.3f} m·s')
    axsN[0].plot(t, iae_y_cum,  '-', color=C_L,    lw=LW, label=f'IAE y  = {iae_y:.3f} m·s')
    axsN[0].plot(t, iae_z_cum,  '-', color=C_dist, lw=LW, label=f'IAE z  = {iae_z:.3f} m·s')
    axsN[0].set_ylabel('IAE [m·s]')
    axsN[0].legend(fontsize=9); axsN[0].grid(True, alpha=0.3)
    axsN[0].set_title('IAE por eje (x, y, z)')

    axsN[1].plot(t, iae_xy_cum, '-', color=[0.5, 0.1, 0.7], lw=LW,
                label=f'IAE XY = {iae_xy:.3f} m·s')
    axsN[1].set_ylabel('IAE [m·s]')
    axsN[1].legend(fontsize=9); axsN[1].grid(True, alpha=0.3)
    axsN[1].set_title('IAE error planar euclidiano ||exy||')

    axsN[2].plot(t, iae_form_cum, '-', color=C_form, lw=LW,
                label=f'IAE formación = {iae_form:.3f} m·s')
    axsN[2].axhline(0, color='k', ls='--', lw=0.6)
    axsN[2].set_ylabel('IAE [m·s]')
    axsN[2].set_xlabel('Tiempo [s]')
    axsN[2].legend(fontsize=9); axsN[2].grid(True, alpha=0.3)
    axsN[2].set_title(f'IAE error de formación |dist_xy − {OFFSET_D:.1f}m|')

    figN.tight_layout()

    ##

    plt.ioff()
    th      = np.linspace(0, 2*math.pi, 400)
    n_d     = 20
    idx_d   = np.round(np.linspace(0, n-1, n_d+2)).astype(int)[1:-1]
    n_fr    = 6
    idx_fr  = np.round(np.linspace(0, n-1, n_fr)).astype(int)

    
    ###########################################################
    
    
    # Fig 1 — Trayectoria XY completa
    fig1, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.plot(lysp, lxsp, '--', color=C_refL, lw=LWS, label='Setpoint Líder', alpha=0.8)
    ax.plot(yd,   xd,   '--', color=C_refS, lw=LWS, label='Setpoint Seguidor', alpha=0.8)
    ax.plot(ly,   lx,   '-',  color=C_L,    lw=LW,  label='Líder real')
    ax.plot(sy,   sx,   '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax.plot(RADIUS*np.sin(th), RADIUS*np.cos(th), ':', color=[0.6,0.6,0.6], lw=1.0, label='Teórico')
    ax.plot(ly[0], lx[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Inicio Líder')
    ax.plot(sy[0], sx[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Inicio Seguidor')
    ax.plot(0, 0, '+k', ms=10, mew=2, label='Centro (0,0)')

    for i in idx_d:
        ax.quiver(ly[i], lx[i],
                sy[i]-ly[i], sx[i]-lx[i],
                angles='xy', scale_units='xy', scale=1,
                color=C_form, width=0.003, headwidth=3.5,
                headlength=3.5, headaxislength=3, alpha=0.65)

    for i in idx_fr:
        _draw_drone_frame(ax, ly[i], lx[i], psiL[i])
        _draw_drone_frame(ax, sy[i], sx[i], psiS[i])

    _draw_rtk_frame(ax)

    ax.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}_D$ (frente)')
    ax.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}_D$ (lateral)')
    ax.plot([], [], '-', color=C_form, lw=1.5, label=r'Vector $\mathbf{d}$')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlabel(r'$y_{ENU}$ — Este [m]', fontsize=11)
    ax.set_ylabel(r'$x_{ENU}$ — Norte [m]', fontsize=11)
    ax.set_title('Trayectoria XY — fase círculo', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9); fig1.tight_layout()

    # Fig 2 — Posición y orientación vs tiempo
    fig2, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True, facecolor='white')
    fig2.suptitle('Posición y orientación vs Tiempo (fase círculo)', fontsize=13, fontweight='bold')
    for a, dl, dd, ds, lbl in zip(axs,
            [lx, ly, lz, np.degrees(psiL)],
            [lxsp, lysp, [PREPOS_Z_LEADER]*n, np.degrees(psidesS)],
            [sx, sy, sz, np.degrees(psiS)],
            ['x [m]', 'y [m]', 'z [m]', 'ψ [°]']):
        a.plot(t, dd, '--', color=C_refL, lw=LWS, label='Setpoint L')
        a.plot(t, dl, '-',  color=C_L,    lw=LW,  label='Líder')
        a.plot(t, ds, '-',  color=C_S,    lw=LW,  label='Seguidor')
        a.set_ylabel(lbl); a.grid(True, alpha=0.3); a.legend(fontsize=8)
    axs[-1].set_xlabel('Tiempo [s]'); fig2.tight_layout()

    # Fig 3 — Errores del seguidor
    fig3, axs = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig3.suptitle('Errores del Seguidor (fase círculo)', fontsize=13, fontweight='bold')
    for a, data, lbl in zip(axs,
            [ex, ey, ez, np.degrees(e_yaw)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        a.plot(t, data, '-', color=C_S, lw=LW)
        a.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        a.set_ylabel(lbl); a.grid(True, alpha=0.3)
    axs[-1].set_xlabel('Tiempo [s]'); fig3.tight_layout()

    # Fig 4 — Distancia L-S
    fig4, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig4.suptitle('Distancia de Formación L-S (fase círculo)', fontsize=13, fontweight='bold')
    axs[0].plot(t, dist_xy, '-', color=C_dist, lw=LW, label='Distancia XY real')
    axs[0].axhline(OFFSET_D, color='r', ls='--', lw=1.5, label=f'Objetivo d={OFFSET_D:.1f}m')
    axs[0].set_ylabel('Dist XY [m]'); axs[0].legend(fontsize=9); axs[0].grid(True, alpha=0.3)
    axs[1].plot(t, dist_z, '-', color=[0.55,0.25,0.65], lw=LW, label='Distancia Z real')
    axs[1].axhline(OFFSET_DZ, color='r', ls='--', lw=1.5, label=f'Objetivo Δz={OFFSET_DZ:.1f}m')
    axs[1].set_ylabel('Dist Z [m]'); axs[1].legend(fontsize=9); axs[1].grid(True, alpha=0.3)
    axs[2].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW, label='ψ Líder')
    axs[2].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW, label='ψ Seguidor')
    axs[2].set_ylabel('ψ [°]'); axs[2].set_xlabel('Tiempo [s]')
    axs[2].legend(fontsize=9); axs[2].grid(True, alpha=0.3); fig4.tight_layout()

    # Fig 5 — Desglose PID
    fig5, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor='white')
    fig5.suptitle('Desglose PID + Feed-forward (fase círculo)', fontsize=13, fontweight='bold')
    for a, ff, pp, dd, cmd, lb in zip(axs,
            [ff_x, ff_y], [vx_p, vy_p], [vx_d, vy_d], [vx_cmd, vy_cmd],
            ['Eje X [m/s]', 'Eje Y [m/s]']):
        a.plot(t, ff,  lw=LWS, label='Feed-forward', color=[0.85,0.55,0.10])
        a.plot(t, pp,  lw=LWS, label='Proporcional', color=C_L)
        a.plot(t, dd,  lw=LWS, label='Derivativo',   color=C_dist)
        a.plot(t, cmd, 'k-',   lw=LW,  alpha=0.85,   label='Cmd total')
        a.axhline(0, color='gray', ls='--', lw=0.6)
        a.set_ylabel(lb); a.legend(fontsize=8); a.grid(True, alpha=0.3)
    axs[-1].set_xlabel('Tiempo [s]'); fig5.tight_layout()

    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  LF_Circulo_RT_v2 — Líder-Seguidor con Despegue")
    print("=" * 65)
    print(f"  Líder    : {LEADER_CONN}  (SYSID {LEADER_SYSID})")
    print(f"  Seguidor : {FOLLOWER_CONN} (SYSID {FOLLOWER_SYSID})")
    print(f"  Círculo  : R={RADIUS}m  ω={ANGULAR_SPEED}rad/s  "
          f"T={2*math.pi/ANGULAR_SPEED:.1f}s")
    print(f"  Offset   : d={OFFSET_D}m  α={math.degrees(OFFSET_ALPHA):.0f}°  "
          f"Δz={OFFSET_DZ}m")
    print(f"  Ganancias: Kp={KP} Ki={KI} Kd={KD} | "
          f"Kp_yaw={KP_YAW} Kd_yaw={KD_YAW}")
    print(f"  Despegue : L={TAKEOFF_ALT_LEADER}m  S={TAKEOFF_ALT_FOLLOWER}m")
    print(f"  SKIP_PREPOS={SKIP_PREPOS}  SKIP_RETURN={SKIP_RETURN}  "
          f"RESET_ORIGIN={RESET_ORIGIN}")
    print("=" * 65)

    # ── Conectar ──────────────────────────────────────────────────────────────
    print(f"\n🔌 Conectando líder    ({LEADER_CONN})...")
    master_leader = mavutil.mavlink_connection(LEADER_CONN)
    master_leader.wait_heartbeat()
    print(f"   ✅ SYS={master_leader.target_system}")

    print(f"🔌 Conectando seguidor ({FOLLOWER_CONN})...")
    master_follower = mavutil.mavlink_connection(FOLLOWER_CONN)
    master_follower.wait_heartbeat()
    print(f"   ✅ SYS={master_follower.target_system}")

    # ── Arrancar lectores y log de trayectoria ────────────────────────────────
    stop_readers = threading.Event()
    thr_read_L = threading.Thread(
        target=_reader, args=(master_leader,   _leader_state,   stop_readers),
        daemon=True, name='reader-leader')
    thr_read_S = threading.Thread(
        target=_reader, args=(master_follower, _follower_state, stop_readers),
        daemon=True, name='reader-follower')
    thr_read_L.start()
    thr_read_S.start()

    prog_t0 = time.monotonic()   # t=0 de TODO el programa
    start_traj_log(prog_t0)      # arranca log de trayectoria (10 Hz desde ya)
    set_traj_phase('init')

    # ── Fijar origen ENU ──────────────────────────────────────────────────────
    origin_loaded = False
    if not RESET_ORIGIN:
        origin_loaded = load_enu_origin()

    if not origin_loaded:
        print("\n⏳ Esperando primer GPS del líder para fijar origen ENU",
              end='', flush=True)
        t0w = time.monotonic()
        while True:
            with _lock:
                lat_ok = _leader_state.get('lat') is not None
            if lat_ok: break
            if time.monotonic() - t0w > 30.0:
                print("\n❌ Timeout."); stop_readers.set()
                stop_traj_log(); raise SystemExit(1)
            print('.', end='', flush=True); time.sleep(0.2)

        with _lock:
            ref_lat = _leader_state['lat']
            ref_lon = _leader_state['lon']
            ref_alt = _leader_state['alt']
        set_enu_origin(ref_lat, ref_lon, ref_alt)
        save_enu_origin()

    # ── Esperar telemetría ENU ────────────────────────────────────────────────
    print("\n⏳ Esperando telemetría ENU de ambos drones", end='', flush=True)
    t0w = time.monotonic()
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if time.monotonic() - t0w > 30.0:
            print("\n❌ Timeout."); stop_readers.set()
            stop_traj_log(); raise SystemExit(1)
        print('.', end='', flush=True); time.sleep(0.3)
    print(" ✅")

    if not capture_leader_ned_home(master_leader):
        print("⚠️  No se pudo capturar el home NED del líder.")
        stop_readers.set(); stop_traj_log(); raise SystemExit(1)

    L, S = get_leader(), get_follower()
    print(f"   Líder    ENU: x={L['x']:+.2f}m  y={L['y']:+.2f}m  z={L['z']:.2f}m"
          f"  ψ={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor ENU: x={S['x']:+.2f}m  y={S['y']:+.2f}m  z={S['z']:.2f}m"
          f"  ψ={math.degrees(S['yaw']):.1f}°")

    # ══════════════════════════════════════════════════════════════════════════
    # DESPEGUE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    print("  FASE DE DESPEGUE")
    print("─"*65)
    print("  Asegúrate de que ambos drones están en tierra y listos.")

    # Líder
    input("\n  ENTER para iniciar despegue del LÍDER...\n")
    set_traj_phase('takeoff_L')
    do_takeoff(master_leader, LEADER_SYSID, LEADER_COMPID,
               get_leader, TAKEOFF_ALT_LEADER, 'LÍDER')

    # Seguidor
    input("\n  ENTER para iniciar despegue del SEGUIDOR...\n")
    set_traj_phase('takeoff_S')
    do_takeoff(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
               get_follower, TAKEOFF_ALT_FOLLOWER, 'SEGUIDOR')

    print("\n  ✅ Ambos drones en el aire.")

    # ══════════════════════════════════════════════════════════════════════════
    # PREPOSICIONAMIENTO
    # ══════════════════════════════════════════════════════════════════════════
    theta0      = math.pi
    yaw_inicial = theta0 + math.pi / 2
    L_x = -RADIUS;  L_y = 0.0;  L_z = PREPOS_Z_LEADER
    dx_off, dy_off, _ = compute_offset(yaw_inicial)
    S_x = L_x + dx_off;  S_y = L_y + dy_off;  S_z = PREPOS_Z_FOLLOWER

    if SKIP_PREPOS:
        print(f"\n⏭️  Preposicionamiento OMITIDO (SKIP_PREPOS=True)")
        print(f"  Líder    esperado en ENU ({L_x:.2f}, {L_y:.2f})  z={L_z:.1f}m")
        print(f"  Seguidor esperado en ENU ({S_x:.2f}, {S_y:.2f})  z={S_z:.1f}m")
        input("  Confirma que los drones están en posición → ENTER\n")
        set_traj_phase('prepos')
    else:
        input("\n  ENTER para iniciar preposicionamiento...\n")
        preposition(master_leader, master_follower)

    # ══════════════════════════════════════════════════════════════════════════
    # CÍRCULO + PID
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─"*65)
    input("  ENTER para iniciar el círculo + seguimiento...\n")

    # El t0 del log de CONTROL se mide desde aquí (igual que v1)
    circle_t0   = time.monotonic()
    leader_sp_q = {}

    thr_pid = threading.Thread(
        target=_thread_pid, args=(master_follower, circle_t0, leader_sp_q),
        daemon=True, name='pid-follower')
    thr_circle = threading.Thread(
        target=_thread_circle, args=(master_leader, leader_sp_q),
        daemon=True, name='circle-leader')

    thr_pid.start()
    thr_circle.start()
    print("🚀 En marcha! Ctrl+C = parada de emergencia.\n")

    # ── Live plot ─────────────────────────────────────────────────────────────
    if LIVE_PLOT_ENABLED:
        if LIVE_PLOT_TRAJ_ONLY:
            fig_live, axes_live, lines_live = _init_live_plot_traj()
        else:
            fig_live, axes_live, lines_live = _init_live_plot_full()

        live_dt     = 1.0 / LIVE_PLOT_RATE
        next_plot_t = time.monotonic()

        try:
            while thr_circle.is_alive() or thr_pid.is_alive():
                now = time.monotonic()
                if now >= next_plot_t:
                    _update_live_plot(fig_live, axes_live, lines_live,
                                      traj_only=LIVE_PLOT_TRAJ_ONLY)
                    next_plot_t = now + live_dt
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n🛑 EMERGENCIA...")
            _stop_all.set(); _circle_done.set()
            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0, 0, 0, 0)
            time.sleep(0.5)

        _update_live_plot(fig_live, axes_live, lines_live,
                          traj_only=LIVE_PLOT_TRAJ_ONLY)
        plt.pause(0.5)
        plt.close(fig_live)

    else:
        try:
            while thr_circle.is_alive() or thr_pid.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 EMERGENCIA...")
            _stop_all.set(); _circle_done.set()
            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0, 0, 0, 0)
            time.sleep(0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # REGRESO
    # ══════════════════════════════════════════════════════════════════════════
    if RETURN_ENABLED and not SKIP_RETURN and not _stop_all.is_set():
        return_to_start(master_leader, master_follower,
                        L_x, L_y, L_z, S_x, S_y, S_z)

    # ══════════════════════════════════════════════════════════════════════════
    # ATERRIZAJE
    # ══════════════════════════════════════════════════════════════════════════
    if not _stop_all.is_set():
        print("\n" + "─"*65)
        input("  ENTER para aterrizar el LÍDER...\n")
        set_traj_phase('land_L')
        do_land(master_leader, LEADER_SYSID, LEADER_COMPID, get_leader, 'LÍDER')

        input("\n  ENTER para aterrizar el SEGUIDOR...\n")
        set_traj_phase('land_S')
        do_land(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, get_follower, 'SEGUIDOR')

        print("\n  ✅ Ambos drones en tierra.")

    # ── Detener todo ──────────────────────────────────────────────────────────
    stop_traj_log()
    stop_readers.set()
    thr_read_L.join(timeout=2.0)
    thr_read_S.join(timeout=2.0)
    save_enu_origin()
    master_leader.close()
    master_follower.close()
    print("🔌 Conexiones cerradas.")

    # ── Guardar CSVs ──────────────────────────────────────────────────────────
    control_csv = save_control_csv()   # log de control (solo fase PID)
    traj_csv    = save_traj_csv()      # log de trayectoria (todo el vuelo)
    generate_rviz_replay(traj_csv)     # replay usa el log de trayectoria

    # ── Gráficas post-vuelo ───────────────────────────────────────────────────
    plot_results()   # gráficas usan el log de control (igual que v1)
#!/usr/bin/env python3
"""
LF_Circulo_RT.py — Líder-Seguidor en Tiempo Real con Círculo
=============================================================
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
SKIP_PREPOS  = False   # True = omitir preposicionamiento
SKIP_RETURN  = False   # True = omitir regreso al origen
RESET_ORIGIN = False   # True = fijar nuevo origen GPS (ignora enu_origin.json)

# ── Gráfica en vivo ───────────────────────────────────────────────────────────
LIVE_PLOT_ENABLED   = True    # False = desactiva completamente la ventana en vivo
LIVE_PLOT_TRAJ_ONLY = True    # True  = solo muestra la trayectoria XY (sin los 6 subplots)
LIVE_PLOT_RATE      = 4       # Hz de refresco de la ventana en vivo

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
LEADER_CONN    = 'udp:127.0.0.1:14552'
FOLLOWER_CONN  = 'udp:127.0.0.1:14553'

R_EARTH = 6371000.0

LEADER_SYSID    = 1
LEADER_COMPID   = 0
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# ══════════════════════════════════════════════════════════════════════════════
# FRAME ENU COMUN
# ══════════════════════════════════════════════════════════════════════════════
_enu_origin = {'lat': None, 'lon': None, 'alt': None}
ENU_ORIGIN_FILE = 'enu_origin.json'

# ── nuevo: captura el home NED del líder (posición LOCAL_NED cuando z≈0 en tierra) ──
_leader_ned_home = {'x': None, 'y': None}   # punto de armado en NED


def set_enu_origin(lat_deg, lon_deg, alt_m):
    _enu_origin['lat'] = lat_deg
    _enu_origin['lon'] = lon_deg
    _enu_origin['alt'] = alt_m
    print(f"\n[ENU] Origen fijado:")
    print(f"      lat={lat_deg:.8f} deg  lon={lon_deg:.8f} deg  alt={alt_m:.2f} m")


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
    print(f"[ENU] Origen CARGADO desde {ENU_ORIGIN_FILE} (corrida anterior)")
    return True


def gps_to_enu(lat_deg, lon_deg, alt_m):
    ref_lat = math.radians(_enu_origin['lat'])
    dlat    = math.radians(lat_deg - _enu_origin['lat'])
    dlon    = math.radians(lon_deg - _enu_origin['lon'])
    x =  dlat * R_EARTH
    y =  dlon * R_EARTH * math.cos(ref_lat)
    z =  alt_m - _enu_origin['alt']
    return x, y, z


def capture_leader_ned_home(master_leader):
    """Lee LOCAL_POSITION_NED del líder para conocer su home en el frame NED."""
    print("[ENU] Capturando home NED del líder...", end='', flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15.0:
        msg = master_leader.recv_match(
            type='LOCAL_POSITION_NED', blocking=True, timeout=1.0)
        if msg:
            # Convertimos el GPS actual a ENU, y la posición NED también la tenemos
            # El home NED es: donde x_ned=0, y_ned=0 → corresponde al punto de armado
            # Guardamos la posición ENU del punto de armado
            with _lock:
                lat = _leader_state.get('lat')
                lon = _leader_state.get('lon')
            if lat and _enu_origin['lat']:
                enu_x, enu_y, _ = gps_to_enu(lat, lon, 0)
                # offset = posición ENU del drone - posición NED del drone
                _leader_ned_home['x'] = enu_x - msg.x   # x_ned apunta Norte ≈ x_ENU
                _leader_ned_home['y'] = enu_y - msg.y
                print(f" ✅  offset=({_leader_ned_home['x']:.2f}, {_leader_ned_home['y']:.2f})")
                return True
    print(" ❌ timeout")
    return False


# Trayectoria líder
RADIUS        = 4.0
ANGULAR_SPEED = 0.2
LEADER_RATE   = 50

# Convergencia (fase cola)
CONV_RADIUS  = 0.15
CONV_SPEED   = 0.10
CONV_HOLD    = 1.0
CONV_TIMEOUT = 15.0

# Preposicionamiento
PREPOS_Z_LEADER   = 2.0
PREPOS_Z_FOLLOWER = 2.0
PREPOS_TIMEOUT    = 40.0
PREPOS_CONV_R     = 0.30
PREPOS_CONV_V     = 0.20
PREPOS_CONV_HOLD  = 2.0

# Regreso
RETURN_ENABLED = True
RETURN_TIMEOUT = 40.0

# PID seguidor
FOLLOWER_RATE = 20

OFFSET_D     = 1.0
OFFSET_ALPHA = -math.pi / 2
OFFSET_DZ    = 1.0

KP, KI, KD              = 0.5, 0.0, 0.0
KP_YAW, KI_YAW, KD_YAW = 0.8, 0.0, 0.1

INTEGRAL_LIMIT     = 2.0
INTEGRAL_YAW_LIMIT = 1.0
V_MAX        = 2.0
V_MAX_Z      = 1.0
YAW_RATE_MAX = 1.0

# Máscaras MAVLink
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

# Log
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

_circle_done  = threading.Event()
_stop_all     = threading.Event()
_leader_phase = 'circle'

# ══════════════════════════════════════════════════════════════════════════════
# LECTORES MAVLINK
# ══════════════════════════════════════════════════════════════════════════════
def _reader(master, state_dict, stop_event, use_enu=True):
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
                state_dict['vx'] =  msg.vx * 0.01
                state_dict['vy'] =  msg.vy * 0.01
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
    return all(s.get(k) is not None for k in ['x','y','z','vx','vy','vz','yaw','yaw_rate'])

# ══════════════════════════════════════════════════════════════════════════════
# ENVÍO DE COMANDOS
# ══════════════════════════════════════════════════════════════════════════════
def enu_global_to_ned_local(enu_x, enu_y, drone_state):
    return enu_x - drone_state['x'], enu_y - drone_state['y']

def send_pos_enu(master, sysid, compid, enu_x, enu_y, z_agl, yaw, drone_state):
    delta_x, delta_y = enu_global_to_ned_local(enu_x, enu_y, drone_state)
    delta_z_ned = -(z_agl - drone_state['z'])
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
        TYPE_MASK_POS_YAW,
        delta_x, delta_y, delta_z_ned,
        0, 0, 0, 0, 0, 0, yaw, 0)

def send_pos_enu_abs(master, sysid, compid, enu_x, enu_y, z_agl, yaw):
    """Envía posición ENU convirtiéndola a NED absoluto usando el home capturado."""
    # ENU→NED: x_ned = enu_x - home_offset_x,  y_ned = enu_y - home_offset_y
    x_ned = enu_x - _leader_ned_home['x']
    y_ned = enu_y - _leader_ned_home['y']
    z_ned = -z_agl
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x_ned, y_ned, z_ned,
        0, 0, 0, 0, 0, 0, yaw, 0)

def send_pos_yaw(master, sysid, compid, x, y, z_ned, yaw):
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z_ned, 0, 0, 0, 0, 0, 0, yaw, 0)

def send_vel_yawrate(master, sysid, compid, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0, vx, vy, vz_ned, 0, 0, 0, 0, yaw_rate)

# ══════════════════════════════════════════════════════════════════════════════
# PID
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

    px, py, pz    = KP*ex,              KP*ey,              KP*ez
    ix, iy, iz    = KI*pid.integral[0], KI*pid.integral[1], KI*pid.integral[2]
    dx_, dy_, dz_ = KD*dv_x,           KD*dv_y,            KD*dv_z

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
        KP_YAW*e_yaw + KI_YAW*pid.integral_yaw + KD_YAW*dyaw,
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
            if t_in_zone is None: t_in_zone = now
            elif now - t_in_zone >= conv_hold:
                print(f"  [{label}] ✅ Convergido en {now-t0:.1f}s  "
                      f"dist_xy={dist:.3f}m  dist_z={dist_z:.3f}m")
                return True
        else:
            t_in_zone = None
        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)
    return False

# ══════════════════════════════════════════════════════════════════════════════
# PREPOSICIONAMIENTO
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
    print(f"  yaw_inicial={math.degrees(yaw_inicial):.1f}deg")
    print(f"  Líder    → ENU ({L_x:.2f}, {L_y:.2f})  z={L_z:.1f}m AGL")
    print(f"  Seguidor → ENU ({S_x:.2f}, {S_y:.2f})  z={S_z:.1f}m AGL")
    print(f"{'─'*65}")

    print(f"\n  Turno 1/2 — SEGUIDOR moviendose...")
    ok_s = _fly_to_wp(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                      S_x, S_y, S_z, get_follower, label='SEGUIDOR prepos')
    if not ok_s:
        print("  ⚠️  Seguidor no convergió.")
    input("  ENTER para mover el LÍDER...\n")

    print(f"  Turno 2/2 — LÍDER moviendose...")
    dt        = 1.0 / 20
    t0        = time.monotonic()
    t_in_zone = None

    while not _stop_all.is_set():
        if time.monotonic() - t0 > PREPOS_TIMEOUT:
            print("  [LIDER prepos] ⚠️  Timeout."); break
        L = get_leader()
        if not state_ready(L): time.sleep(dt); continue
        send_pos_enu(master_leader, LEADER_SYSID, LEADER_COMPID,
                     L_x, L_y, L_z, yaw_inicial, drone_state=L)
        dist   = math.hypot(L['x'] - L_x, L['y'] - L_y)
        dist_z = abs(L['z'] - L_z)
        speed  = math.hypot(L['vx'], L['vy'])
        if dist < PREPOS_CONV_R and dist_z < PREPOS_CONV_R*2 and speed < PREPOS_CONV_V:
            if t_in_zone is None: t_in_zone = time.monotonic()
            elif time.monotonic() - t_in_zone >= PREPOS_CONV_HOLD:
                print(f"  [LIDER prepos] ✅ Convergido en {time.monotonic()-t0:.1f}s")
                break
        else:
            t_in_zone = None
        time.sleep(dt)

    L = get_leader(); S = get_follower()
    if state_ready(L) and state_ready(S):
        sep   = math.hypot(L['x']-S['x'], L['y']-S['y'])
        sep_z = abs(L['z'] - S['z'])
        print(f"\n  Preposicionamiento completado.")
        print(f"  Sep L-S: {sep:.2f}m horiz  {sep_z:.2f}m vert  "
              f"(objetivo d={OFFSET_D}m dz={OFFSET_DZ}m)")
    return L_x, L_y, L_z, S_x, S_y, S_z

# ══════════════════════════════════════════════════════════════════════════════
# REGRESO
# ══════════════════════════════════════════════════════════════════════════════
def return_to_start(master_leader, master_follower, L_x, L_y, L_z, S_x, S_y, S_z):
    print(f"\n{'─'*65}")
    print(f"  REGRESO AL ORIGEN"); print(f"{'─'*65}")
    print(f"  Turno 1/2 — LÍDER regresando...")
    _fly_to_wp(master_leader, LEADER_SYSID, LEADER_COMPID,
               L_x, L_y, L_z, get_leader, label='LIDER regreso', timeout=RETURN_TIMEOUT)
    print(f"  Turno 2/2 — SEGUIDOR regresando...")
    _fly_to_wp(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
               S_x, S_y, S_z, get_follower, label='SEGUIDOR regreso', timeout=RETURN_TIMEOUT)
    print(f"  Regreso completado.")

# ══════════════════════════════════════════════════════════════════════════════
# HILO A — CÍRCULO DEL LÍDER
# ══════════════════════════════════════════════════════════════════════════════
def _thread_circle(master_leader, leader_sp_q):
    global _leader_phase
    dt       = 1.0 / LEADER_RATE
    duration = 2 * math.pi / ANGULAR_SPEED
    steps    = int(duration / dt)

    while True:
        if state_ready(get_leader()): break
        time.sleep(0.05)

    cx, cy   = 0.0, 0.0
    theta0   = math.pi

    print(f"\n[LIDER] Centro círculo: ({cx:.2f}, {cy:.2f}) ENU")
    print(f"[LIDER] R={RADIUS}m  ω={ANGULAR_SPEED}rad/s  T={duration:.1f}s")

    next_t = time.monotonic()
    _leader_phase = 'circle'
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
        # CAMBIO: usar send_pos_enu_abs en lugar de send_pos_yaw
        send_pos_enu_abs(master_leader, LEADER_SYSID, LEADER_COMPID, x_sp, y_sp, PREPOS_Z_LEADER, yaw)
        with _lock:
            leader_sp_q['x_sp'] = x_sp
            leader_sp_q['y_sp'] = y_sp
        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    x_final, y_final = x_sp, y_sp
    _leader_phase = 'tail'
    t_tail = time.monotonic()
    t_in_zone = None
    print("[LÍDER] Fase cola...")

    while not _stop_all.is_set():
        if time.monotonic() - t_tail > CONV_TIMEOUT:
            print("[LÍDER] ⚠️  Timeout cola."); break
        L = get_leader()
        if state_ready(L):
            dist  = math.hypot(L['x'] - x_final, L['y'] - y_final)
            speed = math.hypot(L['vx'], L['vy'])
            # CAMBIO: usar send_pos_enu_abs también en fase cola
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
# HILO B — PID DEL SEGUIDOR
# ══════════════════════════════════════════════════════════════════════════════
def _thread_pid(master_follower, start_time_ref, leader_sp_q):
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

    send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0,0,0,0)
    print("[SEGUIDOR] ⏹️  Control detenido.")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_csv():
    fname = f'lf_circulo_{int(time.time())}.csv'
    with _log_lock:
        rows = list(zip(*[_log[k] for k in _LOG_COLS]))
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_LOG_COLS)
        w.writerows(rows)
    print(f"📄 CSV: {fname}  ({len(rows)} muestras, {len(_LOG_COLS)} columnas)")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# GENERAR REPLAY RVIZ2
# ══════════════════════════════════════════════════════════════════════════════
def generate_rviz_replay(csv_fname):
    replay_fname = csv_fname.replace('.csv', '_rviz_replay.py')
    code = f'''#!/usr/bin/env python3
"""Replay RViz2 — generado desde {csv_fname}"""
import csv, time, math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time as RosTime

CSV_FILE = "{csv_fname}"
FRAME_ID = "map"
SPEED    = 1.0

def _stamp(t_unix):
    sec = int(t_unix); ns = int((t_unix - sec) * 1e9)
    ts = RosTime(); ts.sec = sec; ts.nanosec = ns; return ts

def _pose(x, y, z, yaw, stamp):
    ps = PoseStamped()
    ps.header.frame_id = FRAME_ID; ps.header.stamp = stamp
    ps.pose.position.x = float(x); ps.pose.position.y = float(y); ps.pose.position.z = float(z)
    cy, sy = math.cos(float(yaw)/2), math.sin(float(yaw)/2)
    ps.pose.orientation.w = cy; ps.pose.orientation.z = sy
    return ps

class ReplayNode(Node):
    def __init__(self, rows):
        super().__init__("lf_replay")
        self.rows = rows
        self.pub_L  = self.create_publisher(Path, "/leader/path",       10)
        self.pub_S  = self.create_publisher(Path, "/follower/path",     10)
        self.pub_Sd = self.create_publisher(Path, "/follower/setpoint", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/formation/marker", 10)
        self.path_L  = Path(); self.path_L.header.frame_id  = FRAME_ID
        self.path_S  = Path(); self.path_S.header.frame_id  = FRAME_ID
        self.path_Sd = Path(); self.path_Sd.header.frame_id = FRAME_ID
        self.idx = 0; self.t0_ros = None; self.t0_dat = None
        self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.idx >= len(self.rows): return
        row = self.rows[self.idx]; t_rel = float(row["t"])
        if self.t0_ros is None: self.t0_ros = time.monotonic(); self.t0_dat = t_rel
        if t_rel > (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat: return
        stamp = _stamp(float(row["timestamp_unix"]))
        lx,ly,lz = row["lx"],row["ly"],row["lz"]
        sx,sy,sz = row["sx"],row["sy"],row["sz"]
        xd,yd,zd = row["xd"],row["yd"],row["zd"]
        self.path_L.poses.append(_pose(lx,ly,lz,row["l_yaw"],stamp))
        self.path_S.poses.append(_pose(sx,sy,sz,row["s_yaw"],stamp))
        self.path_Sd.poses.append(_pose(xd,yd,zd,row["s_yaw"],stamp))
        for p in (self.path_L,self.path_S,self.path_Sd): p.header.stamp = stamp
        self.pub_L.publish(self.path_L); self.pub_S.publish(self.path_S); self.pub_Sd.publish(self.path_Sd)
        mk = Marker(); mk.header.frame_id = FRAME_ID; mk.header.stamp = stamp
        mk.ns="formation"; mk.id=0; mk.type=Marker.LINE_LIST; mk.action=Marker.ADD
        mk.scale.x=0.06; mk.color.r=1.0; mk.color.g=0.4; mk.color.b=0.0; mk.color.a=0.9
        p1=Point(); p1.x=float(lx); p1.y=float(ly); p1.z=float(lz)
        p2=Point(); p2.x=float(sx); p2.y=float(sy); p2.z=float(sz)
        mk.points=[p1,p2]; ma=MarkerArray(); ma.markers=[mk]; self.pub_mk.publish(ma)
        self.idx += 1

def main():
    with open(CSV_FILE, newline="") as f: rows = list(csv.DictReader(f))
    print(f"Cargadas {{len(rows)}} muestras. Duración: {{float(rows[-1][\'t\']):.1f}}s")
    rclpy.init(); node = ReplayNode(rows)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    rclpy.shutdown()

if __name__ == "__main__": main()
'''
    with open(replay_fname, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"🤖 Replay RViz2: {replay_fname}")
    return replay_fname

# ══════════════════════════════════════════════════════════════════════════════
# LIVE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def _init_live_plot_full():
    """Ventana completa: trayectoria XY + 6 subplots laterales."""
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
    ln_xy_th,  = ax_xy.plot([], [], 'k:', lw=1,   label='Teórico')
    ln_xy_L,   = ax_xy.plot([], [], 'g-', lw=2,   label='Líder')
    ln_xy_sp,  = ax_xy.plot([], [], 'r--',lw=1.2, label='Setpoint S', alpha=0.7)
    ln_xy_S,   = ax_xy.plot([], [], 'b-', lw=2,   label='Seguidor')
    ax_xy.legend(fontsize=8, loc='upper right')

    ax_x.set_title('X vs t'); ax_x.set_ylabel('X [m]'); ax_x.grid(True, alpha=0.4)
    ln_x_l, = ax_x.plot([], [], 'g-',  lw=1.5, label='Líder')
    ln_x_sp,= ax_x.plot([], [], 'r--', lw=1.2, label='Setpoint S')
    ln_x_s, = ax_x.plot([], [], 'b-',  lw=1.5, label='Seguidor')
    ax_x.legend(fontsize=7)

    ax_y.set_title('Y vs t'); ax_y.set_ylabel('Y [m]'); ax_y.grid(True, alpha=0.4)
    ln_y_l, = ax_y.plot([], [], 'g-',  lw=1.5)
    ln_y_sp,= ax_y.plot([], [], 'r--', lw=1.2)
    ln_y_s, = ax_y.plot([], [], 'b-',  lw=1.5)

    ax_z.set_title('Z vs t'); ax_z.set_ylabel('Z [m]'); ax_z.grid(True, alpha=0.4)
    ln_z_l, = ax_z.plot([], [], 'g-',  lw=1.5)
    ln_z_sp,= ax_z.plot([], [], 'r--', lw=1.2)
    ln_z_s, = ax_z.plot([], [], 'b-',  lw=1.5)

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


def _init_live_plot_traj():
    """Ventana solo con la trayectoria XY."""
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


def _update_live_plot(fig, axes, lines, traj_only=False):
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

    th = np.linspace(0, 2*math.pi, 300)
    lines['xy_th'].set_data(RADIUS * np.cos(th), RADIUS * np.sin(th))
    lines['xy_L'].set_data(lx, ly)
    lines['xy_sp'].set_data(xd, yd)
    lines['xy_S'].set_data(sx, sy)

    if not traj_only:
        for ln_l, ln_sp, ln_s, dl, dd, ds in [
            (lines['x_l'], lines['x_sp'], lines['x_s'], lx, xd, sx),
            (lines['y_l'], lines['y_sp'], lines['y_s'], ly, yd, sy),
            (lines['z_l'], lines['z_sp'], lines['z_s'], lz, zd, sz),
        ]:
            ln_l.set_data(t, dl); ln_sp.set_data(t, dd); ln_s.set_data(t, ds)
        lines['ex'].set_data(t, ex)
        lines['ey'].set_data(t, ey)
        lines['ez'].set_data(t, ez)
        lines['ffx'].set_data(t, ff_x); lines['px'].set_data(t, vx_p)
        lines['dx'].set_data(t, vx_d);  lines['cx'].set_data(t, vx_c)
        lines['ffy'].set_data(t, ff_y); lines['py'].set_data(t, vy_p)
        lines['dy'].set_data(t, vy_d);  lines['cy'].set_data(t, vy_c)

    for ax in axes.values():
        ax.relim(); ax.autoscale_view()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN POST-VUELO — ESTILO MATLAB (para tesis)
# ══════════════════════════════════════════════════════════════════════════════
# Paleta de colores estilo MATLAB
C_refL = np.array([0.40, 0.65, 1.00])   # azul claro — setpoint líder
C_refS = np.array([1.00, 0.60, 0.40])   # naranja claro — setpoint seguidor
C_L    = np.array([0.10, 0.35, 0.75])   # azul oscuro — líder real
C_S    = np.array([0.80, 0.15, 0.10])   # rojo — seguidor real
C_dist = np.array([0.25, 0.75, 0.45])   # verde — distancia
C_form = np.array([0.50, 0.50, 0.50])   # gris — vector formación
LW, LWS = 2.0, 1.2

SC_DRONE = 0.40      # longitud flechas de orientación del drone
SC_RTK   = SC_DRONE * 2.0


def _wrap_np(arr):
    return np.arctan2(np.sin(arr), np.cos(arr))


def _draw_drone_frame(ax, x, y, yaw, scale=SC_DRONE, lw=1.2):
    """Dibuja los ejes X (rojo) e Y (verde) del drone en (x,y) con ángulo yaw."""
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.005, headwidth=4, headlength=3.5, headaxislength=3, alpha=0.85)
    ax.quiver(x, y,  scale*np.cos(yaw),          scale*np.sin(yaw),
              color=[0.85, 0.10, 0.10], **kw)
    ax.quiver(x, y,  scale*np.cos(yaw+np.pi/2),  scale*np.sin(yaw+np.pi/2),
              color=[0.10, 0.65, 0.10], **kw)


def _draw_rtk_frame(ax, scale=SC_RTK):
    """Dibuja el frame RTK en el origen (0,0)."""
    kw = dict(angles='xy', scale_units='xy', scale=1,
              width=0.007, headwidth=4, headlength=4, headaxislength=3.5, alpha=0.9)
    ax.quiver(0, 0, scale, 0, color=[0.85, 0.10, 0.10], **kw)
    ax.quiver(0, 0, 0, scale, color=[0.10, 0.65, 0.10], **kw)
    ax.plot(0, 0, 'ok', ms=6, zorder=5)
    ax.text(scale + 0.12, -0.10, r'$\hat{x}_{ENU}$', fontsize=10, fontweight='bold', va='top')
    ax.text(-0.10, scale + 0.10, r'$\hat{y}_{ENU}$', fontsize=10, fontweight='bold', ha='right')


def plot_results():
    """7 figuras estilo MATLAB para tesis."""
    def arr(key): return np.array([float(v) for v in _log[key]])

    with _log_lock:
        t       = arr('t')
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

    if len(t) == 0:
        print("⚠️  Sin datos para graficar."); return

    n = len(t)
    psidesS = psiL.copy()   # el seguidor debe igualar el yaw del líder
    err_xy  = np.hypot(ex, ey)

    # ── Métricas en consola ───────────────────────────────────────────────────
    def rms(v): return np.sqrt(np.mean(v**2)) if len(v) > 0 else 0.0

    print('\n' + '='*50)
    print('  MÉTRICAS DE VUELO')
    print('='*50)
    print(f'  RMS error x  (seguidor): {rms(ex):.4f} m')
    print(f'  RMS error y  (seguidor): {rms(ey):.4f} m')
    print(f'  RMS error z  (seguidor): {rms(ez):.4f} m')
    print(f'  RMS error ψ  (seguidor): {rms(e_yaw):.4f} rad')
    print(f'  RMS error xy (seguidor): {rms(err_xy):.4f} m')
    print('  ─────────────────────────────────────')
    print(f'  Distancia deseada L-S : {OFFSET_D:.4f} m')
    print(f'  Distancia media   L-S : {np.mean(dist_xy):.4f} m')
    print(f'  Distancia máx     L-S : {np.max(dist_xy):.4f} m')
    print(f'  Distancia mín     L-S : {np.min(dist_xy):.4f} m')
    print(f'  Error formación RMS   : {rms(dist_xy - OFFSET_D):.4f} m')
    print(f'  Distancia Z media L-S : {np.mean(dist_z):.4f} m  (objetivo={OFFSET_DZ} m)')
    print('='*50 + '\n')

    plt.ioff()

# ── Fig 1 — Trayectoria XY completa con frames y vector de formación ──────
    fig1, ax = plt.subplots(figsize=(8, 8), facecolor='white')

    # Trayectorias
    ax.plot(lxsp, lysp, '--', color=C_refL, lw=LWS, label='Setpoint Líder',    alpha=0.8)
    ax.plot(xd,   yd,   '--', color=C_refS, lw=LWS, label='Setpoint Seguidor', alpha=0.8)
    ax.plot(lx,   ly,   '-',  color=C_L,    lw=LW,  label='Líder real')
    ax.plot(sx,   sy,   '-',  color=C_S,    lw=LW,  label='Seguidor real')

    # Círculo teórico
    th = np.linspace(0, 2*math.pi, 400)
    ax.plot(RADIUS*np.cos(th), RADIUS*np.sin(th),
            ':', color=[0.6, 0.6, 0.6], lw=1.0, label='Círculo teórico')

    # Puntos de inicio
    ax.plot(lx[0], ly[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Inicio Líder')
    ax.plot(sx[0], sy[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Inicio Seguidor')
    ax.plot(0, 0, '+k', ms=10, mew=2, label='Centro (0,0)')

    # Vectores de formación L→S (submuestra uniforme)
    n_d = 20
    idx_d = np.round(np.linspace(0, n-1, n_d+2)).astype(int)[1:-1]
    for i in idx_d:
        ax.quiver(lx[i], ly[i], sx[i]-lx[i], sy[i]-ly[i],
                  angles='xy', scale_units='xy', scale=1,
                  color=C_form, width=0.003, headwidth=3.5,
                  headlength=3.5, headaxislength=3, alpha=0.65)

    # Frames de orientación de los drones
    n_frames = 6
    idx_fr = np.round(np.linspace(0, n-1, n_frames)).astype(int)
    for i in idx_fr:
        _draw_drone_frame(ax, lx[i], ly[i], psiL[i])
        _draw_drone_frame(ax, sx[i], sy[i], psiS[i])

    # Frame RTK en el origen
    _draw_rtk_frame(ax)

    # Leyenda extra para los frames
    ax.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}_D$ (frente)')
    ax.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}_D$ (lateral)')
    ax.plot([], [], '-', color=C_form, lw=1.5,        label=r'Vector $\mathbf{d}$')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel(r'$x_{ENU}$ [m]', fontsize=11)
    ax.set_ylabel(r'$y_{ENU}$ [m]', fontsize=11)
    ax.set_title('Trayectoria XY — Líder y Seguidor con frames y formación', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    fig1.tight_layout()

    # ── Fig 2 — Trayectoria del Líder (setpoint vs real) ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax2.plot(lxsp, lysp, '--', color=C_refL, lw=LWS, label='Setpoint Líder')
    ax2.plot(lx,   ly,   '-',  color=C_L,    lw=LW,  label='Líder real')
    ax2.plot(RADIUS*np.cos(th), RADIUS*np.sin(th),
             ':', color=[0.6,0.6,0.6], lw=1.0, label='Teórico')
    ax2.plot(lx[0], ly[0], 'o', ms=9, mfc=C_L, mec='k', mew=1.2, label='Inicio')
    ax2.plot(0, 0, '+k', ms=10, mew=2)
    for i in idx_fr[:4]:
        _draw_drone_frame(ax2, lx[i], ly[i], psiL[i])
    _draw_rtk_frame(ax2)
    ax2.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}_D$')
    ax2.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}_D$')
    ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
    ax2.set_title('Líder: Setpoint vs Trayectoria Real', fontsize=12)
    ax2.legend(loc='best', fontsize=8); fig2.tight_layout()

    # ── Fig 3 — Trayectoria del Seguidor (setpoint vs real) ──────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 7), facecolor='white')
    ax3.plot(xd, yd, '--', color=C_refS, lw=LWS, label='Setpoint Seguidor')
    ax3.plot(sx, sy, '-',  color=C_S,    lw=LW,  label='Seguidor real')
    ax3.plot(sx[0], sy[0], 's', ms=8, mfc=C_S, mec='k', mew=1.2, label='Inicio')
    ax3.plot(0, 0, '+k', ms=10, mew=2)
    for i in idx_fr[:4]:
        _draw_drone_frame(ax3, sx[i], sy[i], psiS[i])
    _draw_rtk_frame(ax3)
    ax3.plot([], [], '-', color=[0.85,0.10,0.10], lw=2, label=r'$\hat{x}_D$')
    ax3.plot([], [], '-', color=[0.10,0.65,0.10], lw=2, label=r'$\hat{y}_D$')
    ax3.set_aspect('equal'); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x [m]'); ax3.set_ylabel('y [m]')
    ax3.set_title('Seguidor: Setpoint vs Trayectoria Real', fontsize=12)
    ax3.legend(loc='best', fontsize=8); fig3.tight_layout()

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

    axs4[2].plot(t, zd, '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[2].plot(t, lz, '-',  color=C_L,    lw=LW,  label='z Líder')
    axs4[2].plot(t, sz, '-',  color=C_S,    lw=LW,  label='z Seguidor')
    axs4[2].set_ylabel('z [m]'); axs4[2].grid(True, alpha=0.3)
    axs4[2].legend(loc='best', fontsize=8); axs4[2].set_title('Altitud')

    axs4[3].plot(t, np.degrees(psidesS), '--', color=C_refS, lw=LWS, label='Setpoint S')
    axs4[3].plot(t, np.degrees(psiL),    '-',  color=C_L,    lw=LW,  label='ψ Líder')
    axs4[3].plot(t, np.degrees(psiS),    '-',  color=C_S,    lw=LW,  label='ψ Seguidor')
    axs4[3].set_ylabel('ψ [°]'); axs4[3].set_xlabel('Tiempo [s]')
    axs4[3].grid(True, alpha=0.3)
    axs4[3].legend(loc='best', ncol=2, fontsize=8); axs4[3].set_title('Yaw')

    fig4.tight_layout()

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

    
    # ── Fig 5b — Errores del líder respecto a su setpoint ────────────────────
    # Setpoint yaw del líder: reconstruido desde la trayectoria del círculo
    # theta se puede aproximar como atan2(lysp, lxsp) y el yaw es theta + pi/2
    theta_sp   = np.arctan2(lysp, lxsp)          # ángulo del punto en el círculo
    psiL_sp    = _wrap_np(theta_sp + math.pi/2)  # yaw deseado del líder

    ex_L_calc   = lxsp - lx
    ey_L_calc   = lysp - ly
    ez_L_calc   = PREPOS_Z_LEADER - lz           # setpoint Z fijo del líder
    epsi_L_calc = _wrap_np(psiL_sp - psiL)       # error real de yaw del líder

    fig5b, axs5b = plt.subplots(4, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig5b.suptitle('Errores del Líder respecto a su setpoint', fontsize=13, fontweight='bold')

    for ax5b, data, lbl in zip(axs5b,
            [ex_L_calc, ey_L_calc, ez_L_calc, np.degrees(epsi_L_calc)],
            [r'$e_x$ [m]', r'$e_y$ [m]', r'$e_z$ [m]', r'$e_\psi$ [°]']):
        ax5b.plot(t, data, '-', color=C_L, lw=LW)
        ax5b.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax5b.set_ylabel(lbl); ax5b.grid(True, alpha=0.3)
    axs5b[-1].set_xlabel('Tiempo [s]')
    fig5b.tight_layout()

    
    # ── Fig 6 — Distancia L-S y yaw comparado ────────────────────────────────
    fig6, axs6 = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor='white')
    fig6.suptitle('Distancia de Formación L-S y Yaw', fontsize=13, fontweight='bold')

    axs6[0].plot(t, dist_xy, '-', color=C_dist, lw=LW, label='Distancia XY real')
    axs6[0].axhline(OFFSET_D, color='r', ls='--', lw=1.5,
                    label=f'Objetivo d = {OFFSET_D:.1f} m')
    axs6[0].set_ylabel('Dist XY [m]'); axs6[0].grid(True, alpha=0.3)
    axs6[0].legend(loc='best', fontsize=9); axs6[0].set_title('Distancia Horizontal L-S')

    axs6[1].plot(t, dist_z, '-', color=[0.55, 0.25, 0.65], lw=LW, label='Distancia Z real')
    axs6[1].axhline(OFFSET_DZ, color='r', ls='--', lw=1.5,
                    label=f'Objetivo Δz = {OFFSET_DZ:.1f} m')
    axs6[1].set_ylabel('Dist Z [m]'); axs6[1].grid(True, alpha=0.3)
    axs6[1].legend(loc='best', fontsize=9); axs6[1].set_title('Distancia Vertical L-S')

    axs6[2].plot(t, np.degrees(psiL), '-', color=C_L, lw=LW,  label='ψ Líder')
    axs6[2].plot(t, np.degrees(psiS), '-', color=C_S, lw=LW,  label='ψ Seguidor')
    axs6[2].set_ylabel('ψ [°]'); axs6[2].set_xlabel('Tiempo [s]')
    axs6[2].grid(True, alpha=0.3)
    axs6[2].legend(loc='best', fontsize=9); axs6[2].set_title('Comparación de Yaw')

    fig6.tight_layout()

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
        ax7.plot(t, cmd, 'k-',   lw=LW,  alpha=0.85,   label='Cmd total')
        ax7.axhline(0, color='gray', ls='--', lw=0.6)
        ax7.set_ylabel(lb); ax7.legend(fontsize=8); ax7.grid(True, alpha=0.3)
    axs7[-1].set_xlabel('Tiempo [s]')
    fig7.tight_layout()

    plt.show()
# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  LF_Circulo_RT — Líder-Seguidor en Tiempo Real")
    print("=" * 65)
    print(f"  Líder    : {LEADER_CONN}  (SYSID {LEADER_SYSID})")
    print(f"  Seguidor : {FOLLOWER_CONN} (SYSID {FOLLOWER_SYSID})")
    print(f"  Círculo  : R={RADIUS}m  ω={ANGULAR_SPEED}rad/s  "
          f"T={2*math.pi/ANGULAR_SPEED:.1f}s")
    print(f"  Offset   : d={OFFSET_D}m  α={math.degrees(OFFSET_ALPHA):.0f}°  "
          f"Δz={OFFSET_DZ}m")
    print(f"  Ganancias: Kp={KP} Ki={KI} Kd={KD} | "
          f"Kp_yaw={KP_YAW} Kd_yaw={KD_YAW}")
    print(f"  SKIP_PREPOS={SKIP_PREPOS}  SKIP_RETURN={SKIP_RETURN}  "
          f"RESET_ORIGIN={RESET_ORIGIN}")
    print(f"  LIVE_PLOT_ENABLED={LIVE_PLOT_ENABLED}  "
          f"LIVE_PLOT_TRAJ_ONLY={LIVE_PLOT_TRAJ_ONLY}")
    print("=" * 65)

    print(f"\n🔌 Conectando líder    ({LEADER_CONN})...")
    master_leader = mavutil.mavlink_connection(LEADER_CONN)
    master_leader.wait_heartbeat()
    print(f"   ✅ SYS={master_leader.target_system}")

    print(f"🔌 Conectando seguidor ({FOLLOWER_CONN})...")
    master_follower = mavutil.mavlink_connection(FOLLOWER_CONN)
    master_follower.wait_heartbeat()
    print(f"   ✅ SYS={master_follower.target_system}")

    stop_readers = threading.Event()
    thr_read_L = threading.Thread(
        target=_reader, args=(master_leader,   _leader_state,   stop_readers),
        daemon=True, name='reader-leader')
    thr_read_S = threading.Thread(
        target=_reader, args=(master_follower, _follower_state, stop_readers),
        daemon=True, name='reader-follower')
    thr_read_L.start()
    thr_read_S.start()

    # ── Paso 1: fijar origen ENU ──────────────────────────────────────────────
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
                print("\n❌ Timeout."); stop_readers.set(); raise SystemExit(1)
            print('.', end='', flush=True); time.sleep(0.2)

        with _lock:
            ref_lat = _leader_state['lat']
            ref_lon = _leader_state['lon']
            ref_alt = _leader_state['alt']
        set_enu_origin(ref_lat, ref_lon, ref_alt)
        save_enu_origin()

    print(f"[ENU] lat={_enu_origin['lat']:.8f}  "
          f"lon={_enu_origin['lon']:.8f}  alt={_enu_origin['alt']:.2f}m")

    # ── Paso 2: esperar telemetría ENU ────────────────────────────────────────
    print("\n⏳ Esperando telemetría ENU de ambos drones", end='', flush=True)
    t0w = time.monotonic()
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if time.monotonic() - t0w > 30.0:
            print("\n❌ Timeout."); stop_readers.set(); raise SystemExit(1)
        print('.', end='', flush=True); time.sleep(0.3)
    print(" ✅")

    # ── Paso 2b: capturar el home NED del líder ────────────────────────────────
    if not capture_leader_ned_home(master_leader):
        print("⚠️  No se pudo capturar el home NED del líder. Verifique telemetría.")
        stop_readers.set()
        raise SystemExit(1)

    L, S = get_leader(), get_follower()
    print(f"   Líder    ENU: x={L['x']:+.2f}m  y={L['y']:+.2f}m  z={L['z']:.2f}m"
          f"  ψ={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor ENU: x={S['x']:+.2f}m  y={S['y']:+.2f}m  z={S['z']:.2f}m"
          f"  ψ={math.degrees(S['yaw']):.1f}°")
    sep_xy = math.hypot(L['x'] - S['x'], L['y'] - S['y'])
    sep_z  = abs(L['z'] - S['z'])
    print(f"   Separación inicial L-S: {sep_xy:.2f}m horiz  {sep_z:.2f}m vert")

    # ── Fase 1: Preposicionamiento ─────────────────────────────────────────────
    theta0      = math.pi
    yaw_inicial = theta0 + math.pi / 2
    L_x = -RADIUS;  L_y = 0.0;  L_z = PREPOS_Z_LEADER
    dx_off, dy_off, _ = compute_offset(yaw_inicial)
    S_x = L_x + dx_off;  S_y = L_y + dy_off;  S_z = PREPOS_Z_FOLLOWER

    print("\n" + "-" * 65)
    print("  Ambos drones: ARMADOS y en modo GUIDED, hover estable")
    print("-" * 65)

    if SKIP_PREPOS:
        print(f"\n⏭️  Preposicionamiento OMITIDO (SKIP_PREPOS=True)")
        print(f"  Líder    esperado en ENU ({L_x:.2f}, {L_y:.2f})  z={L_z:.1f}m")
        print(f"  Seguidor esperado en ENU ({S_x:.2f}, {S_y:.2f})  z={S_z:.1f}m")
        input("  Confirma que los drones están en posición → ENTER\n")
    else:
        input("  ENTER para iniciar preposicionamiento...\n")
        preposition(master_leader, master_follower)

    print("\n" + "-" * 65)
    input("  ENTER para iniciar el círculo + seguimiento...\n")

    start_time  = time.monotonic()
    leader_sp_q = {}

    thr_pid = threading.Thread(
        target=_thread_pid, args=(master_follower, start_time, leader_sp_q),
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
            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0,0,0,0)
            time.sleep(0.5)

        _update_live_plot(fig_live, axes_live, lines_live,
                          traj_only=LIVE_PLOT_TRAJ_ONLY)
        plt.pause(0.5)
        plt.close(fig_live)

    else:
        # Sin live plot: solo esperamos que terminen los hilos
        try:
            while thr_circle.is_alive() or thr_pid.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 EMERGENCIA...")
            _stop_all.set(); _circle_done.set()
            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID, 0,0,0,0)
            time.sleep(0.5)

    # ── Fase 3: Regreso ───────────────────────────────────────────────────────
    if RETURN_ENABLED and not SKIP_RETURN and not _stop_all.is_set():
        return_to_start(master_leader, master_follower,
                        L_x, L_y, L_z, S_x, S_y, S_z)

    stop_readers.set()
    thr_read_L.join(timeout=2.0)
    thr_read_S.join(timeout=2.0)
    save_enu_origin()
    master_leader.close()
    master_follower.close()
    print("🔌 Conexiones cerradas.")

    csv_fname = save_csv()
    generate_rviz_replay(csv_fname)
    plot_results()
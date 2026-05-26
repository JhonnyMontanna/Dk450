#!/usr/bin/env python3
"""
LF_Cuadrado_RT.py — Lider-Seguidor en Tiempo Real: Trayectoria Cuadrada
========================================================================
El LIDER recorre un cuadrado de SIDE_LENGTH metros partiendo de su
posicion actual y a su altura actual (z real en el momento de iniciar).
El SEGUIDOR lo sigue con un PID en velocidad, manteniendose OFFSET_DZ
metros POR ENCIMA del lider en todo momento.

Filosofia de avance del lider: convergencia (igual que MavlinkCuadrado)
  El lider avanza al siguiente waypoint solo cuando llega al actual
  (o tras CONV_TIMEOUT segundos de espera).

Arquitectura de hilos:
  reader_lider    — LOCAL_POSITION_NED + ATTITUDE del lider
  reader_seguidor — LOCAL_POSITION_NED + ATTITUDE del seguidor
  thread_cuadrado — envia setpoints de posicion al lider (convergencia)
  thread_pid      — calcula PID y envia velocidad al seguidor (FOLLOWER_RATE Hz)
  hilo_principal  — live plot + espera fin de mision

Parametros configurables (seccion CONFIGURACION):
  SIDE_LENGTH  : lado del cuadrado [m]
  OFFSET_D     : distancia horizontal de formacion [m]
  OFFSET_ALPHA : angulo de formacion respecto al lider [rad]
  OFFSET_DZ    : metros que el seguidor va POR ENCIMA del lider [m]
  KP, KI, KD   : ganancias PID posicion
  KP_YAW ...   : ganancias PID yaw

CSV generado: lf_cuadrado_<timestamp>.csv  (mismo esquema que LF_Circulo_RT)
Replay RViz2: lf_cuadrado_<timestamp>_rviz_replay.py  (auto-generado)
"""

import math
import time
import csv
import threading
import matplotlib.pyplot as plt
import numpy as np
from pymavlink import mavutil

# ==============================================================================
# CONFIGURACION — edita aqui antes de volar
# ==============================================================================

# Conexiones MAVLink (un puerto UDP por drone)
LEADER_CONN   = 'udp:127.0.0.1:14552'   # Drone 1 — LIDER   (SITL -I0 / COM6)
FOLLOWER_CONN = 'udp:127.0.0.1:14553'   # Drone 2 — SEGUIDOR (SITL -I1 / COM4)

LEADER_SYSID    = 1
LEADER_COMPID   = 0
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# ── Trayectoria del Lider ─────────────────────────────────────────────────────
SIDE_LENGTH = 3.0    # [m] longitud del lado del cuadrado

# Modo de avance del lider entre waypoints
ADVANCE_MODE     = 'timer'   # 'convergence' | 'timer'
WAYPOINT_TIMER   = 10.0            # [s] solo si ADVANCE_MODE = 'timer'

# Criterio de convergencia del lider a cada waypoint
CONV_RADIUS  = 0.25   # [m]   distancia maxima para considerar "llego"
CONV_SPEED   = 0.15   # [m/s] velocidad maxima para considerar "parado"
CONV_HOLD    = 1.0    # [s]   segundos consecutivos dentro del criterio
CONV_TIMEOUT = 30.0   # [s]   tiempo maximo de espera por waypoint

LEADER_RATE  = 20     # [Hz]  frecuencia de envio de setpoints al lider

# ── PID del Seguidor ──────────────────────────────────────────────────────────
FOLLOWER_RATE = 20    # [Hz]

# Offset polar (posicion del seguidor respecto al lider):
#   OFFSET_D     : distancia horizontal de separacion [m]
#   OFFSET_ALPHA : angulo respecto al eje del lider [rad]
#                  math.pi      -> detras
#                  0            -> delante
#                  math.pi/2    -> izquierda
#                 -math.pi/2   -> derecha
#   OFFSET_DZ    : el seguidor va OFFSET_DZ metros MAS ALTO que el lider
#                  (positivo = seguidor mas alto, recomendado por seguridad)
OFFSET_D     = 2.0
OFFSET_ALPHA = 0 #math.pi      # detras del lider
OFFSET_DZ    = 1.0          # seguidor 1 m por encima del lider

# Ganancias PID posicion
KP   = 0.5
KI   = 0.0
KD   = 0.2

# Ganancias PID yaw
KP_YAW = 0.8
KI_YAW = 0.0
KD_YAW = 0.1

# Anti-windup
INTEGRAL_LIMIT     = 2.0   # m*s
INTEGRAL_YAW_LIMIT = 1.0   # rad*s

# Saturacion de salida
V_MAX        = 2.0    # [m/s] velocidad horizontal maxima
V_MAX_Z      = 1.0    # [m/s] velocidad vertical maxima
YAW_RATE_MAX = 1.0    # [rad/s]

# Refresco de la ventana de graficas en vivo
LIVE_PLOT_RATE = 4    # [Hz]

# ==============================================================================
# MASCARAS MAVLINK
# ==============================================================================
TYPE_MASK_POS_NOYAW = None   # se calcula al inicio (ver abajo)
TYPE_MASK_VEL_YAWRATE = None

def _build_masks():
    global TYPE_MASK_POS_NOYAW, TYPE_MASK_VEL_YAWRATE
    TYPE_MASK_POS_NOYAW = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
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

# ==============================================================================
# ESTADO COMPARTIDO (thread-safe)
# ==============================================================================
_lock = threading.Lock()

def _empty_state():
    return dict(x=None, y=None, z=None,
                vx=None, vy=None, vz=None,
                yaw=None, yaw_rate=None)

_leader_state   = _empty_state()
_follower_state = _empty_state()

# Log extendido
_log_lock = threading.Lock()
_LOG_COLS = [
    'timestamp_unix', 't', 'wp_idx', 'wp_phase',
    'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    'lx_sp', 'ly_sp', 'lz_sp',
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    'xd', 'yd', 'zd',
    'ex', 'ey', 'ez', 'err_xy', 'err_z',
    'e_yaw', 'dist_xy', 'dist_z',
    'ff_x', 'ff_y',
    'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i',
    'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd', 'yaw_rate_cmd',
]
_log = {k: [] for k in _LOG_COLS}

# Estado de progreso del lider (para el log del seguidor)
_wp_state = {'idx': 0, 'phase': 'transit',   # 'transit' | 'hold'
             'lx_sp': 0.0, 'ly_sp': 0.0, 'lz_sp': 0.0}

_mission_done = threading.Event()
_stop_all     = threading.Event()

# ==============================================================================
# LECTORES MAVLINK
# ==============================================================================
def _reader(master, state_dict, stop_event):
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1)
        if msg is None:
            continue
        mtype = msg.get_type()
        with _lock:
            if mtype == 'LOCAL_POSITION_NED':
                state_dict['x']  = msg.x
                state_dict['y']  = msg.y
                state_dict['z']  = -msg.z    # NED -> altura positiva hacia arriba (ENU)
                state_dict['vx'] = msg.vx
                state_dict['vy'] = msg.vy
                state_dict['vz'] = -msg.vz
            elif mtype == 'ATTITUDE':
                state_dict['yaw']      = msg.yaw
                state_dict['yaw_rate'] = msg.yawspeed

def get_leader():
    with _lock: return dict(_leader_state)

def get_follower():
    with _lock: return dict(_follower_state)

def state_ready(s):
    return all(v is not None for v in s.values())

# ==============================================================================
# ENVIO DE COMANDOS
# ==============================================================================
def send_pos(master, sysid, compid, x, y, z_ned):
    """Envia setpoint de posicion NED sin control de yaw."""
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_NOYAW,
        x, y, z_ned,
        0, 0, 0, 0, 0, 0, 0, 0)

def send_vel_yawrate(master, sysid, compid, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz_ned,
        0, 0, 0, 0, yaw_rate)

# ==============================================================================
# GENERACION DEL CUADRADO (relativo a posicion inicial del lider)
# ==============================================================================
def build_square(x0, y0, z0_enu, side):
    """
    Genera los 5 waypoints del cuadrado (4 esquinas + regreso al inicio).
    Coordenadas NED: z0_ned = -z0_enu
    La altura del lider es FIJA a z0_enu durante toda la mision.

    Cuadrado en el plano NED (x=norte, y=este):
      inicio -> norte -> noreste -> este -> inicio
    """
    z_ned = -z0_enu   # altura fija del lider en NED
    s = side
    wps_ned = [
        (x0,       y0,       z_ned),   # WP0 — posicion inicial (hover)
        (x0 + s,   y0,       z_ned),   # WP1 — norte
        (x0 + s,   y0 + s,   z_ned),   # WP2 — noreste
        (x0,       y0 + s,   z_ned),   # WP3 — este
        (x0,       y0,       z_ned),   # WP4 — regreso al inicio
    ]
    return wps_ned

# ==============================================================================
# MATEMATICAS PID
# ==============================================================================
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

def clamp(val, limit):
    return max(-limit, min(limit, val))

def compute_offset(psi_L):
    """
    Offset del seguidor respecto al lider en frame NED horizontal.
    OFFSET_DZ se suma a z del lider (seguidor MAS ALTO = +OFFSET_DZ en ENU).
    """
    angle = psi_L + OFFSET_ALPHA
    return (OFFSET_D * math.cos(angle),
            OFFSET_D * math.sin(angle),
            OFFSET_DZ)   # en ENU: positivo = mas alto

def compute_ff(yaw_rate_L, dx, dy):
    return yaw_rate_L * (-dy), yaw_rate_L * dx

class PIDState:
    def __init__(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0

def compute_control(L, S, pid, dt):
    dx, dy, dz = compute_offset(L['yaw'])

    # Setpoint del seguidor (ENU)
    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz   # lider_z_enu + OFFSET_DZ = seguidor va mas alto

    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']

    pid.integral[0] = clamp(pid.integral[0] + ex * dt, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * dt, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * dt, INTEGRAL_LIMIT)

    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    ff_x, ff_y = compute_ff(L['yaw_rate'], dx, dy)

    px, py, pz   = KP * ex,               KP * ey,               KP * ez
    ix, iy, iz   = KI * pid.integral[0],  KI * pid.integral[1],  KI * pid.integral[2]
    dx_, dy_, dz_ = KD * dv_x,            KD * dv_y,             KD * dv_z

    vx = ff_x + px + ix + dx_
    vy = ff_y + py + iy + dy_
    vz =        pz + iz + dz_

    v_h = math.hypot(vx, vy)
    if v_h > V_MAX:
        vx *= V_MAX / v_h
        vy *= V_MAX / v_h
    vz = clamp(vz, V_MAX_Z)
    vz_ned = -vz   # ENU -> NED para el comando MAVLink

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

# ==============================================================================
# HILO A — CUADRADO DEL LIDER (convergencia)
# ==============================================================================
def _thread_square(master_leader, wps_ned):
    """
    Recorre los waypoints del cuadrado usando convergencia.
    Actualiza _wp_state para que el log del seguidor pueda registrar
    el waypoint actual y el setpoint enviado al lider.
    """
    global _wp_state
    n   = len(wps_ned)
    dt  = 1.0 / LEADER_RATE

    print(f"\n[LIDER] {n} waypoints  |  modo: {ADVANCE_MODE}  |  lado={SIDE_LENGTH}m")
    for i, (wx, wy, wz) in enumerate(wps_ned):
        print(f"  WP{i}: x={wx:.2f}  y={wy:.2f}  z_ned={wz:.2f}  "
              f"(z_enu={-wz:.2f}m)")

    for idx, (tx, ty, tz_ned) in enumerate(wps_ned):
        if _stop_all.is_set():
            return

        with _lock:
            _wp_state['idx']   = idx
            _wp_state['phase'] = 'transit'
            _wp_state['lx_sp'] = tx
            _wp_state['ly_sp'] = ty
            _wp_state['lz_sp'] = -tz_ned   # guardamos en ENU

        print(f"\n[LIDER] WP{idx}/{n-1}: x={tx:.2f}  y={ty:.2f}  z_ned={tz_ned:.2f}",
              end='  ', flush=True)

        if ADVANCE_MODE == 'timer':
            # ── Modo temporizador ─────────────────────────────────────────────
            t_wp = time.monotonic()
            next_t = t_wp
            while time.monotonic() - t_wp < WAYPOINT_TIMER:
                if _stop_all.is_set(): return
                send_pos(master_leader, LEADER_SYSID, LEADER_COMPID, tx, ty, tz_ned)
                next_t += dt
                sl = next_t - time.monotonic()
                if sl > 0: time.sleep(sl)
            print(f"avanzado por tiempo ({WAYPOINT_TIMER}s)")

        else:
            # ── Modo convergencia ─────────────────────────────────────────────
            t_wp_start = time.monotonic()
            t_in_zone  = None
            next_t     = t_wp_start

            while not _stop_all.is_set():
                now = time.monotonic()
                if now - t_wp_start > CONV_TIMEOUT:
                    print(f"timeout ({CONV_TIMEOUT}s) — siguiente WP")
                    break

                send_pos(master_leader, LEADER_SYSID, LEADER_COMPID, tx, ty, tz_ned)

                L = get_leader()
                if state_ready(L):
                    # Comparar en NED: L['z'] es ENU, tz_ned es NED
                    dist  = math.sqrt((L['x'] - tx)**2 +
                                      (L['y'] - ty)**2 +
                                      (L['z'] - (-tz_ned))**2)
                    speed = math.hypot(L['vx'], L['vy'])

                    if dist < CONV_RADIUS and speed < CONV_SPEED:
                        if t_in_zone is None:
                            t_in_zone = now
                        elif now - t_in_zone >= CONV_HOLD:
                            elapsed = now - t_wp_start
                            print(f"llegado en {elapsed:.1f}s  "
                                  f"(dist={dist:.3f}m  vel={speed:.3f}m/s)")
                            with _lock:
                                _wp_state['phase'] = 'hold'
                            break
                    else:
                        t_in_zone = None

                next_t += dt
                sl = next_t - time.monotonic()
                if sl > 0: time.sleep(sl)

    print("\n[LIDER] Cuadrado completado.")
    _mission_done.set()

# ==============================================================================
# HILO B — PID DEL SEGUIDOR
# ==============================================================================
def _thread_pid(master_follower, start_time_ref):
    dt  = 1.0 / FOLLOWER_RATE
    pid = PIDState()

    print("[SEGUIDOR] Esperando telemetria...", end='', flush=True)
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if _stop_all.is_set(): return
        time.sleep(0.05)
    print(" OK")

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
                lxsp  = _wp_state['lx_sp']
                lysp  = _wp_state['ly_sp']
                lzsp  = _wp_state['lz_sp']
                widx  = _wp_state['idx']
                wphase= _wp_state['phase']

            with _log_lock:
                row = {
                    'timestamp_unix': f'{ts_unix:.6f}',
                    't':              f'{t_el:.4f}',
                    'wp_idx':         str(widx),
                    'wp_phase':       wphase,
                    'lx':  f'{L["x"]:.4f}',  'ly':  f'{L["y"]:.4f}',  'lz':  f'{L["z"]:.4f}',
                    'lvx': f'{L["vx"]:.4f}', 'lvy': f'{L["vy"]:.4f}', 'lvz': f'{L["vz"]:.4f}',
                    'l_yaw':     f'{L["yaw"]:.4f}',
                    'l_yawrate': f'{L["yaw_rate"]:.4f}',
                    'lx_sp': f'{lxsp:.4f}', 'ly_sp': f'{lysp:.4f}', 'lz_sp': f'{lzsp:.4f}',
                    'sx':  f'{S["x"]:.4f}',  'sy':  f'{S["y"]:.4f}',  'sz':  f'{S["z"]:.4f}',
                    'svx': f'{S["vx"]:.4f}', 'svy': f'{S["vy"]:.4f}', 'svz': f'{S["vz"]:.4f}',
                    's_yaw':     f'{S["yaw"]:.4f}',
                    's_yawrate': f'{S["yaw_rate"]:.4f}',
                    'xd': f'{c["xd"]:.4f}', 'yd': f'{c["yd"]:.4f}', 'zd': f'{c["zd"]:.4f}',
                    'ex': f'{c["ex"]:.4f}', 'ey': f'{c["ey"]:.4f}', 'ez': f'{c["ez"]:.4f}',
                    'err_xy':  f'{err_xy:.4f}',
                    'err_z':   f'{abs(c["ez"]):.4f}',
                    'e_yaw':   f'{c["e_yaw"]:.4f}',
                    'dist_xy': f'{dist_xy:.4f}',
                    'dist_z':  f'{dist_z:.4f}',
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

            # Consola cada 5 s
            if int(t_el) % 5 == 0 and int(t_el - dt) % 5 != 0:
                print(f"  t={t_el:6.1f}s | WP{widx}({wphase[:4]}) "
                      f"err_xy={err_xy:.3f}m  dist={dist_xy:.2f}m  "
                      f"dist_z={dist_z:.2f}m  "
                      f"e_yaw={math.degrees(c['e_yaw']):.1f}deg  "
                      f"cmd=({c['vx']:.2f},{c['vy']:.2f},{-c['vz_ned']:.2f})")

        if _mission_done.is_set():
            break

        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                     0.0, 0.0, 0.0, 0.0)
    print("\n[SEGUIDOR] Control detenido.")

# ==============================================================================
# GUARDAR CSV
# ==============================================================================
def save_csv():
    fname = f'lf_cuadrado_{int(time.time())}.csv'
    with _log_lock:
        rows = list(zip(*[_log[k] for k in _LOG_COLS]))
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(_LOG_COLS)
        w.writerows(rows)
    print(f"\nCSV: {fname}  ({len(rows)} muestras, {len(_LOG_COLS)} columnas)")
    return fname

# ==============================================================================
# GENERAR REPLAY RVIZ2
# ==============================================================================
def generate_rviz_replay(csv_fname):
    replay_fname = csv_fname.replace('.csv', '_rviz_replay.py')
    code = f'''#!/usr/bin/env python3
"""
Replay RViz2 — generado desde {csv_fname}
Publica:
  /leader/path          nav_msgs/Path
  /follower/path        nav_msgs/Path
  /follower/setpoint    nav_msgs/Path
  /formation/marker     visualization_msgs/MarkerArray (linea L->S)

Uso:
  ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
  python3 {replay_fname}
  rviz2   (añadir Path x3 + MarkerArray, Fixed Frame: map)
"""
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
    ts = RosTime(); ts.sec = sec; ts.nanosec = ns
    return ts

def _pose(x, y, z, yaw, stamp):
    ps = PoseStamped()
    ps.header.frame_id = FRAME_ID; ps.header.stamp = stamp
    ps.pose.position.x = float(x)
    ps.pose.position.y = float(y)
    ps.pose.position.z = float(z)
    cy, sy = math.cos(float(yaw)/2), math.sin(float(yaw)/2)
    ps.pose.orientation.w = cy; ps.pose.orientation.z = sy
    return ps

class ReplayNode(Node):
    def __init__(self, rows):
        super().__init__("lf_cuadrado_replay")
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
        row = self.rows[self.idx]
        t_rel = float(row["t"])
        if self.t0_ros is None:
            self.t0_ros = time.monotonic(); self.t0_dat = t_rel
        if t_rel > (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat: return
        stamp = _stamp(float(row["timestamp_unix"]))
        lx,ly,lz = row["lx"],row["ly"],row["lz"]
        sx,sy,sz = row["sx"],row["sy"],row["sz"]
        xd,yd,zd = row["xd"],row["yd"],row["zd"]
        self.path_L.poses.append(_pose(lx,ly,lz,row["l_yaw"],stamp))
        self.path_S.poses.append(_pose(sx,sy,sz,row["s_yaw"],stamp))
        self.path_Sd.poses.append(_pose(xd,yd,zd,row["s_yaw"],stamp))
        for p in (self.path_L,self.path_S,self.path_Sd): p.header.stamp = stamp
        self.pub_L.publish(self.path_L)
        self.pub_S.publish(self.path_S)
        self.pub_Sd.publish(self.path_Sd)
        mk = Marker()
        mk.header.frame_id = FRAME_ID; mk.header.stamp = stamp
        mk.ns="formation"; mk.id=0; mk.type=Marker.LINE_LIST; mk.action=Marker.ADD
        mk.scale.x=0.06; mk.color.r=1.0; mk.color.g=0.4; mk.color.a=0.9
        p1=Point(); p1.x=float(lx); p1.y=float(ly); p1.z=float(lz)
        p2=Point(); p2.x=float(sx); p2.y=float(sy); p2.z=float(sz)
        mk.points=[p1,p2]
        ma=MarkerArray(); ma.markers=[mk]; self.pub_mk.publish(ma)
        self.idx += 1

def main():
    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{{len(rows)}} muestras | duracion: {{float(rows[-1][\'t\']):.1f}}s | speed: {{SPEED}}x")
    rclpy.init()
    node = ReplayNode(rows)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
'''
    with open(replay_fname, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"Replay RViz2: {replay_fname}")
    return replay_fname

# ==============================================================================
# GRAFICAS EN TIEMPO REAL
# ==============================================================================
def _init_live_plot(wps_ned, x0, y0):
    """Crea ventana live plot con el cuadrado teorico ya dibujado."""
    plt.ion()
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f'LF Cuadrado RT — lado={SIDE_LENGTH}m  '
                 f'offset={OFFSET_D}m  dz={OFFSET_DZ}m', fontsize=11)

    gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.35)
    ax_xy  = fig.add_subplot(gs[:, 0])
    ax_x   = fig.add_subplot(gs[0, 1])
    ax_y   = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_z   = fig.add_subplot(gs[2, 1], sharex=ax_x)
    ax_err = fig.add_subplot(gs[3, 1], sharex=ax_x)
    ax_px  = fig.add_subplot(gs[4, 1], sharex=ax_x)
    ax_py  = fig.add_subplot(gs[5, 1], sharex=ax_x)

    # Cuadrado teorico (NED x=norte, y=este -> grafica X horizontal, Y vertical)
    sq_x = [wp[0] for wp in wps_ned] + [wps_ned[0][0]]
    sq_y = [wp[1] for wp in wps_ned] + [wps_ned[0][1]]
    ax_xy.plot(sq_x, sq_y, 'k:', lw=1.5, label='Cuadrado teorico')
    for i, (wx, wy, _) in enumerate(wps_ned):
        ax_xy.scatter(wx, wy, c='k', s=40, zorder=4)
        ax_xy.annotate(f'WP{i}', (wx, wy), fontsize=7,
                       xytext=(4, 4), textcoords='offset points')

    ax_xy.set_title('Trayectoria XY (NED)')
    ax_xy.set_xlabel('X — norte [m]'); ax_xy.set_ylabel('Y — este [m]')
    ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.4)
    ln_xy_l,  = ax_xy.plot([], [], 'g-', lw=2,   label='Lider')
    ln_xy_sp, = ax_xy.plot([], [], 'r--',lw=1.2, label='Setpoint S', alpha=0.7)
    ln_xy_s,  = ax_xy.plot([], [], 'b-', lw=2,   label='Seguidor')
    ax_xy.legend(fontsize=8, loc='upper right')

    ax_x.set_title('Posicion X (norte)'); ax_x.set_ylabel('X [m]'); ax_x.grid(True, alpha=0.4)
    ln_x_l,  = ax_x.plot([], [], 'g-',  lw=1.5, label='Lider')
    ln_x_sp, = ax_x.plot([], [], 'r--', lw=1.2, label='Setpoint S')
    ln_x_s,  = ax_x.plot([], [], 'b-',  lw=1.5, label='Seguidor')
    ax_x.legend(fontsize=7, loc='upper right')

    ax_y.set_title('Posicion Y (este)'); ax_y.set_ylabel('Y [m]'); ax_y.grid(True, alpha=0.4)
    ln_y_l,  = ax_y.plot([], [], 'g-',  lw=1.5)
    ln_y_sp, = ax_y.plot([], [], 'r--', lw=1.2)
    ln_y_s,  = ax_y.plot([], [], 'b-',  lw=1.5)

    ax_z.set_title('Altura Z (ENU)'); ax_z.set_ylabel('Z [m]'); ax_z.grid(True, alpha=0.4)
    ln_z_l,  = ax_z.plot([], [], 'g-',  lw=1.5, label='Lider')
    ln_z_sp, = ax_z.plot([], [], 'r--', lw=1.2, label='Setpoint S')
    ln_z_s,  = ax_z.plot([], [], 'b-',  lw=1.5, label='Seguidor')
    ax_z.legend(fontsize=7, loc='upper right')

    ax_err.set_title('Errores Seguidor'); ax_err.set_ylabel('Error [m]')
    ax_err.axhline(0, color='k', ls='--', lw=0.7); ax_err.grid(True, alpha=0.4)
    ln_ex, = ax_err.plot([], [], lw=1.2, label='ex', color='tab:red')
    ln_ey, = ax_err.plot([], [], lw=1.2, label='ey', color='tab:blue')
    ln_ez, = ax_err.plot([], [], lw=1.2, label='ez', color='tab:green')
    ax_err.legend(fontsize=7, loc='upper right')

    ax_px.set_title('PID Eje X'); ax_px.set_ylabel('m/s')
    ax_px.axhline(0, color='gray', ls='--', lw=0.6); ax_px.grid(True, alpha=0.4)
    ln_ff_x,  = ax_px.plot([], [], lw=1.2, label='FF',  color='tab:orange')
    ln_p_x,   = ax_px.plot([], [], lw=1.2, label='P',   color='tab:blue')
    ln_d_x,   = ax_px.plot([], [], lw=1.2, label='D',   color='tab:green')
    ln_cmd_x, = ax_px.plot([], [], 'k-', lw=1.8, label='Cmd', alpha=0.8)
    ax_px.legend(fontsize=7, loc='upper right')

    ax_py.set_title('PID Eje Y'); ax_py.set_ylabel('m/s'); ax_py.set_xlabel('Tiempo [s]')
    ax_py.axhline(0, color='gray', ls='--', lw=0.6); ax_py.grid(True, alpha=0.4)
    ln_ff_y,  = ax_py.plot([], [], lw=1.2, color='tab:orange')
    ln_p_y,   = ax_py.plot([], [], lw=1.2, color='tab:blue')
    ln_d_y,   = ax_py.plot([], [], lw=1.2, color='tab:green')
    ln_cmd_y, = ax_py.plot([], [], 'k-', lw=1.8, alpha=0.8)

    lines = dict(
        xy_l=ln_xy_l, xy_sp=ln_xy_sp, xy_s=ln_xy_s,
        x_l=ln_x_l,  x_sp=ln_x_sp,  x_s=ln_x_s,
        y_l=ln_y_l,  y_sp=ln_y_sp,  y_s=ln_y_s,
        z_l=ln_z_l,  z_sp=ln_z_sp,  z_s=ln_z_s,
        ex=ln_ex, ey=ln_ey, ez=ln_ez,
        ff_x=ln_ff_x, p_x=ln_p_x, d_x=ln_d_x, cmd_x=ln_cmd_x,
        ff_y=ln_ff_y, p_y=ln_p_y, d_y=ln_d_y, cmd_y=ln_cmd_y,
    )
    axes = dict(xy=ax_xy, x=ax_x, y=ax_y, z=ax_z,
                err=ax_err, px=ax_px, py=ax_py)
    return fig, axes, lines


def _update_live_plot(fig, axes, lines):
    with _log_lock:
        if len(_log['t']) < 2: return
        def a(k): return np.array([float(v) for v in _log[k]])
        t    = a('t')
        lx   = a('lx');  ly  = a('ly');  lz  = a('lz')
        sx   = a('sx');  sy  = a('sy');  sz  = a('sz')
        xd   = a('xd');  yd  = a('yd');  zd  = a('zd')
        ex   = a('ex');  ey  = a('ey');  ez  = a('ez')
        ff_x = a('ff_x'); ff_y = a('ff_y')
        vx_p = a('vx_p'); vy_p = a('vy_p')
        vx_d = a('vx_d'); vy_d = a('vy_d')
        vx_c = a('vx_cmd'); vy_c = a('vy_cmd')

    lines['xy_l'].set_data(lx, ly)
    lines['xy_sp'].set_data(xd, yd)
    lines['xy_s'].set_data(sx, sy)

    for ln_l, ln_sp, ln_s, dl, dd, ds in [
        (lines['x_l'], lines['x_sp'], lines['x_s'], lx, xd, sx),
        (lines['y_l'], lines['y_sp'], lines['y_s'], ly, yd, sy),
        (lines['z_l'], lines['z_sp'], lines['z_s'], lz, zd, sz),
    ]:
        ln_l.set_data(t, dl); ln_sp.set_data(t, dd); ln_s.set_data(t, ds)

    lines['ex'].set_data(t, ex)
    lines['ey'].set_data(t, ey)
    lines['ez'].set_data(t, ez)

    lines['ff_x'].set_data(t, ff_x); lines['p_x'].set_data(t, vx_p)
    lines['d_x'].set_data(t, vx_d);  lines['cmd_x'].set_data(t, vx_c)
    lines['ff_y'].set_data(t, ff_y); lines['p_y'].set_data(t, vy_p)
    lines['d_y'].set_data(t, vy_d);  lines['cmd_y'].set_data(t, vy_c)

    for ax in axes.values():
        ax.relim(); ax.autoscale_view()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

# ==============================================================================
# GRAFICAS POST-VUELO
# ==============================================================================
def plot_results(wps_ned):
    def arr(k): return np.array([float(v) for v in _log[k]])

    with _log_lock:
        t       = arr('t');   lx = arr('lx'); ly = arr('ly'); lz = arr('lz')
        sx      = arr('sx');  sy = arr('sy'); sz = arr('sz')
        xd      = arr('xd');  yd = arr('yd'); zd = arr('zd')
        ex      = arr('ex');  ey = arr('ey'); ez = arr('ez')
        err_xy  = arr('err_xy')
        e_yaw   = arr('e_yaw')
        dist_xy = arr('dist_xy'); dist_z = arr('dist_z')
        ff_x    = arr('ff_x');  ff_y  = arr('ff_y')
        vx_p    = arr('vx_p');  vy_p  = arr('vy_p')
        vx_d    = arr('vx_d');  vy_d  = arr('vy_d')
        vx_cmd  = arr('vx_cmd'); vy_cmd = arr('vy_cmd')
        psiL    = arr('l_yaw'); psiS = arr('s_yaw')
        wp_idx  = np.array([int(v) for v in _log['wp_idx']])

    if len(t) == 0:
        print("Sin datos para graficar."); return

    rms_xy = np.sqrt(np.mean(err_xy**2))
    print(f"\nRMS error_xy = {rms_xy:.4f} m")
    print(f"dist_xy  media={np.mean(dist_xy):.3f}m  std={np.std(dist_xy):.3f}m"
          f"  (objetivo={OFFSET_D}m)")
    print(f"dist_z   media={np.mean(dist_z):.3f}m  (objetivo={OFFSET_DZ}m)")

    colors = plt.cm.tab10(np.linspace(0, 1, len(wps_ned)))
    plt.ioff()

    # Fig 1 — Trayectorias XY
    fig1, ax = plt.subplots(figsize=(7, 7))
    sq_x = [wp[0] for wp in wps_ned] + [wps_ned[0][0]]
    sq_y = [wp[1] for wp in wps_ned] + [wps_ned[0][1]]
    ax.plot(sq_x, sq_y, 'k:', lw=1.5, label='Cuadrado teorico')
    for i in range(len(wps_ned)):
        m = wp_idx == i
        if m.any():
            ax.plot(lx[m], ly[m], '-', color=colors[i], lw=1.5, alpha=0.6)
    ax.plot(lx, ly, 'g-', lw=2,   label='Lider')
    ax.plot(xd, yd, 'r--',lw=1.2, alpha=0.7, label='Setpoint S')
    ax.plot(sx, sy, 'b-', lw=2,   label='Seguidor')
    ax.scatter([lx[0]], [ly[0]], c='g', s=80, zorder=5)
    ax.scatter([sx[0]], [sy[0]], c='b', s=80, zorder=5)
    for i, (wx, wy, _) in enumerate(wps_ned):
        ax.scatter(wx, wy, c='k', s=40, zorder=4)
        ax.annotate(f'WP{i}', (wx, wy), fontsize=8,
                    xytext=(4, 4), textcoords='offset points')
    ax.set(xlabel='X — norte [m]', ylabel='Y — este [m]', title='Trayectorias XY')
    ax.axis('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    fig1.tight_layout()

    # Fig 2 — Posicion vs tiempo (X, Y, Z)
    fig2, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig2.suptitle('Posicion vs Tiempo')
    for ax, dl, ds, dd, lb in zip(axs,
            [lx,ly,lz],[sx,sy,sz],[xd,yd,zd],
            ['X norte [m]','Y este [m]','Z altura [m]']):
        ax.plot(t, dl, 'g-',  lw=1.5, label='Lider')
        ax.plot(t, dd, 'r--', lw=1.2, label='Setpoint S')
        ax.plot(t, ds, 'b-',  lw=1.5, label='Seguidor')
        # Lineas verticales por cambio de waypoint
        for i in range(1, len(wps_ned)):
            m = wp_idx == i
            if m.any():
                ax.axvline(t[m][0], color='gray', ls=':', lw=0.8)
        ax.set_ylabel(lb); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    axs[-1].set_xlabel('Tiempo [s]')
    fig2.tight_layout()

    # Fig 3 — Errores del seguidor
    fig3, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig3.suptitle('Errores del Seguidor')
    for ax, d, lb in zip(axs,
            [ex, ey, ez, np.degrees(e_yaw)],
            ['ex [m]','ey [m]','ez [m]','eYaw [deg]']):
        ax.plot(t, d, lw=1.2)
        ax.axhline(0, color='k', ls='--', lw=0.7)
        for i in range(1, len(wps_ned)):
            m = wp_idx == i
            if m.any(): ax.axvline(t[m][0], color='gray', ls=':', lw=0.8)
        ax.set_ylabel(lb); ax.grid(True, alpha=0.4)
    axs[-1].set_xlabel('Tiempo [s]')
    fig3.tight_layout()

    # Fig 4 — Distancias L-S y yaw
    fig4, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig4.suptitle('Distancias L-S y Yaw')
    a1.plot(t, dist_xy, 'b-', lw=1.5, label='Dist XY real')
    a1.axhline(OFFSET_D, color='r', ls='--', lw=1.2, label=f'Objetivo {OFFSET_D}m')
    a1.set_ylabel('Dist XY [m]'); a1.legend(); a1.grid(True, alpha=0.4)
    a2.plot(t, dist_z, 'm-', lw=1.5, label='Dist Z real')
    a2.axhline(OFFSET_DZ, color='r', ls='--', lw=1.2, label=f'Objetivo {OFFSET_DZ}m')
    a2.set_ylabel('Dist Z [m]'); a2.legend(); a2.grid(True, alpha=0.4)
    a3.plot(t, np.degrees(psiL), 'g-', lw=1.2, label='Lider')
    a3.plot(t, np.degrees(psiS), 'b-', lw=1.2, label='Seguidor')
    a3.set_ylabel('Yaw [deg]'); a3.set_xlabel('Tiempo [s]')
    a3.legend(); a3.grid(True, alpha=0.4)
    fig4.tight_layout()

    # Fig 5 — Desglose PID
    fig5, (b1, b2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig5.suptitle('Desglose PID del Seguidor')
    for ax, ff, pp, dd, cmd, lb in zip(
            [b1, b2],
            [ff_x, ff_y], [vx_p, vy_p], [vx_d, vy_d], [vx_cmd, vy_cmd],
            ['Eje X [m/s]', 'Eje Y [m/s]']):
        ax.plot(t, ff,  lw=1.2, label='Feed-forward', color='tab:orange')
        ax.plot(t, pp,  lw=1.2, label='Proporcional', color='tab:blue')
        ax.plot(t, dd,  lw=1.2, label='Derivativo',   color='tab:green')
        ax.plot(t, cmd, 'k-', lw=1.8, alpha=0.8, label='Cmd total')
        ax.axhline(0, color='gray', ls='--', lw=0.6)
        ax.set_ylabel(lb); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    b2.set_xlabel('Tiempo [s]')
    fig5.tight_layout()

    plt.show()

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    _build_masks()

    print("=" * 65)
    print("  LF_Cuadrado_RT — Lider-Seguidor Cuadrado en Tiempo Real")
    print("=" * 65)
    print(f"  Lider    : {LEADER_CONN}  (SYSID {LEADER_SYSID})")
    print(f"  Seguidor : {FOLLOWER_CONN} (SYSID {FOLLOWER_SYSID})")
    print(f"  Cuadrado : lado={SIDE_LENGTH}m  modo={ADVANCE_MODE}")
    print(f"  Offset   : d={OFFSET_D}m  alpha={math.degrees(OFFSET_ALPHA):.0f}deg"
          f"  Dz=+{OFFSET_DZ}m (seguidor mas alto)")
    print(f"  PID      : Kp={KP} Ki={KI} Kd={KD} | Kp_yaw={KP_YAW}")
    print(f"  Log      : {len(_LOG_COLS)} columnas @ {FOLLOWER_RATE}Hz")
    print("=" * 65)

    # ── Conectar ──────────────────────────────────────────────────────────────
    print(f"\nConectando lider    ({LEADER_CONN})...")
    master_leader = mavutil.mavlink_connection(LEADER_CONN)
    master_leader.wait_heartbeat()
    print(f"  OK SYS={master_leader.target_system}")

    print(f"Conectando seguidor ({FOLLOWER_CONN})...")
    master_follower = mavutil.mavlink_connection(FOLLOWER_CONN)
    master_follower.wait_heartbeat()
    print(f"  OK SYS={master_follower.target_system}")

    # ── Lectores ──────────────────────────────────────────────────────────────
    stop_readers = threading.Event()
    thr_read_L = threading.Thread(
        target=_reader, args=(master_leader,   _leader_state,   stop_readers),
        daemon=True, name='reader-leader')
    thr_read_S = threading.Thread(
        target=_reader, args=(master_follower, _follower_state, stop_readers),
        daemon=True, name='reader-follower')
    thr_read_L.start()
    thr_read_S.start()

    # ── Esperar telemetria ────────────────────────────────────────────────────
    print("\nEsperando telemetria", end='', flush=True)
    t0w = time.monotonic()
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if time.monotonic() - t0w > 30.0:
            print("\nTimeout."); stop_readers.set(); raise SystemExit(1)
        print('.', end='', flush=True); time.sleep(0.3)
    print(" OK")

    L, S = get_leader(), get_follower()
    print(f"  Lider    x={L['x']:.2f}  y={L['y']:.2f}  z={L['z']:.2f}  "
          f"yaw={math.degrees(L['yaw']):.1f}deg")
    print(f"  Seguidor x={S['x']:.2f}  y={S['y']:.2f}  z={S['z']:.2f}  "
          f"yaw={math.degrees(S['yaw']):.1f}deg")

    # ── Construir cuadrado a partir de la posicion ACTUAL del lider ───────────
    x0, y0, z0_enu = L['x'], L['y'], L['z']
    wps_ned = build_square(x0, y0, z0_enu, SIDE_LENGTH)

    print(f"\nCuadrado generado desde posicion actual del lider:")
    print(f"  Altura fija del lider : z_enu = {z0_enu:.2f} m")
    print(f"  Altura del seguidor   : z_enu = {z0_enu + OFFSET_DZ:.2f} m  "
          f"(lider + {OFFSET_DZ}m)")
    for i, (wx, wy, wz) in enumerate(wps_ned):
        print(f"  WP{i}: x={wx:.2f}  y={wy:.2f}  z_ned={wz:.2f}")

    # ── Confirmar inicio ──────────────────────────────────────────────────────
    print("\n" + "-" * 65)
    print("  Ambos drones: ARMADOS y en modo GUIDED")
    print("  Seguidor en hover estable antes de continuar")
    print("-" * 65)
    input("  ENTER para iniciar el cuadrado + seguimiento...\n")

    start_time = time.monotonic()

    # ── Lanzar hilos de control ───────────────────────────────────────────────
    thr_pid = threading.Thread(
        target=_thread_pid, args=(master_follower, start_time),
        daemon=True, name='pid-follower')
    thr_square = threading.Thread(
        target=_thread_square, args=(master_leader, wps_ned),
        daemon=True, name='square-leader')

    thr_pid.start()
    thr_square.start()
    print("En marcha! Ctrl+C = parada de emergencia.\n")

    # ── Live plot en el hilo principal ────────────────────────────────────────
    fig_live, axes_live, lines_live = _init_live_plot(wps_ned, x0, y0)
    live_dt   = 1.0 / LIVE_PLOT_RATE
    next_plot = time.monotonic()

    try:
        while thr_square.is_alive() or thr_pid.is_alive():
            now = time.monotonic()
            if now >= next_plot:
                _update_live_plot(fig_live, axes_live, lines_live)
                next_plot = now + live_dt
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nEMERGENCIA — deteniendo seguidor...")
        _stop_all.set(); _mission_done.set()
        send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                         0.0, 0.0, 0.0, 0.0)
        time.sleep(0.5)

    # Ultima actualizacion del live plot
    _update_live_plot(fig_live, axes_live, lines_live)
    plt.pause(0.5)
    plt.close(fig_live)

    # ── Limpieza ──────────────────────────────────────────────────────────────
    stop_readers.set()
    thr_read_L.join(timeout=2.0)
    thr_read_S.join(timeout=2.0)
    master_leader.close()
    master_follower.close()
    print("Conexiones cerradas.")

    # ── CSV + replay + graficas post-vuelo ────────────────────────────────────
    csv_fname = save_csv()
    generate_rviz_replay(csv_fname)
    plot_results(wps_ned)
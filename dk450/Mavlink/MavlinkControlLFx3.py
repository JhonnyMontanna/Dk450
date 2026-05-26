#!/usr/bin/env python3
"""
LF_Circulo_RT.py — Líder-Seguidor en Tiempo Real con Círculo
=============================================================
Log extendido para análisis completo en Python y replay en RViz/rviz2.

CSV generado: lf_circulo_<timestamp>.csv
  Columnas de tiempo:
    timestamp_unix  — tiempo UNIX absoluto [s] (float, para RViz)
    t               — tiempo relativo al inicio [s]
    leader_phase    — 'circle' | 'tail'

  Líder (posición + velocidad + yaw):
    lx, ly, lz      — posición real ENU [m]
    lvx, lvy, lvz   — velocidad real [m/s]
    l_yaw           — yaw real [rad]
    l_yawrate       — yaw rate real [rad/s]
    lx_sp, ly_sp    — setpoint enviado al líder [m]

  Seguidor (posición + velocidad + yaw):
    sx, sy, sz
    svx, svy, svz
    s_yaw, s_yawrate

  Setpoint del seguidor:
    xd, yd, zd      — posición deseada para el seguidor [m]

  Errores:
    ex, ey, ez      — error de posición por eje [m]
    err_xy          — norma horizontal del error [m]
    err_z           — error vertical [m]
    e_yaw           — error de yaw [rad]
    dist_xy         — distancia horizontal L-S real [m]
    dist_z          — distancia vertical L-S real [m]

  Desglose PID del seguidor:
    ff_x, ff_y      — prealimentación por eje [m/s]
    vx_p, vy_p, vz_p — término proporcional [m/s]
    vx_i, vy_i, vz_i — término integral [m/s]
    vx_d, vy_d, vz_d — término derivativo [m/s]

  Comandos finales (saturados):
    vx_cmd, vy_cmd, vz_cmd  — velocidad comandada [m/s]
    yaw_rate_cmd             — yaw rate comandado [rad/s]

Replay RViz:
  Al terminar se genera lf_circulo_<ts>_rviz_replay.py automáticamente.
  Publica /leader/path, /follower/path, /follower/setpoint como nav_msgs/Path.
"""

import math
import time
import csv
import threading
import matplotlib.pyplot as plt
import numpy as np
from pymavlink import mavutil

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════
LEADER_CONN    = 'udp:127.0.0.1:14552'
FOLLOWER_CONN  = 'udp:127.0.0.1:14553'

LEADER_SYSID    = 1
LEADER_COMPID   = 0
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# Trayectoria líder
RADIUS        = 2.0
ANGULAR_SPEED = 0.2    # rad/s → T = 2π/ω ≈ 31.4 s
LEADER_RATE   = 50     # Hz

# Convergencia
CONV_RADIUS  = 0.15
CONV_SPEED   = 0.10
CONV_HOLD    = 1.0
CONV_TIMEOUT = 15.0

# PID seguidor
FOLLOWER_RATE = 20     # Hz

OFFSET_D     = 1.0
OFFSET_ALPHA = -math.pi / 2   # derecha del líder
OFFSET_DZ    = 1.0

KP, KI, KD          = 0.5, 0.0, 0.0
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
                yaw=None, yaw_rate=None)

_leader_state   = _empty_state()
_follower_state = _empty_state()

# ── Log extendido ─────────────────────────────────────────────────────────────
_log_lock = threading.Lock()
_LOG_COLS = [
    # Tiempo
    'timestamp_unix', 't', 'leader_phase',
    # Líder real
    'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    # Setpoint del líder
    'lx_sp', 'ly_sp',
    # Seguidor real
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    # Setpoint del seguidor
    'xd', 'yd', 'zd',
    # Errores por eje
    'ex', 'ey', 'ez',
    # Normas de error
    'err_xy', 'err_z',
    # Error yaw y distancias reales L-S
    'e_yaw', 'dist_xy', 'dist_z',
    # Desglose PID
    'ff_x', 'ff_y',
    'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i',
    'vx_d', 'vy_d', 'vz_d',
    # Comandos finales
    'vx_cmd', 'vy_cmd', 'vz_cmd', 'yaw_rate_cmd',
]
_log = {k: [] for k in _LOG_COLS}

_circle_done  = threading.Event()
_stop_all     = threading.Event()
_leader_phase = 'circle'   # modificado por _thread_circle

# ══════════════════════════════════════════════════════════════════════════════
# LECTORES MAVLINK
# ══════════════════════════════════════════════════════════════════════════════
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
                state_dict['z']  = -msg.z
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

# ══════════════════════════════════════════════════════════════════════════════
# ENVÍO DE COMANDOS
# ══════════════════════════════════════════════════════════════════════════════
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
# MATEMÁTICAS PID
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

    # Términos aislados (para log)
    px, py, pz   = KP*ex,               KP*ey,               KP*ez
    ix, iy, iz   = KI*pid.integral[0],  KI*pid.integral[1],  KI*pid.integral[2]
    dx_, dy_, dz_ = KD*dv_x,            KD*dv_y,             KD*dv_z

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
# HILO A — CÍRCULO DEL LÍDER
# ══════════════════════════════════════════════════════════════════════════════
def _thread_circle(master_leader, leader_sp_q):
    global _leader_phase

    dt       = 1.0 / LEADER_RATE
    duration = 2 * math.pi / ANGULAR_SPEED
    steps    = int(duration / dt)

    while True:
        L = get_leader()
        if state_ready(L): break
        time.sleep(0.05)

    x0, y0, z0_up = L['x'], L['y'], L['z']
    z0_ned = -z0_up
    cx, cy = x0 + RADIUS, y0
    theta0 = math.pi

    print(f"\n[LÍDER] Inicio: x={x0:.2f}  y={y0:.2f}  z={z0_up:.2f}")
    print(f"[LÍDER] Círculo R={RADIUS}m  ω={ANGULAR_SPEED}rad/s  T={duration:.1f}s")

    next_t = time.monotonic()
    _leader_phase = 'circle'
    x_sp, y_sp, yaw = x0, y0, 0.0   # inicializar para la fase cola

    for i in range(steps):
        if _stop_all.is_set(): return
        t_s   = i * dt
        theta = theta0 + ANGULAR_SPEED * t_s
        x_sp  = cx + RADIUS * math.cos(theta)
        y_sp  = cy + RADIUS * math.sin(theta)
        yaw   = theta + math.pi / 2

        send_pos_yaw(master_leader, LEADER_SYSID, LEADER_COMPID,
                     x_sp, y_sp, z0_ned, yaw)

        with _lock:
            leader_sp_q['x_sp'] = x_sp
            leader_sp_q['y_sp'] = y_sp

        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    # Fase de cola
    x_final, y_final = x_sp, y_sp
    _leader_phase = 'tail'
    t_tail = time.monotonic()
    t_in_zone = None
    print("[LÍDER] Fase cola — esperando convergencia...")

    while not _stop_all.is_set():
        if time.monotonic() - t_tail > CONV_TIMEOUT:
            print("[LÍDER] ⚠️  Timeout cola.")
            break
        L = get_leader()
        if state_ready(L):
            dist  = math.hypot(L['x'] - x_final, L['y'] - y_final)
            speed = math.hypot(L['vx'], L['vy'])
            send_pos_yaw(master_leader, LEADER_SYSID, LEADER_COMPID,
                         x_final, y_final, z0_ned, yaw)
            with _lock:
                leader_sp_q['x_sp'] = x_final
                leader_sp_q['y_sp'] = y_final
            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None: t_in_zone = time.monotonic()
                elif time.monotonic() - t_in_zone >= CONV_HOLD:
                    print(f"[LÍDER] ✅ Convergido dist={dist:.3f}m")
                    break
            else:
                t_in_zone = None
        time.sleep(dt)

    _circle_done.set()
    print("[LÍDER] 🏁 Trayectoria finalizada.")

# ══════════════════════════════════════════════════════════════════════════════
# HILO B — PID DEL SEGUIDOR (log extendido)
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
                    'lx':  f'{L["x"]:.4f}',  'ly':  f'{L["y"]:.4f}',  'lz':  f'{L["z"]:.4f}',
                    'lvx': f'{L["vx"]:.4f}',  'lvy': f'{L["vy"]:.4f}', 'lvz': f'{L["vz"]:.4f}',
                    'l_yaw':     f'{L["yaw"]:.4f}',
                    'l_yawrate': f'{L["yaw_rate"]:.4f}',
                    'lx_sp': f'{lxsp:.4f}', 'ly_sp': f'{lysp:.4f}',
                    'sx':  f'{S["x"]:.4f}',  'sy':  f'{S["y"]:.4f}',  'sz':  f'{S["z"]:.4f}',
                    'svx': f'{S["vx"]:.4f}', 'svy': f'{S["vy"]:.4f}', 'svz': f'{S["vz"]:.4f}',
                    's_yaw':     f'{S["yaw"]:.4f}',
                    's_yawrate': f'{S["yaw_rate"]:.4f}',
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

            # Consola cada 5 s
            if int(t_el) % 5 == 0 and int(t_el - dt) % 5 != 0:
                print(f"  t={t_el:6.1f}s | err_xy={err_xy:.3f}m  dist={dist_xy:.2f}m"
                      f"  e_yaw={math.degrees(c['e_yaw']):.1f}°"
                      f"  ff=({c['ff_x']:.2f},{c['ff_y']:.2f})"
                      f"  cmd=({c['vx']:.2f},{c['vy']:.2f})")

        if _circle_done.is_set():
            break

        next_t += dt
        sl = next_t - time.monotonic()
        if sl > 0: time.sleep(sl)

    send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                     0.0, 0.0, 0.0, 0.0)
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
"""
Replay RViz2 — generado desde {csv_fname}
==========================================
Publica:
  /leader/path          nav_msgs/Path
  /follower/path        nav_msgs/Path
  /follower/setpoint    nav_msgs/Path
  /formation/marker     visualization_msgs/MarkerArray (linea L->S)

Uso:
  # Terminal 1: frame estático
  ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom

  # Terminal 2: replay
  python3 {replay_fname}

  # Terminal 3: RViz2
  rviz2
  Añadir:  Path (/leader/path)
           Path (/follower/path)
           Path (/follower/setpoint)
           MarkerArray (/formation/marker)
  Fixed Frame: map
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
SPEED    = 1.0   # 1.0 = tiempo real, 2.0 = doble velocidad

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
    cy, sy = math.cos(float(yaw)/2), math.sin(float(yaw)/2)
    ps.pose.orientation.w = cy
    ps.pose.orientation.z = sy
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
        self.idx    = 0
        self.t0_ros = None
        self.t0_dat = None
        self.create_timer(0.05, self._tick)

    def _tick(self):
        if self.idx >= len(self.rows):
            return
        row   = self.rows[self.idx]
        t_rel = float(row["t"])
        if self.t0_ros is None:
            self.t0_ros = time.monotonic()
            self.t0_dat = t_rel
        if t_rel > (time.monotonic() - self.t0_ros) * SPEED + self.t0_dat:
            return

        stamp = _stamp(float(row["timestamp_unix"]))
        lx, ly, lz = row["lx"], row["ly"], row["lz"]
        sx, sy, sz = row["sx"], row["sy"], row["sz"]
        xd, yd, zd = row["xd"], row["yd"], row["zd"]

        self.path_L.poses.append(_pose(lx, ly, lz, row["l_yaw"], stamp))
        self.path_S.poses.append(_pose(sx, sy, sz, row["s_yaw"], stamp))
        self.path_Sd.poses.append(_pose(xd, yd, zd, row["s_yaw"], stamp))

        for p in (self.path_L, self.path_S, self.path_Sd):
            p.header.stamp = stamp

        self.pub_L.publish(self.path_L)
        self.pub_S.publish(self.path_S)
        self.pub_Sd.publish(self.path_Sd)

        # Línea de formación
        mk = Marker()
        mk.header.frame_id = FRAME_ID; mk.header.stamp = stamp
        mk.ns = "formation"; mk.id = 0
        mk.type = Marker.LINE_LIST; mk.action = Marker.ADD
        mk.scale.x = 0.06
        mk.color.r = 1.0; mk.color.g = 0.4; mk.color.b = 0.0; mk.color.a = 0.9
        p1 = Point(); p1.x=float(lx); p1.y=float(ly); p1.z=float(lz)
        p2 = Point(); p2.x=float(sx); p2.y=float(sy); p2.z=float(sz)
        mk.points = [p1, p2]
        ma = MarkerArray(); ma.markers = [mk]
        self.pub_mk.publish(ma)

        self.idx += 1

def main():
    with open(CSV_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Cargadas {{len(rows)}} muestras de {{CSV_FILE}}")
    print(f"Duración: {{float(rows[-1][\'t\']):.1f}} s  |  Velocidad replay: {{SPEED}}x")
    rclpy.init()
    node = ReplayNode(rows)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
'''
    with open(replay_fname, 'w', encoding='utf-8') as f:
        f.write(code)
    print(f"🤖 Replay RViz2: {replay_fname}")
    return replay_fname

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS EN TIEMPO REAL
# Ventana con 7 subplots que se actualiza en vivo mientras vuela.
# Corre en el hilo principal (matplotlib necesita el hilo principal en Windows).
# ══════════════════════════════════════════════════════════════════════════════

# Frecuencia de refresco de la ventana en vivo [Hz]
LIVE_PLOT_RATE = 4   # 4 Hz es suficiente para ser legible sin saturar la CPU

def _init_live_plot():
    """
    Crea la figura con 7 subplots:
      Fila 0 (span): Trayectoria XY líder + seguidor
      Fila 1: X vs t  |  Fila 2: Y vs t  |  Fila 3: Z vs t
      Fila 4: ex, ey, ez vs t
      Fila 5: FF, P, D, Cmd — eje X
      Fila 6: FF, P, D, Cmd — eje Y
    Devuelve (fig, dict_of_line_objects).
    """
    plt.ion()
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle('Control Lider-Seguidor — Tiempo Real', fontsize=12)

    # Layout: columna izquierda = XY (ocupa toda la altura), derecha = 6 filas
    gs = fig.add_gridspec(6, 2, hspace=0.55, wspace=0.35)

    ax_xy  = fig.add_subplot(gs[:, 0])          # columna izquierda completa
    ax_x   = fig.add_subplot(gs[0, 1])
    ax_y   = fig.add_subplot(gs[1, 1], sharex=ax_x)
    ax_z   = fig.add_subplot(gs[2, 1], sharex=ax_x)
    ax_err = fig.add_subplot(gs[3, 1], sharex=ax_x)
    ax_px  = fig.add_subplot(gs[4, 1], sharex=ax_x)
    ax_py  = fig.add_subplot(gs[5, 1], sharex=ax_x)

    # ── Trayectoria XY ────────────────────────────────────────────────────────
    ax_xy.set_title('Trayectoria XY')
    ax_xy.set_xlabel('X [m]'); ax_xy.set_ylabel('Y [m]')
    ax_xy.set_aspect('equal'); ax_xy.grid(True, alpha=0.4)
    ln_xy_theory,  = ax_xy.plot([], [], 'k:', lw=1,   label='Teorico')
    ln_xy_lider,   = ax_xy.plot([], [], 'g-', lw=2,   label='Lider')
    ln_xy_sp,      = ax_xy.plot([], [], 'r--',lw=1.2, label='Setpoint S', alpha=0.7)
    ln_xy_seguidor,= ax_xy.plot([], [], 'b-', lw=2,   label='Seguidor')
    ax_xy.legend(fontsize=8, loc='upper right')

    # ── Posición X ───────────────────────────────────────────────────────────
    ax_x.set_title('Posicion X'); ax_x.set_ylabel('X [m]'); ax_x.grid(True, alpha=0.4)
    ln_x_l,  = ax_x.plot([], [], 'g-',  lw=1.5, label='Lider')
    ln_x_sp, = ax_x.plot([], [], 'r--', lw=1.2, label='Setpoint S')
    ln_x_s,  = ax_x.plot([], [], 'b-',  lw=1.5, label='Seguidor')
    ax_x.legend(fontsize=7, loc='upper right')

    # ── Posición Y ───────────────────────────────────────────────────────────
    ax_y.set_title('Posicion Y'); ax_y.set_ylabel('Y [m]'); ax_y.grid(True, alpha=0.4)
    ln_y_l,  = ax_y.plot([], [], 'g-',  lw=1.5)
    ln_y_sp, = ax_y.plot([], [], 'r--', lw=1.2)
    ln_y_s,  = ax_y.plot([], [], 'b-',  lw=1.5)

    # ── Posición Z ───────────────────────────────────────────────────────────
    ax_z.set_title('Posicion Z'); ax_z.set_ylabel('Z [m]'); ax_z.grid(True, alpha=0.4)
    ln_z_l,  = ax_z.plot([], [], 'g-',  lw=1.5)
    ln_z_sp, = ax_z.plot([], [], 'r--', lw=1.2)
    ln_z_s,  = ax_z.plot([], [], 'b-',  lw=1.5)

    # ── Errores ex, ey, ez ───────────────────────────────────────────────────
    ax_err.set_title('Errores Seguidor'); ax_err.set_ylabel('Error [m]')
    ax_err.axhline(0, color='k', ls='--', lw=0.7)
    ax_err.grid(True, alpha=0.4)
    ln_ex, = ax_err.plot([], [], lw=1.2, label='ex', color='tab:red')
    ln_ey, = ax_err.plot([], [], lw=1.2, label='ey', color='tab:blue')
    ln_ez, = ax_err.plot([], [], lw=1.2, label='ez', color='tab:green')
    ax_err.legend(fontsize=7, loc='upper right')

    # ── PID eje X ────────────────────────────────────────────────────────────
    ax_px.set_title('PID Eje X'); ax_px.set_ylabel('m/s')
    ax_px.axhline(0, color='gray', ls='--', lw=0.6); ax_px.grid(True, alpha=0.4)
    ln_ff_x,  = ax_px.plot([], [], lw=1.2, label='FF',   color='tab:orange')
    ln_p_x,   = ax_px.plot([], [], lw=1.2, label='P',    color='tab:blue')
    ln_d_x,   = ax_px.plot([], [], lw=1.2, label='D',    color='tab:green')
    ln_cmd_x, = ax_px.plot([], [], 'k-', lw=1.8, label='Cmd', alpha=0.8)
    ax_px.legend(fontsize=7, loc='upper right')

    # ── PID eje Y ────────────────────────────────────────────────────────────
    ax_py.set_title('PID Eje Y'); ax_py.set_ylabel('m/s'); ax_py.set_xlabel('Tiempo [s]')
    ax_py.axhline(0, color='gray', ls='--', lw=0.6); ax_py.grid(True, alpha=0.4)
    ln_ff_y,  = ax_py.plot([], [], lw=1.2, color='tab:orange')
    ln_p_y,   = ax_py.plot([], [], lw=1.2, color='tab:blue')
    ln_d_y,   = ax_py.plot([], [], lw=1.2, color='tab:green')
    ln_cmd_y, = ax_py.plot([], [], 'k-', lw=1.8, alpha=0.8)

    lines = dict(
        xy_theory=ln_xy_theory, xy_lider=ln_xy_lider,
        xy_sp=ln_xy_sp, xy_seg=ln_xy_seguidor,
        x_l=ln_x_l, x_sp=ln_x_sp, x_s=ln_x_s,
        y_l=ln_y_l, y_sp=ln_y_sp, y_s=ln_y_s,
        z_l=ln_z_l, z_sp=ln_z_sp, z_s=ln_z_s,
        ex=ln_ex, ey=ln_ey, ez=ln_ez,
        ff_x=ln_ff_x, p_x=ln_p_x, d_x=ln_d_x, cmd_x=ln_cmd_x,
        ff_y=ln_ff_y, p_y=ln_p_y, d_y=ln_d_y, cmd_y=ln_cmd_y,
    )
    axes = dict(xy=ax_xy, x=ax_x, y=ax_y, z=ax_z,
                err=ax_err, px=ax_px, py=ax_py)

    return fig, axes, lines


def _update_live_plot(fig, axes, lines, x0, y0):
    """Lee el log actual y actualiza todas las líneas. Sin bloqueo."""
    with _log_lock:
        if len(_log['t']) < 2:
            return
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

    # Círculo teórico (solo se dibuja una vez pero no cuesta nada rehacerlo)
    th = np.linspace(0, 2*math.pi, 300)
    lines['xy_theory'].set_data(x0+RADIUS + RADIUS*np.cos(th),
                                y0         + RADIUS*np.sin(th))

    # Trayectorias XY (completas, sin distinción cola/círculo)
    lines['xy_lider'].set_data(lx, ly)
    lines['xy_sp'].set_data(xd, yd)
    lines['xy_seg'].set_data(sx, sy)

    # Posición vs tiempo
    for ln_l, ln_sp, ln_s, dl, dd, ds in [
        (lines['x_l'], lines['x_sp'], lines['x_s'], lx, xd, sx),
        (lines['y_l'], lines['y_sp'], lines['y_s'], ly, yd, sy),
        (lines['z_l'], lines['z_sp'], lines['z_s'], lz, zd, sz),
    ]:
        ln_l.set_data(t, dl); ln_sp.set_data(t, dd); ln_s.set_data(t, ds)

    # Errores
    lines['ex'].set_data(t, ex)
    lines['ey'].set_data(t, ey)
    lines['ez'].set_data(t, ez)

    # PID X
    lines['ff_x'].set_data(t, ff_x)
    lines['p_x'].set_data(t, vx_p)
    lines['d_x'].set_data(t, vx_d)
    lines['cmd_x'].set_data(t, vx_c)

    # PID Y
    lines['ff_y'].set_data(t, ff_y)
    lines['p_y'].set_data(t, vy_p)
    lines['d_y'].set_data(t, vy_d)
    lines['cmd_y'].set_data(t, vy_c)

    # Reescalar todos los ejes
    for ax in axes.values():
        ax.relim(); ax.autoscale_view()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS POST-VUELO (estáticas, bloqueantes)
# ══════════════════════════════════════════════════════════════════════════════
def plot_results():
    def arr(key): return np.array([float(v) for v in _log[key]])

    with _log_lock:
        t       = arr('t');   lx = arr('lx'); ly = arr('ly'); lz = arr('lz')
        sx      = arr('sx');  sy = arr('sy'); sz = arr('sz')
        xd      = arr('xd');  yd = arr('yd'); zd = arr('zd')
        ex      = arr('ex');  ey = arr('ey'); ez = arr('ez')
        err_xy  = arr('err_xy')
        e_yaw   = arr('e_yaw')
        dist_xy = arr('dist_xy'); dist_z = arr('dist_z')
        ff_x    = arr('ff_x');  ff_y = arr('ff_y')
        vx_p    = arr('vx_p');  vy_p = arr('vy_p')
        vx_d    = arr('vx_d');  vy_d = arr('vy_d')
        vx_cmd  = arr('vx_cmd'); vy_cmd = arr('vy_cmd')
        psiL    = arr('l_yaw'); psiS = arr('s_yaw')

    if len(t) == 0:
        print("⚠️  Sin datos."); return

    rms_xy = np.sqrt(np.mean(err_xy**2))
    print(f"\n📊 RMS error_xy = {rms_xy:.4f} m")
    print(f"   dist_xy  media={np.mean(dist_xy):.3f} m  std={np.std(dist_xy):.3f} m"
          f"  (objetivo={OFFSET_D} m)")
    print(f"   dist_z   media={np.mean(dist_z):.3f} m  (objetivo={OFFSET_DZ} m)")

    plt.ioff()

    # Fig 1 — Trayectorias XY (completa, sin separar cola)
    fig1, ax = plt.subplots(figsize=(7, 7))
    th = np.linspace(0, 2*math.pi, 500)
    ax.plot(lx[0]+RADIUS + RADIUS*np.cos(th), ly[0] + RADIUS*np.sin(th),
            'k:', lw=1, label='Teorico')
    ax.plot(lx, ly, 'g-',  lw=2,   label='Lider')
    ax.plot(xd, yd, 'r--', lw=1.2, alpha=0.7, label='Setpoint S')
    ax.plot(sx, sy, 'b-',  lw=2,   label='Seguidor')
    ax.scatter([lx[0]], [ly[0]], c='g', s=80, zorder=5)
    ax.scatter([sx[0]], [sy[0]], c='b', s=80, zorder=5)
    ax.set(xlabel='X [m]', ylabel='Y [m]', title='Trayectorias XY')
    ax.axis('equal'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    fig1.tight_layout()

    # Fig 2 — Posición vs tiempo
    fig2, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig2.suptitle('Posicion vs Tiempo')
    for ax, dl, ds, dd, lb in zip(axs,
            [lx,ly,lz],[sx,sy,sz],[xd,yd,zd],
            ['X [m]','Y [m]','Z [m]']):
        ax.plot(t, dl, 'g-',  lw=1.5, label='Lider')
        ax.plot(t, dd, 'r--', lw=1.2, label='Setpoint S')
        ax.plot(t, ds, 'b-',  lw=1.5, label='Seguidor')
        ax.set_ylabel(lb); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    axs[-1].set_xlabel('Tiempo [s]')
    fig2.tight_layout()

    # Fig 3 — Errores
    fig3, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig3.suptitle('Errores del Seguidor')
    for ax, d, lb in zip(axs,
            [ex, ey, ez, np.degrees(e_yaw)],
            ['ex [m]','ey [m]','ez [m]','eY [deg]']):
        ax.plot(t, d, lw=1.2)
        ax.axhline(0, color='k', ls='--', lw=0.7)
        ax.set_ylabel(lb); ax.grid(True, alpha=0.4)
    axs[-1].set_xlabel('Tiempo [s]')
    fig3.tight_layout()

    # Fig 4 — Distancias L-S y yaw
    fig4, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig4.suptitle('Distancias L-S y Yaw')
    a1.plot(t, dist_xy, 'b-', lw=1.5, label='Dist XY real')
    a1.axhline(OFFSET_D,  color='r', ls='--', lw=1.2, label=f'Objetivo {OFFSET_D}m')
    a1.set_ylabel('Dist XY [m]'); a1.legend(); a1.grid(True, alpha=0.4)
    a2.plot(t, dist_z, 'm-', lw=1.5, label='Dist Z real')
    a2.axhline(OFFSET_DZ, color='r', ls='--', lw=1.2, label=f'Objetivo {OFFSET_DZ}m')
    a2.set_ylabel('Dist Z [m]'); a2.legend(); a2.grid(True, alpha=0.4)
    a3.plot(t, np.degrees(psiL), 'g-', lw=1.2, label='Lider')
    a3.plot(t, np.degrees(psiS), 'b-', lw=1.2, label='Seguidor')
    a3.set_ylabel('Yaw [deg]'); a3.set_xlabel('Tiempo [s]')
    a3.legend(); a3.grid(True, alpha=0.4)
    fig4.tight_layout()

    # Fig 5 — Desglose PID por eje
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  LF_Circulo_RT — Log extendido + Live Plot + Replay RViz2")
    print("=" * 65)
    print(f"  Lider    : {LEADER_CONN}  (SYSID {LEADER_SYSID})")
    print(f"  Seguidor : {FOLLOWER_CONN} (SYSID {FOLLOWER_SYSID})")
    print(f"  Circulo  : R={RADIUS}m  w={ANGULAR_SPEED}rad/s  "
          f"T={2*math.pi/ANGULAR_SPEED:.1f}s")
    print(f"  Offset   : d={OFFSET_D}m  alpha={math.degrees(OFFSET_ALPHA):.0f}deg  "
          f"Dz={OFFSET_DZ}m")
    print(f"  Ganancias: Kp={KP} Ki={KI} Kd={KD} | "
          f"Kp_yaw={KP_YAW} Kd_yaw={KD_YAW}")
    print(f"  Log      : {len(_LOG_COLS)} columnas @ {FOLLOWER_RATE} Hz")
    print("=" * 65)

    print(f"\n🔌 Conectando lider    ({LEADER_CONN})...")
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

    print("\n⏳ Esperando telemetria", end='', flush=True)
    t0w = time.monotonic()
    while True:
        if state_ready(get_leader()) and state_ready(get_follower()): break
        if time.monotonic() - t0w > 30.0:
            print("\n❌ Timeout."); stop_readers.set(); raise SystemExit(1)
        print('.', end='', flush=True); time.sleep(0.3)
    print(" ✅")

    L, S = get_leader(), get_follower()
    print(f"   Lider    x={L['x']:.2f} y={L['y']:.2f} z={L['z']:.2f} "
          f"yaw={math.degrees(L['yaw']):.1f}deg")
    print(f"   Seguidor x={S['x']:.2f} y={S['y']:.2f} z={S['z']:.2f} "
          f"yaw={math.degrees(S['yaw']):.1f}deg")

    # Guardar posición inicial del líder para el círculo teórico en el live plot
    x0_leader, y0_leader = L['x'], L['y']

    print("\n" + "-" * 65)
    print("  Ambos drones: ARMADOS y en modo GUIDED")
    print("  Seguidor en hover estable antes de continuar")
    print("-" * 65)
    input("  ENTER para iniciar...\n")

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

    # ── Live plot en el hilo principal ────────────────────────────────────────
    fig_live, axes_live, lines_live = _init_live_plot()
    live_dt = 1.0 / LIVE_PLOT_RATE
    next_plot_t = time.monotonic()

    try:
        while thr_circle.is_alive() or thr_pid.is_alive():
            now = time.monotonic()
            if now >= next_plot_t:
                _update_live_plot(fig_live, axes_live, lines_live,
                                  x0_leader, y0_leader)
                next_plot_t = now + live_dt
            time.sleep(0.02)   # ceder CPU sin bloquear

    except KeyboardInterrupt:
        print("\n🛑 EMERGENCIA...")
        _stop_all.set(); _circle_done.set()
        send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                         0.0, 0.0, 0.0, 0.0)
        time.sleep(0.5)

    # Última actualización del live plot antes de cerrar
    _update_live_plot(fig_live, axes_live, lines_live, x0_leader, y0_leader)
    plt.pause(0.5)
    plt.close(fig_live)

    stop_readers.set()
    thr_read_L.join(timeout=2.0)
    thr_read_S.join(timeout=2.0)
    master_leader.close()
    master_follower.close()
    print("🔌 Conexiones cerradas.")

    csv_fname = save_csv()
    generate_rviz_replay(csv_fname)
    plot_results()   # gráficas post-vuelo bloqueantes
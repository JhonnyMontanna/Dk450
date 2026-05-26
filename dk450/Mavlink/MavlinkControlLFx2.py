#!/usr/bin/env python3
"""
LF_Circulo_RT.py — Líder-Seguidor en Tiempo Real con Círculo
=============================================================
Un solo script controla AMBOS drones simultáneamente:

  LÍDER   (Drone 1, UDP:14552) → ejecuta trayectoria circular con setpoints de posición
  SEGUIDOR (Drone 2, UDP:14553) → PID en tiempo real: lee telemetría del líder
                                   y le envía comandos de velocidad

Arquitectura de hilos:
  hilo_reader_lider    — lee LOCAL_POSITION_NED + ATTITUDE del líder
  hilo_reader_seguidor — lee LOCAL_POSITION_NED + ATTITUDE del seguidor
  hilo_circulo         — loop de control del líder (setpoints de posición)
  hilo_pid_seguidor    — loop PID del seguidor (comandos de velocidad)
  hilo_principal       — espera input del usuario, lanza maniobra, grafica al final

Ley de control del seguidor (ec. pid_velocidad):
    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L − v_S)
    FF    = ψ̇_L · Rz(π/2) · d(t)
    d(t)  = [d·cos(ψ_L+α), d·sin(ψ_L+α), Δz]
    e_ψ   = wrap(ψ_L − ψ_S)
"""

import math
import time
import csv
import threading
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from pymavlink import mavutil

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── Conexiones (un puerto por drone) ──────────────────────────────────────────
LEADER_CONN    = 'udp:127.0.0.1:14552'   # Drone 1 — LÍDER   (-I0, TCP 5760)
FOLLOWER_CONN  = 'udp:127.0.0.1:14553'   # Drone 2 — SEGUIDOR (-I1, TCP 5770)

LEADER_SYSID    = 1
LEADER_COMPID   = 0
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# ── Trayectoria del Líder (círculo) ───────────────────────────────────────────
RADIUS        = 2.0    # metros
ANGULAR_SPEED = 0.2    # rad/s  (vuelta completa en ~21 s)
LEADER_RATE   = 50     # Hz — frecuencia de envío de setpoints al líder

# Convergencia al punto final (fase de "cola" del líder)
CONV_RADIUS  = 0.15    # m
CONV_SPEED   = 0.10    # m/s
CONV_HOLD    = 1.0     # s consecutivos dentro del criterio
CONV_TIMEOUT = 15.0    # s máximos de espera

# ── PID del Seguidor ──────────────────────────────────────────────────────────
FOLLOWER_RATE = 20     # Hz — frecuencia del PID del seguidor

# Offset polar (posición del seguidor respecto al líder)
#   math.pi      → detrás
#   0            → delante
#   math.pi/2    → izquierda
#   -math.pi/2   → derecha
OFFSET_D     = 1.5
OFFSET_ALPHA = -math.pi /2   # detrás del líder
OFFSET_DZ    = 1.0        # seguidor 1 m más alto (seguridad)

# Ganancias PID posición
KP   = 0.5
KI   = 0.0    # activar gradualmente una vez validado Kp
KD   = 0.0

# Ganancias PID yaw
KP_YAW = 0.8
KI_YAW = 0.0
KD_YAW = 0.1

# Anti-windup
INTEGRAL_LIMIT     = 2.0   # m·s
INTEGRAL_YAW_LIMIT = 1.0   # rad·s

# Saturación de salida
V_MAX        = 2.0    # m/s horizontal
V_MAX_Z      = 1.0    # m/s vertical
YAW_RATE_MAX = 1.0    # rad/s

# ── Máscaras MAVLink ──────────────────────────────────────────────────────────
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
# ESTADO COMPARTIDO (thread-safe)
# ══════════════════════════════════════════════════════════════════════════════
_lock = threading.Lock()

def _empty_state():
    return dict(x=None, y=None, z=None,
                vx=None, vy=None, vz=None,
                yaw=None, yaw_rate=None)

_leader_state   = _empty_state()
_follower_state = _empty_state()

# Buffers de log (append desde hilos, leídos al final)
_log_lock = threading.Lock()
_log = dict(
    t=[],
    # Líder
    lx=[], ly=[], lz=[], lvx=[], lvy=[], lvz=[], l_yaw=[], l_yawrate=[],
    # Líder setpoint
    lx_sp=[], ly_sp=[],
    # Seguidor
    sx=[], sy=[], sz=[], svx=[], svy=[], svz=[], s_yaw=[], s_yawrate=[],
    # Setpoint seguidor y errores
    xd=[], yd=[], zd=[], ex=[], ey=[], ez=[], e_yaw=[], dist_xy=[],
    # Comandos
    vx_cmd=[], vy_cmd=[], vz_cmd=[], yaw_rate_cmd=[],
)

# Evento global: el líder terminó el círculo (señal para el PID)
_circle_done = threading.Event()

# Evento global: parar todo
_stop_all = threading.Event()

# ══════════════════════════════════════════════════════════════════════════════
# HILOS LECTORES MAVLink
# ══════════════════════════════════════════════════════════════════════════════
def _reader(master, state_dict, stop_event):
    """Lee LOCAL_POSITION_NED + ATTITUDE de un drone y actualiza state_dict."""
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        mtype = msg.get_type()
        with _lock:
            if mtype == 'LOCAL_POSITION_NED':
                state_dict['x']  = msg.x
                state_dict['y']  = msg.y
                state_dict['z']  = -msg.z    # NED → altura positiva hacia arriba
                state_dict['vx'] = msg.vx
                state_dict['vy'] = msg.vy
                state_dict['vz'] = -msg.vz
            elif mtype == 'ATTITUDE':
                state_dict['yaw']      = msg.yaw
                state_dict['yaw_rate'] = msg.yawspeed


def get_leader():
    with _lock:
        return dict(_leader_state)

def get_follower():
    with _lock:
        return dict(_follower_state)

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
        x, y, z_ned,
        0, 0, 0,
        0, 0, 0,
        yaw, 0
    )

def send_vel_yawrate(master, sysid, compid, vx, vy, vz_ned, yaw_rate):
    master.mav.set_position_target_local_ned_send(
        0, sysid, compid,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz_ned,
        0, 0, 0,
        0, yaw_rate
    )

# ══════════════════════════════════════════════════════════════════════════════
# MATEMÁTICAS PID
# ══════════════════════════════════════════════════════════════════════════════
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

def clamp(val, limit):
    return max(-limit, min(limit, val))

def compute_offset(psi_L):
    angle = psi_L + OFFSET_ALPHA
    return (OFFSET_D * math.cos(angle),
            OFFSET_D * math.sin(angle),
            OFFSET_DZ)

def compute_ff(yaw_rate_L, dx, dy):
    return yaw_rate_L * (-dy), yaw_rate_L * dx, 0.0

class PIDState:
    def __init__(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0

def compute_control(L, S, pid, dt):
    dx, dy, dz = compute_offset(L['yaw'])
    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz

    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']

    pid.integral[0] = clamp(pid.integral[0] + ex * dt, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * dt, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * dt, INTEGRAL_LIMIT)

    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    ff_x, ff_y, ff_z = compute_ff(L['yaw_rate'], dx, dy)

    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z

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
        YAW_RATE_MAX
    )

    return vx, vy, vz_ned, yaw_rate_cmd, xd, yd, zd, ex, ey, ez, e_yaw

# ══════════════════════════════════════════════════════════════════════════════
# HILO A — CÍRCULO DEL LÍDER
# ══════════════════════════════════════════════════════════════════════════════
def _thread_circle(master_leader):
    """
    Ejecuta la trayectoria circular del líder.
    Señaliza _circle_done cuando termina (incluida la fase de cola).
    """
    dt       = 1.0 / LEADER_RATE
    duration = 2 * math.pi / ANGULAR_SPEED
    steps    = int(duration / dt)

    # Esperar posición inicial del líder
    while True:
        L = get_leader()
        if state_ready(L):
            break
        time.sleep(0.05)

    x0, y0, z0_up = L['x'], L['y'], L['z']
    z0_ned = -z0_up    # NED: z negativo = altura positiva
    cx     = x0 + RADIUS
    cy     = y0
    theta0 = math.pi

    print(f"\n[LÍDER] Posición inicial: x={x0:.2f}  y={y0:.2f}  z={z0_up:.2f}")
    print(f"[LÍDER] Círculo: R={RADIUS} m  ω={ANGULAR_SPEED} rad/s  "
          f"duración={duration:.1f} s ({steps} pasos @ {LEADER_RATE} Hz)")

    next_t = time.monotonic()

    for i in range(steps):
        if _stop_all.is_set():
            return

        t_sched = i * dt
        theta   = theta0 + ANGULAR_SPEED * t_sched
        x_sp    = cx + RADIUS * math.cos(theta)
        y_sp    = cy + RADIUS * math.sin(theta)
        yaw     = theta + math.pi / 2

        send_pos_yaw(master_leader, LEADER_SYSID, LEADER_COMPID,
                     x_sp, y_sp, z0_ned, yaw)

        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola: mantener el último setpoint hasta convergencia ──
    x_final, y_final = x_sp, y_sp
    t_tail_start = time.monotonic()
    t_in_zone    = None

    print("[LÍDER] Círculo completado. Esperando convergencia al punto final...")

    while not _stop_all.is_set():
        now = time.monotonic()
        if now - t_tail_start > CONV_TIMEOUT:
            print("[LÍDER] ⚠️  Timeout convergencia.")
            break

        L = get_leader()
        if state_ready(L):
            dist  = math.hypot(L['x'] - x_final, L['y'] - y_final)
            speed = math.hypot(L['vx'], L['vy'])
            send_pos_yaw(master_leader, LEADER_SYSID, LEADER_COMPID,
                         x_final, y_final, z0_ned, yaw)

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f"[LÍDER] ✅ Convergido: dist={dist:.3f} m  vel={speed:.3f} m/s")
                    break
            else:
                t_in_zone = None

        time.sleep(dt)

    _circle_done.set()
    print("[LÍDER] 🏁 Trayectoria finalizada.")

# ══════════════════════════════════════════════════════════════════════════════
# HILO B — PID DEL SEGUIDOR
# ══════════════════════════════════════════════════════════════════════════════
def _thread_pid(master_follower, start_time_ref):
    """
    Loop PID del seguidor a FOLLOWER_RATE Hz.
    Corre en paralelo con el hilo del círculo del líder.
    Se detiene cuando _circle_done está seteado Y el seguidor converge,
    o cuando _stop_all se activa.
    """
    dt  = 1.0 / FOLLOWER_RATE
    pid = PIDState()

    # Esperar telemetría de ambos drones
    print("[SEGUIDOR] Esperando telemetría de ambos drones...", end='', flush=True)
    while True:
        L, S = get_leader(), get_follower()
        if state_ready(L) and state_ready(S):
            break
        if _stop_all.is_set():
            return
        time.sleep(0.05)
    print(" ✅ listo.")

    next_t = time.monotonic()

    while not _stop_all.is_set():
        L = get_leader()
        S = get_follower()

        if state_ready(L) and state_ready(S):
            vx, vy, vz_ned, yaw_rate_cmd, xd, yd, zd, ex, ey, ez, e_yaw = \
                compute_control(L, S, pid, dt)

            send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                             vx, vy, vz_ned, yaw_rate_cmd)

            t_el     = time.monotonic() - start_time_ref
            dist_xy  = math.hypot(L['x'] - S['x'], L['y'] - S['y'])

            # Guardar en log global
            with _log_lock:
                _log['t'].append(t_el)
                _log['lx'].append(L['x']);  _log['ly'].append(L['y']);  _log['lz'].append(L['z'])
                _log['lvx'].append(L['vx']); _log['lvy'].append(L['vy']); _log['lvz'].append(L['vz'])
                _log['l_yaw'].append(L['yaw']); _log['l_yawrate'].append(L['yaw_rate'])
                _log['sx'].append(S['x']);  _log['sy'].append(S['y']);  _log['sz'].append(S['z'])
                _log['svx'].append(S['vx']); _log['svy'].append(S['vy']); _log['svz'].append(S['vz'])
                _log['s_yaw'].append(S['yaw']); _log['s_yawrate'].append(S['yaw_rate'])
                _log['xd'].append(xd); _log['yd'].append(yd); _log['zd'].append(zd)
                _log['ex'].append(ex); _log['ey'].append(ey); _log['ez'].append(ez)
                _log['e_yaw'].append(e_yaw); _log['dist_xy'].append(dist_xy)
                _log['vx_cmd'].append(vx); _log['vy_cmd'].append(vy)
                _log['vz_cmd'].append(-vz_ned); _log['yaw_rate_cmd'].append(yaw_rate_cmd)

            # Consola cada 5 s
            if int(t_el) % 5 == 0 and int(t_el - dt) % 5 != 0:
                err_xy = math.hypot(ex, ey)
                print(f"  t={t_el:6.1f}s | err_xy={err_xy:.3f}m  dist={dist_xy:.2f}m"
                      f"  e_yaw={math.degrees(e_yaw):.1f}°"
                      f"  vx={vx:.2f} vy={vy:.2f}")

        # Si el líder ya terminó, detener el PID en el próximo ciclo
        if _circle_done.is_set():
            break

        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Detener seguidor
    send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                     0.0, 0.0, 0.0, 0.0)
    print("[SEGUIDOR] ⏹️  Control detenido.")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_csv():
    fname = f'lf_circulo_{int(time.time())}.csv'
    with _log_lock:
        keys = list(_log.keys())
        rows = list(zip(*[_log[k] for k in keys]))
    with open(fname, 'w', newline='') as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)
    print(f"📄 CSV guardado: {fname}  ({len(rows)} muestras)")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS FINALES
# ══════════════════════════════════════════════════════════════════════════════
def plot_results():
    with _log_lock:
        t       = np.array(_log['t'])
        lx      = np.array(_log['lx']);   ly  = np.array(_log['ly'])
        sx      = np.array(_log['sx']);   sy  = np.array(_log['sy'])
        xd      = np.array(_log['xd']);   yd  = np.array(_log['yd'])
        lz      = np.array(_log['lz']);   sz  = np.array(_log['sz']);  zd = np.array(_log['zd'])
        ex      = np.array(_log['ex']);   ey  = np.array(_log['ey']); ez = np.array(_log['ez'])
        e_yaw   = np.array(_log['e_yaw'])
        dist_xy = np.array(_log['dist_xy'])
        psiL    = np.array(_log['l_yaw'])
        psiS    = np.array(_log['s_yaw'])

    if len(t) == 0:
        print("⚠️  Sin datos para graficar.")
        return

    err_xy = np.sqrt(ex**2 + ey**2)
    print(f"\n📊 Error seguidor RMS xy = {np.sqrt(np.mean(err_xy**2)):.4f} m  "
          f"| máx = {np.max(err_xy):.4f} m")
    print(f"   Distancia L-S  media = {np.mean(dist_xy):.3f} m  "
          f"| std = {np.std(dist_xy):.3f} m  (objetivo={OFFSET_D} m)")

    # ── Fig 1: Trayectorias XY ────────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 7))
    theta_th  = np.linspace(0, 2*math.pi, 500)
    cx_th = lx[0] + RADIUS
    cy_th = ly[0]
    ax.plot(cx_th + RADIUS*np.cos(theta_th),
            cy_th + RADIUS*np.sin(theta_th),
            'k:', lw=1, label='Círculo teórico')
    ax.plot(lx, ly, 'g-',  lw=2,   label='Líder real')
    ax.plot(xd, yd, 'r--', lw=1.2, label='Setpoint seguidor')
    ax.plot(sx, sy, 'b-',  lw=2,   label='Seguidor real')
    ax.scatter(lx[0], ly[0], c='g', marker='o', s=80, zorder=5, label='Inicio líder')
    ax.scatter(sx[0], sy[0], c='b', marker='o', s=80, zorder=5, label='Inicio seguidor')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectorias XY — Líder y Seguidor')
    ax.axis('equal'); ax.legend(); ax.grid(True, alpha=0.4)
    fig1.tight_layout()

    # ── Fig 2: Posición vs tiempo (X, Y, Z) ──────────────────────────────────
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig2.suptitle('Posición vs Tiempo', fontsize=12)
    for ax, data_d, data_l, data_s, lbl in zip(
            axes2,
            [xd, yd, zd], [lx, ly, lz], [sx, sy, sz],
            ['X [m]', 'Y [m]', 'Z altura [m]']):
        ax.plot(t, data_d, 'r--', lw=1.2, label='Setpoint S')
        ax.plot(t, data_l, 'g-',  lw=1.5, label='Líder')
        ax.plot(t, data_s, 'b-',  lw=1.5, label='Seguidor')
        ax.set_ylabel(lbl); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    axes2[-1].set_xlabel('Tiempo [s]')
    fig2.tight_layout()

    # ── Fig 3: Errores del seguidor ───────────────────────────────────────────
    fig3, axes3 = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    fig3.suptitle('Errores del Seguidor', fontsize=12)
    for ax, data, lbl in zip(axes3,
            [ex, ey, ez, np.degrees(e_yaw)],
            ['ex [m]', 'ey [m]', 'ez [m]', 'eψ [°]']):
        ax.plot(t, data, lw=1.2)
        ax.axhline(0, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.4)
    axes3[-1].set_xlabel('Tiempo [s]')
    fig3.tight_layout()

    # ── Fig 4: Distancia L-S y yaw ────────────────────────────────────────────
    fig4, (ax_d, ax_y) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig4.suptitle('Distancia y Yaw', fontsize=12)
    ax_d.plot(t, dist_xy, 'b-', lw=1.5, label='Distancia XY real')
    ax_d.axhline(OFFSET_D, color='r', ls='--', lw=1.2, label=f'Objetivo {OFFSET_D} m')
    ax_d.set_ylabel('Distancia [m]'); ax_d.legend(); ax_d.grid(True, alpha=0.4)
    ax_y.plot(t, np.degrees(psiL), 'g-', lw=1.2, label='ψ Líder')
    ax_y.plot(t, np.degrees(psiS), 'b-', lw=1.2, label='ψ Seguidor')
    ax_y.set_ylabel('Yaw [°]'); ax_y.set_xlabel('Tiempo [s]')
    ax_y.legend(); ax_y.grid(True, alpha=0.4)
    fig4.tight_layout()

    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 65)
    print("  LF_Circulo_RT — Líder-Seguidor en Tiempo Real")
    print("=" * 65)
    print(f"  Líder    → {LEADER_CONN}   (SYSID {LEADER_SYSID})")
    print(f"  Seguidor → {FOLLOWER_CONN}  (SYSID {FOLLOWER_SYSID})")
    print(f"  Círculo  : R={RADIUS} m  ω={ANGULAR_SPEED} rad/s  "
          f"T={2*math.pi/ANGULAR_SPEED:.1f} s")
    print(f"  Offset   : d={OFFSET_D} m  α={math.degrees(OFFSET_ALPHA):.0f}°  "
          f"Δz={OFFSET_DZ} m")
    print(f"  Ganancias: Kp={KP}  Ki={KI}  Kd={KD}  |  "
          f"Kp_yaw={KP_YAW}  Kd_yaw={KD_YAW}")
    print("=" * 65)

    # ── Conectar ──────────────────────────────────────────────────────────────
    print(f"\n🔌 Conectando al LÍDER    ({LEADER_CONN})...")
    master_leader = mavutil.mavlink_connection(LEADER_CONN)
    master_leader.wait_heartbeat()
    print(f"   ✅ Heartbeat líder    — SYS={master_leader.target_system}")

    print(f"🔌 Conectando al SEGUIDOR ({FOLLOWER_CONN})...")
    master_follower = mavutil.mavlink_connection(FOLLOWER_CONN)
    master_follower.wait_heartbeat()
    print(f"   ✅ Heartbeat seguidor — SYS={master_follower.target_system}")

    # ── Hilos lectores ────────────────────────────────────────────────────────
    stop_readers = threading.Event()
    thr_read_L = threading.Thread(
        target=_reader, args=(master_leader,   _leader_state,   stop_readers),
        daemon=True, name='reader-leader')
    thr_read_S = threading.Thread(
        target=_reader, args=(master_follower, _follower_state, stop_readers),
        daemon=True, name='reader-follower')
    thr_read_L.start()
    thr_read_S.start()

    # ── Esperar telemetría inicial ────────────────────────────────────────────
    print("\n⏳ Esperando telemetría de AMBOS drones", end='', flush=True)
    t0_wait = time.monotonic()
    while True:
        L, S = get_leader(), get_follower()
        if state_ready(L) and state_ready(S):
            break
        if time.monotonic() - t0_wait > 30.0:
            print("\n❌ Timeout esperando telemetría. "
                  "Verifica que ambos drones estén activos.")
            stop_readers.set()
            raise SystemExit(1)
        print('.', end='', flush=True)
        time.sleep(0.3)
    print(" ✅")
    L, S = get_leader(), get_follower()
    print(f"   Líder    → x={L['x']:.2f}  y={L['y']:.2f}  z={L['z']:.2f}  "
          f"ψ={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor → x={S['x']:.2f}  y={S['y']:.2f}  z={S['z']:.2f}  "
          f"ψ={math.degrees(S['yaw']):.1f}°")

    # ── Esperar orden del usuario ─────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Ambos drones deben estar armados y en modo GUIDED")
    print("  El seguidor debe estar en hover estable")
    print("─" * 65)
    input("  ▶️  Presiona ENTER para iniciar el círculo + seguimiento...\n")

    start_time = time.monotonic()

    # ── Lanzar hilos de control ───────────────────────────────────────────────
    thr_circle = threading.Thread(
        target=_thread_circle, args=(master_leader,),
        daemon=True, name='circle-leader')

    thr_pid = threading.Thread(
        target=_thread_pid, args=(master_follower, start_time),
        daemon=True, name='pid-follower')

    thr_pid.start()       # PID arranca primero (espera telemetría internamente)
    thr_circle.start()    # Círculo arranca: el líder empieza a moverse

    print("🚀 ¡En marcha! Ctrl+C para parada de emergencia.\n")

    # ── Esperar a que ambos hilos terminen ────────────────────────────────────
    try:
        thr_circle.join()
        thr_pid.join()
    except KeyboardInterrupt:
        print("\n🛑 PARADA DE EMERGENCIA — deteniendo seguidor...")
        _stop_all.set()
        _circle_done.set()
        send_vel_yawrate(master_follower, FOLLOWER_SYSID, FOLLOWER_COMPID,
                         0.0, 0.0, 0.0, 0.0)
        time.sleep(0.5)

    # ── Limpieza ──────────────────────────────────────────────────────────────
    stop_readers.set()
    thr_read_L.join(timeout=2.0)
    thr_read_S.join(timeout=2.0)
    master_leader.close()
    master_follower.close()
    print("🔌 Conexiones cerradas.")

    # ── CSV + gráficas ────────────────────────────────────────────────────────
    save_csv()
    plot_results()
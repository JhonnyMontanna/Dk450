#!/usr/bin/env python3
"""
Vuelo en lemniscata de Bernoulli — sin salto de inicio
El primer setpoint es EXACTAMENTE la posición actual del dron.

Dos modos:
  'center' → dron posicionado en el cruce central de la lemniscata
             t0 = π/2  →  lemniscate_pos(a, b, π/2) = (0, 0)  ✓
  'tip'    → dron posicionado en el extremo del lóbulo derecho
             t0 = 0    →  lemniscate_pos(a, b, 0)   = (a, 0)  ✓

En ambos casos el centro geométrico de la figura se calcula a partir
de la posición del dron, y el recorrido termina exactamente donde empezó.
"""
import time
import math
import threading
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
CONN          = 'udp:127.0.0.1:14552'
SYSID         = 2
COMPID        = 0

AXIS_A        = 4.0    # semieje X — ancho de cada lóbulo [m]
AXIS_B        = 2.0    # semieje Y — alto de cada lóbulo [m]
ANGULAR_SPEED = 0.3    # ω [rad/s]
RATE          = 50     # Hz

# 'center' → dron está en el cruce central de la figura
# 'tip'    → dron está en el extremo del lóbulo derecho
START_MODE    = 'center'

CONV_RADIUS   = 0.15
CONV_SPEED    = 0.10
CONV_HOLD     = 1.0
CONV_TIMEOUT  = 15.0

TYPE_MASK_POS_YAW = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# ===============================
# PARAMETRIZACIÓN DE LA LEMNISCATA
# ===============================
def lemniscate_pos(a, b, t):
    """
    Posición relativa al centro geométrico de la figura.
      t = π/2  →  (0, 0)   cruce central
      t = 0    →  (a, 0)   extremo derecho
    """
    s, c = math.sin(t), math.cos(t)
    d = 1.0 + s * s
    return a * c / d, b * s * c / d

def lemniscate_vel(a, b, omega, t):
    """Derivada analítica × ω."""
    s, c = math.sin(t), math.cos(t)
    d = 1.0 + s * s
    dx_dt = a * (-s * d - c * 2.0 * s * c) / (d * d)
    dy_dt = b * ((c*c - s*s) * d - s * c * 2.0 * s * c) / (d * d)
    return dx_dt * omega, dy_dt * omega

def lemniscate_yaw(a, b, omega, t):
    vx, vy = lemniscate_vel(a, b, omega, t)
    # Evitar yaw indefinido en los ceros de velocidad (cruces)
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return 0.0
    return math.atan2(vy, vx)

def compute_center_offset(x0, y0):
    """
    Calcula las coordenadas del centro geométrico de la lemniscata
    en el marco LOCAL_NED, a partir de la posición del dron.

    - Modo 'center': dron en el cruce → centro = (x0, y0)
    - Modo 'tip':    dron en extremo  → centro = (x0 - a, y0)
                     porque lemniscate_pos(a, b, 0) = (a, 0)
                     y el centro es el origen de esa expresión
    """
    if START_MODE == 'center':
        return x0, y0          # centro coincide con el dron
    else:
        return x0 - AXIS_A, y0 # el dron está a 'a' metros del centro

def theta_start():
    """
    Ángulo paramétrico inicial tal que lemniscate_pos(t0) = posición relativa del dron.
      center → t0 = π/2  (punto de cruce, pos relativa = 0,0)
      tip    → t0 = 0    (extremo derecho, pos relativa = a, 0)
    """
    return math.pi / 2 if START_MODE == 'center' else 0.0

# ===============================
# ESTADO COMPARTIDO (thread-safe)
# ===============================
_state_lock = threading.Lock()
_latest_pos = None

def _mavlink_reader(master, stop_event):
    global _latest_pos
    while not stop_event.is_set():
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.1)
        if msg:
            with _state_lock:
                _latest_pos = msg

def get_latest_pos():
    with _state_lock:
        return _latest_pos

# ===============================
# FUNCIONES AUXILIARES
# ===============================
def send_position_yaw(master, x, y, z, yaw):
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        yaw, 0
    )

def wait_local_position_ready():
    print("⏳ Esperando posición inicial...", end='', flush=True)
    while get_latest_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    return get_latest_pos()

# ===============================
# LOOP DE CONTROL PRINCIPAL
# ===============================
def fly_lemniscate(master, duration):
    pos0 = get_latest_pos()
    x0, y0, z0 = pos0.x, pos0.y, pos0.z

    # Centro geométrico de la figura en coordenadas NED
    cx_lem, cy_lem = compute_center_offset(x0, y0)
    t0 = theta_start()

    # Verificación: el primer setpoint debe coincidir con (x0, y0)
    lx0, ly0 = lemniscate_pos(AXIS_A, AXIS_B, t0)
    sp0_x = cx_lem + lx0
    sp0_y = cy_lem + ly0
    err0 = math.hypot(sp0_x - x0, sp0_y - y0)

    print(f"📍 Posición dron:    x={x0:.3f}, y={y0:.3f}, z={z0:.3f}")
    print(f"📐 Centro lemniscata: x={cx_lem:.3f}, y={cy_lem:.3f}")
    print(f"🎯 Primer setpoint:   x={sp0_x:.3f}, y={sp0_y:.3f}  "
          f"(error={err0:.4f} m — debe ser ~0)")
    print(f"🔧 Modo de inicio: '{START_MODE}'  θ₀ = {math.degrees(t0):.1f}°")

    dt    = 1.0 / RATE
    steps = int(duration / dt)

    log = {k: deque() for k in
           ('t', 'x_sp', 'y_sp', 'x', 'y',
            'vx_real', 'vy_real', 'vx_des', 'vy_des')}

    print(f"∞  Lemniscata: a={AXIS_A} m, b={AXIS_B} m, "
          f"ω={ANGULAR_SPEED} rad/s, T={duration:.2f} s ({steps} pasos @ {RATE} Hz)")

    next_t = time.monotonic()

    for i in range(steps):
        t_sched = i * dt
        theta   = t0 + ANGULAR_SPEED * t_sched   # avanza desde t0

        lx, ly = lemniscate_pos(AXIS_A, AXIS_B, theta)
        x_sp   = cx_lem + lx
        y_sp   = cy_lem + ly
        yaw    = lemniscate_yaw(AXIS_A, AXIS_B, ANGULAR_SPEED, theta)

        send_position_yaw(master, x_sp, y_sp, z0, yaw)

        vx_des, vy_des = lemniscate_vel(AXIS_A, AXIS_B, ANGULAR_SPEED, theta)

        pos = get_latest_pos()
        if pos is not None:
            log['t'].append(t_sched)
            log['x_sp'].append(x_sp);      log['y_sp'].append(y_sp)
            log['x'].append(pos.x);        log['y'].append(pos.y)
            log['vx_real'].append(pos.vx); log['vy_real'].append(pos.vy)
            log['vx_des'].append(vx_des);  log['vy_des'].append(vy_des)

        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola ──
    x_final, y_final = x_sp, y_sp
    t_in_zone    = None
    t_tail_start = time.monotonic()
    t_offset     = steps * dt

    print(f"⏳ Convergiendo al punto final (radio={CONV_RADIUS} m)...")

    while True:
        now = time.monotonic()
        if now - t_tail_start > CONV_TIMEOUT:
            print("⚠️  Timeout — graficando con datos disponibles.")
            break

        pos = get_latest_pos()
        if pos is not None:
            dist  = math.hypot(pos.x - x_final, pos.y - y_final)
            speed = math.hypot(pos.vx, pos.vy)

            send_position_yaw(master, x_final, y_final, z0, yaw)

            t_tail = t_offset + (now - t_tail_start)
            log['t'].append(t_tail)
            log['x_sp'].append(x_final); log['y_sp'].append(y_final)
            log['x'].append(pos.x);      log['y'].append(pos.y)
            log['vx_real'].append(pos.vx); log['vy_real'].append(pos.vy)
            log['vx_des'].append(0.0);     log['vy_des'].append(0.0)

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f"✅ Convergencia: dist={dist:.3f} m, vel={speed:.3f} m/s")
                    break
            else:
                t_in_zone = None

        time.sleep(1.0 / RATE)

    print("⏹️  Registro completo.")
    return {k: list(v) for k, v in log.items()}, (x0, y0), (cx_lem, cy_lem)

# ===============================
# GRÁFICAS
# ===============================
def plot_results(log, x0, y0, cx_lem, cy_lem, duration):
    t_arr = np.array(log['t'])
    mask  = t_arr <= duration

    err = np.sqrt((np.array(log['x']) - np.array(log['x_sp']))**2 +
                  (np.array(log['y']) - np.array(log['y_sp']))**2)
    if mask.sum():
        print(f"📊 Error RMS (vuelo): {err[mask].mean():.4f} m  "
              f"|  máx: {err[mask].max():.4f} m")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Lemniscata a={AXIS_A} m, b={AXIS_B} m  '
                 f'[modo: {START_MODE}]', fontsize=13)

    for ax in axes[:2]:
        ax.axvline(duration, color='gray', linestyle=':', linewidth=1)

    axes[0].plot(t_arr, log['vx_des'], 'b-',  lw=1.5, label='Vx deseada')
    axes[0].plot(t_arr, log['vx_real'], 'r--', lw=1.2, label='Vx real', alpha=0.85)
    axes[0].set(xlabel='Tiempo [s]', ylabel='Vx [m/s]', title='Velocidad en X')
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_arr, log['vy_des'], 'b-',  lw=1.5, label='Vy deseada')
    axes[1].plot(t_arr, log['vy_real'], 'r--', lw=1.2, label='Vy real', alpha=0.85)
    axes[1].set(xlabel='Tiempo [s]', ylabel='Vy [m/s]', title='Velocidad en Y')
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    ax = axes[2]
    th = np.linspace(0, 2 * np.pi, 800)
    lx_t = np.array([lemniscate_pos(AXIS_A, AXIS_B, t)[0] for t in th]) + cx_lem
    ly_t = np.array([lemniscate_pos(AXIS_A, AXIS_B, t)[1] for t in th]) + cy_lem
    ax.plot(lx_t, ly_t, 'g:', lw=1.2, label='Lemniscata teórica')
    ax.plot(np.array(log['x_sp']), np.array(log['y_sp']),
            'b--', lw=1.2, label='Setpoints', alpha=0.7)

    x_arr = np.array(log['x']); y_arr = np.array(log['y'])
    ax.plot(x_arr[mask],  y_arr[mask],  'r-',  lw=2,   label='Real (vuelo)')
    ax.plot(x_arr[~mask], y_arr[~mask], color='orange', lw=1.5,
            ls='--', label='Real (cola)')
    ax.scatter(x0, y0, c='red',    marker='o', s=60, zorder=6, label='Inicio dron')
    ax.scatter(cx_lem, cy_lem, c='purple', marker='+',
               s=80, zorder=6, label='Centro figura')

    ax.set(xlabel='X [m]', ylabel='Y [m]', title='Trayectoria XY')
    ax.axis('equal'); ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

    stop_reader = threading.Event()
    reader = threading.Thread(
        target=_mavlink_reader, args=(master, stop_reader),
        daemon=True, name='mavlink-reader'
    )
    reader.start()

    wait_local_position_ready()

    duration = 2 * math.pi / ANGULAR_SPEED

    log, (x0, y0), (cx_lem, cy_lem) = fly_lemniscate(master, duration)

    stop_reader.set()
    reader.join(timeout=2.0)
    master.close()

    plot_results(log, x0, y0, cx_lem, cy_lem, duration)
#!/usr/bin/env python3
"""
MavlinkCirculoLog.py — Vuelo circular con grabación de trayectoria real
========================================================================
Ejecuta el círculo exactamente igual que MavlinkCirculo.py y además
graba la posición real del drone (x, y, z, vx, vy, vz, yaw, yaw_rate)
en un CSV para ser usado como referencia por MavlinkControlLFLog.py.

Formato CSV generado (una fila por iteración de control):
  t, x, y, z, vx, vy, vz, yaw, yaw_rate

  Ejes: x, y, z con z POSITIVO HACIA ARRIBA (se invierte z_ned).
        yaw en radianes, (-π, π].

Uso:
  python3 MavlinkCirculoLog.py
  → genera circulo_log_<timestamp>.csv al terminar
"""

import time
import math
import threading
import csv
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
CONN          = 'udp:127.0.0.1:14552'
SYSID         = 1
COMPID        = 0

RADIUS        = 4.0     # metros
ANGULAR_SPEED = 0.4     # rad/s
RATE          = 50      # Hz

# Criterio de convergencia al punto final (fase de cola)
CONV_RADIUS  = 0.15     # m
CONV_SPEED   = 0.10     # m/s
CONV_HOLD    = 1.0      # s consecutivos dentro del criterio
CONV_TIMEOUT = 15.0     # s máximos de espera post-loop

OUTPUT_FILE  = f'circulo_log_{int(time.time())}.csv'

# ──────────────────────────────────────────────────────────────────────────────
# MÁSCARAS MAVLINK
# ──────────────────────────────────────────────────────────────────────────────
TYPE_MASK_POS_YAW = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO — hilo lector MAVLink
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_state = dict(x=None, y=None, z=None,
              vx=None, vy=None, vz=None,
              yaw=None, yaw_rate=None)


def _mavlink_reader(master, stop_event):
    """Lee LOCAL_POSITION_NED y ATTITUDE. Filtra solo SYSID del drone."""
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        if msg.get_srcSystem() != SYSID:
            continue

        mtype = msg.get_type()
        with _state_lock:
            if mtype == 'LOCAL_POSITION_NED':
                _state['x']  = msg.x
                _state['y']  = msg.y
                _state['z']  = -msg.z     # NED → altitud positiva
                _state['vx'] = msg.vx
                _state['vy'] = msg.vy
                _state['vz'] = -msg.vz    # ídem velocidad vertical
            elif mtype == 'ATTITUDE':
                _state['yaw']      = msg.yaw
                _state['yaw_rate'] = msg.yawspeed


def get_state():
    with _state_lock:
        return dict(_state)


def state_ready(s):
    return all(v is not None for v in s.values())


def wait_state_ready():
    print('⏳ Esperando telemetría completa (pos + attitude)...', end='', flush=True)
    while True:
        s = get_state()
        if state_ready(s):
            print(' listo.')
            return s
        time.sleep(0.02)


# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE SETPOINT
# ──────────────────────────────────────────────────────────────────────────────
def send_position_yaw(master, x, y, z_ned, yaw):
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z_ned,
        0, 0, 0,
        0, 0, 0,
        yaw, 0
    )


# ──────────────────────────────────────────────────────────────────────────────
# LOOP DE VUELO CON GRABACIÓN
# ──────────────────────────────────────────────────────────────────────────────
def fly_and_record(master, duration):
    s0 = get_state()
    x0, y0, z0_pos = s0['x'], s0['y'], s0['z']
    z0_ned = -z0_pos   # convertir altitud positiva → z_ned
    print(f'📍 Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0_pos:.2f}')

    cx     = x0 + RADIUS
    cy     = y0
    theta0 = math.pi     # empieza desde x0,y0 mirando hacia atrás del centro
    dt     = 1.0 / RATE
    steps  = int(duration / dt)

    # Buffers de log (para gráfica y CSV)
    log = {k: deque() for k in ('t', 'x_sp', 'y_sp', 'x', 'y',
                                 'vx_real', 'vy_real', 'vx_des', 'vy_des')}
    # Buffer completo para CSV del líder (lo que necesita MavlinkControlLFLog)
    csv_buf = deque()   # cada elemento: (t, x, y, z, vx, vy, vz, yaw, yaw_rate)

    print(f'🌀 Círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, '
          f'duración={duration:.1f} s ({steps} pasos @ {RATE} Hz)')
    print(f'💾 Grabando posición real en: {OUTPUT_FILE}')

    next_t  = time.monotonic()
    t_start = next_t

    for i in range(steps):
        t_sched = i * dt
        theta   = theta0 + ANGULAR_SPEED * t_sched

        # Setpoint de posición (frame NED)
        x_sp = cx + RADIUS * math.cos(theta)
        y_sp = cy + RADIUS * math.sin(theta)
        yaw  = theta + math.pi / 2
        send_position_yaw(master, x_sp, y_sp, z0_ned, yaw)

        # Velocidad deseada teórica
        vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta)
        vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta)

        # Leer estado actual
        s = get_state()
        if state_ready(s):
            log['t'].append(t_sched)
            log['x_sp'].append(x_sp)
            log['y_sp'].append(y_sp)
            log['x'].append(s['x'])
            log['y'].append(s['y'])
            log['vx_real'].append(s['vx'])
            log['vy_real'].append(s['vy'])
            log['vx_des'].append(vx_des)
            log['vy_des'].append(vy_des)

            csv_buf.append((
                f'{t_sched:.5f}',
                f'{s["x"]:.6f}',  f'{s["y"]:.6f}',  f'{s["z"]:.6f}',
                f'{s["vx"]:.6f}', f'{s["vy"]:.6f}', f'{s["vz"]:.6f}',
                f'{s["yaw"]:.6f}', f'{s["yaw_rate"]:.6f}',
            ))

        # Timing absoluto
        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola ──────────────────────────────────────────────────────────
    x_final, y_final = x_sp, y_sp
    t_in_zone    = None
    t_tail_start = time.monotonic()
    t_offset     = steps * dt

    print(f'⏳ Esperando convergencia al punto final '
          f'(radio={CONV_RADIUS} m, vel<{CONV_SPEED} m/s)...')

    while True:
        now = time.monotonic()
        if now - t_tail_start > CONV_TIMEOUT:
            print('⚠️  Timeout de convergencia.')
            break

        s = get_state()
        if state_ready(s):
            dist  = math.hypot(s['x'] - x_final, s['y'] - y_final)
            speed = math.hypot(s['vx'], s['vy'])
            send_position_yaw(master, x_final, y_final, z0_ned, yaw)

            t_tail = t_offset + (now - t_tail_start)
            log['t'].append(t_tail)
            log['x_sp'].append(x_final)
            log['y_sp'].append(y_final)
            log['x'].append(s['x'])
            log['y'].append(s['y'])
            log['vx_real'].append(s['vx'])
            log['vy_real'].append(s['vy'])
            log['vx_des'].append(0.0)
            log['vy_des'].append(0.0)

            csv_buf.append((
                f'{t_tail:.5f}',
                f'{s["x"]:.6f}',  f'{s["y"]:.6f}',  f'{s["z"]:.6f}',
                f'{s["vx"]:.6f}', f'{s["vy"]:.6f}', f'{s["vz"]:.6f}',
                f'{s["yaw"]:.6f}', f'{s["yaw_rate"]:.6f}',
            ))

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f'✅ Convergencia: dist={dist:.3f} m, vel={speed:.3f} m/s')
                    break
            else:
                t_in_zone = None

        time.sleep(dt)

    print('⏹️  Vuelo completo.')

    # ── Escribir CSV ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'yaw_rate'])
        w.writerows(csv_buf)
    print(f'💾 CSV guardado: {OUTPUT_FILE}  ({len(csv_buf)} muestras)')

    return {k: list(v) for k, v in log.items()}, (x0, y0)


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(log, x0, y0, duration):
    t    = log['t']
    x_sp = log['x_sp']; y_sp = log['y_sp']
    x    = log['x'];    y    = log['y']
    vx_r = log['vx_real']; vy_r = log['vy_real']
    vx_d = log['vx_des'];  vy_d = log['vy_des']

    t_arr  = np.array(t)
    mask   = t_arr <= duration
    err    = np.sqrt((np.array(x) - np.array(x_sp))**2 +
                     (np.array(y) - np.array(y_sp))**2)
    err_c  = err[mask]
    if len(err_c):
        print(f'📊 Error posición RMS (círculo): {np.mean(err_c):.4f} m  '
              f'| máx: {np.max(err_c):.4f} m')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MavlinkCirculoLog — vuelo circular + grabación', fontsize=13)

    for ax_i in range(2):
        axes[ax_i].axvline(duration, color='gray', linestyle=':', linewidth=1,
                           label='Fin círculo')

    ax = axes[0]
    ax.plot(t, vx_d, 'b-',  linewidth=1.5, label='Vx deseada')
    ax.plot(t, vx_r, 'r--', linewidth=1.2, label='Vx real', alpha=0.85)
    ax.set_xlabel('t [s]'); ax.set_ylabel('Vx [m/s]')
    ax.set_title('Velocidad X'); ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(t, vy_d, 'b-',  linewidth=1.5, label='Vy deseada')
    ax.plot(t, vy_r, 'r--', linewidth=1.2, label='Vy real', alpha=0.85)
    ax.set_xlabel('t [s]'); ax.set_ylabel('Vy [m/s]')
    ax.set_title('Velocidad Y'); ax.legend(); ax.grid(True, alpha=0.4)

    ax = axes[2]
    theta_th = np.linspace(0, 2 * math.pi, 400)
    cx = x0 + RADIUS
    ax.plot(cx + RADIUS * np.cos(theta_th),
            y0 + RADIUS * np.sin(theta_th),
            'g:', linewidth=1.2, label='Círculo teórico')
    ax.plot(x_sp, y_sp, 'b--', linewidth=1.2, label='Setpoints', alpha=0.7)
    x_arr = np.array(x); y_arr = np.array(y)
    ax.plot(x_arr[mask],  y_arr[mask],  'r-', linewidth=2,   label='Real (círculo)')
    ax.plot(x_arr[~mask], y_arr[~mask], color='orange', linewidth=1.5,
            linestyle='--', label='Real (cola)')
    ax.scatter(x0, y0, c='k', marker='o', zorder=5, label='Inicio')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectoria XY'); ax.axis('equal')
    ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f'🔗 Conectado: SYS={master.target_system} COMP={master.target_component}')

    stop_reader = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader, args=(master, stop_reader),
        daemon=True, name='mavlink-reader'
    )
    reader_thread.start()

    wait_state_ready()

    duration = 2 * math.pi / ANGULAR_SPEED

    log, (x0, y0) = fly_and_record(master, duration)

    stop_reader.set()
    reader_thread.join(timeout=2.0)
    master.close()

    plot_results(log, x0, y0, duration)
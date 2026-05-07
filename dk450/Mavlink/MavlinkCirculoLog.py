#!/usr/bin/env python3
"""
MavlinkCirculoLog.py — Vuelo circular con grabación de trayectoria real
========================================================================
Ejecuta el círculo exactamente igual que MavlinkCirculo.py y además
graba la posición real del drone y el setpoint del líder en un CSV
para ser usado como referencia por MavlinkControlLFLog.py.

Formato CSV generado (una fila por iteración de control):
  t, x, y, z, vx, vy, vz, yaw, yaw_rate,
  x_sp, y_sp, z_sp,
  ex_L, ey_L

  Ejes: x, y, z con z POSITIVO HACIA ARRIBA (se invierte z_ned).
        yaw en radianes, (-pi, pi].

  x_sp, y_sp : setpoint de posicion XY del lider en cada muestra.
  z_sp       : altitud de vuelo constante del lider.
  ex_L       : error cartesiano en X:  x_sp - x_real
               Positivo = drone atrasado en X. Negativo = adelantado.
  ey_L       : error cartesiano en Y:  y_sp - y_real
               Positivo = drone atrasado en Y. Negativo = adelantado.

  La misma definicion se mantiene en la fase de cola (setpoint fijo),
  garantizando continuidad en la transicion circulo -> cola sin saltos.

Uso:
  python3 MavlinkCirculoLog.py
  -> genera circulo_log_<timestamp>.csv al terminar
"""

import time
import math
import threading
import csv
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────────────────────
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
CONV_TIMEOUT = 15.0     # s maximos de espera post-loop

OUTPUT_FILE  = f'circulo_log_{int(time.time())}.csv'

# ─────────────────────────────────────────────────────────────────────────────
# MASCARAS MAVLINK
# ─────────────────────────────────────────────────────────────────────────────
TYPE_MASK_POS_YAW = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO — hilo lector MAVLink
# ─────────────────────────────────────────────────────────────────────────────
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
                _state['z']  = -msg.z
                _state['vx'] = msg.vx
                _state['vy'] = msg.vy
                _state['vz'] = -msg.vz
            elif mtype == 'ATTITUDE':
                _state['yaw']      = msg.yaw
                _state['yaw_rate'] = msg.yawspeed


def get_state():
    with _state_lock:
        return dict(_state)


def state_ready(s):
    return all(v is not None for v in s.values())


def wait_state_ready():
    print('Esperando telemetria completa (pos + attitude)...', end='', flush=True)
    while True:
        s = get_state()
        if state_ready(s):
            print(' listo.')
            return s
        time.sleep(0.02)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIO DE SETPOINT
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DE VUELO CON GRABACION
# ─────────────────────────────────────────────────────────────────────────────
def fly_and_record(master, duration):
    s0 = get_state()
    x0, y0, z0_pos = s0['x'], s0['y'], s0['z']
    z0_ned = -z0_pos
    print(f'Posicion inicial: x={x0:.2f}, y={y0:.2f}, z={z0_pos:.2f}')

    cx     = x0 + RADIUS
    cy     = y0
    theta0 = math.pi
    dt     = 1.0 / RATE
    steps  = int(duration / dt)

    log = {k: deque() for k in ('t', 'x_sp', 'y_sp', 'x', 'y',
                                 'vx_real', 'vy_real', 'vx_des', 'vy_des')}
    csv_buf = deque()

    print(f'Circulo: R={RADIUS} m, w={ANGULAR_SPEED} rad/s, '
          f'duracion={duration:.1f} s ({steps} pasos @ {RATE} Hz)')
    print(f'Grabando en: {OUTPUT_FILE}')

    next_t  = time.monotonic()
    t_start = next_t

    # Setpoint enviado en el ciclo anterior
    x_sp_sent   = None
    y_sp_sent   = None
    t_sp_sent   = None
    vx_des_sent = None
    vy_des_sent = None

    for i in range(steps):
        t_sched = i * dt

        # ── 1. Leer telemetria y calcular error contra setpoint anterior ──────
        s = get_state()
        if state_ready(s) and x_sp_sent is not None:
            z_sp = z0_pos

            # Error cartesiano: setpoint enviado - posicion real
            # ex_L = x_sp - x  (positivo = atrasado en X)
            # ey_L = y_sp - y  (positivo = atrasado en Y)
            ex_L = x_sp_sent - s['x']
            ey_L = y_sp_sent - s['y']

            log['t'].append(t_sp_sent)
            log['x_sp'].append(x_sp_sent)
            log['y_sp'].append(y_sp_sent)
            log['x'].append(s['x'])
            log['y'].append(s['y'])
            log['vx_real'].append(s['vx'])
            log['vy_real'].append(s['vy'])
            log['vx_des'].append(vx_des_sent)
            log['vy_des'].append(vy_des_sent)

            csv_buf.append((
                f'{t_sp_sent:.5f}',
                f'{s["x"]:.6f}',  f'{s["y"]:.6f}',  f'{s["z"]:.6f}',
                f'{s["vx"]:.6f}', f'{s["vy"]:.6f}', f'{s["vz"]:.6f}',
                f'{s["yaw"]:.6f}', f'{s["yaw_rate"]:.6f}',
                f'{x_sp_sent:.6f}', f'{y_sp_sent:.6f}', f'{z_sp:.6f}',
                f'{ex_L:.6f}', f'{ey_L:.6f}',
            ))

        # ── 2. Calcular y enviar nuevo setpoint ───────────────────────────────
        theta  = theta0 + ANGULAR_SPEED * t_sched
        x_sp   = cx + RADIUS * math.cos(theta)
        y_sp   = cy + RADIUS * math.sin(theta)
        yaw    = theta + math.pi / 2
        send_position_yaw(master, x_sp, y_sp, z0_ned, yaw)

        vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta)
        vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta)

        x_sp_sent   = x_sp
        y_sp_sent   = y_sp
        t_sp_sent   = t_sched
        vx_des_sent = vx_des
        vy_des_sent = vy_des

        # ── 3. Timing ─────────────────────────────────────────────────────────
        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola ──────────────────────────────────────────────────────────
    x_final, y_final = x_sp, y_sp
    z_sp_final       = z0_pos
    t_in_zone    = None
    t_tail_start = time.monotonic()
    t_offset     = steps * dt

    print(f'Esperando convergencia al punto final '
          f'(radio={CONV_RADIUS} m, vel<{CONV_SPEED} m/s)...')

    while True:
        now = time.monotonic()
        if now - t_tail_start > CONV_TIMEOUT:
            print('Timeout de convergencia.')
            break

        s = get_state()
        if state_ready(s):
            dist  = math.hypot(s['x'] - x_final, s['y'] - y_final)
            speed = math.hypot(s['vx'], s['vy'])
            send_position_yaw(master, x_final, y_final, z0_ned, yaw)

            t_tail = t_offset + (now - t_tail_start)

            # Error cartesiano: misma definicion que en el loop
            # Continuidad garantizada: al entrar en la cola el setpoint
            # es el mismo ultimo punto del circulo, sin cambio de base.
            ex_L = x_final - s['x']
            ey_L = y_final - s['y']

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
                f'{x_final:.6f}', f'{y_final:.6f}', f'{z_sp_final:.6f}',
                f'{ex_L:.6f}', f'{ey_L:.6f}',
            ))

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f'Convergencia: dist={dist:.3f} m, vel={speed:.3f} m/s')
                    break
            else:
                t_in_zone = None

        time.sleep(dt)

    print('Vuelo completo.')

    # ── Escribir CSV ──────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            't', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'yaw_rate',
            'x_sp', 'y_sp', 'z_sp',
            'ex_L',   # error cartesiano X: x_sp - x
            'ey_L',   # error cartesiano Y: y_sp - y
        ])
        w.writerows(csv_buf)
    print(f'CSV guardado: {OUTPUT_FILE}  ({len(csv_buf)} muestras)')

    return {k: list(v) for k, v in log.items()}, (x0, y0)


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICAS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(log, x0, y0, duration):
    t    = log['t']
    x_sp = log['x_sp']; y_sp = log['y_sp']
    x    = log['x'];    y    = log['y']
    vx_r = log['vx_real']; vy_r = log['vy_real']
    vx_d = log['vx_des'];  vy_d = log['vy_des']

    t_arr   = np.array(t)
    x_arr   = np.array(x);    y_arr   = np.array(y)
    xsp_arr = np.array(x_sp); ysp_arr = np.array(y_sp)
    mask    = t_arr <= duration

    # Error cartesiano
    ex_arr = xsp_arr - x_arr
    ey_arr = ysp_arr - y_arr

    ex_c = ex_arr[mask]; ey_c = ey_arr[mask]
    if len(ex_c):
        print(f'Error X RMS (circulo): {np.sqrt(np.mean(ex_c**2)):.4f} m  '
              f'| max |ex|: {np.max(np.abs(ex_c)):.4f} m')
        print(f'Error Y RMS (circulo): {np.sqrt(np.mean(ey_c**2)):.4f} m  '
              f'| max |ey|: {np.max(np.abs(ey_c)):.4f} m')

    # Fig 1: Velocidades + Trayectoria XY
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle('MavlinkCirculoLog - vuelo circular', fontsize=13)

    for ax_i in range(2):
        axes[ax_i].axvline(duration, color='gray', linestyle=':', linewidth=1,
                           label='Fin circulo')

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
    cx_plot = x0 + RADIUS
    ax.plot(cx_plot + RADIUS * np.cos(theta_th),
            y0 + RADIUS * np.sin(theta_th),
            'g:', linewidth=1.2, label='Circulo teorico')
    ax.plot(x_sp, y_sp, 'b--', linewidth=1.2, label='Setpoints', alpha=0.7)
    ax.plot(x_arr[mask],  y_arr[mask],  'r-', linewidth=2,   label='Real (circulo)')
    ax.plot(x_arr[~mask], y_arr[~mask], color='orange', linewidth=1.5,
            linestyle='--', label='Real (cola)')
    ax.scatter(x0, y0, c='k', marker='o', zorder=5, label='Inicio')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectoria XY'); ax.axis('equal')
    ax.legend(); ax.grid(True, alpha=0.4)

    fig1.tight_layout()

    # Fig 2: Errores cartesianos X e Y
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig2.suptitle('Lider - Posicion deseada vs real y errores cartesianos',
                  fontsize=13)

    for ax in axes2:
        ax.axvline(duration, color='gray', linestyle=':', linewidth=0.9)
        ax.grid(True, alpha=0.4)

    ax = axes2[0]
    ax.plot(t_arr, xsp_arr, 'b-',  linewidth=1.5, label='x_sp (deseado)')
    ax.plot(t_arr, x_arr,   'r--', linewidth=1.2, label='x (real)', alpha=0.85)
    ax.set_ylabel('X [m]'); ax.set_title('Posicion X')
    ax.legend(loc='upper right')

    ax = axes2[1]
    ax.plot(t_arr, ysp_arr, 'b-',  linewidth=1.5, label='y_sp (deseado)')
    ax.plot(t_arr, y_arr,   'r--', linewidth=1.2, label='y (real)', alpha=0.85)
    ax.set_ylabel('Y [m]'); ax.set_title('Posicion Y')
    ax.legend(loc='upper right')

    ax = axes2[2]
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.plot(t_arr, ex_arr, color='steelblue', linewidth=1.5,
            label='ex = x_sp - x  (+ = atrasado en X)')
    ax.plot(t_arr, ey_arr, color='crimson',   linewidth=1.5,
            label='ey = y_sp - y  (+ = atrasado en Y)', alpha=0.85)
    ax.set_ylabel('Error [m]')
    ax.set_xlabel('Tiempo [s]')
    ax.set_title('Errores cartesianos (sin cambio de base en la transicion)')
    ax.legend(loc='upper right')

    fig2.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f'Conectado: SYS={master.target_system} COMP={master.target_component}')

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
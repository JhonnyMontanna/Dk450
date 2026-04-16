#!/usr/bin/env python3
"""
Vuelo circular por comandos de posición — versión optimizada
Basado en circle_flight_v2.py, misma arquitectura:
  - Comandos SET_POSITION_TARGET_LOCAL_NED con posición + yaw (sin velocidades)
  - Hilo MAVLink dedicado, sin bloquear el loop de control
  - Loop con tiempo absoluto (next_t += dt), sin deriva acumulada
  - Fase de cola: sigue registrando hasta convergencia física real
  - Gráfica garantizada DESPUÉS de que el dron se detiene
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
CONN          = 'udp:127.0.0.1:14552'   # SITL — cambiar a IP Jetson en real
SYSID         = 1
COMPID        = 0
RADIUS        = 3.0     # metros
ANGULAR_SPEED = 0.5     # rad/s
RATE          = 10      # Hz

# Criterio de convergencia (fase de cola)
CONV_RADIUS   = 0.15    # m   — distancia al punto final para "llegó"
CONV_SPEED    = 0.10    # m/s — velocidad máxima para "parado"
CONV_HOLD     = 1.0     # s   — tiempo consecutivo dentro del criterio
CONV_TIMEOUT  = 15.0    # s   — timeout de seguridad

# Máscara: usar posición + yaw, ignorar velocidades / aceleraciones / yaw_rate
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
# ESTADO COMPARTIDO (thread-safe)
# ===============================
_state_lock = threading.Lock()
_latest_pos = None   # último LOCAL_POSITION_NED


def _mavlink_reader(master, stop_event):
    """Hilo dedicado a leer MAVLink sin interferir con el loop de control."""
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
    """Envía setpoint de posición (NED) + yaw."""
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z,
        0, 0, 0,    # velocidades ignoradas
        0, 0, 0,    # aceleraciones ignoradas
        yaw,        # rad
        0           # yaw_rate ignorado
    )


def wait_position_ready():
    """Bloquea hasta recibir el primer LOCAL_POSITION_NED."""
    print("⏳ Esperando posición inicial...", end='', flush=True)
    while get_latest_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    return get_latest_pos()


# ===============================
# LOOP DE CONTROL
# ===============================
def fly_circle(master, duration):
    """
    Ejecuta el círculo enviando setpoints de posición a RATE Hz.
    Usa tiempo absoluto (next_t += dt) para evitar deriva.
    Después del loop entra en fase de cola hasta convergencia física.
    Retorna el log completo y la posición inicial.
    """
    pos0 = get_latest_pos()
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(f"📍 Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")

    # Centro del círculo desplazado para que (x0, y0) esté sobre él
    cx     = x0 + RADIUS
    cy     = y0
    theta0 = math.pi          # ángulo inicial: (x0,y0) está a la izquierda del centro
    dt     = 1.0 / RATE
    steps  = int(duration / dt)

    log = {k: deque() for k in ('t', 'x_sp', 'y_sp', 'x', 'y',
                                 'vx_real', 'vy_real', 'vx_des', 'vy_des')}

    print(f"🌀 Iniciando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, "
          f"duración≈{duration:.1f} s ({steps} pasos @ {RATE} Hz)")

    next_t = time.monotonic()

    for i in range(steps):
        t_sched = i * dt
        theta   = theta0 + ANGULAR_SPEED * t_sched

        # Setpoint de posición sobre el círculo
        x_sp = cx + RADIUS * math.cos(theta)
        y_sp = cy + RADIUS * math.sin(theta)
        yaw  = theta + math.pi / 2    # apuntar en dirección tangencial

        send_position_yaw(master, x_sp, y_sp, z0, yaw)

        # Velocidad tangencial deseada (derivada analítica)
        vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta)
        vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta)

        pos = get_latest_pos()
        if pos is not None:
            log['t'].append(t_sched)
            log['x_sp'].append(x_sp)
            log['y_sp'].append(y_sp)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['vx_des'].append(vx_des)
            log['vy_des'].append(vy_des)

        # Dormir exactamente hasta la siguiente iteración (sin deriva)
        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola ──────────────────────────────────────────────────────────
    # El loop terminó pero el dron aún se mueve hacia el último punto.
    # Seguimos enviando el setpoint final y registrando hasta convergencia.
    x_final, y_final = x_sp, y_sp
    t_offset     = steps * dt
    t_tail_start = time.monotonic()
    t_in_zone    = None

    print(f"⏳ Esperando convergencia (radio={CONV_RADIUS} m, "
          f"vel<{CONV_SPEED} m/s, hold={CONV_HOLD} s)...")

    while True:
        now = time.monotonic()

        if now - t_tail_start > CONV_TIMEOUT:
            print("⚠️  Timeout de convergencia — graficando con datos disponibles.")
            break

        pos = get_latest_pos()
        if pos is not None:
            dist  = math.hypot(pos.x - x_final, pos.y - y_final)
            speed = math.hypot(pos.vx, pos.vy)

            send_position_yaw(master, x_final, y_final, z0, yaw)

            t_tail = t_offset + (now - t_tail_start)
            log['t'].append(t_tail)
            log['x_sp'].append(x_final)
            log['y_sp'].append(y_final)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['vx_des'].append(0.0)
            log['vy_des'].append(0.0)

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f"✅ Convergencia: dist={dist:.3f} m, vel={speed:.3f} m/s")
                    break
            else:
                t_in_zone = None

        time.sleep(dt)

    print("⏹️  Registro completo.")
    return {k: list(v) for k, v in log.items()}, (x0, y0)


# ===============================
# GRÁFICAS
# ===============================
def plot_results(log, x0, y0, duration):
    t    = log['t'];    x_sp = log['x_sp'];  y_sp = log['y_sp']
    x    = log['x'];    y    = log['y']
    vx_r = log['vx_real']; vy_r = log['vy_real']
    vx_d = log['vx_des'];  vy_d = log['vy_des']

    t_arr = np.array(t)
    mask  = t_arr <= duration

    err_all  = np.sqrt((np.array(x) - np.array(x_sp))**2 +
                       (np.array(y) - np.array(y_sp))**2)
    err_circ = err_all[mask]
    if len(err_circ):
        print(f"📊 Error posición RMS (círculo): {np.mean(err_circ):.4f} m  "
              f"|  máx: {np.max(err_circ):.4f} m")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Vuelo circular (posición) — análisis de seguimiento', fontsize=13)

    for ax_i in range(2):
        axes[ax_i].axvline(duration, color='gray', linestyle=':', linewidth=1,
                           label='Fin círculo')

    # Velocidad X
    ax = axes[0]
    ax.plot(t, vx_d, 'b-',  linewidth=1.5, label='Vx deseada')
    ax.plot(t, vx_r, 'r--', linewidth=1.2, label='Vx real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vx [m/s]')
    ax.set_title('Velocidad en X'); ax.legend(); ax.grid(True, alpha=0.4)

    # Velocidad Y
    ax = axes[1]
    ax.plot(t, vy_d, 'b-',  linewidth=1.5, label='Vy deseada')
    ax.plot(t, vy_r, 'r--', linewidth=1.2, label='Vy real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vy [m/s]')
    ax.set_title('Velocidad en Y'); ax.legend(); ax.grid(True, alpha=0.4)

    # Trayectoria XY
    ax = axes[2]
    theta_th = np.linspace(0, 2 * math.pi, 400)
    ax.plot((x0 + RADIUS) + RADIUS * np.cos(theta_th),
             y0            + RADIUS * np.sin(theta_th),
             'g:', linewidth=1.2, label='Círculo teórico')
    ax.plot(x_sp, y_sp, 'b--', linewidth=1.0, label='Setpoints', alpha=0.6)

    x_arr = np.array(x); y_arr = np.array(y)
    ax.plot(x_arr[mask],  y_arr[mask],  'r-',  linewidth=2,   label='Real (círculo)')
    ax.plot(x_arr[~mask], y_arr[~mask], color='orange', linewidth=1.5,
            linestyle='--', label='Real (cola)')

    ax.scatter(x0, y0, c='k', marker='o', zorder=5, label='Inicio')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectoria XY'); ax.axis('equal')
    ax.legend(); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")
    print("Asegúrate de que el dron esté en GUIDED, armado y en hover.")
    input("Presiona Enter para comenzar el círculo...")

    # Arrancar hilo lector antes de pedir posición
    stop_reader   = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader,
        args=(master, stop_reader),
        daemon=True,
        name="mavlink-reader"
    )
    reader_thread.start()

    wait_position_ready()

    duration = 2 * math.pi / ANGULAR_SPEED

    log, (x0, y0) = fly_circle(master, duration)

    # Detener hilo lector limpiamente
    stop_reader.set()
    reader_thread.join(timeout=2.0)
    master.close()

    # Graficar — garantizado después de convergencia física
    plot_results(log, x0, y0, duration)
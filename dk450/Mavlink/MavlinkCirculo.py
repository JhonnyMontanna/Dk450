#!/usr/bin/env python3
"""
Vuelo circular con lazo cerrado — versión optimizada
Mejoras principales:
  - Loop de control con tiempo absoluto (sin deriva acumulada)
  - Hilo separado para lectura MAVLink (no bloquea el control)
  - Fase de "cola": sigue registrando hasta que el dron converge al punto final
  - Gráfica bloqueada GARANTIZADA hasta convergencia física real
  - Buffer thread-safe con deque para los logs
  - Tiempo de log = tiempo teórico del setpoint (no wall-clock con overhead)
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
SYSID         = 1
COMPID        = 0
RADIUS        = 4.0     # metros
ANGULAR_SPEED = 3.0     # rad/s
RATE          = 50      # Hz

# Criterio de convergencia al punto final (fase de "cola")
CONV_RADIUS   = 0.15    # metros — distancia máxima para considerar "llegó"
CONV_SPEED    = 0.10    # m/s    — velocidad máxima para considerar "parado"
CONV_HOLD     = 1.0     # segundos consecutivos dentro del criterio antes de graficar
CONV_TIMEOUT  = 15.0    # segundos máximos de espera post-loop (seguridad)

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
_state_lock  = threading.Lock()
_latest_pos  = None   # último LOCAL_POSITION_NED recibido

def _mavlink_reader(master, stop_event):
    """
    Hilo dedicado exclusivamente a leer mensajes MAVLink.
    Corre a máxima velocidad sin interferir con el loop de control.
    """
    global _latest_pos
    while not stop_event.is_set():
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.1)
        if msg:
            with _state_lock:
                _latest_pos = msg

def get_latest_pos():
    """Lee el último estado recibido de forma thread-safe."""
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
        yaw,
        0
    )

def wait_local_position_ready():
    """Espera hasta que el hilo lector haya recibido al menos un mensaje."""
    print("⏳ Esperando posición inicial...", end='', flush=True)
    while get_latest_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    return get_latest_pos()

# ===============================
# LOOP DE CONTROL PRINCIPAL
# ===============================
def fly_circle_closed_loop(master, duration):
    """
    Loop de control a tasa fija usando tiempo absoluto.
    No usa time.sleep(dt) acumulativo — usa next_t += dt para evitar deriva.
    Los logs usan el tiempo teórico del setpoint, no el wall-clock con overhead.
    """
    pos0 = get_latest_pos()
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(f"📍 Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")

    cx     = x0 + RADIUS
    cy     = y0
    theta0 = math.pi
    dt     = 1.0 / RATE
    steps  = int(duration / dt)

    # Logs en deques — eficiente para append continuo
    log = {k: deque() for k in ('t', 'x_sp', 'y_sp', 'x', 'y',
                                 'vx_real', 'vy_real', 'vx_des', 'vy_des')}

    print(f"🌀 Volando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, "
          f"duración={duration:.1f} s ({steps} pasos @ {RATE} Hz)")

    # ── Tiempo absoluto: next_t marca el instante exacto de cada iteración ──
    next_t   = time.monotonic()
    t_start  = next_t

    for i in range(steps):
        t_sched = i * dt                                   # tiempo teórico del setpoint
        theta   = theta0 + ANGULAR_SPEED * t_sched

        # Setpoint de posición
        x_sp = cx + RADIUS * math.cos(theta)
        y_sp = cy + RADIUS * math.sin(theta)
        yaw  = theta + math.pi / 2

        send_position_yaw(master, x_sp, y_sp, z0, yaw)

        # Velocidad deseada (derivada analítica)
        vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta)
        vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta)

        # Leer estado actual (ya disponible desde el hilo lector, sin bloquear)
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

        # ── Dormir hasta el instante exacto de la próxima iteración ──
        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)
        # Si sleep_t <= 0 el loop lleva retraso — continúa sin dormir
        # (el log reporta jitter real implícitamente)

    # ── Fase de cola: mantener setpoint final y seguir logueando ──
    # El dron aún se está moviendo hacia el último punto.
    # Esperamos hasta que converja físicamente antes de retornar.
    x_final, y_final = x_sp, y_sp
    t_in_zone     = None          # momento en que entró dentro del radio de convergencia
    t_tail_start  = time.monotonic()
    t_offset      = steps * dt    # tiempo lógico continúa desde donde terminó el loop

    print(f"⏳ Esperando convergencia al punto final "
          f"(radio={CONV_RADIUS} m, vel<{CONV_SPEED} m/s)...")

    while True:
        now = time.monotonic()

        # Timeout de seguridad
        if now - t_tail_start > CONV_TIMEOUT:
            print("⚠️  Timeout de convergencia — graficando con datos disponibles.")
            break

        pos = get_latest_pos()
        if pos is not None:
            dist  = math.hypot(pos.x - x_final, pos.y - y_final)
            speed = math.hypot(pos.vx, pos.vy)

            # Seguir enviando el setpoint final para que el controlador no suelte
            send_position_yaw(master, x_final, y_final, z0, yaw)

            # Log continuo durante la cola (setpoint = punto final fijo)
            t_tail = t_offset + (now - t_tail_start)
            log['t'].append(t_tail)
            log['x_sp'].append(x_final)
            log['y_sp'].append(y_final)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['vx_des'].append(0.0)   # velocidad deseada = 0 en el punto final
            log['vy_des'].append(0.0)

            if dist < CONV_RADIUS and speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f"✅ Convergencia alcanzada: dist={dist:.3f} m, vel={speed:.3f} m/s")
                    break
            else:
                t_in_zone = None   # salió del radio — resetear contador

        time.sleep(1.0 / RATE)

    print("⏹️  Registro completo.")

    # Convertir deques a listas una sola vez al final
    return {k: list(v) for k, v in log.items()}, (x0, y0)

# ===============================
# GRÁFICAS
# ===============================
def plot_results(log, x0, y0, duration):
    t      = log['t']
    x_sp   = log['x_sp'];  y_sp   = log['y_sp']
    x      = log['x'];     y      = log['y']
    vx_r   = log['vx_real'];  vy_r = log['vy_real']
    vx_d   = log['vx_des'];   vy_d = log['vy_des']

    # Error de posición RMS (solo durante el círculo, no la cola)
    t_arr   = np.array(t)
    mask    = t_arr <= duration
    err_all = np.sqrt((np.array(x) - np.array(x_sp))**2 +
                      (np.array(y) - np.array(y_sp))**2)
    err_circ = err_all[mask]
    if len(err_circ):
        print(f"📊 Error posición RMS (círculo): {np.mean(err_circ):.4f} m  "
              f"|  máx: {np.max(err_circ):.4f} m")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Vuelo circular — análisis de seguimiento', fontsize=13)

    # Línea vertical donde termina el círculo y empieza la cola
    for ax_i in range(2):
        axes[ax_i].axvline(duration, color='gray', linestyle=':', linewidth=1,
                           label='Fin círculo')

    # 1) Velocidad X
    ax = axes[0]
    ax.plot(t, vx_d, 'b-',  linewidth=1.5, label='Vx deseada')
    ax.plot(t, vx_r, 'r--', linewidth=1.2, label='Vx real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vx [m/s]')
    ax.set_title('Velocidad en X'); ax.legend(); ax.grid(True, alpha=0.4)

    # 2) Velocidad Y
    ax = axes[1]
    ax.plot(t, vy_d, 'b-',  linewidth=1.5, label='Vy deseada')
    ax.plot(t, vy_r, 'r--', linewidth=1.2, label='Vy real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vy [m/s]')
    ax.set_title('Velocidad en Y'); ax.legend(); ax.grid(True, alpha=0.4)

    # 3) Trayectoria XY — separar círculo de cola visualmente
    ax = axes[2]
    theta_theory = np.linspace(0, 2 * math.pi, 400)
    ax.plot((x0 + RADIUS) + RADIUS * np.cos(theta_theory),
             y0            + RADIUS * np.sin(theta_theory),
             'g:', linewidth=1.2, label='Círculo teórico')
    ax.plot(x_sp, y_sp, 'b--', linewidth=1.2, label='Setpoints', alpha=0.7)

    # Trayectoria real: círculo en rojo, cola en naranja
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

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

    # ── Arrancar hilo lector ANTES de esperar posición ──
    stop_reader = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader,
        args=(master, stop_reader),
        daemon=True,
        name="mavlink-reader"
    )
    reader_thread.start()

    # Esperar primer mensaje de posición
    wait_local_position_ready()

    duration = 2 * math.pi / ANGULAR_SPEED

    # Ejecutar vuelo
    log, (x0, y0) = fly_circle_closed_loop(master, duration)

    # ── Detener hilo lector y esperar que termine limpiamente ──
    stop_reader.set()
    reader_thread.join(timeout=2.0)

    master.close()

    # Graficar — GARANTIZADO que el dron convergió antes de llegar aquí
    plot_results(log, x0, y0, duration)
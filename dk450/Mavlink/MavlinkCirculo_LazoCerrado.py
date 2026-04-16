#!/usr/bin/env python3
"""
Vuelo circular — control en lazo cerrado con path following
===========================================================
El problema del código anterior era open-loop en tiempo:
  theta = theta0 + omega * t
Si el dron se retrasa, el setpoint sigue avanzando y el error crece sin límite.

Solución: path following con proyección
  1. En cada ciclo medir theta_real = atan2(dron - centro)
  2. Setpoint = punto del círculo en (theta_real + lookahead_angle)
  3. Si el dron está fuera/dentro del radio, corregir radialmente

Esto garantiza que el setpoint SIEMPRE esté por delante del dron,
no por delante de donde el dron debería estar según el reloj.
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
RATE          = 20      # Hz

# ── Parámetros del lazo cerrado ──────────────────────────────────────────────
# Lookahead: cuánto adelantar el setpoint respecto a la posición proyectada.
# Valor = tiempo_de_anticipación * omega.
# Demasiado pequeño → dron persigue su propia sombra, oscila.
# Demasiado grande → corta esquinas, el círculo se deforma.
# Recomendado: entre 0.5 y 1.5 ciclos de control.
LOOKAHEAD_TIME   = 1.5          # segundos de anticipación
LOOKAHEAD_ANGLE  = ANGULAR_SPEED * LOOKAHEAD_TIME  # rad

# Corrección radial: si el dron está a (R + err_r) del centro,
# desplazar el setpoint K_RADIAL * err_r metros hacia el círculo.
# Actúa como un resorte que empuja al dron de vuelta al radio correcto.
# Valores entre 0.3 y 0.8 funcionan bien. 0 = sin corrección radial.
K_RADIAL = 0.5

# Criterio de convergencia (fase de cola)
CONV_RADIUS   = 0.15    # m
CONV_SPEED    = 0.10    # m/s
CONV_HOLD     = 1.0     # s
CONV_TIMEOUT  = 15.0    # s

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
# ESTADO COMPARTIDO
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
        yaw,
        0
    )


def wait_position_ready():
    print("⏳ Esperando posición inicial...", end='', flush=True)
    while get_latest_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    return get_latest_pos()


def compute_setpoint(pos_x, pos_y, cx, cy, z0):
    """
    Núcleo del lazo cerrado.

    1. Proyectar la posición real del dron sobre el círculo:
       theta_real = atan2(dron_y - cy, dron_x - cx)

    2. Calcular el error radial (distancia al círculo):
       r_real    = distancia del dron al centro
       err_r     = r_real - RADIUS  (>0 si está fuera, <0 si está dentro)

    3. Setpoint tangencial = punto del círculo en (theta_real + LOOKAHEAD_ANGLE)

    4. Corrección radial: desplazar el setpoint hacia el radio correcto
       proporcional al error:
       x_sp += -K_RADIAL * err_r * cos(theta_real)
       y_sp += -K_RADIAL * err_r * sin(theta_real)
       (el signo negativo empuja hacia el interior si err_r > 0)

    Retorna: (x_sp, y_sp, yaw, theta_real, err_r)
    """
    dx = pos_x - cx
    dy = pos_y - cy
    r_real    = math.hypot(dx, dy)
    theta_real = math.atan2(dy, dx)
    err_r     = r_real - RADIUS

    # Ángulo del setpoint tangencial (adelantado)
    theta_sp = theta_real + LOOKAHEAD_ANGLE

    # Punto sobre el círculo ideal
    x_sp = cx + RADIUS * math.cos(theta_sp)
    y_sp = cy + RADIUS * math.sin(theta_sp)

    # Corrección radial: empuja al setpoint hacia la dirección del centro
    # cuando el dron está fuera del radio, y al contrario si está adentro
    radial_correction = K_RADIAL * err_r
    x_sp -= radial_correction * math.cos(theta_real)
    y_sp -= radial_correction * math.sin(theta_real)

    # Yaw tangencial al círculo en el punto del setpoint
    yaw = theta_sp + math.pi / 2

    return x_sp, y_sp, yaw, theta_real, err_r


# ===============================
# LOOP DE CONTROL EN LAZO CERRADO
# ===============================
def fly_circle_closed_loop(master, cx, cy, z0, duration):
    """
    Ejecuta el círculo con path following durante 'duration' segundos.
    El dron debe comenzar aproximadamente sobre el círculo.
    """
    dt    = 1.0 / RATE
    steps = int(duration / dt)

    log = {k: deque() for k in ('t', 'x_sp', 'y_sp', 'x', 'y',
                                 'vx_real', 'vy_real', 'vx_des', 'vy_des',
                                 'theta_real', 'err_r')}

    print(f"🌀 Path following: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, "
          f"lookahead={math.degrees(LOOKAHEAD_ANGLE):.1f}°, "
          f"duración≈{duration:.1f} s ({steps} pasos @ {RATE} Hz)")

    next_t = time.monotonic()
    x_sp = y_sp = yaw = 0.0   # inicializar para fase de cola

    for i in range(steps):
        t_sched = i * dt

        pos = get_latest_pos()
        if pos is not None:
            x_sp, y_sp, yaw, theta_real, err_r = compute_setpoint(
                pos.x, pos.y, cx, cy, z0
            )
            send_position_yaw(master, x_sp, y_sp, z0, yaw)

            # Velocidad tangencial deseada (para referencia en gráficas)
            vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta_real + LOOKAHEAD_ANGLE)
            vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta_real + LOOKAHEAD_ANGLE)

            log['t'].append(t_sched)
            log['x_sp'].append(x_sp)
            log['y_sp'].append(y_sp)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['vx_des'].append(vx_des)
            log['vy_des'].append(vy_des)
            log['theta_real'].append(theta_real)
            log['err_r'].append(err_r)

        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola: convergencia al punto final ────────────────────────────
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
            log['theta_real'].append(math.atan2(pos.y - cy, pos.x - cx))
            log['err_r'].append(math.hypot(pos.x - cx, pos.y - cy) - RADIUS)

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
    return {k: list(v) for k, v in log.items()}


# ===============================
# GRÁFICAS
# ===============================
def plot_results(log, x0, y0, cx, cy, duration):
    t      = np.array(log['t'])
    x_sp   = np.array(log['x_sp']);  y_sp = np.array(log['y_sp'])
    x      = np.array(log['x']);     y    = np.array(log['y'])
    vx_r   = np.array(log['vx_real']); vy_r = np.array(log['vy_real'])
    vx_d   = np.array(log['vx_des']);  vy_d = np.array(log['vy_des'])
    err_r  = np.array(log['err_r'])
    mask   = t <= duration

    err_pos = np.sqrt((x - x_sp)**2 + (y - y_sp)**2)
    if mask.any():
        print(f"📊 Error posición RMS (círculo): {np.mean(err_pos[mask]):.4f} m  "
              f"|  máx: {np.max(err_pos[mask]):.4f} m")
        print(f"📊 Error radial  RMS (círculo): {np.mean(np.abs(err_r[mask])):.4f} m  "
              f"|  máx: {np.max(np.abs(err_r[mask])):.4f} m")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Path following lazo cerrado — R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, '
                 f'lookahead={math.degrees(LOOKAHEAD_ANGLE):.1f}°', fontsize=12)

    # 1) Trayectoria XY
    ax = axes[0, 0]
    theta_th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(cx + RADIUS * np.cos(theta_th),
            cy + RADIUS * np.sin(theta_th),
            'g:', linewidth=1.2, label='Círculo teórico')
    ax.plot(x_sp[mask], y_sp[mask], 'b--', linewidth=1.0,
            label='Setpoints (lazo cerrado)', alpha=0.6)
    ax.plot(x[mask],    y[mask],    'r-',  linewidth=2,   label='Real (círculo)')
    ax.plot(x[~mask],   y[~mask],   color='orange', linewidth=1.5,
            linestyle='--', label='Real (cola)')
    ax.scatter(x0, y0, c='k', marker='o', zorder=5, label='Inicio')
    ax.scatter(cx, cy, c='g', marker='+', s=80, zorder=5, label='Centro')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectoria XY'); ax.axis('equal')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # 2) Error radial en el tiempo
    ax = axes[0, 1]
    ax.plot(t[mask], err_r[mask], 'purple', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline( CONV_RADIUS, color='orange', linewidth=0.8, linestyle=':')
    ax.axhline(-CONV_RADIUS, color='orange', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Error radial [m]')
    ax.set_title('Error radial (r_real − R)')
    ax.grid(True, alpha=0.4)

    # 3) Velocidad X
    ax = axes[1, 0]
    ax.axvline(duration, color='gray', linestyle=':', linewidth=1)
    ax.plot(t, vx_d, 'b-',  linewidth=1.5, label='Vx deseada')
    ax.plot(t, vx_r, 'r--', linewidth=1.2, label='Vx real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vx [m/s]')
    ax.set_title('Velocidad en X'); ax.legend(); ax.grid(True, alpha=0.4)

    # 4) Velocidad Y
    ax = axes[1, 1]
    ax.axvline(duration, color='gray', linestyle=':', linewidth=1)
    ax.plot(t, vy_d, 'b-',  linewidth=1.5, label='Vy deseada')
    ax.plot(t, vy_r, 'r--', linewidth=1.2, label='Vy real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vy [m/s]')
    ax.set_title('Velocidad en Y'); ax.legend(); ax.grid(True, alpha=0.4)

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

    stop_reader   = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader,
        args=(master, stop_reader),
        daemon=True,
        name="mavlink-reader"
    )
    reader_thread.start()

    pos0 = wait_position_ready()
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(f"📍 Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")

    # Centro del círculo: el dron parte desde el borde izquierdo
    cx = x0 + RADIUS
    cy = y0

    print(f"⚙️  Centro del círculo: cx={cx:.2f}, cy={cy:.2f}")
    print(f"⚙️  Lookahead: {math.degrees(LOOKAHEAD_ANGLE):.1f}° | K_radial: {K_RADIAL}")

    duration = 2 * math.pi / ANGULAR_SPEED

    log = fly_circle_closed_loop(master, cx, cy, z0, duration)

    stop_reader.set()
    reader_thread.join(timeout=2.0)
    master.close()

    plot_results(log, x0, y0, cx, cy, duration)
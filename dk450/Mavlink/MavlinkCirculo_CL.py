#!/usr/bin/env python3
"""
Seguimiento de trayectoria circular — control de VELOCIDAD con feedforward + P
===============================================================================
Arquitectura del controlador
─────────────────────────────
El dron recibe comandos de velocidad (vx, vy) en LOCAL_NED.

Inicio en el perímetro (sin fase de acercamiento):
  El centro se calcula automáticamente como cx = x0 - RADIUS, cy = y0.
  Esto coloca al dron exactamente en θ=0 del círculo desde el primer ciclo.
  Error radial inicial = 0 → sin espiral de convergencia.

Componente feedforward (FF):
  La velocidad tangencial ideal para una circunferencia es la derivada
  de la posición sobre el círculo:
    vx_ff = -R · ω · sin(θ_real)
    vy_ff =  R · ω · cos(θ_real)
  Donde θ_real = atan2(dron - centro) — path following, no tiempo.

Componente proporcional radial (KP_R):
  Si el dron se desvía del radio (por viento, latencia, etc.):
    e_r = dist(dron, centro) - RADIUS
    v_radial_corr = -KP_R · e_r · (dirección_unitaria_al_centro)

Resultado:
  v_cmd = v_ff + v_radial_corr   (con clamp a V_MAX)
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
ANGULAR_SPEED = 0.5     # rad/s  ← conservador para exteriores
RATE          = 20      # Hz

# ── Parámetros del controlador ───────────────────────────────────────────────
# KP_R: ganancia radial. Corrige desviaciones del radio.
#   Alto → corrección agresiva, puede oscilar.
#   Bajo → el radio converge lentamente.
#   Rango típico: 0.3 – 1.0
KP_R  = 0.6

# KP_PH: ganancia de fase (tangencial). Corrige adelanto/retraso angular.
#   Alto → responde rápido a atrasos, puede sobrepasar.
#   Bajo → el dron puede ir muy lento al principio.
#   Rango típico: 0.2 – 0.8
KP_PH = 0.4

# V_MAX: velocidad máxima total comandada (límite de seguridad)
V_MAX = RADIUS * ANGULAR_SPEED * 2.5   # 2.5× la velocidad tangencial nominal

# Criterio de convergencia (fase de cola)
CONV_RADIUS  = 0.20   # m
CONV_SPEED   = 0.10   # m/s
CONV_HOLD    = 1.0    # s
CONV_TIMEOUT = 15.0   # s

# Máscara: solo velocidades vx, vy, vz — ignorar posición, aceleración, yaw
TYPE_MASK_VEL = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

# ===============================
# ESTADO COMPARTIDO
# ===============================
_state_lock = threading.Lock()
_latest_pos = None


def _mavlink_reader(master, stop_event):
    """Hilo dedicado — lee LOCAL_POSITION_NED sin bloquear el loop."""
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
def send_velocity(master, vx, vy, vz=0.0):
    """Envía comando de velocidad en LOCAL_NED."""
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )


def wait_position_ready():
    print("⏳ Esperando posición inicial...", end='', flush=True)
    while get_latest_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    return get_latest_pos()


def angle_diff(a, b):
    """Diferencia angular con signo en [-π, π]: a - b."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def compute_velocity_command(pos_x, pos_y, cx, cy):
    """
    Calcula el comando de velocidad (vx, vy) con feedforward + correcciones P.

    Pasos:
      1. Proyectar el dron sobre el círculo → θ_real
      2. θ_ref = θ_real (path following: el setpoint va anclado al dron)
      3. FF tangencial en θ_ref
      4. Corrección radial proporcional al error r - RADIUS
      5. (θ_ref ya coincide con θ_real, la corrección de fase es cero salvo
         que queramos añadir lookahead — no es necesario con FF puro)
      6. Clamp de velocidad total

    Retorna: (vx, vy, θ_real, e_r, vx_ff, vy_ff)
    """
    dx = pos_x - cx
    dy = pos_y - cy
    r       = math.hypot(dx, dy)
    theta   = math.atan2(dy, dx)
    e_r     = r - RADIUS                # error radial (>0 fuera, <0 dentro)

    # 1) Feedforward tangencial en θ (dirección del movimiento circular)
    vx_ff = -RADIUS * ANGULAR_SPEED * math.sin(theta)
    vy_ff =  RADIUS * ANGULAR_SPEED * math.cos(theta)

    # 2) Corrección radial: empujar hacia el círculo
    #    Si e_r > 0 (dron fuera), la corrección es negativa en dirección radial
    #    (hacia el centro). Si e_r < 0 (dentro), la corrección es hacia afuera.
    #    Dirección radial unitaria: (cos θ, sin θ)
    if r > 1e-3:   # evitar división por cero si el dron está exactamente en el centro
        vx_r = -KP_R * e_r * (dx / r)
        vy_r = -KP_R * e_r * (dy / r)
    else:
        vx_r = vy_r = 0.0

    # 3) Suma
    vx = vx_ff + vx_r
    vy = vy_ff + vy_r

    # 4) Clamp de velocidad total (seguridad)
    v_total = math.hypot(vx, vy)
    if v_total > V_MAX:
        vx = vx * V_MAX / v_total
        vy = vy * V_MAX / v_total

    return vx, vy, theta, e_r, vx_ff, vy_ff


# ===============================
# LOOP DE CONTROL
# ===============================
def fly_circle(master, cx, cy, z0, duration):
    dt    = 1.0 / RATE
    steps = int(duration / dt)

    log = {k: deque() for k in (
        't', 'x_sp', 'y_sp', 'x', 'y',
        'vx_cmd', 'vy_cmd', 'vx_ff', 'vy_ff',
        'vx_real', 'vy_real',
        'error_x', 'error_y', 'err_r'
    )}

    lin_speed = RADIUS * ANGULAR_SPEED
    print(f"🌀 Iniciando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, "
          f"v_tangencial={lin_speed:.2f} m/s, duración≈{duration:.1f} s "
          f"({steps} pasos @ {RATE} Hz)")
    print(f"⚙️  KP_R={KP_R}, KP_PH={KP_PH}, V_MAX={V_MAX:.2f} m/s")
    print(f"⚙️  El dron ya está en el perímetro — error radial inicial ≈ 0")

    next_t = time.monotonic()
    vx_cmd = vy_cmd = 0.0   # inicializar para fase de cola

    for i in range(steps):
        t_sched = i * dt

        pos = get_latest_pos()
        if pos is not None:
            vx_cmd, vy_cmd, theta, e_r, vx_ff, vy_ff = compute_velocity_command(
                pos.x, pos.y, cx, cy
            )
            send_velocity(master, vx_cmd, vy_cmd)

            # Setpoint de posición de referencia (solo para logging y gráficas)
            x_sp = cx + RADIUS * math.cos(theta)
            y_sp = cy + RADIUS * math.sin(theta)

            log['t'].append(t_sched)
            log['x_sp'].append(x_sp)
            log['y_sp'].append(y_sp)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_cmd'].append(vx_cmd)
            log['vy_cmd'].append(vy_cmd)
            log['vx_ff'].append(vx_ff)
            log['vy_ff'].append(vy_ff)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['error_x'].append(x_sp - pos.x)
            log['error_y'].append(y_sp - pos.y)
            log['err_r'].append(e_r)

        next_t += dt
        sleep_t = next_t - time.monotonic()
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── Fase de cola: detener y esperar quietud ──────────────────────────────
    t_tail_start = time.monotonic()
    t_offset     = steps * dt
    t_in_zone    = None

    print(f"⏳ Deteniendo y esperando quietud (vel<{CONV_SPEED} m/s, "
          f"hold={CONV_HOLD} s)...")

    while True:
        now = time.monotonic()
        if now - t_tail_start > CONV_TIMEOUT:
            print("⚠️  Timeout — graficando con datos disponibles.")
            break

        send_velocity(master, 0.0, 0.0)   # comando de parada continuo

        pos = get_latest_pos()
        if pos is not None:
            speed = math.hypot(pos.vx, pos.vy)
            t_tail = t_offset + (now - t_tail_start)

            log['t'].append(t_tail)
            log['x_sp'].append(pos.x)   # en cola el setpoint = posición actual
            log['y_sp'].append(pos.y)
            log['x'].append(pos.x)
            log['y'].append(pos.y)
            log['vx_cmd'].append(0.0)
            log['vy_cmd'].append(0.0)
            log['vx_ff'].append(0.0)
            log['vy_ff'].append(0.0)
            log['vx_real'].append(pos.vx)
            log['vy_real'].append(pos.vy)
            log['error_x'].append(0.0)
            log['error_y'].append(0.0)
            log['err_r'].append(math.hypot(pos.x - cx, pos.y - cy) - RADIUS)

            if speed < CONV_SPEED:
                if t_in_zone is None:
                    t_in_zone = now
                elif now - t_in_zone >= CONV_HOLD:
                    print(f"✅ Dron detenido: vel={speed:.3f} m/s")
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
    x_sp   = np.array(log['x_sp']);   y_sp   = np.array(log['y_sp'])
    x      = np.array(log['x']);      y      = np.array(log['y'])
    vx_cmd = np.array(log['vx_cmd']); vy_cmd = np.array(log['vy_cmd'])
    vx_ff  = np.array(log['vx_ff']);  vy_ff  = np.array(log['vy_ff'])
    vx_r   = np.array(log['vx_real']); vy_r  = np.array(log['vy_real'])
    ex     = np.array(log['error_x']); ey    = np.array(log['error_y'])
    err_r  = np.array(log['err_r'])
    mask   = t <= duration

    if mask.any():
        err_pos = np.sqrt(ex**2 + ey**2)
        print(f"📊 Error posición RMS (círculo): {np.mean(err_pos[mask]):.4f} m  "
              f"|  máx: {np.max(err_pos[mask]):.4f} m")
        print(f"📊 Error radial  RMS (círculo): {np.mean(np.abs(err_r[mask])):.4f} m  "
              f"|  máx: {np.max(np.abs(err_r[mask])):.4f} m")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Control velocidad FF+P — R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, '
                 f'KP_R={KP_R}, KP_PH={KP_PH}', fontsize=11)

    # 1) Trayectoria XY
    ax = axes[0, 0]
    theta_th = np.linspace(0, 2 * math.pi, 400)
    ax.plot(cx + RADIUS * np.cos(theta_th),
            cy + RADIUS * np.sin(theta_th),
            'g:', linewidth=1.2, label='Círculo teórico')
    ax.plot(x[mask],  y[mask],  'r-',  linewidth=2,   label='Real (círculo)')
    ax.plot(x[~mask], y[~mask], color='orange', linewidth=1.5,
            linestyle='--', label='Real (cola)')
    ax.scatter(x0, y0, c='k', marker='o', zorder=5, label='Inicio')
    ax.scatter(cx, cy, c='g', marker='+', s=100, zorder=5, label='Centro')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
    ax.set_title('Trayectoria XY'); ax.axis('equal')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # 2) Error radial
    ax = axes[0, 1]
    ax.plot(t[mask], err_r[mask], 'purple', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axhline( CONV_RADIUS, color='orange', linewidth=0.8, linestyle=':')
    ax.axhline(-CONV_RADIUS, color='orange', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Error radial [m]')
    ax.set_title('Error radial (r_real − R)'); ax.grid(True, alpha=0.4)

    # 3) Velocidades X: FF vs cmd vs real
    ax = axes[1, 0]
    ax.axvline(duration, color='gray', linestyle=':', linewidth=1)
    ax.plot(t, vx_ff,  'g-',  linewidth=1.2, label='Vx feedforward', alpha=0.7)
    ax.plot(t, vx_cmd, 'b-',  linewidth=1.5, label='Vx cmd (FF+P)')
    ax.plot(t, vx_r,   'r--', linewidth=1.2, label='Vx real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vx [m/s]')
    ax.set_title('Velocidad en X'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # 4) Velocidades Y
    ax = axes[1, 1]
    ax.axvline(duration, color='gray', linestyle=':', linewidth=1)
    ax.plot(t, vy_ff,  'g-',  linewidth=1.2, label='Vy feedforward', alpha=0.7)
    ax.plot(t, vy_cmd, 'b-',  linewidth=1.5, label='Vy cmd (FF+P)')
    ax.plot(t, vy_r,   'r--', linewidth=1.2, label='Vy real', alpha=0.85)
    ax.set_xlabel('Tiempo [s]'); ax.set_ylabel('Vy [m/s]')
    ax.set_title('Velocidad en Y'); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    print(f"🔌 Conectando a {CONN}...")
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"✅ Heartbeat. SYS={master.target_system}, COMP={master.target_component}")
    print("Asegúrate de que el dron esté en GUIDED, armado y en hover.")
    input("Presiona Enter para comenzar el círculo...")

    # Arrancar hilo lector
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

    # El dron ya está sobre el perímetro desde el primer ciclo:
    # el centro se desplaza RADIUS metros en -X respecto a la posición inicial.
    # θ_inicial = 0  →  punto del perímetro = (cx + RADIUS, cy) = (x0, y0)  ✓
    cx = x0 - RADIUS
    cy = y0
    print(f"⚙️  Centro del círculo: cx={cx:.2f}, cy={cy:.2f}")
    print(f"⚙️  Dron en perímetro: θ_inicial=0°, "
          f"dist al centro={math.hypot(x0-cx, y0-cy):.3f} m")

    duration = 2 * math.pi / ANGULAR_SPEED

    log = fly_circle(master, cx, cy, z0, duration)

    stop_reader.set()
    reader_thread.join(timeout=2.0)
    master.close()
    print("🔌 Conexión cerrada.")

    plot_results(log, x0, y0, cx, cy, duration)
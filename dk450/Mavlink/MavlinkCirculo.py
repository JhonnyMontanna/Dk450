#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
CONN = 'udp:127.0.0.1:14552'   # SITL
SYSID = 1
COMPID = 0

RADIUS = 4.0
ANGULAR_SPEED = 0.4
LINEAR_SPEED = RADIUS * ANGULAR_SPEED
RATE = 50                      # Hz de envío

# Máscara para usar SOLO vx, vy, vz (ignorar todo lo demás)
TYPE_MASK = 0b0000111111000111

# ===============================
# FUNCIONES AUXILIARES
# ===============================
def send_velocity(master, vx, vy, vz=0.0):
    """Envía comando de velocidad en el frame LOCAL_NED"""
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK,
        0, 0, 0,                # posiciones ignoradas
        vx, vy, vz,             # velocidades
        0, 0, 0,                # aceleraciones ignoradas
        0, 0                    # yaw, yaw_rate ignorados
    )

def wait_local_position(master):
    """Espera el primer mensaje LOCAL_POSITION_NED y lo retorna"""
    while True:
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if msg:
            return msg

def circle_motion_with_logging(master, duration, x0, y0):
    """
    Ejecuta el movimiento circular durante 'duration' segundos,
    registrando tiempo, setpoints de velocidad y valores reales.
    """
    dt = 1.0 / RATE
    steps = int(duration * RATE)

    # Listas para logging
    t_log = []
    vx_sp_log, vy_sp_log = [], []
    x_log, y_log = [], []
    vx_real_log, vy_real_log = [], []

    print(f"🌀 Ejecutando círculo durante {duration:.1f} s, registrando datos...")
    t0 = time.time()

    for i in range(steps):
        t = time.time() - t0
        theta = ANGULAR_SPEED * t

        # Setpoints de velocidad (para círculo centrado en (x0,y0) con radio RADIUS)
        vx_sp = LINEAR_SPEED * math.cos(theta + math.pi/2)
        vy_sp = LINEAR_SPEED * math.sin(theta + math.pi/2)

        # Enviar comando
        send_velocity(master, vx_sp, vy_sp)

        # Leer estado real (sin bloquear)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            x_log.append(msg.x)
            y_log.append(msg.y)
            vx_real_log.append(msg.vx)
            vy_real_log.append(msg.vy)
            # Guardamos el setpoint asociado a este instante (usamos el mismo tiempo)
            t_log.append(t)
            vx_sp_log.append(vx_sp)
            vy_sp_log.append(vy_sp)

        # Mantener la frecuencia
        time.sleep(dt)

    # Detener al finalizar
    send_velocity(master, 0.0, 0.0)
    print("⏹️ Movimiento finalizado, dron detenido.")

    return t_log, vx_sp_log, vy_sp_log, x_log, y_log, vx_real_log, vy_real_log

def plot_results(t, vx_sp, vy_sp, x, y, vx_real, vy_real, x0, y0, R, omega):
    """
    Genera las gráficas:
      - Velocidades en X e Y (setpoint vs real)
      - Trayectoria en XY (real + círculo teórico)
    """
    # Cálculo del círculo teórico para comparación en XY
    t_theory = np.linspace(0, max(t), 300)
    x_theory = x0 + R * np.cos(omega * t_theory)
    y_theory = y0 + R * np.sin(omega * t_theory)

    # 1) Velocidad en X
    plt.figure(figsize=(10, 4))
    plt.plot(t, vx_sp, 'b-', label='Vx setpoint')
    plt.plot(t, vx_real, 'r--', label='Vx real')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Vx [m/s]')
    plt.legend()
    plt.grid(True)
    plt.title('Velocidad en X')

    # 2) Velocidad en Y
    plt.figure(figsize=(10, 4))
    plt.plot(t, vy_sp, 'b-', label='Vy setpoint')
    plt.plot(t, vy_real, 'r--', label='Vy real')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Vy [m/s]')
    plt.legend()
    plt.grid(True)
    plt.title('Velocidad en Y')

    # 3) Trayectoria en el plano XY
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'r-', linewidth=2, label='Trayectoria real')
    plt.plot(x_theory, y_theory, 'b--', linewidth=1.5, label='Círculo teórico')
    plt.scatter(x0, y0, c='k', marker='o', label='Inicio')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.title('Trayectoria en el plano XY')

    plt.show()

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    # 1) Conexión
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

    # 2) Obtener posición inicial (suponemos que el dron ya está en hover)
    pos0 = wait_local_position(master)
    x0, y0 = pos0.x, pos0.y
    print(f"📍 Posición inicial: x={x0:.2f}, y={y0:.2f}")

    # 3) Duración del círculo completo (2π rad)
    circle_duration = 2 * math.pi / ANGULAR_SPEED
    print(f"🌀 Radio = {RADIUS} m, ω = {ANGULAR_SPEED} rad/s, duración ≈ {circle_duration:.1f} s")

    # 4) Ejecutar movimiento y registrar
    t, vx_sp, vy_sp, x, y, vx_real, vy_real = circle_motion_with_logging(
        master, circle_duration, x0, y0
    )

    # 5) Cerrar conexión
    master.close()

    # 6) Graficar
    plot_results(t, vx_sp, vy_sp, x, y, vx_real, vy_real, x0, y0, RADIUS, ANGULAR_SPEED)
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

RADIUS = 4.0                    # metros
ANGULAR_SPEED = 1.0              # rad/s (velocidad angular)
RATE = 50                        # Hz de envío de setpoints

# Máscara para usar posición + yaw (ignorar velocidades, aceleraciones, yaw_rate)
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
# FUNCIONES AUXILIARES
# ===============================
def send_position_yaw(master, x, y, z, yaw):
    """Envía setpoint de posición (x,y,z) y yaw (ángulo de guiñada)"""
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z,
        0, 0, 0,          # velocidades ignoradas
        0, 0, 0,          # aceleraciones ignoradas
        yaw,              # yaw (rad)
        0                  # yaw_rate ignorado
    )

def wait_local_position(master):
    """Espera el primer mensaje LOCAL_POSITION_NED y lo retorna"""
    while True:
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if msg:
            return msg

def fly_circle_closed_loop(master, duration):
    """
    Ejecuta un círculo durante 'duration' segundos enviando setpoints de posición.
    Registra tiempo, setpoints, posición real y velocidad real.
    """
    # Obtener posición inicial (hover)
    pos0 = wait_local_position(master)
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(f"📍 Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0:.2f}")

    # Centro del círculo: desplazado para que el punto inicial esté sobre el círculo
    cx = x0 + RADIUS
    cy = y0
    theta0 = math.pi   # Ángulo inicial: apunta a la izquierda, justo en (x0,y0)

    # Logs
    t_log = []
    x_sp_log, y_sp_log = [], []
    x_log, y_log = [], []
    vx_real_log, vy_real_log = [], []
    vx_des_log, vy_des_log = [], []

    dt = 1.0 / RATE
    steps = int(duration / dt)

    print(f"🌀 Volando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, duración={duration:.1f} s")
    t_start = time.time()

    for i in range(steps):
        t = time.time() - t_start
        theta = theta0 + ANGULAR_SPEED * t

        # Setpoint de posición
        x_sp = cx + RADIUS * math.cos(theta)
        y_sp = cy + RADIUS * math.sin(theta)
        yaw = theta + math.pi/2   # Apuntar en dirección tangencial (opcional)

        send_position_yaw(master, x_sp, y_sp, z0, yaw)

        # Velocidad deseada (derivada de la posición)
        vx_des = -RADIUS * ANGULAR_SPEED * math.sin(theta)
        vy_des =  RADIUS * ANGULAR_SPEED * math.cos(theta)

        # Leer estado real (sin bloquear para mantener frecuencia)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            t_log.append(t)
            x_sp_log.append(x_sp)
            y_sp_log.append(y_sp)
            x_log.append(msg.x)
            y_log.append(msg.y)
            vx_real_log.append(msg.vx)
            vy_real_log.append(msg.vy)
            vx_des_log.append(vx_des)
            vy_des_log.append(vy_des)

        time.sleep(dt)

    # Al terminar, enviar un último setpoint para mantener la posición final (opcional)
    send_position_yaw(master, x_sp, y_sp, z0, yaw)
    print("⏹️ Vuelo terminado, dron en posición final.")

    return (t_log, x_sp_log, y_sp_log, x_log, y_log,
            vx_real_log, vy_real_log, vx_des_log, vy_des_log, x0, y0)

def plot_results(t, x_sp, y_sp, x, y, vx_real, vy_real, vx_des, vy_des, x0, y0):
    """
    Genera las gráficas solicitadas:
      - Velocidades X e Y (setpoint vs real)
      - Trayectoria XY (real + círculo teórico)
    """
    # 1) Velocidad en X
    plt.figure(figsize=(10, 4))
    plt.plot(t, vx_des, 'b-', linewidth=1.5, label='Vx deseada')
    plt.plot(t, vx_real, 'r--', linewidth=1.5, label='Vx real')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Vx [m/s]')
    plt.legend()
    plt.grid(True)
    plt.title('Velocidad en X')

    # 2) Velocidad en Y
    plt.figure(figsize=(10, 4))
    plt.plot(t, vy_des, 'b-', linewidth=1.5, label='Vy deseada')
    plt.plot(t, vy_real, 'r--', linewidth=1.5, label='Vy real')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Vy [m/s]')
    plt.legend()
    plt.grid(True)
    plt.title('Velocidad en Y')

    # 3) Trayectoria en el plano XY
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'r-', linewidth=2, label='Trayectoria real')
    plt.plot(x_sp, y_sp, 'b--', linewidth=1.5, label='Setpoints de posición')
    # Círculo teórico continuo (para referencia)
    theta_theory = np.linspace(0, 2*math.pi, 300)
    x_theory = (x0 + RADIUS) + RADIUS * np.cos(theta_theory)
    y_theory = y0 + RADIUS * np.sin(theta_theory)
    plt.plot(x_theory, y_theory, 'g:', linewidth=1, label='Círculo teórico')
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
    # Conexión
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

    # Duración del círculo completo (2π rad)
    duration = 2 * math.pi / ANGULAR_SPEED

    # Ejecutar vuelo circular con lazo cerrado
    (t, x_sp, y_sp, x, y, vx_real, vy_real, vx_des, vy_des, x0, y0) = fly_circle_closed_loop(master, duration)

    # Cerrar conexión
    master.close()

    # Graficar resultados
    plot_results(t, x_sp, y_sp, x, y, vx_real, vy_real, vx_des, vy_des, x0, y0)
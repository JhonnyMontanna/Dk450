#!/usr/bin/env python3
"""
Seguimiento de trayectoria circular en lazo cerrado usando control proporcional de velocidad.
Requiere: pymavlink, matplotlib, numpy.
"""

import time
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
CONN = 'udp:127.0.0.1:14552'   # Conexión al SITL (ajustar según corresponda)
SYSID = 1                      # Normalmente 1 para el autopiloto
COMPID = 0

# Parámetros del círculo
RADIUS = 1.5                    # metros
ANGULAR_SPEED = 0.4             # rad/s (velocidad angular del setpoint)
CENTER_OFFSET = (0.0, 0.0)      # centro del círculo relativo a la posición inicial? 
                                # Aquí asumimos que el círculo se centra en la posición inicial.

# Parámetros del controlador
KP = 1.0                        # Ganancia proporcional (m/s por metro de error)

# Frecuencia de control (Hz)
RATE = 20                       # Envío de comandos a 20 Hz
DT = 1.0 / RATE

# Duración total del vuelo (tiempo para dar una vuelta completa)
DURATION = 2 * math.pi / ANGULAR_SPEED   # Tiempo para 2π radianes

# Máscara para comandos de velocidad (ignorar posiciones, aceleraciones, yaw)
TYPE_MASK_VEL_ONLY = (
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
# FUNCIONES AUXILIARES
# ===============================
def wait_heartbeat(master):
    """Espera un heartbeat y muestra información de conexión"""
    master.wait_heartbeat()
    print(f"✅ Heartbeat recibido. Sistema {master.target_system}, componente {master.target_component}")

def get_local_position(master):
    """Espera y retorna el último mensaje LOCAL_POSITION_NED (bloqueante)"""
    msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
    return msg

def send_velocity(master, vx, vy, vz=0.0):
    """Envía un comando de velocidad en el frame LOCAL_NED"""
    master.mav.set_position_target_local_ned_send(
        0,                      # tiempo (ignorado)
        SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_ONLY,
        0, 0, 0,                # posiciones (ignoradas)
        vx, vy, vz,             # velocidades (m/s)
        0, 0, 0,                # aceleraciones (ignoradas)
        0, 0                     # yaw, yaw_rate (ignorados)
    )

def arm_and_takeoff(master, target_altitude):
    """
    Arma y despega a una altitud objetivo.
    Nota: El dron debe estar en modo GUIDED.
    """
    # Armar
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    print("🔄 Armando...")
    time.sleep(2)

    # Despegue (takeoff)
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_altitude
    )
    print(f"🛫 Despegando a {target_altitude} m...")

    # Esperar a alcanzar altitud (monitorear posición relativa)
    while True:
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if msg and abs(msg.z + target_altitude) < 0.5:  # NED: z negativo hacia arriba
            print("✅ Altitud alcanzada.")
            break
        time.sleep(0.1)

# ===============================
# BUCLE PRINCIPAL DE CONTROL
# ===============================
def closed_loop_circle(master, center_x, center_y, z, radius, omega, duration, kp, dt):
    """
    Ejecuta el seguimiento de círculo en lazo cerrado durante 'duration' segundos.
    Retorna los logs de tiempo, setpoints y reales.
    """
    t0 = time.time()
    steps = int(duration / dt)

    # Logs
    t_log = []
    x_sp_log, y_sp_log = [], []
    x_real_log, y_real_log = [], []
    vx_cmd_log, vy_cmd_log = [], []
    error_x_log, error_y_log = [], []

    print("🌀 Iniciando seguimiento de círculo en lazo cerrado...")

    for i in range(steps):
        t = time.time() - t0

        # 1. Calcular setpoint de posición circular
        theta = omega * t
        x_sp = center_x + radius * math.cos(theta)
        y_sp = center_y + radius * math.sin(theta)

        # 2. Leer posición real actual (último mensaje disponible)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            x_real = msg.x
            y_real = msg.y
        else:
            # Si no hay mensaje nuevo, usar el último conocido (inicialmente 0)
            # En la primera iteración podría no haber ninguno, entonces esperamos uno.
            if i == 0:
                msg = get_local_position(master)
                x_real, y_real = msg.x, msg.y
            else:
                # Mantener el último valor conocido (puede causar error si se pierden muchos mensajes)
                # Para simplificar, usamos el último guardado.
                x_real = x_real_log[-1] if x_real_log else 0
                y_real = y_real_log[-1] if y_real_log else 0

        # 3. Calcular error
        ex = x_sp - x_real
        ey = y_sp - y_real

        # 4. Comando de velocidad proporcional
        vx_cmd = kp * ex
        vy_cmd = kp * ey

        # 5. Enviar comando
        send_velocity(master, vx_cmd, vy_cmd)

        # 6. Guardar en logs
        t_log.append(t)
        x_sp_log.append(x_sp)
        y_sp_log.append(y_sp)
        x_real_log.append(x_real)
        y_real_log.append(y_real)
        vx_cmd_log.append(vx_cmd)
        vy_cmd_log.append(vy_cmd)
        error_x_log.append(ex)
        error_y_log.append(ey)

        # 7. Esperar para mantener la frecuencia
        time.sleep(dt)

    # Detener el dron al finalizar
    send_velocity(master, 0.0, 0.0)
    print("⏹️ Completado. Dron detenido.")

    return {
        't': t_log,
        'x_sp': x_sp_log, 'y_sp': y_sp_log,
        'x_real': x_real_log, 'y_real': y_real_log,
        'vx_cmd': vx_cmd_log, 'vy_cmd': vy_cmd_log,
        'error_x': error_x_log, 'error_y': error_y_log
    }

# ===============================
# FUNCIÓN DE GRAFICADO
# ===============================
def plot_results(data, radius, omega, center_x, center_y):
    """Genera gráficas de trayectoria, errores y velocidades de comando."""
    t = data['t']
    x_sp, y_sp = data['x_sp'], data['y_sp']
    x_real, y_real = data['x_real'], data['y_real']
    vx_cmd, vy_cmd = data['vx_cmd'], data['vy_cmd']
    ex, ey = data['error_x'], data['error_y']

    # Círculo teórico completo para referencia
    t_theory = np.linspace(0, max(t), 300)
    x_theory = center_x + radius * np.cos(omega * t_theory)
    y_theory = center_y + radius * np.sin(omega * t_theory)

    # Figura 1: Trayectoria en XY
    plt.figure(figsize=(8, 8))
    plt.plot(x_sp, y_sp, 'g--', linewidth=1, label='Setpoint (trayectoria deseada)')
    plt.plot(x_real, y_real, 'b-', linewidth=1.5, label='Real (seguimiento)')
    plt.plot(x_theory, y_theory, 'k:', linewidth=1, label='Círculo teórico')
    plt.scatter(x_real[0], y_real[0], c='green', marker='o', label='Inicio')
    plt.scatter(x_real[-1], y_real[-1], c='red', marker='x', label='Final')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.title('Trayectoria en el plano XY (lazo cerrado)')

    # Figura 2: Error de posición vs tiempo
    plt.figure(figsize=(10, 4))
    plt.plot(t, ex, 'r-', label='Error en X')
    plt.plot(t, ey, 'b-', label='Error en Y')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Error [m]')
    plt.legend()
    plt.grid(True)
    plt.title('Error de seguimiento')

    # Figura 3: Comandos de velocidad enviados
    plt.figure(figsize=(10, 4))
    plt.plot(t, vx_cmd, 'r-', label='Vx cmd')
    plt.plot(t, vy_cmd, 'b-', label='Vy cmd')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Velocidad comandada [m/s]')
    plt.legend()
    plt.grid(True)
    plt.title('Comandos de velocidad (salida del controlador)')

    plt.show()

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    # Conexión
    print(f"🔌 Conectando a {CONN}...")
    master = mavutil.mavlink_connection(CONN)
    wait_heartbeat(master)

    # Opcional: asegurarse de que estamos en GUIDED (ya se asume, pero se podría cambiar)
    # Podríamos cambiar el modo a GUIDED si no lo está (requiere más lógica)

    # Obtener posición inicial
    print("📍 Obteniendo posición inicial...")
    pos0 = get_local_position(master)
    x0, y0, z0 = pos0.x, pos0.y, pos0.z
    print(f"Posición inicial: x={x0:.2f}, y={y0:.2f}, z={z0:.2f} (NED: z negativo arriba)")

    # Centro del círculo (en este ejemplo, lo centramos en la posición inicial)
    center_x, center_y = x0, y0

    # Altitud deseada (usamos la inicial, asumiendo que ya está a la altura de vuelo)
    # Si no, podríamos hacer un takeoff a una altitud específica.
    target_z = z0  # Mantener la misma altitud

    # Armar y despegar si es necesario (si no está en el aire)
    # Comprobamos si z es cercano a 0 (en NED, z negativo arriba, 0 es suelo)
    if abs(z0) < 0.5:  # Si está cerca del suelo, despegamos
        arm_and_takeoff(master, 2.0)  # Despegue a 2 metros (positivo en altitud, pero en NED es -2)
        # Después del takeoff, actualizamos la posición inicial para el centro
        pos0 = get_local_position(master)
        center_x, center_y = pos0.x, pos0.y
        target_z = pos0.z
    else:
        print("El dron ya está en el aire, omitiendo takeoff.")

    # Ejecutar control en lazo cerrado
    data = closed_loop_circle(
        master=master,
        center_x=center_x,
        center_y=center_y,
        z=target_z,
        radius=RADIUS,
        omega=ANGULAR_SPEED,
        duration=DURATION,
        kp=KP,
        dt=DT
    )

    # Cerrar conexión
    master.close()
    print("🔌 Conexión cerrada.")

    # Graficar resultados
    plot_results(data, RADIUS, ANGULAR_SPEED, center_x, center_y)
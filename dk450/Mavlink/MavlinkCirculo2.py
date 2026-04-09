#!/usr/bin/env python3
import time
import math
import matplotlib.pyplot as plt
from pymavlink import mavutil

# Parámetros de conexión y control
CONN = 'udp:127.0.0.1:14552'  # SITL 60 - 51 con jetson
SYSID = 1
COMPID = 0

# Parámetros de la circunferencia
RADIUS = 3              # metros
ANGULAR_SPEED = 1.0       # rad/s (velocidad angular alrededor del centro)
LINEAR_SPEED = RADIUS * ANGULAR_SPEED  # v = ω·R
RATE = 50                 # Hz de envío de comandos

# Máscara para usar SOLO vx, vy, vz
TYPE_MASK = 0b0000111111000111

def send_velocity(master, vx, vy, vz=0.0):
    """Envía comando de velocidad en el frame LOCAL_NED."""
    master.mav.set_position_target_local_ned_send(
        0,
        SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )

def read_telemetry(master, real_t, real_x, real_y, real_vx, real_vy):
    """
    Lee todos los mensajes disponibles y extrae LOCAL_POSITION_NED.
    Los datos se añaden a las listas correspondientes (pasadas por referencia).
    """
    while True:
        msg = master.recv_match(blocking=False)
        if msg is None:
            break
        if msg.get_type() == 'LOCAL_POSITION_NED':
            t = time.time()
            real_t.append(t)
            real_x.append(msg.x)
            real_y.append(msg.y)
            real_vx.append(msg.vx)
            real_vy.append(msg.vy)

def circle_motion(master, duration, setpoint_t, setpoint_vx, setpoint_vy,
                  real_t, real_x, real_y, real_vx, real_vy):
    """
    Mueve el dron describiendo un círculo de radio RADIUS durante `duration` segundos.
    Calcula vx, vy en cada paso según θ = ω·t.
    Durante la ejecución registra setpoints y telemetría real.
    """
    time_period = 1.0 / RATE
    steps = int(duration * RATE)
    for i in range(steps):
        t_start = time.time()
        theta = ANGULAR_SPEED * (i * time_period)   # tiempo acumulado
        # Velocidad lineal en x/y local para círculo
        vx =  LINEAR_SPEED * math.cos(theta + math.pi/2)
        vy =  LINEAR_SPEED * math.sin(theta + math.pi/2)

        # 1) Guardar setpoint
        setpoint_t.append(t_start)
        setpoint_vx.append(vx)
        setpoint_vy.append(vy)

        # 2) Enviar comando
        send_velocity(master, vx, vy)

        # 3) Leer toda la telemetría disponible (no bloqueante)
        read_telemetry(master, real_t, real_x, real_y, real_vx, real_vy)

        # 4) Esperar hasta el siguiente ciclo
        elapsed = time.time() - t_start
        if elapsed < time_period:
            time.sleep(time_period - elapsed)

    # Al terminar, detener el dron
    send_velocity(master, 0.0, 0.0)
    # Leer mensajes finales
    read_telemetry(master, real_t, real_x, real_y, real_vx, real_vy)

def plot_results(setpoint_t, setpoint_vx, setpoint_vy,
                 real_t, real_x, real_y, real_vx, real_vy):
    """Genera las gráficas solicitadas."""
    plt.figure(figsize=(12, 10))

    # Trayectoria real (x vs y)
    plt.subplot(3, 1, 1)
    plt.plot(real_x, real_y, 'b-', linewidth=1, label='Trayectoria real')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trayectoria del dron en el plano XY')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    # Velocidad en X
    plt.subplot(3, 1, 2)
    plt.plot(setpoint_t, setpoint_vx, 'r-', linewidth=1, label='Setpoint vx')
    plt.plot(real_t, real_vx, 'b--', linewidth=1, label='Real vx')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('vx (m/s)')
    plt.title('Velocidad en X: setpoint vs real')
    plt.grid(True)
    plt.legend()

    # Velocidad en Y
    plt.subplot(3, 1, 3)
    plt.plot(setpoint_t, setpoint_vy, 'r-', linewidth=1, label='Setpoint vy')
    plt.plot(real_t, real_vy, 'b--', linewidth=1, label='Real vy')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('vy (m/s)')
    plt.title('Velocidad en Y: setpoint vs real')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 1) Conexión y espera de heartbeat
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")
    print("Asegúrate de que el dron esté en GUIDED, armado y en hover antes de continuar.")
    input("Presiona Enter para comenzar el círculo...")

    # 2) Preparar listas para almacenamiento
    setpoint_t = []
    setpoint_vx = []
    setpoint_vy = []
    real_t = []
    real_x = []
    real_y = []
    real_vx = []
    real_vy = []

    # 3) Ejecutar movimiento circular
    circle_duration = 2 * math.pi / ANGULAR_SPEED
    print(f"🌀 Iniciando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, duración≈{circle_duration:.1f}s")
    circle_motion(master, circle_duration,
                  setpoint_t, setpoint_vx, setpoint_vy,
                  real_t, real_x, real_y, real_vx, real_vy)
    print("⏹️ Círculo completado, dron detenido.")

    # 4) Cerrar conexión
    master.close()

    # 5) Graficar resultados (si hay datos)
    if real_t:
        plot_results(setpoint_t, setpoint_vx, setpoint_vy,
                     real_t, real_x, real_y, real_vx, real_vy)
    else:
        print("No se recibió telemetría. Revisa la conexión.")
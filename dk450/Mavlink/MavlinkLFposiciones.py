#!/usr/bin/env python3
"""
Líder-Seguidor con setpoints de posición (LOCAL_NED).
Lee la posición del líder vía MAVLink, suma un offset y envía el setpoint al seguidor.
Registro CSV y gráficas en tiempo real.
"""

import time
import csv
import matplotlib.pyplot as plt
from pymavlink import mavutil

# ----------------------------------------------------------------------
# Parámetros configurables
# ----------------------------------------------------------------------
CONNECTION_STRING = 'udp:127.0.0.1:14552'  # Ajustar según entorno
LEADER_SYSID = 1       # System ID del dron líder
FOLLOWER_SYSID = 2     # System ID del dron seguidor (el que controlamos)

# Offset deseado respecto al líder (X, Y, Z) en metros (Z positivo hacia arriba)
OFFSET = (1.0, 0.0, 1.5)

# Frecuencia de envío de setpoints (Hz)
RATE = 20
DT = 1.0 / RATE

# Activar gráficas en tiempo real
ENABLE_PLOT = True

# ----------------------------------------------------------------------
# Máscara para setpoints de posición (ignorar velocidades, aceleraciones, yaw_rate)
# ----------------------------------------------------------------------
POSITION_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

def send_position(master, sysid, x, y, z, yaw=0.0):
    """
    Envía un setpoint de posición (x, y, z) en el frame LOCAL_NED.
    z debe ser negativo para altura positiva (NED: z+ abajo).
    """
    master.mav.set_position_target_local_ned_send(
        0,                      # time_boot_ms (ignorado)
        sysid, 1,               # system id, component id
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        POSITION_MASK,
        x, y, z,                # posición
        0, 0, 0,                # velocidades ignoradas
        0, 0, 0,                # aceleraciones ignoradas
        yaw,                    # yaw (opcional)
        0                       # yaw_rate ignorado
    )

# ----------------------------------------------------------------------
# Inicialización MAVLink
# ----------------------------------------------------------------------
master = mavutil.mavlink_connection(CONNECTION_STRING)
master.wait_heartbeat()
print(f"Conectado: sistema {master.target_system}, componente {master.target_component}")

# ----------------------------------------------------------------------
# Variables de estado
# ----------------------------------------------------------------------
leader_x = leader_y = leader_z = None   # posición local del líder (x, y, altura positiva)
follower_x = follower_y = follower_z = None

pos_ready = False   # se han recibido posiciones de ambos?

# ----------------------------------------------------------------------
# Preparación de archivo CSV
# ----------------------------------------------------------------------
csv_filename = f'leader_follower_pos_{int(time.time())}.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['time', 'lx', 'ly', 'lz', 'sx', 'sy', 'sz',
                     'fx', 'fy', 'fz'])
start_time = time.time()

# ----------------------------------------------------------------------
# Configuración de gráficas (opcional)
# ----------------------------------------------------------------------
if ENABLE_PLOT:
    plt.ion()
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(8, 6))
    line_xd, = ax_x.plot([], [], 'r-', label='X set')
    line_x,  = ax_x.plot([], [], 'b-', label='X real')
    ax_x.set_ylabel('X (m)')
    ax_x.legend()
    ax_x.grid(True)

    line_yd, = ax_y.plot([], [], 'r-', label='Y set')
    line_y,  = ax_y.plot([], [], 'b-', label='Y real')
    ax_y.set_ylabel('Y (m)')
    ax_y.legend()
    ax_y.grid(True)

    line_zd, = ax_z.plot([], [], 'r-', label='Z set')
    line_z,  = ax_z.plot([], [], 'b-', label='Z real')
    ax_z.set_ylabel('Z (m)')
    ax_z.set_xlabel('Tiempo (s)')
    ax_z.legend()
    ax_z.grid(True)

    plot_t = []
    plot_xd = []
    plot_x = []
    plot_yd = []
    plot_y = []
    plot_zd = []
    plot_z = []

# ----------------------------------------------------------------------
# Bucle principal de control
# ----------------------------------------------------------------------
print("Iniciando envío de setpoints de posición. Presiona Ctrl+C para finalizar.")
try:
    while True:
        loop_start = time.time()

        # Leer todos los mensajes disponibles
        while True:
            msg = master.recv_match(blocking=False)
            if msg is None:
                break

            if msg.get_type() == 'LOCAL_POSITION_NED':
                sysid = msg.get_srcSystem()
                # Convertimos a altura positiva (z_ned -> -z_ned)
                x = msg.x
                y = msg.y
                z = -msg.z   # altura positiva hacia arriba

                if sysid == LEADER_SYSID:
                    leader_x, leader_y, leader_z = x, y, z
                elif sysid == FOLLOWER_SYSID:
                    follower_x, follower_y, follower_z = x, y, z
                else:
                    continue

                # Verificar si tenemos datos de ambos
                if leader_x is not None and follower_x is not None:
                    pos_ready = True

        # Si tenemos datos suficientes, calcular y enviar setpoint al seguidor
        if pos_ready:
            # Setpoint = líder + offset (en altura positiva)
            xd = leader_x + OFFSET[0]
            yd = leader_y + OFFSET[1]
            zd = leader_z + OFFSET[2]

            # Convertir a NED para el envío (z negativo hacia arriba)
            send_position(master, FOLLOWER_SYSID, xd, yd, -zd)

            # Posición real del seguidor (para registro)
            fx = follower_x
            fy = follower_y
            fz = follower_z

            # Registrar en CSV
            t = time.time() - start_time
            csv_writer.writerow([t,
                                 leader_x, leader_y, leader_z,
                                 xd, yd, zd,
                                 fx, fy, fz])
            csv_file.flush()

            # Actualizar gráficas
            if ENABLE_PLOT:
                plot_t.append(t)
                plot_xd.append(xd)
                plot_x.append(fx)
                plot_yd.append(yd)
                plot_y.append(fy)
                plot_zd.append(zd)
                plot_z.append(fz)

                line_xd.set_data(plot_t, plot_xd)
                line_x.set_data(plot_t, plot_x)
                line_yd.set_data(plot_t, plot_yd)
                line_y.set_data(plot_t, plot_y)
                line_zd.set_data(plot_t, plot_zd)
                line_z.set_data(plot_t, plot_z)

                for ax in (ax_x, ax_y, ax_z):
                    ax.relim()
                    ax.autoscale_view()
                plt.pause(0.001)

        # Mantener frecuencia de control
        elapsed = time.time() - loop_start
        sleep_time = DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nInterrupción por usuario.")

finally:
    csv_file.close()
    print(f"Datos guardados en {csv_filename}")
    if ENABLE_PLOT:
        plt.ioff()
        plt.show()
    master.close()
    print("Conexión cerrada.")
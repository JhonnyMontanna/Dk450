#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt

# ===============================
# CONFIGURACIÓN
# ===============================
CONN = 'udp:127.0.0.1:14552'
SYSID = 1
COMPID = 0

RATE =25
RADIUS = 4.0
ANGULAR_SPEED = 0.4

# ===============================
# MASK: posición + yaw
# ===============================
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
def send_position_yaw(master, x, y, z, yaw):
    master.mav.set_position_target_local_ned_send(
        0,
        SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_POS_YAW,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        yaw,
        0
    )

# ===============================
def wait_local_position(master):
    while True:
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        if msg:
            return msg

# ===============================
def fly_circle_and_log(master):
    # Logs
    t_log = []
    x_sp_log, y_sp_log = [], []
    x_log, y_log = [], []

    pos0 = wait_local_position(master)
    z = pos0.z
    cx = pos0.x + RADIUS
    cy = pos0.y


    duration = 2 * math.pi / ANGULAR_SPEED
    dt = 1.0 / RATE
    steps = int(duration / dt)

    print("🌀 Volando círculo + registrando datos")

    t0 = time.time()

    for i in range(steps):
        now = time.time()
        t = now - t0
        theta0 = math.pi
        theta = theta0 + ANGULAR_SPEED * t


        # Setpoint
        x_sp = cx + RADIUS * math.cos(theta)
        y_sp = cy + RADIUS * math.sin(theta)
        yaw = theta + math.pi / 2

        send_position_yaw(master, x_sp, y_sp, z, yaw)

        # Leer posición real (última disponible)
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
        if msg:
            x_real = msg.x
            y_real = msg.y

            # Guardar logs
            t_log.append(t)
            x_sp_log.append(x_sp)
            y_sp_log.append(y_sp)
            x_log.append(x_real)
            y_log.append(y_real)

        time.sleep(dt)

    print("⏹️ Vuelo terminado")
    return t_log, x_sp_log, y_sp_log, x_log, y_log

# ===============================
def plot_results(t, x_sp, y_sp, x, y):
    # X vs tiempo
    plt.figure()
    plt.plot(t, x_sp, label="x setpoint")
    plt.plot(t, x, label="x real")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("X [m]")
    plt.legend()
    plt.grid()

    # Y vs tiempo
    plt.figure()
    plt.plot(t, y_sp, label="y setpoint")
    plt.plot(t, y, label="y real")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid()

    # Plano XY
    plt.figure()
    plt.plot(x_sp, y_sp, '--', label="Trayectoria pedida")
    plt.plot(x, y, label="Trayectoria real")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.legend()
    plt.grid()

    plt.show()

# ===============================
if __name__ == "__main__":
    print("🔗 Conectando...")
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print("✅ Conectado")

    t, x_sp, y_sp, x, y = fly_circle_and_log(master)
    master.close()

    plot_results(t, x_sp, y_sp, x, y)

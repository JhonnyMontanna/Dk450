#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil

# Parámetros de conexión y control
CONN = 'udp:127.0.0.1:14552'  # SITL 60 - 51 con jetson
SYSID = 2
COMPID = 0

# Parámetros de la circunferencia
RADIUS = 1.5              # metros
ANGULAR_SPEED = 0.8     # rad/s (velocidad angular alrededor del centro)
LINEAR_SPEED = RADIUS * ANGULAR_SPEED  # v = ω·R
RATE = 50                  # Hz de envío de comandos

# Máscara para usar SOLO vx,vy,vz
TYPE_MASK = 0b0000111111000111

def send_velocity(master, vx, vy, vz=0.0):
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

def circle_motion(master, duration):
    """
    Mueve el dron describiendo un círculo de radio RADIUS durante `duration` segundos.
    Calcula vx, vy en cada paso según θ = ω·t.
    """
    time_period = 1.0 / RATE
    steps = int(duration * RATE)
    for i in range(steps):
        t = i * time_period
        theta = ANGULAR_SPEED * t
        # Velocidad lineal en x/y local para círculo
        vx =  LINEAR_SPEED * math.cos(theta + math.pi/2)
        vy =  LINEAR_SPEED * math.sin(theta + math.pi/2)
        send_velocity(master, vx, vy)
        time.sleep(time_period)
    # Al terminar, detener
    send_velocity(master, 0.0, 0.0)

if __name__ == '__main__':
    # 1) Conexión y espera de heartbeat
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

    # (Se asume que ya estás en GUIDED, armado y en hover)

    # 2) Describir un círculo completo
    circle_duration = 2 * math.pi / ANGULAR_SPEED  # tiempo para 2π rad
    print(f"🌀 Iniciando círculo: R={RADIUS} m, ω={ANGULAR_SPEED} rad/s, duración≈{circle_duration:.1f}s")
    circle_motion(master, circle_duration)

    print("⏹️ Círculo completado, deteniendo dron")
    master.close()

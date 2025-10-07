#!/usr/bin/env python3
import time
from pymavlink import mavutil
import keyboard  # pip install keyboard

# — Parámetros de conexión y control —
CONN = 'udp:127.0.0.1:14552'  # Puerto de tu SITL
SYSID = 1
COMPID = 0

RATE = 50                     # Hz de envio
TYPE_MASK = 0b0000111111000111  # Solo vx,vy,vz

# Velocidad base (m/s)
V = 0.5

def send_velocity(master, vx, vy, vz=0.0):
    master.mav.set_position_target_local_ned_send(
        0,
        SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK,
        0, 0, 0,      # x,y,z ignorados
        vx, vy, vz,   # velocidades
        0, 0, 0,      # aceleraciones ignoradas
        0, 0          # yaw, yaw_rate ignorados
    )

def main():
    # 1) Conexión y heartbeat
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"🔗 Conectado al SITL (SYS={master.target_system}, COMP={master.target_component})")
    print("Usa W/A/S/D para mover, R/F para subir/bajar, ESC para terminar.")

    # 2) Bucle principal a RATE Hz
    interval = 1.0 / RATE
    try:
        while True:
            start = time.perf_counter()

            # Lee teclas
            vx = 0.0
            vy = 0.0
            vz = 0.0
            if keyboard.is_pressed('w'):
                vx += V
            if keyboard.is_pressed('s'):
                vx -= V
            if keyboard.is_pressed('d'):
                vy += V
            if keyboard.is_pressed('a'):
                vy -= V
            if keyboard.is_pressed('r'):
                vz -= V  # Z negativo = subir
            if keyboard.is_pressed('f'):
                vz += V  # Z positivo = bajar
            if keyboard.is_pressed('esc'):
                break

            # Envía comando
            send_velocity(master, vx, vy, vz)

            # Ajusta para mantener RATE constante
            elapsed = time.perf_counter() - start
            to_sleep = interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            # si to_sleep < 0, el loop ya está lento, pero no bloqueamos

    finally:
        # 3) Al salir, detén el dron y desconecta
        print("\n⏹️ Deteniendo dron y saliendo…")
        send_velocity(master, 0.0, 0.0, 0.0)
        time.sleep(0.1)
        master.close()
        print("🔌 Desconectado")

if __name__ == '__main__':
    main()

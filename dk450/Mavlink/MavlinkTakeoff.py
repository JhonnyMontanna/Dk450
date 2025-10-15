#!/usr/bin/env python3
import time
from pymavlink import mavutil

# 1) Conexión al SITL
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')

master.wait_heartbeat()
print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

# 2) Cambiar a modo GUIDED
def set_mode_guided():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        4, 0, 0, 0, 0, 0
    )
    time.sleep(1)

# 3) Armar motores
def arm_motors():
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
        1, 0, 0, 0, 0, 0, 0
    )
    # Espera a que el firmware confirme armado
    print("⏳ Armando motores…")
    # Opcional: podrías comprobar master.motors_armed() en un bucle
    time.sleep(3)
    print("🟢 Motores armados")

# 4) Despegue automático
def takeoff(altitude):
    """
    Lanza el comando de takeoff hasta 'altitude' metros.
    """
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
        0, 0, 0, 0,
        0, 0, altitude
    )
    print(f"🚀 Despegando a {altitude} m…")
    # Tiempo de reserva para alcanzar altitud
    time.sleep(10)

# --- Secuencia principal ---
set_mode_guided()
arm_motors()
takeoff(1)   # cambia '5' por la altitud deseada en metros

master.close()
print("🔌 Desconectado")

#!/usr/bin/env python3
from pymavlink import mavutil

# 1) Conexión al SITL
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')
master.wait_heartbeat()
print(f"🔗 Conectado: SYS={master.target_system} COMP={master.target_component}")

# 2) (Se asume que ya estás en GUIDED y armado)

# 3) Enviar un único comando de posición relativa en LOCAL_NED
#    Este ejemplo pide al dron que avance 5 m hacia el norte (X),
#    2 m al este (Y) y mantenga la altitud (Z=0)
master.mav.set_position_target_local_ned_send(
    0,                                  # time_boot_ms (puede ir 0)
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
    0b0000111111111000,                 # type_mask: usar sólo x,y,z
    -4.0, -0.0, -3.0,                      # x=5 m, y=2 m, z=0 m
    0, 0, 0,                            # vx, vy, vz ignorados
    0, 0, 0,                            # ax, ay, az ignorados
    0, 0                                # yaw, yaw_rate ignorados
)

#

print("📍 Comando de posición enviado:")

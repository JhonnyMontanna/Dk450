from pymavlink import mavutil
import time

# Conexión al dron (ajusta según tu sistema)
master = mavutil.mavlink_connection('udp:127.0.0.1:14551')

# Esperar el primer mensaje de heartBeat
master.wait_heartbeat()
print("✅ Conectado con el dron")

# Imprimir encabezado
print(f"{'Tiempo':<10} | {'IMU':<5} | {'AccX':>7} {'AccY':>7} {'AccZ':>7} | {'GyrX':>7} {'GyrY':>7} {'GyrZ':>7}")

# Loop continuo para leer y mostrar datos
while True:
    msg = master.recv_match(type='RAW_IMU', blocking=True, timeout=5)
    if not msg:
        continue

    # Mostrar los valores para la IMU principal (IMU0)
    print(f"{msg.time_usec//1_000_000:<10} | {'IMU0':<5} | {msg.xacc:>7} {msg.yacc:>7} {msg.zacc:>7} | {msg.xgyro:>7} {msg.ygyro:>7} {msg.zgyro:>7}")

    # Buscar mensajes SCALED_IMU2 e IMU3 si están habilitados
    imu2 = master.recv_match(type='SCALED_IMU2', blocking=False)
    imu3 = master.recv_match(type='SCALED_IMU3', blocking=False)

    if imu2:
        print(f"{imu2.time_boot_ms//1000:<10} | {'IMU1':<5} | {imu2.xacc:>7} {imu2.yacc:>7} {imu2.zacc:>7} | {imu2.xgyro:>7} {imu2.ygyro:>7} {imu2.zgyro:>7}")
    if imu3:
        print(f"{imu3.time_boot_ms//1000:<10} | {'IMU2':<5} | {imu3.xacc:>7} {imu3.yacc:>7} {imu3.zacc:>7} | {imu3.xgyro:>7} {imu3.ygyro:>7} {imu3.zgyro:>7}")

    time.sleep(0.5)

#!/usr/bin/env python3
import argparse
from pymavlink import mavutil
import time
import math
from collections import deque
import matplotlib.pyplot as plt

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Monitor de actitud: roll, pitch, yaw y error")
parser.add_argument('--conn', default='udp:127.0.0.1:14552', help="Conexión al dron")
parser.add_argument('--visible', type=int, default=120, help="Cantidad de puntos a mostrar en la gráfica")
parser.add_argument('--csv', default=None, help="Archivo CSV opcional")
args = parser.parse_args()

# ---------------- Conexión MAVLink ----------------
print(f"Conectando a {args.conn} ...")
master = mavutil.mavlink_connection(args.conn)
master.wait_heartbeat()
print(f"Conectado al sistema {master.target_system} comp {master.target_component}")

# ---------------- Datos históricos ----------------
max_points = args.visible
roll_hist, pitch_hist, yaw_hist = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)
roll_des_hist, pitch_des_hist, yaw_des_hist = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)
roll_err_hist, pitch_err_hist, yaw_err_hist = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)

# ---------------- Función de simulación de referencia ----------------
# Aquí deberías reemplazar esto por tus setpoints reales del controlador
def get_desired_angles():
    # Ejemplo: senoidal para prueba
    t = time.time()
    roll_d = 5.0 * math.sin(t*0.5)
    pitch_d = 3.0 * math.sin(t*0.3)
    yaw_d = 10.0 * math.sin(t*0.2)
    return roll_d, pitch_d, yaw_d

# ---------------- Configuración matplotlib ----------------
plt.ion()
fig, axs = plt.subplots(3,1, figsize=(8,6))
lines = []

for ax, label in zip(axs, ['Roll (°)','Pitch (°)','Yaw (°)']):
    l_real, = ax.plot([], [], label='Real')
    l_des, = ax.plot([], [], label='Deseado')
    l_err, = ax.plot([], [], label='Error')
    ax.set_ylabel(label)
    ax.grid(True)
    ax.legend()
    lines.append((l_real, l_des, l_err))
axs[-1].set_xlabel("Tiempo (puntos)")

def update_plot():
    for i, hist in enumerate([(roll_hist, roll_des_hist, roll_err_hist),
                              (pitch_hist, pitch_des_hist, pitch_err_hist),
                              (yaw_hist, yaw_des_hist, yaw_err_hist)]):
        x_data = range(len(hist[0]))
        for l, y in zip(lines[i], hist):
            l.set_data(x_data, y)
        axs[i].relim()
        axs[i].autoscale_view()
    plt.pause(0.001)

# ---------------- Loop principal ----------------
print("Iniciando monitoreo de actitud (Ctrl+C para detener)...")
try:
    while True:
        msg = master.recv_match(type='ATTITUDE', blocking=True)
        if not msg:
            continue

        # Ángulo real en grados
        roll_real  = math.degrees(msg.roll)
        pitch_real = math.degrees(msg.pitch)
        yaw_real   = math.degrees(msg.yaw)

        # Ángulo deseado (simulado o desde tu controlador)
        roll_des, pitch_des, yaw_des = get_desired_angles()

        # Error
        roll_err = roll_des - roll_real
        pitch_err = pitch_des - pitch_real
        yaw_err = yaw_des - yaw_real

        # Guardar historial
        roll_hist.append(roll_real)
        pitch_hist.append(pitch_real)
        yaw_hist.append(yaw_real)

        roll_des_hist.append(roll_des)
        pitch_des_hist.append(pitch_des)
        yaw_des_hist.append(yaw_des)

        roll_err_hist.append(roll_err)
        pitch_err_hist.append(pitch_err)
        yaw_err_hist.append(yaw_err)

        # Mostrar por consola
        print(f"Roll: {roll_real:.2f}°, Pitch: {pitch_real:.2f}°, Yaw: {yaw_real:.2f}° | "
              f"Error: R={roll_err:.2f}°, P={pitch_err:.2f}°, Y={yaw_err:.2f}°")

        # Guardar CSV
        if args.csv:
            with open(args.csv,'a') as f:
                f.write(f"{time.time()},{roll_real},{pitch_real},{yaw_real},"
                        f"{roll_des},{pitch_des},{yaw_des},"
                        f"{roll_err},{pitch_err},{yaw_err}\n")

        # Actualizar gráfico
        update_plot()

except KeyboardInterrupt:
    print("Monitoreo detenido.")

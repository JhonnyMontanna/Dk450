#!/usr/bin/env python3
import math
from pymavlink import mavutil
import matplotlib.pyplot as plt
from collections import deque

# ---------------- Configuración ----------------
MAVLINK_CONNECTION = "udp:127.0.0.1:14552"  # Cambia según tu conexión
MAX_POINTS = 200

# Buffers para gráficos
roll_hist = deque(maxlen=MAX_POINTS)
pitch_hist = deque(maxlen=MAX_POINTS)
yaw_hist = deque(maxlen=MAX_POINTS)

# ---------------- Conexión MAVLink ----------------
master = mavutil.mavlink_connection(MAVLINK_CONNECTION)
master.wait_heartbeat()
print("Conectado a:", master.target_system, master.target_component)

# ---------------- Gráficos ----------------
plt.ion()
fig, axs = plt.subplots(3,1, figsize=(8,6))
lines = [axs[i].plot([], [], label=f"{lbl}")[0] for i,lbl in enumerate(['Roll','Pitch','Yaw'])]
for ax in axs:
    ax.grid(True)
    ax.legend()
axs[-1].set_xlabel("Muestras")

def update_plot():
    for i, data in enumerate([roll_hist, pitch_hist, yaw_hist]):
        lines[i].set_data(range(len(data)), data)
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    plt.pause(0.001)

# ---------------- Loop principal ----------------
while True:
    msg = master.recv_match(type='ATTITUDE', blocking=True)
    if msg:
        roll = math.degrees(msg.roll)
        pitch = math.degrees(msg.pitch)
        yaw = math.degrees(msg.yaw)

        roll_hist.append(roll)
        pitch_hist.append(pitch)
        yaw_hist.append(yaw)

        update_plot()

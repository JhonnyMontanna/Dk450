#!/usr/bin/env python3
import argparse
from pymavlink import mavutil
import time
import matplotlib.pyplot as plt
from collections import deque

# ---------------- Argumentos con valores por defecto ----------------
parser = argparse.ArgumentParser(description="Monitoreo de posición XYZ en tiempo real")
parser.add_argument('--mode', choices=['local', 'global'], default='global', help="Modo de coordenadas (default: global)")
parser.add_argument('--conn', default='udp:127.0.0.1:14552', help="Conexión al dron (default: udp:127.0.0.1:14552)")
parser.add_argument('--origin-lat', type=float, default=19.5942341)
parser.add_argument('--origin-lon', type=float, default=-99.2280871)
parser.add_argument('--origin-alt', type=float, default=2329.0)
parser.add_argument('--calib-mode', choices=['enu','angle','pair'], default='pair')
parser.add_argument('--calib-lat', type=float, default=19.5942429)
parser.add_argument('--calib-lon', type=float, default=-99.2280774)
parser.add_argument('--calib-alt', type=float, default=2329.0)
parser.add_argument('--calib-ang', type=float, default=180.0)
parser.add_argument('--expected-local-x', type=float, default=1.0)
parser.add_argument('--expected-local-y', type=float, default=1.0)
parser.add_argument('--out', choices=['ned','enu'], default='enu')
parser.add_argument('--visible', type=int, default=120)
parser.add_argument('--csv', default=None, help="Archivo CSV opcional para guardar historial")
args = parser.parse_args()

# ---------------- Conexión al dron ----------------
print(f"Conectando a: {args.conn} ...")
master = mavutil.mavlink_connection(args.conn)
master.wait_heartbeat()
print(f"Conectado al sistema {master.target_system} comp {master.target_component}")

# ---------------- Preparar gráficas ----------------
max_points = args.visible
x_hist = deque(maxlen=max_points)
y_hist = deque(maxlen=max_points)
z_hist = deque(maxlen=max_points)

plt.ion()
fig, axs = plt.subplots(3,1, figsize=(8,6))
lines = []
for ax, label in zip(axs, ['X', 'Y', 'Z']):
    line, = ax.plot([], [], label=label)
    ax.set_ylabel(f"{label} (m)")
    ax.grid(True)
    ax.legend()
    lines.append(line)
axs[-1].set_xlabel("Tiempo")

# ---------------- Función para actualizar gráfica ----------------
def update_plot(x_hist, y_hist, z_hist):
    lines[0].set_data(range(len(x_hist)), x_hist)
    lines[1].set_data(range(len(y_hist)), y_hist)
    lines[2].set_data(range(len(z_hist)), z_hist)
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    plt.pause(0.001)

# ---------------- Loop principal ----------------
print("Iniciando monitoreo XYZ (Ctrl+C para detener)...")
try:
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if not msg:
            continue

        # Coordenadas en metros aproximadas (lat/lon -> local ENU simple)
        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0

        # Aquí puedes poner tu transformación real usando el origen y calibración
        x_local = (lat - args.origin_lat) * 111319.5  # aprox metros por grado
        y_local = (lon - args.origin_lon) * 111319.5
        z_local = alt - args.origin_alt

        x_hist.append(x_local)
        y_hist.append(y_local)
        z_hist.append(z_local)

        print(f"X={x_local:.2f}, Y={y_local:.2f}, Z={z_local:.2f}")

        if args.csv:
            with open(args.csv, 'a') as f:
                f.write(f"{time.time()},{x_local},{y_local},{z_local}\n")

        update_plot(x_hist, y_hist, z_hist)

except KeyboardInterrupt:
    print("Monitoreo detenido.")

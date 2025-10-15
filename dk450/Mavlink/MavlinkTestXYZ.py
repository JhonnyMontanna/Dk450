#!/usr/bin/env python3
import argparse
from pymavlink import mavutil
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ---------------- Argumentos ----------------
parser = argparse.ArgumentParser(description="Monitoreo de posición XYZ en tiempo real tipo visualizer")
parser.add_argument('--conn', default='udp:127.0.0.1:14552', help="Conexión al dron (default: udp:127.0.0.1:14552)")
parser.add_argument('--mode', choices=['local','global'], default='global', help="Modo de coordenadas")
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
parser.add_argument('--alt-mode', choices=['ned','computed','lidar'], default='ned', help="Altura a mostrar")
parser.add_argument('--visible', type=int, default=120)
parser.add_argument('--csv', default=None, help="Archivo CSV opcional")
args = parser.parse_args()

# ---------------- Constantes WGS84 ----------------
WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

# ---------------- Conexión MAVLink ----------------
print(f"Conectando a {args.conn} ...")
master = mavutil.mavlink_connection(args.conn)
master.wait_heartbeat()
print(f"Conectado al sistema {master.target_system} comp {master.target_component}")

# ---------------- Variables para lidar (si no hay sensor, queda en 0) ----------------
lidar_z = 0.0

# ---------------- Funciones de transformación ----------------
def geodetic_to_ecef(lat_r, lon_r, alt):
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
    x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
    y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
    z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_r)
    return np.array([x, y, z])

def rotation_matrix(lat0, lon0):
    return np.array([
        [-math.sin(lon0),                 math.cos(lon0),                 0],
        [-math.sin(lat0)*math.cos(lon0), -math.sin(lat0)*math.sin(lon0), math.cos(lat0)],
        [ math.cos(lat0)*math.cos(lon0),  math.cos(lat0)*math.sin(lon0), math.sin(lat0)]
    ])

def compute_theta(origin_ecef, calib_ecef, R_enu, args):
    dx, dy, dz = calib_ecef - origin_ecef
    ref = R_enu.dot([dx, dy, dz])
    east, north = float(ref[0]), float(ref[1])
    if args.calib_mode == 'angle':
        return math.radians(args.calib_ang)
    elif args.calib_mode == 'enu':
        return 0.0
    else:  # pair
        theta_measured = math.atan2(north, east)
        theta_expected = math.atan2(args.expected_local_y, args.expected_local_x)
        return theta_expected - theta_measured

# ---------------- Pre-cálculos ----------------
lat0 = math.radians(args.origin_lat)
lon0 = math.radians(args.origin_lon)
lat_ref = math.radians(args.calib_lat)
lon_ref = math.radians(args.calib_lon)
h0 = args.origin_alt
h_ref = args.calib_alt

X0 = geodetic_to_ecef(lat0, lon0, h0)
Xr = geodetic_to_ecef(lat_ref, lon_ref, h_ref)
R_enu = rotation_matrix(lat0, lon0)
theta = compute_theta(X0, Xr, R_enu, args)

# ---------------- Preparar gráficas ----------------
max_points = args.visible
x_hist, y_hist, z_hist = deque(maxlen=max_points), deque(maxlen=max_points), deque(maxlen=max_points)

plt.ion()
fig, axs = plt.subplots(3,1, figsize=(8,6))
lines = []
for ax, label in zip(axs, ['X','Y','Z']):
    line, = ax.plot([], [], label=label)
    ax.set_ylabel(f"{label} (m)")
    ax.grid(True)
    ax.legend()
    lines.append(line)
axs[-1].set_xlabel("Tiempo")

def update_plot():
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

        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.relative_alt / 1000.0  # metros

        # ---------------- Transformación a ENU ----------------
        Xe, Ye, Ze = geodetic_to_ecef(math.radians(lat), math.radians(lon), alt)
        d = Xe - X0, Ye - X0, Ze - X0
        d = np.array([Xe - X0[0], Ye - X0[1], Ze - X0[2]])
        enu = R_enu.dot(d)
        # Rotación por theta
        xr = enu[0]*math.cos(theta) - enu[1]*math.sin(theta)
        yr = enu[0]*math.sin(theta) + enu[1]*math.cos(theta)
        zr_ned = enu[2]

        # ---------------- Selección de altura ----------------
        if args.alt_mode == 'ned':
            z_plot = zr_ned
        elif args.alt_mode == 'computed':
            z_plot = alt - h0
        else:
            z_plot = lidar_z

        x_hist.append(xr)
        y_hist.append(yr)
        z_hist.append(z_plot)

        print(f"X={xr:.2f}, Y={yr:.2f}, Z={z_plot:.2f}")

        if args.csv:
            with open(args.csv,'a') as f:
                f.write(f"{time.time()},{xr},{yr},{z_plot}\n")

        update_plot()

except KeyboardInterrupt:
    print("Monitoreo detenido.")

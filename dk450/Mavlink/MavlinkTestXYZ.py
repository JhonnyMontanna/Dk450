#!/usr/bin/env python3
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pymavlink import mavutil
import time

# Constants WGS84
WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

def geodetic_to_ecef(lat_r, lon_r, alt):
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
    x = (N + alt) * math.cos(lat_r) * math.cos(lon_r)
    y = (N + alt) * math.cos(lat_r) * math.sin(lon_r)
    z = (N * (1 - WGS84_E2) + alt) * math.sin(lat_r)
    return x, y, z

def get_rotation_matrix(lat_r, lon_r):
    return np.array([
        [-math.sin(lon_r),                 math.cos(lon_r),                 0],
        [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r), math.cos(lat_r)],
        [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)]
    ])

def compute_theta(X0, Y0, Z0, Xr, Yr, Zr, expected_local_x, expected_local_y):
    dx, dy, dz = Xr - X0, Yr - Y0, Zr - Z0
    ref = np.array([dx, dy, dz])
    east, north = float(ref[0]), float(ref[1])
    theta_measured = math.atan2(north, east)
    theta_expected = math.atan2(expected_local_y, expected_local_x)
    return theta_expected - theta_measured

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", default="udp:127.0.0.1:14552")
    parser.add_argument("--mode", choices=['global','local'], required=True)
    parser.add_argument("--origin-lat", type=float, required=True)
    parser.add_argument("--origin-lon", type=float, required=True)
    parser.add_argument("--origin-alt", type=float, required=True)
    parser.add_argument("--calib-lat", type=float, required=True)
    parser.add_argument("--calib-lon", type=float, required=True)
    parser.add_argument("--calib-alt", type=float, required=True)
    parser.add_argument("--calib-mode", choices=['enu','angle','pair'], default='pair')
    parser.add_argument("--calib-ang", type=float, default=0.0)
    parser.add_argument("--expected-local-x", type=float, default=1.0)
    parser.add_argument("--expected-local-y", type=float, default=1.0)
    parser.add_argument("--out", choices=['ned','enu'], default='enu')
    parser.add_argument("--alt-mode", choices=['ned','computed','lidar'], default='computed')
    parser.add_argument("--visible", type=int, default=120)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    # Convert origin/calib to radians
    lat0, lon0 = math.radians(args.origin_lat), math.radians(args.origin_lon)
    latr, lonr = math.radians(args.calib_lat), math.radians(args.calib_lon)
    h0, href = args.origin_alt, args.calib_alt

    # ECEF origin & calibration
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, h0)
    Xr, Yr, Zr = geodetic_to_ecef(latr, lonr, href)

    # Rotation ENU
    R_enu = get_rotation_matrix(lat0, lon0)

    # Compute theta correction
    theta = compute_theta(X0, Y0, Z0, Xr, Yr, Zr, args.expected_local_x, args.expected_local_y)

    # Connect to drone
    print(f"Connecting to {args.conn}...")
    master = mavutil.mavlink_connection(args.conn)
    master.wait_heartbeat()
    print("Heartbeat received")

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(3,1,sharex=True,figsize=(8,6))
    lines = [ax[i].plot([],[],'-')[0] for i in range(3)]
    ax[0].set_ylabel('X [m]'); ax[1].set_ylabel('Y [m]'); ax[2].set_ylabel('Z [m]')
    ax[2].set_xlabel('Time [s]')
    Xdata, Ydata, Zdata, Tdata = [], [], [], []

    start_time = time.time()
    lidar_z = 0.0

    while True:
        msg = master.recv_match(type=['GLOBAL_POSITION_INT','VFR_HUD','DISTANCE_SENSOR'], blocking=True)
        if msg is None:
            continue

        # Altura LIDAR
        if msg.get_type() == 'DISTANCE_SENSOR':
            lidar_z = msg.current_distance / 100.0  # mm→m

        # Coordenadas GPS
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1000.0  # mm→m

            lat_r, lon_r = math.radians(lat), math.radians(lon)
            Xe, Ye, Ze = geodetic_to_ecef(lat_r, lon_r, alt)
            d = np.array([Xe - X0, Ye - Y0, Ze - Z0])
            enu = R_enu.dot(d)

            # Rotate theta
            xr = enu[0]*math.cos(theta) - enu[1]*math.sin(theta)
            yr = enu[0]*math.sin(theta) + enu[1]*math.cos(theta)

            # Select altitude
            if args.alt_mode == 'ned':
                zr = -enu[2]  # Down positive
            elif args.alt_mode == 'computed':
                zr = alt - h0
            else:
                zr = lidar_z

            t = time.time() - start_time
            Xdata.append(xr); Ydata.append(yr); Zdata.append(zr); Tdata.append(t)

            # Update plots
            for l, d in zip(lines, [Xdata,Ydata,Zdata]):
                l.set_data(Tdata, d)
            for a in ax:
                a.relim()
                a.autoscale_view()
            plt.pause(0.001)

            # Print
            print(f"t={t:.1f}s | X={xr:.2f} Y={yr:.2f} Z={zr:.2f}")

            # Optional CSV
            if args.csv:
                with open(args.csv,'a') as f:
                    f.write(f"{t},{xr},{yr},{zr}\n")

if __name__=="__main__":
    main()

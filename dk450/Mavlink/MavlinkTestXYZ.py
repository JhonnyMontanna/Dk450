#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pos_monitor_fixed_mode.py
Monitor de posición X,Y,Z sin modo 'auto': debes indicar --mode local o --mode global.
- local: usa exclusivamente LOCAL_POSITION_NED (x=N, y=E, z=D, m).
- global: usa GLOBAL_POSITION_INT / GPS_RAW_INT y calcula local ENU respecto a origin,
          aplica calibración ('enu'|'angle'|'pair') y devuelve marco según --out (ned|enu).
Histórico en tiempo real y opción de guardar CSV.
"""
import argparse
import math
import time
import threading
import csv
from collections import deque
from pymavlink import mavutil
import numpy as np
import matplotlib.pyplot as plt

# WGS84
WGS84_A  = 6378137.0
WGS84_E2 = 6.69437999014e-3

# ---------- Math helpers ----------
def geodetic_to_ecef(lat_r, lon_r, alt_m):
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_r)**2)
    x = (N + alt_m) * math.cos(lat_r) * math.cos(lon_r)
    y = (N + alt_m) * math.cos(lat_r) * math.sin(lon_r)
    z = (N * (1 - WGS84_E2) + alt_m) * math.sin(lat_r)
    return x, y, z

def get_rotation_matrix(lat_r, lon_r):
    return np.array([
        [-math.sin(lon_r),                  math.cos(lon_r),                 0.0],
        [-math.sin(lat_r)*math.cos(lon_r), -math.sin(lat_r)*math.sin(lon_r), math.cos(lat_r)],
        [ math.cos(lat_r)*math.cos(lon_r),  math.cos(lat_r)*math.sin(lon_r), math.sin(lat_r)]
    ])

def compute_theta(calib_mode, calib_ang_deg, origin, calib, expected_local=(1.0,1.0)):
    if calib_mode == 'enu':
        return 0.0
    if calib_mode == 'angle':
        return math.radians(calib_ang_deg)
    # pair
    lat0_r = math.radians(origin[0]); lon0_r = math.radians(origin[1])
    latr_r = math.radians(calib[0]); lonr_r = math.radians(calib[1])
    X0, Y0, Z0 = geodetic_to_ecef(lat0_r, lon0_r, origin[2])
    Xr, Yr, Zr = geodetic_to_ecef(latr_r, lonr_r, calib[2])
    Renu = get_rotation_matrix(lat0_r, lon0_r)
    dx, dy, dz = Xr - X0, Yr - Y0, Zr - Z0
    ref = Renu.dot(np.array([dx, dy, dz]))
    east_meas = float(ref[0]); north_meas = float(ref[1])
    theta_measured = math.atan2(north_meas, east_meas)
    theta_expected = math.atan2(expected_local[1], expected_local[0])
    return theta_expected - theta_measured

def rotate_xy(x, y, theta_rad):
    ca = math.cos(theta_rad); sa = math.sin(theta_rad)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    return xr, yr

# ---------- Reader ----------
class PosReader(threading.Thread):
    def __init__(self, conn, buffers, stop_event, mode,
                 origin, calib, calib_mode, calib_ang, expected_local, out_frame, verbose=False):
        super().__init__(daemon=True)
        self.conn = conn
        self.buffers = buffers
        self.stop_event = stop_event
        self.mode = mode  # 'local' or 'global' (required)
        self.origin = origin
        self.calib = calib
        self.calib_mode = calib_mode
        self.calib_ang = calib_ang
        self.expected_local = expected_local
        self.out_frame = out_frame.lower()
        self.verbose = verbose

        # precompute origin
        self.lat0_r = math.radians(self.origin[0])
        self.lon0_r = math.radians(self.origin[1])
        self.h0 = float(self.origin[2])
        self.X0, self.Y0, self.Z0 = geodetic_to_ecef(self.lat0_r, self.lon0_r, self.h0)
        self.Renu = get_rotation_matrix(self.lat0_r, self.lon0_r)
        self.theta = compute_theta(self.calib_mode, self.calib_ang, self.origin, self.calib, self.expected_local)
        if self.verbose:
            print(f"[DEBUG] theta = {self.theta:.6f} rad ({math.degrees(self.theta):.3f}°)")

    def connect(self):
        try:
            self.master = mavutil.mavlink_connection(self.conn)
        except Exception as e:
            print(f"[ERROR] crear conexión: {e}")
            return False
        try:
            self.master.wait_heartbeat(timeout=5)
            if self.verbose:
                print(f"[DEBUG] heartbeat sys={self.master.target_system} comp={self.master.target_component}")
        except Exception:
            if self.verbose:
                print("[WARN] no heartbeat en 5s; seguiré escuchando mensajes si llegan.")
        return True

    def run(self):
        if not self.connect():
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            # Decide qué mensajes leer según modo
            if self.mode == 'local':
                msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=2.0)
                if msg is None:
                    # no llegó local en timeout; solo informar y continuar esperando
                    if self.verbose:
                        print("[INFO] esperando LOCAL_POSITION_NED...")
                    continue
                # usar LOCAL_POSITION_NED
                try:
                    x_n = float(getattr(msg, 'x', 0.0))
                    y_e = float(getattr(msg, 'y', 0.0))
                    z_d = float(getattr(msg, 'z', 0.0))
                except Exception:
                    continue
                out_x, out_y, out_z = self._from_local_ned(x_n, y_e, z_d)
                self._push(time.time(), out_x, out_y, out_z, source='LOCAL_POSITION_NED')
                continue

            # mode == 'global'
            if self.mode == 'global':
                msg = self.master.recv_match(type=['GLOBAL_POSITION_INT','GPS_RAW_INT'], blocking=True, timeout=2.0)
                if msg is None:
                    if self.verbose:
                        print("[INFO] esperando GLOBAL_POSITION_INT / GPS_RAW_INT ...")
                    continue
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    lat_i = getattr(msg, 'lat', None)
                    lon_i = getattr(msg, 'lon', None)
                    alt_i = getattr(msg, 'alt', None)
                    if lat_i is None or lon_i is None:
                        continue
                    lat_deg = float(lat_i) * 1e-7
                    lon_deg = float(lon_i) * 1e-7
                    alt_m = float(alt_i) * 1e-3 if alt_i is not None else 0.0
                else:  # GPS_RAW_INT
                    lat_i = getattr(msg, 'lat', None)
                    lon_i = getattr(msg, 'lon', None)
                    alt_i = getattr(msg, 'alt', None)
                    if lat_i is None or lon_i is None:
                        continue
                    lat_deg = float(lat_i) * 1e-7
                    lon_deg = float(lon_i) * 1e-7
                    alt_m = float(alt_i) * 1e-3 if alt_i is not None else 0.0

                # convert geodetic -> ECEF -> ENU (relative to origin) -> rotate by theta -> compose output
                lat_r = math.radians(lat_deg); lon_r = math.radians(lon_deg)
                Xe, Ye, Ze = geodetic_to_ecef(lat_r, lon_r, alt_m)
                d = np.array([Xe - self.X0, Ye - self.Y0, Ze - self.Z0])
                enu = self.Renu.dot(d)   # [east, north, up]
                east, north, up = float(enu[0]), float(enu[1]), float(enu[2])
                xr, yr = rotate_xy(east, north, self.theta)  # xr ~ east', yr ~ north'
                out_x, out_y, out_z = self._compose_output(xr, yr, up)
                self._push(time.time(), out_x, out_y, out_z, source=msg.get_type())
                continue

    # helpers
    def _from_local_ned(self, x_n, y_e, z_d):
        if self.out_frame == 'ned':
            return x_n, y_e, z_d
        # NED -> ENU
        enu_x = y_e
        enu_y = x_n
        enu_z = -z_d
        return enu_x, enu_y, enu_z

    def _compose_output(self, xr_east_rot, yr_north_rot, up):
        if self.out_frame == 'enu':
            return xr_east_rot, yr_north_rot, up
        # to NED: x=N, y=E, z=D=-up
        return yr_north_rot, xr_east_rot, -up

    def _push(self, t, x, y, z, source=''):
        self.buffers['t'].append(t)
        self.buffers['x'].append(x)
        self.buffers['y'].append(y)
        self.buffers['z'].append(z)
        self.buffers['src'].append(source)
        if self.verbose:
            print(f"[SAMPLE] src={source} t={t:.3f} x={x:.3f} y={y:.3f} z={z:.3f}")

# ---------- Plot ----------
def live_plot(buffers, stop_event, visible_seconds=60):
    plt.ion()
    fig, axs = plt.subplots(3,1,sharex=True, figsize=(10,7))
    l_x, = axs[0].plot([], [], label='X')
    l_y, = axs[1].plot([], [], label='Y')
    l_z, = axs[2].plot([], [], label='Z')
    axs[0].set_ylabel('X (m)')
    axs[1].set_ylabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_xlabel('Tiempo (s desde inicio)')
    for ax in axs:
        ax.grid(True)
        ax.legend()

    try:
        while not stop_event.is_set():
            if len(buffers['t']) < 2:
                time.sleep(0.2)
                continue
            t0 = buffers['t'][0]
            t_rel = [tt - t0 for tt in buffers['t']]
            xseries = [v if v is not None else float('nan') for v in buffers['x']]
            yseries = [v if v is not None else float('nan') for v in buffers['y']]
            zseries = [v if v is not None else float('nan') for v in buffers['z']]

            l_x.set_data(t_rel, xseries)
            l_y.set_data(t_rel, yseries)
            l_z.set_data(t_rel, zseries)

            now_rel = t_rel[-1]
            axs[2].set_xlim(max(0.0, now_rel - visible_seconds), now_rel)

            for ax in axs:
                ax.relim()
                ax.autoscale_view()

            fig.suptitle(f"Posición histórica (último {visible_seconds}s visible) - muestras: {len(buffers['t'])}")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1.0/8.0)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        plt.ioff()
        plt.close(fig)

# ---------- CSV ----------
def dump_csv(path, buffers):
    header = ['timestamp', 'x', 'y', 'z', 'source']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, t in enumerate(buffers['t']):
            w.writerow([t, buffers['x'][i], buffers['y'][i], buffers['z'][i], buffers['src'][i]])
    print(f"[INFO] CSV guardado en: {path}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Monitor posición X,Y,Z (modo fijo: local o global).")
    p.add_argument("--conn", default="udp:127.0.0.1:14552", help="Cadena MAVLink")
    p.add_argument("--mode", choices=['local','global'], required=True, help="ELIGE siempre la fuente: local o global")
    p.add_argument("--origin-lat", type=float, default=19.5942341, help="Lat origin (deg)")
    p.add_argument("--origin-lon", type=float, default=-99.2280871, help="Lon origin (deg)")
    p.add_argument("--origin-alt", type=float, default=2329.0, help="Alt origin (m)")
    p.add_argument("--calib-lat", type=float, default=19.5942429, help="Lat calibration point (deg)")
    p.add_argument("--calib-lon", type=float, default=-99.2280774, help="Lon calibration point (deg)")
    p.add_argument("--calib-alt", type=float, default=2329.0, help="Alt calibration point (m)")
    p.add_argument("--calib-mode", choices=['enu','angle','pair'], default='pair', help="Modo calibración")
    p.add_argument("--calib-ang", type=float, default=180.0, help="Ángulo (deg) si calib-mode=angle")
    p.add_argument("--expected-local-x", type=float, default=1.0, help="x esperado para pair (m)")
    p.add_argument("--expected-local-y", type=float, default=1.0, help="y esperado para pair (m)")
    p.add_argument("--out", choices=['ned','enu'], default='ned', help="Marco de salida para plot")
    p.add_argument("--visible", type=int, default=60, help="Segundos visibles en la ventana (scroll)")
    p.add_argument("--max-samples", type=int, default=0, help="Límite de muestras guardadas (0 = ilimitado)")
    p.add_argument("--csv", type=str, default=None, help="Guardar CSV al finalizar (ruta)")
    p.add_argument("--verbose", action='store_true', help="Imprime debug")
    return p.parse_args()

def main():
    args = parse_args()
    maxlen = args.max_samples if args.max_samples and args.max_samples > 0 else None
    buffers = {
        't': deque(maxlen=maxlen),
        'x': deque(maxlen=maxlen),
        'y': deque(maxlen=maxlen),
        'z': deque(maxlen=maxlen),
        'src': deque(maxlen=maxlen),
    }
    stop_event = threading.Event()
    reader = PosReader(
        conn=args.conn,
        buffers=buffers,
        stop_event=stop_event,
        mode=args.mode,
        origin=(args.origin_lat, args.origin_lon, args.origin_alt),
        calib=(args.calib_lat, args.calib_lon, args.calib_alt),
        calib_mode=args.calib_mode,
        calib_ang=args.calib_ang,
        expected_local=(args.expected_local_x, args.expected_local_y),
        out_frame=args.out,
        verbose=args.verbose
    )
    reader.start()

    try:
        live_plot(buffers, stop_event, visible_seconds=args.visible)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        if args.csv:
            dump_csv(args.csv, buffers)
        print("[INFO] terminado. muestras:", len(buffers['t']))

if __name__ == "__main__":
    main()

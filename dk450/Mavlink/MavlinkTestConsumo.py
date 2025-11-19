#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_monitor_sysstatus_metrics.py
Extiende tu monitor original e incluye:

- Cálculo de métricas completas para tabla Excel:
    Voltaje inicial/final/mín/máx
    Corriente mín/máx/prom
    Potencia mín/máx/prom
    mAh consumidos
    Energía Wh
    C-rate promedio
    Caída de voltaje
    Tiempo de vuelo
    Tiempo total de registro
- Guarda CSV opcional (--csv path.csv)
"""

import argparse
import time
import threading
import csv
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import os
import math

DEFAULT_CONN = "udp:0.0.0.0:14552"
DEFAULT_WARN_CURRENT_A = 50.0
DEFAULT_WARN_VOLTAGE_LOW = 10.5
DISPLAY_FPS = 8.0


# ----------------------------------------------
# CONVERSIONES
# ----------------------------------------------
def sys_mv_to_v(mv):
    try:
        return float(mv) * 0.001
    except:
        return None

def sys_10ma_to_a(c10mA):
    try:
        c = int(c10mA)
    except:
        return None
    if c == -1:
        return None
    return float(c) * 0.01


# ----------------------------------------------
# HILO LECTOR DE MAVLINK
# ----------------------------------------------
class SysStatusReader(threading.Thread):
    def __init__(self, conn_str, buffers, stop_event, verbose=False):
        super().__init__(daemon=True)
        self.conn_str = conn_str
        self.buffers = buffers
        self.stop_event = stop_event
        self.master = None
        self.verbose = verbose

    def connect(self):
        try:
            self.master = mavutil.mavlink_connection(self.conn_str)
        except Exception as e:
            print(f"[ERROR] falló conexión: {e}")
            return False

        try:
            self.master.wait_heartbeat(timeout=5)
            print(f"[OK] Heartbeat recibido.")
        except:
            print("[WARN] No llegó heartbeat (pero seguiré escuchando).")

        return True

    def run(self):
        if not self.connect():
            self.stop_event.set()
            return

        last_t = None

        while not self.stop_event.is_set():
            msg = self.master.recv_match(type=['SYS_STATUS', 'BATTERY_STATUS'], blocking=True, timeout=2.0)
            t = time.time()
            if msg is None:
                continue

            mtype = msg.get_type()
            voltage_v = None
            current_a = None

            if mtype == "SYS_STATUS":
                mv = msg.voltage_battery
                ca = msg.current_battery
                voltage_v = sys_mv_to_v(mv)
                current_a = sys_10ma_to_a(ca)

            elif mtype == "BATTERY_STATUS":
                try:
                    valid = [v for v in msg.voltages if v is not None and int(v) not in (0, 65535)]
                    if valid:
                        voltage_v = sum(valid) * 0.001
                    ca = msg.current_battery
                    current_a = sys_10ma_to_a(ca)
                except:
                    pass

            # Rellenar con último valor conocido si falta uno
            if voltage_v is not None or current_a is not None:
                if voltage_v is None:
                    voltage_v = self.buffers["voltage"][-1] if len(self.buffers["voltage"]) > 0 else None
                if current_a is None:
                    current_a = self.buffers["current"][-1] if len(self.buffers["current"]) > 0 else None

                # Potencia
                if voltage_v is not None and current_a is not None:
                    power_w = voltage_v * current_a
                else:
                    power_w = None

                # Energía (trapecio)
                if last_t is None:
                    dt = 0
                else:
                    dt = t - last_t

                if power_w is not None and dt > 0:
                    prev_p = self.buffers["power"][-1] if len(self.buffers["power"]) > 0 else power_w
                    inc_j = 0.5 * (prev_p + power_w) * dt
                    self.buffers["energy_j"] += inc_j

                # Guardar samples
                self.buffers["t"].append(t)
                self.buffers["voltage"].append(voltage_v)
                self.buffers["current"].append(current_a)
                self.buffers["power"].append(power_w)

                last_t = t


# ----------------------------------------------
# CALCULO DE MÉTRICAS
# ----------------------------------------------
def compute_metrics(buffers, battery_capacity_mAh=2200):
    t = buffers["t"]
    v = buffers["voltage"]
    i = buffers["current"]
    p = buffers["power"]

    if len(t) < 2:
        return None

    # Tiempo
    t_total = t[-1] - t[0]

    # Voltaje
    v_init = v[0]
    v_final = v[-1]
    v_min = min(v)
    v_max = max(v)

    # Corriente
    i_valid = [x for x in i if x is not None]
    i_min = min(i_valid)
    i_max = max(i_valid)
    i_avg = sum(i_valid) / len(i_valid)

    # Potencia
    p_valid = [x for x in p if x is not None]
    p_min = min(p_valid)
    p_max = max(p_valid)
    p_avg = sum(p_valid) / len(p_valid)

    # Energía Wh
    E_Wh = buffers["energy_j"] / 3600.0

    # mAh consumidos  (integración simple I_promedio * tiempo)
    mAh = (i_avg * t_total) / 3600.0 * 1000.0

    # C-rate promedio
    C_rate = mAh / battery_capacity_mAh

    # Caída de voltaje (sag)
    sag = v_init - v_min

    return {
        "voltaje_inicial": v_init,
        "voltaje_final": v_final,
        "voltaje_min": v_min,
        "voltaje_max": v_max,
        "corriente_min": i_min,
        "corriente_max": i_max,
        "corriente_prom": i_avg,
        "pot_min": p_min,
        "pot_max": p_max,
        "pot_prom": p_avg,
        "energia_Wh": E_Wh,
        "mAh_consumidos": mAh,
        "C_rate": C_rate,
        "sag": sag,
        "tiempo_total_s": t_total
    }


# ----------------------------------------------
# DUMP CSV
# ----------------------------------------------
def dump_csv(path, buffers):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","voltage(V)","current(A)","power(W)"])
        for t, v, i, p in zip(buffers["t"], buffers["voltage"], buffers["current"], buffers["power"]):
            w.writerow([t, v, i, p])
    print(f"[OK] CSV guardado en {path}")


# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", default=DEFAULT_CONN)
    parser.add_argument("--csv", default=None, help="Ruta CSV para guardar datos")
    parser.add_argument("--capacity", type=float, default=2200, help="Capacidad nominal mAh")
    args = parser.parse_args()

    buffers = {
        "t": [],
        "voltage": [],
        "current": [],
        "power": [],
        "energy_j": 0.0,
    }

    stop_event = threading.Event()
    reader = SysStatusReader(args.conn, buffers, stop_event)
    reader.start()

    print("[INFO] Presiona CTRL+C para terminar la medición.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Finalizando…")
        stop_event.set()
        reader.join()

    # ---- CÁLCULO DE MÉTRICAS ----
    metrics = compute_metrics(buffers, battery_capacity_mAh=args.capacity)

    if metrics is None:
        print("[ERROR] Pocos datos para métricas.")
        return

    print("\n===== MÉTRICAS DE LA SESIÓN =====")
    for k,v in metrics.items():
        print(f"{k}: {v}")

    # ---- CSV ----
    if args.csv is not None:
        dump_csv(args.csv, buffers)


if __name__ == "__main__":
    main()

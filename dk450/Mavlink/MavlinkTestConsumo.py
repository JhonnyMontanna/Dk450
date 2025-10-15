#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_monitor_sysstatus.py
Monitor histórico de consumo usando SYS_STATUS.current_battery y SYS_STATUS.voltage_battery.
- Voltaje: SYS_STATUS.voltage_battery (mV) -> V = mV * 0.001
- Corriente: SYS_STATUS.current_battery (10*mA) -> A = c10mA * 0.01
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

# ---------------- DEFAULTS ----------------
DEFAULT_CONN = "udp:0.0.0.0:14552"  # evite puertos <1024 (ej. 52) salvo que uses sudo
DEFAULT_WARN_CURRENT_A = 50.0       # advertencia si corriente supera esto (A)
DEFAULT_WARN_VOLTAGE_LOW = 10.5     # advertencia si voltaje por debajo (V)
DISPLAY_FPS = 8.0                   # refresco gráfico por segundo

# --------------- HELPERS ---------------
def sys_mv_to_v(mv):
    # mv puede ser None o 0
    try:
        return float(mv) * 0.001
    except Exception:
        return None

def sys_10ma_to_a(c10mA):
    # -1 significa no medido en muchos firmwares
    try:
        c = int(c10mA)
    except Exception:
        return None
    if c == -1:
        return None
    # cada unidad = 10 mA = 0.01 A
    # A = c * 0.01  -> hacemos la multiplicación explícita
    return float(c) * 0.01

# --------------- READER THREAD ---------------
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
            print(f"[ERROR] falló crear conexión: {e}")
            return False
        try:
            # esperamos heartbeat corto para confirmar enlace (no es obligatorio)
            self.master.wait_heartbeat(timeout=5)
            print(f"[OK] Heartbeat: sistema={self.master.target_system}, componente={self.master.target_component}")
        except Exception:
            print("[WARN] No se recibió heartbeat en 5s (pero seguiré escuchando mensajes si llegan).")
        return True

    def run(self):
        if not self.connect():
            self.stop_event.set()
            return

        last_t = None
        while not self.stop_event.is_set():
            # leer SYS_STATUS y BATTERY_STATUS (fallback)
            msg = self.master.recv_match(type=['SYS_STATUS', 'BATTERY_STATUS'], blocking=True, timeout=2.0)
            t = time.time()
            if msg is None:
                # opcional: imprimir avisos periódicos si no llegan mensajes
                # print("[INFO] sin mensajes en 2s...")
                continue

            mtype = msg.get_type()
            voltage_v = None
            current_a = None

            if mtype == 'SYS_STATUS':
                mv = getattr(msg, 'voltage_battery', None)
                ca = getattr(msg, 'current_battery', None)
                if mv is not None:
                    voltage_v = sys_mv_to_v(mv)
                if ca is not None:
                    current_a = sys_10ma_to_a(ca)
                if self.verbose:
                    print(f"[SYS_STATUS] volt(mV)={mv} curr(10mA)={ca}")
            elif mtype == 'BATTERY_STATUS':
                # BATTERY_STATUS puede tener array voltages (mV) y current_battery (10*mA o cA según firmware)
                # Usamos BATTERY_STATUS solo si SYS_STATUS no estuvo disponible recientemente
                try:
                    voltages = getattr(msg, 'voltages', None)
                    if voltages:
                        # sumar celdas válidas (65535 = no disponible)
                        valid = [v for v in voltages if (v is not None and int(v) != 65535 and int(v) != 0)]
                        if valid:
                            # voltaje total en V
                            voltage_v = sum(valid) * 0.001
                    ca = getattr(msg, 'current_battery', None)
                    if ca is not None and int(ca) != -1:
                        current_a = sys_10ma_to_a(ca)
                except Exception:
                    pass
                if self.verbose:
                    print(f"[BATTERY_STATUS] voltages_len={len(getattr(msg,'voltages',[]))} current_battery={getattr(msg,'current_battery',None)}")

            # Si al menos uno de los dos está presente, rellenar con último conocido si falta el otro
            have_any = (voltage_v is not None) or (current_a is not None)
            if have_any:
                # rellenar faltantes con último valor conocido (si existe)
                if voltage_v is None:
                    voltage_v = self.buffers['voltage'][-1] if len(self.buffers['voltage'])>0 else None
                if current_a is None:
                    current_a = self.buffers['current'][-1] if len(self.buffers['current'])>0 else None

                # potencia si ambos presentes
                power_w = None
                if (voltage_v is not None) and (current_a is not None):
                    # P = V * I
                    power_w = voltage_v * current_a

                # integración energía (trapecio)
                if last_t is None:
                    dt = 0.0
                else:
                    dt = t - last_t
                if power_w is not None and dt > 0:
                    prev_p = self.buffers['power'][-1] if len(self.buffers['power'])>0 else power_w
                    # incremento energía (J) = 0.5*(prev_p + power_w)*dt
                    inc_j = 0.5 * (prev_p + power_w) * dt
                    self.buffers['energy_j'] += inc_j

                # push a buffers (histórico)
                self.buffers['t'].append(t)
                self.buffers['voltage'].append(voltage_v)
                self.buffers['current'].append(current_a)
                self.buffers['power'].append(power_w)

                # warnings por rango (imprime valor anómalo)
                if voltage_v is not None and (voltage_v < self.buffers['config']['warn_voltage_low'] or voltage_v > self.buffers['config']['warn_voltage_high']):
                    print(f"[WARN] Voltaje inusual: {voltage_v:.2f} V (esperado {self.buffers['config']['warn_voltage_low']:.2f}-{self.buffers['config']['warn_voltage_high']:.2f} V)")

                if current_a is None:
                    # corriente no disponible
                    print("[WARN] Corriente no disponible (valor = -1 o no provisto).")
                else:
                    if current_a > self.buffers['config']['warn_current_high'] or current_a < self.buffers['config']['warn_current_low']:
                        print(f"[WARN] Corriente inusual: {current_a:.2f} A (esperado {self.buffers['config']['warn_current_low']:.2f}-{self.buffers['config']['warn_current_high']:.2f} A)")

                last_t = t

            # else: no datos relevantes en este mensaje, seguir
        # fin while
# fin class

# ----------------- PLOTTING -----------------
def live_plot_historic(buffers, stop_event, visible_seconds):
    plt.ion()
    fig, axs = plt.subplots(3,1, sharex=True, figsize=(10,7))
    fig.canvas.manager.set_window_title("Power Monitor - SYS_STATUS")
    l_v, = axs[0].plot([], [], label='Voltaje (V)')
    l_i, = axs[1].plot([], [], label='Corriente (A)')
    l_p, = axs[2].plot([], [], label='Potencia (W)')
    axs[0].set_ylabel("V")
    axs[1].set_ylabel("A")
    axs[2].set_ylabel("W")
    axs[2].set_xlabel("Tiempo (s desde inicio)")
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
            v = [x if x is not None else float('nan') for x in buffers['voltage']]
            i = [x if x is not None else float('nan') for x in buffers['current']]
            p = [x if x is not None else float('nan') for x in buffers['power']]

            l_v.set_data(t_rel, v)
            l_i.set_data(t_rel, i)
            l_p.set_data(t_rel, p)

            # Mostrar ventana visible_seconds (pero mantener todo el histórico almacenado)
            now_rel = t_rel[-1]
            axs[2].set_xlim(max(0.0, now_rel - visible_seconds), now_rel)

            for ax in axs:
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)

            energy_wh = buffers['energy_j'] / 3600.0
            fig.suptitle(f"Energía acumulada: {buffers['energy_j']:.1f} J = {energy_wh:.4f} Wh  | muestras: {len(buffers['t'])}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1.0 / DISPLAY_FPS)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        plt.ioff()
        plt.close(fig)

# ----------------- CSV DUMP -----------------
def dump_csv(path, buffers):
    header = ['timestamp', 'voltage_V', 'current_A', 'power_W']
    try:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for idx, t in enumerate(buffers['t']):
                v = buffers['voltage'][idx] if idx < len(buffers['voltage']) else ''
                c = buffers['current'][idx] if idx < len(buffers['current']) else ''
                p = buffers['power'][idx] if idx < len(buffers['power']) else ''
                w.writerow([t, v, c, p])
        print(f"[INFO] CSV guardado en: {path}")
    except Exception as e:
        print(f"[ERROR] Al guardar CSV: {e}")

# Guardado periódico en background
def autosave_thread(path, buffers, stop_event, period_s):
    if period_s <= 0:
        return
    while not stop_event.is_set():
        time.sleep(period_s)
        try:
            # guarda un CSV temporal con timestamp
            base, ext = os.path.splitext(path)
            ts = int(time.time())
            tmpname = f"{base}_{ts}{ext if ext else '.csv'}"
            dump_csv(tmpname, buffers)
        except Exception as e:
            print(f"[ERROR autosave] {e}")

# ----------------- MAIN -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Power monitor (SYS_STATUS) - histórico")
    p.add_argument("--conn", default=DEFAULT_CONN, help="Cadena MAVLink (ej: udp:127.0.0.1:14552)")
    p.add_argument("--visible", type=int, default=60, help="Segundos visibles en la ventana (scroll).")
    p.add_argument("--max-samples", type=int, default=0, help="Limita muestras (0 = ilimitado).")
    p.add_argument("--csv", default=None, help="Guardar CSV final al terminar (ruta).")
    p.add_argument("--autosave", type=int, default=0, help="Autosave periódico en segundos (0 = desactivar).")
    p.add_argument("--warn-current", type=float, default=DEFAULT_WARN_CURRENT_A, help="Umbral de advertencia corriente (A).")
    p.add_argument("--warn-voltage-low", type=float, default=DEFAULT_WARN_VOLTAGE_LOW, help="Umbral bajo de voltaje (V).")
    p.add_argument("--warn-voltage-high", type=float, default=1000.0, help="Umbral alto de voltaje (V) (por si quieres límite superior).")
    p.add_argument("--verbose", action='store_true', help="Imprimir mensajes SYS/BATTERY completos (debug).")
    return p.parse_args()

def main():
    args = parse_args()

    maxlen = args.max_samples if args.max_samples and args.max_samples > 0 else None
    buffers = {
        't': deque(maxlen=maxlen),
        'voltage': deque(maxlen=maxlen),
        'current': deque(maxlen=maxlen),
        'power': deque(maxlen=maxlen),
        'energy_j': 0.0,
        # config interno para límites
        'config': {
            'warn_current_low': -9999.0,
            'warn_current_high': args.warn_current,
            'warn_voltage_low': args.warn_voltage_low,
            'warn_voltage_high': args.warn_voltage_high,
        }
    }

    stop_event = threading.Event()
    reader = SysStatusReader(args.conn, buffers, stop_event, verbose=args.verbose)
    reader.start()

    # autosave si solicitado
    autosave_t = None
    if args.autosave and args.csv:
        autosave_t = threading.Thread(target=autosave_thread, args=(args.csv, buffers, stop_event, args.autosave), daemon=True)
        autosave_t.start()
    elif args.autosave and not args.csv:
        print("[WARN] --autosave requiere --csv para nombrar archivos (desactivado autosave).")

    try:
        live_plot_historic(buffers, stop_event, args.visible)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        if args.csv:
            dump_csv(args.csv, buffers)
        print("[INFO] Finalizado. Muestras totales:", len(buffers['t']))
        energy_wh = buffers['energy_j'] / 3600.0
        print(f"[INFO] Energía acumulada: {buffers['energy_j']:.1f} J = {energy_wh:.4f} Wh")

if __name__ == "__main__":
    main()

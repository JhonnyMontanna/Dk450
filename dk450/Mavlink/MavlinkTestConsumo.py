#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_monitor_sysstatus.py
Monitor histórico de consumo con detección automática de vuelo
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
DEFAULT_CONN = "udp:0.0.0.0:14552"
DEFAULT_WARN_CURRENT_A = 50.0
DEFAULT_WARN_VOLTAGE_LOW = 10.5
DISPLAY_FPS = 8.0

# ---------------- BATTERY PARAMETERS ----------------
BATTERY_CELLS = 3                    # 3S battery
BATTERY_CAPACITY_MAH = 6500          # 6500 mAh
BATTERY_C_RATING = 120               # 120C
BATTERY_NOMINAL_VOLTAGE = 3.7 * 3    # Nominal voltage for 3S
BATTERY_FULL_VOLTAGE = 4.2 * 3       # Full charge voltage for 3S
BATTERY_EMPTY_VOLTAGE = 3.3 * 3      # Empty voltage for 3S

# --------------- HELPERS ---------------
def sys_mv_to_v(mv):
    try:
        return float(mv) * 0.001
    except Exception:
        return None

def sys_10ma_to_a(c10mA):
    try:
        c = int(c10mA)
    except Exception:
        return None
    if c == -1:
        return None
    return float(c) * 0.01

def format_min_sec(total_minutes):
    """Convierte minutos decimal a formato min:seg"""
    if total_minutes <= 0:
        return "0:00"
    
    minutes = int(total_minutes)
    seconds = int((total_minutes - minutes) * 60)
    
    # Asegurar que los segundos estén en el rango 0-59
    if seconds >= 60:
        minutes += 1
        seconds = 0
    
    return f"{minutes}:{seconds:02d}"

# --------------- FLIGHT ANALYSIS ---------------
class FlightAnalyzer:
    def __init__(self):
        self.initial_voltage = None
        self.final_voltage = None
        self.min_voltage = float('inf')
        self.max_voltage = 0
        self.max_current = 0
        self.max_power = 0
        self.total_energy_wh = 0
        self.total_charge_mah = 0
        
        # Flight detection
        self.flight_start_time = None
        self.flight_end_time = None
        self.in_flight = False
        self.landing_detected = False
        
        # Voltage sag detection
        self.voltage_reposo = None
        self.voltage_sag_value = 0.0
        self.reposo_samples = deque(maxlen=20)  # ~2-3 segundos de muestras
        self.reposo_calculado = False
        self.sag_detectado = False
        
        # Landing detection
        self.landing_candidate_time = None
        self.landing_candidate_voltage = None
        self.stable_voltage_period = 0
        
        # For averages
        self.current_sum = 0
        self.power_sum = 0
        self.sample_count = 0
        self.flight_current_sum = 0
        self.flight_power_sum = 0
        self.flight_sample_count = 0
        
    def update(self, timestamp, voltage, current, power):
        # Initialize first voltage
        if self.initial_voltage is None and voltage is not None:
            self.initial_voltage = voltage
        
        if voltage is not None:
            # Update min/max voltage
            self.min_voltage = min(self.min_voltage, voltage)
            self.max_voltage = max(self.max_voltage, voltage)
            self.final_voltage = voltage
            
            # 1. CALCULAR VOLTAJE DE REPOSO (primeros 2-3 segundos)
            if not self.reposo_calculado:
                self.reposo_samples.append(voltage)
                if len(self.reposo_samples) >= 15:  # ~1.5-2 segundos de datos
                    self.voltage_reposo = sum(self.reposo_samples) / len(self.reposo_samples)
                    self.reposo_calculado = True
                    print(f"[FLIGHT] Voltaje de reposo calculado: {self.voltage_reposo:.2f}V")
            
            # 2. DETECTAR INICIO DE VUELO (primera caída brusca + corriente alta)
            if (not self.in_flight and not self.sag_detectado and 
                self.voltage_reposo is not None and current is not None):
                
                # Detectar caída brusca: voltaje baja >0.15V respecto al reposo
                caida_voltage = self.voltage_reposo - voltage
                if caida_voltage > 0.15 and current > 3.0:
                    self.sag_detectado = True
                    self.in_flight = True
                    self.flight_start_time = timestamp
                    self.voltage_sag_value = caida_voltage
                    print(f"[FLIGHT] ¡Inicio de vuelo detectado!")
                    print(f"[FLIGHT] Sag: {caida_voltage:.2f}V (de {self.voltage_reposo:.2f}V a {voltage:.2f}V)")
                    print(f"[FLIGHT] Corriente: {current:.1f}A")
            
            # 3. DETECTAR FIN DE VUELO (corriente mínima + voltaje estable)
            if self.in_flight and not self.landing_detected and current is not None:
                # Condición: corriente muy baja (<2A) indica posible aterrizaje
                if current < 2.0:
                    if self.landing_candidate_time is None:
                        # Primer momento candidato a aterrizaje
                        self.landing_candidate_time = timestamp
                        self.landing_candidate_voltage = voltage
                        self.stable_voltage_period = 0
                    else:
                        # Verificar estabilidad del voltaje (±0.05V)
                        if abs(voltage - self.landing_candidate_voltage) < 0.05:
                            self.stable_voltage_period += 1
                        else:
                            # Voltaje no estable, reiniciar contador
                            self.landing_candidate_voltage = voltage
                            self.stable_voltage_period = 0
                        
                        # Si tenemos 1 segundo de estabilidad, confirmar aterrizaje
                        if self.stable_voltage_period >= 5:  # 5 muestras ≈ 1 segundo
                            self.landing_detected = True
                            self.in_flight = False
                            self.flight_end_time = timestamp
                            print(f"[FLIGHT] ¡Aterrizaje detectado!")
                            print(f"[FLIGHT] Corriente final: {current:.1f}A, Voltaje estable: {voltage:.2f}V")
                else:
                    # Corriente alta, no es aterrizaje
                    self.landing_candidate_time = None
                    self.stable_voltage_period = 0
        
        # Update statistics
        if current is not None:
            self.max_current = max(self.max_current, current)
            self.current_sum += current
            self.sample_count += 1
            if self.in_flight:
                self.flight_current_sum += current
                self.flight_sample_count += 1
            
        if power is not None:
            self.max_power = max(self.max_power, power)
            self.power_sum += power
            if self.in_flight:
                self.flight_power_sum += power
    
    def calculate_metrics(self, total_time, total_energy_j):
        metrics = {}
        
        # Time calculations - guardamos en minutos para cálculos pero formateamos para display
        total_time_min = total_time / 60.0 if total_time else 0
        metrics['Tiempo total (min)'] = total_time_min
        metrics['Tiempo total (formateado)'] = format_min_sec(total_time_min)
        
        # Flight time calculation
        flight_duration_min = 0
        if self.flight_start_time and self.flight_end_time:
            flight_duration = self.flight_end_time - self.flight_start_time
            flight_duration_min = flight_duration / 60.0
            metrics['Tiempo de Vuelo (min)'] = flight_duration_min
            metrics['Tiempo de Vuelo (formateado)'] = format_min_sec(flight_duration_min)
        elif self.flight_start_time and not self.landing_detected:
            # Vuelo en progreso
            flight_duration = time.time() - self.flight_start_time
            flight_duration_min = flight_duration / 60.0
            metrics['Tiempo de Vuelo (min)'] = flight_duration_min
            metrics['Tiempo de Vuelo (formateado)'] = format_min_sec(flight_duration_min)
        else:
            metrics['Tiempo de Vuelo (min)'] = 0
            metrics['Tiempo de Vuelo (formateado)'] = "0:00"
        
        # Voltage metrics
        metrics['Voltaje Inicial (V)'] = self.initial_voltage if self.initial_voltage else 0
        metrics['Voltaje Final (V)'] = self.final_voltage if self.final_voltage else 0
        metrics['Caída de Voltaje(sag) (V)'] = self.voltage_sag_value
        metrics['Voltaje Mínimo (V)'] = self.min_voltage if self.min_voltage != float('inf') else 0
        metrics['Voltaje Máximo (V)'] = self.max_voltage
        
        # Current and power metrics
        metrics['I Máxima (A)'] = self.max_current
        
        # Calcular promedios solo durante el vuelo
        if self.flight_sample_count > 0:
            metrics['I Promedio (A)'] = self.flight_current_sum / self.flight_sample_count
            metrics['Potencia Promedio (W)'] = self.flight_power_sum / self.flight_sample_count
        else:
            metrics['I Promedio (A)'] = 0
            metrics['Potencia Promedio (W)'] = 0
            
        metrics['Potencia Máxima (W)'] = self.max_power
        
        # Energy calculations
        metrics['Energía Consumida (Wh)'] = total_energy_j / 3600.0 if total_energy_j else 0
        
        # Calculate mAh consumed (solo durante vuelo)
        if metrics['I Promedio (A)'] > 0 and flight_duration_min > 0:
            metrics['mAh Consumidos'] = metrics['I Promedio (A)'] * 1000 * (flight_duration_min / 60.0)
        else:
            metrics['mAh Consumidos'] = 0
        
        # C-rate calculations
        if BATTERY_CAPACITY_MAH > 0:
            metrics['C-rate Promedio'] = metrics['I Promedio (A)'] / (BATTERY_CAPACITY_MAH / 1000.0)
        else:
            metrics['C-rate Promedio'] = 0
        
        # SOH and Efficiency (simplified calculations)
        if self.initial_voltage and BATTERY_FULL_VOLTAGE > 0:
            soh_voltage = (self.initial_voltage / BATTERY_FULL_VOLTAGE) * 100
            metrics['SOH'] = min(100.0, soh_voltage)
        else:
            metrics['SOH'] = 100.0
            
        if metrics['Energía Consumida (Wh)'] > 0 and self.initial_voltage and self.final_voltage:
            theoretical_energy = (self.initial_voltage - self.final_voltage) * (BATTERY_CAPACITY_MAH / 1000.0)
            if theoretical_energy > 0:
                metrics['Eficiencia Energética (%)'] = (metrics['Energía Consumida (Wh)'] / theoretical_energy) * 100
            else:
                metrics['Eficiencia Energética (%)'] = 100.0
        else:
            metrics['Eficiencia Energética (%)'] = 100.0
        
        return metrics

    def get_flight_status(self):
        """Return current flight status for real-time display"""
        if self.in_flight and self.flight_start_time:
            current_time = time.time()
            flight_duration = current_time - self.flight_start_time
            flight_duration_min = flight_duration / 60.0
            return {
                'in_flight': True,
                'flight_time_min': flight_duration_min,
                'flight_time_formatted': format_min_sec(flight_duration_min),
                'voltage_sag': self.voltage_sag_value
            }
        else:
            return {
                'in_flight': False,
                'flight_time_min': 0,
                'flight_time_formatted': "0:00",
                'voltage_sag': 0
            }

# --------------- READER THREAD ---------------
class SysStatusReader(threading.Thread):
    def __init__(self, conn_str, buffers, stop_event, verbose=False):
        super().__init__(daemon=True)
        self.conn_str = conn_str
        self.buffers = buffers
        self.stop_event = stop_event
        self.master = None
        self.verbose = verbose
        self.analyzer = FlightAnalyzer()
        self.last_status_print = 0

    def connect(self):
        try:
            self.master = mavutil.mavlink_connection(self.conn_str)
        except Exception as e:
            print(f"[ERROR] falló crear conexión: {e}")
            return False
        try:
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
            msg = self.master.recv_match(type=['SYS_STATUS', 'BATTERY_STATUS'], blocking=True, timeout=2.0)
            t = time.time()
            if msg is None:
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
                try:
                    voltages = getattr(msg, 'voltages', None)
                    if voltages:
                        valid = [v for v in voltages if (v is not None and int(v) != 65535 and int(v) != 0)]
                        if valid:
                            voltage_v = sum(valid) * 0.001
                    ca = getattr(msg, 'current_battery', None)
                    if ca is not None and int(ca) != -1:
                        current_a = sys_10ma_to_a(ca)
                except Exception:
                    pass

            have_any = (voltage_v is not None) or (current_a is not None)
            if have_any:
                if voltage_v is None:
                    voltage_v = self.buffers['voltage'][-1] if len(self.buffers['voltage'])>0 else None
                if current_a is None:
                    current_a = self.buffers['current'][-1] if len(self.buffers['current'])>0 else None

                power_w = None
                if (voltage_v is not None) and (current_a is not None):
                    power_w = voltage_v * current_a

                if last_t is None:
                    dt = 0.0
                else:
                    dt = t - last_t
                if power_w is not None and dt > 0:
                    prev_p = self.buffers['power'][-1] if len(self.buffers['power'])>0 else power_w
                    inc_j = 0.5 * (prev_p + power_w) * dt
                    self.buffers['energy_j'] += inc_j

                self.buffers['t'].append(t)
                self.buffers['voltage'].append(voltage_v)
                self.buffers['current'].append(current_a)
                self.buffers['power'].append(power_w)

                # Update flight analyzer
                self.analyzer.update(t, voltage_v, current_a, power_w)

                # Print real-time status every 3 seconds
                if t - self.last_status_print >= 3.0:
                    self.print_realtime_status(t, voltage_v, current_a)
                    self.last_status_print = t

                last_t = t

    def print_realtime_status(self, current_time, voltage, current):
        """Print real-time flight status"""
        status = self.analyzer.get_flight_status()
        
        if status['in_flight']:
            print(f"[EN VUELO] T: {status['flight_time_formatted']}min, "
                  f"V: {voltage:.2f}V, I: {current:.1f}A, "
                  f"Sag: {status['voltage_sag']:.2f}V")
        elif self.analyzer.landing_detected:
            print(f"[ATERRIZAJE] Vuelo completado: {status['flight_time_formatted']}min")
        elif self.analyzer.reposo_calculado:
            print(f"[ESPERANDO] Listo para despegue - V: {voltage:.2f}V, I: {current:.1f}A")
        else:
            print(f"[CALIBRANDO] Calculando voltaje de reposo... V: {voltage:.2f}V")

    def get_analysis(self):
        """Get flight analysis results"""
        total_time = self.buffers['t'][-1] - self.buffers['t'][0] if len(self.buffers['t']) > 1 else 0
        return self.analyzer.calculate_metrics(total_time, self.buffers['energy_j'])

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

def save_analysis_csv(path, analysis_data):
    """Save flight analysis data to CSV"""
    try:
        # Crear una copia para el CSV manteniendo los valores numéricos
        csv_data = {}
        for key, value in analysis_data.items():
            if '(formateado)' not in key:  # Excluir campos formateados del CSV
                csv_data[key] = value
        
        file_exists = os.path.isfile(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_data)
        print(f"[INFO] Análisis guardado en: {path}")
    except Exception as e:
        print(f"[ERROR] Al guardar análisis: {e}")

def print_analysis(analysis_data):
    """Print formatted analysis results"""
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO DE BATERÍA")
    print("="*80)
    
    # Definir el orden y formato de visualización
    display_fields = [
        ('Tiempo total (formateado)', 'Tiempo total'),
        ('Tiempo de Vuelo (formateado)', 'Tiempo de Vuelo'),
        ('Voltaje Inicial (V)', 'V'),
        ('Voltaje Final (V)', 'V'),
        ('Caída de Voltaje(sag) (V)', 'V'),
        ('Voltaje Mínimo (V)', 'V'), 
        ('Voltaje Máximo (V)', 'V'),
        ('mAh Consumidos', ''),
        ('C-rate Promedio', ''),
        ('I Máxima (A)', 'A'),
        ('I Promedio (A)', 'A'),
        ('Potencia Máxima (W)', 'W'),
        ('Potencia Promedio (W)', 'W'),
        ('Energía Consumida (Wh)', 'Wh'),
        ('SOH', '%'),
        ('Eficiencia Energética (%)', '%')
    ]
    
    for field, unit in display_fields:
        if field in analysis_data:
            value = analysis_data[field]
            if unit:
                if field in ['Tiempo total (formateado)', 'Tiempo de Vuelo (formateado)']:
                    print(f"{field.replace('(formateado)', '(min:seg)')}: {value}")
                elif 'Voltaje' in field or 'Caída' in field:
                    print(f"{field}: {value:.2f} {unit}")
                elif 'I ' in field or 'Potencia' in field:
                    if 'Máxima' in field or 'Promedio' in field:
                        print(f"{field}: {value:.2f} {unit}")
                    else:
                        print(f"{field}: {value:.2f} {unit}")
                elif 'Energía' in field:
                    print(f"{field}: {value:.3f} {unit}")
                elif 'SOH' in field or 'Eficiencia' in field:
                    print(f"{field}: {value:.1f}{unit}")
                else:
                    print(f"{field}: {value} {unit}")
            else:
                if 'mAh' in field:
                    print(f"{field}: {value:.0f}")
                elif 'C-rate' in field:
                    print(f"{field}: {value:.2f}")
                else:
                    print(f"{field}: {value}")
    
    print("="*80)

# ----------------- MAIN -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Power monitor (SYS_STATUS) - histórico")
    p.add_argument("--conn", default=DEFAULT_CONN, help="Cadena MAVLink (ej: udp:127.0.0.1:14552)")
    p.add_argument("--visible", type=int, default=60, help="Segundos visibles en la ventana (scroll).")
    p.add_argument("--max-samples", type=int, default=0, help="Limita muestras (0 = ilimitado).")
    p.add_argument("--csv", default=None, help="Guardar CSV final al terminar (ruta).")
    p.add_argument("--analysis-csv", default="battery_analysis.csv", help="Guardar análisis de batería (ruta).")
    p.add_argument("--autosave", type=int, default=0, help="Autosave periódico en segundos (0 = desactivado).")
    p.add_argument("--warn-current", type=float, default=DEFAULT_WARN_CURRENT_A, help="Umbral de advertencia corriente (A).")
    p.add_argument("--warn-voltage-low", type=float, default=DEFAULT_WARN_VOLTAGE_LOW, help="Umbral bajo de voltaje (V).")
    p.add_argument("--warn-voltage-high", type=float, default=1000.0, help="Umbral alto de voltaje (V).")
    p.add_argument("--verbose", action='store_true', help="Imprimir mensajes SYS/BATTERY completos (debug).")
    p.add_argument("--no-save-analysis", action='store_true', help="No guardar análisis automáticamente.")
    return p.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("MONITOR DE BATERÍA PARA DRONE - DETECCIÓN AUTOMÁTICA DE VUELO")
    print(f"Batería: {BATTERY_CELLS}S {BATTERY_CAPACITY_MAH}mAh {BATTERY_C_RATING}C")
    print(f"Voltaje nominal: {BATTERY_NOMINAL_VOLTAGE:.1f}V")
    print("="*80)
    print("INSTRUCCIONES:")
    print("1. Conecta el drone EN REPOSO (sin armar motores)")
    print("2. Espera 'Listo para despegue'")
    print("3. Despega normalmente")
    print("4. El sistema detectará automáticamente inicio y fin de vuelo")
    print("="*80)

    maxlen = args.max_samples if args.max_samples and args.max_samples > 0 else None
    buffers = {
        't': deque(maxlen=maxlen),
        'voltage': deque(maxlen=maxlen),
        'current': deque(maxlen=maxlen),
        'power': deque(maxlen=maxlen),
        'energy_j': 0.0,
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
        
        # Get and print analysis
        analysis_data = reader.get_analysis()
        print_analysis(analysis_data)
        
        # Save data and analysis
        if args.csv:
            dump_csv(args.csv, buffers)
            
        if not args.no_save_analysis and analysis_data['Tiempo de Vuelo (min)'] > 0:
            save_analysis_csv(args.analysis_csv, analysis_data)
            print(f"[INFO] Análisis guardado en: {args.analysis_csv}")
        elif args.no_save_analysis:
            print("[INFO] Análisis no guardado (--no-save-analysis)")
        else:
            print("[INFO] Sin tiempo de vuelo detectado - análisis no guardado")
            
        print("[INFO] Finalizado. Muestras totales:", len(buffers['t']))
        energy_wh = buffers['energy_j'] / 3600.0
        print(f"[INFO] Energía acumulada: {buffers['energy_j']:.1f} J = {energy_wh:.4f} Wh")

def autosave_thread(path, buffers, stop_event, period_s):
    if period_s <= 0:
        return
    while not stop_event.is_set():
        time.sleep(period_s)
        try:
            base, ext = os.path.splitext(path)
            ts = int(time.time())
            tmpname = f"{base}_{ts}{ext if ext else '.csv'}"
            dump_csv(tmpname, buffers)
        except Exception as e:
            print(f"[ERROR autosave] {e}")

if __name__ == "__main__":
    main()
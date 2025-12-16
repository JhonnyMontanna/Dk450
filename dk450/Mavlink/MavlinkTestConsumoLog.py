#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
power_monitor_log_analysis.py
Análisis completo de consumo energético desde archivo de log MAVLink
Configuración directa para ejecutar en VS Code
DETECTA MÚLTIPLES VUELOS
"""
import time
import csv
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# ==============================================
# CONFIGURACIÓN - MODIFICA ESTAS VARIABLES
# ==============================================

# RUTA DEL ARCHIVO DE LOG (CAMBIAR ESTA RUTA)
LOG_FILE = r"C:\Users\jhonn\Documents\Mission Planner\logs\Posibles\Vuelo D2 tuning 12-dic B2.2.tlog"  # Cambia esta ruta

# PARÁMETROS DE BATERÍA (modificar si es necesario)
BATTERY_CELLS = 3                    # 3S battery
BATTERY_CAPACITY_MAH = 6500          # 6500 mAh
BATTERY_C_RATING = 120               # 120C
BATTERY_NOMINAL_VOLTAGE = 3.7 * 3    # Nominal voltage for 3S
BATTERY_FULL_VOLTAGE = 4.2 * 3       # Full charge voltage for 3S
BATTERY_EMPTY_VOLTAGE = 3.3 * 3      # Empty voltage for 3S

# PARÁMETROS DE DETECCIÓN DE VUELOS
VOLTAGE_SAG_THRESHOLD = 0.15         # Caída mínima de voltaje para detectar despegue (V)
CURRENT_SPIKE_THRESHOLD = 3.0        # Corriente mínima para detectar despegue (A)
LANDING_CURRENT_THRESHOLD = 2.0      # Corriente máxima para detectar aterrizaje (A)
VOLTAGE_STABILITY_THRESHOLD = 0.05   # Variación máxima de voltaje para estabilidad (V)
STABLE_PERIOD_REQUIRED = 5           # Muestras estables necesarias para confirmar aterrizaje

# OPCIONES DE ANÁLISIS (True/False)
GUARDAR_CSV_CRUDOS = False           # Guardar datos crudos en CSV
CSV_CRUDOS_RUTA = "datos_crudos.csv" # Ruta para datos crudos

GUARDAR_ANALISIS_CSV = True          # Guardar análisis en CSV
ANALISIS_CSV_RUTA = "battery_analysis_complete.csv" # Ruta para análisis

MOSTRAR_GRAFICAS = True              # Mostrar gráficas interactivas
MODO_VERBOSE = False                 # Mostrar mensajes detallados de debug

# ==============================================
# FUNCIONES AUXILIARES
# ==============================================

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
    
    if seconds >= 60:
        minutes += 1
        seconds = 0
    
    return f"{minutes}:{seconds:02d}"

# ==============================================
# CLASE DE ANÁLISIS DE VUELO INDIVIDUAL
# ==============================================

class SingleFlight:
    """Representa un vuelo individual"""
    def __init__(self, flight_number, start_time, start_voltage):
        self.flight_number = flight_number
        self.start_time = start_time
        self.end_time = None
        self.duration = 0
        self.start_voltage = start_voltage
        self.end_voltage = None
        self.min_voltage = float('inf')
        self.max_voltage = 0
        self.max_current = 0
        self.max_power = 0
        self.energy_consumed_j = 0
        self.avg_current = 0
        self.avg_power = 0
        
        # Para calcular promedios
        self.current_sum = 0
        self.power_sum = 0
        self.sample_count = 0
        
        # Para voltaje sag
        self.voltage_sag = 0
        
    def update(self, timestamp, voltage, current, power, energy_inc_j=0):
        """Actualiza las métricas del vuelo"""
        if voltage is not None:
            self.min_voltage = min(self.min_voltage, voltage)
            self.max_voltage = max(self.max_voltage, voltage)
            self.end_voltage = voltage
            
        if current is not None:
            self.max_current = max(self.max_current, current)
            self.current_sum += current
            self.sample_count += 1
            
        if power is not None:
            self.max_power = max(self.max_power, power)
            self.power_sum += power
            
        if energy_inc_j > 0:
            self.energy_consumed_j += energy_inc_j
            
        if self.start_time is not None and timestamp is not None:
            self.duration = timestamp - self.start_time
    
    def finalize(self, end_time):
        """Finaliza el vuelo y calcula promedios"""
        self.end_time = end_time
        self.duration = end_time - self.start_time
        
        if self.sample_count > 0:
            self.avg_current = self.current_sum / self.sample_count
            self.avg_power = self.power_sum / self.sample_count
            
        # Calcular voltage sag
        if self.start_voltage is not None and self.min_voltage != float('inf'):
            self.voltage_sag = self.start_voltage - self.min_voltage
    
    def get_metrics(self):
        """Devuelve métricas del vuelo"""
        return {
            'Número de Vuelo': self.flight_number,
            'Inicio (s)': self.start_time,
            'Fin (s)': self.end_time,
            'Duración (s)': self.duration,
            'Duración (min:seg)': format_min_sec(self.duration/60.0),
            'Voltaje Inicial (V)': self.start_voltage,
            'Voltaje Final (V)': self.end_voltage,
            'Voltaje Mínimo (V)': self.min_voltage if self.min_voltage != float('inf') else 0,
            'Voltaje Máximo (V)': self.max_voltage,
            'Caída de Voltaje(sag) (V)': self.voltage_sag,
            'I Máxima (A)': self.max_current,
            'I Promedio (A)': self.avg_current,
            'Potencia Máxima (W)': self.max_power,
            'Potencia Promedio (W)': self.avg_power,
            'Energía Consumida (J)': self.energy_consumed_j,
            'Energía Consumida (Wh)': self.energy_consumed_j / 3600.0,
            'mAh Consumidos': self.avg_current * 1000 * (self.duration / 3600.0) if self.duration > 0 else 0,
            'C-rate Promedio': self.avg_current / (BATTERY_CAPACITY_MAH / 1000.0) if BATTERY_CAPACITY_MAH > 0 else 0
        }

# ==============================================
# CLASE PRINCIPAL DE ANÁLISIS DE MÚLTIPLES VUELOS
# ==============================================

class FlightAnalyzer:
    def __init__(self, log_filename=None):
        self.log_filename = log_filename
        self.initial_voltage = None
        self.final_voltage = None
        self.min_voltage = float('inf')
        self.max_voltage = 0
        self.max_current = 0
        self.max_power = 0
        self.total_energy_j = 0
        
        # Para múltiples vuelos
        self.flights = []  # Lista de objetos SingleFlight
        self.current_flight = None
        self.flight_counter = 0
        self.in_flight = False
        
        # Detección de vuelo
        self.voltage_reposo = None
        self.reposo_samples = deque(maxlen=20)
        self.reposo_calculado = False
        
        # Detección de aterrizaje
        self.landing_candidate_time = None
        self.landing_candidate_voltage = None
        self.stable_voltage_period = 0
        
        # Para tracking de voltaje anterior para detección de sag
        self.last_voltage = None
        self.last_current = None
        
        # Para almacenar todos los datos
        self.timestamps = []
        self.voltages = []
        self.currents = []
        self.powers = []
        self.energy_j = []
        
        # Para calcular energía incremental
        self.prev_power = None
        self.prev_timestamp = None
        
    def update(self, timestamp, voltage, current, power):
        # Almacenar datos crudos
        self.timestamps.append(timestamp)
        self.voltages.append(voltage)
        self.currents.append(current)
        self.powers.append(power)
        
        # Calcular energía incremental
        energy_inc_j = 0
        if power is not None and self.prev_power is not None and self.prev_timestamp is not None:
            dt = timestamp - self.prev_timestamp
            if dt > 0:
                energy_inc_j = 0.5 * (self.prev_power + power) * dt
                self.total_energy_j += energy_inc_j
                
        self.energy_j.append(self.total_energy_j)
        self.prev_power = power
        self.prev_timestamp = timestamp
        
        # Actualizar estadísticas globales
        if voltage is not None:
            if self.initial_voltage is None:
                self.initial_voltage = voltage
            self.min_voltage = min(self.min_voltage, voltage)
            self.max_voltage = max(self.max_voltage, voltage)
            self.final_voltage = voltage
            
        if current is not None:
            self.max_current = max(self.max_current, current)
            
        if power is not None:
            self.max_power = max(self.max_power, power)
        
        # 1. CALCULAR VOLTAJE DE REPOSO (solo al inicio)
        if not self.reposo_calculado and voltage is not None:
            self.reposo_samples.append(voltage)
            if len(self.reposo_samples) >= 15:
                self.voltage_reposo = sum(self.reposo_samples) / len(self.reposo_samples)
                self.reposo_calculado = True
                print(f"[FLIGHT] Voltaje de reposo inicial calculado: {self.voltage_reposo:.2f}V")
        
        # 2. DETECTAR INICIO DE VUELO (caída de voltaje + pico de corriente)
        if not self.in_flight and voltage is not None and current is not None:
            # Para vuelos después del primero, usar voltaje anterior como referencia
            reference_voltage = self.voltage_reposo if self.voltage_reposo is not None else self.last_voltage
            
            if reference_voltage is not None:
                voltage_drop = reference_voltage - voltage
                
                # Condición: caída de voltaje significativa Y corriente alta
                if voltage_drop > VOLTAGE_SAG_THRESHOLD and current > CURRENT_SPIKE_THRESHOLD:
                    self.flight_counter += 1
                    self.in_flight = True
                    self.current_flight = SingleFlight(self.flight_counter, timestamp, voltage)
                    
                    print(f"\n[FLIGHT #{self.flight_counter}] ¡VUELO DETECTADO!")
                    print(f"  Inicio: {timestamp:.1f}s")
                    print(f"  Voltaje inicial: {voltage:.2f}V")
                    print(f"  Corriente: {current:.1f}A")
                    print(f"  Caída de voltaje: {voltage_drop:.2f}V")
        
        # 3. ACTUALIZAR VUELO ACTUAL
        if self.in_flight and self.current_flight is not None:
            self.current_flight.update(timestamp, voltage, current, power, energy_inc_j)
            
            # 4. DETECTAR ATERRIZAJE (corriente baja + voltaje estable)
            if current is not None and current < LANDING_CURRENT_THRESHOLD:
                if self.landing_candidate_time is None:
                    # Primer momento candidato a aterrizaje
                    self.landing_candidate_time = timestamp
                    self.landing_candidate_voltage = voltage
                    self.stable_voltage_period = 0
                else:
                    # Verificar estabilidad del voltaje
                    if voltage is not None and self.landing_candidate_voltage is not None:
                        if abs(voltage - self.landing_candidate_voltage) < VOLTAGE_STABILITY_THRESHOLD:
                            self.stable_voltage_period += 1
                        else:
                            # Voltaje no estable, reiniciar contador
                            self.landing_candidate_voltage = voltage
                            self.stable_voltage_period = 0
                        
                        # Si tenemos suficiente tiempo de estabilidad, confirmar aterrizaje
                        if self.stable_voltage_period >= STABLE_PERIOD_REQUIRED:
                            self.current_flight.finalize(timestamp)
                            self.flights.append(self.current_flight)
                            
                            print(f"\n[FLIGHT #{self.flight_counter}] ¡ATERRIZAJE DETECTADO!")
                            print(f"  Fin: {timestamp:.1f}s")
                            print(f"  Duración: {format_min_sec(self.current_flight.duration/60.0)}")
                            print(f"  Voltaje final: {voltage:.2f}V")
                            print(f"  Corriente final: {current:.1f}A")
                            print(f"  Energía consumida: {self.current_flight.energy_consumed_j/3600.0:.3f} Wh")
                            
                            # Reiniciar para próximo vuelo
                            self.in_flight = False
                            self.current_flight = None
                            self.landing_candidate_time = None
                            self.landing_candidate_voltage = None
                            self.stable_voltage_period = 0
            else:
                # Corriente alta, no es aterrizaje
                self.landing_candidate_time = None
                self.stable_voltage_period = 0
        
        # Guardar valores actuales para próxima detección
        self.last_voltage = voltage
        self.last_current = current
    
    def finalize(self):
        """Finaliza el análisis (útil si el vuelo no terminó)"""
        if self.in_flight and self.current_flight is not None and self.timestamps:
            # El vuelo estaba en progreso al final del log
            last_timestamp = self.timestamps[-1]
            self.current_flight.finalize(last_timestamp)
            self.flights.append(self.current_flight)
            
            print(f"\n[FLIGHT #{self.flight_counter}] VUELO INCOMPLETO DETECTADO")
            print(f"  Último timestamp: {last_timestamp:.1f}s")
            print(f"  Duración parcial: {format_min_sec(self.current_flight.duration/60.0)}")
    
    def get_data_arrays(self):
        """Devuelve arrays numpy de los datos"""
        return {
            'timestamps': np.array(self.timestamps),
            'voltages': np.array(self.voltages, dtype=np.float64),
            'currents': np.array(self.currents, dtype=np.float64),
            'powers': np.array(self.powers, dtype=np.float64),
            'energy_j': np.array(self.energy_j, dtype=np.float64)
        }
    
    def calculate_global_metrics(self):
        """Calcula métricas globales del archivo completo"""
        if len(self.timestamps) < 2:
            return {}
        
        total_time = self.timestamps[-1] - self.timestamps[0]
        total_time_min = total_time / 60.0
        
        # Calcular promedios globales
        valid_currents = [c for c in self.currents if c is not None]
        valid_powers = [p for p in self.powers if p is not None]
        
        avg_current = sum(valid_currents) / len(valid_currents) if valid_currents else 0
        avg_power = sum(valid_powers) / len(valid_powers) if valid_powers else 0
        
        # Sumar tiempos y energías de todos los vuelos
        total_flight_time = sum(f.duration for f in self.flights)
        total_flight_energy = sum(f.energy_consumed_j for f in self.flights)
        
        # Calcular mAh totales consumidos en vuelo
        total_mah_consumed = 0
        for flight in self.flights:
            if flight.duration > 0 and flight.avg_current > 0:
                total_mah_consumed += flight.avg_current * 1000 * (flight.duration / 3600.0)
        
        metrics = {
            'Archivo': os.path.basename(self.log_filename) if self.log_filename else "Desconocido",
            'Ruta Archivo': self.log_filename if self.log_filename else "Desconocido",
            'Tiempo total (min)': total_time_min,
            'Tiempo total (formateado)': format_min_sec(total_time_min),
            'Tiempo total de vuelo (min)': total_flight_time / 60.0,
            'Tiempo total de vuelo (formateado)': format_min_sec(total_flight_time / 60.0),
            'Número de vuelos detectados': len(self.flights),
            'Voltaje Inicial (V)': self.initial_voltage if self.initial_voltage else 0,
            'Voltaje Final (V)': self.final_voltage if self.final_voltage else 0,
            'Voltaje Mínimo (V)': self.min_voltage if self.min_voltage != float('inf') else 0,
            'Voltaje Máximo (V)': self.max_voltage,
            'I Máxima Global (A)': self.max_current,
            'I Promedio Global (A)': avg_current,
            'Potencia Máxima Global (W)': self.max_power,
            'Potencia Promedio Global (W)': avg_power,
            'Energía Consumida Total (Wh)': self.total_energy_j / 3600.0,
            'Energía Consumida en Vuelo (Wh)': total_flight_energy / 3600.0,
            'mAh Consumidos Totales': total_mah_consumed,
            'C-rate Promedio Global': avg_current / (BATTERY_CAPACITY_MAH / 1000.0) if BATTERY_CAPACITY_MAH > 0 else 0,
        }
        
        # Calcular eficiencia
        if self.initial_voltage and self.final_voltage and BATTERY_CAPACITY_MAH > 0:
            theoretical_energy = (self.initial_voltage - self.final_voltage) * (BATTERY_CAPACITY_MAH / 1000.0)
            if theoretical_energy > 0:
                efficiency = (self.total_energy_j / 3600.0 / theoretical_energy) * 100
                metrics['Eficiencia Energética (%)'] = efficiency
            else:
                metrics['Eficiencia Energética (%)'] = 100.0
        else:
            metrics['Eficiencia Energética (%)'] = 100.0
        
        # Calcular SOH
        if self.initial_voltage and BATTERY_FULL_VOLTAGE > 0:
            soh = (self.initial_voltage / BATTERY_FULL_VOLTAGE) * 100
            metrics['SOH'] = min(100.0, soh)
        else:
            metrics['SOH'] = 100.0
        
        return metrics

# ==============================================
# PROCESADOR DE LOG
# ==============================================

def process_log_file(log_file, verbose=False):
    """Procesa un archivo de log MAVLink y extrae datos de batería"""
    print(f"[INFO] Procesando archivo de log: {log_file}")
    
    if not os.path.exists(log_file):
        print(f"[ERROR] El archivo no existe: {log_file}")
        return None
    
    try:
        mlog = mavutil.mavlink_connection(log_file)
        
        # Crear analizador
        analyzer = FlightAnalyzer(log_filename=log_file)
        
        message_count = 0
        battery_messages_count = 0
        prev_msg_time = time.time()
        
        print("[INFO] Leyendo log...")
        
        while True:
            msg = mlog.recv_match(blocking=False)
            
            if msg is None:
                break
            
            message_count += 1
            
            if message_count % 1000 == 0:
                elapsed = time.time() - prev_msg_time
                print(f"\r[PROGRESO] Mensajes: {message_count}, "
                      f"Batería: {battery_messages_count} ({battery_messages_count/message_count*100:.1f}%)", end='')
                sys.stdout.flush()
                prev_msg_time = time.time()
            
            mtype = msg.get_type()
            voltage_v = None
            current_a = None
            
            if mtype in ['SYS_STATUS', 'BATTERY_STATUS']:
                battery_messages_count += 1
                
                if mtype == 'SYS_STATUS':
                    mv = getattr(msg, 'voltage_battery', None)
                    ca = getattr(msg, 'current_battery', None)
                    if mv is not None:
                        voltage_v = sys_mv_to_v(mv)
                    if ca is not None:
                        current_a = sys_10ma_to_a(ca)
                    if verbose:
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
                    if hasattr(msg, '_timestamp'):
                        log_timestamp = msg._timestamp
                        
                        # Calcular potencia
                        power_w = None
                        if (voltage_v is not None) and (current_a is not None):
                            power_w = voltage_v * current_a
                        
                        # Actualizar analizador
                        analyzer.update(log_timestamp, voltage_v, current_a, power_w)
        
        print(f"\n[INFO] Log procesado exitosamente")
        print(f"[INFO] Mensajes totales: {message_count}")
        print(f"[INFO] Mensajes de batería: {battery_messages_count}")
        print(f"[INFO] Datos de batería procesados: {len(analyzer.timestamps)} muestras")
        
        # Finalizar vuelos en progreso
        analyzer.finalize()
        
        mlog.close()
        return analyzer
        
    except Exception as e:
        print(f"[ERROR] Error procesando el log: {e}")
        return None

# ==============================================
# GRÁFICAS
# ==============================================

def plot_complete_flight(analyzer):
    """Muestra gráficas completas del vuelo con marcas de múltiples vuelos"""
    if len(analyzer.timestamps) < 2:
        print("[ERROR] No hay suficientes datos para graficar")
        return
    
    data = analyzer.get_data_arrays()
    
    # Crear figura con 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Usar nombre del archivo en el título de la ventana
    filename = os.path.basename(analyzer.log_filename) if analyzer.log_filename else "Log Desconocido"
    fig.canvas.manager.set_window_title(f"Análisis de Consumo - {filename}")
    
    # Ajustar tiempos para comenzar desde 0
    t0 = data['timestamps'][0]
    time_rel = data['timestamps'] - t0
    
    # 1. Voltaje vs Tiempo
    axs[0].plot(time_rel, data['voltages'], 'b-', linewidth=1.5, label='Voltaje (V)')
    axs[0].set_ylabel('Voltaje (V)', color='b')
    axs[0].tick_params(axis='y', labelcolor='b')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc='upper left')
    
    # Marcar inicio y fin de cada vuelo
    for i, flight in enumerate(analyzer.flights):
        start_rel = flight.start_time - t0
        end_rel = flight.end_time - t0 if flight.end_time else time_rel[-1]
        
        # Área sombreada para cada vuelo
        axs[0].axvspan(start_rel, end_rel, alpha=0.2, color='green', label=f'Vuelo {i+1}' if i == 0 else "")
        
        # Líneas verticales
        axs[0].axvline(x=start_rel, color='g', linestyle='--', alpha=0.7, linewidth=1)
        axs[0].axvline(x=end_rel, color='r', linestyle='--', alpha=0.7, linewidth=1)
        
        # Etiqueta del vuelo
        axs[0].text(start_rel, axs[0].get_ylim()[1] * 0.95, f'V{i+1}', 
                   verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 2. Corriente vs Tiempo
    axs[1].plot(time_rel, data['currents'], 'r-', linewidth=1.5, label='Corriente (A)')
    axs[1].set_ylabel('Corriente (A)', color='r')
    axs[1].tick_params(axis='y', labelcolor='r')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc='upper left')
    
    # Marcar umbral de corriente para vuelo
    axs[1].axhline(y=CURRENT_SPIKE_THRESHOLD, color='orange', linestyle=':', alpha=0.5, label='Umbral despegue')
    axs[1].axhline(y=LANDING_CURRENT_THRESHOLD, color='purple', linestyle=':', alpha=0.5, label='Umbral aterrizaje')
    
    # 3. Potencia vs Tiempo
    axs[2].plot(time_rel, data['powers'], 'g-', linewidth=1.5, label='Potencia (W)')
    axs[2].set_ylabel('Potencia (W)', color='g')
    axs[2].tick_params(axis='y', labelcolor='g')
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc='upper left')
    
    # 4. Energía Acumulada vs Tiempo
    energy_wh = data['energy_j'] / 3600.0
    axs[3].plot(time_rel, energy_wh, 'm-', linewidth=1.5, label='Energía (Wh)')
    axs[3].set_ylabel('Energía (Wh)', color='m')
    axs[3].set_xlabel('Tiempo (s)')
    axs[3].tick_params(axis='y', labelcolor='m')
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc='upper left')
    
    # Marcar energía al final de cada vuelo
    for i, flight in enumerate(analyzer.flights):
        if flight.end_time:
            end_rel = flight.end_time - t0
            energy_at_end = flight.energy_consumed_j / 3600.0
            axs[3].plot(end_rel, energy_at_end, 'ko', markersize=5)
            axs[3].text(end_rel, energy_at_end, f' V{i+1}', 
                       verticalalignment='bottom', horizontalalignment='left')
    
    # Configurar eje X
    total_duration = time_rel[-1]
    axs[3].set_xlim(0, total_duration)
    
    # Añadir título general con información del archivo
    display_filename = os.path.basename(analyzer.log_filename) if analyzer.log_filename else "Log"
    
    fig.suptitle(f'Análisis de Consumo - {display_filename}\n'
                 f'Duración total: {format_min_sec(total_duration/60.0)} - '
                 f'Vuelos detectados: {len(analyzer.flights)}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ==============================================
# FUNCIONES DE IMPRESIÓN Y GUARDADO
# ==============================================

def print_flight_details(flights):
    """Imprime detalles de cada vuelo individual"""
    print("\n" + "="*80)
    print("DETALLES DE CADA VUELO")
    print("="*80)
    
    for flight in flights:
        metrics = flight.get_metrics()
        print(f"\nVUELO #{flight.flight_number}:")
        print(f"  Duración: {metrics['Duración (min:seg)']}")
        print(f"  Voltaje: {metrics['Voltaje Inicial (V)']:.2f}V → {metrics['Voltaje Final (V)']:.2f}V")
        print(f"  Sag: {metrics['Caída de Voltaje(sag) (V)']:.2f}V")
        print(f"  Corriente: Prom {metrics['I Promedio (A)']:.1f}A, Max {metrics['I Máxima (A)']:.1f}A")
        print(f"  Potencia: Prom {metrics['Potencia Promedio (W)']:.1f}W, Max {metrics['Potencia Máxima (W)']:.1f}W")
        print(f"  Energía: {metrics['Energía Consumida (Wh)']:.3f} Wh")
        print(f"  mAh: {metrics['mAh Consumidos']:.0f}")
        print(f"  C-rate: {metrics['C-rate Promedio']:.2f}")

def print_global_analysis(global_metrics):
    """Imprime análisis global formateado"""
    print("\n" + "="*80)
    print("ANÁLISIS GLOBAL COMPLETO")
    print("="*80)
    
    display_fields = [
        ('Archivo', ''),
        ('Ruta Archivo', ''),
        ('Tiempo total (formateado)', 'Tiempo total'),
        ('Tiempo total de vuelo (formateado)', 'Tiempo total de vuelo'),
        ('Número de vuelos detectados', ''),
        ('Voltaje Inicial (V)', 'V'),
        ('Voltaje Final (V)', 'V'),
        ('Voltaje Mínimo (V)', 'V'),
        ('Voltaje Máximo (V)', 'V'),
        ('I Máxima Global (A)', 'A'),
        ('I Promedio Global (A)', 'A'),
        ('Potencia Máxima Global (W)', 'W'),
        ('Potencia Promedio Global (W)', 'W'),
        ('Energía Consumida Total (Wh)', 'Wh'),
        ('Energía Consumida en Vuelo (Wh)', 'Wh'),
        ('mAh Consumidos Totales', ''),
        ('C-rate Promedio Global', ''),
        ('SOH', '%'),
        ('Eficiencia Energética (%)', '%')
    ]
    
    for field, unit in display_fields:
        if field in global_metrics:
            value = global_metrics[field]
            if unit:
                if field in ['Tiempo total (formateado)', 'Tiempo total de vuelo (formateado)']:
                    print(f"{field}: {value}")
                elif 'Voltaje' in field:
                    print(f"{field}: {value:.2f} {unit}")
                elif 'I ' in field or 'Potencia' in field:
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

def save_raw_csv(path, analyzer):
    """Guarda datos crudos en CSV"""
    try:
        data = analyzer.get_data_arrays()
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'voltage_V', 'current_A', 'power_W', 'energy_J', 'filename', 'flight_number'])
            
            # Determinar en qué vuelo está cada muestra
            flight_numbers = []
            current_flight_idx = 0
            
            for timestamp in data['timestamps']:
                flight_num = 0
                for i, flight in enumerate(analyzer.flights):
                    if flight.start_time <= timestamp <= flight.end_time:
                        flight_num = i + 1
                        break
                flight_numbers.append(flight_num)
            
            for i in range(len(data['timestamps'])):
                writer.writerow([
                    data['timestamps'][i],
                    data['voltages'][i] if not np.isnan(data['voltages'][i]) else '',
                    data['currents'][i] if not np.isnan(data['currents'][i]) else '',
                    data['powers'][i] if not np.isnan(data['powers'][i]) else '',
                    data['energy_j'][i] if not np.isnan(data['energy_j'][i]) else '',
                    analyzer.log_filename if analyzer.log_filename else '',
                    flight_numbers[i]
                ])
        print(f"[INFO] Datos crudos guardados en: {path}")
    except Exception as e:
        print(f"[ERROR] Al guardar CSV: {e}")

def save_analysis_csv(path, flights, global_metrics):
    """Guarda análisis en CSV"""
    try:
        # Guardar vuelos individuales
        flights_path = os.path.splitext(path)[0] + "_vuelos.csv"
        with open(flights_path, 'w', newline='') as f:
            if flights:
                fieldnames = list(flights[0].get_metrics().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for flight in flights:
                    writer.writerow(flight.get_metrics())
        
        # Guardar métricas globales
        global_path = os.path.splitext(path)[0] + "_global.csv"
        with open(global_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=global_metrics.keys())
            writer.writeheader()
            writer.writerow(global_metrics)
        
        print(f"[INFO] Análisis guardado en: {flights_path} y {global_path}")
    except Exception as e:
        print(f"[ERROR] Al guardar análisis: {e}")

# ==============================================
# FUNCIÓN PRINCIPAL
# ==============================================

def main():
    """Función principal - Ejecuta el análisis"""
    print("="*80)
    print("ANÁLISIS COMPLETO DE CONSUMO ENERGÉTICO - MÚLTIPLES VUELOS")
    print(f"Batería: {BATTERY_CELLS}S {BATTERY_CAPACITY_MAH}mAh {BATTERY_C_RATING}C")
    print(f"Archivo de log: {LOG_FILE}")
    print("="*80)
    
    # Verificar que el archivo existe
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] El archivo no existe: {LOG_FILE}")
        print("[INFO] Verifica que la ruta en la variable LOG_FILE sea correcta.")
        return
    
    # Procesar log completo
    analyzer = process_log_file(LOG_FILE, verbose=MODO_VERBOSE)
    
    if analyzer is None or len(analyzer.timestamps) < 2:
        print("[ERROR] No se pudieron procesar datos suficientes del log")
        return
    
    # Mostrar resumen de vuelos detectados
    print(f"\n[RESUMEN] Vuelos detectados: {len(analyzer.flights)}")
    
    if analyzer.flights:
        print("\n" + "="*80)
        print("RESUMEN DE VUELOS:")
        print("="*80)
        for flight in analyzer.flights:
            print(f"Vuelo #{flight.flight_number}: {format_min_sec(flight.duration/60.0)}, "
                  f"Energía: {flight.energy_consumed_j/3600.0:.3f} Wh")
    
    # Mostrar detalles de cada vuelo
    if analyzer.flights:
        print_flight_details(analyzer.flights)
    
    # Calcular y mostrar métricas globales
    global_metrics = analyzer.calculate_global_metrics()
    print_global_analysis(global_metrics)
    
    # Mostrar gráficas completas
    if MOSTRAR_GRAFICAS:
        print("\n[INFO] Generando gráficas completas...")
        plot_complete_flight(analyzer)
    
    # Guardar datos si está configurado
    if GUARDAR_CSV_CRUDOS:
        save_raw_csv(CSV_CRUDOS_RUTA, analyzer)
    
    if GUARDAR_ANALISIS_CSV and analyzer.flights:
        save_analysis_csv(ANALISIS_CSV_RUTA, analyzer.flights, global_metrics)
    elif GUARDAR_ANALISIS_CSV:
        print("[INFO] Sin vuelos detectados - análisis no guardado")
    
    print(f"[INFO] Proceso completado exitosamente")
    print(f"[INFO] Muestras procesadas: {len(analyzer.timestamps)}")
    print(f"[INFO] Vuelos detectados: {len(analyzer.flights)}")
    print(f"[INFO] Archivo analizado: {LOG_FILE}")

# ==============================================
# EJECUCIÓN
# ==============================================

if __name__ == "__main__":
    # Mostrar configuración actual
    print("="*80)
    print("CONFIGURACIÓN ACTUAL:")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"GUARDAR_CSV_CRUDOS: {GUARDAR_CSV_CRUDOS}")
    print(f"MOSTRAR_GRAFICAS: {MOSTRAR_GRAFICAS}")
    print(f"MODO_VERBOSE: {MODO_VERBOSE}")
    print(f"\nParámetros de detección:")
    print(f"  Umbral voltaje sag: {VOLTAGE_SAG_THRESHOLD} V")
    print(f"  Umbral corriente despegue: {CURRENT_SPIKE_THRESHOLD} A")
    print(f"  Umbral corriente aterrizaje: {LANDING_CURRENT_THRESHOLD} A")
    print(f"  Estabilidad voltaje: {VOLTAGE_STABILITY_THRESHOLD} V")
    print("="*80)
    
    # Preguntar confirmación antes de continuar
    respuesta = input("\n¿Continuar con el análisis? (s/n): ").strip().lower()
    
    if respuesta == 's' or respuesta == 'si' or respuesta == '':
        main()
    else:
        print("[INFO] Análisis cancelado por el usuario")
    
    print("\n[INFO] Presiona Enter para salir...")
    input()
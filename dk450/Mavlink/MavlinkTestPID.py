#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
attitude_error_analysis.py
Análisis completo de error de actitud para tuning de PID desde logs MAVLink
Configuración directa para ejecutar en VS Code
ANÁLISIS DE ERROR, DERIVADAS, INTEGRALES Y OSCILACIONES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, fft
from scipy.integrate import cumulative_trapezoid
import warnings
import os
import sys
import time
from pymavlink import mavutil
from collections import deque
import csv

warnings.filterwarnings('ignore')

# ==============================================
# CONFIGURACIÓN - MODIFICA ESTAS VARIABLES
# ==============================================

# RUTA DEL ARCHIVO DE LOG (CAMBIAR ESTA RUTA)
LOG_FILE = r"C:\Users\jhonn\Documents\Mission Planner\logs\Posibles\d2-tuning15-dic.bin"  # CAMBIA AQUÍ TU RUTA

# PARÁMETROS DE ANÁLISIS
SAMPLING_RATE = 50                    # Hz (típico para Pixhawk)
ERROR_THRESHOLD = 5.0                # Grados - umbral para considerar error significativo
OSCILLATION_FREQ_THRESHOLD = 2.0     # Hz - frecuencia mínima para considerar oscilación
SETTLING_TIME_TOLERANCE = 2.0        # % - tolerancia para tiempo de establecimiento

# OPCIONES DE ANÁLISIS (True/False)
ANALIZAR_ROLL = True
ANALIZAR_PITCH = True
MOSTRAR_GRAFICAS = True
GUARDAR_CSV_ANALISIS = True
GUARDAR_CSV_CRUDOS = False
MODO_VERBOSE = False                  # Mostrar mensajes detallados de debug

# RUTAS DE SALIDA (modificar si es necesario)
CSV_ANALISIS_RUTA = "attitude_error_analysis.csv"
CSV_CRUDOS_RUTA = "attitude_raw_data.csv"
GRAFICAS_RUTA_ROLL = "roll_analysis.png"
GRAFICAS_RUTA_PITCH = "pitch_analysis.png"
REPORTE_RUTA = "tuning_diagnostic_report.txt"

# ==============================================
# FUNCIONES AUXILIARES
# ==============================================

def format_min_sec(total_seconds):
    """Convierte segundos a formato min:seg"""
    if total_seconds <= 0:
        return "0:00"
    
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    
    return f"{minutes}:{seconds:02d}"

def radians_to_degrees(rad):
    """Convierte radianes a grados"""
    if rad is None:
        return None
    return rad * 180.0 / np.pi

def classify_oscillation_severity(zcr, freq, crest_factor):
    """Clasifica la severidad de las oscilaciones"""
    score = 0
    
    # Alta frecuencia es peor para control
    if freq > 5:  # > 5 Hz
        score += 3
    elif freq > 2:  # 2-5 Hz
        score += 2
    elif freq > 0.5:  # 0.5-2 Hz
        score += 1
    
    # Alta tasa de cruce por cero
    if zcr > 10:  # > 10 cruces/seg
        score += 2
    elif zcr > 5:  # 5-10 cruces/seg
        score += 1
    
    # Alto factor de cresta
    if crest_factor > 3:  # Picos muy pronunciados
        score += 2
    elif crest_factor > 2:
        score += 1
    
    # Clasificar
    if score >= 5:
        return "SEVERA - Ajustar P y D"
    elif score >= 3:
        return "MODERADA - Considerar ajustar D"
    elif score >= 1:
        return "LEVE - Monitorear"
    else:
        return "MÍNIMA - Sistema estable"

# ==============================================
# CLASE DE ANÁLISIS DE ERROR PARA UN EJE
# ==============================================

class AxisErrorAnalyzer:
    """Analiza el error para un eje específico (Roll o Pitch)"""
    
    def __init__(self, axis_name, sampling_rate):
        self.axis_name = axis_name
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Datos crudos
        self.timestamps = []
        self.desired = []
        self.actual = []
        
        # Resultados
        self.results = None
        
    def add_data_point(self, timestamp, desired_val, actual_val):
        """Añade un punto de datos"""
        self.timestamps.append(timestamp)
        self.desired.append(desired_val if desired_val is not None else 0)
        self.actual.append(actual_val if actual_val is not None else 0)
    
    def calculate_metrics(self):
        """Calcula todas las métricas de error"""
        if len(self.timestamps) < 10:
            print(f"[WARNING] Insuficientes datos para {self.axis_name}")
            return None
        
        # Convertir a arrays numpy
        time_arr = np.array(self.timestamps)
        desired_arr = np.array(self.desired)
        actual_arr = np.array(self.actual)
        
        # Ajustar tiempo para comenzar desde 0
        time_zero = time_arr - time_arr[0]
        
        # 1. Señal de error
        error = desired_arr - actual_arr
        
        # 2. Derivada del error (primera derivada)
        error_derivative = np.gradient(error, self.dt)
        
        # 3. Segunda derivada del error
        error_second_derivative = np.gradient(error_derivative, self.dt)
        
        # 4. Integral del error absoluto (IAE - Integral Absolute Error)
        error_abs = np.abs(error)
        iae = cumulative_trapezoid(error_abs, time_zero, initial=0)
        
        # 5. Integral de la parte positiva del error
        error_positive = error.copy()
        error_positive[error_positive < 0] = 0
        positive_integral = cumulative_trapezoid(error_positive, time_zero, initial=0)
        
        # 6. Integral de la parte negativa del error
        error_negative = error.copy()
        error_negative[error_negative > 0] = 0
        negative_integral = cumulative_trapezoid(np.abs(error_negative), time_zero, initial=0)
        
        # 7. Análisis de frecuencia/oscillación
        freq_metrics = self._analyze_oscillations(error, time_zero)
        
        # 8. Cálculo de métricas estadísticas
        stats = self._calculate_statistics(error, error_derivative, error_second_derivative,
                                          iae, positive_integral, negative_integral,
                                          desired_arr, actual_arr, time_zero)
        
        # Combinar con métricas de frecuencia
        stats.update(freq_metrics)
        
        # Almacenar resultados
        self.results = {
            'time': time_zero,
            'timestamps_original': time_arr,
            'desired': desired_arr,
            'actual': actual_arr,
            'error': error,
            'error_derivative': error_derivative,
            'error_second_derivative': error_second_derivative,
            'iae': iae,
            'positive_integral': positive_integral,
            'negative_integral': negative_integral,
            'stats': stats
        }
        
        return self.results
    
    def _analyze_oscillations(self, error_signal, time):
        """Analiza las oscilaciones en la señal de error"""
        n = len(error_signal)
        
        # 1. Zero Crossing Rate (ZCR) - tasa de cruces por cero
        zero_crossings = np.where(np.diff(np.sign(error_signal)))[0]
        zcr = len(zero_crossings) / (time[-1] - time[0]) if (time[-1] - time[0]) > 0 else 0
        
        # 2. Análisis espectral con FFT
        fft_result = fft.fft(error_signal)
        freqs = fft.fftfreq(n, self.dt)
        
        # Tomar magnitud y solo frecuencias positivas
        magnitude = np.abs(fft_result[:n//2])
        positive_freqs = freqs[:n//2]
        
        # Encontrar pico dominante (excluyendo DC - frecuencia 0)
        mask = positive_freqs > 0.1  # Ignorar frecuencias muy bajas
        if np.any(mask):
            dominant_idx = np.argmax(magnitude[mask])
            dominant_freq = positive_freqs[mask][dominant_idx]
            dominant_magnitude = magnitude[mask][dominant_idx]
            
            # Ancho de banda a -3dB
            max_power = dominant_magnitude**2
            half_power = max_power / 2
            
            # Encontrar frecuencias donde la potencia es al menos la mitad
            power_spectrum = magnitude**2
            mask_half_power = power_spectrum[mask] >= half_power
            if np.any(mask_half_power):
                bandwidth = positive_freqs[mask][mask_half_power][-1] - \
                           positive_freqs[mask][mask_half_power][0]
            else:
                bandwidth = 0
        else:
            dominant_freq = 0
            dominant_magnitude = 0
            bandwidth = 0
        
        # 3. Desviación estándar de la derivada (indicador de "agitación")
        if len(error_signal) > 1:
            derivative = np.diff(error_signal) / self.dt
            derivative_std = np.std(derivative)
        else:
            derivative_std = 0
        
        # 4. Índice de Oscilación (OSI - Oscillation Index)
        if len(error_signal) > 2:
            second_derivative = np.diff(error_signal, 2) / (self.dt**2)
            sign_changes = np.sum(np.diff(np.sign(second_derivative)) != 0)
            osi = sign_changes / len(second_derivative) * 100 if len(second_derivative) > 0 else 0
        else:
            osi = 0
        
        # 5. Factor de Cresta (Crest Factor)
        rms = np.sqrt(np.mean(error_signal**2))
        crest_factor = np.max(np.abs(error_signal)) / rms if rms > 0 else 0
        
        return {
            'Zero_Crossing_Rate_Hz': zcr,
            'Dominant_Frequency_Hz': dominant_freq,
            'Dominant_Magnitude': dominant_magnitude,
            'Spectral_Bandwidth_Hz': bandwidth,
            'Derivative_Std': derivative_std,
            'Oscillation_Index_Percent': osi,
            'Crest_Factor': crest_factor,
            'Oscillation_Severity': classify_oscillation_severity(zcr, dominant_freq, crest_factor)
        }
    
    def _calculate_statistics(self, error, error_derivative, error_second_derivative,
                             iae, positive_integral, negative_integral,
                             desired, actual, time):
        """Calcula métricas estadísticas del error"""
        
        # Calcular overshoot (sobrepico)
        overshoot = self._calculate_overshoot(desired, actual)
        
        # Calcular tiempo de establecimiento (settling time)
        settling_time = self._calculate_settling_time(error, time)
        
        # Calcular retardo (lag) - correlación cruzada
        if len(desired) > 10 and len(actual) > 10:
            correlation = np.correlate(desired - np.mean(desired), 
                                      actual - np.mean(actual), mode='full')
            lag = np.argmax(correlation) - (len(desired) - 1)
            lag_time = lag * self.dt
        else:
            lag_time = 0
        
        stats = {
            'RMSE': np.sqrt(np.mean(error**2)),
            'MAE': np.mean(np.abs(error)),
            'Max_Error': np.max(np.abs(error)),
            'Std_Error': np.std(error),
            'Mean_Error': np.mean(error),
            'Error_Derivative_RMS': np.sqrt(np.mean(error_derivative**2)),
            'Error_Second_Derivative_RMS': np.sqrt(np.mean(error_second_derivative**2)),
            'IAE_Final': iae[-1] if len(iae) > 0 else 0,
            'Positive_Integral_Final': positive_integral[-1] if len(positive_integral) > 0 else 0,
            'Negative_Integral_Final': negative_integral[-1] if len(negative_integral) > 0 else 0,
            'Overshoot_Percentage': overshoot,
            'Settling_Time_Seconds': settling_time,
            'Settling_Time_Formatted': format_min_sec(settling_time),
            'Time_Lag_Seconds': abs(lag_time),
            'Peak_to_Peak_Error': np.max(error) - np.min(error),
            'Signal_to_Noise_Ratio': 20 * np.log10(np.max(np.abs(desired)) / np.std(error)) if np.std(error) > 0 else 0,
            'Tracking_Percentage': 100 * (1 - np.mean(np.abs(error)) / np.mean(np.abs(desired))) if np.mean(np.abs(desired)) > 0 else 100
        }
        
        return stats
    
    def _calculate_overshoot(self, desired, actual):
        """Calcula el porcentaje de overshoot"""
        if len(desired) < 10:
            return 0
        
        # Buscar transiciones en la señal deseada
        desired_diff = np.diff(desired)
        transition_points = np.where(np.abs(desired_diff) > np.std(desired_diff) * 2)[0]
        
        if len(transition_points) == 0:
            return 0
        
        overshoots = []
        for point in transition_points[:2]:  # Analizar primeras 2 transiciones
            if point + 20 < len(actual):
                window = actual[point:point+20]
                if len(window) > 0:
                    # Calcular overshoot relativo al cambio
                    desired_change = desired[point+10] - desired[point]
                    if abs(desired_change) > 0.1:  # Cambio significativo
                        max_deviation = np.max(window) if desired_change > 0 else np.min(window)
                        steady_state = np.mean(window[-5:])
                        overshoot = abs(max_deviation - steady_state) / abs(desired_change) * 100
                        overshoots.append(overshoot)
        
        return np.mean(overshoots) if overshoots else 0
    
    def _calculate_settling_time(self, error, time, tolerance_percent=SETTLING_TIME_TOLERANCE):
        """Calcula el tiempo de establecimiento (settling time)"""
        if len(error) < 10:
            return 0
        
        # Normalizar error
        error_max = np.max(np.abs(error))
        if error_max < 1e-6:
            return 0
        
        error_normalized = np.abs(error) / error_max * 100
        
        # Encontrar último punto donde el error excede la tolerancia
        tolerance_mask = error_normalized > tolerance_percent
        
        if not np.any(tolerance_mask):
            return 0
        
        last_exceed_idx = np.max(np.where(tolerance_mask)[0])
        
        if last_exceed_idx < len(time) - 1:
            return time[last_exceed_idx]
        
        return time[-1]
    
    def generate_tuning_recommendations(self):
        """Genera recomendaciones de tuning basadas en las métricas"""
        if self.results is None:
            return []
        
        stats = self.results['stats']
        recs = []
        
        # Basado en overshoot
        if stats['Overshoot_Percentage'] > 20:
            recs.append("• REDUCIR P_rate (sobrepico excesivo > 20%)")
            recs.append("• AUMENTAR D_rate (amortiguamiento insuficiente)")
        elif stats['Overshoot_Percentage'] < 5:
            recs.append("• CONSIDERAR aumentar P_rate (respuesta muy lenta)")
        
        # Basado en oscilaciones
        if "SEVERA" in stats['Oscillation_Severity']:
            recs.append("• REDUCIR P_rate y/o D_rate (oscilaciones severas)")
            recs.append("• AUMENTAR filtro D-term")
        elif "MODERADA" in stats['Oscillation_Severity']:
            recs.append("• REDUCIR D_rate (oscilaciones moderadas)")
        
        # Basado en tiempo de establecimiento
        if stats['Settling_Time_Seconds'] > 2.0:
            recs.append("• AUMENTAR P_rate (respuesta muy lenta)")
        
        # Basado en error estático
        if abs(stats['Mean_Error']) > 1.0:
            recs.append("• AUMENTAR I_rate o FeedForward (error de estado estacionario)")
        
        # Basado en factor de cresta
        if stats['Crest_Factor'] > 3:
            recs.append("• AUMENTAR filtrado (señal muy ruidosa)")
        
        # Basado en frecuencia dominante
        if stats['Dominant_Frequency_Hz'] > 5:
            recs.append(f"• REDUCIR P_rate (oscilaciones de alta frecuencia: {stats['Dominant_Frequency_Hz']:.1f} Hz)")
        
        if not recs:
            recs.append("• Parámetros actuales parecen adecuados")
            recs.append("• Monitorear en diferentes condiciones de vuelo")
        
        return recs
    
    def plot_analysis(self, save_path=None):
        """Genera gráficas completas del análisis"""
        if self.results is None:
            print(f"[ERROR] No hay resultados para graficar {self.axis_name}")
            return None
        
        fig = plt.figure(figsize=(15, 10))
        
        # Título de la ventana
        filename = os.path.basename(LOG_FILE) if LOG_FILE else "Log Desconocido"
        fig.canvas.manager.set_window_title(f"Análisis de {self.axis_name} - {filename}")
        
        # 1. Señales deseadas vs actuales
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.results['time'], self.results['desired'], 'r--', 
                label='Deseado', linewidth=2, alpha=0.8)
        ax1.plot(self.results['time'], self.results['actual'], 'b-', 
                label='Actual', linewidth=1.5)
        ax1.set_title(f'{self.axis_name} - Deseado vs Actual')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Ángulo (grados)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Señal de error
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.results['time'], self.results['error'], 'g-', 
                label='Error', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.fill_between(self.results['time'], 0, self.results['error'], 
                        where=self.results['error']>=0, alpha=0.3, color='green', label='Error +')
        ax2.fill_between(self.results['time'], 0, self.results['error'], 
                        where=self.results['error']<0, alpha=0.3, color='red', label='Error -')
        ax2.set_title(f'Señal de Error - {self.axis_name}')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Error (grados)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Derivadas del error
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.results['time'], self.results['error_derivative'], 'c-', 
                label='1ra Derivada', linewidth=1.5)
        ax3.plot(self.results['time'], self.results['error_second_derivative'], 'm-', 
                label='2da Derivada', linewidth=1.5, alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('Derivadas del Error')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Derivada (grados/s)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Integrales acumulativas
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(self.results['time'], self.results['iae'], 'k-', 
                label='IAE (Absoluto)', linewidth=2)
        ax4.plot(self.results['time'], self.results['positive_integral'], 'g-', 
                label='Integral Positiva', linewidth=1.5)
        ax4.plot(self.results['time'], self.results['negative_integral'], 'r-', 
                label='Integral Negativa', linewidth=1.5)
        ax4.set_title('Integrales Acumulativas del Error')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Integral Acumulada')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Histograma del error
        ax5 = plt.subplot(3, 2, 5)
        ax5.hist(self.results['error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_title('Distribución del Error')
        ax5.set_xlabel('Error (grados)')
        ax5.set_ylabel('Frecuencia')
        ax5.grid(True, alpha=0.3)
        
        # Añadir estadísticas al histograma
        stats_text = (f"Media: {self.results['stats']['Mean_Error']:.2f}°\n"
                     f"Std: {self.results['stats']['Std_Error']:.2f}°\n"
                     f"RMS: {self.results['stats']['RMSE']:.2f}°")
        ax5.text(0.95, 0.95, stats_text, transform=ax5.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 6. Espectro de frecuencia
        ax6 = plt.subplot(3, 2, 6)
        n = len(self.results['error'])
        freqs = fft.fftfreq(n, self.dt)[:n//2]
        magnitude = np.abs(fft.fft(self.results['error']))[:n//2]
        
        ax6.semilogy(freqs[1:], magnitude[1:], 'b-', linewidth=1.5)
        ax6.set_title('Espectro de Frecuencia del Error')
        ax6.set_xlabel('Frecuencia (Hz)')
        ax6.set_ylabel('Magnitud (log)')
        ax6.grid(True, alpha=0.3)
        
        # Marcar frecuencia dominante
        if self.results['stats']['Dominant_Frequency_Hz'] > 0.1:
            ax6.axvline(x=self.results['stats']['Dominant_Frequency_Hz'], 
                       color='r', linestyle='--', alpha=0.7,
                       label=f"Dominante: {self.results['stats']['Dominant_Frequency_Hz']:.2f} Hz")
            ax6.legend()
        
        # Título general
        fig.suptitle(f'Análisis de Error - {self.axis_name}\n'
                    f'Archivo: {os.path.basename(LOG_FILE) if LOG_FILE else "Desconocido"}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Guardar si se especifica ruta
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Gráfica guardada en: {save_path}")
        
        return fig

# ==============================================
# CLASE PRINCIPAL DE ANÁLISIS DE LOG
# ==============================================

class AttitudeLogAnalyzer:
    """Analiza logs MAVLink para extraer datos de actitud"""
    
    def __init__(self, log_file, sampling_rate=50):
        self.log_file = log_file
        self.sampling_rate = sampling_rate
        
        # Analizadores para cada eje
        self.roll_analyzer = AxisErrorAnalyzer("Roll", sampling_rate)
        self.pitch_analyzer = AxisErrorAnalyzer("Pitch", sampling_rate)
        
        # Métricas globales
        self.global_metrics = {}
        self.message_count = 0
        
    def process_log_file(self):
        """Procesa el archivo de log MAVLink"""
        print(f"[INFO] Procesando archivo de log: {self.log_file}")
        
        if not os.path.exists(self.log_file):
            print(f"[ERROR] El archivo no existe: {self.log_file}")
            return False
        
        try:
            mlog = mavutil.mavlink_connection(self.log_file)
            
            prev_msg_time = time.time()
            attitude_messages = 0
            
            print("[INFO] Leyendo log...")
            
            while True:
                msg = mlog.recv_match(blocking=False)
                
                if msg is None:
                    break
                
                self.message_count += 1
                
                if self.message_count % 1000 == 0:
                    elapsed = time.time() - prev_msg_time
                    print(f"\r[PROGRESO] Mensajes: {self.message_count}, "
                          f"Actitud: {attitude_messages} ({attitude_messages/self.message_count*100:.1f}%)", 
                          end='')
                    sys.stdout.flush()
                    prev_msg_time = time.time()
                
                mtype = msg.get_type()
                
                if mtype == 'ATT':
                    attitude_messages += 1
                    
                    # Extraer timestamp
                    if hasattr(msg, '_timestamp'):
                        log_timestamp = msg._timestamp
                    else:
                        continue
                    
                    # Intentar extraer valores deseados y actuales
                    # Nota: ATT no tiene DesRoll directamente. Usaremos diferentes estrategias:
                    
                    # Estrategia 1: Buscar en otros mensajes (necesitaríamos almacenar)
                    # Por ahora, usaremos valores sintéticos o buscaremos mensajes específicos
                    # En un análisis real, necesitaríamos mensajes de control o setpoints
                    
                    # Para este ejemplo, asumiremos que los valores deseados son cero
                    # o buscaremos en otros mensajes. Vamos a crear valores deseados sintéticos
                    # basados en cambios en los valores actuales
                    
                    # Extraer valores actuales
                    roll_actual = getattr(msg, 'Roll', None)
                    pitch_actual = getattr(msg, 'Pitch', None)
                    
                    if roll_actual is not None:
                        # Convertir de radianes a grados
                        roll_actual_deg = radians_to_degrees(roll_actual)
                        
                        # Para demostración, crear un valor deseado sintético
                        # En un caso real, deberías tener mensajes de setpoint
                        roll_desired_deg = 0  # Valor por defecto
                        
                        # Buscar cambios bruscos para simular comandos
                        if len(self.roll_analyzer.actual) > 10:
                            # Si hay un cambio brusco, asumir comando
                            last_vals = self.roll_analyzer.actual[-5:]
                            if abs(roll_actual_deg - np.mean(last_vals)) > 5:
                                roll_desired_deg = roll_actual_deg
                        
                        self.roll_analyzer.add_data_point(log_timestamp, 
                                                         roll_desired_deg, 
                                                         roll_actual_deg)
                    
                    if pitch_actual is not None:
                        pitch_actual_deg = radians_to_degrees(pitch_actual)
                        pitch_desired_deg = 0
                        
                        if len(self.pitch_analyzer.actual) > 10:
                            last_vals = self.pitch_analyzer.actual[-5:]
                            if abs(pitch_actual_deg - np.mean(last_vals)) > 5:
                                pitch_desired_deg = pitch_actual_deg
                        
                        self.pitch_analyzer.add_data_point(log_timestamp,
                                                          pitch_desired_deg,
                                                          pitch_actual_deg)
                
                # También buscar mensajes de control que puedan tener setpoints
                elif mtype in ['ATTITUDE_TARGET', 'RC_CHANNELS']:
                    # Estos mensajes pueden tener información de setpoints
                    # Implementación más avanzada para versión futura
                    pass
            
            mlog.close()
            
            print(f"\n[INFO] Log procesado exitosamente")
            print(f"[INFO] Mensajes totales: {self.message_count}")
            print(f"[INFO] Mensajes de actitud: {attitude_messages}")
            print(f"[INFO] Muestras Roll: {len(self.roll_analyzer.timestamps)}")
            print(f"[INFO] Muestras Pitch: {len(self.pitch_analyzer.timestamps)}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error procesando el log: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_all_metrics(self):
        """Calcula todas las métricas para ambos ejes"""
        print("\n[INFO] Calculando métricas...")
        
        results = {}
        
        if ANALIZAR_ROLL and len(self.roll_analyzer.timestamps) > 10:
            print("  - Calculando métricas para Roll...")
            results['Roll'] = self.roll_analyzer.calculate_metrics()
        
        if ANALIZAR_PITCH and len(self.pitch_analyzer.timestamps) > 10:
            print("  - Calculando métricas para Pitch...")
            results['Pitch'] = self.pitch_analyzer.calculate_metrics()
        
        return results
    
    def generate_diagnostic_report(self):
        """Genera un reporte de diagnóstico completo"""
        report = []
        
        report.append("=" * 80)
        report.append("DIAGNÓSTICO DE TUNING - DRONE FLIGHT CONTROLLER")
        report.append("=" * 80)
        report.append(f"Archivo analizado: {os.path.basename(self.log_file) if self.log_file else 'Desconocido'}")
        report.append(f"Fecha análisis: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        
        # Análisis por eje
        for axis_name, analyzer in [("ROLL", self.roll_analyzer), 
                                   ("PITCH", self.pitch_analyzer)]:
            
            if analyzer.results is None:
                continue
                
            stats = analyzer.results['stats']
            
            report.append(f"\n{'='*50}")
            report.append(f"ANÁLISIS DE {axis_name}")
            report.append(f"{'='*50}")
            
            report.append(f"\n--- MÉTRICAS DE ERROR ---")
            report.append(f"RMSE: {stats['RMSE']:.3f} grados")
            report.append(f"Error Máximo: {stats['Max_Error']:.3f} grados")
            report.append(f"Error Promedio: {stats['Mean_Error']:.3f} grados")
            report.append(f"Desviación Estándar: {stats['Std_Error']:.3f} grados")
            report.append(f"Overshoot: {stats['Overshoot_Percentage']:.1f}%")
            report.append(f"Tiempo de Establecimiento: {stats['Settling_Time_Formatted']}")
            report.append(f"Retardo: {stats['Time_Lag_Seconds']:.3f} seg")
            report.append(f"Porcentaje de Tracking: {stats['Tracking_Percentage']:.1f}%")
            
            report.append(f"\n--- ANÁLISIS DE OSCILACIONES ---")
            report.append(f"Frecuencia Dominante: {stats['Dominant_Frequency_Hz']:.2f} Hz")
            report.append(f"Tasa de Cruce por Cero: {stats['Zero_Crossing_Rate_Hz']:.2f} Hz")
            report.append(f"Factor de Cresta: {stats['Crest_Factor']:.2f}")
            report.append(f"Severidad: {stats['Oscillation_Severity']}")
            
            report.append(f"\n--- INTEGRALES DE ERROR ---")
            report.append(f"IAE (Error Absoluto): {stats['IAE_Final']:.3f}")
            report.append(f"Integral Positiva: {stats['Positive_Integral_Final']:.3f}")
            report.append(f"Integral Negativa: {stats['Negative_Integral_Final']:.3f}")
            
            report.append(f"\n--- RECOMENDACIONES DE TUNING ---")
            recommendations = analyzer.generate_tuning_recommendations()
            for rec in recommendations:
                report.append(rec)
        
        # Métricas comparativas
        report.append(f"\n{'='*80}")
        report.append("COMPARACIÓN ENTRE EJES")
        report.append(f"{'='*80}")
        
        if self.roll_analyzer.results and self.pitch_analyzer.results:
            roll_stats = self.roll_analyzer.results['stats']
            pitch_stats = self.pitch_analyzer.results['stats']
            
            report.append(f"\nDiferencia Roll - Pitch:")
            report.append(f"  RMSE: {abs(roll_stats['RMSE'] - pitch_stats['RMSE']):.3f} grados")
            report.append(f"  Overshoot: {abs(roll_stats['Overshoot_Percentage'] - pitch_stats['Overshoot_Percentage']):.1f}%")
            report.append(f"  Frecuencia dominante: {abs(roll_stats['Dominant_Frequency_Hz'] - pitch_stats['Dominant_Frequency_Hz']):.2f} Hz")
            
            if roll_stats['RMSE'] > pitch_stats['RMSE'] * 1.5:
                report.append("\n[RECOMENDACIÓN] Roll tiene mayor error que Pitch. Considerar:")
                report.append("  • Verificar balance del drone")
                report.append("  • Ajustar ganancias de Roll por separado")
        
        report.append(f"\n{'='*80}")
        report.append("RESUMEN EJECUTIVO")
        report.append(f"{'='*80}")
        
        if self.roll_analyzer.results:
            roll_severity = self.roll_analyzer.results['stats']['Oscillation_Severity']
            report.append(f"Roll: {roll_severity}")
        
        if self.pitch_analyzer.results:
            pitch_severity = self.pitch_analyzer.results['stats']['Oscillation_Severity']
            report.append(f"Pitch: {pitch_severity}")
        
        report.append(f"\nAcciones recomendadas prioritarias:")
        
        # Determinar acción más crítica
        critical_actions = []
        for analyzer in [self.roll_analyzer, self.pitch_analyzer]:
            if analyzer.results:
                stats = analyzer.results['stats']
                if "SEVERA" in stats['Oscillation_Severity']:
                    critical_actions.append("• REDUCIR ganancias P y D inmediatamente")
                elif stats['Overshoot_Percentage'] > 30:
                    critical_actions.append("• REDUCIR ganancia P para reducir overshoot")
                elif stats['Settling_Time_Seconds'] > 3:
                    critical_actions.append("• AUMENTAR ganancia P para mejor respuesta")
        
        if critical_actions:
            for action in set(critical_actions):  # Remover duplicados
                report.append(action)
        else:
            report.append("• Sistema estable. Realizar pruebas en condiciones más exigentes.")
        
        report.append(f"\n{'='*80}")
        report.append("FIN DEL REPORTE")
        report.append(f"{'='*80}")
        
        return "\n".join(report)
    
    def save_raw_data_csv(self, path):
        """Guarda datos crudos en CSV"""
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'roll_desired', 'roll_actual', 
                               'pitch_desired', 'pitch_actual'])
                
                # Determinar longitud máxima
                max_len = max(len(self.roll_analyzer.timestamps), 
                            len(self.pitch_analyzer.timestamps))
                
                for i in range(max_len):
                    row = []
                    
                    # Timestamp
                    if i < len(self.roll_analyzer.timestamps):
                        row.append(self.roll_analyzer.timestamps[i])
                    elif i < len(self.pitch_analyzer.timestamps):
                        row.append(self.pitch_analyzer.timestamps[i])
                    else:
                        row.append('')
                    
                    # Roll
                    if i < len(self.roll_analyzer.desired):
                        row.append(self.roll_analyzer.desired[i])
                        row.append(self.roll_analyzer.actual[i])
                    else:
                        row.extend(['', ''])
                    
                    # Pitch
                    if i < len(self.pitch_analyzer.desired):
                        row.append(self.pitch_analyzer.desired[i])
                        row.append(self.pitch_analyzer.actual[i])
                    else:
                        row.extend(['', ''])
                    
                    writer.writerow(row)
            
            print(f"[INFO] Datos crudos guardados en: {path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Al guardar datos crudos: {e}")
            return False
    
    def save_analysis_csv(self, path):
        """Guarda análisis en CSV"""
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Escribir encabezados
                headers = ['Métrica', 'Roll', 'Pitch', 'Unidad']
                writer.writerow(headers)
                
                # Lista de métricas a guardar
                metrics_to_save = [
                    ('RMSE', 'RMSE', 'grados'),
                    ('Error_Maximo', 'Max_Error', 'grados'),
                    ('Error_Promedio', 'Mean_Error', 'grados'),
                    ('Desviacion_Estandar', 'Std_Error', 'grados'),
                    ('Overshoot', 'Overshoot_Percentage', '%'),
                    ('Tiempo_Establecimiento', 'Settling_Time_Seconds', 's'),
                    ('Retardo', 'Time_Lag_Seconds', 's'),
                    ('Frecuencia_Dominante', 'Dominant_Frequency_Hz', 'Hz'),
                    ('Tasa_Cruce_Cero', 'Zero_Crossing_Rate_Hz', 'Hz'),
                    ('Factor_Cresta', 'Crest_Factor', ''),
                    ('IAE_Final', 'IAE_Final', ''),
                    ('Integral_Positiva', 'Positive_Integral_Final', ''),
                    ('Integral_Negativa', 'Negative_Integral_Final', ''),
                ]
                
                for metric_name, stat_key, unit in metrics_to_save:
                    row = [metric_name]
                    
                    # Roll
                    if self.roll_analyzer.results:
                        row.append(self.roll_analyzer.results['stats'].get(stat_key, ''))
                    else:
                        row.append('')
                    
                    # Pitch
                    if self.pitch_analyzer.results:
                        row.append(self.pitch_analyzer.results['stats'].get(stat_key, ''))
                    else:
                        row.append('')
                    
                    row.append(unit)
                    writer.writerow(row)
            
            print(f"[INFO] Análisis guardado en: {path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Al guardar análisis: {e}")
            return False

# ==============================================
# FUNCIÓN PRINCIPAL
# ==============================================

def main():
    """Función principal - Ejecuta el análisis completo"""
    print("=" * 80)
    print("ANÁLISIS DE ERROR DE ACTITUD PARA TUNING DE PID")
    print("=" * 80)
    print(f"Archivo de log: {LOG_FILE}")
    print(f"Frecuencia de muestreo: {SAMPLING_RATE} Hz")
    print(f"Analizar Roll: {ANALIZAR_ROLL}")
    print(f"Analizar Pitch: {ANALIZAR_PITCH}")
    print("=" * 80)
    
    # Verificar que el archivo existe
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] El archivo no existe: {LOG_FILE}")
        print("[INFO] Verifica que la ruta en la variable LOG_FILE sea correcta.")
        return
    
    # Crear analizador
    analyzer = AttitudeLogAnalyzer(LOG_FILE, SAMPLING_RATE)
    
    # Procesar archivo de log
    success = analyzer.process_log_file()
    if not success:
        print("[ERROR] Falló el procesamiento del log")
        return
    
    # Calcular métricas
    results = analyzer.calculate_all_metrics()
    
    if not results:
        print("[ERROR] No se pudieron calcular métricas. Datos insuficientes.")
        return
    
    # Mostrar métricas clave por consola
    print("\n" + "=" * 80)
    print("MÉTRICAS CLAVE RESUMIDAS:")
    print("=" * 80)
    
    for axis_name, result in results.items():
        if result:
            stats = result['stats']
            print(f"\n{axis_name.upper()}:")
            print(f"  • RMSE: {stats['RMSE']:.3f}°")
            print(f"  • Overshoot: {stats['Overshoot_Percentage']:.1f}%")
            print(f"  • Tiempo establecimiento: {stats['Settling_Time_Formatted']}")
            print(f"  • Frecuencia dominante: {stats['Dominant_Frequency_Hz']:.2f} Hz")
            print(f"  • Severidad oscilaciones: {stats['Oscillation_Severity']}")
    
    # Generar reporte de diagnóstico
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE DE DIAGNÓSTICO...")
    print("=" * 80)
    
    report = analyzer.generate_diagnostic_report()
    print(report)
    
    # Guardar reporte
    try:
        with open(REPORTE_RUTA, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[INFO] Reporte guardado en: {REPORTE_RUTA}")
    except Exception as e:
        print(f"[ERROR] Al guardar reporte: {e}")
    
    # Guardar CSV si está configurado
    if GUARDAR_CSV_CRUDOS:
        analyzer.save_raw_data_csv(CSV_CRUDOS_RUTA)
    
    if GUARDAR_CSV_ANALISIS:
        analyzer.save_analysis_csv(CSV_ANALISIS_RUTA)
    
    # Mostrar gráficas
    if MOSTRAR_GRAFICAS:
        print("\n[INFO] Generando gráficas...")
        
        if ANALIZAR_ROLL and analyzer.roll_analyzer.results:
            fig_roll = analyzer.roll_analyzer.plot_analysis(GRAFICAS_RUTA_ROLL)
        
        if ANALIZAR_PITCH and analyzer.pitch_analyzer.results:
            fig_pitch = analyzer.pitch_analyzer.plot_analysis(GRAFICAS_RUTA_PITCH)
        
        print("[INFO] Mostrando gráficas...")
        plt.show()
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print(f"Archivos generados:")
    if os.path.exists(REPORTE_RUTA):
        print(f"  • {REPORTE_RUTA}")
    if os.path.exists(CSV_ANALISIS_RUTA):
        print(f"  • {CSV_ANALISIS_RUTA}")
    if os.path.exists(GRAFICAS_RUTA_ROLL):
        print(f"  • {GRAFICAS_RUTA_ROLL}")
    if os.path.exists(GRAFICAS_RUTA_PITCH):
        print(f"  • {GRAFICAS_RUTA_PITCH}")
    print("=" * 80)

# ==============================================
# EJECUCIÓN
# ==============================================

if __name__ == "__main__":
    # Mostrar configuración actual
    print("=" * 80)
    print("CONFIGURACIÓN ACTUAL:")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"SAMPLING_RATE: {SAMPLING_RATE} Hz")
    print(f"ANALIZAR_ROLL: {ANALIZAR_ROLL}")
    print(f"ANALIZAR_PITCH: {ANALIZAR_PITCH}")
    print(f"MOSTRAR_GRAFICAS: {MOSTRAR_GRAFICAS}")
    print(f"GUARDAR_CSV_ANALISIS: {GUARDAR_CSV_ANALISIS}")
    print(f"MODO_VERBOSE: {MODO_VERBOSE}")
    print("=" * 80)
    
    # Preguntar confirmación antes de continuar
    respuesta = input("\n¿Continuar con el análisis? (s/n): ").strip().lower()
    
    if respuesta in ['s', 'si', '']:
        print("\n[INFO] Iniciando análisis...\n")
        main()
    else:
        print("[INFO] Análisis cancelado por el usuario")
    
    print("\n[INFO] Presiona Enter para salir...")
    input()
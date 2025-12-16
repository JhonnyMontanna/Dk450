from pymavlink import mavutil
import time
import sys
import threading
from collections import defaultdict

class PacketMonitor:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.master = None
        self.running = False
        self.counters = defaultdict(int)  # Contadores por tipo de mensaje
        self.total_counters = defaultdict(int)  # Contadores totales desde inicio
        self.last_print_time = time.time()
        
        # Categorías de mensajes
        self.categories = {
            'attitude': ['ATTITUDE', 'ATTITUDE_QUATERNION'],
            'position': [
                'GLOBAL_POSITION_INT', 
                'GPS_RAW_INT', 
                'GPS2_RAW', 
                'LOCAL_POSITION_NED',
                'VFR_HUD'
            ],
            'sensor': [
                'RANGEFINDER',
                'DISTANCE_SENSOR',
                'SCALED_IMU',
                'SCALED_PRESSURE',
                'RAW_IMU'
            ]
        }
        
        # Invertir el diccionario para búsqueda rápida
        self.message_to_category = {}
        for category, messages in self.categories.items():
            for msg in messages:
                self.message_to_category[msg] = category
    
    def connect(self):
        """Establecer conexión con el dron"""
        try:
            print(f"Conectando a {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string)
            self.master.wait_heartbeat(timeout=5)
            print(f"✓ Conexión establecida con el dron")
            print(f"  Sistema: {self.master.target_system}, Componente: {self.master.target_component}")
            
            # Solicitar streams de datos
            self.request_data_streams()
            return True
            
        except Exception as e:
            print(f"✗ Error al conectar: {e}")
            return False
    
    def request_data_streams(self):
        """Solicitar streams de datos específicos"""
        try:
            # Request all data streams (reduced rate)
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # Hz
                1    # Enable
            )
            print("✓ Streams de datos solicitados")
        except Exception as e:
            print(f"✗ Error solicitando streams: {e}")
    
    def get_message_type(self, msg):
        """Obtener el tipo de mensaje"""
        return msg.get_type() if hasattr(msg, 'get_type') else None
    
    def get_message_category(self, msg_type):
        """Determinar la categoría del mensaje"""
        if msg_type in self.message_to_category:
            return self.message_to_category[msg_type]
        return 'other'
    
    def calculate_rates(self):
        """Calcular tasas de recepción por segundo"""
        current_time = time.time()
        elapsed = current_time - self.last_print_time
        
        if elapsed >= 1.0:  # Mostrar cada segundo
            rates = {}
            for category in ['attitude', 'position', 'sensor', 'other']:
                rate = self.counters[category] / elapsed if elapsed > 0 else 0
                rates[category] = rate
            
            # Calcular tasa total
            total_rate = sum(rates.values())
            
            # Imprimir resultados
            self.print_rates(rates, total_rate, elapsed)
            
            # Reiniciar contadores del segundo
            self.counters.clear()
            self.last_print_time = current_time
    
    def print_rates(self, rates, total_rate, elapsed):
        """Imprimir las tasas de recepción formateadas"""
        # Limpiar línea anterior
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        
        # Imprimir encabezado cada 10 segundos
        current_time_seconds = int(time.time())
        if current_time_seconds % 10 == 0:
            print("\n" + "="*80)
            print("MONITOR DE VELOCIDAD DE PAQUETES MAVLINK")
            print("="*80)
            print(f"{'TIPO':<15} {'PAQ/SEG':<10} {'TOTAL':<12} {'%':<8}")
            print("-"*45)
        
        # Imprimir tasas
        categories_order = ['attitude', 'position', 'sensor', 'other']
        for category in categories_order:
            rate = rates[category]
            total = self.total_counters[category]
            percentage = (rate / total_rate * 100) if total_rate > 0 else 0
            
            # Nombre en español
            names = {
                'attitude': 'Actitud',
                'position': 'Posición',
                'sensor': 'Sensores',
                'other': 'Otros'
            }
            
            print(f"{names[category]:<15} {rate:<10.1f} {total:<12} {percentage:<8.1f}%")
        
        # Imprimir total
        print("-"*45)
        print(f"{'TOTAL':<15} {total_rate:<10.1f} {sum(self.total_counters.values()):<12} 100%")
        print(f"Tiempo transcurrido: {time.strftime('%H:%M:%S', time.gmtime(time.time() - self.start_time))}")
        
        # Mover cursor para sobreescribir
        sys.stdout.write("\033[7A")  # Subir 7 líneas
    
    def start_monitoring(self):
        """Iniciar monitoreo"""
        if not self.connect():
            return
        
        self.running = True
        self.start_time = time.time()
        print("\nIniciando monitoreo de paquetes... (Ctrl+C para detener)\n")
        
        # Imprimir encabezado inicial
        print("\n" + "="*80)
        print("MONITOR DE VELOCIDAD DE PAQUETES MAVLINK")
        print("="*80)
        print(f"{'TIPO':<15} {'PAQ/SEG':<10} {'TOTAL':<12} {'%':<8}")
        print("-"*45)
        for _ in range(6):  # Espacio para las 6 líneas de datos
            print()
        
        try:
            while self.running:
                # Leer mensaje con timeout corto
                msg = self.master.recv_match(blocking=True, timeout=0.1)
                
                if msg:
                    msg_type = self.get_message_type(msg)
                    if msg_type:
                        category = self.get_message_category(msg_type)
                        self.counters[category] += 1
                        self.total_counters[category] += 1
                
                # Calcular y mostrar tasas
                self.calculate_rates()
                
        except KeyboardInterrupt:
            print("\n\n✗ Monitoreo interrumpido por usuario")
        except Exception as e:
            print(f"\n✗ Error durante el monitoreo: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Detener monitoreo"""
        self.running = False
        if self.master:
            self.master.close()
        
        # Mostrar estadísticas finales
        total_time = time.time() - self.start_time
        print("\n" + "="*80)
        print("ESTADÍSTICAS FINALES")
        print("="*80)
        print(f"Tiempo total de monitoreo: {total_time:.1f} segundos")
        print(f"Tiempo total formateado: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print()
        
        categories_order = ['attitude', 'position', 'sensor', 'other']
        names = {
            'attitude': 'Actitud',
            'position': 'Posición',
            'sensor': 'Sensores',
            'other': 'Otros'
        }
        
        total_packets = sum(self.total_counters.values())
        print(f"{'TIPO':<15} {'TOTAL':<12} {'PAQ/SEG':<10} {'%':<8}")
        print("-"*45)
        
        for category in categories_order:
            total = self.total_counters[category]
            rate = total / total_time if total_time > 0 else 0
            percentage = (total / total_packets * 100) if total_packets > 0 else 0
            print(f"{names[category]:<15} {total:<12} {rate:<10.1f} {percentage:<8.1f}%")
        
        print("-"*45)
        print(f"{'TOTAL':<15} {total_packets:<12} {total_packets/total_time if total_time>0 else 0:<10.1f} 100%")
        print("="*80)

def main():
    """Función principal"""
    # Opciones de conexión (descomenta la que necesites)
    connection_options = {
        '1': 'udp:127.0.0.1:14550',
        '2': 'udp:127.0.0.1:14551',
        '3': 'udp:127.0.0.1:14552',
        '4': 'COM3:57600',
        '5': 'COM6:9600'
    }
    
    print("="*80)
    print("MONITOR DE VELOCIDAD DE PAQUETES MAVLINK")
    print("="*80)
    print("\nSelecciona el tipo de conexión:")
    for key, value in connection_options.items():
        print(f"  {key}. {value}")
    
    choice = input("\nTu selección (1-5, por defecto 3): ").strip()
    
    if choice in connection_options:
        connection_string = connection_options[choice]
    else:
        connection_string = connection_options['3']  # Por defecto
    
    print(f"\nUsando conexión: {connection_string}")
    
    # Crear y ejecutar monitor
    monitor = PacketMonitor(connection_string)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()
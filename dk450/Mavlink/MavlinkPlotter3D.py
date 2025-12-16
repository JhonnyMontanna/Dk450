from pymavlink import mavutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

style.use('ggplot')

# Lista para almacenar puntos (lat, lon, alt)
trajectory = []

# Configurar conexión MAVLink
connection_string = 'COM3'
baudrate = 57600
#master = mavutil.mavlink_connection('udp:127.0.0.1:14551')
master = mavutil.mavlink_connection('udp:127.0.0.1:14552')
#master = mavutil.mavlink_connection(connection_string, baud=baudrate)
master.wait_heartbeat()
print("¡Conexión establecida con el dron!")

# Configurar gráfico 3D con líneas
fig = plt.figure("Trayectoria 3D del Dron")
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Trayectoria 3D del Dron')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_zlabel('Altura (m)')
line, = ax.plot([], [], [], 'bo-', markersize=5, linestyle='-')  # Línea conectada

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    global trajectory
    
    # Leer mensajes
    while True:
        msg = master.recv_match(blocking=False)
        if not msg:
            break
            
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = None  # Esperar dato de altura
            
        elif msg.get_type() == 'RANGEFINDER':
            alt = msg.distance
            if 'lat' in locals() and 'lon' in locals() and alt is not None:
                trajectory.append((lat, lon, alt))
                print(f"Punto agregado: Lat {lat:.6f}°, Lon {lon:.6f}°, Altura: {alt:.2f}m")
    
    # Actualizar línea
    if trajectory:
        lats, lons, alts = zip(*trajectory)
        line.set_data(lons, lats)
        line.set_3d_properties(alts)
        
        # Ajustar límites dinámicos
        ax.set_xlim(min(lons), max(lons))
        ax.set_ylim(min(lats), max(lats))
        ax.set_zlim(min(alts), max(alts))
    
    return line,

# Configurar animación
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    init_func=init,
    interval=1000,  # Ajusta la velocidad aquí
    blit=False
)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nMonitor detenido por el usuario")
    master.close()
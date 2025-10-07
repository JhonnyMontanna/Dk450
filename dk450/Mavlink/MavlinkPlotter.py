from pymavlink import mavutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import math
import numpy as np

# --------------------------- CONFIGURACIÓN RTK LOCAL --------------------------- #
lat0 = math.radians(19.5951429)
lon0 = math.radians(-99.2278926)
h0 = 2333.045
lat_ref = math.radians(19.595135)
lon_ref = math.radians(-99.227897)
h_ref = h0
expected_local = np.array([0, 1])

def geodetic_to_ecef(lat, lon, h):
    a = 6378137.0
    e2 = 0.00669437999014
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    X = (N + h) * math.cos(lat) * math.cos(lon)
    Y = (N + h) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - e2) + h) * math.sin(lat)
    return np.array([X, Y, Z])

def get_rotation_matrix(lat, lon):
    return np.array([
        [-math.sin(lon),               math.cos(lon),              0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [math.cos(lat)*math.cos(lon),  math.cos(lat)*math.sin(lon),  math.sin(lat)]
    ])

def compute_definitive_rotation():
    origin_ecef = geodetic_to_ecef(lat0, lon0, h0)
    ref_ecef = geodetic_to_ecef(lat_ref, lon_ref, h_ref)
    R_enu = get_rotation_matrix(lat0, lon0)
    d_ref = ref_ecef - origin_ecef
    enu_ref = R_enu @ d_ref
    theta_expected = math.atan2(expected_local[1], expected_local[0])
    theta_measured = math.atan2(enu_ref[1], enu_ref[0])
    theta = theta_expected - theta_measured
    return origin_ecef, R_enu, theta

origin_ecef, R_enu, theta = compute_definitive_rotation()
R_correction = np.array([
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta),  math.cos(theta)]
])

def gps_to_local(lat_deg, lon_deg, alt):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    ecef = geodetic_to_ecef(lat, lon, alt)
    delta = ecef - origin_ecef
    enu = R_enu @ delta
    enu_rotated = R_correction @ enu[:2]
    return enu_rotated[0], enu_rotated[1]

# ----------------------- CONFIGURACIÓN DE GRAFICACIÓN ------------------------ #
style.use('ggplot')
lats_local = []
lons_local = []

# Variables de conexión para debug o uso flexible
usar_com = False  # True para usar COM, False para usar UDP
default_com_port = 'COM3'
default_baudrate = 57600
default_udp_port = '127.0.0.1:14560'


if usar_com:
    print(f"Conectando por puerto serie: {default_com_port} @ {default_baudrate} baud")
    master = mavutil.mavlink_connection(default_com_port, baud=default_baudrate)
else:
    print(f"Conectando por UDP en: {default_udp_port}")
    master = mavutil.mavlink_connection(f'udp:{default_udp_port}')

master.wait_heartbeat()
print("\u00a1Conexión establecida con el dron!")

fig = plt.figure("Posición Local del Dron (Sistema RTK)")
ax = fig.add_subplot(111)
ax.set_title('Trayectoria del Dron (Coordenadas Locales)')
ax.set_xlabel('Este (m)')
ax.set_ylabel('Norte (m)')
line, = ax.plot([], [], 'bo-', markersize=5)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global lats_local, lons_local
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if not msg:
            break

        lat = msg.lat / 1e7
        lon = msg.lon / 1e7
        alt = msg.alt / 1e3
        x_local, y_local = gps_to_local(lat, lon, alt)

        lats_local.append(y_local)
        lons_local.append(x_local)

        print(f"Local: X={x_local:.2f} m, Y={y_local:.2f} m")

    if lats_local and lons_local:
        line.set_data(lons_local, lats_local)
        ax.relim()
        ax.autoscale_view()

    return line,

ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    init_func=init,
    interval=500,
    blit=True,
    cache_frame_data=False
)

try:
    plt.show()
except KeyboardInterrupt:
    print("\nMonitor detenido por el usuario")
    master.close()

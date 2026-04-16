#!/usr/bin/env python3
"""
Controlador PID líder-seguidor con prealimentación
====================================================
Implementación directa del planteamiento matemático (sección líder-seguidor).

Versión con conexiones UDP separadas para líder y seguidor.
"""

import math
import time
import csv
import threading
from collections import deque
import matplotlib.pyplot as plt
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE CONEXIONES (MODIFICAR AQUÍ)
# ──────────────────────────────────────────────────────────────────────────────
# Configuración del LÍDER
LEADER_CONFIG = {
    'udp_port': 'udp:127.0.0.1:14552',  # Puerto UDP del líder
    'sysid': 1,                          # System ID del líder
    'compid': 1                          # Component ID del líder
}

# Configuración del SEGUIDOR
FOLLOWER_CONFIG = {
    'udp_port': 'udp:127.0.0.1:14553',  # Puerto UDP del seguidor
    'sysid': 2,                          # System ID del seguidor
    'compid': 1                          # Component ID del seguidor
}

# Offset en coordenadas polares
OFFSET_D     = 2.0          # metros
OFFSET_ALPHA = math.pi      # detrás del líder (π rad = 180°)
OFFSET_DZ    = 0.0          # misma altitud

# Ganancias PID posición
KP   = 0.5
KI   = 0.05
KD   = 0.2

# Ganancias PID yaw
KP_YAW = 0.8
KI_YAW = 0.0
KD_YAW = 0.1

# Anti-windup
INTEGRAL_LIMIT = 2.0        # m·s por eje
INTEGRAL_YAW_LIMIT = 1.0    # rad·s

# Velocidades máximas
V_MAX = 3.0                 # m/s
YAW_RATE_MAX = 1.0          # rad/s

# Frecuencia de control
RATE = 20                   # Hz
DT   = 1.0 / RATE

# Activar gráficas en tiempo real
ENABLE_PLOT = True

# Máscara MAVLink
TYPE_MASK_VEL_YAWRATE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()

_leader = dict(x=None, y=None, z=None,
               vx=None, vy=None, vz=None,
               yaw=None, yaw_rate=None)

_follower = dict(x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 yaw=None, yaw_rate=None)


# ──────────────────────────────────────────────────────────────────────────────
# HILOS LECTORES SEPARADOS (uno por dron)
# ──────────────────────────────────────────────────────────────────────────────
def _leader_reader(stop_event):
    """
    Hilo dedicado a leer datos del LÍDER desde su puerto UDP.
    """
    print(f"📡 Conectando al LÍDER en {LEADER_CONFIG['udp_port']}...")
    leader_master = mavutil.mavlink_connection(LEADER_CONFIG['udp_port'])
    
    # Esperar heartbeat del líder
    try:
        leader_master.wait_heartbeat(timeout=5)
        print(f"✅ Heartbeat del LÍDER recibido (SYSID={leader_master.target_system})")
    except:
        print(f"❌ Error: No se recibió heartbeat del LÍDER en {LEADER_CONFIG['udp_port']}")
        return
    
    while not stop_event.is_set():
        msg = leader_master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        
        mtype = msg.get_type()
        
        with _state_lock:
            if mtype == 'LOCAL_POSITION_NED':
                _leader['x']  = msg.x
                _leader['y']  = msg.y
                _leader['z']  = -msg.z    # altura positiva
                _leader['vx'] = msg.vx
                _leader['vy'] = msg.vy
                _leader['vz'] = -msg.vz   # velocidad vertical positiva
            
            elif mtype == 'ATTITUDE':
                _leader['yaw']      = msg.yaw
                _leader['yaw_rate'] = msg.yawspeed


def _follower_reader(stop_event):
    """
    Hilo dedicado a leer datos del SEGUIDOR desde su puerto UDP.
    """
    print(f"📡 Conectando al SEGUIDOR en {FOLLOWER_CONFIG['udp_port']}...")
    follower_master = mavutil.mavlink_connection(FOLLOWER_CONFIG['udp_port'])
    
    # Esperar heartbeat del seguidor
    try:
        follower_master.wait_heartbeat(timeout=5)
        print(f"✅ Heartbeat del SEGUIDOR recibido (SYSID={follower_master.target_system})")
    except:
        print(f"❌ Error: No se recibió heartbeat del SEGUIDOR en {FOLLOWER_CONFIG['udp_port']}")
        return
    
    while not stop_event.is_set():
        msg = follower_master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue
        
        mtype = msg.get_type()
        
        with _state_lock:
            if mtype == 'LOCAL_POSITION_NED':
                _follower['x']  = msg.x
                _follower['y']  = msg.y
                _follower['z']  = -msg.z
                _follower['vx'] = msg.vx
                _follower['vy'] = msg.vy
                _follower['vz'] = -msg.vz
            
            elif mtype == 'ATTITUDE':
                _follower['yaw']      = msg.yaw
                _follower['yaw_rate'] = msg.yawspeed


def get_states():
    """Devuelve copias thread-safe del estado de ambos drones."""
    with _state_lock:
        L = dict(_leader)
        S = dict(_follower)
    return L, S


def states_ready(L, S):
    """True si tenemos todos los campos necesarios de ambos drones."""
    needed = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'yaw_rate')
    return all(L[k] is not None for k in needed) and \
           all(S[k] is not None for k in needed)


# ──────────────────────────────────────────────────────────────────────────────
# FUNCIONES MATEMÁTICAS
# ──────────────────────────────────────────────────────────────────────────────
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_offset(psi_L):
    angle = psi_L + OFFSET_ALPHA
    dx = OFFSET_D * math.cos(angle)
    dy = OFFSET_D * math.sin(angle)
    dz = OFFSET_DZ
    return dx, dy, dz


def compute_offset_dot(yaw_rate_L, dx, dy):
    ff_x = yaw_rate_L * (-dy)
    ff_y = yaw_rate_L * ( dx)
    ff_z = 0.0
    return ff_x, ff_y, ff_z


def clamp(value, limit):
    return max(-limit, min(limit, value))


# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE COMANDOS AL SEGUIDOR (usando su conexión)
# ──────────────────────────────────────────────────────────────────────────────
class FollowerCommander:
    def __init__(self, config):
        self.config = config
        self.master = None
        self.connect()
    
    def connect(self):
        """Conectar al seguidor para enviar comandos."""
        print(f"🎮 Conectando canal de comandos al SEGUIDOR en {self.config['udp_port']}...")
        self.master = mavutil.mavlink_connection(self.config['udp_port'])
        try:
            self.master.wait_heartbeat(timeout=5)
            print(f"✅ Canal de comandos del SEGUIDOR listo")
        except:
            print(f"⚠️  Advertencia: No se recibió heartbeat, pero se intentará enviar comandos")
    
    def send_velocity_yawrate(self, vx, vy, vz_ned, yaw_rate):
        """
        Envía setpoint de velocidad + yaw_rate al seguidor.
        """
        if self.master is None:
            return
        
        self.master.mav.set_position_target_local_ned_send(
            0, 
            self.config['sysid'], 
            self.config['compid'],
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            TYPE_MASK_VEL_YAWRATE,
            0, 0, 0,
            vx, vy, vz_ned,
            0, 0, 0,
            0, yaw_rate
        )


# ──────────────────────────────────────────────────────────────────────────────
# ESTADO DEL CONTROLADOR PID
# ──────────────────────────────────────────────────────────────────────────────
class PIDState:
    def __init__(self):
        self.integral    = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0


def compute_control(L, S, pid):
    """
    Calcula el comando de velocidad e yaw_rate.
    """
    # Offset polar rotante
    dx, dy, dz = compute_offset(L['yaw'])
    
    # Posición deseada
    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz
    
    # Errores
    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']
    
    # Integrador
    pid.integral[0] = clamp(pid.integral[0] + ex * DT, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * DT, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * DT, INTEGRAL_LIMIT)
    
    # Diferencia de velocidades
    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']
    
    # Prealimentación
    ff_x, ff_y, ff_z = compute_offset_dot(L['yaw_rate'], dx, dy)
    
    # Ley de control
    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z
    
    # Clamp velocidad
    v_horiz = math.hypot(vx, vy)
    if v_horiz > V_MAX:
        vx *= V_MAX / v_horiz
        vy *= V_MAX / v_horiz
    vz = clamp(vz, V_MAX)
    vz_ned = -vz
    
    # Control de yaw
    e_yaw = wrap(L['yaw'] - S['yaw'])
    pid.integral_yaw = clamp(pid.integral_yaw + e_yaw * DT, INTEGRAL_YAW_LIMIT)
    dyaw = L['yaw_rate'] - S['yaw_rate']
    
    yaw_rate_cmd = clamp(
        KP_YAW * e_yaw + KI_YAW * pid.integral_yaw + KD_YAW * dyaw,
        YAW_RATE_MAX
    )
    
    components = dict(
        xd=xd, yd=yd, zd=zd,
        ex=ex, ey=ey, ez=ez, e_yaw=e_yaw,
        ff_x=ff_x, ff_y=ff_y, ff_z=ff_z,
        vx_p=KP*ex, vy_p=KP*ey, vz_p=KP*ez,
        vx_i=KI*pid.integral[0], vy_i=KI*pid.integral[1], vz_i=KI*pid.integral[2],
        vx_d=KD*dv_x, vy_d=KD*dv_y, vz_d=KD*dv_z,
        vx=vx, vy=vy, vz=vz,
    )
    return vx, vy, vz_ned, yaw_rate_cmd, components


# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_HEADER = [
    'time', 'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    'xd', 'yd', 'zd', 'ex', 'ey', 'ez', 'e_yaw',
    'ff_x', 'ff_y', 'ff_z',
    'vx_p', 'vy_p', 'vz_p', 'vx_i', 'vy_i', 'vz_i', 'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd',
]


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS
# ──────────────────────────────────────────────────────────────────────────────
def init_plot():
    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    lines = {}
    
    for ax, key, label in zip(axes[:3], ['x', 'y', 'z'], ['X (m)', 'Y (m)', 'Z (m)']):
        lines[f'{key}d'], = ax.plot([], [], 'r-', linewidth=1, label='Setpoint')
        lines[f'{key}r'], = ax.plot([], [], 'b-', linewidth=1, label='Real')
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    
    ax = axes[3]
    lines['eyaw'], = ax.plot([], [], 'g-', linewidth=1, label='Error yaw (rad)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Yaw error (rad)')
    ax.set_xlabel('Tiempo (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)
    
    fig.suptitle('Seguimiento líder-seguidor - Conexiones UDP separadas', fontsize=11)
    return fig, axes, lines


def update_plot(axes, lines, buf):
    t = buf['t']
    for key in ('x', 'y', 'z'):
        lines[f'{key}d'].set_data(t, buf[f'{key}d'])
        lines[f'{key}r'].set_data(t, buf[f'{key}r'])
    lines['eyaw'].set_data(t, buf['eyaw'])
    for ax in axes:
        ax.relim()
        ax.autoscale_view()
    plt.pause(0.001)


# ──────────────────────────────────────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("CONTROLADOR LÍDER-SEGUIDOR CON CONEXIONES UDP SEPARADAS")
    print("=" * 60)
    print(f"\n📋 CONFIGURACIÓN:")
    print(f"   LÍDER    → {LEADER_CONFIG['udp_port']} (SYSID={LEADER_CONFIG['sysid']})")
    print(f"   SEGUIDOR → {FOLLOWER_CONFIG['udp_port']} (SYSID={FOLLOWER_CONFIG['sysid']})")
    
    # Iniciar hilos lectores
    stop_reader = threading.Event()
    
    leader_thread = threading.Thread(
        target=_leader_reader,
        args=(stop_reader,),
        daemon=True,
        name="leader-reader"
    )
    
    follower_thread = threading.Thread(
        target=_follower_reader,
        args=(stop_reader,),
        daemon=True,
        name="follower-reader"
    )
    
    leader_thread.start()
    follower_thread.start()
    
    # Conectar canal de comandos para el seguidor
    commander = FollowerCommander(FOLLOWER_CONFIG)
    
    # Esperar datos de ambos drones
    print("\n⏳ Esperando telemetría de líder y seguidor...", end='', flush=True)
    while True:
        L, S = get_states()
        if states_ready(L, S):
            break
        time.sleep(0.05)
    print(" listo.")
    print(f"\n📊 ESTADO INICIAL:")
    print(f"   Líder   → x={L['x']:.2f}, y={L['y']:.2f}, z={L['z']:.2f}, yaw={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor → x={S['x']:.2f}, y={S['y']:.2f}, z={S['z']:.2f}, yaw={math.degrees(S['yaw']):.1f}°")
    
    # CSV
    csv_filename = f'lf_log_{int(time.time())}.csv'
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)
    start_time = time.time()
    print(f"\n📄 Log en: {csv_filename}")
    
    # Gráficas
    if ENABLE_PLOT:
        fig, axes, lines = init_plot()
        plot_buf = dict(t=[], xd=[], xr=[], yd=[], yr=[], zd=[], zr=[], eyaw=[])
    
    # Estado PID
    pid = PIDState()
    
    print("\n🚀 Iniciando bucle de control. Ctrl+C para detener.")
    print(f"⚙️  Offset: d={OFFSET_D} m, α={math.degrees(OFFSET_ALPHA):.1f}°, Δz={OFFSET_DZ} m")
    print(f"⚙️  Ganancias pos: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"⚙️  Ganancias yaw: Kp={KP_YAW}, Ki={KI_YAW}, Kd={KD_YAW}")
    print("-" * 60)
    
    try:
        next_t = time.monotonic()
        while True:
            L, S = get_states()
            
            if states_ready(L, S):
                vx, vy, vz_ned, yaw_rate_cmd, c = compute_control(L, S, pid)
                commander.send_velocity_yawrate(vx, vy, vz_ned, yaw_rate_cmd)
                
                t_elapsed = time.time() - start_time
                
                # Escribir CSV
                csv_writer.writerow([
                    t_elapsed,
                    L['x'], L['y'], L['z'], L['vx'], L['vy'], L['vz'], L['yaw'], L['yaw_rate'],
                    S['x'], S['y'], S['z'], S['vx'], S['vy'], S['vz'], S['yaw'], S['yaw_rate'],
                    c['xd'], c['yd'], c['zd'],
                    c['ex'], c['ey'], c['ez'], c['e_yaw'],
                    c['ff_x'], c['ff_y'], c['ff_z'],
                    c['vx_p'], c['vy_p'], c['vz_p'],
                    c['vx_i'], c['vy_i'], c['vz_i'],
                    c['vx_d'], c['vy_d'], c['vz_d'],
                    c['vx'], c['vy'], c['vz'],
                ])
                csv_file.flush()
                
                # Actualizar gráficas
                if ENABLE_PLOT:
                    plot_buf['t'].append(t_elapsed)
                    plot_buf['xd'].append(c['xd'])
                    plot_buf['xr'].append(S['x'])
                    plot_buf['yd'].append(c['yd'])
                    plot_buf['yr'].append(S['y'])
                    plot_buf['zd'].append(c['zd'])
                    plot_buf['zr'].append(S['z'])
                    plot_buf['eyaw'].append(c['e_yaw'])
                    update_plot(axes, lines, plot_buf)
            
            # Control de tiempo
            next_t += DT
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupción por usuario.")
    
    finally:
        # Detener el seguidor
        commander.send_velocity_yawrate(0.0, 0.0, 0.0, 0.0)
        time.sleep(0.2)
        
        # Detener hilos
        stop_reader.set()
        leader_thread.join(timeout=2.0)
        follower_thread.join(timeout=2.0)
        
        csv_file.close()
        print(f"\n📄 Datos guardados en {csv_filename}")
        print("🔌 Conexiones cerradas.")
        
        if ENABLE_PLOT:
            plt.ioff()
            plt.show()
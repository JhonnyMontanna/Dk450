#!/usr/bin/env python3
"""
Controlador PID líder-seguidor con prealimentación
====================================================
Implementación directa del planteamiento matemático (sección líder-seguidor).

Ley de control de posición (ec. pid_velocidad):
    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L - v_S)

donde:
    FF = ψ̇_L · Rz(π/2) · d(t)          prealimentación por giro del líder
    d(t) = [d·cos(ψ_L+α), d·sin(ψ_L+α), Δz]   offset polar rotante (ec. offset_polar)
    e_p = p_L + d(t) - p_S              error de posición (ec. error_posicion)
    Kd usa v_L - v_S, no diferencias numéricas de e_p (ec. derivada_error)

Ley de control de yaw (ec. pid_yaw):
    r_cmd = Kp_ψ · e_ψ + Ki_ψ · ∫e_ψ dt + Kd_ψ · (ψ̇_L - ψ̇_S)

    e_ψ = wrap(ψ_L - ψ_S)              (ec. error_yaw)
    wrap proyecta al intervalo (-π, π]
"""

import math
import time
import csv
import threading
from collections import deque
import matplotlib.pyplot as plt
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
CONNECTION_STRING = 'udp:127.0.0.1:14552'
LEADER_SYSID   = 2
FOLLOWER_SYSID = 1
FOLLOWER_COMPID = 1

# Offset en coordenadas polares (ec. offset_polar)
#   d     : distancia de separación horizontal [m]
#   alpha : ángulo relativo al eje longitudinal del líder [rad]
#           0 = detrás del líder, π/2 = a su izquierda, etc.
#   dz    : diferencia de altitud deseada (seguidor - líder) [m], positivo = más alto
OFFSET_D     = 2.0          # metros
OFFSET_ALPHA = math.pi      # detrás del líder
OFFSET_DZ    = 0.0          # misma altitud

# Ganancias PID posición
KP   = 0.5
KI   = 0.05
KD   = 0.2

# Ganancias PID yaw (ec. pid_yaw)
KP_YAW = 0.8
KI_YAW = 0.0
KD_YAW = 0.1

# Anti-windup: límite del integrador [m·s → m/s cuando multiplicado por Ki]
INTEGRAL_LIMIT = 2.0        # m·s por eje
INTEGRAL_YAW_LIMIT = 1.0    # rad·s

# Velocidad máxima comandada al seguidor [m/s]
V_MAX = 3.0
YAW_RATE_MAX = 1.0          # rad/s

# Frecuencia de control
RATE = 20                   # Hz
DT   = 1.0 / RATE

# Activar gráficas en tiempo real
ENABLE_PLOT = True

# Máscara MAVLink: velocidades lineales + yaw rate
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
# ESTADO COMPARTIDO (hilo lector MAVLink)
# ──────────────────────────────────────────────────────────────────────────────
_state_lock = threading.Lock()

# Estado del líder
_leader = dict(x=None, y=None, z=None,
               vx=None, vy=None, vz=None,
               yaw=None, yaw_rate=None)

# Estado del seguidor
_follower = dict(x=None, y=None, z=None,
                 vx=None, vy=None, vz=None,
                 yaw=None, yaw_rate=None)


def _mavlink_reader(master, stop_event):
    """
    Hilo dedicado. Lee LOCAL_POSITION_NED y ATTITUDE de ambos drones.
    LOCAL_POSITION_NED provee posición y velocidad lineal.
    ATTITUDE provee yaw y yaw_rate sin diferenciación numérica.
    """
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue

        sysid = msg.get_srcSystem()
        mtype = msg.get_type()

        with _state_lock:
            if mtype == 'LOCAL_POSITION_NED':
                # NED: z negativo hacia arriba → convertir a altitud positiva
                state = _leader if sysid == LEADER_SYSID else (
                        _follower if sysid == FOLLOWER_SYSID else None)
                if state is not None:
                    state['x']  = msg.x
                    state['y']  = msg.y
                    state['z']  = -msg.z    # altura positiva hacia arriba
                    state['vx'] = msg.vx
                    state['vy'] = msg.vy
                    state['vz'] = -msg.vz   # velocidad vertical positiva hacia arriba

            elif mtype == 'ATTITUDE':
                state = _leader if sysid == LEADER_SYSID else (
                        _follower if sysid == FOLLOWER_SYSID else None)
                if state is not None:
                    state['yaw']      = msg.yaw        # rad, rango (-π, π]
                    state['yaw_rate'] = msg.yawspeed   # rad/s, positivo antihorario


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
    """
    Proyecta θ al intervalo (-π, π] (ec. wrap).
    Equivalente a atan2(sin θ, cos θ).
    """
    return math.atan2(math.sin(theta), math.cos(theta))


def compute_offset(psi_L):
    """
    Calcula el vector de offset d(t) en RTK (ec. offset_polar):
        d = [d·cos(ψ_L + α), d·sin(ψ_L + α), Δz]
    El offset rota solidariamente con la orientación del líder.
    """
    angle = psi_L + OFFSET_ALPHA
    dx = OFFSET_D * math.cos(angle)
    dy = OFFSET_D * math.sin(angle)
    dz = OFFSET_DZ
    return dx, dy, dz


def compute_offset_dot(psi_L, yaw_rate_L, dx, dy):
    """
    Derivada temporal del offset (ec. derivada_offset_compacta):
        ḋ = ψ̇_L · Rz(π/2) · d
    Rz(π/2) · [dx, dy, dz]ᵀ = [-dy, dx, 0]ᵀ

    Este es el término de prealimentación que compensa el giro del líder.
    """
    ff_x = yaw_rate_L * (-dy)
    ff_y = yaw_rate_L * ( dx)
    ff_z = 0.0
    return ff_x, ff_y, ff_z


def clamp(value, limit):
    return max(-limit, min(limit, value))


# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE COMANDOS
# ──────────────────────────────────────────────────────────────────────────────
def send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate):
    """
    Envía setpoint de velocidad lineal (NED) + yaw_rate al seguidor.
    vz_ned ya está en frame NED (negativo hacia arriba).
    yaw_rate en rad/s, positivo = antihorario desde arriba.
    """
    master.mav.set_position_target_local_ned_send(
        0, FOLLOWER_SYSID, FOLLOWER_COMPID,
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
    """Encapsula las variables de memoria del PID (integral, prev_error)."""
    def __init__(self):
        self.integral    = [0.0, 0.0, 0.0]   # eje x, y, z
        self.integral_yaw = 0.0

    def reset(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# LEY DE CONTROL (ec. pid_velocidad + ec. pid_yaw)
# ──────────────────────────────────────────────────────────────────────────────
def compute_control(L, S, pid):
    """
    Calcula el comando de velocidad e yaw_rate según las ecuaciones del
    planteamiento matemático.

    Retorna:
        vx, vy, vz_ned, yaw_rate_cmd   (todos en frame NED)
        y un dict con los componentes internos para logging.
    """
    # ── 1. Offset polar rotante d(t) (ec. offset_polar) ──────────────────────
    dx, dy, dz = compute_offset(L['yaw'])

    # ── 2. Posición deseada del seguidor (ec. posicion_deseada) ──────────────
    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz

    # ── 3. Error de posición (ec. error_posicion) ─────────────────────────────
    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']

    # ── 4. Integrador con anti-windup ─────────────────────────────────────────
    pid.integral[0] = clamp(pid.integral[0] + ex * DT, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * DT, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * DT, INTEGRAL_LIMIT)

    # ── 5. Acción derivativa: diferencia de velocidades (ec. derivada_error) ──
    #    Kd·(v_L - v_S) en lugar de Kd·ė_p numérico → sin amplificación de ruido
    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    # ── 6. Prealimentación: ψ̇_L · Rz(π/2) · d (ec. derivada_offset_compacta) ─
    ff_x, ff_y, ff_z = compute_offset_dot(L['yaw'], L['yaw_rate'], dx, dy)

    # ── 7. Ley de control de posición (ec. pid_velocidad) ────────────────────
    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z   # positivo hacia arriba

    # Clamp de velocidad total (seguridad)
    v_horiz = math.hypot(vx, vy)
    if v_horiz > V_MAX:
        vx *= V_MAX / v_horiz
        vy *= V_MAX / v_horiz
    vz = clamp(vz, V_MAX)

    # Convertir vz a frame NED (negativo hacia arriba)
    vz_ned = -vz

    # ── 8. Control de yaw (ec. pid_yaw) ──────────────────────────────────────
    e_yaw = wrap(L['yaw'] - S['yaw'])                   # (ec. error_yaw)
    pid.integral_yaw = clamp(
        pid.integral_yaw + e_yaw * DT, INTEGRAL_YAW_LIMIT
    )
    dyaw = L['yaw_rate'] - S['yaw_rate']                # ψ̇_L - ψ̇_S de telemetría

    yaw_rate_cmd = clamp(
        KP_YAW * e_yaw + KI_YAW * pid.integral_yaw + KD_YAW * dyaw,
        YAW_RATE_MAX
    )

    # ── 9. Componentes para logging ───────────────────────────────────────────
    components = dict(
        xd=xd, yd=yd, zd=zd,
        ex=ex, ey=ey, ez=ez,
        e_yaw=e_yaw,
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
    'time',
    # Líder
    'lx', 'ly', 'lz', 'lvx', 'lvy', 'lvz', 'l_yaw', 'l_yawrate',
    # Seguidor
    'sx', 'sy', 'sz', 'svx', 'svy', 'svz', 's_yaw', 's_yawrate',
    # Setpoint
    'xd', 'yd', 'zd',
    # Errores
    'ex', 'ey', 'ez', 'e_yaw',
    # Componentes del controlador
    'ff_x', 'ff_y', 'ff_z',
    'vx_p', 'vy_p', 'vz_p',
    'vx_i', 'vy_i', 'vz_i',
    'vx_d', 'vy_d', 'vz_d',
    'vx_cmd', 'vy_cmd', 'vz_cmd',
]


def write_csv_row(writer, t, L, S, c):
    writer.writerow([
        t,
        L['x'], L['y'], L['z'], L['vx'], L['vy'], L['vz'], L['yaw'], L['yaw_rate'],
        S['x'], S['y'], S['z'], S['vx'], S['vy'], S['vz'], S['yaw'], S['yaw_rate'],
        c['xd'], c['yd'], c['zd'],
        c['ex'], c['ey'], c['ez'], c['e_yaw'],
        c['ff_x'], c['ff_y'], c['ff_z'],
        c['vx_p'], c['vy_p'], c['vz_p'],
        c['vx_i'], c['vy_i'], c['vz_i'],
        c['vx_d'], c['vy_d'], c['vz_d'],
        c['vx'],   c['vy'],   c['vz'],
    ])


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────
def init_plot():
    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    lines = {}

    for ax, key, label in zip(axes[:3],
                               ['x', 'y', 'z'],
                               ['X (m)', 'Y (m)', 'Z (m)']):
        lines[f'{key}d'], = ax.plot([], [], 'r-', linewidth=1, label='Setpoint')
        lines[f'{key}r'], = ax.plot([], [], 'b-', linewidth=1, label='Real')
        ax.set_ylabel(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[3]
    lines['eyaw'], = ax.plot([], [], 'g-', linewidth=1, label='Error yaw (rad)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_ylabel('Yaw error (rad)')
    ax.set_xlabel('Tiempo (s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle('Seguimiento líder-seguidor', fontsize=11)
    return fig, axes, lines


def update_plot(axes, lines, buf):
    t = buf['t']
    for key in ('x', 'y', 'z'):
        lines[f'{key}d'].set_data(t, buf[f'{key}d'])
        lines[f'{key}r'].set_data(t, buf[f'{key}r'])
    lines['eyaw'].set_data(t, buf['eyaw'])
    for ax in axes:
        ax.relim(); ax.autoscale_view()
    plt.pause(0.001)


# ──────────────────────────────────────────────────────────────────────────────
# PROGRAMA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"🔌 Conectando a {CONNECTION_STRING}...")
    master = mavutil.mavlink_connection(CONNECTION_STRING)
    master.wait_heartbeat()
    print(f"✅ Heartbeat. SYS={master.target_system}, COMP={master.target_component}")

    # Hilo lector MAVLink
    stop_reader   = threading.Event()
    reader_thread = threading.Thread(
        target=_mavlink_reader,
        args=(master, stop_reader),
        daemon=True, name="mavlink-reader"
    )
    reader_thread.start()

    # Esperar datos de ambos drones
    print("⏳ Esperando telemetría de líder y seguidor...", end='', flush=True)
    while True:
        L, S = get_states()
        if states_ready(L, S):
            break
        time.sleep(0.05)
    print(" listo.")
    print(f"   Líder   → x={L['x']:.2f}, y={L['y']:.2f}, z={L['z']:.2f}, yaw={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor → x={S['x']:.2f}, y={S['y']:.2f}, z={S['z']:.2f}, yaw={math.degrees(S['yaw']):.1f}°")

    # CSV
    csv_filename = f'lf_log_{int(time.time())}.csv'
    csv_file   = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)
    start_time = time.time()
    print(f"📄 Log en: {csv_filename}")

    # Gráficas
    if ENABLE_PLOT:
        fig, axes, lines = init_plot()
        plot_buf = dict(t=[], xd=[], xr=[], yd=[], yr=[], zd=[], zr=[], eyaw=[])

    # Estado PID
    pid = PIDState()

    print("🚀 Iniciando bucle de control. Ctrl+C para detener.")
    print(f"⚙️  Offset: d={OFFSET_D} m, α={math.degrees(OFFSET_ALPHA):.1f}°, Δz={OFFSET_DZ} m")
    print(f"⚙️  Ganancias pos: Kp={KP}, Ki={KI}, Kd={KD}")
    print(f"⚙️  Ganancias yaw: Kp={KP_YAW}, Ki={KI_YAW}, Kd={KD_YAW}")

    try:
        next_t = time.monotonic()   # tiempo absoluto para evitar deriva
        while True:
            L, S = get_states()

            if states_ready(L, S):
                vx, vy, vz_ned, yaw_rate_cmd, c = compute_control(L, S, pid)
                send_velocity_yawrate(master, vx, vy, vz_ned, yaw_rate_cmd)

                t_elapsed = time.time() - start_time
                write_csv_row(csv_writer, t_elapsed, L, S, c)
                csv_file.flush()

                if ENABLE_PLOT:
                    plot_buf['t'].append(t_elapsed)
                    plot_buf['xd'].append(c['xd']); plot_buf['xr'].append(S['x'])
                    plot_buf['yd'].append(c['yd']); plot_buf['yr'].append(S['y'])
                    plot_buf['zd'].append(c['zd']); plot_buf['zr'].append(S['z'])
                    plot_buf['eyaw'].append(c['e_yaw'])
                    update_plot(axes, lines, plot_buf)

            # Tiempo absoluto: sin deriva acumulada
            next_t += DT
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupción por usuario.")

    finally:
        # Detener el seguidor antes de cerrar
        send_velocity_yawrate(master, 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.2)

        stop_reader.set()
        reader_thread.join(timeout=2.0)

        csv_file.close()
        print(f"📄 Datos guardados en {csv_filename}")

        master.close()
        print("🔌 Conexión cerrada.")

        if ENABLE_PLOT:
            plt.ioff()
            plt.show()
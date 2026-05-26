#!/usr/bin/env python3
"""
MavlinkControlLF_DualConn.py — Controlador PID líder-seguidor en tiempo real
=============================================================================
Dos drones, dos conexiones MAVLink independientes (un puerto UDP por drone).

Ley de control de posición (ec. pid_velocidad):
    v_cmd = FF + Kp·e_p + Ki·∫e_p dt + Kd·(v_L - v_S)

    FF   = ψ̇_L · Rz(π/2) · d(t)
    d(t) = [d·cos(ψ_L+α), d·sin(ψ_L+α), Δz]   offset polar rotante
    e_p  = p_L + d(t) - p_S

Ley de control de yaw (ec. pid_yaw):
    r_cmd = Kp_ψ·e_ψ + Ki_ψ·∫e_ψ dt + Kd_ψ·(ψ̇_L − ψ̇_S)
    e_ψ   = wrap(ψ_L − ψ_S)

Setup SITL (bat de ejemplo):
    Drone LÍDER   → mavproxy → udp:127.0.0.1:14552   (SYSID 1, instancia -I0)
    Drone SEGUIDOR → mavproxy → udp:127.0.0.1:14553  (SYSID 2, instancia -I1)
"""

import math
import time
import csv
import threading
import matplotlib.pyplot as plt
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estos valores a tu entorno
# ──────────────────────────────────────────────────────────────────────────────

# Conexiones MAVLink (un puerto por drone, tal como genera tu .bat)
LEADER_CONN   = 'udp:127.0.0.1:14552'   # Drone 1 (LÍDER)   — instancia -I0
FOLLOWER_CONN = 'udp:127.0.0.1:14553'   # Drone 2 (SEGUIDOR) — instancia -I1

# SYSID esperados (para verificar que el heartbeat viene del drone correcto)
LEADER_SYSID    = 1
FOLLOWER_SYSID  = 2
FOLLOWER_COMPID = 1

# ── Offset polar (posición del seguidor respecto al líder) ────────────────────
#   OFFSET_D     : distancia de separación horizontal [m]
#   OFFSET_ALPHA : ángulo respecto al eje del líder [rad]
#                  math.pi      → detrás del líder
#                  0            → delante del líder
#                  math.pi/2    → izquierda del líder
#                  -math.pi/2   → derecha del líder
#   OFFSET_DZ    : diferencia de altitud (seguidor − líder) [m]
#                  positivo → seguidor más alto (recomendado por seguridad)
OFFSET_D     = 1.0
OFFSET_ALPHA = math.pi    # detrás del líder
OFFSET_DZ    = 1.0        # seguidor 1 m más alto (seguridad en primera prueba)

# ── Ganancias PID posición ────────────────────────────────────────────────────
KP   = 0.5
KI   = 0.0    # activa solo cuando el sistema esté estable
KD   = 0.0

# ── Ganancias PID yaw ─────────────────────────────────────────────────────────
KP_YAW = 0.8
KI_YAW = 0.0
KD_YAW = 0.1

# ── Anti-windup ───────────────────────────────────────────────────────────────
INTEGRAL_LIMIT     = 2.0   # m·s por eje
INTEGRAL_YAW_LIMIT = 1.0   # rad·s

# ── Límites de salida ─────────────────────────────────────────────────────────
V_MAX        = 2.0    # m/s horizontal (conservador para primera prueba)
V_MAX_Z      = 1.0    # m/s vertical
YAW_RATE_MAX = 1.0    # rad/s

# ── Frecuencia de control ─────────────────────────────────────────────────────
RATE = 20             # Hz
DT   = 1.0 / RATE

# ── Gráficas en tiempo real ───────────────────────────────────────────────────
ENABLE_PLOT = True

# ── Máscara MAVLink: solo velocidades lineales + yaw rate ─────────────────────
TYPE_MASK_VEL_YAWRATE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE  |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
)

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO — un dict por drone, protegido por lock
# ──────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()

def _empty_state():
    return dict(x=None, y=None, z=None,
                vx=None, vy=None, vz=None,
                yaw=None, yaw_rate=None)

_leader_state   = _empty_state()
_follower_state = _empty_state()


def _reader(master, state_dict, label, stop_event):
    """
    Hilo lector MAVLink para UN drone.
    Lee LOCAL_POSITION_NED y ATTITUDE y actualiza state_dict.
    """
    while not stop_event.is_set():
        msg = master.recv_match(
            type=['LOCAL_POSITION_NED', 'ATTITUDE'],
            blocking=True, timeout=0.1
        )
        if msg is None:
            continue

        mtype = msg.get_type()
        with _lock:
            if mtype == 'LOCAL_POSITION_NED':
                state_dict['x']  = msg.x
                state_dict['y']  = msg.y
                state_dict['z']  = -msg.z    # NED→altura positiva hacia arriba
                state_dict['vx'] = msg.vx
                state_dict['vy'] = msg.vy
                state_dict['vz'] = -msg.vz
            elif mtype == 'ATTITUDE':
                state_dict['yaw']      = msg.yaw
                state_dict['yaw_rate'] = msg.yawspeed


def get_states():
    with _lock:
        return dict(_leader_state), dict(_follower_state)


def state_ready(s):
    return all(v is not None for v in s.values())


def states_ready(L, S):
    return state_ready(L) and state_ready(S)


# ──────────────────────────────────────────────────────────────────────────────
# MATEMÁTICAS
# ──────────────────────────────────────────────────────────────────────────────
def wrap(theta):
    return math.atan2(math.sin(theta), math.cos(theta))


def clamp(val, limit):
    return max(-limit, min(limit, val))


def compute_offset(psi_L):
    """Vector de offset d(t) en frame NED horizontal, rotante con el líder."""
    angle = psi_L + OFFSET_ALPHA
    return (OFFSET_D * math.cos(angle),
            OFFSET_D * math.sin(angle),
            OFFSET_DZ)


def compute_ff(yaw_rate_L, dx, dy):
    """Prealimentación: ψ̇_L · Rz(π/2) · d — compensa el giro del líder."""
    return yaw_rate_L * (-dy), yaw_rate_L * dx, 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ESTADO PID
# ──────────────────────────────────────────────────────────────────────────────
class PIDState:
    def __init__(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0

    def reset(self):
        self.integral     = [0.0, 0.0, 0.0]
        self.integral_yaw = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# LEY DE CONTROL
# ──────────────────────────────────────────────────────────────────────────────
def compute_control(L, S, pid):
    dx, dy, dz = compute_offset(L['yaw'])

    xd = L['x'] + dx
    yd = L['y'] + dy
    zd = L['z'] + dz

    ex = xd - S['x']
    ey = yd - S['y']
    ez = zd - S['z']

    pid.integral[0] = clamp(pid.integral[0] + ex * DT, INTEGRAL_LIMIT)
    pid.integral[1] = clamp(pid.integral[1] + ey * DT, INTEGRAL_LIMIT)
    pid.integral[2] = clamp(pid.integral[2] + ez * DT, INTEGRAL_LIMIT)

    dv_x = L['vx'] - S['vx']
    dv_y = L['vy'] - S['vy']
    dv_z = L['vz'] - S['vz']

    ff_x, ff_y, ff_z = compute_ff(L['yaw_rate'], dx, dy)

    vx = ff_x + KP*ex + KI*pid.integral[0] + KD*dv_x
    vy = ff_y + KP*ey + KI*pid.integral[1] + KD*dv_y
    vz = ff_z + KP*ez + KI*pid.integral[2] + KD*dv_z

    # Saturación horizontal
    v_h = math.hypot(vx, vy)
    if v_h > V_MAX:
        vx *= V_MAX / v_h
        vy *= V_MAX / v_h
    vz = clamp(vz, V_MAX_Z)

    vz_ned = -vz   # positivo hacia arriba → NED negativo hacia arriba

    # Control de yaw
    e_yaw = wrap(L['yaw'] - S['yaw'])
    pid.integral_yaw = clamp(pid.integral_yaw + e_yaw * DT, INTEGRAL_YAW_LIMIT)
    dyaw = L['yaw_rate'] - S['yaw_rate']
    yaw_rate_cmd = clamp(
        KP_YAW*e_yaw + KI_YAW*pid.integral_yaw + KD_YAW*dyaw,
        YAW_RATE_MAX
    )

    c = dict(
        xd=xd, yd=yd, zd=zd,
        ex=ex, ey=ey, ez=ez, e_yaw=e_yaw,
        ff_x=ff_x, ff_y=ff_y, ff_z=ff_z,
        vx_p=KP*ex, vy_p=KP*ey, vz_p=KP*ez,
        vx_i=KI*pid.integral[0], vy_i=KI*pid.integral[1], vz_i=KI*pid.integral[2],
        vx_d=KD*dv_x, vy_d=KD*dv_y, vz_d=KD*dv_z,
        vx=vx, vy=vy, vz=vz,
    )
    return vx, vy, vz_ned, yaw_rate_cmd, c


# ──────────────────────────────────────────────────────────────────────────────
# ENVÍO DE COMANDOS AL SEGUIDOR
# ──────────────────────────────────────────────────────────────────────────────
def send_velocity_yawrate(master_follower, vx, vy, vz_ned, yaw_rate):
    master_follower.mav.set_position_target_local_ned_send(
        0, FOLLOWER_SYSID, FOLLOWER_COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK_VEL_YAWRATE,
        0, 0, 0,
        vx, vy, vz_ned,
        0, 0, 0,
        0, yaw_rate
    )


# ──────────────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────────────
CSV_HEADER = [
    'time',
    'lx','ly','lz','lvx','lvy','lvz','l_yaw','l_yawrate',
    'sx','sy','sz','svx','svy','svz','s_yaw','s_yawrate',
    'xd','yd','zd',
    'ex','ey','ez','e_yaw',
    'ff_x','ff_y','ff_z',
    'vx_p','vy_p','vz_p',
    'vx_i','vy_i','vz_i',
    'vx_d','vy_d','vz_d',
    'vx_cmd','vy_cmd','vz_cmd','yaw_rate_cmd',
]


def write_csv_row(writer, t, L, S, c, yaw_rate_cmd):
    writer.writerow([
        f'{t:.4f}',
        L['x'],L['y'],L['z'],L['vx'],L['vy'],L['vz'],L['yaw'],L['yaw_rate'],
        S['x'],S['y'],S['z'],S['vx'],S['vy'],S['vz'],S['yaw'],S['yaw_rate'],
        c['xd'],c['yd'],c['zd'],
        c['ex'],c['ey'],c['ez'],c['e_yaw'],
        c['ff_x'],c['ff_y'],c['ff_z'],
        c['vx_p'],c['vy_p'],c['vz_p'],
        c['vx_i'],c['vy_i'],c['vz_i'],
        c['vx_d'],c['vy_d'],c['vz_d'],
        c['vx'],c['vy'],c['vz'],yaw_rate_cmd,
    ])


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS EN TIEMPO REAL
# ──────────────────────────────────────────────────────────────────────────────
def init_plot():
    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    lines = {}

    labels = [('x','X (m)'), ('y','Y (m)'), ('z','Z altura (m)')]
    for ax, (key, label) in zip(axes[:3], labels):
        lines[f'{key}d'], = ax.plot([], [], 'r--', lw=1.2, label='Setpoint S')
        lines[f'{key}l'], = ax.plot([], [], 'g-',  lw=1.0, label='Líder')
        lines[f'{key}r'], = ax.plot([], [], 'b-',  lw=1.0, label='Seguidor')
        ax.set_ylabel(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[3]
    lines['eyaw'], = ax.plot([], [], 'm-', lw=1, label='Error yaw (rad)')
    lines['dist'],  = ax.plot([], [], 'k-', lw=1, label='Dist XY (m)')
    ax.axhline(OFFSET_D, color='r', lw=0.8, ls='--', label=f'd={OFFSET_D}m')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_ylabel('Yaw error / Distancia')
    ax.set_xlabel('Tiempo (s)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle(
        f'Líder-Seguidor en tiempo real  |  d={OFFSET_D}m  α={math.degrees(OFFSET_ALPHA):.0f}°  Δz={OFFSET_DZ}m',
        fontsize=11
    )
    return fig, axes, lines


def update_plot(axes, lines, buf):
    t = buf['t']
    for key in ('x', 'y', 'z'):
        lines[f'{key}d'].set_data(t, buf[f'{key}d'])
        lines[f'{key}l'].set_data(t, buf[f'{key}l'])
        lines[f'{key}r'].set_data(t, buf[f'{key}r'])
    lines['eyaw'].set_data(t, buf['eyaw'])
    lines['dist'].set_data(t, buf['dist'])
    for ax in axes:
        ax.relim(); ax.autoscale_view()
    plt.pause(0.001)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  CONTROLADOR LÍDER-SEGUIDOR — Tiempo Real (Dual UDP)")
    print("=" * 60)
    print(f"  Líder    → {LEADER_CONN}   (SYSID {LEADER_SYSID})")
    print(f"  Seguidor → {FOLLOWER_CONN}  (SYSID {FOLLOWER_SYSID})")
    print(f"  Offset   : d={OFFSET_D} m  α={math.degrees(OFFSET_ALPHA):.0f}°  Δz={OFFSET_DZ} m")
    print(f"  Ganancias: Kp={KP}  Ki={KI}  Kd={KD}")
    print(f"  Yaw      : Kp={KP_YAW}  Ki={KI_YAW}  Kd={KD_YAW}")
    print(f"  V_MAX={V_MAX} m/s   YAW_RATE_MAX={YAW_RATE_MAX} rad/s")
    print("=" * 60)

    # ── Conectar al líder ─────────────────────────────────────────────────────
    print(f"\n🔌 Conectando al LÍDER   ({LEADER_CONN})...")
    master_leader = mavutil.mavlink_connection(LEADER_CONN)
    master_leader.wait_heartbeat()
    print(f"   ✅ Heartbeat líder   — SYS={master_leader.target_system}")

    # ── Conectar al seguidor ──────────────────────────────────────────────────
    print(f"🔌 Conectando al SEGUIDOR ({FOLLOWER_CONN})...")
    master_follower = mavutil.mavlink_connection(FOLLOWER_CONN)
    master_follower.wait_heartbeat()
    print(f"   ✅ Heartbeat seguidor — SYS={master_follower.target_system}")

    # ── Lanzar hilos lectores ─────────────────────────────────────────────────
    stop_ev = threading.Event()

    t_leader = threading.Thread(
        target=_reader,
        args=(master_leader, _leader_state, 'LÍDER', stop_ev),
        daemon=True, name='reader-leader'
    )
    t_follower = threading.Thread(
        target=_reader,
        args=(master_follower, _follower_state, 'SEGUIDOR', stop_ev),
        daemon=True, name='reader-follower'
    )
    t_leader.start()
    t_follower.start()

    # ── Esperar telemetría completa de ambos drones ───────────────────────────
    print("\n⏳ Esperando telemetría de AMBOS drones", end='', flush=True)
    timeout_tel = 30.0
    t0_tel = time.monotonic()
    while True:
        L, S = get_states()
        if states_ready(L, S):
            break
        if time.monotonic() - t0_tel > timeout_tel:
            print("\n❌ Timeout esperando telemetría. Verifica conexiones y que")
            print("   ambos drones estén activos y enviando LOCAL_POSITION_NED + ATTITUDE.")
            stop_ev.set()
            raise SystemExit(1)
        print('.', end='', flush=True)
        time.sleep(0.2)

    print(" ✅ listo.")
    print(f"   Líder    → x={L['x']:.2f}  y={L['y']:.2f}  z={L['z']:.2f}  ψ={math.degrees(L['yaw']):.1f}°")
    print(f"   Seguidor → x={S['x']:.2f}  y={S['y']:.2f}  z={S['z']:.2f}  ψ={math.degrees(S['yaw']):.1f}°")

    # ── CSV de salida ─────────────────────────────────────────────────────────
    csv_filename = f'lf_realtime_{int(time.time())}.csv'
    csv_file   = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)
    print(f"\n📄 Log: {csv_filename}")

    # ── Gráficas ──────────────────────────────────────────────────────────────
    if ENABLE_PLOT:
        fig, axes, plot_lines = init_plot()
        plot_buf = dict(
            t=[],
            xd=[], xl=[], xr=[],
            yd=[], yl=[], yr=[],
            zd=[], zl=[], zr=[],
            eyaw=[], dist=[]
        )

    pid        = PIDState()
    start_time = time.monotonic()

    print("\n🚀 Control activo. Pilotea el LÍDER normalmente.")
    print("   Ctrl+C para detener (envía velocidad 0 al seguidor).\n")

    try:
        next_t = time.monotonic()
        while True:
            L, S = get_states()

            if states_ready(L, S):
                vx, vy, vz_ned, yaw_rate_cmd, c = compute_control(L, S, pid)
                send_velocity_yawrate(master_follower, vx, vy, vz_ned, yaw_rate_cmd)

                t_el = time.monotonic() - start_time
                write_csv_row(csv_writer, t_el, L, S, c, yaw_rate_cmd)
                csv_file.flush()

                dist_xy = math.hypot(L['x'] - S['x'], L['y'] - S['y'])

                if ENABLE_PLOT:
                    plot_buf['t'].append(t_el)
                    plot_buf['xd'].append(c['xd']); plot_buf['xl'].append(L['x']); plot_buf['xr'].append(S['x'])
                    plot_buf['yd'].append(c['yd']); plot_buf['yl'].append(L['y']); plot_buf['yr'].append(S['y'])
                    plot_buf['zd'].append(c['zd']); plot_buf['zl'].append(L['z']); plot_buf['zr'].append(S['z'])
                    plot_buf['eyaw'].append(c['e_yaw'])
                    plot_buf['dist'].append(dist_xy)
                    update_plot(axes, plot_lines, plot_buf)

                # Consola: imprimir cada 5 s
                if int(t_el) % 5 == 0 and int(t_el - DT) % 5 != 0:
                    err_xy = math.hypot(c['ex'], c['ey'])
                    print(f"t={t_el:6.1f}s | err_xy={err_xy:.3f}m  err_z={c['ez']:.3f}m"
                          f"  dist={dist_xy:.2f}m  e_yaw={math.degrees(c['e_yaw']):.1f}°"
                          f"  vx={c['vx']:.2f} vy={c['vy']:.2f} vz={c['vz']:.2f}")

            next_t += DT
            sleep_t = next_t - time.monotonic()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n⏹️  Interrupción por usuario.")

    finally:
        # Detener seguidor
        send_velocity_yawrate(master_follower, 0.0, 0.0, 0.0, 0.0)
        time.sleep(0.3)

        stop_ev.set()
        t_leader.join(timeout=2.0)
        t_follower.join(timeout=2.0)

        csv_file.close()
        master_leader.close()
        master_follower.close()

        print(f"📄 Log guardado: {csv_filename}")
        print("🔌 Conexiones cerradas.")

        if ENABLE_PLOT:
            plt.ioff()
            plt.show()
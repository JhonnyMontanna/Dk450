#!/usr/bin/env python3
"""
Seguidor de waypoints NED con lazo cerrado
Envía cada punto de la lista cuando el dron converge al anterior,
o bien por tiempo fijo (modo temporizador), según la configuración.

Modos de avance:
  'convergence' → espera que el dron llegue al waypoint actual antes de pasar al siguiente
  'timer'       → avanza cada WAYPOINT_INTERVAL segundos sin importar si llegó
"""
import time
import math
import threading
from collections import deque
from pymavlink import mavutil
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
CONN   = 'udp:127.0.0.1:14552'
SYSID  = 1
COMPID = 0

# Lista de waypoints en coordenadas NED absolutas (x, y, z)
# x = norte [m], y = este [m], z = abajo [m] (negativo = altitud)
# El primer punto suele ser la posición de inicio (o cerca de ella)
WAYPOINTS = [
    ( 0.0,  0.0, -3.0),   # punto inicial / hover
    ( 5.0,  0.0, -3.0),   # norte 5 m
    ( 5.0,  5.0, -3.0),   # este  5 m
    ( 0.0,  5.0, -3.0),   # sur   5 m
    ( 0.0,  0.0, -3.0),   # regreso al origen
]

# Modo de avance entre waypoints
ADVANCE_MODE      = 'convergence'  # 'convergence' | 'timer'

# Parámetros modo timer
WAYPOINT_INTERVAL = 10.0    # segundos por waypoint (solo en modo 'timer')

# Parámetros de convergencia (solo en modo 'convergence')
CONV_RADIUS  = 0.30   # distancia máxima para considerar "llegó" [m]
CONV_SPEED   = 0.15   # velocidad máxima para "parado" [m/s]
CONV_HOLD    = 1.0    # segundos consecutivos dentro del criterio
CONV_TIMEOUT = 20.0   # segundos máximos esperando convergencia

# Yaw fijo durante todo el vuelo (en radianes, None = ignorar yaw)
YAW = None   # ejemplo: 0.0 apunta al norte, math.pi/2 apunta al este

# Tasa de envío de setpoints y lectura de telemetría [Hz]
RATE = 20

# Usar coordenadas absolutas NED (True) o relativas al punto de despegue (False)
ABSOLUTE_NED = True

TYPE_MASK_POS = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)

TYPE_MASK_POS_NO_YAW = TYPE_MASK_POS | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE

# ===============================
# ESTADO COMPARTIDO (thread-safe)
# ===============================
_lock       = threading.Lock()
_latest_pos = None

def _mavlink_reader(master, stop_event):
    global _latest_pos
    while not stop_event.is_set():
        msg = master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=0.1)
        if msg:
            with _lock:
                _latest_pos = msg

def get_pos():
    with _lock:
        return _latest_pos

# ===============================
# ENVÍO DE SETPOINT
# ===============================
def send_waypoint(master, x, y, z, yaw=None):
    """Envía un setpoint de posición NED. Si yaw es None lo ignora."""
    mask = TYPE_MASK_POS_NO_YAW if yaw is None else TYPE_MASK_POS
    master.mav.set_position_target_local_ned_send(
        0, SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        mask,
        x, y, z,
        0, 0, 0,
        0, 0, 0,
        yaw if yaw is not None else 0,
        0
    )

# ===============================
# ESPERA DE POSICIÓN INICIAL
# ===============================
def wait_ready():
    print("Esperando posición inicial...", end='', flush=True)
    while get_pos() is None:
        time.sleep(0.02)
    print(" listo.")
    p = get_pos()
    print(f"Posición de despegue: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f}")
    return p

# ===============================
# NÚCLEO: EJECUTAR WAYPOINTS
# ===============================
def run_waypoints(master):
    home = wait_ready()

    # Si los waypoints son relativos, sumar la posición de despegue
    if ABSOLUTE_NED:
        wps = list(WAYPOINTS)
        print("Modo: coordenadas NED absolutas")
    else:
        wps = [(home.x + dx, home.y + dy, home.z + dz) for dx, dy, dz in WAYPOINTS]
        print(f"Modo: coordenadas relativas al despegue ({home.x:.2f}, {home.y:.2f}, {home.z:.2f})")

    n = len(wps)
    print(f"Waypoints cargados: {n}  |  modo avance: {ADVANCE_MODE}\n")

    # Log de telemetría
    log = {k: deque() for k in ('t', 'wp_idx', 'x_sp', 'y_sp', 'z_sp', 'x', 'y', 'z')}
    t_start = time.monotonic()
    dt      = 1.0 / RATE

    for idx, (tx, ty, tz) in enumerate(wps):
        print(f"  [{idx+1}/{n}] -> x={tx:.2f}  y={ty:.2f}  z={tz:.2f}", end='  ', flush=True)
        send_waypoint(master, tx, ty, tz, YAW)

        if ADVANCE_MODE == 'timer':
            # ── Modo temporizador: mantener setpoint y loguear durante WAYPOINT_INTERVAL ──
            t_wp_start = time.monotonic()
            next_t     = t_wp_start
            while time.monotonic() - t_wp_start < WAYPOINT_INTERVAL:
                send_waypoint(master, tx, ty, tz, YAW)
                pos = get_pos()
                if pos:
                    t_now = time.monotonic() - t_start
                    log['t'].append(t_now)
                    log['wp_idx'].append(idx)
                    log['x_sp'].append(tx); log['y_sp'].append(ty); log['z_sp'].append(tz)
                    log['x'].append(pos.x); log['y'].append(pos.y); log['z'].append(pos.z)
                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)
            print(f"avanzado por tiempo ({WAYPOINT_INTERVAL}s)")

        else:
            # ── Modo convergencia: esperar hasta que el dron llega al punto ──
            t_in_zone  = None
            t_wp_start = time.monotonic()
            next_t     = t_wp_start

            while True:
                now = time.monotonic()

                if now - t_wp_start > CONV_TIMEOUT:
                    print(f"timeout ({CONV_TIMEOUT}s) — avanzando")
                    break

                send_waypoint(master, tx, ty, tz, YAW)
                pos = get_pos()

                if pos:
                    t_now = now - t_start
                    log['t'].append(t_now)
                    log['wp_idx'].append(idx)
                    log['x_sp'].append(tx); log['y_sp'].append(ty); log['z_sp'].append(tz)
                    log['x'].append(pos.x); log['y'].append(pos.y); log['z'].append(pos.z)

                    dist  = math.sqrt((pos.x-tx)**2 + (pos.y-ty)**2 + (pos.z-tz)**2)
                    speed = math.sqrt(pos.vx**2 + pos.vy**2 + pos.vz**2)

                    if dist < CONV_RADIUS and speed < CONV_SPEED:
                        if t_in_zone is None:
                            t_in_zone = now
                        elif now - t_in_zone >= CONV_HOLD:
                            elapsed = now - t_wp_start
                            print(f"llegado en {elapsed:.1f}s  (dist={dist:.3f}m, vel={speed:.3f}m/s)")
                            break
                    else:
                        t_in_zone = None

                next_t += dt
                sleep_t = next_t - time.monotonic()
                if sleep_t > 0:
                    time.sleep(sleep_t)

    print("\nSecuencia completada.")
    return {k: list(v) for k, v in log.items()}

# ===============================
# GRÁFICAS
# ===============================
def plot_results(log, wps):
    t     = np.array(log['t'])
    x, y, z_   = np.array(log['x']), np.array(log['y']), np.array(log['z'])
    xsp   = np.array(log['x_sp'])
    ysp   = np.array(log['y_sp'])
    zsp   = np.array(log['z_sp'])
    idx   = np.array(log['wp_idx'])
    err   = np.sqrt((x-xsp)**2 + (y-ysp)**2 + (z_-zsp)**2)

    print(f"\nError posición RMS total: {err.mean():.4f} m  |  máx: {err.max():.4f} m")

    colors = plt.cm.tab10(np.linspace(0, 1, len(wps)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Seguimiento de waypoints NED', fontsize=13)

    # — Error en el tiempo con banda por waypoint —
    ax = axes[0, 0]
    ax.plot(t, err, 'k-', lw=1.2, label='Error 3D [m]')
    for i in range(len(wps)):
        m = idx == i
        if m.sum():
            ax.axvspan(t[m][0], t[m][-1], alpha=0.12, color=colors[i], label=f'WP{i+1}')
    ax.set(xlabel='Tiempo [s]', ylabel='Error [m]', title='Error de posición')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.4)

    # — X e Y en el tiempo —
    ax = axes[0, 1]
    ax.plot(t, xsp, 'b--', lw=1, label='X_sp', alpha=0.7)
    ax.plot(t, x,   'b-',  lw=1.5, label='X real')
    ax.plot(t, ysp, 'r--', lw=1, label='Y_sp', alpha=0.7)
    ax.plot(t, y,   'r-',  lw=1.5, label='Y real')
    ax.set(xlabel='Tiempo [s]', ylabel='Posición [m]', title='X e Y vs tiempo')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # — Altitud en el tiempo —
    ax = axes[1, 0]
    ax.plot(t, zsp, 'g--', lw=1, label='Z_sp (NED)', alpha=0.7)
    ax.plot(t, z_,  'g-',  lw=1.5, label='Z real')
    ax.set(xlabel='Tiempo [s]', ylabel='Z NED [m]', title='Altitud (Z NED)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # — Trayectoria XY top-down —
    ax = axes[1, 1]
    ax.plot(xsp, ysp, 'k:', lw=1.2, label='Ruta programada', zorder=1)
    for i in range(len(wps)):
        m = idx == i
        if m.sum():
            ax.plot(x[m], y[m], '-', color=colors[i], lw=2, label=f'WP{i+1}')
    # Marcar cada waypoint
    for i, (wx, wy, _) in enumerate(wps):
        ax.scatter(wx, wy, color=colors[i], s=60, zorder=5, edgecolors='k', lw=0.5)
        ax.annotate(f'{i+1}', (wx, wy), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)
    ax.set(xlabel='X — norte [m]', ylabel='Y — este [m]', title='Trayectoria XY')
    ax.axis('equal'); ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
if __name__ == '__main__':
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    print(f"Conectado: SYS={master.target_system} COMP={master.target_component}")

    stop_reader = threading.Event()
    reader = threading.Thread(
        target=_mavlink_reader, args=(master, stop_reader),
        daemon=True, name='mavlink-reader'
    )
    reader.start()

    try:
        log = run_waypoints(master)
    finally:
        stop_reader.set()
        reader.join(timeout=2.0)
        master.close()

    plot_results(log, list(WAYPOINTS) if ABSOLUTE_NED else
                 [(home.x+dx, home.y+dy, home.z+dz) for dx,dy,dz in WAYPOINTS])
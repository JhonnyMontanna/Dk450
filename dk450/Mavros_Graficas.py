#!/usr/bin/env python3
"""
telemetry_node.py — Telemetría en vivo + logs CSV para N drones
================================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.

Se ejecuta en paralelo con cualquier otro nodo. Se suscribe a los
namespaces que se le indiquen y:
  - Guarda posición, velocidad y yaw de cada dron en un CSV unificado.
  - Muestra una ventana matplotlib con posición XY en tiempo real.
  - Al recibir Ctrl+C guarda el CSV y opcionalmente imprime gráficas
    post-vuelo de todas las trayectorias registradas.

Parámetros ROS2:
  namespaces   : lista de namespaces separados por coma  (default: 'uav1')
                 Ejemplo: --ros-args -p namespaces:='uav1,uav2'
  live_plot    : activar ventana en tiempo real          (default: true)
  plot_interval: segundos entre refresco de la ventana  (default: 0.5)
  csv_prefix   : prefijo del archivo CSV                 (default: 'telem')

Uso típico:
  # Terminal 1: lanzar telemetría
  python3 telemetry_node.py --ros-args -p namespaces:='uav1,uav2' -p live_plot:=true

  # Terminal 2: lanzar misión
  python3 master_node.py --ros-args -p uav_ns:=uav1
"""

import math
import csv
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped

# ─────────────────────────────────────────────────────────────────────────────
# QoS
# ─────────────────────────────────────────────────────────────────────────────
_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Nodo
# ─────────────────────────────────────────────────────────────────────────────
class TelemetryNode(Node):

    def __init__(self):
        super().__init__('telemetry_node')

        self.declare_parameter('namespaces',    'uav1')
        self.declare_parameter('live_plot',      True)
        self.declare_parameter('plot_interval',  0.5)
        self.declare_parameter('csv_prefix',    'telem')

        ns_str        = self.get_parameter('namespaces').value
        self.live     = self.get_parameter('live_plot').value
        plot_ivl      = self.get_parameter('plot_interval').value
        csv_prefix    = self.get_parameter('csv_prefix').value

        self.namespaces = [s.strip() for s in ns_str.split(',')]

        # Estado por namespace: x, y, z, vx, vy, vz, yaw, t
        self._lock  = threading.Lock()
        self._state = {ns: dict(x=0.0, y=0.0, z=0.0,
                                vx=0.0, vy=0.0, vz=0.0,
                                yaw=0.0, t=0.0)
                       for ns in self.namespaces}

        # Historial para gráficas post-vuelo: listas por namespace
        self._hist = {ns: dict(t=[], x=[], y=[], z=[], yaw=[])
                      for ns in self.namespaces}

        # CSV ─────────────────────────────────────────────────────────────────
        fname = f'{csv_prefix}_{int(time.time())}.csv'
        self._csv_f = open(fname, 'w', newline='')
        self._csv_w = csv.writer(self._csv_f)
        header = ['t', 'ns', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw']
        self._csv_w.writerow(header)
        self._t0 = time.monotonic()
        self.get_logger().info(f'CSV: {fname}')

        # Suscripciones por namespace ─────────────────────────────────────────
        for ns in self.namespaces:
            self.create_subscription(
                PoseStamped,
                f'/{ns}/local_position/pose',
                lambda msg, n=ns: self._cb_pose(msg, n),
                _STATE_QOS
            )
            self.create_subscription(
                TwistStamped,
                f'/{ns}/local_position/velocity_local',
                lambda msg, n=ns: self._cb_vel(msg, n),
                _STATE_QOS
            )

        # Timer de logging a CSV (10 Hz) ──────────────────────────────────────
        self.create_timer(0.1, self._log_tick)

        # Gráfica en vivo (hilo separado para no bloquear el executor) ─────────
        self._plot_lines  = {}
        self._plot_data   = {ns: dict(x=[], y=[]) for ns in self.namespaces}
        if self.live:
            self._plot_thread = threading.Thread(
                target=self._plot_loop,
                args=(plot_ivl,),
                daemon=True,
                name='telem-plot'
            )
            self._plot_thread.start()

        self.get_logger().info(
            f'TelemetryNode: namespaces={self.namespaces} '
            f'live_plot={self.live}'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped, ns: str):
        with self._lock:
            s = self._state[ns]
            s['x']   = msg.pose.position.x
            s['y']   = msg.pose.position.y
            s['z']   = msg.pose.position.z
            s['yaw'] = _quat_to_yaw(msg.pose.orientation)
            s['t']   = time.monotonic() - self._t0

    def _cb_vel(self, msg: TwistStamped, ns: str):
        with self._lock:
            s = self._state[ns]
            s['vx'] = msg.twist.linear.x
            s['vy'] = msg.twist.linear.y
            s['vz'] = msg.twist.linear.z

    # ── Logging CSV ──────────────────────────────────────────────────────────
    def _log_tick(self):
        with self._lock:
            snapshot = {ns: dict(s) for ns, s in self._state.items()}

        for ns, s in snapshot.items():
            self._csv_w.writerow([
                f'{s["t"]:.4f}', ns,
                f'{s["x"]:.4f}', f'{s["y"]:.4f}', f'{s["z"]:.4f}',
                f'{s["vx"]:.4f}', f'{s["vy"]:.4f}', f'{s["vz"]:.4f}',
                f'{s["yaw"]:.4f}'
            ])
            # Acumular historial para post-vuelo
            h = self._hist[ns]
            h['t'].append(s['t'])
            h['x'].append(s['x'])
            h['y'].append(s['y'])
            h['z'].append(s['z'])
            h['yaw'].append(s['yaw'])

    # ── Gráfica en vivo ───────────────────────────────────────────────────────
    def _plot_loop(self, interval: float):
        """Corre en hilo separado. Matplotlib debe usarse desde el mismo hilo."""
        import matplotlib
        matplotlib.use('TkAgg')       # backend no interactivo en hilo
        import matplotlib.pyplot as plt

        colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'tab:red', 'tab:purple', 'tab:brown']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Telemetría en vivo', fontsize=12)

        ax_xy  = axes[0]
        ax_z   = axes[1]

        lines_xy = {}
        lines_z  = {}

        for i, ns in enumerate(self.namespaces):
            c = colors[i % len(colors)]
            lines_xy[ns], = ax_xy.plot([], [], '-', color=c,
                                        linewidth=1.5, label=ns)
            lines_z[ns],  = ax_z.plot([], [], '-', color=c,
                                       linewidth=1.5, label=ns)

        ax_xy.set_xlabel('X — este [m]')
        ax_xy.set_ylabel('Y — norte [m]')
        ax_xy.set_title('Trayectoria XY')
        ax_xy.legend(); ax_xy.grid(True, alpha=0.4)

        ax_z.set_xlabel('Tiempo [s]')
        ax_z.set_ylabel('Z — altitud [m]')
        ax_z.set_title('Altitud')
        ax_z.legend(); ax_z.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.ion()
        plt.show()

        while True:
            try:
                with self._lock:
                    snap = {ns: dict(h) for ns, h in self._hist.items()}

                for ns in self.namespaces:
                    h = snap[ns]
                    lines_xy[ns].set_data(h['x'], h['y'])
                    lines_z[ns].set_data(h['t'], h['z'])

                ax_xy.relim(); ax_xy.autoscale_view()
                ax_z.relim();  ax_z.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                time.sleep(interval)
            except Exception:
                break

    # ── Gráficas post-vuelo ───────────────────────────────────────────────────
    def plot_post_flight(self):
        """
        Genera gráficas completas del vuelo registrado.
        Llamar después de detener el nodo (no bloquea si no se muestra aún).
        """
        import matplotlib.pyplot as plt
        import numpy as np

        colors = ['tab:blue', 'tab:orange', 'tab:green',
                  'tab:red', 'tab:purple', 'tab:brown']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Post-vuelo — trayectorias registradas', fontsize=13)

        ax_xy  = axes[0, 0]
        ax_z   = axes[0, 1]
        ax_vx  = axes[1, 0]
        ax_vy  = axes[1, 1]

        for i, ns in enumerate(self.namespaces):
            c = colors[i % len(colors)]
            h = self._hist[ns]
            if not h['t']:
                continue
            t = np.array(h['t'])
            x = np.array(h['x'])
            y = np.array(h['y'])
            z = np.array(h['z'])

            ax_xy.plot(x, y, '-', color=c, lw=1.5, label=ns)
            ax_xy.scatter(x[0], y[0], c=c, marker='o', s=50, zorder=5)
            ax_z.plot(t, z, '-', color=c, lw=1.5, label=ns)

        ax_xy.set(xlabel='X [m]', ylabel='Y [m]', title='Trayectoria XY')
        ax_xy.axis('equal'); ax_xy.legend(); ax_xy.grid(True, alpha=0.4)

        ax_z.set(xlabel='Tiempo [s]', ylabel='Z [m]', title='Altitud')
        ax_z.legend(); ax_z.grid(True, alpha=0.4)

        # Eliminar subplots vacíos si no hay más datos
        for ax in (ax_vx, ax_vy):
            ax.set_visible(False)

        plt.tight_layout()
        plt.show(block=True)

    def destroy_node(self):
        self._csv_f.flush()
        self._csv_f.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Deteniendo telemetría...')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        # Gráficas post-vuelo al salir
        # node.plot_post_flight()   # descomentar si se quiere automático


if __name__ == '__main__':
    main()

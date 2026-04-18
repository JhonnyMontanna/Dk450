#!/usr/bin/env python3
"""
master_node.py — Nodo maestro con máquina de estados por terminal
=================================================================
ROS2 Humble + MAVROS2. Archivo autónomo, sin dependencias internas.

Controla un dron a través de una máquina de estados interactiva.
Cada estado instancia el nodo correspondiente y lo corre en el mismo
executor ROS2, luego lo destruye limpiamente al salir del modo.

Estados disponibles:
  IDLE       → en espera, sin enviar comandos
  TAKEOFF    → despegue a altitud configurable
  LAND       → aterrizaje
  KEYBOARD   → control manual con W/A/S/D/R/F (requiere 'pynput')
  CIRCLE     → círculo con path following (parámetros configurables)
  LEMNISCATE → lemniscata de Bernoulli (parámetros configurables)
  SQUARE     → waypoints en cuadrado (parámetros configurables)
  FOLLOW     → controlador líder-seguidor PID (especificar líder)

Parámetros ROS2:
  uav_ns     : namespace de este dron   (default: 'uav1')
  rate       : frecuencia base [Hz]     (default: 20)

Uso:
  python3 master_node.py --ros-args -p uav_ns:=uav1

Dependencias extra para control por teclado:
  pip install pynput
"""

import math
import time
import threading
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# ─────────────────────────────────────────────────────────────────────────────
# QoS compartido
# ─────────────────────────────────────────────────────────────────────────────
_MAVROS_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
_STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ─────────────────────────────────────────────────────────────────────────────
# Estados
# ─────────────────────────────────────────────────────────────────────────────
class State_:
    IDLE       = 'IDLE'
    TAKEOFF    = 'TAKEOFF'
    LAND       = 'LAND'
    KEYBOARD   = 'KEYBOARD'
    CIRCLE     = 'CIRCLE'
    LEMNISCATE = 'LEMNISCATE'
    SQUARE     = 'SQUARE'
    FOLLOW     = 'FOLLOW'

# ─────────────────────────────────────────────────────────────────────────────
# Helpers inline
# ─────────────────────────────────────────────────────────────────────────────
def _quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )

def _make_pose(node, x, y, z, yaw) -> PoseStamped:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    msg = PoseStamped()
    msg.header.stamp    = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.z = float(sy)
    msg.pose.orientation.w = float(cy)
    return msg

def _make_twist(node, vx, vy, vz, yr=0.0) -> TwistStamped:
    msg = TwistStamped()
    msg.header.stamp    = node.get_clock().now().to_msg()
    msg.header.frame_id = 'map'
    msg.twist.linear.x  = float(vx)
    msg.twist.linear.y  = float(vy)
    msg.twist.linear.z  = float(vz)
    msg.twist.angular.z = float(yr)
    return msg

def _clamp(v, lim):
    return max(-lim, min(lim, v))

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

# ─────────────────────────────────────────────────────────────────────────────
# Nodo maestro
# ─────────────────────────────────────────────────────────────────────────────
class MasterNode(Node):

    def __init__(self):
        super().__init__('master_node')

        self.declare_parameter('uav_ns', 'uav1')
        self.declare_parameter('rate',   20)

        self.ns   = self.get_parameter('uav_ns').value
        self.rate = self.get_parameter('rate').value
        self.dt   = 1.0 / self.rate

        # ── Estado actual y flag de parada del modo activo ────────────────────
        self._state       = State_.IDLE
        self._stop_mode   = threading.Event()   # señal para detener el modo activo
        self._mode_thread = None                # hilo del modo activo

        # ── Posición y velocidad del dron (actualizado por callbacks) ─────────
        self._lock = threading.Lock()
        self._pos  = None    # (x, y, z) ENU
        self._vel  = None    # (vx, vy, vz)
        self._yaw  = None

        # ── Suscripciones de estado del dron ──────────────────────────────────
        self.create_subscription(
            PoseStamped,
            f'/{self.ns}/local_position/pose',
            self._cb_pose, _STATE_QOS
        )
        self.create_subscription(
            TwistStamped,
            f'/{self.ns}/local_position/velocity_local',
            self._cb_vel, _STATE_QOS
        )

        # ── Publicadores ──────────────────────────────────────────────────────
        self._pub_pos = self.create_publisher(
            PoseStamped,
            f'/{self.ns}/setpoint_position/local',
            _MAVROS_QOS
        )
        self._pub_vel = self.create_publisher(
            TwistStamped,
            f'/{self.ns}/setpoint_velocity/cmd_vel',
            _MAVROS_QOS
        )

        # ── Clientes de servicio MAVROS ───────────────────────────────────────
        self._arm_cli  = self.create_client(
            CommandBool, f'/{self.ns}/mavros/cmd/arming'
        )
        self._mode_cli = self.create_client(
            SetMode, f'/{self.ns}/mavros/set_mode'
        )
        self._tol_cli  = self.create_client(
            CommandTOL, f'/{self.ns}/mavros/cmd/takeoff'
        )
        self._land_cli = self.create_client(
            CommandTOL, f'/{self.ns}/mavros/cmd/land'
        )

        # ── Hilo de la terminal (interfaz de usuario) ─────────────────────────
        self._ui_thread = threading.Thread(
            target=self._ui_loop,
            daemon=True,
            name='master-ui'
        )
        self._ui_thread.start()

        self.get_logger().info(
            f'MasterNode iniciado — namespace: {self.ns}\n'
            f'Escribe "help" en la terminal para ver comandos.'
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _cb_pose(self, msg: PoseStamped):
        with self._lock:
            p = msg.pose.position
            self._pos = (p.x, p.y, p.z)
            self._yaw = _quat_to_yaw(msg.pose.orientation)

    def _cb_vel(self, msg: TwistStamped):
        with self._lock:
            v = msg.twist.linear
            self._vel = (v.x, v.y, v.z)

    def _get_pos(self):
        with self._lock:
            return self._pos, self._vel, self._yaw

    # ── Servicios MAVROS ──────────────────────────────────────────────────────
    def _set_guided(self):
        if not self._mode_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Servicio set_mode no disponible')
            return False
        req = SetMode.Request()
        req.custom_mode = 'GUIDED'
        fut = self._mode_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        return fut.result() and fut.result().mode_sent

    def _arm(self):
        if not self._arm_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Servicio arming no disponible')
            return False
        req = CommandBool.Request()
        req.value = True
        fut = self._arm_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        return fut.result() and fut.result().success

    def _call_takeoff(self, altitude: float):
        if not self._tol_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Servicio takeoff no disponible')
            return False
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        fut = self._tol_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() and fut.result().success

    def _call_land(self):
        if not self._land_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Servicio land no disponible')
            return False
        req = CommandTOL.Request()
        fut = self._land_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() and fut.result().success

    # ── Cambio de modo ─────────────────────────────────────────────────────────
    def _stop_current_mode(self):
        """Señaliza al modo activo que se detenga y espera a que termine."""
        self._stop_mode.set()
        if self._mode_thread and self._mode_thread.is_alive():
            self._mode_thread.join(timeout=3.0)
        self._stop_mode.clear()
        self._mode_thread = None
        self._state = State_.IDLE

    def _start_mode(self, target, fn, args=()):
        self._stop_current_mode()
        self._state = target
        self._mode_thread = threading.Thread(
            target=fn, args=args, daemon=True,
            name=f'mode-{target.lower()}'
        )
        self._mode_thread.start()
        self.get_logger().info(f'Modo activo: {target}')

    # ─────────────────────────────────────────────────────────────────────────
    # IMPLEMENTACIONES DE MODOS
    # ─────────────────────────────────────────────────────────────────────────

    # ── TAKEOFF ───────────────────────────────────────────────────────────────
    def _mode_takeoff(self, altitude: float):
        self.get_logger().info(f'Takeoff → {altitude} m')
        ok_g = self._set_guided()
        ok_a = self._arm()
        if not (ok_g and ok_a):
            self.get_logger().warn('No se pudo armar/guided — abortando takeoff')
            self._state = State_.IDLE
            return
        self._call_takeoff(altitude)
        # Esperar hasta alcanzar altitud o stop
        while not self._stop_mode.is_set():
            pos, _, _ = self._get_pos()
            if pos and pos[2] >= altitude * 0.95:
                self.get_logger().info(f'Altitud alcanzada: {pos[2]:.2f} m')
                break
            time.sleep(0.2)
        self._state = State_.IDLE

    # ── LAND ──────────────────────────────────────────────────────────────────
    def _mode_land(self):
        self.get_logger().info('Aterrizando...')
        self._call_land()
        self._state = State_.IDLE

    # ── KEYBOARD ──────────────────────────────────────────────────────────────
    def _mode_keyboard(self, speed: float):
        """
        Control por teclado usando pynput (no requiere root en Linux).
        Teclas: W/S → X (norte/sur), A/D → Y (oeste/este),
                R/F → Z (subir/bajar), ESC o Q → salir del modo.
        """
        try:
            from pynput import keyboard as kb
        except ImportError:
            self.get_logger().error(
                'pynput no instalado. Ejecuta: pip install pynput'
            )
            self._state = State_.IDLE
            return

        pressed = set()

        def on_press(key):
            try:
                pressed.add(key.char.lower())
            except AttributeError:
                pressed.add(key)

        def on_release(key):
            try:
                pressed.discard(key.char.lower())
            except AttributeError:
                pressed.discard(key)

        print(
            '\n--- MODO TECLADO ---\n'
            '  W/S : norte/sur\n'
            '  A/D : oeste/este\n'
            '  R/F : subir/bajar\n'
            '  Q o ESC : salir\n'
            '--------------------\n'
        )

        listener = kb.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        next_t = time.monotonic()
        while not self._stop_mode.is_set():
            vx = vy = vz = 0.0

            if 'w' in pressed:  vx += speed
            if 's' in pressed:  vx -= speed
            if 'd' in pressed:  vy += speed
            if 'a' in pressed:  vy -= speed
            if 'r' in pressed:  vz += speed
            if 'f' in pressed:  vz -= speed
            if 'q' in pressed or kb.Key.esc in pressed:
                break

            self._pub_vel.publish(_make_twist(self, vx, vy, vz))

            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

        listener.stop()
        self._pub_vel.publish(_make_twist(self, 0.0, 0.0, 0.0))
        self._state = State_.IDLE

    # ── CIRCLE ────────────────────────────────────────────────────────────────
    def _mode_circle(self, radius: float, omega: float,
                     lookahead_t: float, kp_r: float):
        """Path following circular en lazo cerrado (lógica de circle_node)."""
        lookahead = omega * lookahead_t
        cx = cy = cz = None

        self.get_logger().info(
            f'Círculo: R={radius}m ω={omega}rad/s '
            f'lookahead={math.degrees(lookahead):.1f}° kp_r={kp_r}'
        )

        next_t = time.monotonic()
        while not self._stop_mode.is_set():
            pos, _, _ = self._get_pos()
            if pos is None:
                time.sleep(0.02)
                continue

            x, y, z = pos
            if cx is None:
                cx, cy, cz = x - radius, y, z
                self.get_logger().info(
                    f'Centro círculo: ({cx:.2f},{cy:.2f},{cz:.2f})'
                )

            dx, dy = x - cx, y - cy
            r      = math.hypot(dx, dy)
            theta  = math.atan2(dy, dx)
            err_r  = r - radius

            theta_sp = theta + lookahead
            x_sp = cx + radius * math.cos(theta_sp)
            y_sp = cy + radius * math.sin(theta_sp)

            if r > 1e-3:
                x_sp -= kp_r * err_r * (dx / r)
                y_sp -= kp_r * err_r * (dy / r)

            self._pub_pos.publish(
                _make_pose(self, x_sp, y_sp, cz, theta_sp + math.pi / 2)
            )

            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

        self._state = State_.IDLE

    # ── LEMNISCATE ────────────────────────────────────────────────────────────
    def _mode_lemniscate(self, a: float, b: float,
                         omega: float, start_mode: str):
        """Lemniscata de Bernoulli (lógica de lemniscate_node)."""
        def lem_pos(t):
            s, c = math.sin(t), math.cos(t)
            d = 1.0 + s * s
            return a * c / d, b * s * c / d

        def lem_yaw(t):
            s, c = math.sin(t), math.cos(t)
            d = 1.0 + s * s
            vx = a * (-s * d - c * 2.0 * s * c) / (d * d) * omega
            vy = b * ((c*c - s*s) * d - s * c * 2.0 * s * c) / (d * d) * omega
            return math.atan2(vy, vx) if abs(vx) > 1e-6 or abs(vy) > 1e-6 else 0.0

        # Esperar posición inicial
        while not self._stop_mode.is_set():
            pos, _, _ = self._get_pos()
            if pos is not None:
                break
            time.sleep(0.05)

        if self._stop_mode.is_set():
            self._state = State_.IDLE
            return

        x0, y0, z0 = pos
        if start_mode == 'center':
            cx, cy = x0, y0
            t0_p   = math.pi / 2
        else:
            cx, cy = x0 - a, y0
            t0_p   = 0.0

        self.get_logger().info(
            f'Lemniscata: a={a}m b={b}m ω={omega}rad/s modo={start_mode}'
        )

        next_t = time.monotonic()
        t_start = time.monotonic()

        while not self._stop_mode.is_set():
            t_s   = time.monotonic() - t_start
            theta = t0_p + omega * t_s
            lx, ly = lem_pos(theta)
            x_sp   = cx + lx
            y_sp   = cy + ly
            yaw    = lem_yaw(theta)

            self._pub_pos.publish(_make_pose(self, x_sp, y_sp, z0, yaw))

            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

        self._state = State_.IDLE

    # ── SQUARE ────────────────────────────────────────────────────────────────
    def _mode_square(self, side: float, altitude: float,
                     conv_r: float, conv_v: float):
        """Cuadrado en lazo cerrado (convergencia por waypoint)."""
        while not self._stop_mode.is_set():
            pos, _, _ = self._get_pos()
            if pos is not None:
                break
            time.sleep(0.05)

        if self._stop_mode.is_set():
            self._state = State_.IDLE
            return

        x0, y0, z0 = pos
        # La altitud del cuadrado es RELATIVA a la posición actual del dron
        # Si altitude=3.0 y el dron ya está a z0=5.0, el cuadrado vuela a z0+3=8 ✗
        # Por eso usamos z0 directamente y altitude como offset opcional
        tz = z0 if altitude == 0.0 else altitude
        wps = [
            (x0,        y0,        tz),
            (x0 + side, y0,        tz),
            (x0 + side, y0 + side, tz),
            (x0,        y0 + side, tz),
            (x0,        y0,        tz),
        ]

        self.get_logger().info(
            f'Cuadrado: lado={side}m z={tz:.2f}m '
            f'({len(wps)} waypoints) conv_r={conv_r}m conv_v={conv_v}m/s'
        )

        for idx, (tx, ty, tz_wp) in enumerate(wps):
            if self._stop_mode.is_set():
                break
            self.get_logger().info(
                f'  WP{idx+1}/{len(wps)}: ({tx:.1f},{ty:.1f},{tz_wp:.1f})'
            )
            zone_t        = None
            next_t        = time.monotonic()
            timeout_start = time.monotonic()

            while not self._stop_mode.is_set():
                now = time.monotonic()
                if now - timeout_start > 30.0:
                    self.get_logger().warn(f'  WP{idx+1} timeout — avanzando')
                    break

                self._pub_pos.publish(_make_pose(self, tx, ty, tz_wp, 0.0))
                pos, vel, _ = self._get_pos()

                if pos is not None:
                    x, y, z = pos
                    if vel is not None:
                        vx, vy, vz = vel
                        speed = math.sqrt(vx**2 + vy**2 + vz**2)
                    else:
                        speed = float('inf')   # sin vel → no convergemos aún

                    dist = math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz_wp)**2)
                    self.get_logger().debug(
                        f'  WP{idx+1} dist={dist:.3f}m speed={speed:.3f}m/s'
                    )

                    if dist < conv_r and speed < conv_v:
                        zone_t = zone_t or now
                        if now - zone_t >= 1.0:
                            self.get_logger().info(
                                f'  WP{idx+1} OK en {now-timeout_start:.1f}s '
                                f'dist={dist:.3f}m speed={speed:.3f}m/s'
                            )
                            break
                    else:
                        zone_t = None

                next_t += self.dt
                s = next_t - now
                if s > 0:
                    time.sleep(s)

        self._state = State_.IDLE

    # ── FOLLOW ────────────────────────────────────────────────────────────────
    def _mode_follow(self, leader_ns: str, d: float, alpha: float, dz: float,
                     kp: float, ki: float, kd: float,
                     kp_y: float, kd_y: float):
        """PID líder-seguidor con prealimentación (lógica de leader_follower_node)."""
        # Estado del líder
        leader_state = {k: None for k in
                        ('x','y','z','vx','vy','vz','yaw','yaw_rate')}
        leader_lock  = threading.Lock()

        def cb_lpose(msg):
            with leader_lock:
                p = msg.pose.position
                leader_state['x']   = p.x
                leader_state['y']   = p.y
                leader_state['z']   = p.z
                leader_state['yaw'] = _quat_to_yaw(msg.pose.orientation)

        def cb_lvel(msg):
            with leader_lock:
                v = msg.twist.linear
                leader_state['vx']       = v.x
                leader_state['vy']       = v.y
                leader_state['vz']       = v.z
                leader_state['yaw_rate'] = msg.twist.angular.z

        # Suscripciones temporales al líder
        sub_p = self.create_subscription(
            PoseStamped,
            f'/{leader_ns}/local_position/pose',
            cb_lpose, _STATE_QOS
        )
        sub_v = self.create_subscription(
            TwistStamped,
            f'/{leader_ns}/local_position/velocity_local',
            cb_lvel, _STATE_QOS
        )

        # PID state
        integ     = [0.0, 0.0, 0.0]
        integ_yaw = 0.0
        i_lim     = 2.0
        iy_lim    = 1.0
        v_max     = 3.0
        yr_max    = 1.0

        self.get_logger().info(
            f'Seguidor de {leader_ns}: d={d}m α={math.degrees(alpha):.1f}° '
            f'Kp={kp} Ki={ki} Kd={kd}'
        )

        next_t = time.monotonic()
        while not self._stop_mode.is_set():
            with leader_lock:
                L = dict(leader_state)
            pos, vel, yaw = self._get_pos()

            keys = ('x','y','z','vx','vy','vz','yaw','yaw_rate')
            if any(L[k] is None for k in keys) or pos is None:
                time.sleep(0.02)
                continue

            Sx, Sy, Sz = pos
            Svx, Svy, Svz = vel or (0, 0, 0)
            S_yaw = yaw or 0.0

            # Offset polar rotante
            angle = L['yaw'] + alpha
            dx = d * math.cos(angle)
            dy = d * math.sin(angle)

            xd = L['x'] + dx
            yd = L['y'] + dy
            zd = L['z'] + dz

            ex = xd - Sx
            ey = yd - Sy
            ez = zd - Sz

            integ[0] = _clamp(integ[0] + ex * self.dt, i_lim)
            integ[1] = _clamp(integ[1] + ey * self.dt, i_lim)
            integ[2] = _clamp(integ[2] + ez * self.dt, i_lim)

            dv_x = L['vx'] - Svx
            dv_y = L['vy'] - Svy
            dv_z = L['vz'] - Svz

            ff_x = L['yaw_rate'] * (-dy)
            ff_y = L['yaw_rate'] * ( dx)

            vx = ff_x + kp*ex + ki*integ[0] + kd*dv_x
            vy = ff_y + kp*ey + ki*integ[1] + kd*dv_y
            vz =        kp*ez + ki*integ[2] + kd*dv_z

            v_h = math.hypot(vx, vy)
            if v_h > v_max:
                vx *= v_max / v_h
                vy *= v_max / v_h
            vz = _clamp(vz, v_max)

            e_yaw = _wrap(L['yaw'] - S_yaw)
            integ_yaw = _clamp(integ_yaw + e_yaw * self.dt, iy_lim)
            yr_cmd = _clamp(
                kp_y * e_yaw + kd_y * (L['yaw_rate'] - 0.0),
                yr_max
            )

            self._pub_vel.publish(_make_twist(self, vx, vy, vz, yr_cmd))

            next_t += self.dt
            s = next_t - time.monotonic()
            if s > 0:
                time.sleep(s)

        # Limpiar suscripciones temporales
        self.destroy_subscription(sub_p)
        self.destroy_subscription(sub_v)
        self._pub_vel.publish(_make_twist(self, 0.0, 0.0, 0.0))
        self._state = State_.IDLE

    # ─────────────────────────────────────────────────────────────────────────
    # INTERFAZ DE TERMINAL
    # ─────────────────────────────────────────────────────────────────────────
    def _ui_loop(self):
        """Bucle de terminal en hilo separado. Lee comandos del usuario."""
        time.sleep(1.0)   # esperar que el nodo arranque
        self._print_help()

        while rclpy.ok():
            try:
                raw = input(f'\n[{self.ns}|{self._state}] > ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue
            parts = raw.split()
            cmd   = parts[0].lower()
            args  = parts[1:]
            self._dispatch(cmd, args)

    def _dispatch(self, cmd: str, args: list):
        """Despacha el comando al modo correspondiente."""

        if cmd in ('help', 'h', '?'):
            self._print_help()

        elif cmd == 'status':
            pos, vel, yaw = self._get_pos()
            print(f'  Estado:   {self._state}')
            print(f'  Posición: {pos}')
            print(f'  Yaw:      {math.degrees(yaw):.1f}°' if yaw else '  Yaw: None')

        elif cmd == 'stop':
            print('  Deteniendo modo activo...')
            self._stop_current_mode()
            self._pub_vel.publish(_make_twist(self, 0.0, 0.0, 0.0))

        elif cmd == 'takeoff':
            alt = float(args[0]) if args else 2.0
            self._start_mode(State_.TAKEOFF, self._mode_takeoff, (alt,))

        elif cmd == 'land':
            self._start_mode(State_.LAND, self._mode_land)

        elif cmd == 'keyboard':
            spd = float(args[0]) if args else 0.5
            self._start_mode(State_.KEYBOARD, self._mode_keyboard, (spd,))

        elif cmd == 'circle':
            # circle [radius] [omega] [lookahead_time] [kp_r]
            r   = float(args[0]) if len(args) > 0 else 4.0
            w   = float(args[1]) if len(args) > 1 else 0.5
            lt  = float(args[2]) if len(args) > 2 else 1.5
            kpr = float(args[3]) if len(args) > 3 else 0.5
            print(f'  Círculo: R={r}m ω={w}rad/s lookahead={lt}s kp_r={kpr}')
            self._start_mode(State_.CIRCLE, self._mode_circle, (r, w, lt, kpr))

        elif cmd == 'lemniscate':
            # lemniscate [a] [b] [omega] [start_mode]
            a    = float(args[0]) if len(args) > 0 else 4.0
            b    = float(args[1]) if len(args) > 1 else 2.0
            w    = float(args[2]) if len(args) > 2 else 0.3
            smod = args[3]        if len(args) > 3 else 'center'
            print(f'  Lemniscata: a={a}m b={b}m ω={w}rad/s modo={smod}')
            self._start_mode(State_.LEMNISCATE, self._mode_lemniscate,
                             (a, b, w, smod))

        elif cmd == 'square':
            # square [side] [altitude] [conv_r] [conv_v]
            side = float(args[0]) if len(args) > 0 else 5.0
            alt  = float(args[1]) if len(args) > 1 else 0.0
            cr   = float(args[2]) if len(args) > 2 else 0.30
            cv   = float(args[3]) if len(args) > 3 else 0.15
            print(f'  Cuadrado: lado={side}m alt={alt}m')
            self._start_mode(State_.SQUARE, self._mode_square,
                             (side, alt, cr, cv))

        elif cmd == 'follow':
            # follow [leader_ns] [d] [alpha_deg] [dz] [kp] [ki] [kd] [kp_y] [kd_y]
            if not args:
                print('  Uso: follow <leader_ns> [d] [alpha_deg] [dz] '
                      '[kp] [ki] [kd] [kp_y] [kd_y]')
                return
            lns  = args[0]
            d    = float(args[1]) if len(args) > 1 else 2.0
            alph = math.radians(float(args[2])) if len(args) > 2 else math.pi
            dz   = float(args[3]) if len(args) > 3 else 0.0
            kp   = float(args[4]) if len(args) > 4 else 0.5
            ki   = float(args[5]) if len(args) > 5 else 0.05
            kd   = float(args[6]) if len(args) > 6 else 0.2
            kpy  = float(args[7]) if len(args) > 7 else 0.8
            kdy  = float(args[8]) if len(args) > 8 else 0.1
            print(f'  Siguiendo a {lns}: d={d}m α={math.degrees(alph):.1f}°')
            self._start_mode(State_.FOLLOW, self._mode_follow,
                             (lns, d, alph, dz, kp, ki, kd, kpy, kdy))

        else:
            print(f'  Comando desconocido: "{cmd}". Escribe "help".')

    @staticmethod
    def _print_help():
        print("""
╔══════════════════════════════════════════════════════════════╗
║              NODO MAESTRO — COMANDOS DISPONIBLES            ║
╠══════════════════════════════════════════════════════════════╣
║  help                        → esta ayuda                   ║
║  status                      → posición y estado actual     ║
║  stop                        → detener modo activo          ║
║                                                              ║
║  takeoff [altitud=2.0]       → despegar a altitud (m)       ║
║  land                        → aterrizar                    ║
║                                                              ║
║  keyboard [velocidad=0.5]    → control manual por teclado   ║
║    W/S=norte/sur  A/D=este/oeste  R/F=subir/bajar  Q=salir  ║
║                                                              ║
║  circle [R=4] [ω=0.5] [la=1.5] [kp_r=0.5]                 ║
║    R=radio(m)  ω=vel.angular(rad/s)                         ║
║    la=lookahead(s)  kp_r=ganancia radial                    ║
║                                                              ║
║  lemniscate [a=4] [b=2] [ω=0.3] [modo=center]              ║
║    a=semieje X(m)  b=semieje Y(m)  modo=center|tip          ║
║                                                              ║
║  square [lado=5] [alt=0] [cr=0.3] [cv=0.15]                ║
║    lado(m)  alt=z absoluto (0=usar z actual)                ║
║             cr=radio conv  cv=vel conv                      ║
║                                                              ║
║  follow <lider_ns> [d=2] [α°=180] [Δz=0]                   ║
║         [Kp=0.5] [Ki=0.05] [Kd=0.2] [Kp_ψ=0.8] [Kd_ψ=0.1]║
║    lider_ns: namespace del dron líder (ej: uav1)            ║
╚══════════════════════════════════════════════════════════════╝
""")

    def destroy_node(self):
        self._stop_current_mode()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRADA
# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = MasterNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
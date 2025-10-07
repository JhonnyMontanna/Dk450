#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import threading
import time
from collections import deque, defaultdict

import matplotlib.pyplot as plt
from pymavlink import mavutil

# -------------------------
# Configuración por defecto
# -------------------------
DEFAULT_CONN = "udp:127.0.0.1:14552"  # Cámbialo si necesitas otro puerto/host
HISTORY_SECONDS = 15                  # Ventana de tiempo a mostrar en el gráfico
SAMPLE_HZ = 30                        # Frecuencia objetivo para refrescar buffers
YLIM = (900, 2100)                    # Rango típico PWM para visualizar

# Rango permitido por canal (puedes ajustar por canal)
DEFAULT_LIMITS = {
    1: (987, 2011),
    2: (987, 2011),
    3: (987, 2011),
    4: (987, 2011),
    5: (987, 2011),
    6: (0, 2011),
    7: (0, 2011),
    8: (0, 2011),
}

# -------------------------
# Hilo de lectura MAVLink
# -------------------------
class RCReader(threading.Thread):
    def __init__(self, conn_str, buffers, limits, stop_event):
        super().__init__(daemon=True)
        self.conn_str = conn_str
        self.buffers = buffers
        self.limits = limits
        self.stop_event = stop_event
        self.master = None
        self.last_warn_time = defaultdict(float)
        self.warn_cooldown = 0.5  # s para no inundar prints

    def _connect(self):
        print(f"[INFO] Conectando a {self.conn_str} ...")
        self.master = mavutil.mavlink_connection(self.conn_str)
        self.master.wait_heartbeat(timeout=10)
        print(f"[OK] Heartbeat recibido del sistema {self.master.target_system}, componente {self.master.target_component}")

    def run(self):
        try:
            self._connect()
        except Exception as e:
            print(f"[ERROR] No se pudo conectar: {e}")
            self.stop_event.set()
            return

        # Bucle de recepción
        while not self.stop_event.is_set():
            # Intenta RC_CHANNELS primero (tiene 18 canales posibles en 2 mensajes)
            msg = self.master.recv_match(type=['RC_CHANNELS', 'RC_CHANNELS_RAW'], blocking=True, timeout=1.0)
            t = time.time()
            if msg is None:
                continue

            # Normaliza a un dict ch->valor PWM
            rc_values = {}

            if msg.get_type() == 'RC_CHANNELS':
                # ch1..ch8 están en msg.chan1_raw ... msg.chan8_raw (y hasta chan18 según versión)
                for i in range(1, 9):
                    val = getattr(msg, f'chan{i}_raw', 0) or 0
                    rc_values[i] = int(val)
            elif msg.get_type() == 'RC_CHANNELS_RAW':
                # Este mensaje suele traer chan1_raw..chan4_raw
                rc_values[1] = int(getattr(msg, 'chan1_raw', 0) or 0)
                rc_values[2] = int(getattr(msg, 'chan2_raw', 0) or 0)
                rc_values[3] = int(getattr(msg, 'chan3_raw', 0) or 0)
                rc_values[4] = int(getattr(msg, 'chan4_raw', 0) or 0)

            # Actualiza buffers y valida rangos
            for ch, val in rc_values.items():
                # Guarda tiempo y valor
                self.buffers['t'][ch].append(t)
                self.buffers['v'][ch].append(val)

                # Chequeo de límites (si existen para ese canal)
                if ch in self.limits:
                    low, high = self.limits[ch]
                    out_of_range = (val < low) or (val > high) or (val == -1)
                    now = time.time()
                    if out_of_range and (now - self.last_warn_time[ch] > self.warn_cooldown):
                        if val == -1:
                            print(f"[WARN] CH{ch}: sin señal (valor = -1)")
                        elif val < low:
                            print(f"[WARN] CH{ch}: {val} debajo del mínimo ({low})")
                        elif val > high:
                            print(f"[WARN] CH{ch}: {val} encima del máximo ({high})")
                        self.last_warn_time[ch] = now

            # Limita la tasa de muestreo objetivo
            time.sleep(max(0.0, 1.0 / SAMPLE_HZ))

# -------------------------
# Plot en vivo
# -------------------------
def live_plot(buffers, stop_event):
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("RC Channels Monitor (pymavlink)")
    ax.set_title("Canales RC (últimos {} s)".format(HISTORY_SECONDS))
    ax.set_xlabel("Tiempo (s relativo)")
    ax.set_ylabel("PWM")
    ax.set_ylim(*YLIM)
    ax.grid(True, which='both', alpha=0.3)

    # Prepara líneas para CH1..CH8
    lines = {}
    for ch in range(1, 9):
        (ln,) = ax.plot([], [], label=f"CH{ch}")
        lines[ch] = ln

    ax.legend(loc="upper right", ncol=2)

    # Bucle de actualización
    while not stop_event.is_set():
        try:
            now = time.time()
            t_min = now - HISTORY_SECONDS

            for ch in range(1, 9):
                # Filtra los puntos dentro de la ventana
                tbuf = buffers['t'][ch]
                vbuf = buffers['v'][ch]

                # Limpieza rápida de buffers antiguos (pop left)
                while len(tbuf) > 0 and tbuf[0] < t_min:
                    tbuf.popleft()
                    vbuf.popleft()

                if len(tbuf) >= 2:
                    t_rel = [ti - tbuf[-1] for ti in tbuf]  # tiempo relativo (último = 0)
                    lines[ch].set_data(t_rel, list(vbuf))
                else:
                    lines[ch].set_data([], [])

            # Ajusta xlim para cubrir ventana (tiempo relativo va negativo a 0)
            ax.set_xlim(-HISTORY_SECONDS, 0)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(1.0 / 20.0)  # ~20 FPS de refresco
        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as e:
            print(f"[ERROR plot] {e}")
            time.sleep(0.2)

    plt.ioff()
    plt.close(fig)

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Monitor en vivo de canales RC vía pymavlink con advertencias por rango.")
    p.add_argument("--conn", default=DEFAULT_CONN,
                   help="Cadena de conexión MAVLink (ej. 'udp:127.0.0.1:14552', 'udp:0.0.0.0:14552')")
    p.add_argument("--history", type=int, default=HISTORY_SECONDS,
                   help="Segundos de historial a mostrar en el gráfico")
    p.add_argument("--low", type=int, default=1000, help="Límite inferior por defecto para todos los canales")
    p.add_argument("--high", type=int, default=2000, help="Límite superior por defecto para todos los canales")
    return p.parse_args()

def main():
    args = parse_args()

    # Aplica límites por defecto a todos los canales
    limits = {ch: (args.low, args.high) for ch in range(1, 9)}

    # Buffers por canal
    buffers = {
        't': {ch: deque(maxlen=args.history * SAMPLE_HZ * 2) for ch in range(1, 9)},
        'v': {ch: deque(maxlen=args.history * SAMPLE_HZ * 2) for ch in range(1, 9)},
    }

    stop_event = threading.Event()

    # Lector MAVLink en hilo
    reader = RCReader(args.conn, buffers, limits, stop_event)
    reader.start()

    # Plot en el hilo principal
    try:
        live_plot(buffers, stop_event)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        print("[INFO] Finalizado.")

if __name__ == "__main__":
    main()

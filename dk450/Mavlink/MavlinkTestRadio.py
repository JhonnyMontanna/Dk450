#!/usr/bin/env python3
"""
radio_monitor.py — Monitor de calidad de enlace dual RFD SiK
=============================================================
Lee RADIO_STATUS (MAVLink #109) de ambos pares de radios en paralelo
y calcula las métricas de calidad del enlace en tiempo real.

Métricas capturadas por radio:
  RSSI local / remoto     [0-255 → dBm]
  Noise local / remoto    [0-255 → dBm]
  SNR local / remoto      [dB]
  Paquetes recibidos      [pkts]
  Errores de TX / RX      [counts]
  Correcciones ECC        [counts]
  Packet Loss Rate        [%]  — calculado como errores/total
  Link Budget             [dB] — margen sobre el piso de ruido

Métricas derivadas del control:
  Latencia estimada       [ms]  — basada en Air Speed configurado
  Throughput disponible   [kbps]
  Utilización del canal   [%]   — MAVLink telemetría vs capacidad

Requisitos del control LF a 20 Hz:
  Latencia máxima tolerable : 50 ms  (1 ciclo de control)
  Throughput mínimo         : ~8 kbps (paquetes MAVLink típicos)
  PLR máximo tolerable      : 5%

Uso:
  python radio_monitor.py                  # monitoreo en tiempo real
  python radio_monitor.py --log 60         # monitorea 60 s y guarda CSV
  python radio_monitor.py --compare        # muestra comparativa final
"""

import argparse
import math
import time
import threading
import csv
import sys
from collections import deque
from pymavlink import mavutil

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

RADIOS = {
    'LIDER': {
        'conn':       'udp:127.0.0.1:14550',   # puerto GCS del lider
        'air_speed':  128,                      # kbps configurado en el radio
        'net_id':     25,
        'channels':   50,
        'color':      '\033[92m',               # verde
    },
    'SEGUIDOR': {
        'conn':       'udp:127.0.0.1:14551',   # puerto GCS del seguidor
        'air_speed':  128,
        'net_id':     26,
        'channels':   50,
        'color':      '\033[94m',               # azul
    },
}

# Requisitos del algoritmo de control LF @ 20 Hz
CTRL_RATE_HZ     = 20
CTRL_LATENCY_MAX = 50.0    # ms — latencia máxima tolerable (1 ciclo)
CTRL_PLR_MAX     = 5.0     # %  — packet loss rate máximo
CTRL_SNR_MIN     = 10.0    # dB — SNR mínimo para enlace estable

# Tamaño de ventana deslizante para promedios
WINDOW_SIZE = 100   # muestras

# Intervalo de polling de RADIO_STATUS [s]
POLL_INTERVAL = 0.5   # 2 Hz — suficiente para métricas de radio

# ══════════════════════════════════════════════════════════════════════════════
# CONVERSIONES SiK
# ══════════════════════════════════════════════════════════════════════════════
def rssi_to_dbm(rssi_raw):
    """
    RFD SiK: RSSI en escala 0-255 donde 255 = -54 dBm, 0 = ~-120 dBm.
    Fórmula del firmware SiK: dBm = (rssi / 1.9) - 127
    """
    if rssi_raw == 0:
        return -120.0
    return round((rssi_raw / 1.9) - 127, 1)

def noise_to_dbm(noise_raw):
    return rssi_to_dbm(noise_raw)

def snr_db(rssi_raw, noise_raw):
    rssi  = rssi_to_dbm(rssi_raw)
    noise = noise_to_dbm(noise_raw)
    return round(rssi - noise, 1)

def estimate_latency_ms(air_speed_kbps, channels):
    """
    Latencia estimada de un ciclo MAVLink basada en la configuración del radio.
    Un paquete MAVLink típico de telemetría = ~100 bytes = 800 bits.
    Más overhead de hopping (~5 ms por salto).
    """
    pkt_bits      = 800   # bits por paquete MAVLink promedio
    tx_time_ms    = (pkt_bits / (air_speed_kbps * 1000)) * 1000
    hop_overhead  = 5.0   # ms por salto de canal
    ack_time_ms   = tx_time_ms * 0.3
    return round(tx_time_ms + hop_overhead + ack_time_ms, 1)

def channel_utilization(air_speed_kbps, ctrl_rate_hz):
    """
    Porcentaje del canal usado por la telemetría del control LF.
    Asume ~100 bytes por mensaje MAVLink de posición/velocidad.
    """
    mavlink_pkt_bytes = 100
    used_kbps = (mavlink_pkt_bytes * 8 * ctrl_rate_hz) / 1000
    return round((used_kbps / air_speed_kbps) * 100, 1)

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO POR RADIO
# ══════════════════════════════════════════════════════════════════════════════
class RadioState:
    def __init__(self, name, config):
        self.name       = name
        self.config     = config
        self.connected  = False
        self.last_msg   = None
        self.t_start    = None

        # Ventanas deslizantes
        self.rssi_l_buf   = deque(maxlen=WINDOW_SIZE)
        self.rssi_r_buf   = deque(maxlen=WINDOW_SIZE)
        self.noise_l_buf  = deque(maxlen=WINDOW_SIZE)
        self.noise_r_buf  = deque(maxlen=WINDOW_SIZE)
        self.snr_l_buf    = deque(maxlen=WINDOW_SIZE)
        self.snr_r_buf    = deque(maxlen=WINDOW_SIZE)

        # Contadores acumulados
        self.pkts_total   = 0
        self.rx_errors    = 0
        self.tx_errors    = 0
        self.fixed        = 0
        self.plr_buf      = deque(maxlen=WINDOW_SIZE)

        # Log completo
        self.log_rows = []

        # Lock
        self.lock = threading.Lock()

    def update(self, msg, t_elapsed):
        with self.lock:
            self.last_msg  = msg
            self.connected = True

            rl = msg.rssi
            rr = msg.remrssi
            nl = msg.noise
            nr = msg.remnoise

            self.rssi_l_buf.append(rl)
            self.rssi_r_buf.append(rr)
            self.noise_l_buf.append(nl)
            self.noise_r_buf.append(nr)
            self.snr_l_buf.append(snr_db(rl, nl))
            self.snr_r_buf.append(snr_db(rr, nr))

            # Packet loss rate instantáneo
            rxe = msg.rxerrors
            txb = msg.txbuf    # buffer TX libre [%] — 0 = saturado
            fix = msg.fixed

            if self.pkts_total > 0:
                new_errors = max(0, rxe - self.rx_errors)
                new_pkts   = max(1, msg.rssi)   # proxy: siempre > 0
                plr = min(100.0, (new_errors / max(1, WINDOW_SIZE)) * 100)
            else:
                plr = 0.0

            self.rx_errors  = rxe
            self.fixed      = fix
            self.pkts_total += 1
            self.plr_buf.append(plr)

            # Log
            row = {
                't':          round(t_elapsed, 3),
                'radio':      self.name,
                'rssi_l_raw': rl,  'rssi_r_raw': rr,
                'noise_l_raw':nl,  'noise_r_raw': nr,
                'rssi_l_dbm': rssi_to_dbm(rl),
                'rssi_r_dbm': rssi_to_dbm(rr),
                'noise_l_dbm':noise_to_dbm(nl),
                'noise_r_dbm':noise_to_dbm(nr),
                'snr_l':      snr_db(rl, nl),
                'snr_r':      snr_db(rr, nr),
                'rxerrors':   rxe,
                'fixed':      fix,
                'txbuf':      txb,
                'plr':        round(plr, 2),
                'lat_est_ms': estimate_latency_ms(
                    self.config['air_speed'], self.config['channels']),
            }
            self.log_rows.append(row)

    def snapshot(self):
        """Devuelve métricas actuales thread-safe."""
        with self.lock:
            if not self.rssi_l_buf:
                return None
            msg = self.last_msg

            def avg(buf): return round(sum(buf)/len(buf), 1) if buf else 0

            return {
                'connected':   self.connected,
                'rssi_l':      rssi_to_dbm(msg.rssi),
                'rssi_r':      rssi_to_dbm(msg.remrssi),
                'noise_l':     noise_to_dbm(msg.noise),
                'noise_r':     noise_to_dbm(msg.remnoise),
                'snr_l':       snr_db(msg.rssi, msg.noise),
                'snr_r':       snr_db(msg.remrssi, msg.remnoise),
                'snr_l_avg':   avg(self.snr_l_buf),
                'snr_r_avg':   avg(self.snr_r_buf),
                'rssi_l_avg':  round(avg(self.rssi_l_buf), 1),
                'rssi_l_min':  rssi_to_dbm(min(self.rssi_l_buf)),
                'rssi_r_avg':  round(avg(self.rssi_r_buf), 1),
                'rxerrors':    msg.rxerrors,
                'fixed':       msg.fixed,
                'txbuf':       msg.txbuf,
                'plr':         round(sum(self.plr_buf)/max(1,len(self.plr_buf)),2),
                'lat_est':     estimate_latency_ms(
                    self.config['air_speed'], self.config['channels']),
                'chan_util':   channel_utilization(
                    self.config['air_speed'], CTRL_RATE_HZ),
                'samples':     self.pkts_total,
            }

# ══════════════════════════════════════════════════════════════════════════════
# HILO LECTOR POR RADIO
# ══════════════════════════════════════════════════════════════════════════════
def _reader_thread(name, config, state, stop_event):
    print(f"  [{name}] Conectando a {config['conn']}...")
    try:
        master = mavutil.mavlink_connection(config['conn'])
        master.wait_heartbeat(timeout=10)
        print(f"  [{name}] OK — SYS={master.target_system}")
        state.connected = True
        state.t_start   = time.monotonic()
    except Exception as e:
        print(f"  [{name}] ERROR de conexion: {e}")
        return

    while not stop_event.is_set():
        msg = master.recv_match(type='RADIO_STATUS', blocking=True, timeout=1.0)
        if msg:
            t_el = time.monotonic() - state.t_start
            state.update(msg, t_el)
        time.sleep(POLL_INTERVAL)

    master.close()

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY EN TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
RESET  = '\033[0m'
BOLD   = '\033[1m'
RED    = '\033[91m'
YELLOW = '\033[93m'
GREEN  = '\033[92m'
CYAN   = '\033[96m'
WHITE  = '\033[97m'
DIM    = '\033[2m'

def status_color(value, good, warn, reverse=False):
    """Colorea un valor según umbrales."""
    if reverse:
        if value <= good:   return GREEN
        if value <= warn:   return YELLOW
        return RED
    else:
        if value >= good:   return GREEN
        if value >= warn:   return YELLOW
        return RED

def bar(value, min_v, max_v, width=20, char='█'):
    """Barra de progreso ASCII."""
    frac = max(0, min(1, (value - min_v) / (max_v - min_v)))
    filled = int(frac * width)
    return char * filled + '░' * (width - filled)

def check(value, threshold, reverse=False):
    """Simbolo OK/WARN/FAIL."""
    if reverse:
        ok = value <= threshold
    else:
        ok = value >= threshold
    return f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"

def render_dashboard(states, elapsed):
    """Renderiza el dashboard completo en terminal."""
    lines = []
    W = 70

    lines.append(f"\033[H\033[2J")   # limpiar pantalla
    lines.append(f"{BOLD}{CYAN}{'━'*W}{RESET}")
    lines.append(f"{BOLD}{CYAN}  MONITOR DE ENLACE RF — FORMACION LIDER-SEGUIDOR{RESET}"
                 f"   t={elapsed:.0f}s")
    lines.append(f"{BOLD}{CYAN}{'━'*W}{RESET}")

    # Requisitos del control
    lines.append(f"\n{BOLD}{WHITE}  REQUISITOS DEL CONTROL  "
                 f"[LF @ {CTRL_RATE_HZ} Hz]{RESET}")
    lines.append(f"{DIM}  Latencia max: {CTRL_LATENCY_MAX}ms  |  "
                 f"PLR max: {CTRL_PLR_MAX}%  |  "
                 f"SNR min: {CTRL_SNR_MIN}dB{RESET}\n")

    snaps = {}
    for name, state in states.items():
        snaps[name] = state.snapshot()

    # Una columna por radio
    for name, snap in snaps.items():
        cfg   = RADIOS[name]
        color = cfg['color']

        lines.append(f"{BOLD}{color}  ▌ {name}  "
                     f"Net ID={cfg['net_id']}  "
                     f"AirSpeed={cfg['air_speed']}kbps  "
                     f"CH={cfg['channels']}{RESET}")

        if snap is None or not snap['connected']:
            lines.append(f"  {RED}  Sin conexion{RESET}\n")
            continue

        # RSSI
        c_rssi = status_color(snap['rssi_l'], -80, -95)
        lines.append(
            f"  RSSI Local  : {c_rssi}{snap['rssi_l']:>7.1f} dBm{RESET}  "
            f"{bar(snap['rssi_l'], -120, -40)}  "
            f"avg={snap['rssi_l_avg']:.1f}  min={snap['rssi_l_min']:.1f}")

        c_rrssi = status_color(snap['rssi_r'], -80, -95)
        lines.append(
            f"  RSSI Remote : {c_rrssi}{snap['rssi_r']:>7.1f} dBm{RESET}  "
            f"{bar(snap['rssi_r'], -120, -40)}")

        # Noise
        c_noise = status_color(snap['noise_l'], -95, -85, reverse=True)
        lines.append(
            f"  Noise Local : {c_noise}{snap['noise_l']:>7.1f} dBm{RESET}  "
            f"{bar(-snap['noise_l'], 40, 120)}")
        c_rnoise = status_color(snap['noise_r'], -95, -85, reverse=True)
        lines.append(
            f"  Noise Remote: {c_rnoise}{snap['noise_r']:>7.1f} dBm{RESET}  "
            f"{bar(-snap['noise_r'], 40, 120)}")

        # SNR
        c_snr_l = status_color(snap['snr_l'], CTRL_SNR_MIN+5, CTRL_SNR_MIN)
        c_snr_r = status_color(snap['snr_r'], CTRL_SNR_MIN+5, CTRL_SNR_MIN)
        lines.append(
            f"  SNR Local   : {c_snr_l}{snap['snr_l']:>7.1f} dB{RESET}  "
            f"{bar(snap['snr_l'], 0, 40)}  avg={snap['snr_l_avg']:.1f}")
        lines.append(
            f"  SNR Remote  : {c_snr_r}{snap['snr_r']:>7.1f} dB{RESET}  "
            f"{bar(snap['snr_r'], 0, 40)}  avg={snap['snr_r_avg']:.1f}")

        # Latencia estimada
        lat = snap['lat_est']
        c_lat = status_color(lat, CTRL_LATENCY_MAX, CTRL_LATENCY_MAX*1.5, reverse=True)
        lines.append(
            f"  Lat estimada: {c_lat}{lat:>7.1f} ms{RESET}  "
            f"{check(lat, CTRL_LATENCY_MAX, reverse=True)} "
            f"limite={CTRL_LATENCY_MAX}ms")

        # Utilizacion del canal
        util = snap['chan_util']
        c_util = status_color(util, 80, 60, reverse=True)
        lines.append(
            f"  Canal util  : {c_util}{util:>7.1f} %{RESET}   "
            f"{bar(util, 0, 100)}  libre={100-util:.0f}%")

        # Packet Loss
        plr = snap['plr']
        c_plr = status_color(plr, CTRL_PLR_MAX, CTRL_PLR_MAX*2, reverse=True)
        lines.append(
            f"  PLR         : {c_plr}{plr:>7.2f} %{RESET}  "
            f"{check(plr, CTRL_PLR_MAX, reverse=True)} "
            f"limite={CTRL_PLR_MAX}%")

        # RX Errors / Fixed
        lines.append(
            f"  RX Errors   : {snap['rxerrors']:>4}   "
            f"Fixed: {snap['fixed']:>4}   "
            f"TXbuf libre: {snap['txbuf']:>3}%   "
            f"Muestras: {snap['samples']}")

        # Veredicto
        ok_snr  = snap['snr_l'] >= CTRL_SNR_MIN and snap['snr_r'] >= CTRL_SNR_MIN
        ok_lat  = lat  <= CTRL_LATENCY_MAX
        ok_plr  = plr  <= CTRL_PLR_MAX
        ok_buf  = snap['txbuf'] > 20

        if ok_snr and ok_lat and ok_plr and ok_buf:
            verdict = f"{GREEN}{BOLD}  ✓ ENLACE OPTIMO para control LF{RESET}"
        elif ok_lat and ok_plr:
            verdict = f"{YELLOW}{BOLD}  ⚠ ENLACE MARGINAL — monitorear{RESET}"
        else:
            failures = []
            if not ok_snr:  failures.append(f"SNR bajo ({snap['snr_l']:.0f}dB)")
            if not ok_lat:  failures.append(f"Latencia alta ({lat}ms)")
            if not ok_plr:  failures.append(f"PLR alto ({plr:.1f}%)")
            if not ok_buf:  failures.append(f"Buffer TX lleno ({snap['txbuf']}%)")
            verdict = f"{RED}{BOLD}  ✗ ENLACE DEGRADADO: {', '.join(failures)}{RESET}"

        lines.append(f"\n{verdict}\n")
        lines.append(f"  {'─'*60}")

    # Comparativa entre radios (si ambos conectados)
    if all(s is not None for s in snaps.values()):
        s_list = list(snaps.values())
        n_list = list(snaps.keys())
        if all(s and s['connected'] for s in s_list):
            lines.append(f"\n{BOLD}{WHITE}  COMPARATIVA LIDER vs SEGUIDOR{RESET}")
            diff_rssi  = s_list[0]['rssi_l']  - s_list[1]['rssi_l']
            diff_snr   = s_list[0]['snr_l']   - s_list[1]['snr_l']
            diff_noise = s_list[0]['noise_l'] - s_list[1]['noise_l']

            better_rssi  = n_list[0] if diff_rssi  >= 0 else n_list[1]
            better_snr   = n_list[0] if diff_snr   >= 0 else n_list[1]
            noisier      = n_list[0] if diff_noise >= 0 else n_list[1]

            lines.append(
                f"  RSSI delta  : {abs(diff_rssi):.1f} dB  "
                f"→ mejor enlace: {GREEN}{better_rssi}{RESET}")
            lines.append(
                f"  SNR delta   : {abs(diff_snr):.1f} dB  "
                f"→ mejor SNR  : {GREEN}{better_snr}{RESET}")
            lines.append(
                f"  Noise delta : {abs(diff_noise):.1f} dB  "
                f"→ mas ruidoso: {YELLOW}{noisier}{RESET}")

            # Restriccion al algoritmo
            lat_max = max(s['lat_est'] for s in s_list)
            plr_max = max(s['plr']     for s in s_list)
            snr_min = min(s['snr_l']   for s in s_list)

            lines.append(f"\n{BOLD}{WHITE}  RESTRICCION AL ALGORITMO DE CONTROL{RESET}")
            lines.append(f"  Latencia maxima del sistema : "
                         f"{status_color(lat_max,CTRL_LATENCY_MAX,CTRL_LATENCY_MAX*1.5,True)}"
                         f"{lat_max:.1f} ms{RESET}  "
                         f"(ciclo de control = {1000/CTRL_RATE_HZ:.0f} ms)")
            margin = (1000/CTRL_RATE_HZ) - lat_max
            c_m = GREEN if margin > 15 else (YELLOW if margin > 0 else RED)
            lines.append(f"  Margen disponible           : "
                         f"{c_m}{margin:.1f} ms{RESET} por ciclo")
            lines.append(f"  SNR minimo del sistema      : "
                         f"{status_color(snr_min,CTRL_SNR_MIN+5,CTRL_SNR_MIN)}"
                         f"{snr_min:.1f} dB{RESET}")
            lines.append(f"  PLR maximo del sistema      : "
                         f"{status_color(plr_max,CTRL_PLR_MAX,CTRL_PLR_MAX*2,True)}"
                         f"{plr_max:.2f}%{RESET}")

            # Cuello de botella
            bottleneck = None
            if lat_max > CTRL_LATENCY_MAX:
                bottleneck = f"LATENCIA ({lat_max:.0f}ms > {CTRL_LATENCY_MAX}ms)"
            elif plr_max > CTRL_PLR_MAX:
                bottleneck = f"PACKET LOSS ({plr_max:.1f}% > {CTRL_PLR_MAX}%)"
            elif snr_min < CTRL_SNR_MIN:
                bottleneck = f"SNR BAJO ({snr_min:.1f}dB < {CTRL_SNR_MIN}dB)"

            if bottleneck:
                lines.append(f"\n  {RED}{BOLD}  CUELLO DE BOTELLA: {bottleneck}{RESET}")
            else:
                lines.append(f"\n  {GREEN}{BOLD}  Sin cuellos de botella detectados{RESET}")

    lines.append(f"\n{DIM}  Ctrl+C para detener  |  "
                 f"Actualizando cada {POLL_INTERVAL}s  |  "
                 f"ventana={WINDOW_SIZE} muestras{RESET}")

    print('\n'.join(lines))

# ══════════════════════════════════════════════════════════════════════════════
# GUARDAR CSV
# ══════════════════════════════════════════════════════════════════════════════
def save_csv(states):
    fname = f'radio_monitor_{int(time.time())}.csv'
    cols = ['t','radio','rssi_l_raw','rssi_r_raw','noise_l_raw','noise_r_raw',
            'rssi_l_dbm','rssi_r_dbm','noise_l_dbm','noise_r_dbm',
            'snr_l','snr_r','rxerrors','fixed','txbuf','plr','lat_est_ms']
    with open(fname, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for state in states.values():
            with state.lock:
                w.writerows(state.log_rows)
    print(f"\nCSV guardado: {fname}")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# REPORTE FINAL
# ══════════════════════════════════════════════════════════════════════════════
def print_final_report(states):
    import statistics

    print(f"\n{'═'*65}")
    print(f"  REPORTE FINAL DE CALIDAD DE ENLACE")
    print(f"{'═'*65}\n")

    for name, state in states.items():
        with state.lock:
            rows = state.log_rows

        if not rows:
            print(f"  [{name}] Sin datos\n")
            continue

        snr_l_vals  = [r['snr_l']      for r in rows]
        snr_r_vals  = [r['snr_r']      for r in rows]
        rssi_l_vals = [r['rssi_l_dbm'] for r in rows]
        lat_vals    = [r['lat_est_ms'] for r in rows]
        plr_vals    = [r['plr']        for r in rows]

        print(f"  [{name}]  Net ID={RADIOS[name]['net_id']}  "
              f"AirSpeed={RADIOS[name]['air_speed']}kbps  "
              f"Muestras={len(rows)}")
        print(f"  {'─'*55}")
        print(f"  RSSI Local  : avg={statistics.mean(rssi_l_vals):.1f}  "
              f"min={min(rssi_l_vals):.1f}  "
              f"max={max(rssi_l_vals):.1f}  dBm")
        print(f"  SNR Local   : avg={statistics.mean(snr_l_vals):.1f}  "
              f"min={min(snr_l_vals):.1f}  "
              f"std={statistics.stdev(snr_l_vals) if len(snr_l_vals)>1 else 0:.1f}  dB")
        print(f"  SNR Remote  : avg={statistics.mean(snr_r_vals):.1f}  "
              f"min={min(snr_r_vals):.1f}  dB")
        print(f"  Latencia est: avg={statistics.mean(lat_vals):.1f}  ms")
        print(f"  PLR         : avg={statistics.mean(plr_vals):.2f}  "
              f"max={max(plr_vals):.2f}  %")

        # Veredicto contra requisitos del control
        ok = (statistics.mean(snr_l_vals) >= CTRL_SNR_MIN and
              statistics.mean(lat_vals)   <= CTRL_LATENCY_MAX and
              statistics.mean(plr_vals)   <= CTRL_PLR_MAX)

        print(f"  Veredicto   : {'APTO' if ok else 'NO APTO'} para control LF @ {CTRL_RATE_HZ}Hz\n")

    # Generar grafica comparativa
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle('Calidad de enlace RF — Comparativa Lider vs Seguidor', fontsize=12)

        colors = {'LIDER': 'tab:green', 'SEGUIDOR': 'tab:blue'}

        for name, state in states.items():
            with state.lock:
                rows = state.log_rows
            if not rows:
                continue

            t       = [r['t']        for r in rows]
            snr_l   = [r['snr_l']    for r in rows]
            snr_r   = [r['snr_r']    for r in rows]
            rssi_l  = [r['rssi_l_dbm'] for r in rows]
            noise_l = [r['noise_l_dbm'] for r in rows]
            plr     = [r['plr']      for r in rows]
            rxerr   = [r['rxerrors'] for r in rows]
            c = colors[name]

            # SNR local y remoto
            axes[0,0].plot(t, snr_l, '-',  color=c, lw=1.5, label=f'{name} SNR local')
            axes[0,0].plot(t, snr_r, '--', color=c, lw=1.0, alpha=0.7, label=f'{name} SNR remoto')

            # RSSI y Noise
            axes[0,1].plot(t, rssi_l,  '-',  color=c, lw=1.5, label=f'{name} RSSI')
            axes[0,1].plot(t, noise_l, '--', color=c, lw=1.0, alpha=0.6, label=f'{name} Noise')

            # PLR
            axes[1,0].plot(t, plr, '-', color=c, lw=1.5, label=name)

            # RX Errors acumulados
            axes[1,1].plot(t, rxerr, '-', color=c, lw=1.5, label=name)

        # Líneas de referencia
        axes[0,0].axhline(CTRL_SNR_MIN, color='red', ls='--', lw=1, label=f'Min {CTRL_SNR_MIN}dB')
        axes[1,0].axhline(CTRL_PLR_MAX, color='red', ls='--', lw=1, label=f'Max {CTRL_PLR_MAX}%')

        axes[0,0].set(title='SNR [dB]',           ylabel='dB',  xlabel='Tiempo [s]')
        axes[0,1].set(title='RSSI y Noise [dBm]', ylabel='dBm', xlabel='Tiempo [s]')
        axes[1,0].set(title='Packet Loss Rate',   ylabel='%',   xlabel='Tiempo [s]')
        axes[1,1].set(title='RX Errors acumulados', ylabel='count', xlabel='Tiempo [s]')

        for ax in axes.flat:
            ax.legend(fontsize=7); ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("  (matplotlib no disponible — instala con: pip install matplotlib)")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor de enlace RF dual')
    parser.add_argument('--log', type=float, default=0,
                        help='Duración en segundos (0 = indefinido)')
    parser.add_argument('--compare', action='store_true',
                        help='Mostrar reporte comparativo al final')
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print(f"  RADIO MONITOR — Control LF Lider-Seguidor")
    print(f"{'═'*65}")
    print(f"  Requisitos del control @ {CTRL_RATE_HZ} Hz:")
    print(f"    Latencia max : {CTRL_LATENCY_MAX} ms")
    print(f"    PLR max      : {CTRL_PLR_MAX} %")
    print(f"    SNR min      : {CTRL_SNR_MIN} dB")
    print(f"\n  Conectando a los radios...")

    # Crear estados y lanzar hilos lectores
    states = {name: RadioState(name, cfg) for name, cfg in RADIOS.items()}
    stop   = threading.Event()

    threads = []
    for name, cfg in RADIOS.items():
        t = threading.Thread(
            target=_reader_thread,
            args=(name, cfg, states[name], stop),
            daemon=True, name=f'reader-{name}')
        t.start()
        threads.append(t)

    time.sleep(2.0)   # dar tiempo a conectar

    t_start = time.monotonic()
    print("\033[2J")  # limpiar pantalla

    try:
        while True:
            elapsed = time.monotonic() - t_start
            render_dashboard(states, elapsed)

            if args.log > 0 and elapsed >= args.log:
                print(f"\n  Tiempo de monitoreo completado ({args.log}s)")
                break

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n  Detenido por usuario.")

    finally:
        stop.set()

    # Guardar CSV si hay datos
    has_data = any(state.log_rows for state in states.values())
    if has_data:
        save_csv(states)
        if args.compare or args.log > 0:
            print_final_report(states)
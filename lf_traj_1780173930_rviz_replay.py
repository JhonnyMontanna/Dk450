#!/usr/bin/env python3
"""
plot_traj_3d.py — Visualizador 3D interactivo de trayectoria de drones
=======================================================================
Lee un archivo lf_traj_*.csv generado por LF_Circulo_RT_v2.py y genera:

  Fig 1 — Vista 3D interactiva (rotar con el ratón) con líder y seguidor
           coloreados por fase, eventos marcados, y vector de formación L→S.
  Fig 2 — Tres vistas 2D: XY (top-down), XZ (perfil Norte), YZ (perfil Este)
  Fig 3 — Altitud vs tiempo de ambos drones con fases y eventos
  Fig 4 — Separación L-S (horizontal y vertical) vs tiempo

Uso:
  python3 plot_traj_3d.py                        # archivo más reciente
  python3 plot_traj_3d.py lf_traj_TIMESTAMP.csv  # archivo específico
"""

import sys
import os
import glob
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

# ══════════════════════════════════════════════════════════════════════════════
# COLORES POR FASE
# ══════════════════════════════════════════════════════════════════════════════
PHASE_COLORS = {
    'init':      '#888888',
    'takeoff_L': '#22BB22',
    'takeoff_S': '#22BBAA',
    'prepos':    '#CCAA00',
    'circle':    '#EE5500',
    'return':    '#4444EE',
    'land_L':    '#AA22AA',
    'land_S':    '#CC66CC',
}
DEFAULT_COLOR = '#333333'

LW       = 2.0
LW_S     = 1.4
LW_FORM  = 0.6
ALPHA_FORM = 0.35
N_FORM   = 30       # número de vectores de formación L→S a dibujar en 3D


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y PREPARACIÓN
# ══════════════════════════════════════════════════════════════════════════════
def _latest():
    files = sorted(glob.glob('lf_traj_*.csv'))
    return files[-1] if files else None


def load(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows


def to_arrays(rows):
    """Convierte filas a arrays numpy."""
    t    = np.array([float(r['t'])    for r in rows])
    lx   = np.array([float(r['lx'])  for r in rows])
    ly   = np.array([float(r['ly'])  for r in rows])
    lz   = np.array([float(r['lz'])  for r in rows])
    sx   = np.array([float(r['sx'])  for r in rows])
    sy   = np.array([float(r['sy'])  for r in rows])
    sz   = np.array([float(r['sz'])  for r in rows])
    lyaw = np.array([float(r['l_yaw']) for r in rows])
    syaw = np.array([float(r['s_yaw']) for r in rows])
    phases = [r['phase'] for r in rows]
    events = [(float(r['t']), r['event'])
              for r in rows if r.get('event', '').strip()]
    return t, lx, ly, lz, sx, sy, sz, lyaw, syaw, phases, events


def _phase_segments(t, x, y, z, phases):
    """
    Devuelve lista de (color, xs, ys, zs) por segmento de fase contigua.
    Cada segmento incluye el primer punto del siguiente para que no haya huecos.
    """
    segments = []
    if len(phases) == 0:
        return segments
    cur_phase = phases[0]
    i_start   = 0
    for i in range(1, len(phases)):
        if phases[i] != cur_phase:
            color = PHASE_COLORS.get(cur_phase, DEFAULT_COLOR)
            # +1 para conectar con el inicio del siguiente segmento
            end = min(i + 1, len(t))
            segments.append((color, x[i_start:end],
                                    y[i_start:end],
                                    z[i_start:end]))
            i_start   = i
            cur_phase = phases[i]
    color = PHASE_COLORS.get(cur_phase, DEFAULT_COLOR)
    segments.append((color, x[i_start:], y[i_start:], z[i_start:]))
    return segments


def _phase_spans(ax, t, phases):
    """Colorea el fondo de un eje 2D por fase."""
    if not phases:
        return
    cur = phases[0]
    t0  = t[0]
    for i in range(1, len(phases)):
        if phases[i] != cur:
            ax.axvspan(t0, t[i], color=PHASE_COLORS.get(cur, DEFAULT_COLOR),
                       alpha=0.10)
            t0  = t[i]
            cur = phases[i]
    ax.axvspan(t0, t[-1], color=PHASE_COLORS.get(cur, DEFAULT_COLOR), alpha=0.10)


def _mark_events(ax, events, y_frac=0.93, fontsize=7, vertical=True):
    """Marca eventos con líneas verticales y etiquetas."""
    ymin, ymax = ax.get_ylim()
    y_txt = ymin + (ymax - ymin) * y_frac
    for t_ev, ev in events:
        ax.axvline(t_ev, color='k', lw=0.9, ls='--', alpha=0.55)
        if vertical:
            ax.text(t_ev + 0.4, y_txt, ev, fontsize=fontsize,
                    rotation=90, va='top', ha='left', color='k', alpha=0.7)
        else:
            ax.text(t_ev + 0.4, y_txt, ev, fontsize=fontsize,
                    va='top', ha='left', color='k', alpha=0.7)


def _phase_legend_handles():
    return [mpatches.Patch(color=c, alpha=0.7, label=ph)
            for ph, c in PHASE_COLORS.items()]


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — VISTA 3D INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
def plot_3d(t, lx, ly, lz, sx, sy, sz, lyaw, syaw, phases, events):
    fig = plt.figure(figsize=(13, 9), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.suptitle('Trayectoria 3D — Líder y Seguidor', fontsize=13, fontweight='bold')

    # ── Trayectorias coloreadas por fase ──────────────────────────────────────
    for color, xs, ys, zs in _phase_segments(t, lx, ly, lz, phases):
        ax.plot(ys, xs, zs, '-', color=color, lw=LW, alpha=0.9)   # ENU: Y=Este, X=Norte

    for color, xs, ys, zs in _phase_segments(t, sx, sy, sz, phases):
        ax.plot(ys, xs, zs, '--', color=color, lw=LW_S, alpha=0.75)

    # ── Vectores de formación L→S (submuestra uniforme) ───────────────────────
    n   = len(t)
    idx = np.round(np.linspace(0, n - 1, N_FORM)).astype(int)
    for i in idx:
        ax.plot([ly[i], sy[i]], [lx[i], sx[i]], [lz[i], sz[i]],
                '-', color='gray', lw=LW_FORM, alpha=ALPHA_FORM)

    # ── Marcadores de inicio y fin ────────────────────────────────────────────
    ax.scatter([ly[0]],  [lx[0]],  [lz[0]],  c='green',  s=60, zorder=5,
               depthshade=False, label='Inicio L')
    ax.scatter([ly[-1]], [lx[-1]], [lz[-1]], c='green',  s=60, marker='s',
               zorder=5, depthshade=False, label='Fin L')
    ax.scatter([sy[0]],  [sx[0]],  [sz[0]],  c='blue',   s=60, zorder=5,
               depthshade=False, label='Inicio S')
    ax.scatter([sy[-1]], [sx[-1]], [sz[-1]], c='blue',   s=60, marker='s',
               zorder=5, depthshade=False, label='Fin S')

    # ── Eventos en 3D (punto + texto sobre el líder) ──────────────────────────
    for t_ev, ev in events:
        # Encontrar el índice más cercano en el tiempo
        idx_ev = int(np.argmin(np.abs(t - t_ev)))
        ax.scatter([ly[idx_ev]], [lx[idx_ev]], [lz[idx_ev] + 0.15],
                   c='k', s=25, zorder=6, depthshade=False)
        ax.text(ly[idx_ev], lx[idx_ev], lz[idx_ev] + 0.25,
                ev, fontsize=6, color='k', alpha=0.75)

    # ── Plano del suelo (Z=0) como referencia ────────────────────────────────
    all_y = np.concatenate([ly, sy])
    all_x = np.concatenate([lx, sx])
    pad   = 0.5
    yg = np.array([all_y.min() - pad, all_y.max() + pad])
    xg = np.array([all_x.min() - pad, all_x.max() + pad])
    Yg, Xg = np.meshgrid(yg, xg)
    ax.plot_surface(Yg, Xg, np.zeros_like(Yg),
                    alpha=0.06, color='gray', zorder=0)

    # ── Ejes y leyenda ────────────────────────────────────────────────────────
    ax.set_xlabel('Y — Este [m]',   fontsize=9, labelpad=8)
    ax.set_ylabel('X — Norte [m]',  fontsize=9, labelpad=8)
    ax.set_zlabel('Z — Altitud [m]', fontsize=9, labelpad=8)

    # Leyenda de fases
    ph_handles = _phase_legend_handles()
    extra = [Line2D([0],[0], color='k', lw=LW,  label='Líder'),
             Line2D([0],[0], color='k', lw=LW_S, ls='--', label='Seguidor'),
             Line2D([0],[0], color='gray', lw=1,  label='Vector L→S')]
    ax.legend(handles=ph_handles + extra, fontsize=7, loc='upper left',
              ncol=2, framealpha=0.85)

    # Orientación inicial cómoda para ver el círculo
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — TRES PROYECCIONES 2D
# ══════════════════════════════════════════════════════════════════════════════
def plot_projections(t, lx, ly, lz, sx, sy, sz, phases, events):
    fig, axs = plt.subplots(1, 3, figsize=(16, 6), facecolor='white')
    fig.suptitle('Proyecciones 2D de la trayectoria', fontsize=13, fontweight='bold')

    configs = [
        # (ax, título, xlabel,       ylabel,       ldata_a, ldata_b, sdata_a, sdata_b)
        (axs[0], 'Vista superior XY\n(top-down)',
         'Y — Este [m]', 'X — Norte [m]',   ly, lx, sy, sx),
        (axs[1], 'Perfil Norte XZ',
         'X — Norte [m]', 'Z — Altitud [m]', lx, lz, sx, sz),
        (axs[2], 'Perfil Este YZ',
         'Y — Este [m]', 'Z — Altitud [m]', ly, lz, sy, sz),
    ]

    for ax, title, xl, yl, la, lb, sa, sb in configs:
        # Segmentos coloreados por fase
        for color, xs, ys, _ in _phase_segments(t, la, lb, lz, phases):
            ax.plot(xs, ys, '-', color=color, lw=LW, alpha=0.9)
        for color, xs, ys, _ in _phase_segments(t, sa, sb, sz, phases):
            ax.plot(xs, ys, '--', color=color, lw=LW_S, alpha=0.75)

        # Vectores de formación L→S
        n   = len(t)
        idx = np.round(np.linspace(0, n - 1, N_FORM)).astype(int)
        for i in idx:
            ax.annotate('', xy=(sa[i], sb[i]), xytext=(la[i], lb[i]),
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=0.7, alpha=0.4))

        # Inicio / fin
        ax.plot(la[0],  lb[0],  'g^', ms=9, zorder=5, label='Inicio L')
        ax.plot(la[-1], lb[-1], 'gs', ms=9, zorder=5, label='Fin L')
        ax.plot(sa[0],  sb[0],  'b^', ms=8, zorder=5, label='Inicio S')
        ax.plot(sa[-1], sb[-1], 'bs', ms=8, zorder=5, label='Fin S')

        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    # Leyenda de fases solo en el primer panel
    ph_handles = _phase_legend_handles()
    axs[0].legend(handles=ph_handles, fontsize=7, loc='upper right',
                  ncol=2, framealpha=0.9)

    fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — ALTITUD VS TIEMPO
# ══════════════════════════════════════════════════════════════════════════════
def plot_altitude(t, lz, sz, phases, events):
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='white')
    fig.suptitle('Altitud vs Tiempo — todas las fases', fontsize=13, fontweight='bold')

    _phase_spans(ax, t, phases)
    ax.plot(t, lz, '-',  color='#EE5500', lw=LW,  label='Líder Z')
    ax.plot(t, sz, '--', color='#2255CC', lw=LW_S, label='Seguidor Z')

    ax.set_ylabel('Altitud [m]', fontsize=10)
    ax.set_xlabel('Tiempo [s]',  fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    _mark_events(ax, events)
    # Actualizar límites antes de marcar
    ax.autoscale_view()
    _mark_events(ax, events)   # segunda vez ya con ylim correcto

    # Leyenda de fases
    ph_handles = _phase_legend_handles()
    ax.legend(handles=ph_handles +
              [Line2D([0],[0], color='#EE5500', lw=LW,       label='Líder Z'),
               Line2D([0],[0], color='#2255CC', lw=LW_S, ls='--', label='Seguidor Z')],
              fontsize=7, loc='upper left', ncol=3, framealpha=0.9)

    fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — SEPARACIÓN L-S VS TIEMPO
# ══════════════════════════════════════════════════════════════════════════════
def plot_separation(t, lx, ly, lz, sx, sy, sz, phases, events):
    dist_xy = np.hypot(lx - sx, ly - sy)
    dist_z  = np.abs(lz - sz)
    dist_3d = np.sqrt((lx-sx)**2 + (ly-sy)**2 + (lz-sz)**2)

    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True, facecolor='white')
    fig.suptitle('Separación Líder–Seguidor vs Tiempo', fontsize=13, fontweight='bold')

    for ax in axs:
        _phase_spans(ax, t, phases)

    axs[0].plot(t, dist_xy, '-',  color='#2255CC', lw=LW,  label='Dist XY [m]')
    axs[0].plot(t, dist_3d, '--', color='#555555', lw=LW_S, label='Dist 3D [m]', alpha=0.7)
    axs[0].set_ylabel('Distancia [m]', fontsize=10)
    axs[0].legend(fontsize=9); axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, dist_z, '-', color='#AA3300', lw=LW, label='Dist Z [m]')
    axs[1].set_ylabel('Separación vertical [m]', fontsize=10)
    axs[1].set_xlabel('Tiempo [s]', fontsize=10)
    axs[1].legend(fontsize=9); axs[1].grid(True, alpha=0.3)

    # Estadísticas por fase en consola
    unique_phases = []
    seen = set()
    for ph in phases:
        if ph not in seen:
            seen.add(ph); unique_phases.append(ph)

    print("\n  Separación L-S por fase:")
    print(f"  {'Fase':<14} {'XY media':>10} {'XY max':>9} {'Z media':>9} {'Z max':>8}")
    print("  " + "─"*52)
    for ph in unique_phases:
        mask = np.array([p == ph for p in phases])
        if mask.sum() == 0:
            continue
        print(f"  {ph:<14} "
              f"{dist_xy[mask].mean():>9.3f}m "
              f"{dist_xy[mask].max():>8.3f}m "
              f"{dist_z[mask].mean():>8.3f}m "
              f"{dist_z[mask].max():>7.3f}m")

    _mark_events(axs[0], events)
    axs[0].autoscale_view(); _mark_events(axs[0], events)

    # Leyenda de fases
    axs[0].legend(handles=_phase_legend_handles() +
                  [Line2D([0],[0], color='#2255CC', lw=LW, label='Dist XY'),
                   Line2D([0],[0], color='#555555', lw=LW_S, ls='--', label='Dist 3D')],
                  fontsize=7, ncol=3, framealpha=0.9)

    fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Resolver ruta
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = _latest()

    if not path or not os.path.exists(path):
        print(f"❌ No se encontró lf_traj_*.csv  (buscado: {path})")
        print("   Uso: python3 plot_traj_3d.py lf_traj_TIMESTAMP.csv")
        sys.exit(1)

    print(f"📍 Cargando: {path}")
    rows = load(path)
    print(f"   {len(rows)} muestras")

    t, lx, ly, lz, sx, sy, sz, lyaw, syaw, phases, events = to_arrays(rows)

    # Resumen en consola
    dur = t[-1] - t[0]
    unique_phases = list(dict.fromkeys(phases))   # orden de aparición
    print(f"   Duración  : {dur:.1f} s")
    print(f"   Fases     : {' → '.join(unique_phases)}")
    if events:
        print(f"   Eventos   : {len(events)}")
        for t_ev, ev in events:
            print(f"     t={t_ev:7.2f}s  {ev}")
    print(f"   Altitud L : {lz.min():.2f} – {lz.max():.2f} m")
    print(f"   Altitud S : {sz.min():.2f} – {sz.max():.2f} m")
    print(f"   Sep XY    : {np.hypot(lx-sx,ly-sy).mean():.3f} m (media)")

    # Generar figuras
    plot_3d(t, lx, ly, lz, sx, sy, sz, lyaw, syaw, phases, events)
    plot_projections(t, lx, ly, lz, sx, sy, sz, phases, events)
    plot_altitude(t, lz, sz, phases, events)
    plot_separation(t, lx, ly, lz, sx, sy, sz, phases, events)

    plt.show()
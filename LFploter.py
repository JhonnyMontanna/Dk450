"""
Visualizador de trayectorias Líder-Seguidor
=============================================
Uso:
    python plot_lf_trajectories.py                      # busca todos los lf_*.csv en el directorio actual
    python plot_lf_trajectories.py archivo1.csv archivo2.csv ...
    python plot_lf_trajectories.py --dir /ruta/a/csvs   # busca en otro directorio

Controles interactivos:
    → / ←   : siguiente / anterior archivo
    g       : ir a archivo por número
    s       : guardar figura actual como PNG
    q       : cerrar
"""

import sys
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# ── Colores ──────────────────────────────────────────────────────────────────
C_LEADER   = "#378ADD"
C_FOLLOWER = "#D85A30"
C_ERROR    = "#639922"
C_BG       = "#F8F8F6"
C_GRID     = "#E0DED6"

# ── Estado global del navegador ───────────────────────────────────────────────
state = {"idx": 0, "files": []}


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def plot_file(df: pd.DataFrame, filename: str, fig: plt.Figure, axes):
    for ax in axes:
        ax.cla()

    name = os.path.basename(filename)
    n    = len(df)

    # CSV vacío o sin filas de datos
    if n == 0:
        idx   = state["idx"] + 1
        total = len(state["files"])
        fig.suptitle(f"[{idx}/{total}]  {name}  —  archivo vacío", fontsize=12, y=0.98)
        for ax in axes:
            ax.text(0.5, 0.5, "Archivo sin datos", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=12)
        fig.canvas.draw_idle()
        return

    dur  = round(df["t"].iloc[-1] - df["t"].iloc[0], 2) if "t" in df.columns else "?"
    phase = df["leader_phase"].iloc[0] if "leader_phase" in df.columns else "?"

    # ── Panel 1: Trayectoria XY ───────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(df["lx"], df["ly"], color=C_LEADER,   lw=2,   label="Líder",    zorder=3)
    ax1.plot(df["sx"], df["sy"], color=C_FOLLOWER, lw=2, ls="--", label="Seguidor", zorder=3)

    # Marcadores de inicio y fin
    ax1.scatter(df["lx"].iloc[0],  df["ly"].iloc[0],  color=C_LEADER,   s=60, zorder=5, marker="o")
    ax1.scatter(df["sx"].iloc[0],  df["sy"].iloc[0],  color=C_FOLLOWER, s=60, zorder=5, marker="o")
    ax1.scatter(df["lx"].iloc[-1], df["ly"].iloc[-1], color=C_LEADER,   s=80, zorder=5, marker="D")
    ax1.scatter(df["sx"].iloc[-1], df["sy"].iloc[-1], color=C_FOLLOWER, s=80, zorder=5, marker="D")

    ax1.set_xlabel("X (m)", fontsize=11)
    ax1.set_ylabel("Y (m)", fontsize=11)
    ax1.set_title("Trayectoria XY", fontsize=12, fontweight="normal")
    ax1.set_aspect("equal", "datalim")
    ax1.legend(fontsize=10, framealpha=0.6)
    ax1.set_facecolor(C_BG)
    ax1.grid(True, color=C_GRID, linewidth=0.8)

    # ── Panel 2: Error XY vs tiempo ──────────────────────────────────────────
    ax2 = axes[1]
    if "err_xy" in df.columns and "t" in df.columns:
        ax2.plot(df["t"], df["err_xy"], color=C_ERROR, lw=1.5)
        ax2.axhline(df["err_xy"].mean(), color=C_ERROR, lw=1, ls=":", alpha=0.7,
                    label=f"Media: {df['err_xy'].mean():.3f} m")
        ax2.set_ylabel("Error XY (m)", fontsize=11)
        ax2.legend(fontsize=9, framealpha=0.6)
    else:
        ax2.text(0.5, 0.5, "Columna err_xy no disponible",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")

    ax2.set_xlabel("Tiempo (s)", fontsize=11)
    ax2.set_title("Error XY vs Tiempo", fontsize=12, fontweight="normal")
    ax2.set_facecolor(C_BG)
    ax2.grid(True, color=C_GRID, linewidth=0.8)

    # ── Panel 3: Error Z vs tiempo ───────────────────────────────────────────
    ax3 = axes[2]
    if "err_z" in df.columns and "t" in df.columns:
        ax3.plot(df["t"], df["err_z"], color="#9F77DD", lw=1.5)
        ax3.axhline(df["err_z"].mean(), color="#9F77DD", lw=1, ls=":", alpha=0.7,
                    label=f"Media: {df['err_z'].mean():.3f} m")
        ax3.set_ylabel("Error Z (m)", fontsize=11)
        ax3.legend(fontsize=9, framealpha=0.6)
    else:
        ax3.text(0.5, 0.5, "Columna err_z no disponible",
                 ha="center", va="center", transform=ax3.transAxes, color="gray")

    ax3.set_xlabel("Tiempo (s)", fontsize=11)
    ax3.set_title("Error Z vs Tiempo", fontsize=12, fontweight="normal")
    ax3.set_facecolor(C_BG)
    ax3.grid(True, color=C_GRID, linewidth=0.8)

    # ── Título general ────────────────────────────────────────────────────────
    idx  = state["idx"] + 1
    total = len(state["files"])
    fig.suptitle(
        f"[{idx}/{total}]  {name}   |   fase: {phase}   |   "
        f"n={n}   duración={dur} s",
        fontsize=12, fontweight="normal", y=0.98
    )

    fig.canvas.draw_idle()


def refresh(fig, axes):
    path = state["files"][state["idx"]]
    try:
        df = load_csv(path)
        plot_file(df, path, fig, axes)
    except Exception as e:
        for ax in axes:
            ax.cla()
            ax.text(0.5, 0.5, f"Error al leer:\n{e}",
                    ha="center", va="center", transform=ax.transAxes, color="red")
        fig.canvas.draw_idle()


def on_key(event, fig, axes):
    idx = state["idx"]
    n   = len(state["files"])

    if event.key == "right" and idx < n - 1:
        state["idx"] += 1
        refresh(fig, axes)

    elif event.key == "left" and idx > 0:
        state["idx"] -= 1
        refresh(fig, axes)

    elif event.key == "g":
        try:
            num = int(input(f"  Ir a archivo (1–{n}): ")) - 1
            if 0 <= num < n:
                state["idx"] = num
                refresh(fig, axes)
        except ValueError:
            pass

    elif event.key == "s":
        out = os.path.splitext(os.path.basename(state["files"][state["idx"]]))[0] + ".png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Guardado: {out}")

    elif event.key == "q":
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Visualizador líder-seguidor")
    parser.add_argument("csvs", nargs="*", help="Archivos CSV a graficar")
    parser.add_argument("--dir", default=".", help="Directorio donde buscar lf_*.csv")
    args = parser.parse_args()

    if args.csvs:
        files = [f for f in args.csvs if os.path.isfile(f)]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "lf_*.csv")))

    if not files:
        print("No se encontraron archivos CSV.")
        print("Uso: python plot_lf_trajectories.py [archivo1.csv ...] [--dir <directorio>]")
        sys.exit(1)

    print(f"  {len(files)} archivo(s) encontrado(s).")
    print("  Controles: ← → navegar   g ir a número   s guardar PNG   q salir\n")

    state["files"] = files
    state["idx"]   = 0

    fig = plt.figure(figsize=(15, 5), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.12, wspace=0.28)
    gs   = gridspec.GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    refresh(fig, axes)
    fig.canvas.mpl_connect("key_press_event", lambda e: on_key(e, fig, axes))

    plt.show()


if __name__ == "__main__":
    main()
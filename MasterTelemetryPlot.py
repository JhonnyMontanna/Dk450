#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

CSV_FILE = "master_telemetry.csv"

LW = 2.0

# ==========================================================
# CARGA
# ==========================================================

df = pd.read_csv(CSV_FILE)

print(f"\nArchivo cargado: {CSV_FILE}")
print(f"Filas: {len(df)}")

# ==========================================================
# COLUMNAS AUX
# ==========================================================

t = df["t"].to_numpy()

# ==========================================================
# DISTANCIAS
# ==========================================================

if all(c in df.columns for c in
       ["L_x","L_y","S_x","S_y"]):

    df["dist_xy"] = np.sqrt(
        (df["L_x"] - df["S_x"])**2 +
        (df["L_y"] - df["S_y"])**2
    )

if all(c in df.columns for c in
       ["L_x","L_y","L_z",
        "S_x","S_y","S_z"]):

    df["dist_3d"] = np.sqrt(
        (df["L_x"] - df["S_x"])**2 +
        (df["L_y"] - df["S_y"])**2 +
        (df["L_z"] - df["S_z"])**2
    )

# ==========================================================
# ESTADISTICAS
# ==========================================================

print("\n==============================")
print("ESTADISTICAS")
print("==============================")

if "dist_xy" in df.columns:

    print(
        f"Distancia XY media : "
        f"{df['dist_xy'].mean():.3f} m"
    )

    print(
        f"Distancia XY max   : "
        f"{df['dist_xy'].max():.3f} m"
    )

if "dist_3d" in df.columns:

    print(
        f"Distancia 3D media : "
        f"{df['dist_3d'].mean():.3f} m"
    )

    print(
        f"Distancia 3D max   : "
        f"{df['dist_3d'].max():.3f} m"
    )

for drone in ["L","S"]:

    if f"{drone}_lvx" in df.columns:

        v = np.sqrt(
            df[f"{drone}_lvx"]**2 +
            df[f"{drone}_lvy"]**2 +
            df[f"{drone}_lvz"]**2
        )

        print(
            f"Velocidad max {drone}: "
            f"{v.max():.3f} m/s"
        )


# ==========================================================
# FIGURA 0
# TRAYECTORIA 3D
# ==========================================================

from mpl_toolkits.mplot3d import Axes3D

fig0 = plt.figure(figsize=(10,8))
ax = fig0.add_subplot(111, projection='3d')

# ----------------------------------------------------------
# Lider
# ----------------------------------------------------------

if all(c in df.columns for c in ["L_x","L_y","L_z"]):

    ax.plot(
        df["L_y"],
        df["L_x"],
        df["L_z"],
        lw=2,
        label="Lider"
    )

    ax.scatter(
        df["L_y"].iloc[0],
        df["L_x"].iloc[0],
        df["L_z"].iloc[0],
        s=60,
        marker="o",
        label="Inicio Lider"
    )

    ax.scatter(
        df["L_y"].iloc[-1],
        df["L_x"].iloc[-1],
        df["L_z"].iloc[-1],
        s=60,
        marker="s",
        label="Fin Lider"
    )

# ----------------------------------------------------------
# Seguidor
# ----------------------------------------------------------

if all(c in df.columns for c in ["S_x","S_y","S_z"]):

    ax.plot(
        df["S_y"],
        df["S_x"],
        df["S_z"],
        "--",
        lw=2,
        label="Seguidor"
    )

    ax.scatter(
        df["S_y"].iloc[0],
        df["S_x"].iloc[0],
        df["S_z"].iloc[0],
        s=60,
        marker="o",
        label="Inicio Seguidor"
    )

    ax.scatter(
        df["S_y"].iloc[-1],
        df["S_x"].iloc[-1],
        df["S_z"].iloc[-1],
        s=60,
        marker="s",
        label="Fin Seguidor"
    )

# ----------------------------------------------------------
# Vectores de formacion
# ----------------------------------------------------------

if all(c in df.columns for c in
       ["L_x","L_y","L_z",
        "S_x","S_y","S_z"]):

    N_FORM = 50

    idx = np.linspace(
        0,
        len(df)-1,
        N_FORM
    ).astype(int)

    for i in idx:

        ax.plot(
            [df["L_y"].iloc[i], df["S_y"].iloc[i]],
            [df["L_x"].iloc[i], df["S_x"].iloc[i]],
            [df["L_z"].iloc[i], df["S_z"].iloc[i]],
            alpha=0.3,
            linewidth=0.7
        )

# ----------------------------------------------------------
# Plano suelo
# ----------------------------------------------------------

all_x = np.concatenate([
    df["L_x"].dropna(),
    df["S_x"].dropna()
])

all_y = np.concatenate([
    df["L_y"].dropna(),
    df["S_y"].dropna()
])

xmin = all_x.min()
xmax = all_x.max()

ymin = all_y.min()
ymax = all_y.max()

Xg, Yg = np.meshgrid(
    np.linspace(xmin, xmax, 2),
    np.linspace(ymin, ymax, 2)
)

Zg = np.zeros_like(Xg)

ax.plot_surface(
    Yg,
    Xg,
    Zg,
    alpha=0.08
)

# ----------------------------------------------------------
# FORMATO
# ----------------------------------------------------------

ax.set_title(
    "Trayectoria 3D Lider-Seguidor"
)

ax.set_xlabel(
    "Y Este [m]"
)

ax.set_ylabel(
    "X Norte [m]"
)

ax.set_zlabel(
    "Altitud [m]"
)

ax.legend()

ax.view_init(
    elev=25,
    azim=-60
)

# ==========================================================
# FIGURA 1
# TRAYECTORIA XY
# ==========================================================

fig1, ax = plt.subplots(figsize=(8,8))

if all(c in df.columns for c in
       ["L_x","L_y"]):

    ax.plot(
        df["L_y"],
        df["L_x"],
        lw=LW,
        label="Lider"
    )

    ax.plot(
        df["L_y"].iloc[0],
        df["L_x"].iloc[0],
        "go",
        ms=10,
        label="Inicio Lider"
    )

    ax.plot(
        df["L_y"].iloc[-1],
        df["L_x"].iloc[-1],
        "gs",
        ms=10,
        label="Fin Lider"
    )

if all(c in df.columns for c in
       ["S_x","S_y"]):

    ax.plot(
        df["S_y"],
        df["S_x"],
        "--",
        lw=LW,
        label="Seguidor"
    )

    ax.plot(
        df["S_y"].iloc[0],
        df["S_x"].iloc[0],
        "bo",
        ms=10,
        label="Inicio Seguidor"
    )

    ax.plot(
        df["S_y"].iloc[-1],
        df["S_x"].iloc[-1],
        "bs",
        ms=10,
        label="Fin Seguidor"
    )

ax.set_title("Trayectoria XY")
ax.set_xlabel("Y Este [m]")
ax.set_ylabel("X Norte [m]")
ax.grid(True)
ax.axis("equal")
ax.legend()

# ==========================================================
# FIGURA 2
# ALTITUD
# ==========================================================

fig2, ax = plt.subplots(figsize=(12,5))

if "L_z" in df.columns:
    ax.plot(
        t,
        df["L_z"],
        label="Lider"
    )

if "S_z" in df.columns:
    ax.plot(
        t,
        df["S_z"],
        "--",
        label="Seguidor"
    )

ax.set_title("Altitud")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Z [m]")
ax.grid(True)
ax.legend()

# ==========================================================
# FIGURA 3
# DISTANCIA
# ==========================================================

fig3, ax = plt.subplots(figsize=(12,5))

if "dist_xy" in df.columns:

    ax.plot(
        t,
        df["dist_xy"],
        label="Distancia XY"
    )

if "dist_3d" in df.columns:

    ax.plot(
        t,
        df["dist_3d"],
        "--",
        label="Distancia 3D"
    )

ax.set_title("Separacion Lider-Seguidor")
ax.set_xlabel("Tiempo [s]")
ax.set_ylabel("Distancia [m]")
ax.grid(True)
ax.legend()

# ==========================================================
# FIGURA 4
# VELOCIDADES
# ==========================================================

fig4, axs = plt.subplots(
    3,
    1,
    figsize=(12,8),
    sharex=True
)

componentes = [
    ("x","lvx"),
    ("y","lvy"),
    ("z","lvz")
]

for ax, (nombre, col) in zip(axs, componentes):

    L = f"L_{col}"
    S = f"S_{col}"

    if L in df.columns:

        ax.plot(
            t,
            df[L],
            label=f"Lider {nombre}"
        )

    if S in df.columns:

        ax.plot(
            t,
            df[S],
            "--",
            label=f"Seguidor {nombre}"
        )

    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel("Tiempo [s]")

fig4.suptitle("Velocidades Locales")

# ==========================================================
# FIGURA 5
# ACTITUD
# ==========================================================

fig5, axs = plt.subplots(
    3,
    1,
    figsize=(12,8),
    sharex=True
)

att = [
    ("roll","Roll"),
    ("pitch","Pitch"),
    ("yaw","Yaw")
]

for ax, (col, titulo) in zip(axs, att):

    L = f"L_{col}"
    S = f"S_{col}"

    if L in df.columns:

        ax.plot(
            t,
            np.degrees(df[L]),
            label="Lider"
        )

    if S in df.columns:

        ax.plot(
            t,
            np.degrees(df[S]),
            "--",
            label="Seguidor"
        )

    ax.set_ylabel("deg")
    ax.grid(True)
    ax.legend()
    ax.set_title(titulo)

axs[-1].set_xlabel("Tiempo [s]")

fig5.suptitle("Actitud")

# ==========================================================
# FIGURA 6
# MODOS
# ==========================================================

if "L_mode_name" in df.columns:

    fig6, ax = plt.subplots(
        figsize=(12,4)
    )

    cambios = (
        df["L_mode_name"] !=
        df["L_mode_name"].shift()
    )

    eventos = df[cambios]

    for _, row in eventos.iterrows():

        ax.axvline(
            row["t"],
            alpha=0.4
        )

        ax.text(
            row["t"],
            0.5,
            str(row["L_mode_name"]),
            rotation=90
        )

    ax.set_title(
        "Cambios de modo Lider"
    )

    ax.set_xlabel(
        "Tiempo [s]"
    )

    ax.set_yticks([])

# ==========================================================
# MOSTRAR
# ==========================================================

plt.tight_layout()
plt.show()
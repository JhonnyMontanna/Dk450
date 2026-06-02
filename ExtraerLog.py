from pymavlink import mavutil
import pandas as pd
import numpy as np

# ==========================================================
# CONFIGURACION
# ==========================================================

LEADER_TLOG   = "CirculoReal_Lider.tlog"
FOLLOWER_TLOG = "CirculoReal_Seguidor.tlog"

OUTPUT_CSV    = "master_telemetry.csv"

MERGE_TOLERANCE_SEC = 0.10

# ==========================================================
# MODOS ARDUCOPTER
# ==========================================================

COPTER_MODES = {
    0: "STABILIZE",
    1: "ACRO",
    2: "ALT_HOLD",
    3: "AUTO",
    4: "GUIDED",
    5: "LOITER",
    6: "RTL",
    7: "CIRCLE",
    9: "LAND",
    11: "DRIFT",
    13: "SPORT",
    14: "FLIP",
    15: "AUTOTUNE",
    16: "POSHOLD",
    17: "BRAKE",
    18: "THROW",
    19: "AVOID_ADSB",
    20: "GUIDED_NOGPS",
    21: "SMART_RTL"
}

# ==========================================================
# FUNCION DE EXTRACCION
# ==========================================================

def load_tlog(tlog_file, prefix):

    print(f"Leyendo {tlog_file}")

    mlog = mavutil.mavlink_connection(tlog_file)

    rows = []

    data = {

        f"{prefix}_mode": None,
        f"{prefix}_mode_name": None,

        f"{prefix}_roll": None,
        f"{prefix}_pitch": None,
        f"{prefix}_yaw": None,

        f"{prefix}_lat": None,
        f"{prefix}_lon": None,
        f"{prefix}_alt": None,

        f"{prefix}_gvx": None,
        f"{prefix}_gvy": None,
        f"{prefix}_gvz": None,

        f"{prefix}_x": None,
        f"{prefix}_y": None,
        f"{prefix}_z": None,

        f"{prefix}_lvx": None,
        f"{prefix}_lvy": None,
        f"{prefix}_lvz": None,

        f"{prefix}_target_lat": None,
        f"{prefix}_target_lon": None,
        f"{prefix}_target_alt": None,

        f"{prefix}_rangefinder": None,
    }

    t0 = None

    while True:

        msg = mlog.recv_match(blocking=False)

        if msg is None:
            break

        timestamp = getattr(msg, "_timestamp", None)

        if timestamp is None:
            continue

        if t0 is None:
            t0 = timestamp

        t = timestamp - t0

        mtype = msg.get_type()

        # --------------------------------------------------
        # ATTITUDE
        # --------------------------------------------------

        if mtype == "ATTITUDE":

            data[f"{prefix}_roll"] = msg.roll
            data[f"{prefix}_pitch"] = msg.pitch
            data[f"{prefix}_yaw"] = msg.yaw

        # --------------------------------------------------
        # GLOBAL POSITION
        # --------------------------------------------------

        elif mtype == "GLOBAL_POSITION_INT":

            data[f"{prefix}_lat"] = msg.lat / 1e7
            data[f"{prefix}_lon"] = msg.lon / 1e7
            data[f"{prefix}_alt"] = msg.alt / 1000.0

            data[f"{prefix}_gvx"] = msg.vx / 100.0
            data[f"{prefix}_gvy"] = msg.vy / 100.0
            data[f"{prefix}_gvz"] = msg.vz / 100.0

        # --------------------------------------------------
        # LOCAL POSITION NED
        # --------------------------------------------------

        elif mtype == "LOCAL_POSITION_NED":

            data[f"{prefix}_x"] = msg.x
            data[f"{prefix}_y"] = msg.y
            data[f"{prefix}_z"] = msg.z

            data[f"{prefix}_lvx"] = msg.vx
            data[f"{prefix}_lvy"] = msg.vy
            data[f"{prefix}_lvz"] = msg.vz

        # --------------------------------------------------
        # POSITION TARGET
        # --------------------------------------------------

        elif mtype == "POSITION_TARGET_GLOBAL_INT":

            data[f"{prefix}_target_lat"] = msg.lat_int / 1e7
            data[f"{prefix}_target_lon"] = msg.lon_int / 1e7
            data[f"{prefix}_target_alt"] = msg.alt

        # --------------------------------------------------
        # RANGEFINDER
        # --------------------------------------------------

        elif mtype == "RANGEFINDER":

            data[f"{prefix}_rangefinder"] = msg.distance

        # --------------------------------------------------
        # HEARTBEAT
        # --------------------------------------------------

        elif mtype == "HEARTBEAT":

            data[f"{prefix}_mode"] = msg.custom_mode

            data[f"{prefix}_mode_name"] = COPTER_MODES.get(
                msg.custom_mode,
                f"UNKNOWN_{msg.custom_mode}"
            )

        # --------------------------------------------------
        # GUARDAR FILA
        # --------------------------------------------------

        row = {"t": t}

        row.update(data)

        rows.append(row.copy())

    df = pd.DataFrame(rows)

    print(f"  {len(df)} filas")

    return df

# ==========================================================
# CARGAR LOGS
# ==========================================================

leader = load_tlog(
    LEADER_TLOG,
    "L"
)

follower = load_tlog(
    FOLLOWER_TLOG,
    "S"
)

# ==========================================================
# ORDENAR
# ==========================================================

leader = leader.sort_values("t")
follower = follower.sort_values("t")

# ==========================================================
# FUSION TEMPORAL
# ==========================================================

print("\nFusionando...")

master = pd.merge_asof(
    leader,
    follower,
    on="t",
    direction="nearest",
    tolerance=MERGE_TOLERANCE_SEC
)

# ==========================================================
# DISTANCIAS
# ==========================================================

if all(c in master.columns for c in
       ["L_x","L_y","L_z","S_x","S_y","S_z"]):

    master["dist_xy"] = np.sqrt(
        (master["L_x"] - master["S_x"])**2 +
        (master["L_y"] - master["S_y"])**2
    )

    master["dist_3d"] = np.sqrt(
        (master["L_x"] - master["S_x"])**2 +
        (master["L_y"] - master["S_y"])**2 +
        (master["L_z"] - master["S_z"])**2
    )

# ==========================================================
# ERROR DE ALTURA
# ==========================================================

if all(c in master.columns for c in
       ["L_z","S_z"]):

    master["dz"] = master["L_z"] - master["S_z"]

# ==========================================================
# EXPORTAR
# ==========================================================

master.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n====================================")
print("CSV maestro generado")
print("====================================")
print(OUTPUT_CSV)
print(f"Filas: {len(master)}")

print("\nColumnas:")

for c in master.columns:
    print(c)
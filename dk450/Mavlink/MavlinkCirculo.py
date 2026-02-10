#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil
from collections import deque

# ===============================
# CONFIGURACIÓN GENERAL
# ===============================
CONN = 'udp:127.0.0.1:14552'   # USB: 'COM6' o '/dev/ttyACM0'
SYSID = 1
COMPID = 0

RATE_CMD = 30          # Hz REALISTA para dron real
RATE_MONITOR = 5       # Hz para prints
MAX_NO_HEARTBEAT = 1.0 # s

# Trayectoria
RADIUS = 2.0
ANGULAR_SPEED = 0.9
LINEAR_SPEED = RADIUS * ANGULAR_SPEED

# MAVLink
TYPE_MASK = 0b0000111111000111  # solo velocidades

# ===============================
# ESTADO GLOBAL
# ===============================
last_heartbeat_time = 0.0
send_times = deque(maxlen=200)
recv_times = deque(maxlen=200)

# ===============================
def send_velocity(master, vx, vy, vz=0.0):
    master.mav.set_position_target_local_ned_send(
        int(time.time() * 1e6),
        SYSID, COMPID,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        TYPE_MASK,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, 0
    )
    send_times.append(time.time())

# ===============================
def read_feedback(master):
    global last_heartbeat_time
    msg = master.recv_match(blocking=False)
    if not msg:
        return

    t = time.time()
    recv_times.append(t)

    if msg.get_type() == "HEARTBEAT":
        last_heartbeat_time = t

# ===============================
def check_link_health():
    now = time.time()
    if now - last_heartbeat_time > MAX_NO_HEARTBEAT:
        print("❌ Heartbeat perdido → STOP")
        return False
    return True

# ===============================
def rate_info():
    if len(send_times) > 1:
        send_rate = len(send_times) / (send_times[-1] - send_times[0])
    else:
        send_rate = 0.0

    if len(recv_times) > 1:
        recv_rate = len(recv_times) / (recv_times[-1] - recv_times[0])
    else:
        recv_rate = 0.0

    return send_rate, recv_rate

# ===============================
def circle_motion(master, duration):
    dt = 1.0 / RATE_CMD
    steps = int(duration / dt)
    last_monitor = time.time()

    for i in range(steps):
        t = i * dt
        theta = ANGULAR_SPEED * t

        vx = LINEAR_SPEED * math.cos(theta + math.pi / 2)
        vy = LINEAR_SPEED * math.sin(theta + math.pi / 2)

        send_velocity(master, vx, vy)
        read_feedback(master)

        if not check_link_health():
            send_velocity(master, 0, 0, 0)
            return

        if time.time() - last_monitor > 1.0 / RATE_MONITOR:
            sr, rr = rate_info()
            print(f"📤 Cmd: {sr:5.1f} Hz | 📥 Rx: {rr:5.1f} Hz")
            last_monitor = time.time()

        time.sleep(dt)

    send_velocity(master, 0, 0, 0)

# ===============================
if __name__ == "__main__":
    print("🔗 Conectando...")
    master = mavutil.mavlink_connection(CONN)
    master.wait_heartbeat()
    last_heartbeat_time = time.time()

    print(f"✅ Conectado SYS={master.target_system}")

    duration = 2 * math.pi / ANGULAR_SPEED
    print(f"🌀 Círculo: R={RADIUS} m, duración≈{duration:.1f} s")

    circle_motion(master, duration)

    print("⏹️ Finalizado")
    master.close()

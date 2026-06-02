"""
Mide exactamente dónde se pierde el tiempo en el loop de replay.
Corre en tu máquina con ROS 2 activo y un CSV real:

  python3 bench_replay_loop.py --master vuelo1.csv --speed 10.0

Imprime un desglose por sección del loop.
"""
import sys
import time
import math
import argparse
import threading

import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time as RosTime

R_EARTH = 6_371_000.0

def gps_to_enu(lat, lon, ref_lat, ref_lon):
    x = math.radians(lat - ref_lat) * R_EARTH
    y = math.radians(lon - ref_lon) * R_EARTH * math.cos(math.radians(ref_lat))
    return x, y

def load_simple(path):
    df = pd.read_csv(path, sep=None, engine="python").sort_values("t").reset_index(drop=True)
    mask = df["L_lat"].notna() & df["L_lon"].notna()
    ref_lat = float(df[mask].iloc[0]["L_lat"])
    ref_lon = float(df[mask].iloc[0]["L_lon"])
    for pfx in ("L","S"):
        has = df[f"{pfx}_lat"].notna() & df[f"{pfx}_lon"].notna()
        ex = np.full(len(df), np.nan); ey = np.full(len(df), np.nan)
        if has.any():
            vals = df.loc[has,[f"{pfx}_lat",f"{pfx}_lon"]].apply(
                lambda r: gps_to_enu(r[f"{pfx}_lat"],r[f"{pfx}_lon"],ref_lat,ref_lon),axis=1)
            ex[has.values]=[v[0] for v in vals]; ey[has.values]=[v[1] for v in vals]
        df[f"{pfx}_rx"]=ex; df[f"{pfx}_ry"]=ey
    for pfx in ("L","S"):
        if f"{pfx}_alt" in df.columns and df[f"{pfx}_alt"].notna().any():
            av=df[f"{pfx}_alt"].values.astype(float)
            df[f"{pfx}_cz"]=av-np.nanmin(av[:20])
        elif f"{pfx}_z" in df.columns:
            df[f"{pfx}_cz"]=-df[f"{pfx}_z"].values.astype(float)
        else:
            df[f"{pfx}_cz"]=np.nan
    for c in ["L_qx","L_qy","L_qz","L_qw","S_qx","S_qy","S_qz","S_qw"]:
        if c not in df.columns: df[c]=np.nan
    return df

def _ros_stamp(t):
    msg=RosTime(); msg.sec=int(t); msg.nanosec=int((t%1)*1e9); return msg

def _arr(df, col):
    v=df[col].values.astype(float); v[np.isnan(v)]=0.0; return v

class BenchLoopNode(Node):
    def __init__(self, df, speed):
        super().__init__("bench_replay_loop")
        self.speed = speed
        self.t_arr = df["t"].values.astype(float)
        self.n     = len(self.t_arr)
        self.lx=_arr(df,"L_rx"); self.ly=_arr(df,"L_ry"); self.lz=_arr(df,"L_cz")
        self.sx=_arr(df,"S_rx"); self.sy=_arr(df,"S_ry"); self.sz=_arr(df,"S_cz")
        self.lqw=_arr(df,"L_qw"); self.sqw=_arr(df,"S_qw")
        if df["L_qw"].isna().all(): self.lqw[:]=1.0
        if df["S_qw"].isna().all(): self.sqw[:]=1.0

        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
        self.tf_br=TransformBroadcaster(self)
        self.odom1=self.create_publisher(Odometry,"/uav1/gps/odom",qos)
        self.odom2=self.create_publisher(Odometry,"/uav2/gps/odom",qos)
        self.mk1  =self.create_publisher(Marker,  "/uav1/gps/drone_marker",qos)
        self.mk2  =self.create_publisher(Marker,  "/uav2/gps/drone_marker",qos)

        self._stop=threading.Event()
        self._thread=threading.Thread(target=self._bench_loop,daemon=True)
        self._thread.start()

    def _pub(self, odom_pub, mk_pub, ns, x, y, z, qw, stamp):
        q=Quaternion(); q.w=float(qw)
        tf=TransformStamped()
        tf.header.stamp=stamp; tf.header.frame_id="rtk_odom"
        tf.child_frame_id=f"{ns}_base_link"
        tf.transform.translation.x=x; tf.transform.translation.y=y; tf.transform.translation.z=z
        tf.transform.rotation=q
        self.tf_br.sendTransform(tf)
        od=Odometry(); od.header.stamp=stamp; od.header.frame_id="rtk_odom"
        od.pose.pose.position.x=x; od.pose.pose.position.y=y; od.pose.pose.position.z=z
        od.pose.pose.orientation=q
        odom_pub.publish(od)
        mk=Marker(); mk.header=od.header; mk.type=Marker.SPHERE; mk.action=Marker.ADD
        mk.pose.position.x=x; mk.pose.position.y=y; mk.pose.position.z=z
        mk.pose.orientation.w=1.0; mk.scale.x=1.2; mk.scale.y=1.2; mk.scale.z=0.4
        mk.color=ColorRGBA(r=0.0,g=1.0,b=0.0,a=0.3)
        mk_pub.publish(mk)

    def _bench_loop(self):
        N        = min(500, self.n)
        t_arr    = self.t_arr
        speed    = self.speed

        # ── Tiempos por sección ──────────────────────────────────────────────
        t_sleep  = 0.0
        t_pub    = 0.0
        t_stamp  = 0.0
        overruns = 0

        wall_t0  = time.perf_counter()
        log_t0   = t_arr[0]

        print(f"\n[Bench] Midiendo {N} muestras a speed={speed}x …\n")

        for i in range(N):
            log_t    = t_arr[i]
            deadline = wall_t0 + (log_t - log_t0) / speed

            # --- sección sleep ---
            t0 = time.perf_counter()
            remaining = deadline - time.perf_counter()
            if remaining > 1e-4:
                time.sleep(remaining)
            slept = time.perf_counter() - t0

            # ¿llegamos tarde?
            late = time.perf_counter() - deadline
            if late > 0.001:
                overruns += 1

            t_sleep += slept

            # --- sección stamp ---
            t0 = time.perf_counter()
            stamp = _ros_stamp(log_t)
            t_stamp += time.perf_counter() - t0

            # --- sección publish ---
            t0 = time.perf_counter()
            self._pub(self.odom1, self.mk1, "uav1",
                      self.lx[i], self.ly[i], self.lz[i], self.lqw[i], stamp)
            self._pub(self.odom2, self.mk2, "uav2",
                      self.sx[i], self.sy[i], self.sz[i], self.sqw[i], stamp)
            t_pub += time.perf_counter() - t0

        wall_total = time.perf_counter() - wall_t0
        log_total  = t_arr[N-1] - log_t0
        expected   = log_total / speed

        print("=" * 55)
        print(f"  Muestras medidas   : {N}")
        print(f"  Duración log       : {log_total:.2f} s")
        print(f"  Esperado (speed={speed}x): {expected:.2f} s")
        print(f"  Tiempo real total  : {wall_total:.2f} s")
        print(f"  Ratio real/esperado: {wall_total/expected:.2f}x  (1.0 = perfecto)")
        print()
        print(f"  Desglose por muestra:")
        print(f"    sleep   : {t_sleep/N*1000:.3f} ms  ({t_sleep/wall_total*100:.1f}%)")
        print(f"    publish : {t_pub/N*1000:.3f} ms  ({t_pub/wall_total*100:.1f}%)")
        print(f"    stamp   : {t_stamp/N*1000:.3f} ms  ({t_stamp/wall_total*100:.1f}%)")
        print(f"    overruns: {overruns}/{N} muestras llegaron tarde (>1ms)")
        print("=" * 55)

        if wall_total / expected > 1.5:
            print("\n⚠️  El replay va más lento de lo esperado.")
            if t_pub/N*1000 > 5:
                print("   → Cuello de botella: publish() (ROS/DDS lento en este contexto)")
            elif overruns > N * 0.1:
                print("   → Cuello de botella: sleep() no es preciso o hay contención de hilos")
            else:
                print("   → Cuello de botella: executor de ROS bloqueando el hilo de replay")
        else:
            print("\n✅ Timing correcto. Si RViz va lento, el problema es de visualización.")

        self._stop.set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--master", default="master_telemetry.csv")
    ap.add_argument("--speed",  type=float, default=10.0)
    args=ap.parse_args()

    try:
        df=load_simple(args.master)
    except FileNotFoundError:
        print(f"[ERROR] No se encontró '{args.master}'"); sys.exit(1)

    rclpy.init()
    node=BenchLoopNode(df, args.speed)
    executor=MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        while not node._stop.is_set():
            executor.spin_once(timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()
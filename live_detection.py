#live_detection.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, Imu
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped
import numpy as np
import bisect
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_msgs.msg import TFMessage
import math
from collections import defaultdict, deque
import sensor_msgs_py.point_cloud2 as pc2
from scipy import ndimage
import dendromatics as dm
import time
import json
import os
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from rclpy.qos import QoSProfile, DurabilityPolicy, qos_profile_sensor_data


DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_ccl")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_output")
os.makedirs(OUT_DIR, exist_ok=True)

# per-component verdicts recorded in the CCL dumps
ACCEPTED, REJ_NMIN, REJ_SUPPORT, REJ_RADIUS = 0, 1, 2, 3
VERDICT_NAME = {ACCEPTED: "accepted", REJ_NMIN: "n_min",
                REJ_SUPPORT: "support", REJ_RADIUS: "radius"}

def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9

def quat_to_R(q): #converts [w,x,y,z] to 3x3 rotation
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
def slerp(q0, q1, a):

    #dot product of two quaternions to determine how similar they are; 
    d = float(np.dot(q0, q1))  

    if d < 0.0: # close to -1->pointing in opposite directions
        q1, d = -q1, -d #align their directions by flipping one

    if d > 0.9995:  #close to 1->nearly parallel; 
        q = (1.0 - a) * q0 + a * q1
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    th = th0 * a
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return s0 * q0 + s1 * q1

class Cfg:
    voxel = 0.10            
    z_min = 0.15 #0.4 did not work        
    z_max = 0.5            
    n_min = 5 #100->30 trees
    detect_every = 10
    det_radius = 10.0        
    self_radius = 0.0      
    z_gnd_min = -0.4      
    z_gnd_max = 0.2        
    ground_support_frac = 0.25 
    r_min = 0.05            
    r_max = 2.7             
    d_match = max(5 * voxel, r_max)
    reg_cell = 5.0           
    pose_max_dt = 0.05       
    max_pending = 20        
    log_sensor_msgs = False 
    quiet_every = 50         
    cloth_resolution = 0.3   
    dtm_rebuild_every = 20   # detection cycles between DTM rebuilds
    ground_cell = 0.25     
    ground_min_hits = 5 
    ground_fill_rings = 2    
    merge_floor = 0
    debug_ccl_every = 1

class SensorStreamReader(Node):
    def __init__(self):
        super().__init__('sensor_stream_reader')
        self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.start_time = time.time()   #wall-clock start
        self.total_detect_time = 0.0    #cumulative time spent in detect_trunks
        self.n_detect_calls = 0         #how many times detect_trunks actually ran
        self.cfg = Cfg
        self.poses = PoseBuffer()
        self.occ_map = OccupancyMap(Cfg.voxel)
        self.registry = TreeRegistry(Cfg.d_match, Cfg.reg_cell, Cfg.merge_floor)
        self.n_skipped_no_pose = 0
        self.n_pose_msgs = 0
        self.pending = deque()   # scans held until their GLIM pose arrives
        self.ground = GroundModel(Cfg.dtm_rebuild_every, Cfg.cloth_resolution)
        self.detect_cycles = 0
        self.total_pipeline_time = 0.0
        self.n_pipeline_runs = 0

        #publisher for centroids
        tree_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_trees = self.create_publisher(Float64MultiArray, '/trees/registry', tree_qos)

        # voxel map from octomap (replaces the occ map accumulation)
        self.latest_oct = None
        self.sub_oct = self.create_subscription(
            PointCloud2, '/trunks/octomap_point_cloud_centers',
            self.octomap_callback, qos_profile_sensor_data)

        #lidar message
        self.lidar_count = 0
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/unitree_go2/lidar/point_cloud', self.lidar_callback, 10)

        #GLIM's lidar pose in the map frame, loop-closure corrected.
        self.sub_pose = self.create_subscription(
            PoseStamped, '/glim_ros/lidar_pose', self.pose_callback, 100)
        self.sub_pose_corr = self.create_subscription(
            PoseStamped, '/glim_ros/lidar_pose_corrected', self.pose_callback, 100)

        #camera message
        self.camera_count = 0        
        self.sub_camera = self.create_subscription(
            Image, '/unitree_go2/front_cam/color_image', self.camera_callback, 10)

        #imu message
        self.imu_count = 0
        self.sub_imu = self.create_subscription(
            Imu, '/unitree_go2/imu', self.imu_callback, 10)

        #tf node
        self.tf_sub = self.create_subscription(
            TFMessage, '/tf', self.tf_callback, 50)

        #tf static node
        #using qos to retain the tf_static msg since generally if a msg is published and there is not subscriber to 
        # receive it's discarded, and since tf_static is only published once, we hv to retain it 
        #tf2_ros.StaticTransformBroadcaster publishes with transient-local durability
        #  by default, so any subscriber needs to match that setting to actually 
        # receive it. depth=10 just caps how many recent msgs the cache holds.
        static_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static_sub = self.create_subscription(
            TFMessage, '/tf_static', self.tf_static_callback, static_qos)


        self.get_logger().info("Subscribed to lidar, camera, imu, tf, tf_static, glim pose")

    #work on incoming scan
    def lidar_callback(self, msg):
        self.lidar_count += 1
        pts, _ = read_xyzt(msg)
        if pts is None or len(pts) == 0:
            return
        self.pending.append((stamp_to_sec(msg.header.stamp), pts, self.lidar_count, time.time()))#added time to measure time diff btw frame and its pose being published
        self._drain() #wait for pose from glim, there is a one scan frame latency when waiting. drain when found.

    def pose_callback(self, msg):
            p = msg.pose.position
            q = msg.pose.orientation
            self.n_pose_msgs += 1
            if self.n_pose_msgs == 1:
                p = msg.pose.position
                self.get_logger().info(
                    f"first GLIM pose received: stamp={stamp_to_sec(msg.header.stamp):.3f} "
                    f"pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) frame={msg.header.frame_id}")            #add pose to posebuffer
            self.poses.add(
                stamp_to_sec(msg.header.stamp),
                np.array([p.x, p.y, p.z], dtype=np.float64),
                np.array([q.w, q.x, q.y, q.z], dtype=np.float64),
            )
            self._drain()
    def _warn_no_pose(self, stamp, frame_idx):
        """Say WHICH of the two failure modes this is: no poses arriving at
        all, or poses arriving whose stamps don't cover this scan."""
        if self.n_pose_msgs == 0:
            self.get_logger().warn(
                f"frame {frame_idx}: no GLIM pose for stamp={stamp:.3f} "
                f"-- 0 messages on /glim_ros/lidar_pose[_corrected]. GLIM is "
                f"not publishing: check `ros2 topic hz /glim_ros/lidar_pose`, "
                f"that glim_rosnode is up with librviz_viewer.so loaded, and "
                f"that it was started BEFORE the bag. "
                f"(skipped {self.n_skipped_no_pose})")
            return
        lo, hi = self.poses.stamps[0], self.poses.stamps[-1]
        i = bisect.bisect_left(self.poses.stamps, stamp)
        window = self.poses.stamps[max(0, i - 1):i + 1]
        near = min((abs(v - stamp) for v in window), default=float('inf'))
        self.get_logger().warn(
            f"frame {frame_idx}: no GLIM pose for stamp={stamp:.3f} -- "
            f"{self.n_pose_msgs} poses received, buffer spans "
            f"[{lo:.3f}..{hi:.3f}], nearest dt={near:.3f}s > pose_max_dt="
            f"{self.cfg.pose_max_dt}s (skipped {self.n_skipped_no_pose})")


    #to keep dropping timestamps from pending queue if glim has moved to next pose 
    #so move on to next timestamp if there is no pose for this stamp or give warning if no pose for entire frame
    # or queue has outgrown its cap
    #else keep waiting
 
    def _drain(self):
        while self.pending:
            stamp, pts, frame_idx, arrival_time = self.pending[0] 
            pose = self.poses.lookup(stamp, self.cfg.pose_max_dt)
            if pose is None: #new timestamp                
                newest = self.poses.stamps[-1] if self.poses.stamps else None
                overtaken = newest is not None and newest > stamp + self.cfg.pose_max_dt #if new stamp w no pose and outside max_dt
                if overtaken or len(self.pending) > self.cfg.max_pending: #if no glim pose exists
                    self.pending.popleft()
                    self.n_skipped_no_pose += 1
                    if self.n_skipped_no_pose % 20 == 1: #if no pose for whole frame
                        self._warn_no_pose(stamp, frame_idx)
                    continue
                return
            wait_time = time.time() - arrival_time
            #self.get_logger().info(f"frame {frame_idx}: waited {wait_time:.4f} s for pose") #log time for each frame's pose coming in
            with open(os.path.join(OUT_DIR, "pose_wait_times.csv"), "a") as f:
                f.write(f"{frame_idx},{wait_time:.6f}\n")
            self.pending.popleft() #if pose exists, move to next timestamp after acquiring
            self._process(stamp, pts, frame_idx, pose, arrival_time)

    def octomap_callback(self, msg):
        self.latest_oct = msg      # just keep the newest; convert once per detection cycle

    def _process(self, stamp, pts, frame_idx, pose, arrival_time):
        R, t = pose

        #all 20K points per frame use the same sensor pose since its firing at same time
        pts_map = pts @ R.T + t #here pts_map= R @ pts +t but since pts=1x3 and not 3x1:(R @ p.T).T = p @ R.T

        # fold the scan into the accumulated occupancy map. Algorithm 1
        #    consumes the map, not this scan 

        #keep = ((pts_map[:, :2] - t[:2]) ** 2).sum(axis=1) >= self.cfg.self_radius ** 2
        #self.occ_map.update(pts_map[keep])

        #Algorithm 1 over the map window around the sensor
        if frame_idx % self.cfg.detect_every:
            return

        #OCTOMAP APPROACH: the octomap is already an accumulated occupancy map, 
        # so don't accumulate it again. Just convert it to OccupancyMap and run detection on that.
        if self.latest_oct is None:
            return                              # octomap not publishing yet
        oct_pts, _ = read_xyzt(self.latest_oct)
        if oct_pts is None or len(oct_pts) == 0:
            return
        snap = OccupancyMap(self.cfg.voxel)
        snap.update(oct_pts)
        self.occ_map = snap                     # REPLACE, don't accumulate: octomap already does that


        t0 = time.time()  #time detect trunks pipeline

        # the DTM is rebuilt on a slow cadence and cached in between
        self.detect_cycles += 1
        built = self.ground.maybe_rebuild(self.occ_map, self.detect_cycles)
        if built:
            self.get_logger().info(
                f"DTM rebuild #{self.ground.n_builds}: {built:.2f} s over "
                f"{len(self.occ_map)} columns")
       # dump this cycle's CCL raster for visualize_ccl.py?
        dbg_path = None
        if (self.cfg.debug_ccl_every
                and self.detect_cycles % self.cfg.debug_ccl_every == 0):
            os.makedirs(DEBUG_DIR, exist_ok=True)
            dbg_path = os.path.join(DEBUG_DIR, f"ccl_{self.detect_cycles:05d}.npz")

        dets = detect_trunks(self.occ_map, self.ground, t[:2], self.cfg,
                             debug_path=dbg_path)

        total_pipeline_time = time.time() - arrival_time
        self.get_logger().info(
            f"frame {frame_idx}: FULL PIPELINE (arrival -> centroids) took {total_pipeline_time:.4f} s"
        )

        self.total_pipeline_time += total_pipeline_time
        self.n_pipeline_runs += 1
        self.total_detect_time += time.time() - t0      
        self.n_detect_calls += 1                      

        if frame_idx % 500 == 0:  # to visualize dump occ map every 500 accepted frames, tune as needed
            cand = self.occ_map.ground_candidates()
            np.save(os.path.join(OUT_DIR, f"debug_occmap_frame{frame_idx}.npy"), cand)
            if self.ground.dtm is not None:
                np.save(os.path.join(OUT_DIR, f"debug_dtm_frame{frame_idx}.npy"), self.ground.dtm)


        # Algorithm 2: persistent identity association against the
        #    global list. Detections are already world-frame.
        events = self.registry.update(dets, frame_idx)

        # registry state + this cycle's association decisions, alongside the
        # raster, so the two can be overlaid frame by frame
        if dbg_path is not None:
            np.save(os.path.join(DEBUG_DIR,
                                 f"registry_{self.detect_cycles:05d}.npy"),
                    self.registry.rows(min_obs=1))
            with open(os.path.join(DEBUG_DIR,
                                   f"events_{self.detect_cycles:05d}.json"), "w") as fh:
                json.dump({
                    "cycle": int(self.detect_cycles),
                    "frame": int(frame_idx),
                    "sensor_xy": [float(t[0]), float(t[1])],
                    "det_radius": float(self.cfg.det_radius),
                    "n_dets": int(len(dets)),
                    "n_trees": int(len(self.registry)),
                    "events": [[str(k),
                                int(a) if not isinstance(a, tuple) else a,
                                (float(b) if np.isscalar(b) else
                                 [float(v) for v in b])]
                               for k, a, b in events],
                }, fh, indent=1)

        #publishing the registry's centroids x,y
        rows = self.registry.rows(min_obs=1)
        msg = Float64MultiArray()
        msg.layout.dim = [MultiArrayDimension(label='rows', size=rows.shape[0], stride=rows.size),
                        MultiArrayDimension(label='cols', size=7, stride=7)]
        msg.data = rows.ravel().tolist()
        self.pub_trees.publish(msg)



        if not len(dets):
            if frame_idx % self.cfg.quiet_every == 0:
                self.get_logger().info(
                    f"frame {frame_idx}: 0 components | "
                    f"trees={len(self.registry)}")
            return

        matched = [(k, d) for kind, k, d in events if kind == 'match']
        new = [(k, v) for kind, k, v in events if kind == 'new']
        merges = [(a, b) for kind, a, b in events if kind == 'merge']
        parts = [f"frame {frame_idx}: {len(dets)} components"]
        if matched:
            parts.append("same-tree " + ", ".join(
                f"#{k} (d={d:.2f}m)" for k, d in matched))
        if new:
            parts.append("NEW " + ", ".join(
                f"#{k} (x={v[0]:.2f} y={v[1]:.2f} r={v[2]:.2f})" for k, v in new))
        if merges:
            parts.append("overlap-merge " + ", ".join(
                f"#{a}<-#{b}" for a, b in merges))
        parts.append(f"trees={len(self.registry)} "
                        f"(map={len(self.occ_map)} cols)")
        self.get_logger().info(" | ".join(parts))


    # ------------------------------------------------------------------ #
    def camera_callback(self, msg):
        self.camera_count += 1
        if not self.cfg.log_sensor_msgs:
            return
        self.get_logger().info(
            f"[CAMERA] frame {self.camera_count}: {msg.width}x{msg.height}, "
            f"encoding={msg.encoding}, stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def imu_callback(self, msg):
        self.imu_count += 1
        if self.cfg.log_sensor_msgs and self.imu_count % 100 == 0:
            self.get_logger().info(
                f"[IMU] msg {self.imu_count}: "
                f"accel=({msg.linear_acceleration.x:.2f}, {msg.linear_acceleration.y:.2f}, {msg.linear_acceleration.z:.2f}), "
                f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

    def tf_callback(self, msg):
        pass  # GLIM's pose comes from the pose topic, not TF -- see NOTE below

    def tf_static_callback(self, msg):
        if not self.cfg.log_sensor_msgs:
            return
        for transform in msg.transforms:
            self.get_logger().info(
                f"[TF_STATIC] {transform.header.frame_id} -> {transform.child_frame_id}")
            
    def save_registry(self, path=os.path.join(OUT_DIR, "tree_registry.npy")):
        """Print the global list and save it.

        print, not get_logger: on Ctrl-C the rclpy context is already down and
        rosout publishing fails.
        """
        rows = self.registry.rows(min_obs=1)
        np.save(path, rows)

        elapsed = time.time() - self.start_time   
        avg_detect_ms = (self.total_detect_time / self.n_detect_calls * 1000
                      if self.n_detect_calls else 0.0)               
        print()
        print("=" * 68)
        print("TOTAL DETECTED TREES: %d" % len(rows))
        print("=" * 68)
        print("%4s  %5s  %-9s  %9s  %9s  %7s  %6s"
              % ("id", "disc", "status", "x [m]", "y [m]", "r [m]", "n_obs"))
        print("-" * 68)
        for gid, x, y, r, status, disc, n_obs in rows:
            print("%4d  %5d  %-9s  %9.2f  %9.2f  %7.3f  %6d"
                  % (gid, disc, STATUS_NAME[int(status)], x, y, r, n_obs))
        print("-" * 68)
        print("%d lidar frames, %d skipped for want of a GLIM pose, "
              "map %d columns, d_match=%.2f m"
              % (self.lidar_count, self.n_skipped_no_pose,
                 len(self.occ_map), self.registry.d_match))
        print("-" * 68)
        print("TIMING: %.1f s wall-clock total | %d detection cycles | "
          "avg %.1f ms per detect_trunks() call | %.1f s total in detect_trunks()"
          % (elapsed, self.n_detect_calls, avg_detect_ms, self.total_detect_time))

        print("saved to %s : columns "
              "[global_id, x, y, r, status, discovery_order, n_obs]" % path)

        avg_pipeline_ms = (self.total_pipeline_time / self.n_pipeline_runs * 1000) if self.n_pipeline_runs else 0.0
        print(f"FULL PIPELINE: {self.n_pipeline_runs} runs | avg {avg_pipeline_ms:.1f} ms | "
            f"{self.total_pipeline_time:.1f}s total")

#dump final map to visualize
        final_map = self.occ_map.ground_candidates()
        np.save(os.path.join(OUT_DIR, "final_occmap.npy"), final_map)
        if self.ground.dtm is not None:
            np.save(os.path.join(OUT_DIR, "final_dtm.npy"), self.ground.dtm)

    #full occupancy map, all heights ,matches static script's pcd_arr
        full_map = self.occ_map.all_voxels()
        np.save(os.path.join(OUT_DIR, "final_occmap_full.npy"), full_map)
        print(f"saved full occupancy map: {len(full_map)} voxels "
            f"(vs {len(final_map)} ground-candidate columns)")




class PoseBuffer:
#to store sensor poses recieved from glim in map frame
    def __init__(self, maxlen=2000):
        self.maxlen = maxlen
        self.stamps = []        # sorted list of float seconds
        self.pos = []           # list of (3,) float64
        self.quat = []          # list of (4,) float64, [w, x, y, z]

    def add(self, stamp_sec, pos, quat):
        i = bisect.bisect_left(self.stamps, stamp_sec) #binary search to correctly place new stamp

        #if stamp already exists, replace w new one
        if i < len(self.stamps) and abs(self.stamps[i] - stamp_sec) < 1e-9:
            self.pos[i], self.quat[i] = pos, quat      # corrected re-publish
            return

        #else add to list
        self.stamps.insert(i, stamp_sec)
        self.pos.insert(i, pos)
        self.quat.insert(i, quat)

        #if buffer full, remove the first one
        if len(self.stamps) > self.maxlen:
            del self.stamps[0], self.pos[0], self.quat[0]

    #check if sensor pose for this stamp exists
    #the pose returned is used for LiDAR points -> world frame transform
    def lookup(self, stamp_sec, max_dt):
        #returns (R, t) for stamp_sec, or None if we can't cover it
        if not self.stamps:
            return None
        i = bisect.bisect_left(self.stamps, stamp_sec)

        # exact hit
        if i < len(self.stamps) and abs(self.stamps[i] - stamp_sec) < 1e-9:
            return quat_to_R(self.quat[i]), self.pos[i]

        #timestamp lies outside the poses stored; either before the first pose or after last
        if i == 0 or i == len(self.stamps):
            j = 0 if i == 0 else len(self.stamps) - 1
            if abs(self.stamps[j] - stamp_sec) > max_dt:
                return None
            return quat_to_R(self.quat[j]), self.pos[j]

        #interpolation -> this doesnt happen since i am getting pose for every frame
        #take previous and next stamp  
        t0, t1 = self.stamps[i - 1], self.stamps[i]

        #check if time difference btw desired time and neighboring time lie within max_dt
        if (stamp_sec - t0) > max_dt and (t1 - stamp_sec) > max_dt:
            return None
        
        #interpolation fraction; determines how far along the timestamp is from t0 to t1 (btw 0-1)
        a = (stamp_sec - t0) / (t1 - t0)

        #interpolating position
        pos = (1.0 - a) * self.pos[i - 1] + a * self.pos[i]
        #interpolating orientation
        quat = slerp(self.quat[i - 1], self.quat[i], a)
        return quat_to_R(quat), pos
   

class GroundGrid: #superseded by dtm dendromatics
#to replace DTM, incremental expandable ground grid storing min Z per cell

    def __init__(self, cell, min_hits, fill_rings):
        self.cell = cell #cell size
        self.min_hits = min_hits #min points required to trust min Z 
        self.fill_rings = fill_rings #checks if cell points> min_hits if not then check 8 neighbors (1 ring), or 16 neighboring cells(2 rings) 
        self.origin = None #(i0, j0) cell index of array [0,0]
        self.min_z = None   # +inf where unseen, a 2D array of size shape(x,y)
        self.count = None   # hit count per cell, a 2D array of size shape(x,y)

    def _cell_idx(self, xy): #where real world index is placed on which grid cell
        return np.floor(xy / self.cell).astype(np.int64)

    def _ensure(self, i_lo, i_hi, j_lo, j_hi):
        #grow the array so [i_lo, i_hi] x [j_lo, j_hi] cell indices fit
        if self.origin is None:
            pad = 64 #can change if needed
            self.origin = (i_lo - pad, j_lo - pad)
            shape = (i_hi - i_lo + 1 + 2 * pad, j_hi - j_lo + 1 + 2 * pad)
            self.min_z = np.full(shape, np.inf, dtype=np.float32)
            self.count = np.zeros(shape, dtype=np.uint32)
            return
        
        #check if the new frame fits within the existing array bounds
        i0, j0 = self.origin
        h, w = self.min_z.shape
        gi_lo = min(0, i_lo - i0) #if cell index lies below
        gj_lo = min(0, j_lo - j0) 
        gi_hi = max(h, i_hi - i0 + 1)
        gj_hi = max(w, j_hi - j0 + 1)
        if gi_lo == 0 and gj_lo == 0 and gi_hi == h and gj_hi == w: #if fits return
            return

        #if new frame exceeds the current bounds, resize
        pad = 64
        gi_lo -= pad if gi_lo < 0 else 0
        gj_lo -= pad if gj_lo < 0 else 0
        gi_hi += pad if gi_hi > h else 0
        gj_hi += pad if gj_hi > w else 0

        new_z = np.full((gi_hi - gi_lo, gj_hi - gj_lo), np.inf, dtype=np.float32)
        new_c = np.zeros(new_z.shape, dtype=np.uint32)

        #copy the old data into new correct spots
        new_z[-gi_lo:-gi_lo + h, -gj_lo:-gj_lo + w] = self.min_z
        new_c[-gi_lo:-gi_lo + h, -gj_lo:-gj_lo + w] = self.count
        #discard old values and update origin
        self.min_z, self.count = new_z, new_c
        self.origin = (i0 + gi_lo, j0 + gj_lo)


    def update(self, pts_map):
        idx = self._cell_idx(pts_map[:, :2])
        self._ensure(idx[:, 0].min(), idx[:, 0].max(),
                     idx[:, 1].min(), idx[:, 1].max())
        i0, j0 = self.origin
        r = idx[:, 0] - i0 #real world coordinates correspondence to array coordinates
        c = idx[:, 1] - j0

        #checks the cell at (r,c) and compares all z values received
        np.minimum.at(self.min_z, (r, c), pts_map[:, 2].astype(np.float32))

        #add point count for the respective updated cell
        np.add.at(self.count, (r, c), 1)

    def trusted(self):
        #boolean mask of cells with enough hits to believe their min-Z
        return self.count >= self.min_hits

    def query(self, xy):
    #this returns min z for queried cells with min_hits check
        n = len(xy)
        z = np.full(n, np.nan, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)
        if self.origin is None:
            return z, valid

        i0, j0 = self.origin
        h, w = self.min_z.shape
        idx = self._cell_idx(xy)
        r = idx[:, 0] - i0
        c = idx[:, 1] - j0

        trusted = self.trusted()
        # rings 0..fill_rings: take the min over a growing square neighbourhood
        for ring in range(self.fill_rings + 1):
            todo = ~valid #new array, opp of valid; set all True
            if not todo.any(): #if all are False->valid then exit
                break
            best = np.full(todo.sum(), np.inf, dtype=np.float32) #sums all True and adds inf as value
            rr, cc = r[todo], c[todo] #take r and c for True ones
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if ring > 0 and abs(di) != ring and abs(dj) != ring:
                        continue          # interior already covered by prev ring
                    ri, cj = rr + di, cc + dj
                    ok = (ri >= 0) & (ri < h) & (cj >= 0) & (cj < w) #check if within bounds
                    if not ok.any():
                        continue
                    v = np.full(len(ri), np.inf, dtype=np.float32)
                    v[ok] = np.where(trusted[ri[ok], cj[ok]],
                                     self.min_z[ri[ok], cj[ok]], np.inf) #v stores min z value if present else inf
                    best = np.minimum(best, v) #take min from best vs v and store in best
            got = np.isfinite(best) #returns an array where values are true if best[element] is not inf else false 
            sel = np.flatnonzero(todo)[got] #takes a boolean array and returns position indices array where value=true
            z[sel] = best[got]
            valid[sel] = True
        return z, valid


class OccupancyMap:
    TILE = 64                                  # columns per tile edge

    def __init__(self, voxel):
        self.voxel = voxel #voxel size
        self.cols = defaultdict(set)           # (i, j) -> {k}
        self.tiles = defaultdict(set)          # (ti, tj) -> {(i, j)}

    def update(self, pts_map):
        if len(pts_map) == 0:
            return

        #convert pts in world frame to voxel indices
        idx = np.floor(pts_map / self.voxel).astype(np.int64)
        
        # dedup within the frame first: 20k points collapse to a few thousand
        # voxels, and the dict work below is the expensive part.
        idx = np.unique(idx, axis=0)
        for i, j, k in idx:
            key = (int(i), int(j))
            col = self.cols[key]
            if not col:
                self.tiles[(int(i) // self.TILE, int(j) // self.TILE)].add(key)
            col.add(int(k))

    def window(self, centre_xy, radius):
        lo = np.floor((np.asarray(centre_xy) - radius) / self.voxel).astype(np.int64)
        hi = np.floor((np.asarray(centre_xy) + radius) / self.voxel).astype(np.int64)
        keys = []
        for ti in range(lo[0] // self.TILE, hi[0] // self.TILE + 1):
            for tj in range(lo[1] // self.TILE, hi[1] // self.TILE + 1):
                for key in self.tiles.get((ti, tj), ()):
                    if lo[0] <= key[0] <= hi[0] and lo[1] <= key[1] <= hi[1]:
                        keys.append(key)
        if not keys:
            return np.zeros((0, 2), dtype=np.int64), []
        arr = np.array(keys, dtype=np.int64)
        xy = (arr + 0.5) * self.voxel
        keep = ((xy - centre_xy) ** 2).sum(axis=1) <= radius ** 2
        arr = arr[keep]
        return arr, [self.cols[(int(a), int(b))] for a, b in arr]

    def ground_candidates(self):
        out = np.empty((len(self.cols), 3), dtype=np.float64)
        for n, ((i, j), ks) in enumerate(self.cols.items()):
            out[n, 0] = (i + 0.5) * self.voxel
            out[n, 1] = (j + 0.5) * self.voxel
            out[n, 2] = (min(ks) + 0.5) * self.voxel
        return out

    def __len__(self):
        return len(self.cols)

#to get the fully accumulated occ map
    def all_voxels(self):
        """Every occupied voxel in the map, as an (N, 3) real-world cloud —
        not just the per-column minimum. Matches the static script's raw
        pcd_arr for comparison."""
        total = sum(len(ks) for ks in self.cols.values())
        out = np.empty((total, 3), dtype=np.float64)
        n = 0
        for (i, j), ks in self.cols.items():
            x = (i + 0.5) * self.voxel
            y = (j + 0.5) * self.voxel
            for k in ks:
                out[n, 0] = x
                out[n, 1] = y
                out[n, 2] = (k + 0.5) * self.voxel
                n += 1
        return out
# --------------------------------------------------------------------------- #
# ground model: cloth-simulation DTM, rebuilt periodically from the map
# --------------------------------------------------------------------------- #
class GroundModel:
    def __init__(self, rebuild_every, cloth_resolution):
        self.rebuild_every = rebuild_every
        self.cloth_resolution = cloth_resolution
        self.dtm = None
        self.built_at = -10 ** 9
        self.n_builds = 0
        self.build_seconds = 0.0

    def maybe_rebuild(self, occ_map, cycle):
        """Returns the build time in seconds, or 0.0 if the cache still holds."""
        if self.dtm is not None and cycle - self.built_at < self.rebuild_every:
            return 0.0
        cand = occ_map.ground_candidates()
        if len(cand) < 100:                 # too early to fit anything
            return 0.0
        t0 = time.perf_counter()
        self.dtm = dm.complete_dtm(
            dm.generate_dtm(cand, bSloopSmooth=True,
                            cloth_resolution=self.cloth_resolution))
        dt = time.perf_counter() - t0
        self.built_at = cycle
        self.n_builds += 1
        self.build_seconds += dt
        return dt

    def ground_z(self, xy):
        """Ground height under each XY. Returns (gz, ok).

        normalize_heights(p, dtm) is p_z - ground(p_xy), so probing at z = 0
        returns -ground.
        """
        if self.dtm is None:
            return np.zeros(len(xy)), np.zeros(len(xy), dtype=bool)
        probe = np.column_stack([xy, np.zeros(len(xy))])
        gz = -np.asarray(dm.normalize_heights(probe, self.dtm), dtype=np.float64)
        return gz, np.isfinite(gz)


# --------------------------------------------------------------------------- #
# Algorithm 1: occupied-voxel 2-D CCL with multi-stage filtering
# --------------------------------------------------------------------------- #
def detect_trunks(occ_map, ground, sensor_xy, cfg=Cfg, debug_path=None):
    empty = np.zeros((0, 3))
    keys, ks_list = occ_map.window(sensor_xy, cfg.det_radius)
    if len(keys) == 0:
        return empty

    xy = (keys + 0.5) * cfg.voxel
    far = ((xy - sensor_xy) ** 2).sum(axis=1) >= cfg.self_radius ** 2
    if not far.any():
        return empty
    keys, xy = keys[far], xy[far]
    ks_list = [ks for ks, f in zip(ks_list, far) if f]

    # ground height per column
    gz, gok = ground.ground_z(xy)       # one ground-Z estimate per column
    keys, xy, gz = keys[gok], xy[gok], gz[gok]
    ks_list = [ks for ks, ok in zip(ks_list, gok) if ok]
    if len(keys) == 0:
        return empty

    counts = np.fromiter((len(ks) for ks in ks_list), dtype=np.int64, count=len(ks_list))
    col_id = np.repeat(np.arange(len(keys)), counts)
    kk = np.fromiter((k for ks in ks_list for k in ks), dtype=np.int64, count=int(counts.sum()))
    z_abs = (kk + 0.5) * cfg.voxel
    z0 = z_abs - gz[col_id]              # height above the DTM, per voxel

    in_slab = (z0 >= cfg.z_min) & (z0 <= cfg.z_max)
    in_gnd = (z0 >= cfg.z_gnd_min) & (z0 <= cfg.z_gnd_max)
    n = len(keys)
    slab_col = np.bincount(col_id[in_slab], minlength=n) > 0
    gnd_col = np.bincount(col_id[in_gnd], minlength=n) > 0

    if not slab_col.any():
        return empty
    
#remaining pipeline
    sel = keys[slab_col]
    i_min, j_min = sel.min(axis=0)
    i_max, j_max = sel.max(axis=0)
    shape = (i_max - i_min + 1, j_max - j_min + 1)
    occ = np.zeros(shape, dtype=bool)
    occ[sel[:, 0] - i_min, sel[:, 1] - j_min] = True
    # ground support carried on the same raster, so a component can be tested
    # cell by cell against what lies under it
    sup = np.zeros(shape, dtype=bool)
    gsel = keys[slab_col & gnd_col]
    if len(gsel):
        sup[gsel[:, 0] - i_min, gsel[:, 1] - j_min] = True

    labels, n_comp = ndimage.label(occ, structure=np.ones((3, 3), dtype=int))
    if n_comp == 0:
        return empty

    dets = []
    dbg = [] if debug_path is not None else None

    for label_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sub = labels[sl] == label_id
        n_cells = int(sub.sum())
        if n_cells < cfg.n_min:                                  # filter (i)
            if dbg is not None:
                dbg.append((label_id, n_cells, float(sup[sl][sub].mean()),
                            np.nan, np.nan, np.nan, REJ_NMIN))
            continue

        li, lj = np.nonzero(sub)
        gi = li + sl[0].start
        gj = lj + sl[1].start

        # filter (ii): does the component stand on the ground?
        support = sup[sl][sub].mean()
        if support < cfg.ground_support_frac:
            if dbg is not None:
                dbg.append((label_id, n_cells, float(support),
                            np.nan, np.nan, np.nan, REJ_SUPPORT))
            continue

        # centroid: mean of the component's cells, in world metres
        cx = (gi + i_min + 0.5).mean() * cfg.voxel
        cy = (gj + j_min + 0.5).mean() * cfg.voxel
        # bounding radius: furthest cell from the centroid, plus half a voxel
        px = (gi + i_min + 0.5) * cfg.voxel
        py = (gj + j_min + 0.5) * cfg.voxel
        radius = float(np.hypot(px - cx, py - cy).max()) + cfg.voxel / 2.0

        if not (cfg.r_min <= radius <= cfg.r_max):               # filter (iii)
            if dbg is not None:
                dbg.append((label_id, n_cells, float(support),
                            cx, cy, radius, REJ_RADIUS))
            continue
        if dbg is not None:
            dbg.append((label_id, n_cells, float(support),
                        cx, cy, radius, ACCEPTED))
        dets.append((cx, cy, radius))

    if dbg is not None:
        # i_min/j_min and voxel are what map a raster index back to world
        # metres:  x = (row + i_min + 0.5) * voxel,  y = (col + j_min + 0.5) * voxel
        np.savez_compressed(
            debug_path,
            labels=labels.astype(np.int32), occ=occ, sup=sup,
            i_min=np.int64(i_min), j_min=np.int64(j_min),
            voxel=np.float64(cfg.voxel), sensor_xy=np.asarray(sensor_xy),
            comps=np.array(dbg, dtype=np.float64).reshape(-1, 7))

    return np.array(dets, dtype=np.float64) if dets else empty

UNVISITED, VISITED = 0, 1
STATUS_NAME = {UNVISITED: "UNVISITED", VISITED: "VISITED"}


class Tree:

    __slots__ = ("gid", "x", "y", "r", "status", "discovery_order",
                 "n_obs", "last_frame")

    def __init__(self, gid, x, y, r, discovery_order, frame_idx):
        self.gid = gid
        self.x, self.y, self.r = float(x), float(y), float(r)
        self.status = UNVISITED
        self.discovery_order = discovery_order
        self.n_obs = 1
        self.last_frame = frame_idx


class TreeRegistry:
  
    def __init__(self, d_match, cell, merge_floor):
        self.d_match = d_match
        self.cell = cell
        self.merge_floor = merge_floor   # tune this according to our ground truth, should be lesser than min distance between two trees
        self.trees = {}                   # global_id -> Tree
        self._hash = defaultdict(set)     # (ci, cj) -> {global_id}
        self._next_gid = 0
        self._next_discovery = 0

    # -- spatial hash: matching only looks at neighbouring cells --------- #
    def _key(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _neighbours(self, x, y):
        ci, cj = self._key(x, y)
        out = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out.extend(self._hash.get((ci + di, cj + dj), ()))
        return out

    def _index(self, t):
        self._hash[self._key(t.x, t.y)].add(t.gid)

    def _unindex(self, t):
        self._hash[self._key(t.x, t.y)].discard(t.gid)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _blend(xa, ya, ra, xb, yb, rb):
        wa, wb = ra * ra, rb * rb
        if wa + wb <= 0.0:                     
            return (xa + xb) / 2.0, (ya + yb) / 2.0, (ra + rb) / 2.0
        return ((wa * xa + wb * xb) / (wa + wb),
                (wa * ya + wb * yb) / (wa + wb),
                (ra + rb) / 2.0)

    # ------------------------------------------------------------------ #
    def update(self, dets, frame_idx):
        """dets: (N, 3) world-frame [x, y, r] from ONE scan.

        Returns one event per decision, for logging:
            ('match', gid, distance)      component folded into an entry
            ('new',   gid, (x, y, r))     component started a new identity
            ('merge', kept_gid, gone_gid) two entries collapsed in step 2
        """
        events = []
        gate = self.d_match

        # ---- step 1: sort every admissible pair, then take greedily ----
        pairs = []
        for i in range(len(dets)):
            x, y = dets[i][0], dets[i][1]
            for gid in self._neighbours(x, y):
                t = self.trees[gid]
                d = math.hypot(t.x - x, t.y - y)
                if d < gate:
                    pairs.append((d, i, gid))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))

        taken_det, taken_gid = set(), set()
        for d, i, gid in pairs:
            if i in taken_det or gid in taken_gid:
                continue                  # one component <-> one entry
            taken_det.add(i)
            taken_gid.add(gid)
            x, y, r = dets[i]
            t = self.trees[gid]
            self._unindex(t)
            t.x, t.y, t.r = float(x), float(y), float(r)   # line 6: replace
            t.n_obs += 1
            t.last_frame = frame_idx
            self._index(t)
            events.append(("match", gid, d))

        # ---- step 1b: unmatched components become new identities -------
        for i in range(len(dets)):
            if i in taken_det:
                continue
            x, y, r = dets[i]
            
            host = next((g for g in self._neighbours(x, y) if g in taken_gid
                         and math.hypot(self.trees[g].x - x, self.trees[g].y - y)
                         < max(self.trees[g].r + r, self.merge_floor)),
                        None)
            if host is not None:
                events.append(("absorb", host, i))
                continue
            t = Tree(self._next_gid, x, y, r, self._next_discovery, frame_idx)
            self._next_gid += 1
            self._next_discovery += 1
            self.trees[t.gid] = t
            self._index(t)
            events.append(("new", t.gid, (t.x, t.y, t.r)))

        # ---- step 2 ----------------------------------------------------
        events.extend(self._merge_overlaps())
        return events

    def _merge_overlaps(self):
        
        events = []
        for gi in sorted(self.trees):
            while True:
                ti = self.trees.get(gi)
                if ti is None:
                    break
                gone = None
                # recomputed each round: ti moves when it absorbs a partner
                for gj in sorted(self._neighbours(ti.x, ti.y)):
                    if gj <= gi:
                        continue
                    tj = self.trees.get(gj)
                    if tj is None:
                        continue
                    merge_dist = max(ti.r + tj.r, self.merge_floor)  
                    if math.hypot(ti.x - tj.x, ti.y - tj.y) < merge_dist:
                    # if math.hypot(ti.x - tj.x, ti.y - tj.y) < ti.r + tj.r:
                        gone = gj
                        break
                if gone is None:
                    break
                tj = self.trees[gone]
                self._unindex(ti)
                self._unindex(tj)
                ti.x, ti.y, ti.r = self._blend(ti.x, ti.y, ti.r,
                                               tj.x, tj.y, tj.r)
                ti.status = max(ti.status, tj.status)
                ti.discovery_order = min(ti.discovery_order, tj.discovery_order)
                ti.n_obs += tj.n_obs
                ti.last_frame = max(ti.last_frame, tj.last_frame)
                del self.trees[gone]
                self._index(ti)
                events.append(("merge", gi, gone))
        return events

    # ------------------------------------------------------------------ #
    def rows(self, min_obs=1):
        """The global list in discovery order, one row per tree:
        [global_id, x, y, r, status, discovery_order, n_obs]"""
        ts = sorted((t for t in self.trees.values() if t.n_obs >= min_obs),
                key=lambda t: t.discovery_order)
        return np.array([[t.gid, t.x, t.y, t.r, t.status,
                          t.discovery_order, t.n_obs] for t in ts],
                        dtype=np.float64).reshape(-1, 7)

    def __len__(self):
        return len(self.trees)


def read_xyzt(msg):
    names = {f.name for f in msg.fields}
    tname = "t" if "t" in names else ("time" if "time" in names else None)
    fields = ("x", "y", "z") + ((tname,) if tname else ())
    arr = pc2.read_points(msg, field_names=fields, skip_nans=True)
    if arr.dtype.names is None:                       # older tuple-list API
        arr = np.asarray(list(arr), dtype=np.float64)
        if arr.size == 0:
            return None, None
        return arr[:, :3], (arr[:, 3].astype(np.int64) if tname else None)
    if arr.shape[0] == 0:
        return None, None
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    return pts, (arr[tname].astype(np.int64) if tname else None)


def main():
    rclpy.init()
    node = SensorStreamReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_registry()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

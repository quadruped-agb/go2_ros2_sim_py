#!/usr/bin/env python3
"""
Frontier + tree-inspection exploration node (terminating version, v4).

What changed vs. v3 (everything not listed here is unchanged)
--------------------------------------------------------------
TREE VIEWING-DISTANCE WINDOW  (new, all distances are TRUNK SURFACE -> robot centre)
  * Every tree viewpoint, merged or not, must satisfy
        TREE_DIST_MIN_M <= (centre distance - tree radius r) <= TREE_DIST_MAX_M
    Defaults 0.7 m / 1.8 m (derivation in the constants block). Both are ROS
    parameters (tree_dist_min_m / tree_dist_max_m) so you can overwrite them with
    your own point-cloud-density numbers without touching the code.
  * The old viewpoint lattice (radius steps 0/+0.4/-0.25/+0.8 m, angle steps
    0/+-20/+-40 deg) already lived inside that window, so un-merged viewpoints
    come out exactly where they did before.

VIEWPOINT MERGING  (new)
  * Pending viewpoints of DIFFERENT trees that lie within merge_radius_m (1.5 m)
    of each other are merged into ONE viewpoint if a single point exists that
    satisfies, for every tree it serves: the distance window, the slot's azimuth
    tolerance (+-40 deg, the largest offset the old lattice already allowed),
    obstacle clearance (0.30 m, 0.22 m fallback), the plot boundary, and a clear
    line of sight to the trunk. Among feasible points the one with the best
    WORST-CASE point density is chosen (see joint_viewpoint).
  * A tree never gets two slots merged into the same point, so it still keeps its
    3 azimuth slots; merging only lets several trees share one trip.
  * If a merged viewpoint fails in Nav2, its members are un-merged, the position
    is blacklisted for them, and they continue individually (no merge retry).

FACING  (changed: per tree, not "first viewpoint of a tree")
  * A tree must be CAUGHT IN THE CAMERA at least once ('faced'). A tree whose
    viewpoints are all used up but that was never faced is NOT done: the node
    adds an extra viewpoint slot (max 2) to try again.
  * At every tree-viewpoint arrival the robot first checks whether its current
    heading already catches a tree, then spins to face every still-unfaced tree
    that is in range with a clear line of sight (members of the merged viewpoint
    and any other tree in range). Trees whose bearings differ by < 2 x camera
    tolerance are caught with ONE spin. Success is decided from the real yaw read
    back from TF, not from the Spin action's return code.
  * Each tree is attempted at most MAX_FACE_TRIES (3) times in total and at most
    once per arrival, so this cannot loop.

Unchanged: frontier logic, watchdog / stall / escape logic, run limits, report.
"""

import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose, Spin
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from std_msgs.msg import Float64MultiArray


FREE = 0
UNKNOWN = -1
OCC_THRESHOLD = 50

# ---------------- frontier detection / scoring ----------------
MIN_FRONTIER_CELLS = 12
UNKNOWN_SPECK_M = 0.25
GOAL_STANDOFF_M = 0.6
MIN_OBSTACLE_CLEARANCE_M = 0.30     # was 0.45 (a 0.6 m square in practice)
TREE_FALLBACK_CLEARANCE_M = 0.22    # second pass for tree viewpoints
REPLAN_ON_FAILURE_BLACKLIST_M = 0.75
BLACKLIST_TTL_S = 90.0

SENSOR_RANGE_M = 6.0
NUM_RAYS = 72
TOP_K_CANDIDATES = 5
UTILITY_DISTANCE_ALPHA = 2.5
MAX_CANDIDATE_RADIUS_M = 8.0
MAX_CANDIDATE_RADIUS_LIMIT_M = 60.0
RADIUS_EXPAND_FACTOR = 1.5
MAX_FRONTIER_BONUS_M = 2.5

MIN_INFO_GAIN = 40
FRONTIER_DONE_RADIUS_M = 2.0
FRONTIER_VISITED_RADIUS_M = 1.5     # robot has been this close -> region is done
MAX_FRONTIER_FAILS = 2
COVERAGE_DONE_PCT = 95.0

# ---------------- tree inspection ----------------
CENTROID_BOUND_MARGIN_M = 0.0       # tree centroids outside the bounds are ignored (was 1.0)
GOAL_BOUND_MARGIN_M = 0.0           # every goal must be inside the bounds (was 0.5)
MIN_TREE_OBS = 10
TREE_INSPECTION_RADIUS_M = 1.0      # nominal standoff of the viewpoint lattice
TREE_MATCH_RADIUS_M = 1.25
MAX_NEW_TREES_PER_CYCLE = 4
TREE_BATCH = 3
TREE_MIN_VISITED = 3                # viewpoints that must be visited per tree

VP_RADIUS_STEPS_M = (0.0, 0.4, -0.25, 0.8)       # offsets from the nominal standoff
VP_ANGLE_STEPS_DEG = (0, 20, -20, 40, -40)       # offsets from the slot's base angle
MAX_SLOT_ATTEMPTS = 4               # distinct positions tried before a slot is given up
TREE_GIVEUP_FAILS = 6               # a tree with this many failed goals and NO visit is given up entirely
SLOT_COOLDOWN_S = 25.0              # wait before retrying a slot elsewhere
SLOT_DRY_LIMIT_S = 120.0            # no feasible position for this long (map covers it) -> give up
SYSTEMIC_DISTINCT = 4               # failures at this many DIFFERENT places with no success in between:
                                    # print a loud 'nav2 is the problem' error (counting continues)
TREE_COOLDOWN_S = 40.0              # after a failure, move on to other trees for this long

# ---- viewing-distance window (trunk SURFACE to robot centre) ----------------------
# Point density on a surface facing the sensor falls with the inverse square of range:
#     rho(d) = (point_rate / fov_solid_angle) * T / d^2      [points / m^2]
# so the density relative to the reference standoff is  (D_REF / d)^2.
#   D_REF = 1.0 m : the standoff the run has been using (TREE_INSPECTION_RADIUS_M)
#   D_MAX         : density floor of 30 % of the reference  ->  D_REF / sqrt(0.30) = 1.83 m  -> 1.8 m
#                   (1.8 m is also the farthest point the old lattice ever used)
#   D_MIN         : robot circumscribed radius sqrt(0.35^2 + 0.175^2) = 0.39 m
#                   + MIN_OBSTACLE_CLEARANCE_M 0.30 m = 0.69 m  -> 0.7 m
#                   (so the robot can still turn in place next to the trunk)
# Replace with your own numbers: D_MAX = sqrt(rate * T / (solid_angle * rho_required)).
TREE_DIST_REF_M = TREE_INSPECTION_RADIUS_M
TREE_DIST_MIN_M = 0.7
TREE_DIST_MAX_M = 1.8

# ---- viewpoint merging -------------------------------------------------------------
# Two viewpoints that were each ~D_REF from their own tree and are R apart can be
# replaced by their midpoint, which is at most R/2 farther from each tree:
#     D_REF + R/2 <= D_MAX   ->   R <= 2 * (D_MAX - D_REF) = 1.6 m   -> 1.5 m (margin)
# Candidate grouping only; the real acceptance test is joint_viewpoint().
MERGE_RADIUS_M = 1.5
MERGE_AZ_TOL_DEG = 40.0             # slots are 120 deg apart; +-40 keeps two views of one tree >= 40 deg apart
MERGE_GRID_M = 0.1                  # lattice spacing of the joint-viewpoint search (<= half a 0.2 m cell)
MERGE_MAX_TREES = 4                 # trees served by one merged viewpoint
MERGE_MAX_EVAL = 200                # best-scored lattice points checked against the map per clearance pass

# ---- facing (camera) -------------------------------------------------------------
CAMERA_HFOV_DEG = 88.0              # ASSUMPTION: conservative camera horizontal field of view
CAMERA_FOV_USE = 0.6                # use 60 % of the half-FOV so the trunk is not at the image edge
FACE_RANGE_SLACK_M = 0.25           # robot may stop this far (nav2 xy tolerance) from the viewpoint
MAX_FACE_TRIES = 3                  # spin attempts per tree over the whole run
MAX_EXTRA_FACE_SLOTS = 2            # extra viewpoint slots a never-faced tree may get
SPIN_CLEARANCE_M = 0.40             # ~ circumscribed radius 0.39 m: room to turn in place

# ---------------- run limits ----------------
GOAL_TIMEOUT_S = 90.0
MAX_FRONTIER_GOALS = 40
MAX_RUNTIME_S = 3600.0
COMPLETE_HOLD_S = 20.0

# ---------------- getting unstuck ----------------
STALL_TIMEOUT_S = 15.0              # not moved this long while navigating -> cancel the goal
STALL_MOVE_M = 0.25                 # "moved" means further than this from where the timer started
ESCAPE_SPEED_MPS = 0.12             # gentle back-off speed
ESCAPE_TIME_S = 2.5                 # ~0.3 m
ESCAPE_TRIGGER_DIST_M = 0.75        # only back off if a mapped obstacle is this close

# (x, y, bonus, yaw)
DUMMY_TARGETS = []


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.declare_parameter('map_topic', '/projected_map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('min_frontier_cells', MIN_FRONTIER_CELLS)
        self.declare_parameter('bounds_min_x', -5.0)
        self.declare_parameter('bounds_max_x', 5.0)
        self.declare_parameter('bounds_min_y', -5.0)
        self.declare_parameter('bounds_max_y', 5.0)
        self.declare_parameter('sensor_range_m', SENSOR_RANGE_M)
        self.declare_parameter('top_k_candidates', TOP_K_CANDIDATES)
        self.declare_parameter('coverage_done_pct', COVERAGE_DONE_PCT)
        self.declare_parameter('tree_batch', TREE_BATCH)
        self.declare_parameter('tree_min_visited', TREE_MIN_VISITED)
        self.declare_parameter('max_frontier_goals', MAX_FRONTIER_GOALS)
        self.declare_parameter('max_runtime_s', MAX_RUNTIME_S)
        self.declare_parameter('goal_timeout_s', GOAL_TIMEOUT_S)
        self.declare_parameter('complete_hold_s', COMPLETE_HOLD_S)
        self.declare_parameter('shutdown_on_complete', True)
        self.declare_parameter('tree_bound_margin', CENTROID_BOUND_MARGIN_M)
        self.declare_parameter('goal_bound_margin', GOAL_BOUND_MARGIN_M)
        self.declare_parameter('min_tree_obs', MIN_TREE_OBS)
        self.declare_parameter('stall_timeout_s', STALL_TIMEOUT_S)
        self.declare_parameter('unstick_enabled', True)
        # --- new in v4 ---
        self.declare_parameter('tree_dist_min_m', TREE_DIST_MIN_M)
        self.declare_parameter('tree_dist_max_m', TREE_DIST_MAX_M)
        self.declare_parameter('merge_viewpoints', True)
        self.declare_parameter('merge_radius_m', MERGE_RADIUS_M)
        self.declare_parameter('camera_hfov_deg', CAMERA_HFOV_DEG)

        self.map_topic = self.get_parameter('map_topic').value
        self.base_frame = self.get_parameter('robot_base_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.min_frontier_cells = self.get_parameter('min_frontier_cells').value
        self.bounds_min_x = self.get_parameter('bounds_min_x').value
        self.bounds_max_x = self.get_parameter('bounds_max_x').value
        self.bounds_min_y = self.get_parameter('bounds_min_y').value
        self.bounds_max_y = self.get_parameter('bounds_max_y').value
        self.sensor_range_m = self.get_parameter('sensor_range_m').value
        self.top_k_candidates = self.get_parameter('top_k_candidates').value
        self.coverage_done_pct = self.get_parameter('coverage_done_pct').value
        self.tree_batch = self.get_parameter('tree_batch').value
        self.tree_min_visited = self.get_parameter('tree_min_visited').value
        self.max_frontier_goals = self.get_parameter('max_frontier_goals').value
        self.max_runtime_s = self.get_parameter('max_runtime_s').value
        self.goal_timeout_s = self.get_parameter('goal_timeout_s').value
        self.complete_hold_s = self.get_parameter('complete_hold_s').value
        self.shutdown_on_complete = self.get_parameter('shutdown_on_complete').value
        self.tree_bound_margin = self.get_parameter('tree_bound_margin').value
        self.goal_bound_margin = self.get_parameter('goal_bound_margin').value
        self.min_tree_obs = self.get_parameter('min_tree_obs').value
        self.stall_timeout_s = self.get_parameter('stall_timeout_s').value
        self.unstick_enabled = self.get_parameter('unstick_enabled').value
        self.tree_dist_min = float(self.get_parameter('tree_dist_min_m').value)
        self.tree_dist_max = float(self.get_parameter('tree_dist_max_m').value)
        if self.tree_dist_min <= 0.0 or self.tree_dist_min >= self.tree_dist_max:
            self.get_logger().warn(
                f'invalid tree distance window [{self.tree_dist_min}, {self.tree_dist_max}] '
                f'-- using defaults [{TREE_DIST_MIN_M}, {TREE_DIST_MAX_M}]')
            self.tree_dist_min, self.tree_dist_max = TREE_DIST_MIN_M, TREE_DIST_MAX_M
        self.merge_enabled = bool(self.get_parameter('merge_viewpoints').value)
        self.merge_radius = float(self.get_parameter('merge_radius_m').value)
        self.face_tol = math.radians(
            0.5 * float(self.get_parameter('camera_hfov_deg').value) * CAMERA_FOV_USE)

        # live tree registry: rows of [gid, x, y, r, status, disc, n_obs]
        tree_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                              reliability=QoSReliabilityPolicy.RELIABLE)
        self.tree_sub = self.create_subscription(Float64MultiArray, '/trees/registry',
                                                 self.trees_cb, tree_qos)
        self.tree_rows = np.zeros((0, 7))

        # persistent tree records (see module docstring)
        self.trees = []
        self.next_tree_id = 0

        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_cb, qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')

        self.latest_map = None
        self.exploring = False
        self.failed_goals = []              # (x, y, t) time-limited blacklist
        self.candidate_radius = MAX_CANDIDATE_RADIUS_M
        self.last_n_too_far = 0
        self.last_n_blacklisted = 0
        self.last_cluster_count = 0
        self.visited_frontier_count = 0
        self.done_frontier_centroids = []   # permanently retired frontier regions
        self.frontier_fails = []            # [cx, cy, count]
        self.goal_centroids = {}            # (round(gx,2), round(gy,2)) -> (cx, cy)
        self.fail_counts = {}               # only used for external dummy targets
        self.goal_queue = []
        self.stall_count = 0
        self.declared_complete = False
        self.stall_limit = 5
        self.dummy_targets = list(DUMMY_TARGETS)
        self.current_goal_handle = None

        # completion / safety bookkeeping
        self.visit_pts = []                 # places the robot has actually been
        self.frontier_goals_sent = 0
        self.goal_sent_t = None
        self.watchdog_fired = False
        self.budget_hit = False
        self.frontier_cap_logged = False
        self.complete_since = None
        self.finished = False
        self.fail_keys = set()
        self.face_ctx = None                # {'trees': ids to re-check, 'tried': ids spun at this arrival}
        self.merge_fail_cache = set()       # joint viewpoints already proven infeasible
        self.stall_ref = None               # (x, y, t) where the 'not moving' timer started
        self.escaping = False
        self.escape_until = 0.0
        self.escape_dir = -1.0

        self.timer = self.create_timer(2.0, self.explore_step)
        self.start_time = time.monotonic()
        self.complete_elapsed = None
        self.n_centroids_outside = 0

        self.path_pub = self.create_publisher(Path, 'frontier_explorer/robot_path', 10)
        self.trail_path = Path()
        self.trail_path.header.frame_id = self.global_frame
        self.trail_max_poses = 20000
        self.trail_timer = self.create_timer(0.5, self.record_trail)

        self.marker_pub = self.create_publisher(MarkerArray, 'frontier_explorer/markers', 10)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)   # same topic nav2 drives
        self.escape_timer = self.create_timer(0.1, self.escape_tick)
        self.last_candidates = []

        self.get_logger().info(
            f'Frontier explorer up. Listening on {self.map_topic}. '
            f'Bounds x[{self.bounds_min_x},{self.bounds_max_x}] '
            f'y[{self.bounds_min_y},{self.bounds_max_y}]  '
            f'(tree_min_visited={self.tree_min_visited}, '
            f'max_frontier_goals={self.max_frontier_goals}, '
            f'max_runtime_s={self.max_runtime_s:.0f}, goal_timeout_s={self.goal_timeout_s:.0f})')
        self.get_logger().info(
            f'Tree viewing window {self.tree_dist_min:.2f}-{self.tree_dist_max:.2f} m from trunk '
            f'surface; merge={"on" if self.merge_enabled else "off"} (radius '
            f'{self.merge_radius:.2f} m); camera catch tolerance '
            f'+-{math.degrees(self.face_tol):.0f} deg')

    # ------------------------------------------------------------------
    # basic callbacks / helpers
    # ------------------------------------------------------------------
    def map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    def get_robot_pose_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'tf lookup {self.global_frame}->{self.base_frame} failed: {e}')
            return None

    def get_robot_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.global_frame, self.base_frame, rclpy.time.Time())
            q = t.transform.rotation
            return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        except Exception as e:
            self.get_logger().warn(f'tf lookup for yaw failed: {e}')
            return None

    @staticmethod
    def _wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def trees_cb(self, msg):
        rows = np.array(msg.data, dtype=np.float64).reshape(-1, 7)
        if len(rows):
            m = self.tree_bound_margin
            ok = ((rows[:, 1] >= self.bounds_min_x - m) & (rows[:, 1] <= self.bounds_max_x + m) &
                  (rows[:, 2] >= self.bounds_min_y - m) & (rows[:, 2] <= self.bounds_max_y + m))
            self.n_centroids_outside = int((~ok).sum())
            rows = rows[ok]
        self.tree_rows = rows

    def bounds_mask(self, grid_msg):
        """Boolean HxW mask of cells whose centre lies inside the configured bounds."""
        res = grid_msg.info.resolution
        ox, oy = grid_msg.info.origin.position.x, grid_msg.info.origin.position.y
        h, w = grid_msg.info.height, grid_msg.info.width
        xs = ox + (np.arange(w) + 0.5) * res
        ys = oy + (np.arange(h) + 0.5) * res
        in_x = (xs >= self.bounds_min_x) & (xs <= self.bounds_max_x)
        in_y = (ys >= self.bounds_min_y) & (ys <= self.bounds_max_y)
        return in_y[:, None] & in_x[None, :]

    @staticmethod
    def _erode(m, r):
        h, w = m.shape
        p = np.pad(m, r, mode='constant', constant_values=True)
        out = np.ones_like(m)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                out &= p[r + dy:r + dy + h, r + dx:r + dx + w]
        return out

    @staticmethod
    def _dilate(m, r):
        h, w = m.shape
        p = np.pad(m, r, mode='constant', constant_values=False)
        out = np.zeros_like(m)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                out |= p[r + dy:r + dy + h, r + dx:r + dx + w]
        return out

    def near_visited(self, x, y, radius):
        r2 = radius * radius
        return any((x - px) ** 2 + (y - py) ** 2 < r2 for px, py in self.visit_pts)

    # ------------------------------------------------------------------
    # run limits / watchdog
    # ------------------------------------------------------------------
    def check_budget(self):
        if self.budget_hit:
            return
        if time.monotonic() - self.start_time < self.max_runtime_s:
            return
        self.budget_hit = True
        n = 0
        for t in self.trees:
            for s in t['slots']:
                if s['state'] == 'pending':
                    s['state'] = 'failed'
                    s['cur'] = None
                    n += 1
            self.update_tree_done(t)
        self.goal_queue.clear()
        self.get_logger().warn(
            f'Run-time budget ({self.max_runtime_s:.0f} s) reached -- giving up on '
            f'{n} pending tree viewpoint(s) and stopping the frontier search.')

    def check_goal_watchdog(self):
        if self.goal_sent_t is None:
            return
        now = time.monotonic()

        # (a) stuck: the robot has not moved for stall_timeout_s while navigating
        if (self.unstick_enabled and self.current_goal_handle is not None
                and not self.watchdog_fired):
            xy = self.get_robot_pose_xy()
            if xy is not None:
                if (self.stall_ref is None or
                        math.hypot(xy[0] - self.stall_ref[0], xy[1] - self.stall_ref[1]) > STALL_MOVE_M):
                    self.stall_ref = (xy[0], xy[1], now)
                elif now - self.stall_ref[2] >= self.stall_timeout_s:
                    self.watchdog_fired = True
                    self.get_logger().warn(
                        f'Robot has not moved for {now - self.stall_ref[2]:.0f} s while navigating '
                        f'(stuck?) -- cancelling the goal')
                    self.current_goal_handle.cancel_goal_async()
                    return

        # (b) overall time limit for one goal
        if now - self.goal_sent_t <= self.goal_timeout_s:
            return
        if self.current_goal_handle is not None:
            if not self.watchdog_fired:
                self.watchdog_fired = True
                self.get_logger().warn(
                    f'Goal running longer than {self.goal_timeout_s:.0f} s -- cancelling it')
                self.current_goal_handle.cancel_goal_async()
        else:
            # never got an answer from nav2
            self.get_logger().warn('No response to goal request -- resetting')
            self.goal_sent_t = None
            self.exploring = False

    # ---- backing away from a tree the robot is pressed against ----
    def nearest_obstacle(self):
        """(bearing relative to the robot's heading, distance) of the nearest mapped obstacle."""
        msg = self.latest_map
        xy = self.get_robot_pose_xy()
        yaw = self.get_robot_yaw()
        if msg is None or xy is None or yaw is None:
            return None
        h, w = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        rr, cc = np.nonzero(data >= OCC_THRESHOLD)
        if rr.size == 0:
            return None
        res = msg.info.resolution
        x = msg.info.origin.position.x + (cc + 0.5) * res - xy[0]
        y = msg.info.origin.position.y + (rr + 0.5) * res - xy[1]
        d = np.hypot(x, y)
        i = int(np.argmin(d))
        bearing = math.atan2(float(y[i]), float(x[i])) - yaw
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return bearing, float(d[i])

    def start_escape(self):
        if not self.unstick_enabled:
            return
        found = self.nearest_obstacle()
        if found is None or found[1] > ESCAPE_TRIGGER_DIST_M:
            return
        bearing, dist = found
        self.escape_dir = -1.0 if abs(bearing) < math.radians(100.0) else 1.0
        self.escape_until = time.monotonic() + ESCAPE_TIME_S
        self.escaping = True
        self.get_logger().info(
            f'Obstacle {dist:.2f} m away at {math.degrees(bearing):.0f} deg -- '
            f"backing {'away' if self.escape_dir < 0 else 'forward'} a little before the next goal")

    def escape_tick(self):
        if not self.escaping:
            return
        cmd = Twist()
        if time.monotonic() < self.escape_until:
            cmd.linear.x = self.escape_dir * ESCAPE_SPEED_MPS
        else:
            self.escaping = False            # publishes one final zero command
        self.cmd_pub.publish(cmd)

    def frontier_phase_enabled(self, coverage_reached):
        if self.budget_hit or coverage_reached:
            return False
        if self.frontier_goals_sent >= self.max_frontier_goals:
            if not self.frontier_cap_logged:
                self.frontier_cap_logged = True
                self.get_logger().info(
                    f'Frontier goal cap ({self.max_frontier_goals}) reached -- '
                    f'no more frontier sweeps.')
            return False
        return True

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def explore_step(self):
        grid_msg = self.latest_map
        if grid_msg is None:
            self.get_logger().info('waiting for map...', throttle_duration_sec=5.0)
            return
        h, w = grid_msg.info.height, grid_msg.info.width
        data = np.array(grid_msg.data, dtype=np.int8).reshape(h, w)

        self.check_budget()
        # detection + slot upkeep run every cycle, even while a goal is in flight
        self.scan_for_new_trees(data, grid_msg)

        if self.escaping:
            return

        if self.exploring:
            self.check_goal_watchdog()
            return

        # commit to the current sweep / tour before planning anything new
        if self.goal_queue:
            self.send_nav_goal(self.goal_queue.pop(0))
            return

        robot_xy = self.get_robot_pose_xy()
        if robot_xy is None:
            return

        # share viewpoints between nearby trees. Safe here: no goal is queued or in
        # flight, so no queued goal can refer to a position that is about to change.
        if self.merge_enabled:
            self.merge_viewpoints(data, grid_msg)

        n_pending = self.count_pending_trees()
        n_ready = self.count_ready_trees()

        # 1) enough trees ready -> inspect them as one TSP tour
        if n_ready >= self.tree_batch and self.start_tree_tour(robot_xy):
            return

        # 2) frontier sweep (bounded)
        cov = self.coverage_in_plot()
        coverage_reached = cov is not None and cov >= self.coverage_done_pct
        if self.frontier_phase_enabled(coverage_reached):
            goals = self.plan_frontier_sweep(data, grid_msg, robot_xy)
            if goals:
                self.goal_queue = goals
                self.stall_count = 0
                self.declared_complete = False
                self.complete_since = None
                self.publish_planning_markers(robot_xy)
                return
        else:
            self.last_cluster_count = 0
            self.last_n_blacklisted = 0
            self.last_candidates = []

        # 3) finish whatever trees are ready
        if n_ready > 0 and self.start_tree_tour(robot_xy):
            return

        # 3b) trees still pending but cooling down / waiting for a feasible
        #     position: wait (all of these have a hard end, so this terminates)
        if n_pending > 0:
            self.complete_since = None
            self.get_logger().info(
                f'Waiting on {n_pending} tree(s): {self.pending_summary()}',
                throttle_duration_sec=10.0)
            self.publish_planning_markers(robot_xy)
            return

        # 4) externally injected targets
        if self.dummy_targets:
            ordered = self.solve_viewpoint_order([], robot_xy)
            if ordered:
                self.goal_queue = [(x, y, yaw) for (x, y, _b, yaw) in ordered]
                self.declared_complete = False
                self.complete_since = None
                self.get_logger().info(f'Routing to {len(ordered)} dummy target(s).')
                self.publish_planning_markers(robot_xy)
                return

        # 5) wait briefly if the only remaining frontiers are temporarily blacklisted
        if self.last_n_blacklisted > 0 and self.stall_count < self.stall_limit:
            self.stall_count += 1
            self.publish_planning_markers(robot_xy)
            return

        # 6) nothing left to do: require it to stay that way for a moment
        now = time.monotonic()
        if self.complete_since is None:
            self.complete_since = now
            return
        if now - self.complete_since < self.complete_hold_s:
            return

        if self.budget_hit:
            reason = 'run-time budget reached'
        elif coverage_reached:
            reason = f'plot coverage {cov:.1f}% >= {self.coverage_done_pct:.0f}%'
        elif self.frontier_goals_sent >= self.max_frontier_goals:
            reason = 'frontier goal cap reached'
        else:
            reason = 'no selectable frontiers remain'
        self.announce_complete(data, reason)
        self.publish_planning_markers(robot_xy)

    def start_tree_tour(self, robot_xy):
        tour = self.plan_tree_tour(robot_xy)
        if not tour:
            return False
        self.goal_queue = tour
        self.declared_complete = False
        self.complete_since = None
        self.get_logger().info(
            f'Planned tree tour: {len(tour)} viewpoints over '
            f'{self.count_ready_trees()} tree(s).')
        self.publish_planning_markers(robot_xy)
        return True

    def plan_frontier_sweep(self, data, grid_msg, robot_xy):
        clusters = self.detect_frontier_clusters(grid_msg, data)
        self.last_cluster_count = len(clusters)
        if not clusters:
            self.last_candidates = []
            self.last_n_blacklisted = 0
            return []

        candidates = self.score_candidates(clusters, data, grid_msg, robot_xy)
        while (not candidates and self.last_n_too_far > 0
               and self.candidate_radius < MAX_CANDIDATE_RADIUS_LIMIT_M):
            self.candidate_radius = min(self.candidate_radius * RADIUS_EXPAND_FACTOR,
                                        MAX_CANDIDATE_RADIUS_LIMIT_M)
            self.get_logger().info(
                f'No local frontier selectable -- widening search to '
                f'{self.candidate_radius:.1f} m')
            candidates = self.score_candidates(clusters, data, grid_msg, robot_xy)
        self.candidate_radius = MAX_CANDIDATE_RADIUS_M

        self.last_candidates = candidates
        if not candidates:
            return []

        top = sorted(candidates, key=lambda c: c['utility'], reverse=True)[:self.top_k_candidates]
        max_gain = max(c['gain'] for c in top) or 1
        for c in top:
            c['tsp_bonus'] = (c['gain'] / max_gain) * MAX_FRONTIER_BONUS_M

        ordered = self.solve_viewpoint_order(top, robot_xy)
        goals = [(x, y, yaw) for (x, y, _b, yaw) in ordered]
        self.get_logger().info(
            f'Planned a {len(goals)}-viewpoint sweep from {len(candidates)} candidates '
            f'(of {len(clusters)} frontier clusters, '
            f'{len(self.done_frontier_centroids)} regions retired, '
            f'{self.frontier_goals_sent}/{self.max_frontier_goals} frontier goals used).')
        return goals

    # ------------------------------------------------------------------
    # TSP helpers
    # ------------------------------------------------------------------
    def tsp_order(self, pts, start):
        """Order of pts (list of (x,y)) as an open path from start."""
        n = len(pts) + 1
        nodes = [start] + list(pts)
        m = [[int(round(math.hypot(a[0] - b[0], a[1] - b[1]) * 1000)) for b in nodes] for a in nodes]
        for i in range(n):
            m[i][0] = 0
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
        rt = pywrapcp.RoutingModel(mgr)
        cb = rt.RegisterTransitCallback(lambda i, j: m[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
        rt.SetArcCostEvaluatorOfAllVehicles(cb)
        p = pywrapcp.DefaultRoutingSearchParameters()
        p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        p.time_limit.FromMilliseconds(500)
        sol = rt.SolveWithParameters(p)
        if sol is None:
            return list(range(len(pts)))
        order, idx = [], rt.Start(0)
        while not rt.IsEnd(idx):
            k = mgr.IndexToNode(idx)
            if k:
                order.append(k - 1)
            idx = sol.Value(rt.NextVar(idx))
        return order

    def build_cost_matrix(self, nodes):
        n = len(nodes)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            xi, yi, _, _ = nodes[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj, bonus_j, _ = nodes[j]
                cost = math.hypot(xj - xi, yj - yi) - bonus_j
                matrix[i][j] = max(0, int(round(cost * 1000)))
        return matrix

    def solve_viewpoint_order(self, candidates, robot_xy):
        nodes = [(robot_xy[0], robot_xy[1], 0.0, 0.0)]
        for c in candidates:
            nodes.append((c['x'], c['y'], c.get('tsp_bonus', 0.0), c['yaw']))
        for (tx, ty, tbonus, tyaw) in self.dummy_targets:
            if self.is_blacklisted(tx, ty):
                continue
            nodes.append((tx, ty, tbonus, tyaw))

        n = len(nodes)
        if n <= 1:
            return []
        if n == 2:
            return [nodes[1]]

        matrix = self.build_cost_matrix(nodes)
        for i in range(n):
            matrix[i][0] = 0   # open path

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_params.time_limit.FromSeconds(1)

        solution = routing.SolveWithParameters(search_params)
        if solution is None:
            self.get_logger().warn('OR-Tools TSP solve failed, falling back to input order')
            return list(nodes[1:])

        order = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                order.append(nodes[node])
            index = solution.Value(routing.NextVar(index))
        return order

    # ------------------------------------------------------------------
    # trees: persistent records with retryable viewpoint slots
    # ------------------------------------------------------------------
    def tree_near(self, x, y, radius=TREE_MATCH_RADIUS_M):
        for t in self.trees:
            if math.hypot(x - t['x'], y - t['y']) < radius:
                return t
        return None

    @staticmethod
    def new_slot(base_angle, extra=False):
        return {'angle': base_angle, 'state': 'pending',   # pending|visited|failed|skipped
                'cur': None,        # (x, y, yaw) currently assigned position
                'bad': [],          # positions that failed in nav2
                'attempts': 0, 'cool': 0.0, 'dry_t0': None,
                'merged': False,    # cur is shared with a slot of another tree
                'no_merge': False,  # a merged viewpoint failed: stay individual
                'extra': extra}     # added because the tree was never faced

    @staticmethod
    def slot_index(tree, slot):
        for k, s in enumerate(tree['slots']):
            if s is slot:
                return k
        return -1

    def count_pending_trees(self):
        return sum(1 for t in self.trees
                   if not t['done'] and any(s['state'] == 'pending' for s in t['slots']))

    def pending_summary(self):
        now = time.monotonic()
        parts = []
        for t in self.trees:
            if t['done']:
                continue
            why = []
            for s in t['slots']:
                if s['state'] != 'pending':
                    continue
                wait = max(t['cool'], s['cool']) - now
                if s['cur'] is None and wait <= 0:
                    why.append('no free position'
                               + (f" for {now - s['dry_t0']:.0f}s" if s['dry_t0'] else ''))
                elif wait > 0:
                    why.append(f"cooldown {wait:.0f}s after {s['attempts']} failed")
                else:
                    why.append('ready')
            parts.append(f"#{t['id']}({t['x']:.1f},{t['y']:.1f}) [{'; '.join(why)}]")
        return ' | '.join(parts)

    def count_ready_trees(self):
        now = time.monotonic()
        n = 0
        for t in self.trees:
            if t['done'] or now < t['cool']:
                continue
            if any(s['state'] == 'pending' and s['cur'] is not None and now >= s['cool']
                   for s in t['slots']):
                n += 1
        return n

    def find_tree_vp(self, gx, gy):
        """First slot (any state) whose position is (gx, gy)."""
        for t in self.trees:
            for s in t['slots']:
                c = s['cur']
                if c is not None and abs(c[0] - gx) < 0.02 and abs(c[1] - gy) < 0.02:
                    return t, s
        return None

    def find_tree_vps(self, gx, gy):
        """ALL pending slots at (gx, gy): more than one when the viewpoint is merged."""
        out = []
        for t in self.trees:
            for s in t['slots']:
                c = s['cur']
                if (s['state'] == 'pending' and c is not None
                        and abs(c[0] - gx) < 0.02 and abs(c[1] - gy) < 0.02):
                    out.append((t, s))
        return out

    def face_exhausted(self, tree):
        return tree['faced'] or tree['face_tries'] >= MAX_FACE_TRIES

    def add_face_slot(self, tree):
        """The tree has no pending viewpoint left but was never caught in the camera:
        add one more slot (max MAX_EXTRA_FACE_SLOTS), first on the side the robot is on."""
        if self.budget_hit or tree['gave_up'] or tree['extra_slots'] >= MAX_EXTRA_FACE_SLOTS:
            return False
        angles = []
        xy = self.get_robot_pose_xy()
        if xy is not None:
            angles.append(math.atan2(xy[1] - tree['y'], xy[0] - tree['x']))
        angles += [s['angle'] for s in tree['slots'] if not s['extra']]
        for a in angles:
            if self.slot_possible(tree, a):
                tree['slots'].append(self.new_slot(a, extra=True))
                tree['extra_slots'] += 1
                self.get_logger().info(
                    f"Tree #{tree['id']} has not been caught in the camera yet -- adding "
                    f"extra viewpoint slot {tree['extra_slots']}/{MAX_EXTRA_FACE_SLOTS} "
                    f"at {math.degrees(a):.0f} deg")
                return True
        return False

    def update_tree_done(self, tree):
        if tree['done']:
            return
        n_ok = sum(1 for s in tree['slots'] if s['state'] == 'visited')
        face_ok = self.face_exhausted(tree)
        # enough viewpoints visited: drop the rest, but ONLY once the tree has been
        # caught in the camera (or its spin attempts are used up)
        if n_ok >= self.tree_min_visited and face_ok:
            for s in tree['slots']:
                if s['state'] == 'pending':
                    s['state'] = 'skipped'
                    s['cur'] = None
                    s['merged'] = False
        if any(s['state'] == 'pending' for s in tree['slots']):
            return
        if not face_ok and self.add_face_slot(tree):
            return
        tree['done'] = True
        n_bad = sum(1 for s in tree['slots'] if s['state'] == 'failed')
        extra = f', {n_bad} UNREACHABLE' if n_bad else ''
        seen = 'caught in camera' if tree['faced'] else 'NOT CAUGHT IN CAMERA'
        self.get_logger().info(
            f"Tree #{tree['id']} at ({tree['x']:.1f},{tree['y']:.1f}) DONE "
            f"({n_ok} viewpoint(s) visited{extra}; {seen}). "
            f"{sum(1 for t in self.trees if t['done'])}/{len(self.trees)} trees done.")

    # ---- viewpoint geometry --------------------------------------------------
    def slot_radii(self, tree):
        """Centre distances of the old lattice that fall inside the viewing window."""
        standoff = min(tree['r'], 0.6) + TREE_INSPECTION_RADIUS_M
        out = []
        for dr in VP_RADIUS_STEPS_M:
            rad = standoff + dr
            ds = rad - tree['r']                       # trunk surface -> robot centre
            if self.tree_dist_min - 1e-6 <= ds <= self.tree_dist_max + 1e-6:
                out.append(rad)
        return out

    def slot_candidates(self, data, grid_msg, tree, base_angle, clearances=None):
        """All feasible positions for one slot, best first."""
        if clearances is None:
            clearances = (MIN_OBSTACLE_CLEARANCE_M, TREE_FALLBACK_CLEARANCE_M)
        out = []
        for clearance in clearances:
            for rad in self.slot_radii(tree):
                for da in VP_ANGLE_STEPS_DEG:
                    ang = base_angle + math.radians(da)
                    gx = tree['x'] + rad * math.cos(ang)
                    gy = tree['y'] + rad * math.sin(ang)
                    if not self.in_bounds_margin(gx, gy, self.goal_bound_margin):
                        continue
                    if not self.has_clearance(data, grid_msg, gx, gy, clearance, outside_ok=True):
                        continue
                    if any(math.hypot(gx - c[0], gy - c[1]) < 0.15 for c in out):
                        continue
                    out.append((gx, gy, math.atan2(tree['y'] - gy, tree['x'] - gx)))
        return out

    def pick_slot_candidate(self, data, grid_msg, tree, slot):
        # extra (never-faced) slots prefer room to turn in place
        clearances = ((SPIN_CLEARANCE_M, MIN_OBSTACLE_CLEARANCE_M, TREE_FALLBACK_CLEARANCE_M)
                      if slot['extra'] else None)
        for c in self.slot_candidates(data, grid_msg, tree, slot['angle'], clearances):
            if all(math.hypot(c[0] - bx, c[1] - by) > 0.3 for bx, by in slot['bad']):
                return c
        return None

    def slot_possible(self, tree, base_angle):
        """Could this slot EVER have a position inside the boundary (map-independent)?"""
        for rad in self.slot_radii(tree):
            for da in VP_ANGLE_STEPS_DEG:
                ang = base_angle + math.radians(da)
                if self.in_bounds_margin(tree['x'] + rad * math.cos(ang),
                                         tree['y'] + rad * math.sin(ang),
                                         self.goal_bound_margin):
                    return True
        return False

    def line_of_sight(self, data, grid_msg, px, py, tree):
        """No occupied cell between (px, py) and the trunk (the trunk's own cells are
        excluded by stopping r + 1.5 cells short of its centre)."""
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        h, w = data.shape
        dx, dy = tree['x'] - px, tree['y'] - py
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        stop = dist - (tree['r'] + 1.5 * res)
        if stop <= res:
            return True
        n = int(stop / (0.5 * res)) + 1
        ts = np.linspace(0.0, stop, n)
        cols = np.floor((px + dx / dist * ts - ox) / res).astype(int)
        rows = np.floor((py + dy / dist * ts - oy) / res).astype(int)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if not inside.any():
            return True
        return not np.any(data[rows[inside], cols[inside]] >= OCC_THRESHOLD)

    # ---- viewpoint merging ------------------------------------------------------
    def merge_key(self, members):
        return tuple(sorted((t['id'], self.slot_index(t, s),
                             round(s['cur'][0], 2), round(s['cur'][1], 2)) for t, s in members))

    def joint_viewpoint(self, data, grid_msg, members):
        """One position that serves every (tree, slot) in `members`, or None.

        Hard constraints, per member:  distance window, azimuth within MERGE_AZ_TOL_DEG of
        the slot's nominal angle, not within 0.3 m of a position that already failed for
        that slot, in bounds, obstacle clearance (0.30 m, 0.22 m fallback) and a clear
        line of sight to the trunk.
        Objective (minimise) on the lattice points that pass the cheap geometric tests:
            cost = 10 * (1 - min_i q_i) + mean_i |az_dev_i|/tol + mean_i |d_i - D_REF|/(D_MAX-D_MIN)
        with q_i = min(1, (D_REF/d_i)^2) the relative point density at tree i. The first
        term makes the WORST-served tree decide; the others keep the viewpoint close to
        the old azimuths and to the reference distance."""
        dmin, dmax, ref = self.tree_dist_min, self.tree_dist_max, TREE_DIST_REF_M
        az_tol = math.radians(MERGE_AZ_TOL_DEG)
        x0 = max(t['x'] - (t['r'] + dmax) for t, _ in members)
        x1 = min(t['x'] + (t['r'] + dmax) for t, _ in members)
        y0 = max(t['y'] - (t['r'] + dmax) for t, _ in members)
        y1 = min(t['y'] + (t['r'] + dmax) for t, _ in members)
        if x0 > x1 or y0 > y1:
            return None
        gx_, gy_ = np.meshgrid(np.arange(x0, x1 + 1e-9, MERGE_GRID_M),
                               np.arange(y0, y1 + 1e-9, MERGE_GRID_M))
        X, Y = gx_.ravel(), gy_.ravel()
        m = self.goal_bound_margin
        ok = ((X >= self.bounds_min_x - m) & (X <= self.bounds_max_x + m) &
              (Y >= self.bounds_min_y - m) & (Y <= self.bounds_max_y + m))
        min_q = np.ones_like(X)
        az_sum = np.zeros_like(X)
        d_sum = np.zeros_like(X)
        for t, s in members:
            dx, dy = X - t['x'], Y - t['y']
            ds = np.hypot(dx, dy) - t['r']
            ok &= (ds >= dmin - 1e-6) & (ds <= dmax + 1e-6)
            az = np.arctan2(dy, dx)
            dev = np.abs(np.arctan2(np.sin(az - s['angle']), np.cos(az - s['angle'])))
            ok &= dev <= az_tol + 1e-6
            min_q = np.minimum(min_q, np.minimum(1.0, (ref / np.maximum(ds, 1e-3)) ** 2))
            az_sum += dev / az_tol
            d_sum += np.abs(ds - ref) / (dmax - dmin)
        idx = np.nonzero(ok)[0]
        if idx.size == 0:
            return None
        cost = 10.0 * (1.0 - min_q) + az_sum / len(members) + d_sum / len(members)
        order = idx[np.argsort(cost[idx], kind='stable')][:MERGE_MAX_EVAL]
        for clearance in (MIN_OBSTACLE_CLEARANCE_M, TREE_FALLBACK_CLEARANCE_M):
            for i in order:
                x, y = float(X[i]), float(Y[i])
                if any(math.hypot(x - bx, y - by) <= 0.3
                       for _, s in members for bx, by in s['bad']):
                    continue
                if not self.has_clearance(data, grid_msg, x, y, clearance, outside_ok=True):
                    continue
                if not all(self.line_of_sight(data, grid_msg, x, y, t) for t, _ in members):
                    continue
                bs = [math.atan2(t['y'] - y, t['x'] - x) for t, _ in members]
                yaw = math.atan2(sum(math.sin(b) for b in bs), sum(math.cos(b) for b in bs))
                return (x, y, yaw)
        return None

    def merge_viewpoints(self, data, grid_msg):
        """Greedy grouping of pending viewpoints of DIFFERENT trees that are within
        merge_radius of each other. Returns the number of merged viewpoints created."""
        now = time.monotonic()
        items = []
        for t in self.trees:
            if t['done'] or now < t['cool']:
                continue
            for s in t['slots']:
                if (s['state'] == 'pending' and s['cur'] is not None and now >= s['cool']
                        and not s['merged'] and not s['no_merge']):
                    items.append((t, s))
        if len(items) < 2:
            return 0
        items.sort(key=lambda ts: (ts[1]['cur'][0], ts[1]['cur'][1]))
        used = set()
        n_groups = 0
        for i, (ti, si) in enumerate(items):
            if i in used:
                continue
            ax, ay = si['cur'][0], si['cur'][1]
            nbrs = sorted((j for j in range(len(items)) if j != i and j not in used),
                          key=lambda j: math.hypot(items[j][1]['cur'][0] - ax,
                                                   items[j][1]['cur'][1] - ay))
            group, gidx, best = [(ti, si)], [i], None
            for j in nbrs:
                tj, sj = items[j]
                if math.hypot(sj['cur'][0] - ax, sj['cur'][1] - ay) > self.merge_radius:
                    break                                   # sorted: all further ones are farther
                if len(group) >= MERGE_MAX_TREES:
                    break
                if any(tj is t for t, _ in group):
                    continue                                # one slot per tree in a group
                trial = group + [(tj, sj)]
                key = self.merge_key(trial)
                if key in self.merge_fail_cache:
                    continue
                jv = self.joint_viewpoint(data, grid_msg, trial)
                if jv is None:
                    self.merge_fail_cache.add(key)
                    continue
                group, best = trial, jv
                gidx.append(j)
            if best is None or len(group) < 2:
                continue
            for t, s in group:
                s['cur'] = best
                s['merged'] = True
            used.update(gidx)
            n_groups += 1
            desc = ', '.join(
                f"#{t['id']} ({math.hypot(best[0] - t['x'], best[1] - t['y']) - t['r']:.2f} m)"
                for t, _ in group)
            self.get_logger().info(
                f'Merged {len(group)} viewpoints into one at ({best[0]:.2f},{best[1]:.2f}); '
                f'trees (trunk-surface distance): {desc}')
        return n_groups

    # ---- trees from the registry --------------------------------------------------
    def maintain_tree_slots(self, data, grid_msg):
        """Give every pending slot a position; retire slots that cannot get one."""
        now = time.monotonic()
        for t in self.trees:
            if t['done']:
                continue
            for s in t['slots']:
                if s['state'] != 'pending' or s['cur'] is not None or now < s['cool']:
                    continue
                cand = self.pick_slot_candidate(data, grid_msg, t, s)
                if cand is not None:
                    s['cur'] = cand
                    s['merged'] = False
                    s['dry_t0'] = None
                elif s['dry_t0'] is None:
                    s['dry_t0'] = now
                elif now - s['dry_t0'] >= SLOT_DRY_LIMIT_S:
                    s['state'] = 'failed'
                    why = (f"after {s['attempts']} failed attempt(s)" if s['attempts']
                           else 'no feasible position found')
                    self.get_logger().warn(
                        f"Tree #{t['id']}: no usable viewpoint position left ({why}) "
                        f"-- marking this viewpoint unreachable")
            self.update_tree_done(t)

    def scan_for_new_trees(self, data, grid_msg):
        good = [(float(r[1]), float(r[2]), float(r[3]))
                for r in self.tree_rows if r[6] >= self.min_tree_obs]

        # Drop only trees we have NOT started and that the registry no longer
        # reports nearby. Started/done trees are frozen forever.
        kept = []
        for t in self.trees:
            if t['done'] or t['started']:
                kept.append(t)
            elif any(math.hypot(gx - t['x'], gy - t['y']) < TREE_MATCH_RADIUS_M
                     for gx, gy, _ in good):
                kept.append(t)
            else:
                self.get_logger().info(
                    f"Tree #{t['id']} at ({t['x']:.1f},{t['y']:.1f}) no longer in registry "
                    f"and not started -- dropping")
        self.trees = kept

        added = 0
        for tx, ty, r in good:
            if self.tree_near(tx, ty) is not None:
                continue
            if added >= MAX_NEW_TREES_PER_CYCLE:
                break
            self.next_tree_id += 1
            tree = {
                'id': self.next_tree_id, 'x': tx, 'y': ty, 'r': r,
                'slots': [self.new_slot(k * (2.0 * math.pi / 3.0)) for k in range(3)],
                'started': False, 'done': False, 'cool': 0.0, 'fails': 0,
                'faced': False,         # caught in the camera at least once?
                'face_tries': 0,        # spin attempts so far
                'extra_slots': 0,       # extra viewpoint slots added for facing
                'gave_up': False,
            }
            for s in tree['slots']:
                if not self.slot_possible(tree, s['angle']):
                    s['state'] = 'skipped'
                    self.get_logger().info(
                        f"Tree #{tree['id']} at ({tx:.1f},{ty:.1f}): viewpoint at "
                        f"{math.degrees(s['angle']):.0f} deg would be outside the boundary -- skipped")
            self.trees.append(tree)
            self.update_tree_done(tree)
            added += 1

        self.maintain_tree_slots(data, grid_msg)
        return added

    def plan_tree_tour(self, robot_xy):
        """Goals for all ready tree viewpoints, one whole tree at a time. A merged
        viewpoint is listed once, however many trees it serves."""
        now = time.monotonic()
        trees = []
        for t in self.trees:
            if t['done'] or now < t['cool']:
                continue
            ready = [s for s in t['slots']
                     if s['state'] == 'pending' and s['cur'] is not None and now >= s['cool']]
            if ready:
                trees.append((t, ready))
        if not trees:
            return []
        order = self.tsp_order([(t['x'], t['y']) for t, _ in trees], robot_xy)
        goals, cur_xy, seen = [], robot_xy, set()
        for k in order:
            pend = list(trees[k][1])
            while pend:
                s = min(pend, key=lambda s: math.hypot(s['cur'][0] - cur_xy[0],
                                                       s['cur'][1] - cur_xy[1]))
                pend.remove(s)
                key = (round(s['cur'][0], 3), round(s['cur'][1], 3))
                if key in seen:
                    continue
                seen.add(key)
                goals.append(s['cur'])
                cur_xy = (s['cur'][0], s['cur'][1])
        return goals

    # ------------------------------------------------------------------
    # goal bookkeeping
    # ------------------------------------------------------------------
    def retire_frontier(self, cx, cy):
        self.done_frontier_centroids.append((cx, cy))

    def register_goal_success(self, gx, gy):
        """Mark every slot at this position visited (several if the viewpoint is merged).
        update_tree_done is NOT called here: it needs the facing result, see face_finish()."""
        self.fail_keys.clear()
        matches = self.find_tree_vps(gx, gy)
        if matches:
            for tree, slot in matches:
                slot['state'] = 'visited'
                tree['started'] = True
        else:
            c = self.goal_centroids.pop((round(gx, 2), round(gy, 2)), None)
            if c is not None:
                self.retire_frontier(*c)
        self.dummy_targets = [t for t in self.dummy_targets
                              if math.hypot(t[0] - gx, t[1] - gy) > 0.3]

    def register_goal_failure(self, gx, gy):
        self.goal_queue.clear()
        matches = self.find_tree_vps(gx, gy)
        key = ('t', matches[0][0]['id']) if matches else ('g', round(gx), round(gy))
        self.fail_keys.add(key)

        if len(self.fail_keys) >= SYSTEMIC_DISTINCT:
            self.get_logger().error(
                f'nav2 has failed goals at {len(self.fail_keys)} different places with no '
                f'success in between. If this happens right from the start, look at the '
                f'costmaps around the robot (planner "failed to plan from ..." / spin '
                f'"Collision Ahead") -- the explorer cannot fix that.',
                throttle_duration_sec=30.0)

        self.failed_goals.append((gx, gy, time.monotonic()))

        if matches:
            now = time.monotonic()
            for tree, slot in matches:
                slot['attempts'] += 1
                slot['bad'].append((slot['cur'][0], slot['cur'][1]))
                if slot['merged']:
                    slot['no_merge'] = True          # un-merge: carry on individually
                slot['merged'] = False
                slot['cur'] = None                   # next position is picked after the cooldown
                slot['cool'] = now + SLOT_COOLDOWN_S
                tree['cool'] = now + TREE_COOLDOWN_S    # let the tour visit other trees first
                tree['started'] = True
                tree['fails'] += 1
                if (tree['fails'] >= TREE_GIVEUP_FAILS and
                        not any(x['state'] == 'visited' for x in tree['slots'])):
                    tree['gave_up'] = True
                    for x in tree['slots']:
                        if x['state'] == 'pending':
                            x['state'] = 'failed'
                            x['cur'] = None
                            x['merged'] = False
                    self.get_logger().warn(
                        f"Tree #{tree['id']} at ({tree['x']:.1f},{tree['y']:.1f}): "
                        f"{tree['fails']} failed goals and no viewpoint ever reached -- giving up on this tree")
                elif slot['attempts'] >= MAX_SLOT_ATTEMPTS:
                    slot['state'] = 'failed'
                    self.get_logger().warn(
                        f"Tree #{tree['id']}: viewpoint failed {slot['attempts']} times "
                        f"-- marking unreachable")
                else:
                    self.get_logger().info(
                        f"Tree #{tree['id']}: viewpoint failed ({slot['attempts']}/"
                        f"{MAX_SLOT_ATTEMPTS}); will retry from a different position later")
                self.update_tree_done(tree)
            return

        c = self.goal_centroids.pop((round(gx, 2), round(gy, 2)), None)
        if c is not None:
            for rec in self.frontier_fails:
                if math.hypot(rec[0] - c[0], rec[1] - c[1]) < FRONTIER_DONE_RADIUS_M:
                    rec[2] += 1
                    n = rec[2]
                    break
            else:
                self.frontier_fails.append([c[0], c[1], 1])
                n = 1
            if n >= MAX_FRONTIER_FAILS:
                self.get_logger().info(
                    f'Giving up on frontier region near ({c[0]:.1f},{c[1]:.1f}) '
                    f'after {n} failed goals')
                self.retire_frontier(*c)
            return

        # external dummy target
        key = (round(gx, 1), round(gy, 1))
        self.fail_counts[key] = self.fail_counts.get(key, 0) + 1
        if self.fail_counts[key] >= MAX_SLOT_ATTEMPTS:
            self.dummy_targets = [t for t in self.dummy_targets
                                  if math.hypot(t[0] - gx, t[1] - gy) > 0.3]

    # ------------------------------------------------------------------
    # facing: every tree must be caught in the camera at least once
    # ------------------------------------------------------------------
    def tree_bearing_if_visible(self, tree, px, py, data, grid_msg):
        """Bearing from (px, py) to the tree if it is in range and not occluded, else None."""
        ds = math.hypot(tree['x'] - px, tree['y'] - py) - tree['r']
        if ds > self.tree_dist_max + FACE_RANGE_SLACK_M:
            return None
        if not self.line_of_sight(data, grid_msg, px, py, tree):
            return None
        return math.atan2(tree['y'] - py, tree['x'] - px)

    def credit_faced(self, px, py, yaw, data, grid_msg):
        """Mark every in-range, unoccluded tree that lies within the camera tolerance
        of the robot's heading as faced."""
        for t in self.trees:
            if t['faced']:
                continue
            b = self.tree_bearing_if_visible(t, px, py, data, grid_msg)
            if b is None or abs(self._wrap(b - yaw)) > self.face_tol:
                continue
            t['faced'] = True
            if self.face_ctx is not None:
                self.face_ctx['trees'].add(t['id'])
            self.get_logger().info(
                f"Tree #{t['id']} at ({t['x']:.1f},{t['y']:.1f}) caught in the camera "
                f"(bearing error {math.degrees(abs(self._wrap(b - yaw))):.0f} deg)")

    def begin_face_sequence(self, affected_trees):
        """Called on arrival at a tree viewpoint. True if a spin was started (the caller
        must then leave `exploring` set); False if nothing needs facing."""
        self.face_ctx = {'trees': {t['id'] for t in affected_trees}, 'tried': set()}
        return self.face_step()

    def face_step(self):
        ctx = self.face_ctx
        if ctx is None:
            return False
        grid_msg = self.latest_map
        xy = self.get_robot_pose_xy()
        yaw = self.get_robot_yaw()
        if grid_msg is None or xy is None or yaw is None:
            self.face_finish()
            return False
        data = np.array(grid_msg.data, dtype=np.int8).reshape(
            grid_msg.info.height, grid_msg.info.width)

        # (1) the current heading may already catch a tree -- no spin needed for those
        self.credit_faced(xy[0], xy[1], yaw, data, grid_msg)

        # (2) trees still to face from here: unfaced, attempts left, not tried at this
        #     arrival yet, in range and unoccluded (members of this viewpoint and others)
        targets = []
        for t in self.trees:
            if t['faced'] or t['face_tries'] >= MAX_FACE_TRIES or t['id'] in ctx['tried']:
                continue
            b = self.tree_bearing_if_visible(t, xy[0], xy[1], data, grid_msg)
            if b is not None:
                targets.append((t, b))
        if not targets:
            self.face_finish()
            return False

        # (3) one spin may catch several trees: start with the tree needing the smallest
        #     turn, then add trees while the whole bundle spans <= 2 x camera tolerance
        targets.sort(key=lambda tb: abs(self._wrap(tb[1] - yaw)))
        b0 = targets[0][1]
        lo = hi = 0.0
        group = []
        for t, b in sorted(targets, key=lambda tb: abs(self._wrap(tb[1] - b0))):
            rel = self._wrap(b - b0)
            nlo, nhi = min(lo, rel), max(hi, rel)
            if nhi - nlo <= 2.0 * self.face_tol:
                lo, hi = nlo, nhi
                group.append(t)
        target_yaw = b0 + 0.5 * (lo + hi)
        for t in group:
            t['face_tries'] += 1
            ctx['tried'].add(t['id'])
            ctx['trees'].add(t['id'])
        return self.send_face_spin(target_yaw, group)

    def face_finish(self):
        """End of the facing sequence: now that 'faced' is known, re-evaluate the trees."""
        ctx = self.face_ctx
        self.face_ctx = None
        if ctx is None:
            return
        for t in list(self.trees):
            if t['id'] in ctx['trees']:
                self.update_tree_done(t)

    def send_face_spin(self, target_absolute_yaw, group):
        current_yaw = self.get_robot_yaw()
        if current_yaw is None:
            self.get_logger().warn('Could not read current yaw, skipping facing spin')
            self.face_finish()
            return False
        rel = self._wrap(target_absolute_yaw - current_yaw)

        if not self.spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('spin action server not available, skipping facing spin')
            self.face_finish()
            return False

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = rel
        goal_msg.time_allowance = Duration(sec=8)
        ids = ', '.join(f"#{t['id']}" for t in group)
        self.get_logger().info(
            f'Facing tree(s) {ids}: spinning {math.degrees(rel):.0f} deg')
        future = self.spin_client.send_goal_async(goal_msg)
        future.add_done_callback(self.spin_response_cb)
        return True

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    def record_trail(self):
        robot_xy = self.get_robot_pose_xy()
        if robot_xy is None:
            return
        if (not self.visit_pts or
                math.hypot(robot_xy[0] - self.visit_pts[-1][0],
                           robot_xy[1] - self.visit_pts[-1][1]) > 0.75):
            self.visit_pts.append((robot_xy[0], robot_xy[1]))
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = robot_xy[0]
        pose.pose.position.y = robot_xy[1]
        pose.pose.orientation.w = 1.0
        self.trail_path.poses.append(pose)
        if len(self.trail_path.poses) > self.trail_max_poses:
            self.trail_path.poses.pop(0)
        self.trail_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.trail_path)

    def publish_planning_markers(self, robot_xy):
        """blue spheres: frontier candidates | orange cubes: pending trees |
        grey cubes: DONE trees | green arrows: pending tree viewpoints |
        cyan arrows: MERGED tree viewpoints | yellow arrows: dummy targets |
        magenta: planned route."""
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        now = self.get_clock().now().to_msg()
        mid = 0

        def base(ns, mtype, x, y, z):
            nonlocal mid
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = now
            m.ns = ns
            m.id = mid
            mid += 1
            m.type = mtype
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            return m

        def make_blob(ns, mtype, x, y, z, rgb, scale):
            m = base(ns, mtype, x, y, z)
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.9
            return m

        def make_arrow(x, y, yaw, rgb, ns, length=0.4):
            m = base(ns, Marker.ARROW, x, y, 0.25)
            m.pose.orientation.z = math.sin(yaw / 2.0)
            m.pose.orientation.w = math.cos(yaw / 2.0)
            m.scale.x, m.scale.y, m.scale.z = length, 0.08, 0.08
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 0.95
            return m

        for c in self.last_candidates:
            markers.markers.append(make_blob('candidates', Marker.SPHERE, c['x'], c['y'], 0.2,
                                             (0.2, 0.4, 1.0), 0.25))
        for t in self.trees:
            rgb = (0.5, 0.5, 0.5) if t['done'] else (1.0, 0.5, 0.0)
            markers.markers.append(make_blob('trees', Marker.CUBE, t['x'], t['y'], 0.3, rgb, 0.3))
            for s in t['slots']:
                if s['state'] == 'pending' and s['cur'] is not None:
                    markers.markers.append(
                        make_arrow(s['cur'][0], s['cur'][1], s['cur'][2],
                                   (0.0, 1.0, 1.0) if s['merged'] else (0.0, 1.0, 0.0),
                                   'tree_viewpoints'))
        for (gx, gy, _bonus, yaw) in self.dummy_targets:
            markers.markers.append(make_arrow(gx, gy, yaw, (1.0, 1.0, 0.0), 'dummy_targets'))

        if self.goal_queue:
            line = Marker()
            line.header.frame_id = self.global_frame
            line.header.stamp = now
            line.ns = 'planned_route'
            line.id = mid
            mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.06
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.0, 1.0, 0.9
            line.pose.orientation.w = 1.0
            line.points.append(Point(x=robot_xy[0], y=robot_xy[1], z=0.1))
            for (gx, gy, gyaw) in self.goal_queue:
                line.points.append(Point(x=gx, y=gy, z=0.1))
                markers.markers.append(make_arrow(gx, gy, gyaw if gyaw is not None else 0.0,
                                                  (1.0, 0.0, 1.0), 'planned_route'))
            markers.markers.append(line)

        self.marker_pub.publish(markers)

    # ------------------------------------------------------------------
    # stats / completion
    # ------------------------------------------------------------------
    def compute_coverage_pct(self, data):
        total = data.size
        if total == 0:
            return 0.0
        return 100.0 * np.count_nonzero(data != UNKNOWN) / total

    def coverage_in_plot(self):
        """% of the plot area (the bounds) that is known (free or occupied)."""
        msg = self.latest_map
        if msg is None:
            return None
        h, w = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        inb = self.bounds_mask(msg)
        total = int(inb.sum())
        if total == 0:
            return None
        known = np.count_nonzero((data != UNKNOWN) & inb)
        return min(100.0, 100.0 * known / total)

    def tree_report_lines(self):
        lines = []
        for t in self.trees:
            st = [s['state'] for s in t['slots']]
            lines.append(
                f"  tree #{t['id']} at ({t['x']:.1f},{t['y']:.1f}): "
                f"{st.count('visited')} visited, {st.count('failed')} unreachable, "
                f"{st.count('skipped')} skipped, {st.count('pending')} pending, "
                f"{t['face_tries']} spin attempt(s)"
                f"{'' if t['done'] else '  [NOT DONE]'}"
                f"{'' if t['faced'] else '  [NOT CAUGHT IN CAMERA]'}")
        return lines

    def announce_complete(self, data, reason):
        if self.declared_complete:
            return
        self.declared_complete = True
        self.complete_elapsed = time.monotonic() - self.start_time
        pct = self.compute_coverage_pct(data)
        n_done = sum(1 for t in self.trees if t['done'])
        n_faced = sum(1 for t in self.trees if t['faced'])
        n_vis = sum(1 for t in self.trees for s in t['slots'] if s['state'] == 'visited')
        n_bad = sum(1 for t in self.trees for s in t['slots'] if s['state'] == 'failed')
        self.get_logger().info(
            f'Exploration complete: {reason}. '
            f'{self.visited_frontier_count} goal(s) reached, '
            f'{n_done}/{len(self.trees)} trees done, {n_faced}/{len(self.trees)} caught in camera '
            f'({n_vis} viewpoints visited, {n_bad} unreachable), '
            f'{len(self.done_frontier_centroids)} frontier regions retired. '
            f'Map coverage (known cells in grid): {pct:.1f}%.')
        for line in self.tree_report_lines():
            self.get_logger().info(line)
        if self.shutdown_on_complete:
            self.finished = True

    def print_final_stats(self):
        elapsed = time.monotonic() - self.start_time
        cov_plot = self.coverage_in_plot()
        cov_grid = (self.compute_coverage_pct(
            np.array(self.latest_map.data, dtype=np.int8)) if self.latest_map else None)
        print()
        print("=" * 60)
        print("EXPLORATION STATS")
        print("=" * 60)
        print("run time (wall clock)  : %.1f s (%.1f min)" % (elapsed, elapsed / 60))
        if self.complete_elapsed is not None:
            print("declared complete at   : %.1f s" % self.complete_elapsed)
        else:
            print("declared complete at   : not reached")
        if cov_plot is not None and cov_grid is not None:
            print("coverage of plot area  : %.1f %%" % cov_plot)
            print("coverage of map grid   : %.1f %%" % cov_grid)
        else:
            print("coverage               : no map received")
        print("goals reached          : %d" % self.visited_frontier_count)
        print("frontier goals sent    : %d" % self.frontier_goals_sent)
        print("trees accepted         : %d" % len(self.trees))
        print("trees done             : %d" % sum(1 for t in self.trees if t['done']))
        print("trees caught in camera : %d" % sum(1 for t in self.trees if t['faced']))
        print("frontier regions retired: %d" % len(self.done_frontier_centroids))
        print("centroids out of range : %d (at last registry update)" % self.n_centroids_outside)
        print("failed goals (active)  : %d" % len(self.failed_goals))
        for line in self.tree_report_lines():
            print(line)
        print("=" * 60)

    # ------------------------------------------------------------------
    # frontier detection
    # ------------------------------------------------------------------
    def detect_frontier_clusters(self, grid_msg: OccupancyGrid, data):
        h, w = data.shape
        res = grid_msg.info.resolution
        inb = self.bounds_mask(grid_msg)

        # remove unknown specks (octomap projection holes), THEN restrict to bounds
        unknown_raw = (data == UNKNOWN)
        r = int(UNKNOWN_SPECK_M / res / 2.0)
        if r >= 1:
            unknown_raw = self._dilate(self._erode(unknown_raw, r), r)
        unknown_mask = unknown_raw & inb
        free_mask = (data == FREE) & inb

        frontier_mask = np.zeros_like(free_mask)
        padded = np.pad(unknown_mask, 1, mode='constant', constant_values=False)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                frontier_mask |= (free_mask & padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])

        visited = np.zeros_like(frontier_mask)
        clusters = []
        ys, xs = np.nonzero(frontier_mask)
        frontier_cells = set(zip(ys.tolist(), xs.tolist()))

        for start in frontier_cells:
            if visited[start]:
                continue
            q = deque([start])
            visited[start] = True
            cluster = []
            while q:
                cy, cx = q.popleft()
                cluster.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (ny, nx) in frontier_cells and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
            if len(cluster) >= self.min_frontier_cells:
                clusters.append(cluster)
        return clusters

    def in_bounds(self, wx, wy):
        return (self.bounds_min_x <= wx <= self.bounds_max_x and
                self.bounds_min_y <= wy <= self.bounds_max_y)

    def in_bounds_margin(self, x, y, m):
        return (self.bounds_min_x - m <= x <= self.bounds_max_x + m and
                self.bounds_min_y - m <= y <= self.bounds_max_y + m)

    def has_clearance(self, data, grid_msg, wx, wy, min_clearance_m, outside_ok=False):
        """True if no occupied cell centre lies within min_clearance_m of (wx, wy).
        outside_ok: a point outside the known map counts as clear (unknown, not blocked)."""
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        h, w = data.shape
        col = int((wx - ox) / res)
        row = int((wy - oy) / res)
        if not (0 <= row < h and 0 <= col < w):
            return bool(outside_ok)
        r_cells = max(1, int(math.ceil(min_clearance_m / res)))
        r0, r1 = max(0, row - r_cells), min(h, row + r_cells + 1)
        c0, c1 = max(0, col - r_cells), min(w, col + r_cells + 1)
        rr, cc = np.nonzero(data[r0:r1, c0:c1] >= OCC_THRESHOLD)
        if rr.size == 0:
            return True
        dy = oy + (rr + r0 + 0.5) * res - wy
        dx = ox + (cc + c0 + 0.5) * res - wx
        return not np.any(dx * dx + dy * dy <= min_clearance_m * min_clearance_m)

    def find_standoff_goal(self, data, grid_msg, cluster, robot_xy):
        wx, wy = self.cluster_centroid_world(cluster, grid_msg)
        rx, ry = robot_xy
        dx, dy = rx - wx, ry - wy
        base_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-3 else 0.0
        angle_offsets = [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6]
        standoffs = (GOAL_STANDOFF_M, GOAL_STANDOFF_M * 0.6, GOAL_STANDOFF_M * 0.3, 0.0)
        for angle_off in angle_offsets:
            ux, uy = math.cos(base_angle + angle_off), math.sin(base_angle + angle_off)
            for standoff in standoffs:
                gx, gy = wx + ux * standoff, wy + uy * standoff
                if (self.in_bounds_margin(gx, gy, self.goal_bound_margin) and
                        self.has_clearance(data, grid_msg, gx, gy, MIN_OBSTACLE_CLEARANCE_M)):
                    return (gx, gy, None)   # frontier goals carry no final heading
        return None

    def cluster_centroid_world(self, cluster, grid_msg: OccupancyGrid):
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        mean_row = sum(c[0] for c in cluster) / len(cluster)
        mean_col = sum(c[1] for c in cluster) / len(cluster)
        return ox + (mean_col + 0.5) * res, oy + (mean_row + 0.5) * res

    # ------------------------------------------------------------------
    # information gain / candidate scoring
    # ------------------------------------------------------------------
    def compute_information_gain(self, data, grid_msg, wx, wy):
        """Raycast a 360-degree sweep and count UNKNOWN cells it would observe."""
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        h, w = data.shape
        col0 = (wx - ox) / res
        row0 = (wy - oy) / res
        max_range_cells = int(self.sensor_range_m / res)

        gain = 0
        for i in range(NUM_RAYS):
            angle = 2.0 * math.pi * i / NUM_RAYS
            dx, dy = math.cos(angle), math.sin(angle)
            for step in range(1, max_range_cells + 1):
                c = int(round(col0 + dx * step))
                r = int(round(row0 + dy * step))
                if not (0 <= r < h and 0 <= c < w):
                    break
                val = data[r, c]
                if val >= OCC_THRESHOLD:
                    break
                if val == UNKNOWN:
                    gain += 1
        return gain

    def score_candidates(self, clusters, data, grid_msg, robot_xy):
        rx, ry = robot_xy
        candidates = []
        n_blacklisted = n_no_clearance = n_too_close = 0
        n_out_of_bounds = n_too_far = n_retired = n_low_gain = n_visited = 0

        inb = self.bounds_mask(grid_msg)
        gain_data = np.where(inb, data, 100).astype(np.int8)

        for cluster in clusters:
            cx, cy = self.cluster_centroid_world(cluster, grid_msg)
            if not self.in_bounds(cx, cy):
                n_out_of_bounds += 1
                continue
            if any(math.hypot(cx - dx_, cy - dy_) < FRONTIER_DONE_RADIUS_M
                   for dx_, dy_ in self.done_frontier_centroids):
                n_retired += 1
                continue
            if self.near_visited(cx, cy, FRONTIER_VISITED_RADIUS_M):
                # the robot has been right here and it is still unknown: not worth chasing
                n_visited += 1
                continue
            if math.hypot(cx - rx, cy - ry) > self.candidate_radius:
                n_too_far += 1
                continue

            goal = self.find_standoff_goal(data, grid_msg, cluster, robot_xy)
            if goal is None:
                n_no_clearance += 1
                continue
            gx, gy, yaw = goal

            if self.is_blacklisted(gx, gy):
                n_blacklisted += 1
                continue
            dist = math.hypot(gx - rx, gy - ry)
            if dist < 0.3:
                n_too_close += 1
                continue

            gain = self.compute_information_gain(gain_data, grid_msg, gx, gy)
            if gain < MIN_INFO_GAIN:
                self.retire_frontier(cx, cy)
                n_low_gain += 1
                continue

            self.goal_centroids[(round(gx, 2), round(gy, 2))] = (cx, cy)
            candidates.append({
                'x': gx, 'y': gy, 'yaw': yaw, 'cx': cx, 'cy': cy,
                'gain': gain, 'dist': dist,
                'utility': gain / ((dist + 0.5) ** UTILITY_DISTANCE_ALPHA),
                'cluster_size': len(cluster),
            })

        if not candidates and clusters:
            self.get_logger().info(
                f'{len(clusters)} frontier cluster(s) found but none selectable: '
                f'{n_out_of_bounds} outside bounds, {n_retired} already retired, '
                f'{n_visited} already visited, '
                f'{n_low_gain} low info gain (now retired), '
                f'{n_too_far} too far (>{self.candidate_radius:.1f}m), '
                f'{n_no_clearance} failed clearance, {n_blacklisted} blacklisted, '
                f'{n_too_close} at robot position.',
                throttle_duration_sec=10.0)
        self.last_n_too_far = n_too_far
        self.last_n_blacklisted = n_blacklisted
        return candidates

    def is_blacklisted(self, x, y):
        now = time.monotonic()
        self.failed_goals = [(bx, by, t) for (bx, by, t) in self.failed_goals
                             if now - t < BLACKLIST_TTL_S]
        return any(math.hypot(x - bx, y - by) < REPLAN_ON_FAILURE_BLACKLIST_M
                   for (bx, by, _t) in self.failed_goals)

    # ------------------------------------------------------------------
    # nav2 goal dispatch
    # ------------------------------------------------------------------
    def send_nav_goal(self, xyyaw):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('navigate_to_pose action server not available yet')
            return

        wx, wy, align_yaw = xyyaw
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = wx
        pose.pose.position.y = wy

        robot_now = self.get_robot_pose_xy()
        if robot_now is not None and math.hypot(wx - robot_now[0], wy - robot_now[1]) > 1e-3:
            travel_yaw = math.atan2(wy - robot_now[1], wx - robot_now[0])
        else:
            travel_yaw = 0.0
        pose.pose.orientation.z = math.sin(travel_yaw / 2.0)
        pose.pose.orientation.w = math.cos(travel_yaw / 2.0)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        if self.find_tree_vp(wx, wy) is None:
            self.frontier_goals_sent += 1
        self.exploring = True
        self.complete_since = None
        self.watchdog_fired = False
        self.stall_ref = None
        self.goal_sent_t = time.monotonic()
        self.get_logger().info(
            f'Sending goal: ({wx:.2f}, {wy:.2f}) [{len(self.goal_queue)} more queued]')
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.goal_response_cb(f, (wx, wy, align_yaw)))

    def goal_response_cb(self, future, goal_info):
        wx, wy, align_yaw = goal_info
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by nav2')
            self.goal_sent_t = None
            self.register_goal_failure(wx, wy)
            self.exploring = False
            self.explore_step()
            return
        self.current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(
            lambda f: self.goal_result_cb(f, goal_info))

    def goal_result_cb(self, future, goal_info):
        gx, gy, align_yaw = goal_info
        status = future.result().status
        self.current_goal_handle = None
        self.goal_sent_t = None
        if status == 4:  # SUCCEEDED
            self.watchdog_fired = False
            self.get_logger().info(f'Reached goal ({gx:.2f}, {gy:.2f})')
            self.visited_frontier_count += 1
            arrived = self.find_tree_vps(gx, gy)       # >1 entry when the viewpoint is merged
            self.register_goal_success(gx, gy)
            if arrived and self.begin_face_sequence([t for t, _ in arrived]):
                return                                  # a facing spin is in flight
        elif status == 5 and not self.watchdog_fired:  # CANCELED by someone else
            self.get_logger().info(f'Goal to ({gx:.2f},{gy:.2f}) canceled')
        else:  # ABORTED, or cancelled by our own watchdog
            self.watchdog_fired = False
            self.get_logger().warn(
                f'Goal to ({gx:.2f},{gy:.2f}) did not succeed (status={status})')
            self.register_goal_failure(gx, gy)
            self.start_escape()
        self.exploring = False
        self.explore_step()

    def spin_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Facing spin rejected')
            self.face_finish()
            self.exploring = False
            return
        goal_handle.get_result_async().add_done_callback(self.spin_result_cb)

    def spin_result_cb(self, future):
        # Whatever the Spin action reported, face_step() re-reads the real heading from
        # TF, credits the trees now in the camera, and starts the next spin if some
        # in-range tree is still unfaced (each tree is tried once per arrival).
        if self.face_step():
            return
        self.exploring = False
        self.explore_step()


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.print_final_stats()
        except Exception as e:
            print("could not print stats:", e)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

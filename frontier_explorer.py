#!/usr/bin/env python3
"""
Frontier + information-gain viewpoint exploration node.

Pipeline (upgraded from the greedy-nearest MVP):
  1. Subscribe to a 2D nav_msgs/OccupancyGrid (/projected_map from octomap_server,
     itself fed by GLIM's 3D map). Cell values: 0=free, 100=occupied, -1=unknown.
  2. Detect frontier cells: free cells 8-adjacent to >=1 unknown cell.
  3. Cluster frontier cells into connected components (poor-man's WFD).
  4. For each cluster, compute a safe standoff viewpoint (pulled back from the
     obstacle-adjacent boundary into clear free space), oriented to face the
     unknown region.
  5. Score each viewpoint by an actual visibility/information-gain estimate
     (raycast a synthetic sensor sweep from the candidate pose and count how
     many currently-unknown cells it would newly observe), not just raw
     frontier cluster pixel count.
  6. Take the top-K candidate viewpoints by utility (gain / distance^alpha,
     a cost-utility formulation in the same spirit as next-best-view / FALCON-
     style planners) and order them into a short visitation queue via a cheap
     greedy nearest-neighbor chain from the robot's current pose -- a light
     stand-in for FALCON's global coverage-path-guided visitation ordering,
     without the full TSP solver.
  7. Drain the queue one NavigateToPose goal at a time; only re-run full
     detection + rescoring once the queue empties or a goal fails, so the
     robot commits to a locally coherent sweep instead of re-planning myopically
     after every single step.
  8. Stop when no frontier clusters remain above the minimum size, anywhere
     inside the configured world bounds.
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


FREE = 0
UNKNOWN = -1
OCC_THRESHOLD = 50  # cells >= this are treated as occupied/obstacle

MIN_FRONTIER_CELLS = 6          # ignore tiny noisy clusters
GOAL_STANDOFF_M = 0.6           # pull goal back from the frontier boundary into free space
MIN_OBSTACLE_CLEARANCE_M = 0.28  # matches robot footprint half-width (~0.18m) + small margin,
                                  # not an arbitrarily conservative fixed value
REPLAN_ON_FAILURE_BLACKLIST_M = 0.75  # don't re-pick a goal within this radius of a failed one

SENSOR_RANGE_M = 6.0     # synthetic sensor range used for the info-gain raycast estimate
NUM_RAYS = 72            # angular resolution of the info-gain raycast sweep (5 deg steps)
TOP_K_CANDIDATES = 5     # how many best-utility viewpoints to chain into a visitation queue
UTILITY_DISTANCE_ALPHA = 2.5  # cost-utility exponent: utility = gain / (dist + eps)^alpha
                                # steep on purpose -- a far, big-gain frontier should NOT
                                # be able to outrank a close, modest one; this forces
                                # genuinely local, expanding-outward exploration instead
                                # of jumping to whichever single spot has the most reward
MAX_CANDIDATE_RADIUS_M = 8.0    # ignore frontier clusters farther than this from the
                                 # robot entirely this cycle -- they'll be picked up
                                 # naturally once the robot is actually near them

# ---- EDIT HERE to inject externally-supplied "must visit" / high-priority targets ----
# Each entry: (x, y, bonus, yaw). `bonus` is subtracted from every edge cost *into*
# that node in the TSP cost matrix (see build_cost_matrix), so a bigger bonus makes
# the solver more strongly prefer routing to that node early/directly. Set bonus
# very large (e.g. 1e6) to effectively force it to be visited immediately after the
# nearest opportunity; set it near 0 to let it compete on equal footing with
# ordinary frontier viewpoints. `yaw` (radians) is the heading the robot should
# face on arrival -- e.g. facing into a tree trunk for an inspection viewpoint.
DUMMY_TARGETS = [
    # (x, y, bonus, yaw)  -- example, delete/replace with real targets:
    # (3.5, -2.0, 50.0, 0.0),
]

# ---- Stand-in "perception pipeline" for tree inspection ----
# Until the real perception pipeline exists, every occupied-cell cluster the
# explorer discovers is treated as a "detected tree". For each newly-discovered
# tree, we auto-generate 3 viewpoints spaced 120 degrees apart around it at
# TREE_INSPECTION_RADIUS_M, each oriented to face the tree, and inject them as
# dummy targets with TREE_VIEWPOINT_BONUS. This is the exact same mechanism a
# real perception node would use -- it would just call
# add_tree_inspection_viewpoints(x, y) itself instead of us discovering trees
# from the occupancy grid.
MIN_TREE_CLUSTER_CELLS = 4        # ignore tiny obstacle-noise blobs, not real trees
TREE_INSPECTION_RADIUS_M = 1.0    # standoff distance for the 3 circular viewpoints
MAX_TREE_CLUSTER_DIAMETER_M = 0.8  # cap a single "tree" cluster's bounding-box span --
                                    # trees close enough together to be 8-connected in
                                    # the occupancy grid would otherwise merge into one
                                    # giant blob treated as a single tree. Capping growth
                                    # here forces them to split back into separate,
                                    # per-trunk-sized detections.
TREE_VIEWPOINT_BONUS = 2.5        # priority of tree viewpoints vs frontier viewpoints
                                   # (in the same meters-equivalent unit as
                                   # MAX_FRONTIER_BONUS_M -- keep these comparable)
MAX_FRONTIER_BONUS_M = 2.5         # cap on how much a frontier viewpoint's raw info-gain
                                    # can bias the TSP route, expressed in the same units
                                    # as real travel distance (meters). Gain is normalized
                                    # to [0, MAX_FRONTIER_BONUS_M] before entering the cost
                                    # matrix -- this keeps it a tiebreaker between similarly-
                                    # distant options, not something that can override real
                                    # distance and send the robot on a long detour.
TREE_DEDUP_RADIUS_M = 2.0         # don't re-detect/re-target a tree we've already handled
                                   # (loose enough to tolerate cluster centroid jitter as
                                   # the octomap updates between scans of the same tree)
MAX_NEW_TREES_PER_CYCLE = 4       # rate-limit how many new trees get queued per 2s tick
MAX_DUMMY_TARGETS = 36            # hard cap so a burst of detections can't pile up
                                   # unboundedly and trap the robot locally


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')

        self.declare_parameter('map_topic', '/projected_map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('min_frontier_cells', MIN_FRONTIER_CELLS)
        self.declare_parameter('bounds_min_x', -25.0)
        self.declare_parameter('bounds_max_x', 25.0)
        self.declare_parameter('bounds_min_y', -25.0)
        self.declare_parameter('bounds_max_y', 25.0)
        self.declare_parameter('sensor_range_m', SENSOR_RANGE_M)
        self.declare_parameter('top_k_candidates', TOP_K_CANDIDATES)

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

        qos = QoSProfile(depth=5)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.VOLATILE

        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_cb, qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.latest_map = None
        self.exploring = False
        self.failed_goals = []       # list of (x, y) we shouldn't retry near
        self.visited_frontier_count = 0
        self.goal_queue = []         # ordered list of (x, y, yaw) still to visit this "sweep"
        self.stall_count = 0         # consecutive cycles where no candidate was selectable
        self.declared_complete = False
        self.stall_limit = 5         # ~10s of no-progress before declaring complete
        # EDIT HERE (or reassign at runtime / add a topic to push into this list)
        # to inject externally-supplied target locations into the TSP solve --
        # see DUMMY_TARGETS and solve_viewpoint_order for how bonus is used.
        self.dummy_targets = list(DUMMY_TARGETS)
        # (x, y) centroids of trees we've already generated inspection viewpoints
        # for, so we don't regenerate the same 3 viewpoints every 2s forever.
        self.processed_trees = []

        self.timer = self.create_timer(2.0, self.explore_step)

        # --- Visualization: robot trail (nav_msgs/Path, native RViz Path display) ---
        self.path_pub = self.create_publisher(Path, 'frontier_explorer/robot_path', 10)
        self.trail_path = Path()
        self.trail_path.header.frame_id = self.global_frame
        self.trail_max_poses = 20000  # cap so this doesn't grow unbounded over a long run
        self.trail_timer = self.create_timer(0.5, self.record_trail)  # 2Hz, smoother than the 2s planning tick

        # --- Visualization: planned viewpoints / tree detections / route (MarkerArray) ---
        self.marker_pub = self.create_publisher(MarkerArray, 'frontier_explorer/markers', 10)
        self.last_candidates = []  # most recent frontier candidate scoring, kept for visualization only

        self.get_logger().info(
            f'Frontier explorer up. Listening on {self.map_topic}. '
            f'Bounds x[{self.bounds_min_x},{self.bounds_max_x}] '
            f'y[{self.bounds_min_y},{self.bounds_max_y}]')

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

    def explore_step(self):
        if self.exploring:
            return  # a NavigateToPose goal is already in flight

        # Drain the current visitation queue before doing any new detection work --
        # this is what makes the robot commit to a locally coherent sweep instead of
        # re-planning myopically after every single goal.
        if self.goal_queue:
            next_goal = self.goal_queue.pop(0)
            self.send_nav_goal(next_goal)
            return

        if self.latest_map is None:
            self.get_logger().info('waiting for map...', throttle_duration_sec=5.0)
            return

        robot_xy = self.get_robot_pose_xy()
        if robot_xy is None:
            return

        grid_msg = self.latest_map
        w, h = grid_msg.info.width, grid_msg.info.height
        data = np.array(grid_msg.data, dtype=np.int8).reshape(h, w)

        # Stand-in perception pipeline: pick up newly-seen "trees" (obstacle
        # clusters) every cycle, regardless of frontier state, and queue their
        # inspection viewpoints as dummy targets.
        self.scan_for_new_trees(data, grid_msg)

        clusters = self.detect_frontier_clusters(grid_msg, data)
        if not clusters:
            self.last_candidates = []
            if self.dummy_targets:
                ordered_nodes = self.solve_viewpoint_order([], robot_xy)
                self.goal_queue = [(x, y, yaw) for (x, y, _bonus, yaw) in ordered_nodes]
                self.get_logger().info(
                    f'No frontier clusters remain, but {len(self.dummy_targets)} '
                    f'dummy target(s) still pending -- routing to them.')
                self.publish_planning_markers(robot_xy)
                return
            self.announce_complete(data, 'no frontier clusters remain')
            self.publish_planning_markers(robot_xy)
            return

        candidates = self.score_candidates(clusters, data, grid_msg, robot_xy)
        self.last_candidates = candidates
        if not candidates:
            if self.dummy_targets:
                ordered_nodes = self.solve_viewpoint_order([], robot_xy)
                self.goal_queue = [(x, y, yaw) for (x, y, _bonus, yaw) in ordered_nodes]
                self.publish_planning_markers(robot_xy)
                return
            self.stall_count += 1
            if self.stall_count >= self.stall_limit:
                self.announce_complete(
                    data, f'{len(clusters)} frontier cluster(s) remain but none have '
                          f'been reachable/selectable for {self.stall_count} consecutive checks')
            self.publish_planning_markers(robot_xy)
            return
        self.stall_count = 0
        self.declared_complete = False

        # Take the top-K by utility, then solve a real TSP ordering (OR-Tools)
        # over the robot's current position + these candidates + any injected
        # dummy targets -- this is the actual coverage-path-guided visitation
        # ordering, not just a greedy nearest-neighbor approximation.
        top = sorted(candidates, key=lambda c: c['utility'], reverse=True)[:self.top_k_candidates]
        # Rescale each candidate's info-gain into a small, bounded, meters-equivalent
        # "tsp_bonus" so it nudges the route toward higher-gain viewpoints without
        # ever being able to outweigh real travel distance (see MAX_FRONTIER_BONUS_M).
        # Using raw gain here (not 'utility', which already divides by distance --
        # dividing by distance twice would double-penalize far candidates).
        if top:
            gains = [c['gain'] for c in top]
            max_gain = max(gains) if max(gains) > 0 else 1
            for c in top:
                c['tsp_bonus'] = (c['gain'] / max_gain) * MAX_FRONTIER_BONUS_M
        ordered_nodes = self.solve_viewpoint_order(top, robot_xy)
        self.goal_queue = [(x, y, yaw) for (x, y, _bonus, yaw) in ordered_nodes]

        self.get_logger().info(
            f'Planned a {len(self.goal_queue)}-viewpoint sweep from '
            f'{len(candidates)} candidates (of {len(clusters)} frontier clusters).')
        self.publish_planning_markers(robot_xy)

    def record_trail(self):
        """Append the robot's current position to a growing nav_msgs/Path so RViz
        can render its full traversal history (Add -> By topic -> robot_path -> Path)."""
        robot_xy = self.get_robot_pose_xy()
        if robot_xy is None:
            return
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
        """Publish everything needed to visually verify the planner's decisions:
        - blue spheres: all frontier candidates considered this cycle
        - orange cubes: detected 'tree' centroids (stand-in perception results) --
          compare these against real tree trunk positions in Gazebo to sanity-check
          the detection
        - green arrows: pending tree-inspection viewpoints (dummy_targets), oriented
          to show which way they face
        - magenta line + arrows: the current planned goal_queue, in visiting order
        Re-published fresh every cycle with a DELETEALL first so nothing goes stale
        (we got bitten by stale markers earlier today -- this avoids that)."""
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        now = self.get_clock().now().to_msg()
        mid = 0

        def make_sphere(x, y, r, g, b, scale=0.25):
            nonlocal mid
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = now
            m.ns = 'candidates'
            m.id = mid; mid += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9
            return m

        def make_cube(x, y, r, g, b, scale=0.3):
            nonlocal mid
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = now
            m.ns = 'trees'
            m.id = mid; mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9
            return m

        def make_arrow(x, y, yaw, r, g, b, ns, length=0.4):
            nonlocal mid
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = now
            m.ns = ns
            m.id = mid; mid += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.25
            m.pose.orientation.z = math.sin(yaw / 2.0)
            m.pose.orientation.w = math.cos(yaw / 2.0)
            m.scale.x = length
            m.scale.y = 0.08
            m.scale.z = 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.95
            return m

        # frontier candidates considered this cycle (blue)
        for c in self.last_candidates:
            markers.markers.append(make_sphere(c['x'], c['y'], 0.2, 0.4, 1.0))

        # detected tree centroids (orange) -- compare against real trunks in Gazebo
        for (tx, ty) in self.processed_trees:
            markers.markers.append(make_cube(tx, ty, 1.0, 0.5, 0.0))

        # pending tree-inspection viewpoints (green arrows, oriented)
        for (gx, gy, bonus, yaw) in self.dummy_targets:
            markers.markers.append(make_arrow(gx, gy, yaw, 0.0, 1.0, 0.0, 'tree_viewpoints'))

        # planned route: robot -> each queued goal in order (magenta line + arrows)
        if self.goal_queue:
            line = Marker()
            line.header.frame_id = self.global_frame
            line.header.stamp = now
            line.ns = 'planned_route'
            line.id = mid; mid += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.06
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.0, 1.0, 0.9
            line.pose.orientation.w = 1.0
            line.points.append(Point(x=robot_xy[0], y=robot_xy[1], z=0.1))
            for (gx, gy, gyaw) in self.goal_queue:
                line.points.append(Point(x=gx, y=gy, z=0.1))
                markers.markers.append(make_arrow(gx, gy, gyaw, 1.0, 0.0, 1.0, 'planned_route'))
            markers.markers.append(line)

        self.marker_pub.publish(markers)

    def compute_coverage_pct(self, data):
        """Fraction of the grid that is known (free or occupied) vs still unknown --
        the real ground-truth completion metric, independent of frontier-cluster edge cases."""
        total = data.size
        if total == 0:
            return 0.0
        known = np.count_nonzero(data != UNKNOWN)
        return 100.0 * known / total

    def announce_complete(self, data, reason):
        if self.declared_complete:
            return  # already announced, don't spam every 2s
        self.declared_complete = True
        pct = self.compute_coverage_pct(data)
        self.get_logger().info(
            f'Exploration complete: {reason}. '
            f'{self.visited_frontier_count} viewpoint(s) visited. '
            f'Map coverage (known/unknown cells within current grid): {pct:.1f}%.')

    # ---------- stand-in perception: treat obstacle clusters as "detected trees" ----------

    def detect_obstacle_clusters(self, data, grid_msg):
        """Same connected-component approach as detect_frontier_clusters, but over
        OCCUPIED cells instead of frontier cells. This is the stand-in for a real
        perception pipeline's tree-trunk detector -- swap this out for an actual
        detection callback later; everything downstream (viewpoint generation,
        TSP injection) stays the same either way.

        Cluster growth is capped at MAX_TREE_CLUSTER_DIAMETER_M: a cell whose
        addition would push the cluster's bounding box past that span is left
        unvisited rather than absorbed, so it seeds its own separate cluster on
        a later iteration. Without this, trees close enough together to touch
        in the occupancy grid would merge into one giant blob and get treated
        as a single "tree" -- this splits them back into per-trunk detections."""
        h, w = data.shape
        res = grid_msg.info.resolution
        max_span_cells = max(1, int(round(MAX_TREE_CLUSTER_DIAMETER_M / res)))
        occ_mask = (data >= OCC_THRESHOLD)
        visited = np.zeros_like(occ_mask)
        clusters = []
        ys, xs = np.nonzero(occ_mask)
        occ_cells = set(zip(ys.tolist(), xs.tolist()))

        for start in occ_cells:
            if visited[start]:
                continue
            q = deque([start])
            visited[start] = True
            cluster = [start]
            min_row = max_row = start[0]
            min_col = max_col = start[1]
            while q:
                cy, cx = q.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if (ny, nx) not in occ_cells or visited[ny, nx]:
                            continue
                        new_min_row, new_max_row = min(min_row, ny), max(max_row, ny)
                        new_min_col, new_max_col = min(min_col, nx), max(max_col, nx)
                        if (new_max_row - new_min_row > max_span_cells or
                                new_max_col - new_min_col > max_span_cells):
                            continue  # would grow this cluster too big -- leave it
                                      # unvisited so it seeds a separate cluster instead
                        visited[ny, nx] = True
                        cluster.append((ny, nx))
                        min_row, max_row, min_col, max_col = new_min_row, new_max_row, new_min_col, new_max_col
                        q.append((ny, nx))
            if len(cluster) >= MIN_TREE_CLUSTER_CELLS:
                clusters.append(cluster)
        return clusters

    def is_known_tree(self, tx, ty):
        for (px, py) in self.processed_trees:
            if math.hypot(tx - px, ty - py) < TREE_DEDUP_RADIUS_M:
                return True
        return False

    def generate_tree_viewpoints(self, data, grid_msg, tx, ty):
        """Generate up to 3 viewpoints spaced 120 degrees apart around a detected
        tree at (tx, ty), each pulled back to TREE_INSPECTION_RADIUS_M and checked
        for real clearance -- a tree with a close neighbor may only yield 1 or 2
        valid viewpoints instead of 3, which is fine, we just skip the blocked ones."""
        viewpoints = []
        for k in range(3):
            angle = k * (2.0 * math.pi / 3.0)
            gx = tx + TREE_INSPECTION_RADIUS_M * math.cos(angle)
            gy = ty + TREE_INSPECTION_RADIUS_M * math.sin(angle)
            if self.in_bounds(gx, gy) and self.has_clearance(data, grid_msg, gx, gy, MIN_OBSTACLE_CLEARANCE_M):
                yaw = math.atan2(ty - gy, tx - gx)  # face into the tree
                viewpoints.append((gx, gy, yaw))
        return viewpoints

    def scan_for_new_trees(self, data, grid_msg):
        """Run every cycle: find obstacle clusters the explorer hasn't seen before,
        generate their 3 circular inspection viewpoints, and inject them as dummy
        targets. Rate-limited to MAX_NEW_TREES_PER_CYCLE so a big already-mapped
        area doesn't dump dozens of targets into the solver in one tick."""
        if len(self.dummy_targets) >= MAX_DUMMY_TARGETS:
            return  # already have plenty queued, work through the backlog first
        clusters = self.detect_obstacle_clusters(data, grid_msg)
        new_trees_added = 0
        for cluster in clusters:
            if new_trees_added >= MAX_NEW_TREES_PER_CYCLE:
                break
            tx, ty = self.cluster_centroid_world(cluster, grid_msg)
            if self.is_known_tree(tx, ty):
                continue
            self.processed_trees.append((tx, ty))
            viewpoints = self.generate_tree_viewpoints(data, grid_msg, tx, ty)
            for (gx, gy, yaw) in viewpoints:
                self.dummy_targets.append((gx, gy, TREE_VIEWPOINT_BONUS, yaw))
            new_trees_added += 1
            if viewpoints:
                self.get_logger().info(
                    f'Detected tree at ({tx:.2f}, {ty:.2f}) -- queued '
                    f'{len(viewpoints)}/3 inspection viewpoint(s) around it.')

    # ---------- frontier detection ----------

    def detect_frontier_clusters(self, grid_msg: OccupancyGrid, data):
        h, w = data.shape
        free_mask = (data == FREE)
        unknown_mask = (data == UNKNOWN)

        frontier_mask = np.zeros_like(free_mask)
        padded = np.pad(unknown_mask, 1, mode='constant', constant_values=False)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                frontier_mask |= (free_mask & shifted)

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

    def has_clearance(self, data, grid_msg, wx, wy, min_clearance_m):
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        h, w = data.shape
        col = int((wx - ox) / res)
        row = int((wy - oy) / res)
        r_cells = max(1, int(math.ceil(min_clearance_m / res)))
        r0, r1 = max(0, row - r_cells), min(h, row + r_cells + 1)
        c0, c1 = max(0, col - r_cells), min(w, col + r_cells + 1)
        if r1 <= r0 or c1 <= c0:
            return False
        window = data[r0:r1, c0:c1]
        return not np.any(window >= OCC_THRESHOLD)

    def find_standoff_goal(self, data, grid_msg, cluster, robot_xy):
        """Search for a safe standoff point near the frontier centroid, pulled back
        into free space. Tries multiple angular directions around the centroid (not
        just straight back toward the robot) so a frontier in a narrow gap between
        obstacles still gets a fair chance -- a single blocked line-of-retreat no
        longer kills the whole cluster."""
        wx, wy = self.cluster_centroid_world(cluster, grid_msg)
        rx, ry = robot_xy
        dx, dy = rx - wx, ry - wy
        dist_to_robot = math.hypot(dx, dy)
        base_angle = math.atan2(dy, dx) if dist_to_robot > 1e-3 else 0.0

        # try the direct line back toward the robot first (cheapest, usually best),
        # then fan out to either side in case that line is blocked but the gap
        # is still passable from a slightly different angle
        angle_offsets = [0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6]
        standoffs = (GOAL_STANDOFF_M, GOAL_STANDOFF_M * 0.6, GOAL_STANDOFF_M * 0.3, 0.0)

        for angle_off in angle_offsets:
            angle = base_angle + angle_off
            ux, uy = math.cos(angle), math.sin(angle)
            for standoff in standoffs:
                gx = wx + ux * standoff
                gy = wy + uy * standoff
                if self.in_bounds(gx, gy) and self.has_clearance(data, grid_msg, gx, gy, MIN_OBSTACLE_CLEARANCE_M):
                    yaw = math.atan2(wy - gy, wx - gx)
                    return (gx, gy, yaw)
        return None

    def cluster_centroid_world(self, cluster, grid_msg: OccupancyGrid):
        res = grid_msg.info.resolution
        ox = grid_msg.info.origin.position.x
        oy = grid_msg.info.origin.position.y
        ys = [c[0] for c in cluster]
        xs = [c[1] for c in cluster]
        mean_row = sum(ys) / len(ys)
        mean_col = sum(xs) / len(xs)
        wx = ox + (mean_col + 0.5) * res
        wy = oy + (mean_row + 0.5) * res
        return wx, wy

    # ---------- information-gain scoring ----------

    def compute_information_gain(self, data, grid_msg, wx, wy):
        """Raycast a synthetic 360-degree sensor sweep from (wx, wy) and count how
        many currently-UNKNOWN cells would become observable (rays stop at the
        first occupied cell or at sensor_range_m). This is a real visibility-based
        utility estimate, not just a proxy like frontier cluster pixel count."""
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
            dx = math.cos(angle)
            dy = math.sin(angle)
            for step in range(1, max_range_cells + 1):
                c = int(round(col0 + dx * step))
                r = int(round(row0 + dy * step))
                if not (0 <= r < h and 0 <= c < w):
                    break
                val = data[r, c]
                if val >= OCC_THRESHOLD:
                    break  # ray blocked by an obstacle
                if val == UNKNOWN:
                    gain += 1
        return gain

    def score_candidates(self, clusters, data, grid_msg, robot_xy):
        rx, ry = robot_xy
        candidates = []
        n_blacklisted = 0
        n_no_clearance = 0
        n_too_close = 0
        n_out_of_bounds = 0
        n_too_far = 0

        for cluster in clusters:
            cx, cy = self.cluster_centroid_world(cluster, grid_msg)
            if not self.in_bounds(cx, cy):
                n_out_of_bounds += 1
                continue
            if math.hypot(cx - rx, cy - ry) > MAX_CANDIDATE_RADIUS_M:
                n_too_far += 1
                continue  # too far to consider this cycle -- keep exploration local

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

            gain = self.compute_information_gain(data, grid_msg, gx, gy)
            utility = gain / ((dist + 0.5) ** UTILITY_DISTANCE_ALPHA)
            candidates.append({
                'x': gx, 'y': gy, 'yaw': yaw,
                'gain': gain, 'dist': dist, 'utility': utility,
                'cluster_size': len(cluster),
            })

        if not candidates and clusters:
            self.get_logger().info(
                f'{len(clusters)} frontier cluster(s) found but none selectable: '
                f'{n_out_of_bounds} outside world bounds, '
                f'{n_too_far} too far (>{MAX_CANDIDATE_RADIUS_M}m) this cycle, '
                f'{n_no_clearance} failed obstacle clearance, '
                f'{n_blacklisted} blacklisted from prior failures, '
                f'{n_too_close} already at robot position.')

        return candidates

    def build_cost_matrix(self, nodes):
        """Build the N x N cost matrix handed to the TSP solver. nodes[0] is always
        the robot's current position (the depot). Every other node is a candidate
        viewpoint: (x, y, bonus, yaw). `bonus` is subtracted from the cost of every
        edge *arriving* at that node -- this is the direct, hand-editable lever for
        prioritizing specific targets (dummy or otherwise): raise a node's bonus
        and the solver will bias its route to reach that node sooner / more
        directly, without you having to hand-craft the route yourself.

        EDIT HERE if you want a different cost model -- e.g. swap the Euclidean
        distance() call below for a real path-length query (Nav2's ComputePathToPose
        service) if straight-line distance is too optimistic around obstacles.
        """
        n = len(nodes)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            xi, yi, _, _ = nodes[i]
            for j in range(n):
                if i == j:
                    continue
                xj, yj, bonus_j, _ = nodes[j]
                dist = math.hypot(xj - xi, yj - yi)
                cost = dist - bonus_j
                matrix[i][j] = max(0, int(round(cost * 1000)))  # ortools wants integers
        return matrix

    def solve_viewpoint_order(self, candidates, robot_xy):
        """Real TSP solve (Google OR-Tools) over the robot's current position plus
        every candidate viewpoint (plus any DUMMY_TARGETS), replacing the old
        greedy nearest-neighbor chain. This is an OPEN path (robot does not need
        to return to its start), implemented via the standard trick of zeroing
        the cost of every edge back into the depot."""
        nodes = [(robot_xy[0], robot_xy[1], 0.0, 0.0)]
        for c in candidates:
            nodes.append((c['x'], c['y'], c.get('tsp_bonus', 0.0), c['yaw']))
        for (tx, ty, tbonus, tyaw) in self.dummy_targets:
            if self.is_blacklisted(tx, ty):
                continue  # this exact point already failed -- don't resend it forever
            nodes.append((tx, ty, tbonus, tyaw))

        n = len(nodes)
        if n <= 1:
            return []
        if n == 2:
            return [nodes[1]]

        matrix = self.build_cost_matrix(nodes)
        # open-path trick: make returning to the depot free, so the solver isn't
        # penalized for "coming home" -- we only care about the outbound tour
        for i in range(n):
            matrix[i][0] = 0

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # n nodes, 1 vehicle, depot=0
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return matrix[from_node][to_node]

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_params.time_limit.FromSeconds(1)  # candidate sets are small (<=~10), no need for more

        solution = routing.SolveWithParameters(search_params)
        if solution is None:
            self.get_logger().warn('OR-Tools TSP solve failed, falling back to input order')
            return [n for n in nodes[1:]]

        order = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                order.append(nodes[node])
            index = solution.Value(routing.NextVar(index))
        return order

    def is_blacklisted(self, x, y):
        for (bx, by) in self.failed_goals:
            if math.hypot(x - bx, y - by) < REPLAN_ON_FAILURE_BLACKLIST_M:
                return True
        return False

    # ---------- nav2 goal dispatch ----------

    def send_nav_goal(self, xyyaw):
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('navigate_to_pose action server not available yet')
            return

        wx, wy, yaw = xyyaw
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = wx
        pose.pose.position.y = wy
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.exploring = True
        self.get_logger().info(
            f'Sending exploration goal: ({wx:.2f}, {wy:.2f}, yaw={math.degrees(yaw):.0f}deg) '
            f'[{len(self.goal_queue)} more queued]')
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.goal_response_cb(f, (wx, wy)))

    def goal_response_cb(self, future, xy):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by nav2')
            self.failed_goals.append(xy)
            self.exploring = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.goal_result_cb(f, xy))

    def goal_result_cb(self, future, xy):
        status = future.result().status
        gx, gy = xy
        if status != 4:  # 4 = SUCCEEDED
            self.get_logger().warn(f'Goal to {xy} did not succeed (status={status}), blacklisting')
            self.failed_goals.append(xy)
            self.goal_queue.clear()  # abandon the rest of this sweep, replan fresh next step
            # remove it from dummy_targets too -- a permanently unreachable viewpoint
            # shouldn't sit there forever as dead weight
            self.dummy_targets = [
                t for t in self.dummy_targets if math.hypot(t[0] - gx, t[1] - gy) > 0.3
            ]
        else:
            self.get_logger().info(f'Reached frontier goal {xy}')
            self.visited_frontier_count += 1
            self.dummy_targets = [
                t for t in self.dummy_targets if math.hypot(t[0] - gx, t[1] - gy) > 0.3
            ]
        self.exploring = False


def main():
    rclpy.init()
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

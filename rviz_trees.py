#!/usr/bin/env python3
"""RViz bridge for live_detection.py -- reads nothing from its process.

live_detection.py has no publishers, so nothing in it reaches RViz directly.
It does, however, already dump three files per detection cycle into
debug_ccl/ (Cfg.debug_ccl_every = 1):

    ccl_NNNNN.npz       occ/labels/sup rasters + i_min, j_min, voxel,
                        sensor_xy, comps = [label, n_cells, support,
                        cx, cy, radius, verdict]
    registry_NNNNN.npy  the whole global list: [gid, x, y, r, status,
                        discovery_order, n_obs]
    events_NNNNN.json   frame, sensor_xy, det_radius, match/new/merge events

This node tails that directory and republishes it as markers, so
live_detection.py needs no change at all. It therefore lags by one detection
cycle -- it is for watching, not for closing a control loop.

Topics (all in frame `map`, which is the frame GLIM publishes poses in):

    /trees/map          PointCloud2   slab cells, accumulated over cycles
    /trees/detections   MarkerArray   registry circles + "#gid n=obs"
    /trees/truth        MarkerArray   static-map ground truth, if available
    /trees/error        MarkerArray   detection -> nearest true tree, labelled
                                      with the distance in metres
    /trees/sensor       MarkerArray   sensor position + the det_radius window
    /trees/stats        MarkerArray   running count / mean / max error

Run it alongside the pipeline:

    python3 rviz_trees.py                      # tail: ignore dumps already there
    python3 rviz_trees.py --ros-args -p mode:=replay -p replay_hz:=4.0

In RViz set **Fixed Frame = map**, then add the topics above. The markers are
published TRANSIENT_LOCAL, so RViz picks up the latest state even if you
connect it after the run has started.
"""

import glob
import json
import os
import re
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Duration
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

HERE = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(HERE, "debug_ccl")
# ground truth is per dataset: ground_truth_<name>.npy, selected with the
# `truth` parameter. "none" turns it off, which is what a Gazebo run wants --
# the forest_56 tree positions mean nothing in a different world.
def truth_path(name):
    return os.path.join(HERE, "ground_truth_%s.npy" % name)

# must match live_detection.py
ACCEPTED = 0
UNVISITED, VISITED = 0, 1

FRAME = "map"
Z_MAP = 0.0        # the dumps carry no per-cell height, so the map is drawn flat
Z_MARK = 0.05      # rings sit just above the cloud so they are not z-fought
Z_TOP = 0.20       # centroid glyphs and error lines sit above the rings
Z_TEXT = 0.40      # labels above everything


def rgba(r, g, b, a=1.0):
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


# Two palettes: RViz's default background is dark (48;48;48) but the Isaac Sim
# config uses a light one (222;221;218), where white text is invisible and pale
# cyan washes out. `theme` picks between them and rewrites these globals.
DARK = dict(
    C_MAP=rgba(0.55, 0.60, 0.68, 0.85),    # occupancy cells
    C_DET=rgba(0.20, 0.85, 1.00, 0.85),    # detected trunk ring, UNVISITED
    C_VIS=rgba(0.30, 0.95, 0.45, 0.85),    # detected trunk ring, VISITED
    C_DET_M=rgba(0.30, 0.95, 1.00, 1.00),  # detected centroid PLUS
    C_TRU=rgba(1.00, 0.75, 0.15, 0.75),    # ground-truth ring
    C_TRU_M=rgba(1.00, 0.85, 0.20, 1.00),  # ground-truth centroid CROSS
    C_ERR=rgba(1.00, 0.35, 0.35, 1.00),    # error line
    C_TXT=rgba(1.00, 1.00, 1.00, 1.00),
    C_SEN=rgba(1.00, 0.95, 0.25, 1.00),
    C_REJ=rgba(0.45, 0.45, 0.50, 0.35),    # slab cells of a REJECTED component
    PAL_V=1.00, PAL_S=0.62,
)
LIGHT = dict(
    C_MAP=rgba(0.42, 0.45, 0.52, 0.85),
    C_DET=rgba(0.00, 0.40, 0.85, 0.95),
    C_VIS=rgba(0.00, 0.50, 0.18, 0.95),
    C_DET_M=rgba(0.00, 0.30, 0.80, 1.00),
    C_TRU=rgba(0.85, 0.45, 0.00, 0.85),
    C_TRU_M=rgba(0.70, 0.33, 0.00, 1.00),
    C_ERR=rgba(0.80, 0.05, 0.10, 1.00),
    C_TXT=rgba(0.08, 0.08, 0.10, 1.00),
    C_SEN=rgba(0.60, 0.45, 0.00, 1.00),
    C_REJ=rgba(0.35, 0.35, 0.40, 0.30),
    PAL_V=0.68, PAL_S=0.85,
)
globals().update(DARK)


def palette(k):
    """Distinct colour per component. Golden-ratio hue stepping, so adjacent
    label ids never get adjacent hues."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((k * 0.6180339887) % 1.0, PAL_S, PAL_V)
    return rgba(r, g, b, 0.80)


CYCLE_RE = re.compile(r"ccl_(\d+)\.npz$")


def cycle_of(path):
    m = CYCLE_RE.search(path)
    return int(m.group(1)) if m else -1


class RvizTrees(Node):

    def __init__(self):
        super().__init__("rviz_trees")

        self.declare_parameter("mode", "live")       # live | replay
        self.declare_parameter("replay_hz", 4.0)
        self.declare_parameter("poll_hz", 4.0)
        self.declare_parameter("match_max", 5.0)     # m, beyond this a detection
                                                     # is called unmatched
        self.mode = self.get_parameter("mode").value
        self.match_max = float(self.get_parameter("match_max").value)
        self.declare_parameter("mark_size", 0.7)     # m, half-length of the
        self.mark = float(self.get_parameter("mark_size").value)   # + and x arms
        self.declare_parameter("truth", "forest_56")   # dataset name, or "none"
        self.declare_parameter("slab_z", 0.4)          # m, centre of the drawn slab
        self.declare_parameter("slab_h", 0.97)         # m, z_max - z_min
        self.slab_z = float(self.get_parameter("slab_z").value)
        self.slab_h = float(self.get_parameter("slab_h").value)
        self.declare_parameter("session_gap", 120.0)   # s, see newest_session()
        self.declare_parameter("theme", "dark")        # dark | light
        # The sensor dot and det_radius ring are the only markers that should
        # track the robot CONTINUOUSLY. Published in `map` they can only move
        # when a new dump lands -- once per detect_every scans, which at Isaac's
        # RTF is tens of wall-clock seconds, so they look nailed to the spawn
        # point. Published in the robot's own frame at the origin instead, RViz
        # moves them via TF at TF rate. Set sensor_frame:="" to go back to map.
        self.declare_parameter("sensor_frame", "unitree_go2/lidar_frame")
        # "map"/"none"/"" all mean: draw the sensor markers in the GLIM map frame
        # using sensor_xy from the dumps. An empty string is awkward to pass through
        # a shell (`-p sensor_frame:=""` arrives as `-p sensor_frame:=` and rcl
        # rejects it), so the words are the usable form.
        self.sensor_frame = str(self.get_parameter("sensor_frame").value)
        if self.sensor_frame.lower() in ("map", "none", ""):
            self.sensor_frame = ""
        theme = str(self.get_parameter("theme").value).lower()
        globals().update(LIGHT if theme == "light" else DARK)

        # TRANSIENT_LOCAL so RViz gets the current state when it subscribes late
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_map = self.create_publisher(PointCloud2, "/trees/map", qos)
        self.pub_det = self.create_publisher(MarkerArray, "/trees/detections", qos)
        self.pub_tru = self.create_publisher(MarkerArray, "/trees/truth", qos)
        self.pub_err = self.create_publisher(MarkerArray, "/trees/error", qos)
        self.pub_sen = self.create_publisher(MarkerArray, "/trees/sensor", qos)
        self.pub_sta = self.create_publisher(MarkerArray, "/trees/stats", qos)
        self.pub_slab = self.create_publisher(MarkerArray, "/trees/slab", qos)

        # ground truth is optional: without it the error overlay is simply empty
        self.truth = None
        name = str(self.get_parameter("truth").value)
        if name.lower() in ("none", "", "off"):
            self.get_logger().info(
                "truth:=none -- no ground-truth overlay and no error lines "
                "(use this for Gazebo, where forest_56 positions do not apply)")
        elif os.path.exists(truth_path(name)):
            self.truth = np.load(truth_path(name)).reshape(-1, 3)
            self.get_logger().info(
                f"ground truth '{name}': {len(self.truth)} trees")
        else:
            self.get_logger().warn(
                f"truth:={name} but {os.path.basename(truth_path(name))} does "
                f"not exist -- continuing with no ground truth")
        self.pub_tru.publish(self.truth_markers())

        # accumulated occupancy, as a set of (i, j) raster cells in ABSOLUTE
        # voxel indices, so windows from different cycles overlay correctly
        self.cells = set()
        self.voxel = 0.10
        # path -> mtime of the version already consumed. Keyed on mtime, NOT on
        # path alone: live_detection.py restarts cycle numbering at 1 every run and
        # overwrites ccl_00001.npz onward, so a path-only set makes every dump of
        # the second run look "already seen" and nothing is ever published.
        self.seen = {}
        self.started = time.time()

        if self.mode == "replay":
            self.queue = self.newest_session()
            self.get_logger().info(f"replay: {len(self.queue)} dumps queued")
            hz = float(self.get_parameter("replay_hz").value)
            self.timer = self.create_timer(1.0 / max(hz, 0.1), self.step_replay)
        else:
            # ignore whatever is already on disk: those are previous runs, and
            # cycle numbering restarts each run so old files would replay first
            n_old = len(glob.glob(os.path.join(DEBUG_DIR, "ccl_*.npz")))
            self.get_logger().info(
                f"live: ignoring {n_old} dump(s) written before now, waiting for "
                f"fresh ones in {DEBUG_DIR}")
            hz = float(self.get_parameter("poll_hz").value)
            self.timer = self.create_timer(1.0 / max(hz, 0.1), self.step_live)

    def newest_session(self):
        """Dumps from the most recent run only, in cycle order.

        live_detection.py never clears debug_ccl/ and restarts cycle numbering
        at 1 every run, so a short run leaves the tail of a longer earlier one
        behind: replaying by cycle number walks off the end of this run and
        into an older one, and the registry appears to collapse. Group the
        files by modification time instead and keep only the newest cluster --
        a gap longer than session_gap means a different run.
        """
        files = glob.glob(os.path.join(DEBUG_DIR, "ccl_*.npz"))
        if not files:
            return []
        gap = float(self.get_parameter("session_gap").value)
        by_time = sorted(files, key=os.path.getmtime, reverse=True)
        session = [by_time[0]]
        for prev, cur in zip(by_time, by_time[1:]):
            if os.path.getmtime(prev) - os.path.getmtime(cur) > gap:
                break
            session.append(cur)
        if len(session) < len(files):
            import time as _t
            self.get_logger().warn(
                "ignoring %d dump(s) from earlier runs still in debug_ccl/ "
                "(newest run: %d cycles, written %s)"
                % (len(files) - len(session), len(session),
                   _t.strftime("%Y-%m-%d %H:%M",
                               _t.localtime(os.path.getmtime(by_time[0])))))
        return sorted(session, key=cycle_of)

    # ------------------------------------------------------------------ #
    def step_replay(self):
        if not self.queue:
            return
        self.consume(self.queue.pop(0))

    def step_live(self):
        for p in sorted(glob.glob(os.path.join(DEBUG_DIR, "ccl_*.npz")), key=cycle_of):
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if mt <= self.started:          # left over from an earlier run
                continue
            if self.seen.get(p) == mt:      # this version already published
                continue
            self.seen[p] = mt
            self.consume(p)

    # ------------------------------------------------------------------ #
    def consume(self, path):
        # NB: not `handle` -- rclpy.node.Node.handle is a property, and
        # shadowing it makes Node.__init__ fail with AttributeError: __enter__
        try:
            c = np.load(path)
            occ = c["occ"]
            labels = c["labels"]
            i_min, j_min = int(c["i_min"]), int(c["j_min"])
            self.voxel = float(c["voxel"])
            sensor_xy = np.asarray(c["sensor_xy"], dtype=np.float64)
            comps = np.asarray(c["comps"], dtype=np.float64).reshape(-1, 7)
        except Exception as exc:           # a dump caught mid-write
            self.get_logger().debug(f"skipping {os.path.basename(path)}: {exc}")
            return

        cyc = cycle_of(path)
        reg = self.load_sibling(cyc, "registry_%05d.npy")
        meta = self.load_sibling(cyc, "events_%05d.json") or {}

        # accumulate this window's occupied cells into the running map
        r, cc = np.nonzero(occ)
        self.cells.update(zip((r + i_min).tolist(), (cc + j_min).tolist()))

        self.pub_map.publish(self.map_cloud())
        self.pub_slab.publish(self.slab_markers(labels, comps, i_min, j_min))
        self.pub_sen.publish(self.sensor_markers(sensor_xy, meta.get("det_radius")))
        if reg is not None and len(reg):
            self.pub_det.publish(self.detection_markers(reg))
            err = self.pair(reg)
            self.pub_err.publish(self.error_markers(err))
            self.pub_sta.publish(self.stats_markers(cyc, meta, reg, err))
        n_acc = int((comps[:, 6] == ACCEPTED).sum()) if len(comps) else 0
        self.get_logger().info(
            "cycle %d (frame %s): %d accepted, %d in registry, map %d cells"
            % (cyc, meta.get("frame", "?"), n_acc,
               0 if reg is None else len(reg), len(self.cells)))

    def load_sibling(self, cyc, pattern):
        p = os.path.join(DEBUG_DIR, pattern % cyc)
        if not os.path.exists(p):
            return None
        try:
            if p.endswith(".json"):
                with open(p) as fh:
                    return json.load(fh)
            return np.load(p).reshape(-1, 7)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def map_cloud(self):
        """Accumulated slab cells as a flat cloud in world metres."""
        if not self.cells:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            a = np.fromiter((v for c in self.cells for v in c),
                            dtype=np.int64, count=2 * len(self.cells))
            a = a.reshape(-1, 2)
            pts = np.empty((len(a), 3), dtype=np.float32)
            pts[:, 0] = (a[:, 0] + 0.5) * self.voxel
            pts[:, 1] = (a[:, 1] + 0.5) * self.voxel
            pts[:, 2] = Z_MAP
        hdr = self.header()
        return pc2.create_cloud_xyz32(hdr, pts.tolist())

    def slab_markers(self, labels, comps, i_min, j_min):
        """The height slab itself, as one box per occupied cell.

        Each connected component gets its own colour, so you can see which
        blob of slab voxels became which trunk, and rejected components stay
        grey -- that is what tells you whether a centroid sits on the tree or
        beside it. Boxes are voxel x voxel x slab_h so the band has real
        thickness; the dumps carry no per-cell ground height, so the slab is
        drawn at a fixed slab_z rather than following the DTM.
        """
        out = MarkerArray()
        out.markers.append(self.clear())
        if not len(comps):
            return out
        for row in comps:
            label_id, verdict = int(row[0]), int(row[6])
            cells = np.argwhere(labels == label_id)
            if not len(cells):
                continue
            m = self.base("slab", label_id, Marker.CUBE_LIST)
            m.scale = Vector3(x=self.voxel, y=self.voxel, z=self.slab_h)
            m.color = palette(label_id) if verdict == ACCEPTED else C_REJ
            m.pose.position = Point(x=0.0, y=0.0, z=0.0)
            for r, cc in cells:
                m.points.append(Point(
                    x=float((r + i_min + 0.5) * self.voxel),
                    y=float((cc + j_min + 0.5) * self.voxel),
                    z=float(self.slab_z)))
            out.markers.append(m)
        return out

    def detection_markers(self, reg):
        """Ring at the bounding radius, a PLUS at the centroid, and a label."""
        out = MarkerArray()
        out.markers.append(self.clear())
        for k, (gid, x, y, rad, status, _disc, n_obs) in enumerate(reg):
            col = C_VIS if int(status) == VISITED else C_DET
            out.markers.append(self.ring("det_ring", k, x, y, rad, col, 0.04))
            out.markers.append(self.glyph("det_plus", k, x, y, C_DET_M, plus=True))
            out.markers.append(self.text("det_txt", k, x, y + self.mark + 0.35,
                                         "#%d n=%d" % (int(gid), int(n_obs)),
                                         0.45, C_DET_M))
        return out

    def truth_markers(self):
        """Ring at the stem radius, a CROSS at the centroid, and a label."""
        out = MarkerArray()
        out.markers.append(self.clear())
        if self.truth is None:
            return out
        for k, (x, y, rad) in enumerate(self.truth):
            out.markers.append(self.ring("tru_ring", k, x, y, rad, C_TRU, 0.03))
            out.markers.append(self.glyph("tru_cross", k, x, y, C_TRU_M, plus=False))
            out.markers.append(self.text("tru_txt", k, x, y - self.mark - 0.45,
                                         "T%d" % k, 0.45, C_TRU_M))
        return out

    def pair(self, reg):
        """Nearest ground-truth tree for every registry entry.

        Returns a list of (gid, dx, dy, tx, ty, distance, truth_index).
        Nearest-neighbour, not one-to-one: two detections of the same trunk
        should both point at it rather than one being forced onto a different
        tree. The stats line counts DISTINCT truth indices, so a duplicated
        trunk does not read as two trees found.
        """
        if self.truth is None or not len(reg):
            return []
        out = []
        for gid, x, y, _r, _s, _d, _n in reg:
            d = np.hypot(self.truth[:, 0] - x, self.truth[:, 1] - y)
            k = int(d.argmin())
            out.append((int(gid), float(x), float(y),
                        float(self.truth[k, 0]), float(self.truth[k, 1]),
                        float(d[k]), k))
        return out

    def error_markers(self, err):
        out = MarkerArray()
        out.markers.append(self.clear())
        if not err:
            return out
        line = self.base("err_line", 0, Marker.LINE_LIST)
        line.scale = Vector3(x=0.06, y=0.0, z=0.0)
        line.color = C_ERR
        for k, (gid, x, y, tx, ty, dist, _ti) in enumerate(err):
            if dist > self.match_max:
                continue                     # spurious: no true tree near it
            line.points.append(Point(x=x, y=y, z=Z_TOP))
            line.points.append(Point(x=tx, y=ty, z=Z_TOP))
            out.markers.append(self.text(
                "err_txt", k, (x + tx) / 2.0, (y + ty) / 2.0,
                "%.2f m" % dist, 0.40, C_ERR))
        out.markers.append(line)
        return out

    def stats_markers(self, cyc, meta, reg, err):
        out = MarkerArray()
        near = [e for e in err if e[5] <= self.match_max]
        matched = [e[5] for e in near]
        found = {e[6] for e in near}          # distinct true trees accounted for
        txt = ["cycle %d   frame %s" % (cyc, meta.get("frame", "?")),
               "registry: %d" % len(reg)]
        if self.truth is not None:
            txt.append("true trees found: %d / %d" % (len(found), len(self.truth)))
            if matched:
                txt.append("error  mean %.2f m   max %.2f m"
                           % (float(np.mean(matched)), float(np.max(matched))))
            txt.append("duplicates: %d   spurious (>%.0f m): %d"
                       % (len(near) - len(found), self.match_max,
                          sum(1 for e in err if e[5] > self.match_max)))
        m = self.text("stats", 0, 0.0, 0.0, "\n".join(txt), 0.4, C_TXT)
        out.markers.append(m)
        return out

    def sensor_markers(self, xy, det_radius):
        """Sensor dot + det_radius window.

        With sensor_frame set these are drawn at the ORIGIN of the robot frame
        and carried by TF, so they follow the robot smoothly between detection
        cycles. With it empty they fall back to absolute map coordinates from
        the dump, which only step once per cycle.
        """
        out = MarkerArray()
        out.markers.append(self.clear())
        if self.sensor_frame:
            cx = cy = 0.0
        else:
            cx, cy = float(xy[0]), float(xy[1])
        m = self.base("sen_dot", 0, Marker.SPHERE, frame=self.sensor_frame)
        m.pose.position = Point(x=cx, y=cy, z=Z_TOP)
        m.scale = Vector3(x=0.5, y=0.5, z=0.5)
        m.color = C_SEN
        out.markers.append(m)
        if det_radius:
            out.markers.append(self.ring("sen_ring", 0, cx, cy, det_radius,
                                         C_SEN, 0.05, frame=self.sensor_frame))
        return out

    # ------------------------------------------------------------------ #
    def header(self, frame=None):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = frame or FRAME
        h.stamp = self.get_clock().now().to_msg()
        return h

    def base(self, ns, mid, mtype, frame=None):
        m = Marker()
        m.header = self.header(frame)
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = Duration(sec=0, nanosec=0)      # until replaced
        return m

    def clear(self):
        """One DELETEALL for the whole array.

        It carries an EMPTY namespace, which clears every namespace on the
        topic. Giving it a real namespace collides with the first marker of
        that namespace at id 0, and RViz then rejects the entire array with
        "Duplicate Marker: multiple markers with the same id" -- which is why
        nothing was drawing.
        """
        m = self.base("", 0, Marker.SPHERE)
        m.action = Marker.DELETEALL
        return m

    def ring(self, ns, mid, x, y, radius, colour, width=0.04, frame=None):
        """Circle outline, so overlapping discs stay readable and the centroid
        glyph inside them is not hidden by a filled cylinder."""
        m = self.base(ns, mid, Marker.LINE_STRIP, frame=frame)
        m.scale = Vector3(x=float(width), y=0.0, z=0.0)
        m.color = colour
        for t in np.linspace(0.0, 2.0 * np.pi, 48):
            m.points.append(Point(x=float(x + radius * np.cos(t)),
                                  y=float(y + radius * np.sin(t)), z=Z_MARK))
        return m

    def glyph(self, ns, mid, x, y, colour, plus=True):
        """PLUS (+) for a detected centroid, CROSS (x) for a true one."""
        m = self.base(ns, mid, Marker.LINE_LIST)
        m.scale = Vector3(x=0.07, y=0.0, z=0.0)
        m.color = colour
        s = self.mark
        arms = ([(-s, 0.0), (s, 0.0), (0.0, -s), (0.0, s)] if plus
                else [(-s, -s), (s, s), (-s, s), (s, -s)])
        for dx, dy in arms:
            m.points.append(Point(x=float(x + dx), y=float(y + dy), z=Z_TOP))
        return m

    def text(self, ns, mid, x, y, s, size, colour):
        m = self.base(ns, mid, Marker.TEXT_VIEW_FACING)
        m.pose.position = Point(x=float(x), y=float(y), z=Z_TEXT)
        m.scale = Vector3(x=0.0, y=0.0, z=float(size))
        m.color = colour
        m.text = s
        return m


def main():
    rclpy.init()
    node = RvizTrees()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

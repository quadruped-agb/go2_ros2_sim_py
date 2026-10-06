#!/usr/bin/env python3
"""make_gazebo_truth.py -- Gazebo ground truth for rviz_trees.py, cut to the
slab live_detection.py is currently tuned to.

The registry radius is a BOUNDING radius of whatever the slab cuts, so the true
radius depends on the band: 0.1-0.5 m is bare trunk, but move z_min above the
first branches and the same tree is a 2 m crown. This reads voxel / z_min /
z_max / r_min / r_max / n_min straight out of Cfg in live_detection.py (parsed,
not imported, so it needs no ROS), slices every tree mesh over that band and
rewrites ground_truth_gazebo.npy, so the truth overlay tracks the tuning.

Run on the HOST (the simulator container has no trimesh), then restart
rviz_trees.py -- it only reads the truth file at startup:

    python3 ~/go_sim/src/make_gazebo_truth.py
    ... rviz_trees.py --ros-args -p truth:=gazebo -p sensor_frame:=map

Outputs, next to this script:
    ground_truth_gazebo.npy   (N, 3) [x, y, r]  -- the format rviz_trees.py loads
    ground_truth_gazebo.csv   per-tree detail (species, scale, height, trunk r,
                              branch start, band r, pass/fail r_min..r_max)

Geometric truth only: the LiDAR sees the facing side of a trunk and neighbours
occlude it, so a registry radius can come out smaller than this one and its
centroid can sit toward the sensor.

Assumes flat ground at z = 0, which is what cafe.world has (visual plane at 0,
collision-grid tops at 0.02), so "height above the DTM" is just world z.
"""
import argparse
import ast
import csv
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(HERE, "gazebo_sim", "world", "cafe.world")
MODELS = os.path.join(HERE, "gazebo_sim", "models")
DETECTOR = os.path.join(HERE, "live_detection.py")
CACHE_DIR = os.path.join(HERE, "detection_output", "truth_cache")

TREE_PREFIXES = ("spruce", "fir")
CFG_KEYS = ("voxel", "z_min", "z_max", "r_min", "r_max", "n_min")
WORLD_STEP = 0.05        # m, profile resolution in WORLD height at the largest scale
PROFILE_TOP = 4.0        # m, world height the profile covers at the smallest scale
TRUNK_BAND = (0.1, 0.4)  # m, world heights that are bare trunk for every tree here
BRANCH_FACTOR = 2.0      # cross-section wider than this x trunk r = branches


def read_cfg(path):
    """Constant assignments in `class Cfg`, without importing the module."""
    tree = ast.parse(open(path).read(), path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Cfg")
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass                       # computed values, e.g. d_match
    missing = [k for k in CFG_KEYS if k not in out]
    if missing:
        raise SystemExit("Cfg in %s is missing %s" % (path, missing))
    return {k: out[k] for k in CFG_KEYS}


def read_trees(world):
    """(name, species, x, y, z, yaw, scale, mesh_path) for every tree model."""
    w = ET.parse(world).getroot().find("world")
    trees = []
    for m in w.findall("model"):
        name = m.get("name")
        if not name.startswith(TREE_PREFIXES):
            continue
        x, y, z, _, _, yaw = (float(v) for v in m.find("pose").text.split())
        mesh = m.find(".//visual/geometry/mesh")
        scale = float(mesh.find("scale").text.split()[0])
        uri = mesh.find("uri").text.strip()
        path = os.path.join(MODELS, uri.replace("model://", "", 1))
        trees.append((name, name.split("_")[0], x, y, z, yaw, scale, path))
    return trees


def mesh_profile(path, s_min, s_max):
    """Bounding radius about the trunk axis vs native mesh height, cached.

    Returns (zn, rn, axis_xy) in native mesh units. One mesh is shared by all
    trees of a species, so this is computed once per mesh, not per tree.
    """
    step = WORLD_STEP / s_max
    top = PROFILE_TOP / s_min
    key = "%s_%d_%.4f_%.2f.npz" % (os.path.basename(path),
                                   int(os.path.getmtime(path)), step, top)
    cache = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache):
        c = np.load(cache)
        return c["zn"], c["rn"], c["axis"]

    print("slicing %s (one-off, cached afterwards) ..." % os.path.basename(path))
    sc = trimesh.load(path, force="scene")
    mesh = trimesh.util.concatenate(
        [sc.geometry[sc.graph[n][1]].copy().apply_transform(sc.graph[n][0])
         for n in sc.graph.nodes_geometry])
    v = mesh.vertices
    z0 = v[:, 2].min()
    # trunk axis: centre of the lowest 2 % of the mesh, which is trunk only
    axis = v[v[:, 2] < z0 + 0.02 * (v[:, 2].max() - z0), :2].mean(axis=0)

    zn = np.arange(step / 2, top, step)
    rn = np.full(len(zn), np.nan)
    for i, z in enumerate(zn):
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is not None:
            rn[i] = np.linalg.norm(sec.vertices[:, :2] - axis, axis=1).max()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, zn=zn, rn=rn, axis=axis)
    return zn, rn, axis


def band_radius(zn, rn, scale, lo, hi):
    """Widest native cross-section between world heights lo..hi, in world m."""
    sel = (zn * scale >= lo) & (zn * scale <= hi)
    if not sel.any():                          # band thinner than one step
        sel = np.zeros(len(zn), dtype=bool)
        sel[np.argmin(np.abs(zn * scale - (lo + hi) / 2))] = True
    return float(np.nanmax(rn[sel])) * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", default=WORLD)
    ap.add_argument("--detector", default=DETECTOR,
                    help="live_detection.py whose Cfg sets the slab")
    ap.add_argument("--out", default=os.path.join(HERE, "ground_truth_gazebo"),
                    help="output path without extension (.npy and .csv)")
    args = ap.parse_args()

    cfg = read_cfg(args.detector)
    trees = read_trees(args.world)
    print("Cfg from %s: %s" % (os.path.basename(args.detector),
                               "  ".join("%s=%s" % kv for kv in cfg.items())))

    profiles = {}
    for path in sorted({t[7] for t in trees}):
        scales = [t[6] for t in trees if t[7] == path]
        profiles[path] = mesh_profile(path, min(scales), max(scales))

    rows = []
    for name, sp, x, y, z, yaw, s, path in trees:
        zn, rn, axis = profiles[path]
        # trunk axis offset, mesh frame -> world (scale, then yaw)
        c, sn = np.cos(yaw), np.sin(yaw)
        ax = np.array([c * axis[0] - sn * axis[1], sn * axis[0] + c * axis[1]]) * s
        hz = zn * s + z                         # world height of each slice
        trunk = band_radius(zn, rn, s, *TRUNK_BAND)
        wide = np.flatnonzero((hz > TRUNK_BAND[1]) & (rn * s > BRANCH_FACTOR * trunk))
        branch = float(hz[wide[0]]) if len(wide) else np.nan
        r = band_radius(zn, rn, s, cfg["z_min"] - z, cfg["z_max"] - z) + cfg["voxel"] / 2
        ok = cfg["r_min"] <= r <= cfg["r_max"]
        rows.append(dict(name=name, species=sp, x=x + ax[0], y=y + ax[1], r=r,
                         scale=s, trunk_r=trunk, branch_start_z=branch,
                         passes_r=ok, band_in_branches=bool(cfg["z_max"] > branch)))

    np.save(args.out + ".npy",
            np.array([[t["x"], t["y"], t["r"]] for t in rows], dtype=np.float64))
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for t in rows:
            w.writerow({k: ("%.3f" % v if isinstance(v, float) else v)
                        for k, v in t.items()})

    print("\nslab %.2f..%.2f m, radius filter %.2f..%.2f m (r includes +voxel/2)"
          % (cfg["z_min"], cfg["z_max"], cfg["r_min"], cfg["r_max"]))
    print("%-7s %4s  %-13s %-13s %-13s %s" % ("species", "n", "band r [m]",
                                              "trunk r [m]", "branches at",
                                              "pass r filter"))
    for sp in sorted({t["species"] for t in rows}):
        g = [t for t in rows if t["species"] == sp]
        r = np.array([t["r"] for t in g])
        tr = np.array([t["trunk_r"] for t in g])
        br = np.array([t["branch_start_z"] for t in g])
        print("%-7s %4d  %5.2f..%-5.2f   %5.2f..%-5.2f   %5.2f..%-5.2f   %d / %d"
              % (sp, len(g), r.min(), r.max(), tr.min(), tr.max(),
                 np.nanmin(br), np.nanmax(br), sum(t["passes_r"] for t in g), len(g)))
        n_br = sum(t["band_in_branches"] for t in g)
        if n_br:
            print("        ! z_max %.2f reaches into the branches of %d/%d %s "
                  "-- the slab cuts crown, not trunk" % (cfg["z_max"], n_br, len(g), sp))
    print("\nwrote %s.npy (%d trees) and .csv -- restart rviz_trees.py "
          "with -p truth:=gazebo to load it" % (args.out, len(rows)))


if __name__ == "__main__":
    main()

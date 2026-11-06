#!/usr/bin/env python3
"""terrain.py - single source of truth for terrain height. created to add slopes"""
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MESH_DIR   = os.path.join(SCRIPT_DIR, 'materials', 'meshes')
OBJ_PATH   = os.path.join(MESH_DIR, 'terrain.obj')

TERRAIN_W, TERRAIN_H = 10.0, 10.0
TEXTURE_SIZE = 4.0
N = 201                      # grid vertices per side (0.05 m spacing)
SMOOTH_PASSES = 1            # rounds off sharp ridges; 0 = crisp, exact angles
FLAT_ZONES = [(0.0, 0.0, 1.0)]   # (x, y, radius) kept flat, e.g. robot spawn
FLAT_BLEND = 0.5                 # m, fade from flat zone into terrain


# ---- EDIT THESE: each patch is a hill (ramp up, plateau, ramp down) ----
# angle = slope in degrees, ramp = horizontal run of each slope (m),
# plateau = flat top length (m), width = across-slope size (m),
# yaw = direction the slope runs along (deg). Peak height = ramp*tan(angle).
import random as _random

ANGLE_MIN, ANGLE_MAX = 10, 45    # degrees
NUM_SLOPES   = 10
SLOPE_SEED   = 100 #7,8,42,100                # change for a different layout
MAX_RISE     = 1.5               # m, cap on hill height so 30 deg stays reasonable
KEEP_OUT     = [(-4.96, 1.5, 0.8)]   # (x, y, radius) e.g. the AprilTag wall
MIN_GAP      = 0.5               # m of flat ground between hills

def _gen_slopes():
    rnd = _random.Random(SLOPE_SEED)
    out = []
    for _ in range(NUM_SLOPES):
        for _try in range(200):
            angle   = rnd.uniform(ANGLE_MIN, ANGLE_MAX)
            ramp    = min(rnd.uniform(1.0, 1.8), MAX_RISE / np.tan(np.radians(angle)))
            plateau = rnd.uniform(0.3, 1.0)
            width   = rnd.uniform(2.0, 3.5)
            rad     = np.hypot(plateau / 2 + ramp, width / 2) + 0.6   # footprint incl. edge fade
            x = rnd.uniform(-TERRAIN_W/2 + rad, TERRAIN_W/2 - rad)
            y = rnd.uniform(-TERRAIN_H/2 + rad, TERRAIN_H/2 - rad)
            ok = all(np.hypot(x - fx, y - fy) > rad + fr for fx, fy, fr in FLAT_ZONES + KEEP_OUT)
            ok = ok and all(np.hypot(x - o['x'], y - o['y']) > rad + o['_rad'] + MIN_GAP for o in out)
            if ok:
                out.append(dict(x=x, y=y, yaw=rnd.uniform(0, 360), angle=angle,
                                ramp=ramp, plateau=plateau, width=width, _rad=rad))
                break
    return out

SLOPES = _gen_slopes()

def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _patch(X, Y, p, edge=None):
    edge = 1.5 * p['ramp'] if edge is None else edge

    th = np.radians(p['yaw'])
    dx, dy = X - p['x'], Y - p['y']
    u =  dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    beyond = np.clip(np.abs(u) - p['plateau'] / 2, 0, None)
    prof = np.clip(p['ramp'] - beyond, 0, None) * np.tan(np.radians(p['angle']))
    return prof * _smoothstep((p['width'] / 2 - np.abs(v)) / edge)

def _blur(a):
    k = np.array([1, 2, 1], dtype=np.float64) / 4
    for _ in range(SMOOTH_PASSES):
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, 'edge'), k, 'valid'), 0, a)
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, 'edge'), k, 'valid'), 1, a)
    return a


def build_height(slopes=None):
    slopes = SLOPES if slopes is None else slopes
    xs = np.linspace(-TERRAIN_W / 2, TERRAIN_W / 2, N)
    ys = np.linspace(-TERRAIN_H / 2, TERRAIN_H / 2, N)
    X, Y = np.meshgrid(xs, ys)               # arrays indexed [j(y), i(x)]
    Z = np.zeros_like(X)
    for p in slopes:
        Z = np.maximum(Z, _patch(X, Y, p))
    Z = _blur(Z)
    for (fx, fy, r) in FLAT_ZONES:
        d = np.hypot(X - fx, Y - fy)
        Z *= _smoothstep((d - r) / FLAT_BLEND)
    return xs, ys, Z

XS, YS, HEIGHT = build_height()
STEP = XS[1] - XS[0]


def terrain_z(wx, wy):
    """Bilinear sample of the same grid the mesh/collision use."""
    fx = np.clip((wx + TERRAIN_W / 2) / STEP, 0, N - 1.000001)
    fy = np.clip((wy + TERRAIN_H / 2) / STEP, 0, N - 1.000001)
    i, j = int(fx), int(fy)
    tx, ty = fx - i, fy - j
    h = HEIGHT
    return float((1-tx)*(1-ty)*h[j, i] + tx*(1-ty)*h[j, i+1]
                 + (1-tx)*ty*h[j+1, i] + tx*ty*h[j+1, i+1])


def terrain_normal(wx, wy):
    e = STEP
    gx = (terrain_z(wx + e, wy) - terrain_z(wx - e, wy)) / (2 * e)
    gy = (terrain_z(wx, wy + e) - terrain_z(wx, wy - e)) / (2 * e)
    n = np.array([-gx, -gy, 1.0])
    return n / np.linalg.norm(n)


def write_obj(path=OBJ_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gy_, gx_ = np.gradient(HEIGHT, STEP)
    nrm = np.dstack([-gx_, -gy_, np.ones_like(HEIGHT)])
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    with open(path, 'w') as f:
        for j in range(N):
            for i in range(N):
                f.write(f"v {XS[i]:.5f} {YS[j]:.5f} {HEIGHT[j, i]:.5f}\n")
        for j in range(N):
            for i in range(N):
                f.write(f"vt {XS[i]/TEXTURE_SIZE:.5f} {YS[j]/TEXTURE_SIZE:.5f}\n")
        for j in range(N):
            for i in range(N):
                f.write(f"vn {nrm[j,i,0]:.5f} {nrm[j,i,1]:.5f} {nrm[j,i,2]:.5f}\n")
        for j in range(N - 1):
            for i in range(N - 1):
                a = j*N + i + 1; b = a + 1; c = a + N + 1; d = a + N
                f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
                f.write(f"f {a}/{a}/{a} {c}/{c}/{c} {d}/{d}/{d}\n")
    print(f"Wrote {path}  max height {HEIGHT.max():.3f} m")


if __name__ == '__main__':
    write_obj()
    gy_, gx_ = np.gradient(HEIGHT, STEP)
    print("max slope:", np.degrees(np.arctan(np.hypot(gx_, gy_).max())), "deg")

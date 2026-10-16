#!/usr/bin/env python3
import psutil, time, csv, sys, numpy as np

TARGETS = {                      # label -> substring in the command line
    "live_detection":   "live_detection.py",
    "frontier_explorer":"frontier_explorer",
    "octomap_trunks":   "__ns:=/trunks",          # 0.1 m full-height instance
    "octomap_explorer": "octomap_server_node",    # matched last, see below
}
DT = 1.0
ncpu = psutil.cpu_count()

def find():
    found = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info["cmdline"] or [])
        for label, key in TARGETS.items():
            if key in cmd and "measure.py" not in cmd:
                if label == "octomap_explorer" and "__ns:=/trunks" in cmd:
                    continue                      # that one is octomap_trunks
                found[label] = p
                break
    return found

procs = find()
print("tracking:", {k: v.pid for k, v in procs.items()})
for p in procs.values():
    p.cpu_percent(None)                           # prime the counters
t0 = time.time()
cpu0 = {k: sum(p.cpu_times()[:2]) for k, p in procs.items()}
rows = []
try:
    while True:
        time.sleep(DT)
        for k, p in procs.items():
            try:
                rows.append((time.time() - t0, k, p.cpu_percent(None),
                             p.memory_info().rss / 1e6))
            except psutil.NoSuchProcess:
                pass
except KeyboardInterrupt:
    pass

dur = time.time() - t0
with open("usage.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_s", "proc", "cpu_pct_of_1core", "rss_mb"]); w.writerows(rows)

print(f"\nduration {dur:.0f} s, {ncpu} logical cores")
print(f"{'process':18s} {'avg cpu%':>9s} {'p95':>7s} {'max':>7s} {'avg MB':>8s} {'peak MB':>8s} {'cpu-sec':>8s}")
for k, p in procs.items():
    c = np.array([r[2] for r in rows if r[1] == k]); m = np.array([r[3] for r in rows if r[1] == k])
    if len(c) == 0: continue
    try: cs = sum(p.cpu_times()[:2]) - cpu0[k]
    except psutil.NoSuchProcess: cs = float("nan")
    print(f"{k:18s} {c.mean():9.1f} {np.percentile(c,95):7.1f} {c.max():7.1f} "
          f"{m.mean():8.0f} {m.max():8.0f} {cs:8.1f}")

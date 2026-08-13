#!/usr/bin/env python3
import json
import os
import sys

import numpy as np

NECK, ROOT_TAIL = 3, 4
L_SHO, R_SHO = 5, 8
L_HIP, R_HIP = 11, 14
PAWS = (7, 10, 13, 16)

MIN_PAW_DROP = 0.32
MIN_DROP_OVER_THICK = 1.6
MIN_USABLE_FRAMES = 20


def load_frames(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("keypoints", "frames", "stabilised", "data"):
            if key in data:
                data = data[key]
                break
    frames = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("keypoints", item.get("kps"))
        if item is None:
            continue
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 17:
            frames.append(arr[:17])
    return frames


def measure(frames):
    drops, thicks = [], []
    for kp in frames:
        xy = kp[:, :2]
        if not np.all(np.isfinite(xy[[NECK, ROOT_TAIL]])):
            continue
        neck, tail = xy[NECK], xy[ROOT_TAIL]
        spine_len = float(np.hypot(*(neck - tail)))
        if spine_len < 20:
            continue

        dx = neck[0] - tail[0]
        if abs(dx) < 1e-6:
            continue
        slope = (neck[1] - tail[1]) / dx

        def spine_y(x):
            return tail[1] + slope * (x - tail[0])

        pd = [xy[i, 1] - spine_y(xy[i, 0])
              for i in PAWS if np.all(np.isfinite(xy[i]))]
        if len(pd) < 2:
            continue
        drops.append(float(np.mean(pd)) / spine_len)

        girth = [abs(xy[i, 1] - spine_y(xy[i, 0]))
                 for i in (L_SHO, R_SHO, L_HIP, R_HIP) if np.all(np.isfinite(xy[i]))]
        if girth:
            thicks.append(float(np.mean(girth)) / spine_len)

    if len(drops) < MIN_USABLE_FRAMES:
        return None
    drop = float(np.median(drops))
    thick = float(np.median(thicks)) if thicks else float("nan")
    ratio = drop / thick if thick and thick > 1e-6 else float("inf")
    return dict(frames=len(drops), paw_drop=drop, spine_thickness=thick, ratio=ratio)


def verdict(m):
    if m is None:
        return "NO DATA", ["fewer than %d usable frames" % MIN_USABLE_FRAMES]
    reasons = []
    if m["paw_drop"] < MIN_PAW_DROP:
        reasons.append("paw_drop %.2f < %.2f (legs foreshortened)"
                       % (m["paw_drop"], MIN_PAW_DROP))
    if m["ratio"] < MIN_DROP_OVER_THICK:
        reasons.append("drop/thickness %.2f < %.2f (body wide + legs short = high angle)"
                       % (m["ratio"], MIN_DROP_OVER_THICK))
    if len(reasons) == 2:
        return "REJECT", reasons
    if reasons:
        return "CHECK", reasons
    return "PASS", []


def find_jsons(root):
    if os.path.isfile(root):
        return [(os.path.basename(os.path.dirname(root)) or root, root)]
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn == "stabilised_keypoints.json":
                full = os.path.join(dirpath, fn)
                parts = os.path.normpath(full).split(os.sep)
                name = parts[-3] if len(parts) >= 3 and parts[-2] == "outputs" else parts[-2]
                out.append((name, full))
    return sorted(out)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs/batch"
    targets = find_jsons(root)
    if not targets:
        print("no stabilised_keypoints.json found under %r" % root)
        return 1

    print("%-38s %6s %9s %10s %7s  %s"
          % ("clip", "frames", "paw_drop", "thickness", "ratio", "verdict"))
    print("-" * 100)
    rows = []
    for name, path in targets:
        m = measure(load_frames(path))
        v, reasons = verdict(m)
        rows.append((name, m, v, reasons))
        if m is None:
            print("%-38s %6s %9s %10s %7s  %s" % (name[:38], "-", "-", "-", "-", v))
        else:
            print("%-38s %6d %9.3f %10.3f %7.2f  %s"
                  % (name[:38], m["frames"], m["paw_drop"],
                     m["spine_thickness"], m["ratio"], v))
        for r in reasons:
            print("%-38s   -> %s" % ("", r))

    tally = {}
    for _n, _m, v, _r in rows:
        tally[v] = tally.get(v, 0) + 1
    print("-" * 100)
    print(" ".join("%s:%d" % kv for kv in sorted(tally.items())))
    print("\nCalibrate MIN_PAW_DROP / MIN_DROP_OVER_THICK against known-good clips")
    print("before using this to reject anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
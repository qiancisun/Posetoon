#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys

import numpy as np

BATCH = "outputs/batch"
KEYPOINT_NAMES = ["L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
                  "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow",
                  "R_F_Paw", "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee",
                  "R_B_Paw"]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
PAWS = [KP["L_F_Paw"], KP["R_F_Paw"], KP["L_B_Paw"], KP["R_B_Paw"]]

EDGE_MARGIN_FRAC = 0.006
MIN_EDGE_POINTS = 3
TURN_FRACTION = 0.72
STAND_RATIO = 0.45
MIN_STRIDE = 0.10


def find_keypoints(clip_dir):
    for cand in ("stabilised_keypoints.json",
                 os.path.join("outputs", "stabilised_keypoints.json"),
                 "full_video_keypoints.json"):
        p = os.path.join(clip_dir, cand)
        if os.path.exists(p):
            return p
    return None


VIDEO_DIRS = ["videos", "videos_raw", "."]


def frame_size(clip_dir):
    name = os.path.basename(clip_dir.rstrip("/"))
    for d in VIDEO_DIRS:
        for ext in (".mp4", ".mov", ".MP4", ".m4v"):
            p = os.path.join(d, name + ext)
            if not os.path.exists(p):
                continue
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height", "-of",
                     "csv=p=0:s=x", p],
                    capture_output=True, text=True, timeout=20)
                a, b = r.stdout.strip().split("x")[:2]
                return float(a), float(b)
            except Exception:
                return None
    return None


def audit(clip_dir):
    kp_path = find_keypoints(clip_dir)
    if kp_path is None:
        return None
    try:
        with open(kp_path) as f:
            frames = json.load(f)
    except (ValueError, OSError):
        return None
    if not frames:
        return None

    pts = []
    for fr in frames:
        a = np.array(fr.get("keypoints", []), dtype=float)
        if a.ndim == 2 and len(a) >= 17:
            pts.append(a[:17])
    if len(pts) < 10:
        return None
    pts = np.stack(pts)

    wh = frame_size(clip_dir)
    if wh is None:
        w = float(np.nanmax(pts[:, :, 0]))
        h = float(np.nanmax(pts[:, :, 1]))
        inferred = True
    else:
        w, h = wh
        inferred = False
    margin = EDGE_MARGIN_FRAC * min(w, h)

    valid = (pts[:, :, 0] > 0.5) & (pts[:, :, 1] > 0.5)
    at_edge = (((pts[:, :, 0] <= margin) | (pts[:, :, 0] >= w - margin)
                | (pts[:, :, 1] <= margin) | (pts[:, :, 1] >= h - margin))
               & valid)
    frac_edge = float(np.mean(at_edge.sum(axis=1) >= MIN_EDGE_POINTS))
    frac_missing = float(np.mean(~valid))

    spine = np.linalg.norm(pts[:, KP["Neck"]] - pts[:, KP["root_of_tail"]],
                           axis=1)
    med = float(np.median(spine[spine > 1e-6])) if np.any(spine > 1e-6) else 0.0
    frac_turn = (float(np.mean(spine < TURN_FRACTION * med))
                 if med > 1e-6 else 1.0)

    paw_y = pts[:, PAWS, 1]
    spread = np.nanmax(paw_y, axis=1) - np.nanmin(paw_y, axis=1)
    frac_stand = (float(np.mean(spread > STAND_RATIO * med))
                  if med > 1e-6 else 0.0)

    d_spine = np.abs(np.diff(spine)) / max(med, 1e-6)
    frac_jitter = float(np.mean(d_spine > 0.18)) if len(d_spine) else 1.0

    body_ax = pts[:, KP["root_of_tail"]] - pts[:, KP["Neck"]]
    n = np.linalg.norm(body_ax, axis=1, keepdims=True)
    body_ax = body_ax / np.maximum(n, 1e-6)
    reach = pts[:, KP["L_F_Paw"]] - pts[:, KP["L_Elbow"]]
    along = np.sum(reach * body_ax, axis=1) / max(med, 1e-6)
    swing = float(np.percentile(along, 90) - np.percentile(along, 10))

    return {"frames": len(pts), "edge": frac_edge, "turn": frac_turn,
            "unsettled": frac_stand, "jitter": frac_jitter,
            "stride": float(swing), "inferred_size": inferred,
            "missing": frac_missing}


def verdict(a):
    bad = []
    if a["edge"] > 0.15:
        bad.append(f"out of frame in {a['edge']:.0%} of frames")
    if a["turn"] > 0.12:
        bad.append(f"turns off side-on in {a['turn']:.0%} of frames")
    if a["unsettled"] > 0.35:
        bad.append(f"paws off one ground line in {a['unsettled']:.0%} "
                   f"(sitting, jumping or not walking)")
    if a.get("jitter", 0) > 0.18:
        bad.append(f"spine length jumps in {a['jitter']:.0%} of frames -- the "
                   f"tracker is losing the dog (occlusion or motion blur)")
    if a.get("missing", 0) > 0.25:
        bad.append(f"{a['missing']:.0%} of keypoints were never detected -- "
                   f"the skeleton is incomplete")
    if a.get("stride", 9.0) < MIN_STRIDE:
        bad.append(f"the front paw travels only {a['stride']:.02f} of a spine "
                   f"length relative to its own elbow -- the dog is barely "
                   f"moving, so there is no gait to retarget")
    return bad


def load_grades(path):
    out = {}
    if not os.path.exists(path):
        return out
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        g, rest = line[0].upper(), line[1:].strip()
        name = rest.split("  ")[0].strip() if "  " in rest else rest
        out[name] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades.txt")
    a = ap.parse_args()

    if not os.path.isdir(BATCH):
        sys.exit(f"{BATCH} not found")
    grades = load_grades(a.grades)

    rows = []
    for name in sorted(os.listdir(BATCH)):
        d = os.path.join(BATCH, name)
        if not os.path.isdir(d):
            continue
        r = audit(d)
        if r is None:
            print(f"  no keypoints found for {name}")
            continue
        r["name"] = name
        r["grade"] = grades.get(name, "?")
        rows.append(r)

    n_inf = sum(1 for r in rows if r.get("inferred_size"))
    if n_inf:
        print(f"  !! {n_inf} clip(s) had no source video to read the frame "
              f"size from; their edge figure is inferred and reads high\n")

    if not rows:
        sys.exit("nothing audited -- are the per-clip keypoint files still "
                 "in outputs/batch/<name>/ ?")

    print(f"{'clip':32} {'gr':>3} {'edge':>6} {'turn':>6} {'unset':>6} "
          f"{'jitter':>7} {'stride':>7} {'miss':>6}")
    print("-" * 74)
    for r in sorted(rows, key=lambda x: -(x["edge"] + x["turn"] + x["jitter"])):
        flag = "  <<" if verdict(r) else ""
        print(f"{r['name'][:32]:32} {r['grade']:>3} {r['edge']:6.0%} "
              f"{r['turn']:6.0%} {r['unsettled']:6.0%} {r['jitter']:7.0%} "
              f"{r['stride']:7.2f} {r.get('missing', 0):6.0%}{flag}")

    bad_a = [(r, verdict(r)) for r in rows
             if r["grade"] == "A" and verdict(r)]
    print(f"\n{len(bad_a)} clip(s) graded A that should not be:")
    for r, why in bad_a:
        print(f"\n  {r['name']}")
        for w in why:
            print(f"      {w}")
    if not bad_a:
        print("  none -- every A clip stays side-on, in frame and on its feet")

    clean = [r for r in rows if not verdict(r)]
    print(f"\n{len(clean)} of {len(rows)} clip(s) pass all three checks. "
          f"Those are the ones that can carry the demo reel:")
    for r in sorted(clean, key=lambda x: x["name"]):
        print(f"  {r['grade']}  {r['name']}")


if __name__ == "__main__":
    main()
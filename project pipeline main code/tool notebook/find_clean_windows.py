#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np

NOSE, NECK, ROOT_TAIL = 2, 3, 4
L_SHO, R_SHO = 5, 8
L_ELB, L_F_PAW = 6, 7
R_ELB, R_F_PAW = 9, 10
L_HIP, R_HIP = 11, 14
L_B_PAW, R_B_PAW = 13, 16
PAWS = (L_F_PAW, R_F_PAW, L_B_PAW, R_B_PAW)

TURN_RATIO = 0.72
JUMP_RATIO = 0.18
BORDER_PX = 6
MIN_PAW_DROP = 0.22
MIN_SWING_DEG = 18.0
DEFAULT_MIN_SECONDS = 3.0
DEFAULT_FPS = 30.0

MIN_DROP_OVER_THICK = 1.60
RATIO_SMOOTH = 15


def load_frames(path):
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in ("keypoints", "frames", "stabilised", "data"):
            if key in data:
                data = data[key]
                break
    out = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("keypoints", item.get("kps"))
        if item is None:
            out.append(None)
            continue
        arr = np.asarray(item, dtype=float)
        out.append(arr[:17] if arr.ndim == 2 and arr.shape[0] >= 17 else None)
    return out


def frame_size(frames, video_path=None):
    if video_path and os.path.exists(video_path):
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return w, h, "measured"
        except Exception:
            pass
    return None, None, "unknown"


def score_frames(frames, W, H):
    n = len(frames)
    spine = np.full(n, np.nan)
    drop = np.full(n, np.nan)
    thick = np.full(n, np.nan)
    border = np.zeros(n, dtype=bool)
    present = np.zeros(n, dtype=bool)

    for i, kp in enumerate(frames):
        if kp is None:
            continue
        xy = kp[:, :2]
        if not np.all(np.isfinite(xy[[NECK, ROOT_TAIL]])):
            continue
        present[i] = True
        neck, tail = xy[NECK], xy[ROOT_TAIL]
        L = float(np.hypot(*(neck - tail)))
        spine[i] = L
        if L > 20 and abs(neck[0] - tail[0]) > 1e-6:
            slope = (neck[1] - tail[1]) / (neck[0] - tail[0])

            def _sy(x, tail=tail, slope=slope):
                return tail[1] + slope * (x - tail[0])

            pd = [xy[j, 1] - _sy(xy[j, 0])
                  for j in PAWS if np.all(np.isfinite(xy[j]))]
            if len(pd) >= 2:
                drop[i] = float(np.mean(pd)) / L
            girth = [abs(xy[j, 1] - _sy(xy[j, 0]))
                     for j in (L_SHO, R_SHO, L_HIP, R_HIP)
                     if np.all(np.isfinite(xy[j]))]
            if girth:
                thick[i] = float(np.mean(girth)) / L
        if W:
            fin = xy[np.all(np.isfinite(xy), axis=1)]
            if len(fin) and (fin[:, 0].min() < BORDER_PX
                             or fin[:, 1].min() < BORDER_PX
                             or fin[:, 0].max() > W - BORDER_PX
                             or fin[:, 1].max() > H - BORDER_PX):
                border[i] = True

    med = np.nanmedian(spine)
    ok = present.copy()
    reasons = {}

    turning = present & (spine < TURN_RATIO * med)
    ok &= ~turning
    reasons["turning"] = turning

    highcam = present & np.isfinite(drop) & (drop < MIN_PAW_DROP)
    ok &= ~highcam
    reasons["high camera"] = highcam

    ratio = np.full(n, np.nan)
    good = np.isfinite(drop) & np.isfinite(thick) & (thick > 1e-6)
    ratio[good] = drop[good] / thick[good]
    smooth = np.full(n, np.nan)
    half = max(1, RATIO_SMOOTH // 2)
    for i in range(n):
        seg = ratio[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            smooth[i] = float(np.median(seg))
    tipped = present & np.isfinite(smooth) & (smooth < MIN_DROP_OVER_THICK)
    ok &= ~tipped
    reasons["camera above the dog"] = tipped

    jump = np.zeros(n, dtype=bool)
    d = np.abs(np.diff(spine)) / np.maximum(med, 1e-6)
    bad = np.nonzero(d > JUMP_RATIO)[0]
    jump[bad] = True
    jump[np.minimum(bad + 1, n - 1)] = True
    ok &= ~jump
    reasons["tracking jump"] = jump

    ok &= ~border
    reasons["out of frame"] = border
    reasons["no detection"] = ~present
    return ok, reasons, med


def swing_deg(frames, a, b):
    angs = []
    for kp in frames[a:b]:
        if kp is None:
            continue
        xy = kp[:, :2]
        for elb, paw in ((L_ELB, L_F_PAW), (R_ELB, R_F_PAW)):
            if np.all(np.isfinite(xy[[elb, paw]])):
                v = xy[paw] - xy[elb]
                angs.append(np.degrees(np.arctan2(v[1], v[0])))
                break
    if len(angs) < 10:
        return 0.0
    a_ = np.unwrap(np.radians(angs))
    return float(np.degrees(np.percentile(a_, 95) - np.percentile(a_, 5)))


def longest_run(ok):
    best = (0, 0, 0)
    i = 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j < len(ok) and ok[j]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j
        else:
            i += 1
    return best


def find_video(name, roots):
    for root in roots:
        for ext in (".mp4", ".mov", ".MP4", ".webm"):
            p = os.path.join(root, name + ext)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="outputs/batch")
    ap.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help="only used to convert frames to seconds")
    ap.add_argument("--video-dirs", default="videos,videos_raw")
    ap.add_argument("--write-cuts", default=None,
                    help="write a shell script of ffmpeg cut commands")
    args = ap.parse_args()

    vdirs = [d.strip() for d in args.video_dirs.split(",") if d.strip()]
    clips = []
    for dirpath, _dn, fns in os.walk(args.root):
        if "stabilised_keypoints.json" in fns:
            parts = os.path.normpath(dirpath).split(os.sep)
            name = parts[-2] if parts[-1] == "outputs" else parts[-1]
            clips.append((name, os.path.join(dirpath, "stabilised_keypoints.json")))
    clips.sort()
    if not clips:
        print("no stabilised_keypoints.json under %r" % args.root)
        return 1

    min_f = int(round(args.min_seconds * args.fps))
    print("Longest clean window per clip  (min %d frames = %.1fs at %.0f fps)\n"
          % (min_f, args.min_seconds, args.fps))
    print("%-36s %6s %7s %8s %7s  %s"
          % ("clip", "frames", "clean", "window", "swing", "verdict"))
    print("-" * 104)

    cuts = []
    whole, trim, nothing = 0, 0, 0
    no_dims = []
    for name, path in clips:
        frames = load_frames(path)
        vid = find_video(name, vdirs)
        W, H, src = frame_size(frames, vid)
        if W is None:
            no_dims.append(name)
        ok, reasons, med = score_frames(frames, W, H)
        n = len(frames)
        length, a, b = longest_run(ok)
        sw = swing_deg(frames, a, b) if length else 0.0

        if length < min_f:
            verdict = "NOTHING USABLE"
            nothing += 1
        elif sw < MIN_SWING_DEG:
            verdict = "no gait in window (%.0f deg)" % sw
            nothing += 1
        elif length >= 0.97 * n:
            verdict = "ALREADY CLEAN"
            whole += 1
        else:
            verdict = "TRIM -> %.2f..%.2fs" % (a / args.fps, b / args.fps)
            trim += 1
            if vid:
                cuts.append((name, vid, a / args.fps, (b - a) / args.fps))

        print("%-36s %6d %6.0f%% %4d..%-4d %6.0f  %s"
              % (name[:36], n, 100.0 * ok.sum() / max(n, 1), a, b, sw, verdict))

        if verdict.startswith(("TRIM", "NOTHING")):
            worst = sorted(((r.sum(), k) for k, r in reasons.items()),
                           reverse=True)[:2]
            detail = ", ".join("%s %d" % (k, c) for c, k in worst if c)
            if detail:
                print("%-36s   why: %s" % ("", detail))
            if not vid and verdict.startswith("TRIM"):
                print("%-36s   source video not found in %s -- cut manually"
                      % ("", args.video_dirs))

    print("-" * 104)
    print("already clean %d | trimmable %d | nothing usable %d"
          % (whole, trim, nothing))
    if no_dims:
        print("\n%d clip(s) had no source video in %s, so the OUT OF FRAME test "
              "was skipped\nfor them -- a dog walking in from offscreen will not "
              "be caught. Put the videos\nthere, or pass --video-dirs, to enable "
              "it: %s" % (len(no_dims), args.video_dirs,
                          ", ".join(no_dims[:4]) + (" ..." if len(no_dims) > 4 else "")))

    if args.write_cuts and cuts:
        with open(args.write_cuts, "w") as fh:
            fh.write("#!/bin/sh\n# generated by find_clean_windows.py\n")
            fh.write("# -c copy does not re-encode, so no new duplicate "
                     "frames are introduced.\n")
            fh.write("# Each cut writes a NEW filename: the batch runner keys "
                     "its working\n# directory off the name, so reusing the old "
                     "one would silently reuse the\n# old keypoints.\n\n")
            for name, vid, ss, dur in cuts:
                ext = os.path.splitext(vid)[1]
                out = os.path.join(os.path.dirname(vid), name + "_cut" + ext)
                fh.write('ffmpeg -v error -y -ss %.2f -t %.2f -i "%s" '
                         '-c copy "%s" || echo "CUT FAILED: %s"\n'
                         % (ss, dur, vid, out, out))
            fh.write("\n# then re-run only these:\n")
            for name, vid, _s, _d in cuts:
                ext = os.path.splitext(vid)[1]
                out = os.path.join(os.path.dirname(vid), name + "_cut" + ext)
                fh.write('# $PY run_one.py "%s"\n' % out)
        print("\nwrote %s (%d cut commands)" % (args.write_cuts, len(cuts)))
        print("Read it before running it. Watch each cut for three seconds:")
        print("this script cannot see edit points.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
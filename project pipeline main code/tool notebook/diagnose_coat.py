#!/usr/bin/env python3
import glob
import json
import os
import sys

import numpy as np

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

SAMPLE_EVERY = 5
DISC_RADIUS = 4
MAD_REJECT = 3.0
INWARD_FRACTION = 0.60


def old_points(kp):
    neck, tail = kp[KP["Neck"]], kp[KP["root_of_tail"]]
    sh = (kp[KP["L_Shoulder"]] + kp[KP["R_Shoulder"]]) / 2
    hip = (kp[KP["L_Hip"]] + kp[KP["R_Hip"]]) / 2
    pts = [neck + (tail - neck) * t for t in (0.30, 0.45, 0.60, 0.75)]
    pts += [(neck + sh) / 2 * 0.5 + (neck * 0.5), (tail + hip) / 2]
    centre = (neck + tail) / 2
    return [p + (centre - p) * 0.18 for p in pts]


def new_points(kp, inward_fraction=INWARD_FRACTION):
    neck, tail = kp[KP["Neck"]], kp[KP["root_of_tail"]]
    sh = (kp[KP["L_Shoulder"]] + kp[KP["R_Shoulder"]]) / 2
    hip = (kp[KP["L_Hip"]] + kp[KP["R_Hip"]]) / 2
    axis = tail - neck
    span = float(np.linalg.norm(axis))
    if span < 1e-6:
        return [(neck + tail) / 2]
    u = axis / span
    n = np.array([-u[1], u[0]])
    depth = float(np.median([float(np.dot(sh - neck, n)),
                             float(np.dot(hip - tail, n))]))
    inward = n * depth * inward_fraction
    pts = [neck + axis * t + inward for t in (0.30, 0.45, 0.60, 0.75)]
    pts += [sh + (neck - sh) * 0.35, hip + (tail - hip) * 0.35]
    return pts


def rgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def reduce_samples(samples):
    if len(samples) < 20:
        return None, 0
    a = np.array(samples, dtype=float)
    lab = rgb_to_lab(a)
    med = np.median(lab, axis=0)
    dist = np.linalg.norm(lab - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
    keep = dist < np.median(dist) + MAD_REJECT * mad
    return np.median(a[keep], axis=0), int(keep.sum())


def lightness(rgb):
    r, g, b = [v / 255.0 for v in rgb]
    return 0.5 * (max(r, g, b) + min(r, g, b))


def hexs(rgb):
    return "#{:02X}{:02X}{:02X}".format(*[int(round(v)) for v in rgb])


def find_video(name, dirs=("videos", "videos_raw")):
    for d in dirs:
        for ext in (".mp4", ".mov", ".MP4", ".webm"):
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p
    return None


def find_stab(name):
    direct = "outputs/batch/%s/outputs/stabilised_keypoints.json" % name
    if os.path.exists(direct):
        return direct
    root = os.path.join("outputs/batch", name)
    if os.path.isdir(root):
        for dirpath, _dn, fns in os.walk(root):
            for fn in fns:
                if fn.endswith("stabilised_keypoints.json"):
                    return os.path.join(dirpath, fn)
    for dirpath, _dn, fns in os.walk("outputs"):
        if os.path.basename(dirpath.rstrip("/")) == name or \
           os.path.basename(os.path.dirname(dirpath)) == name:
            for fn in fns:
                if fn.endswith("stabilised_keypoints.json"):
                    return os.path.join(dirpath, fn)
    return None


def load_kp_frames(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        for k in ("keypoints", "frames", "stabilised", "data"):
            if k in data:
                data = data[k]
                break
    out = []
    for item in data:
        if isinstance(item, dict):
            item = item.get("keypoints", item.get("kps"))
        if item is None:
            continue
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 17:
            out.append(arr[:17, :2])
    return out


def run(name, write_overlay=True):
    import cv2

    stab = find_stab(name)
    if not stab:
        print("%-38s no stabilised_keypoints.json under outputs/batch/%s"
              % (name[:38], name))
        return
    video = find_video(name)
    if not video:
        print("%-38s keypoints OK, but no source video in videos/ videos_raw/"
              % name[:38])
        return

    frames = load_kp_frames(stab)
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def disc(frame, p):
        x, y = int(round(p[0])), int(round(p[1]))
        r = DISC_RADIUS
        if not (r <= x < W - r and r <= y < H - r):
            return None
        patch = frame[y - r:y + r + 1, x - r:x + r + 1]
        return np.median(patch.reshape(-1, 3), axis=0)[::-1]

    old_s, new_s, idx = [], [], 0
    overlay = None
    overlay_pts = None
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(frames):
            break
        if idx % SAMPLE_EVERY == 0:
            kp = frames[idx]
            po, pn = old_points(kp), new_points(kp)
            for p in po:
                v = disc(frame, p)
                if v is not None:
                    old_s.append(v)
            for p in pn:
                v = disc(frame, p)
                if v is not None:
                    new_s.append(v)
            if overlay is None and idx > len(frames) // 4:
                overlay, overlay_pts = frame.copy(), (po, pn)
        idx += 1
    cap.release()

    o, no = reduce_samples(old_s)
    n, nn = reduce_samples(new_s)
    if o is None or n is None:
        print("%-38s too few samples" % name[:38])
        return

    dl = lightness(n) - lightness(o)
    flag = ""
    if dl < -0.08:
        flag = "  <-- NEW IS DARKER; inward offset is the suspect"
    elif dl > 0.08:
        flag = "  <-- NEW IS LIGHTER"
    print("%-38s OLD %s L=%.2f | NEW %s L=%.2f | dL %+.2f%s"
          % (name[:38], hexs(o), lightness(o), hexs(n), lightness(n), dl, flag))

    if write_overlay and overlay is not None:
        po, pn = overlay_pts
        for p in po:
            cv2.circle(overlay, (int(p[0]), int(p[1])), DISC_RADIUS + 1,
                       (0, 0, 255), 1)
        for p in pn:
            cv2.circle(overlay, (int(p[0]), int(p[1])), DISC_RADIUS + 1,
                       (0, 255, 0), 1)
        os.makedirs("outputs/coat_debug", exist_ok=True)
        out = "outputs/coat_debug/%s.png" % name
        cv2.imwrite(out, overlay)


def sweep(name):
    import cv2
    stab = find_stab(name)
    if not stab:
        print("%-30s no stabilised_keypoints.json found" % name[:30])
        return
    video = find_video(name)
    if not video:
        print("%-30s keypoints OK, but no source video found" % name[:30])
        return
    frames = load_kp_frames(stab)
    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fracs = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    acc = {f: [] for f in fracs}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(frames):
            break
        if idx % SAMPLE_EVERY == 0:
            kp = frames[idx]
            for f in fracs:
                for p in new_points(kp, f):
                    x, y = int(round(p[0])), int(round(p[1]))
                    r = DISC_RADIUS
                    if r <= x < W - r and r <= y < H - r:
                        acc[f].append(np.median(
                            frame[y-r:y+r+1, x-r:x+r+1].reshape(-1, 3), 0)[::-1])
        idx += 1
    cap.release()
    print("\n%s" % name)
    for f in fracs:
        v, kept = reduce_samples(acc[f])
        if v is None:
            print("   %.2f   too few samples" % f)
        else:
            print("   %.2f   %s  L=%.2f" % (f, hexs(v), lightness(v)))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--sweep" in sys.argv:
        for n in args:
            sweep(n)
        return 0
    if "--all" in sys.argv:
        args = sorted(
            os.path.basename(os.path.dirname(os.path.dirname(p)))
            for p in glob.glob("outputs/batch/*/outputs/stabilised_keypoints.json")
            if "__superseded" not in p)
    if not args:
        print("Usage: python diagnose_coat.py <clip-name> [more clips...]")
        print("       python diagnose_coat.py --all")
        print("       python diagnose_coat.py --sweep <clip-name>")
        return 1
    print("red circles = old sample points, green = new. "
          "Overlays in outputs/coat_debug/\n")
    for name in args:
        run(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
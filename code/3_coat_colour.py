import os
import sys
import json
import colorsys
import numpy as np

OUT_DIR = "outputs"
STAB_PATH = "stabilised_keypoints.json"
DEFAULT_VIDEO = "dogvideo.mp4"

WHITE_BALANCE = True
WB_PERCENTILE = 95
WB_MAX_GAIN = 1.5

SAMPLE_EVERY = 5
DISC_RADIUS = 4
PART_DISC_RADIUS = 2
MIN_PART_SAMPLES = 12
MAD_REJECT = 3.0
LIGHTNESS_FLOOR = 0.06
LIGHTNESS_CEIL = 0.94

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

L_OFFSET = {
    "body": 0.00, "neck": 0.02, "head": 0.04, "muzzle": 0.11,
    "ear_upper": -0.05, "ear_lower": -0.08, "tail": -0.09,
    "near_upper": -0.01, "near_mid": -0.04, "near_lower": -0.09,
    "far_upper": 0.13, "far_mid": 0.10, "far_lower": 0.06,
    "outline": -0.14, "outline_dark": -0.18,
}


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


def shift_lightness(rgb, dl, base_l=None):
    r, g, b = [v / 255.0 for v in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if base_l is None:
        base_l = l

    target = base_l + dl
    if target < LIGHTNESS_FLOOR or target > LIGHTNESS_CEIL:
        target = base_l - dl
    l = float(np.clip(target, LIGHTNESS_FLOOR, LIGHTNESS_CEIL))

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


PART_SITES = {
    "head":       [("Neck", "Nose", 0.35), ("Neck", "Nose", 0.55)],
    "near_upper": [("L_Shoulder", "L_Elbow", 0.5), ("L_Hip", "L_Knee", 0.5)],
    "near_lower": [("L_Elbow", "L_F_Paw", 0.45), ("L_Knee", "L_B_Paw", 0.45)],
    "far_upper":  [("R_Shoulder", "R_Elbow", 0.5), ("R_Hip", "R_Knee", 0.5)],
    "tail":       [("Neck", "root_of_tail", 1.06),
                   ("Neck", "root_of_tail", 1.12)],
}


INWARD_FRACTION = 0.15


def torso_sample_points(kp):
    neck, tail = kp[KP["Neck"]], kp[KP["root_of_tail"]]
    sh = (kp[KP["L_Shoulder"]] + kp[KP["R_Shoulder"]]) / 2
    hip = (kp[KP["L_Hip"]] + kp[KP["R_Hip"]]) / 2

    axis = tail - neck
    span = float(np.linalg.norm(axis))
    if span < 1e-6:
        return [(neck + tail) / 2]
    u = axis / span
    n = np.array([-u[1], u[0]])

    depths = [float(np.dot(sh - neck, n)), float(np.dot(hip - tail, n))]
    depth = float(np.median(depths))
    inward = n * depth * INWARD_FRACTION

    pts = [neck + axis * t + inward for t in (0.30, 0.45, 0.60, 0.75)]
    pts += [sh + (neck - sh) * 0.35, hip + (tail - hip) * 0.35]
    return pts


def main():
    import cv2

    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        return
    with open(STAB_PATH) as f:
        frames = json.load(f)

    cap = cv2.VideoCapture(video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {video}: {n_video} frames, {w}x{h}")
    print(f"Keypoint frames: {len(frames)}")
    if abs(n_video - len(frames)) > 5:
        print("  NOTE: frame counts differ; sampling by keypoint index.")

    def sample_at(frame, p, radius):
        x, y = int(round(p[0])), int(round(p[1]))
        if not (radius <= x < w - radius and radius <= y < h - radius):
            return None
        patch = frame[y - radius:y + radius + 1, x - radius:x + radius + 1]
        return np.median(patch.reshape(-1, 3), axis=0)[::-1]

    def frame_gain(frame):
        if not WHITE_BALANCE:
            return None
        wp = np.percentile(frame.reshape(-1, 3), WB_PERCENTILE, axis=0)[::-1]
        wp = np.maximum(wp, 1.0)
        return np.clip(255.0 / wp, 1.0, WB_MAX_GAIN)

    samples, idx = [], 0
    gains = []
    part_samples = {k: [] for k in PART_SITES}
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(frames):
            break
        if idx % SAMPLE_EVERY == 0:
            kp = np.array(frames[idx]["keypoints"], dtype=float)
            gain = frame_gain(frame)
            if gain is not None:
                gains.append(gain)
            for p in torso_sample_points(kp):
                v = sample_at(frame, p, DISC_RADIUS)
                if v is not None:
                    samples.append(np.clip(v * gain, 0, 255)
                                   if gain is not None else v)
            for part, sites in PART_SITES.items():
                for a, b, t in sites:
                    if a not in KP or b not in KP:
                        continue
                    pa, pb = kp[KP[a]], kp[KP[b]]
                    if np.linalg.norm(pb - pa) < 8:
                        continue
                    v = sample_at(frame, pa + (pb - pa) * t, PART_DISC_RADIUS)
                    if v is not None:
                        part_samples[part].append(v)
        idx += 1
    cap.release()

    if len(samples) < 20:
        print(f"Only {len(samples)} usable samples -- check that the keypoints "
              f"are in the same pixel space as the video.")
        return

    samples = np.array(samples, dtype=float)
    print(f"Collected {len(samples)} raw samples")
    if WHITE_BALANCE and gains:
        g = np.median(np.array(gains), axis=0)
        print(f"  illuminant normalisation ON: median gain "
              f"R{g[0]:.2f} G{g[1]:.2f} B{g[2]:.2f}")
        if max(g) < 1.03:
            print("  (frame is already well exposed; this changed almost "
                  "nothing, which is the intended behaviour)")
    elif not WHITE_BALANCE:
        print("  illuminant normalisation OFF -- a light coat in shade will "
              "measure as mid grey.")
        print("  Set WHITE_BALANCE = True at the top of this file to correct "
              "for it.")

    lab = rgb_to_lab(samples)
    med = np.median(lab, axis=0)
    dist = np.linalg.norm(lab - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
    keep = dist < np.median(dist) + MAD_REJECT * mad
    kept = samples[keep]
    print(f"Kept {keep.sum()} after outlier rejection "
          f"({100*keep.sum()/len(samples):.0f}%)")

    base = np.median(kept, axis=0)
    base_hex = "#{:02X}{:02X}{:02X}".format(*[int(v) for v in base])
    r, g, b = base / 255.0
    _, base_l, base_s = colorsys.rgb_to_hls(r, g, b)
    print(f"\nBase coat colour: {base_hex}   "
          f"lightness={base_l:.2f}  saturation={base_s:.2f}")
    if base_l < 0.20:
        print("  Very dark coat -- lightness offsets are doing the work here;")
        print("  a multiplicative shading scheme would have flattened this.")
    elif base_l > 0.80:
        print("  Very light coat -- outline colours carry the definition.")

    palette = {k: shift_lightness(base, dl, base_l) for k, dl in L_OFFSET.items()}

    n_unique = len(set(palette.values()))
    print(f"Palette has {n_unique} distinct colours out of {len(palette)} slots")
    if n_unique < len(palette) - 2:
        print("  Several parts share a colour -- the coat is near the extreme")
        print("  of the lightness range and separation is limited.")

    measured = {}
    for part, vals in part_samples.items():
        if len(vals) < MIN_PART_SAMPLES:
            continue
        arr = np.array(vals, dtype=float)
        plab = rgb_to_lab(arr)
        pmed = np.median(plab, axis=0)
        pdist = np.linalg.norm(plab - pmed, axis=1)
        pmad = np.median(np.abs(pdist - np.median(pdist))) + 1e-6
        pkeep = pdist < np.median(pdist) + MAD_REJECT * pmad
        if pkeep.sum() < MIN_PART_SAMPLES // 2:
            continue
        pv = np.median(arr[pkeep], axis=0)
        measured[part] = "#{:02X}{:02X}{:02X}".format(*[int(v) for v in pv])

    if measured:
        def _lum(hx):
            hh = hx.lstrip("#")
            r_, g_, b_ = (int(hh[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            return 0.5 * (max(r_, g_, b_) + min(r_, g_, b_))
        lums = [_lum(v) for v in measured.values()]
        spread = max(lums) - min(lums)
        print(f"\nMeasured {len(measured)} part(s) independently:")
        for k, v in measured.items():
            print(f"  {k:12} {v}   lightness {_lum(v):.2f}")
        print(f"  lightness spread across parts: {spread:.2f}")
        print("  (small = a solid-coloured dog; the rig uses this to decide "
              "how much of\n   the breed template's markings to apply)")
    else:
        print("\nNo part could be sampled reliably -- the rig will keep the "
              "template's markings in full.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "coat_palette.json")
    with open(out_json, "w") as f:
        json.dump({
            "video": video,
            "base_rgb": [int(v) for v in base],
            "base_hex": base_hex,
            "measured": measured,
            "base_lightness": base_l,
            "base_saturation": base_s,
            "n_samples_raw": int(len(samples)),
            "n_samples_kept": int(keep.sum()),
            "palette": palette,
        }, f, indent=2)
    print(f"\nSaved {out_json}")
    for k in ["body", "head", "near_upper", "far_upper", "tail", "outline"]:
        print(f"  {k:12} {palette[k]}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5),
                                    gridspec_kw={"height_ratios": [1, 2]})
    strip = kept[np.linspace(0, len(kept) - 1, min(80, len(kept))).astype(int)]
    ax0.imshow((strip / 255.0)[None, :, :], aspect="auto")
    ax0.set_yticks([])
    ax0.set_title(f"Kept torso samples ({keep.sum()} of {len(samples)})", fontsize=10)

    keys = list(L_OFFSET.keys())
    for i, k in enumerate(keys):
        ax1.add_patch(plt.Rectangle((i, 0), 0.92, 1, color=palette[k]))
        ax1.text(i + 0.46, -0.12, k, ha="center", va="top", fontsize=7, rotation=45)
    ax1.set_xlim(0, len(keys)); ax1.set_ylim(-0.9, 1)
    ax1.axis("off")
    ax1.set_title(f"Derived palette from base {base_hex}", fontsize=10)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "coat_palette.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

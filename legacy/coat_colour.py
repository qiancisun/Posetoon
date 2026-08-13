import os
import sys
import json
import colorsys
import numpy as np

OUT_DIR = "outputs"
STAB_PATH = "stabilised_keypoints.json"
DEFAULT_VIDEO = "dogvideo.mp4"

SAMPLE_EVERY = 5
DISC_RADIUS = 4
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


def torso_sample_points(kp):
    neck, tail = kp[KP["Neck"]], kp[KP["root_of_tail"]]
    sh = (kp[KP["L_Shoulder"]] + kp[KP["R_Shoulder"]]) / 2
    hip = (kp[KP["L_Hip"]] + kp[KP["R_Hip"]]) / 2

    pts = [neck + (tail - neck) * t for t in (0.30, 0.45, 0.60, 0.75)]
    pts += [(neck + sh) / 2 * 0.5 + (neck * 0.5), (tail + hip) / 2]
    centre = (neck + tail) / 2
    return [p + (centre - p) * 0.18 for p in pts]


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

    samples, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= len(frames):
            break
        if idx % SAMPLE_EVERY == 0:
            kp = np.array(frames[idx]["keypoints"], dtype=float)
            for p in torso_sample_points(kp):
                x, y = int(round(p[0])), int(round(p[1]))
                if not (DISC_RADIUS <= x < w - DISC_RADIUS
                        and DISC_RADIUS <= y < h - DISC_RADIUS):
                    continue
                patch = frame[y - DISC_RADIUS:y + DISC_RADIUS + 1,
                              x - DISC_RADIUS:x + DISC_RADIUS + 1]
                bgr = np.median(patch.reshape(-1, 3), axis=0)
                samples.append(bgr[::-1])
        idx += 1
    cap.release()

    if len(samples) < 20:
        print(f"Only {len(samples)} usable samples -- check that the keypoints "
              f"are in the same pixel space as the video.")
        return

    samples = np.array(samples, dtype=float)
    print(f"Collected {len(samples)} raw samples")

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

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "coat_palette.json")
    with open(out_json, "w") as f:
        json.dump({
            "video": video,
            "base_rgb": [int(v) for v in base],
            "base_hex": base_hex,
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

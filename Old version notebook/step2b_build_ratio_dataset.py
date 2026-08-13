import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANN_DIR = "data/ap10k/annotations"
OUT_DIR = "outputs"

MIN_SPINE_PX = 60.0
MIN_HORIZONTALITY = 0.75
STRICT_VISIBILITY = 2

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
K = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

LIMB_CHAINS = {
    "front_L": (("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_F_Paw")),
    "front_R": (("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_F_Paw")),
    "hind_L":  (("L_Hip", "L_Knee"), ("L_Knee", "L_B_Paw")),
    "hind_R":  (("R_Hip", "R_Knee"), ("R_Knee", "R_B_Paw")),
}


def parse_keypoints(flat):
    arr = np.asarray(flat, dtype=float).reshape(-1, 3)
    return arr[:, :2], arr[:, 2]


def bone_length(xy, vis, a, b, min_vis):
    ia, ib = K[a], K[b]
    if vis[ia] < min_vis or vis[ib] < min_vis:
        return None
    return float(np.linalg.norm(xy[ib] - xy[ia]))


def find_dog_category(all_categories):
    for cat in all_categories:
        if "dog" in cat["name"].lower():
            return cat["id"], cat["name"]
    return None, None


def load_annotations(ann_dir):
    files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No annotation json found in {ann_dir}")

    anns_by_id = {}
    images_by_id = {}
    categories = None

    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        if categories is None:
            categories = data.get("categories", [])
        for img in data.get("images", []):
            images_by_id[img["id"]] = img
        for ann in data.get("annotations", []):
            anns_by_id[ann["id"]] = ann
        print(f"  loaded {os.path.basename(path):40} "
              f"imgs={len(data.get('images', [])):6} anns={len(data.get('annotations', [])):6}")

    return anns_by_id, images_by_id, categories, files


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading AP-10K annotations...")
    anns_by_id, images_by_id, categories, files = load_annotations(ANN_DIR)
    print(f"Merged: {len(images_by_id)} images, {len(anns_by_id)} annotations "
          f"from {len(files)} file(s)\n")

    dog_id, dog_name = find_dog_category(categories)
    if dog_id is None:
        print("Could not find a 'dog' category. Available categories:")
        for cat in categories:
            print(f"  id={cat['id']:3}  {cat['name']}")
        return
    print(f"Dog category: id={dog_id}  name='{dog_name}'")

    dog_anns = [a for a in anns_by_id.values() if a["category_id"] == dog_id]
    print(f"Dog annotations: {len(dog_anns)}\n")

    records = []
    reject = {
        "no_keypoints": 0,
        "spine_missing": 0,
        "spine_too_short": 0,
        "not_side_view": 0,
        "incomplete_front": 0,
        "incomplete_hind": 0,
    }

    for ann in dog_anns:
        if "keypoints" not in ann:
            reject["no_keypoints"] += 1
            continue

        xy, vis = parse_keypoints(ann["keypoints"])

        spine = bone_length(xy, vis, "Neck", "root_of_tail", STRICT_VISIBILITY)
        if spine is None:
            reject["spine_missing"] += 1
            continue
        if spine < MIN_SPINE_PX:
            reject["spine_too_short"] += 1
            continue

        dx = abs(xy[K["root_of_tail"]][0] - xy[K["Neck"]][0])
        horizontality = dx / spine
        if horizontality < MIN_HORIZONTALITY:
            reject["not_side_view"] += 1
            continue

        limb_vals = {}
        for chain_name, (upper_pair, lower_pair) in LIMB_CHAINS.items():
            up = bone_length(xy, vis, *upper_pair, STRICT_VISIBILITY)
            lo = bone_length(xy, vis, *lower_pair, STRICT_VISIBILITY)
            limb_vals[chain_name] = None if (up is None or lo is None) else (up, lo)

        def merge(side_a, side_b):
            a, b = limb_vals[side_a], limb_vals[side_b]
            usable = [v for v in (a, b) if v is not None]
            if not usable:
                return None
            up = float(np.mean([v[0] for v in usable]))
            lo = float(np.mean([v[1] for v in usable]))
            return up, lo, len(usable)

        front = merge("front_L", "front_R")
        hind = merge("hind_L", "hind_R")
        if front is None:
            reject["incomplete_front"] += 1
            continue
        if hind is None:
            reject["incomplete_hind"] += 1
            continue

        f_up, f_lo, f_n = front
        h_up, h_lo, h_n = hind

        muzzle = bone_length(xy, vis, "Nose", "Neck", STRICT_VISIBILITY)

        rec = {
            "ann_id": ann["id"],
            "image_id": ann["image_id"],
            "file_name": images_by_id.get(ann["image_id"], {}).get("file_name", ""),
            "spine_px": spine,
            "horizontality": horizontality,
            "n_front_sides": f_n,
            "n_hind_sides": h_n,
            "front_upper": f_up / spine,
            "front_lower": f_lo / spine,
            "front_total": (f_up + f_lo) / spine,
            "hind_upper": h_up / spine,
            "hind_lower": h_lo / spine,
            "hind_total": (h_up + h_lo) / spine,
            "muzzle": (muzzle / spine) if muzzle is not None else None,
        }
        rec["leg_to_spine"] = 0.5 * (rec["front_total"] + rec["hind_total"])
        rec["hind_over_front"] = rec["hind_total"] / rec["front_total"]
        records.append(rec)

    print("Rejections:")
    for reason, count in reject.items():
        print(f"  {reason:20} {count:5}")
    print(f"\nAccepted samples: {len(records)}  "
          f"({100.0 * len(records) / max(1, len(dog_anns)):.1f}% of dog annotations)\n")

    if len(records) < 20:
        print("!! Too few samples to characterise a distribution.")
        print("!! Try relaxing MIN_HORIZONTALITY or STRICT_VISIBILITY (set to 1).")
        return

    def stats(key):
        vals = np.array([r[key] for r in records if r[key] is not None])
        return vals, {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p33": float(np.percentile(vals, 33.3)),
            "p67": float(np.percentile(vals, 66.7)),
            "p90": float(np.percentile(vals, 90)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    print(f"{'ratio':16} {'n':>5} {'median':>8} {'std':>7} "
          f"{'p10':>7} {'p33':>7} {'p67':>7} {'p90':>7}")
    print("-" * 72)
    summary = {}
    for key in ["leg_to_spine", "front_total", "hind_total",
                "front_upper", "front_lower", "hind_upper", "hind_lower",
                "hind_over_front", "muzzle"]:
        vals, s = stats(key)
        summary[key] = s
        print(f"{key:16} {s['n']:5} {s['median']:8.3f} {s['std']:7.3f} "
              f"{s['p10']:7.3f} {s['p33']:7.3f} {s['p67']:7.3f} {s['p90']:7.3f}")

    s = summary["leg_to_spine"]
    spread = s["p90"] / s["p10"] if s["p10"] > 0 else float("nan")
    print(f"\nleg_to_spine p90/p10 spread = {spread:.2f}x")
    print("  (a real short-leg vs long-leg population should show >= ~1.6x;")
    print("   much less than that means the ratio alone will not separate breeds)")
    print(f"\nPreview of an equal-thirds split on leg_to_spine:")
    print(f"  small  : leg_to_spine <  {s['p33']:.3f}")
    print(f"  medium : {s['p33']:.3f} - {s['p67']:.3f}")
    print(f"  large  : leg_to_spine >  {s['p67']:.3f}")

    out_json = os.path.join(OUT_DIR, "dog_ratio_dataset.json")
    with open(out_json, "w") as f:
        json.dump({
            "source": "AP-10K ground-truth keypoints",
            "config": {
                "min_spine_px": MIN_SPINE_PX,
                "min_horizontality": MIN_HORIZONTALITY,
                "strict_visibility": STRICT_VISIBILITY,
            },
            "n_dog_annotations": len(dog_anns),
            "n_accepted": len(records),
            "rejections": reject,
            "summary": summary,
            "records": records,
        }, f, indent=2)
    print(f"\nSaved {out_json}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    leg = np.array([r["leg_to_spine"] for r in records])
    axes[0, 0].hist(leg, bins=40, color="steelblue", edgecolor="white")
    for p, c in [(s["p33"], "darkorange"), (s["p67"], "darkorange")]:
        axes[0, 0].axvline(p, color=c, ls="--", lw=1.5)
    axes[0, 0].set_title("leg_to_spine (mean of front & hind leg / spine)")
    axes[0, 0].set_xlabel("ratio")
    axes[0, 0].set_ylabel("count")

    ft = np.array([r["front_total"] for r in records])
    ht = np.array([r["hind_total"] for r in records])
    axes[0, 1].hist(ft, bins=40, alpha=0.6, label="front_total", color="seagreen")
    axes[0, 1].hist(ht, bins=40, alpha=0.6, label="hind_total", color="indianred")
    axes[0, 1].legend()
    axes[0, 1].set_title("front vs hind leg length / spine")
    axes[0, 1].set_xlabel("ratio")

    axes[1, 0].scatter(ft, ht, s=8, alpha=0.4, color="slateblue")
    axes[1, 0].set_xlabel("front_total / spine")
    axes[1, 0].set_ylabel("hind_total / spine")
    axes[1, 0].set_title("front vs hind (clusters would show as blobs)")

    hz = np.array([r["horizontality"] for r in records])
    axes[1, 1].scatter(hz, leg, s=8, alpha=0.4, color="darkcyan")
    axes[1, 1].set_xlabel("horizontality (view quality)")
    axes[1, 1].set_ylabel("leg_to_spine")
    axes[1, 1].set_title("view-angle confound check\n(a slope here = perspective bias, not breed)")

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "ratio_distribution.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

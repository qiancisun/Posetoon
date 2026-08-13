import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmpose.apis import MMPoseInferencer

AP10K_IMG_DIR = "data/ap10k/data"
RATIO_DATASET = "outputs/dog_ratio_dataset.json"
OUT_DIR = "outputs"

SCORE_THRESHOLD = 0.35
MIN_SPINE_PX = 40.0
MIN_HORIZONTALITY = 0.70

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


def bone_length(xy, scores, a, b, min_score):
    ia, ib = K[a], K[b]
    if scores[ia] < min_score or scores[ib] < min_score:
        return None
    return float(np.linalg.norm(np.array(xy[ib]) - np.array(xy[ia])))


def measure_leg_to_spine(xy, scores):
    spine = bone_length(xy, scores, "Neck", "root_of_tail", SCORE_THRESHOLD)
    if spine is None or spine < MIN_SPINE_PX:
        return None, "spine"
    dx = abs(xy[K["root_of_tail"]][0] - xy[K["Neck"]][0])
    if dx / spine < MIN_HORIZONTALITY:
        return None, "view"

    def merge(side_a, side_b):
        vals = []
        for (upper_pair, lower_pair) in [LIMB_CHAINS[side_a], LIMB_CHAINS[side_b]]:
            up = bone_length(xy, scores, *upper_pair, SCORE_THRESHOLD)
            lo = bone_length(xy, scores, *lower_pair, SCORE_THRESHOLD)
            if up is not None and lo is not None:
                vals.append(up + lo)
        return float(np.mean(vals)) if vals else None

    front = merge("front_L", "front_R")
    hind = merge("hind_L", "hind_R")
    if front is None or hind is None:
        return None, "limbs"
    return 0.5 * (front / spine + hind / spine), "ok"


def main():
    with open(RATIO_DATASET, "r") as f:
        gt_data = json.load(f)
    records = gt_data["records"]
    print(f"AP-10K samples that passed step2b's view filter: {len(records)}")

    print("Loading MMPose inferencer...")
    inferencer = MMPoseInferencer("animal")
    print("Model loaded.\n")

    inferred_ratios = []
    gt_ratios_same_samples = []
    reject_counts = {"spine": 0, "view": 0, "limbs": 0, "no_detection": 0}

    for i, rec in enumerate(records):
        img_path = os.path.join(AP10K_IMG_DIR, rec["file_name"])
        if not os.path.exists(img_path):
            continue
        try:
            gen = inferencer(img_path, show=False, return_vis=False)
            res = [r for r in gen]
            preds = res[0]["predictions"][0]
            if not preds:
                reject_counts["no_detection"] += 1
                continue
            xy = preds[0]["keypoints"]
            scores = preds[0]["keypoint_scores"]
        except Exception:
            reject_counts["no_detection"] += 1
            continue

        val, status = measure_leg_to_spine(xy, scores)
        if val is None:
            reject_counts[status] += 1
            continue

        inferred_ratios.append(val)
        gt_ratios_same_samples.append(rec["leg_to_spine"])

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(records)}")

    print(f"\nInference-usable samples: {len(inferred_ratios)}/{len(records)}")
    print(f"Rejections: {reject_counts}")

    if len(inferred_ratios) < 30:
        print("Too few samples survived inference -- cannot recompute thresholds reliably.")
        return

    inferred_ratios = np.array(inferred_ratios)
    gt_ratios_same_samples = np.array(gt_ratios_same_samples)

    p33 = float(np.percentile(inferred_ratios, 33.3))
    p67 = float(np.percentile(inferred_ratios, 66.7))

    print(f"\nInference-domain thresholds:")
    print(f"  p33 = {p33:.3f}   (GT-domain was {gt_data['summary']['leg_to_spine']['p33']:.3f})")
    print(f"  p67 = {p67:.3f}   (GT-domain was {gt_data['summary']['leg_to_spine']['p67']:.3f})")

    diff = inferred_ratios - gt_ratios_same_samples
    print(f"\nPaired GT-vs-inference gap on the same {len(diff)} images:")
    print(f"  mean diff (inferred - GT) = {np.mean(diff):.4f}")
    print(f"  median diff              = {np.median(diff):.4f}")
    print(f"  std of diff              = {np.std(diff):.4f}")
    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(gt_ratios_same_samples, inferred_ratios)
        print(f"  Pearson r (GT vs inferred) = {r:.3f} (p={p:.2e})")
    except ImportError:
        pass

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "inferred_thresholds.json")
    with open(out_json, "w") as f:
        json.dump({
            "n": len(inferred_ratios),
            "p33": p33,
            "p67": p67,
            "gt_domain_p33": gt_data["summary"]["leg_to_spine"]["p33"],
            "gt_domain_p67": gt_data["summary"]["leg_to_spine"]["p67"],
            "mean_gt_minus_inferred_gap": float(-np.mean(diff)),
            "inferred_ratios": inferred_ratios.tolist(),
        }, f, indent=2)
    print(f"\nSaved {out_json}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(gt_ratios_same_samples, bins=30, alpha=0.6, label="GT keypoints", color="steelblue")
    ax[0].hist(inferred_ratios, bins=30, alpha=0.6, label="MMPose inferred", color="tomato")
    ax[0].axvline(gt_data["summary"]["leg_to_spine"]["p33"], color="steelblue", ls="--")
    ax[0].axvline(gt_data["summary"]["leg_to_spine"]["p67"], color="steelblue", ls="--")
    ax[0].axvline(p33, color="tomato", ls="--")
    ax[0].axvline(p67, color="tomato", ls="--")
    ax[0].set_xlabel("leg_to_spine")
    ax[0].set_title("GT vs inferred distribution\n(dashed = each domain's own p33/p67)")
    ax[0].legend()

    ax[1].scatter(gt_ratios_same_samples, inferred_ratios, s=10, alpha=0.4, color="purple")
    lims = [min(gt_ratios_same_samples.min(), inferred_ratios.min()),
            max(gt_ratios_same_samples.max(), inferred_ratios.max())]
    ax[1].plot(lims, lims, "k--", lw=1, label="y=x")
    ax[1].set_xlabel("GT leg_to_spine")
    ax[1].set_ylabel("Inferred leg_to_spine")
    ax[1].set_title("Same-image paired comparison")
    ax[1].legend()

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "gt_vs_inferred.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

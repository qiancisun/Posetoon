import os
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmpose.apis import MMPoseInferencer

STANFORD_DIR = "images/Images"
RATIO_DATASET = "outputs/dog_ratio_dataset.json"
OUT_DIR = "outputs"
N_PER_BREED = 20

MIN_SPINE_PX = 40.0
MIN_HORIZONTALITY = 0.70
SCORE_THRESHOLD = 0.35

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

EXPECTED_BREEDS = {
    "basset":               ("small",  "Basset Hound"),
    "Pekinese":             ("small",  "Pekinese"),
    "Chihuahua":            ("small",  "Chihuahua"),
    "Cardigan":             ("small",  "Cardigan Corgi"),
    "beagle":               ("medium", "Beagle"),
    "cocker_spaniel":       ("medium", "Cocker Spaniel"),
    "Labrador_retriever":   ("medium", "Labrador Retriever"),
    "borzoi":               ("large",  "Borzoi"),
    "Great_Dane":           ("large",  "Great Dane"),
    "German_shepherd":      ("large",  "German Shepherd"),
    "Afghan_hound":         ("large",  "Afghan Hound"),
    "whippet":              ("large",  "Whippet"),
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


def find_breed_folder(substr):
    candidates = glob.glob(os.path.join(STANFORD_DIR, f"*{substr}*"))
    return candidates[0] if candidates else None


def tier_for(value, p33, p67):
    if value < p33:
        return "small"
    if value > p67:
        return "large"
    return "medium"


def main():
    with open(RATIO_DATASET, "r") as f:
        base = json.load(f)
    p33 = base["summary"]["leg_to_spine"]["p33"]
    p67 = base["summary"]["leg_to_spine"]["p67"]
    print(f"Thresholds from AP-10K (step2b): p33={p33:.3f}  p67={p67:.3f}\n")

    print("Loading MMPose inferencer...")
    inferencer = MMPoseInferencer("animal")
    print("Model loaded.\n")

    results = []
    for substr, (expected_tier, display_name) in EXPECTED_BREEDS.items():
        folder = find_breed_folder(substr)
        if folder is None:
            print(f"[skip] no folder found for '{substr}'")
            continue

        img_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))[: N_PER_BREED * 3]

        ratios = []
        reject_counts = {"spine": 0, "view": 0, "limbs": 0}
        for img_path in img_paths:
            if len(ratios) >= N_PER_BREED:
                break
            try:
                gen = inferencer(img_path, show=False, return_vis=False)
                res = [r for r in gen]
                preds = res[0]["predictions"][0]
                if not preds:
                    continue
                xy = preds[0]["keypoints"]
                scores = preds[0]["keypoint_scores"]
            except Exception as e:
                continue

            val, status = measure_leg_to_spine(xy, scores)
            if val is None:
                reject_counts[status] += 1
                continue
            ratios.append(val)

        if len(ratios) < 5:
            print(f"[{display_name:20}] only {len(ratios)} usable samples "
                  f"(rejects: {reject_counts}) -- too few, skipping")
            continue

        ratios = np.array(ratios)
        median_ratio = float(np.median(ratios))
        measured_tier = tier_for(median_ratio, p33, p67)
        match = "OK " if measured_tier == expected_tier else "MISMATCH"

        print(f"[{display_name:20}] n={len(ratios):3}  median={median_ratio:.3f}  "
              f"expected={expected_tier:6}  measured={measured_tier:6}  {match}")

        results.append({
            "breed": display_name,
            "folder": os.path.basename(folder),
            "n": len(ratios),
            "median_leg_to_spine": median_ratio,
            "all_ratios": ratios.tolist(),
            "expected_tier": expected_tier,
            "measured_tier": measured_tier,
            "match": measured_tier == expected_tier,
        })

    if not results:
        print("\nNo breeds produced usable results -- check STANFORD_DIR path "
              "and folder naming.")
        return

    n_match = sum(r["match"] for r in results)
    print(f"\n{n_match}/{len(results)} breeds matched their expected tier "
          f"({100.0*n_match/len(results):.0f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "breed_validation.json")
    with open(out_json, "w") as f:
        json.dump({
            "thresholds": {"p33": p33, "p67": p67},
            "n_matched": n_match,
            "n_total": len(results),
            "results": results,
        }, f, indent=2)
    print(f"Saved {out_json}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"small": "tomato", "medium": "goldenrod", "large": "seagreen"}

    for i, r in enumerate(sorted(results, key=lambda x: x["median_leg_to_spine"])):
        c = colors[r["expected_tier"]]
        marker = "o" if r["match"] else "x"
        ax.scatter(r["median_leg_to_spine"], i, color=c, marker=marker, s=80)
        ax.text(r["median_leg_to_spine"] + 0.02, i, r["breed"], va="center", fontsize=9)

    ax.axvline(p33, color="gray", ls="--", lw=1)
    ax.axvline(p67, color="gray", ls="--", lw=1)
    ax.set_xlabel("median leg_to_spine")
    ax.set_yticks([])
    ax.set_title("Breed validation: circle = matched expected tier, x = mismatch\n"
                  "(color = expected tier: red=small, gold=medium, green=large)")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "breed_validation.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

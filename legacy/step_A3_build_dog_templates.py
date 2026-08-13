import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
TARGET_SPINE = 200.0

TIER_NAMES = ["small", "medium", "large"]
RATIO_FIELDS = ["front_upper", "front_lower", "hind_upper", "hind_lower", "muzzle"]


def tier_for(value, p33, p67):
    if value < p33:
        return "small"
    if value > p67:
        return "large"
    return "medium"


def robust_stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)) if v.size else None,
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)) if v.size else None,
    }


def build_templates():
    with open(os.path.join(OUT_DIR, "dog_ratio_dataset.json"), "r") as f:
        gt_data = json.load(f)
    with open(os.path.join(OUT_DIR, "inferred_thresholds.json"), "r") as f:
        inf_thr = json.load(f)

    gt_p33 = gt_data["summary"]["leg_to_spine"]["p33"]
    gt_p67 = gt_data["summary"]["leg_to_spine"]["p67"]
    records = gt_data["records"]

    print(f"Splitting {len(records)} GT-domain AP-10K samples using "
          f"GT thresholds p33={gt_p33:.3f} p67={gt_p67:.3f}\n")

    tiers = {name: [] for name in TIER_NAMES}
    for rec in records:
        t = tier_for(rec["leg_to_spine"], gt_p33, gt_p67)
        tiers[t].append(rec)

    templates = {}
    for name in TIER_NAMES:
        recs = tiers[name]
        print(f"[{name:6}] n={len(recs)}")

        field_stats = {}
        for field in RATIO_FIELDS:
            stats = robust_stats([r[field] for r in recs])
            field_stats[field] = stats
            print(f"    {field:12} median={stats['median']:.3f}  "
                  f"iqr={stats['iqr']:.3f}  n={stats['n']}")

        hum_px = field_stats["front_upper"]["median"] * TARGET_SPINE
        rad_px = field_stats["front_lower"]["median"] * TARGET_SPINE
        fem_px = field_stats["hind_upper"]["median"] * TARGET_SPINE
        tib_px = field_stats["hind_lower"]["median"] * TARGET_SPINE
        head_px = field_stats["muzzle"]["median"] * TARGET_SPINE

        measure_dict = {
            "lower": TARGET_SPINE / 2.0,
            "upper": TARGET_SPINE / 2.0,
            "head": head_px,
            "hum": hum_px,
            "rad": rad_px,
            "fem": fem_px,
            "tib": tib_px,
        }
        templates[name] = {
            "measure": measure_dict,
            "n_samples": len(recs),
            "source_ratio_stats": field_stats,
        }
        print()

    out = {
        "target_spine": TARGET_SPINE,
        "spine_split_assumption": "symmetric 50/50 (lower=upper=target_spine/2); "
                                    "see docstring for why",
        "template_content_thresholds_domain": "GT (AP-10K ground-truth keypoints)",
        "template_content_thresholds": {"p33": gt_p33, "p67": gt_p67},
        "runtime_selection_thresholds_domain": "MMPose inference (must match how a "
                                                 "new video's ratio is measured)",
        "runtime_selection_thresholds": {"p33": inf_thr["p33"], "p67": inf_thr["p67"]},
        "templates": templates,
    }

    out_path = os.path.join(OUT_DIR, "dog_templates.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    return out


def plot_templates(templates_out):
    fields = ["hum", "rad", "fem", "tib", "head"]
    labels = ["front upper\n(humerus)", "front lower\n(radius)",
              "hind upper\n(femur)", "hind lower\n(tibia)", "head/muzzle"]
    colors = {"small": "tomato", "medium": "goldenrod", "large": "seagreen"}

    x = np.arange(len(fields))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(TIER_NAMES):
        vals = [templates_out["templates"][name]["measure"][f] for f in fields]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"pixels (spine normalised to {int(templates_out['target_spine'])})")
    ax.set_title("Template bone-segment lengths by tier")
    ax.legend()
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "dog_templates.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


def select_and_blend(video_leg_to_spine, video_measure=None, alpha=0.5,
                      templates_path=os.path.join(OUT_DIR, "dog_templates.json")):
    with open(templates_path, "r") as f:
        data = json.load(f)

    p33 = data["runtime_selection_thresholds"]["p33"]
    p67 = data["runtime_selection_thresholds"]["p67"]
    tier = tier_for(video_leg_to_spine, p33, p67)
    template_measure = data["templates"][tier]["measure"]

    if video_measure is None:
        return tier, dict(template_measure)

    blended = {
        k: (1 - alpha) * template_measure[k] + alpha * video_measure[k]
        for k in template_measure
    }
    return tier, blended


if __name__ == "__main__":
    result = build_templates()
    plot_templates(result)

    print("\n--- demo: select_and_blend() with no live video (pure template) ---")
    for probe_ratio in [0.40, 0.53, 0.65]:
        tier, m = select_and_blend(probe_ratio)
        print(f"leg_to_spine={probe_ratio:.2f} -> tier={tier:6} -> "
              f"hum={m['hum']:.1f} rad={m['rad']:.1f} fem={m['fem']:.1f} tib={m['tib']:.1f}")

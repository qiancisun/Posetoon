import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"


def tier_for(value, p33, p67):
    if value < p33:
        return "small"
    if value > p67:
        return "large"
    return "medium"


def main():
    with open(os.path.join(OUT_DIR, "breed_validation.json"), "r") as f:
        old = json.load(f)
    with open(os.path.join(OUT_DIR, "inferred_thresholds.json"), "r") as f:
        thr = json.load(f)

    p33, p67 = thr["p33"], thr["p67"]
    print(f"Using inference-domain thresholds: p33={p33:.3f}  p67={p67:.3f}\n")

    results = []
    n_match = 0
    for r in old["results"]:
        ratios = np.array(r["all_ratios"])
        median_ratio = float(np.median(ratios))
        measured_tier = tier_for(median_ratio, p33, p67)
        match = measured_tier == r["expected_tier"]
        n_match += match
        flag = "OK " if match else "MISMATCH"
        print(f"[{r['breed']:20}] median={median_ratio:.3f}  "
              f"expected={r['expected_tier']:6}  measured={measured_tier:6}  {flag}")
        results.append({**r, "measured_tier_v2": measured_tier, "match_v2": match})

    print(f"\n{n_match}/{len(old['results'])} breeds matched "
          f"({100.0*n_match/len(old['results']):.0f}%) "
          f"-- was {old['n_matched']}/{old['n_total']} "
          f"({100.0*old['n_matched']/old['n_total']:.0f}%) with GT-domain thresholds")

    out_json = os.path.join(OUT_DIR, "breed_validation_v2.json")
    with open(out_json, "w") as f:
        json.dump({
            "thresholds": {"p33": p33, "p67": p67},
            "n_matched": n_match,
            "n_total": len(old["results"]),
            "results": results,
        }, f, indent=2)
    print(f"Saved {out_json}")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"small": "tomato", "medium": "goldenrod", "large": "seagreen"}
    for i, r in enumerate(sorted(results, key=lambda x: np.median(x["all_ratios"]))):
        med = float(np.median(r["all_ratios"]))
        c = colors[r["expected_tier"]]
        marker = "o" if r["match_v2"] else "x"
        ax.scatter(med, i, color=c, marker=marker, s=80)
        ax.text(med + 0.01, i, r["breed"], va="center", fontsize=9)
    ax.axvline(p33, color="gray", ls="--", lw=1)
    ax.axvline(p67, color="gray", ls="--", lw=1)
    ax.set_xlabel("median leg_to_spine (inference domain)")
    ax.set_yticks([])
    ax.set_title("Breed validation v2: inference-domain thresholds\n"
                  "circle = matched, x = mismatch")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "breed_validation_v2.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

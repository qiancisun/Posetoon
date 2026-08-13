import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ablation_E1_alpha_stability import (
    measure_proportions, video_ratio, tier_for, blend, prop_distance,
    FIELDS, TARGET_SPINE, STAB_PATH, TEMPLATES_PATH, OUT_DIR,
)

WINDOW_SIZES = [5, 10, 20, 40, 80, 139]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_DRAWS = 200
SEED = 0


def centroid_measure(measures):
    return {k: float(np.mean([m[k] for m in measures])) for k in FIELDS}


def main():
    with open(STAB_PATH, "r") as f:
        frames = json.load(f)
    with open(TEMPLATES_PATH, "r") as f:
        tdata = json.load(f)

    p33 = tdata["runtime_selection_thresholds"]["p33"]
    p67 = tdata["runtime_selection_thresholds"]["p67"]

    n = len(frames)
    m_full = measure_proportions(frames)
    tier_full = tier_for(video_ratio(m_full), p33, p67)
    print(f"Frames: {n}   full-video tier: {tier_full}   "
          f"leg_to_spine: {video_ratio(m_full):.3f}\n")

    rng = np.random.default_rng(SEED)
    results = []

    print(f"{'k':>5} {'tier_flip':>10} " + " ".join(f"{'a=' + str(a):>9}" for a in ALPHAS))
    print("-" * (16 + 10 * len(ALPHAS)))

    for k in WINDOW_SIZES:
        if k > n:
            continue
        window_measures, window_tiers = [], []
        for _ in range(N_DRAWS):
            start = int(rng.integers(0, n - k + 1))
            try:
                m = measure_proportions(frames[start:start + k])
            except (ValueError, IndexError):
                continue
            window_measures.append(m)
            window_tiers.append(tier_for(video_ratio(m), p33, p67))

        if len(window_measures) < 10:
            continue

        tier_flip = float(np.mean([t != tier_full for t in window_tiers]))

        row = {"k": k, "n_draws": len(window_measures), "tier_flip": tier_flip,
               "spread": {}}
        spreads = []
        for alpha in ALPHAS:
            chars = [blend(tdata["templates"][t]["measure"], m, alpha)
                     for m, t in zip(window_measures, window_tiers)]
            cen = centroid_measure(chars)
            spread = float(np.mean([prop_distance(c, cen) for c in chars]))
            row["spread"][str(alpha)] = spread
            spreads.append(spread)
        results.append(row)

        print(f"{k:5} {tier_flip:10.3f} " + " ".join(f"{s:9.4f}" for s in spreads))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "ablation_E1b_sample_size.json")
    with open(out_json, "w") as fh:
        json.dump({"n_frames": n, "tier_full": tier_full,
                   "n_draws": N_DRAWS, "results": results}, fh, indent=2)
    print(f"\nSaved {out_json}")

    print("\nHow to read this:")
    print("  - spread falling as alpha decreases means the template IS")
    print("    absorbing measurement noise at that window length.")
    print("  - if spread is flat across alpha, the template adds nothing")
    print("    at that length and alpha should be chosen on other grounds.")
    print("  - tier_flip > 0 means short clips can select the wrong")
    print("    template entirely, which blending cannot fix -- that is an")
    print("    argument for a minimum-frames requirement, not for a")
    print("    particular alpha.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ks = [r["k"] for r in results]
    for alpha in ALPHAS:
        ax[0].plot(ks, [r["spread"][str(alpha)] for r in results],
                   "o-", label=f"α={alpha}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("frames used (log scale)")
    ax[0].set_ylabel("character spread across random clips")
    ax[0].set_title("E1b: does the template reduce variance?\n"
                    "separation between lines = template is helping")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(ks, [r["tier_flip"] for r in results], "o-", color="indianred")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("frames used (log scale)")
    ax[1].set_ylabel("fraction selecting a different tier")
    ax[1].set_title("E1b: tier-selection robustness\n"
                    "blending cannot repair a wrong tier")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "ablation_E1b_sample_size.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

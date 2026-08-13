import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
STAB_PATH = "stabilised_keypoints.json"
TEMPLATES_PATH = os.path.join(OUT_DIR, "dog_templates.json")

TARGET_SPINE = 200.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
KP = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
FIELDS = ["lower", "upper", "head", "hum", "rad", "fem", "tib"]


def get_kp(frame, name):
    return np.array(frame["keypoints"][KP[name]], dtype=float)


def measure_proportions(frames, target_spine=TARGET_SPINE):
    acc = {k: [] for k in FIELDS}
    for fr in frames:
        neck = get_kp(fr, "Neck")
        tail = get_kp(fr, "root_of_tail")
        sl = np.linalg.norm(neck - tail)
        if sl < 1e-6:
            continue
        sc = target_spine / sl
        hips = (get_kp(fr, "L_Hip") + get_kp(fr, "R_Hip")) / 2
        pelvis = tail * 0.55 + hips * 0.45
        waist = (pelvis + neck) / 2

        acc["lower"].append(np.linalg.norm(waist - pelvis) * sc)
        acc["upper"].append(np.linalg.norm(neck - waist) * sc)
        acc["head"].append(np.linalg.norm(get_kp(fr, "Nose") - neck) * sc)
        acc["hum"].append(np.linalg.norm(get_kp(fr, "L_Elbow") - get_kp(fr, "L_Shoulder")) * sc)
        acc["rad"].append(np.linalg.norm(get_kp(fr, "L_F_Paw") - get_kp(fr, "L_Elbow")) * sc)
        acc["fem"].append(np.linalg.norm(get_kp(fr, "L_Knee") - get_kp(fr, "L_Hip")) * sc)
        acc["tib"].append(np.linalg.norm(get_kp(fr, "L_B_Paw") - get_kp(fr, "L_Knee")) * sc)

    m = {}
    for k in ["lower", "upper", "head"]:
        m[k] = float(np.median(acc[k]))
    for k in ["hum", "rad", "fem", "tib"]:
        m[k] = float(np.percentile(acc[k], 75))
    return m


def video_ratio(measure):
    return 0.5 * ((measure["hum"] + measure["rad"])
                   + (measure["fem"] + measure["tib"])) / TARGET_SPINE


def tier_for(v, p33, p67):
    return "small" if v < p33 else ("large" if v > p67 else "medium")


def blend(template_measure, individual_measure, alpha):
    return {k: (1 - alpha) * template_measure[k] + alpha * individual_measure[k]
            for k in FIELDS}


def prop_distance(a, b, reference=TARGET_SPINE):
    return float(np.mean([abs(a[k] - b[k]) for k in FIELDS]) / reference)


def main():
    with open(STAB_PATH, "r") as f:
        frames = json.load(f)
    with open(TEMPLATES_PATH, "r") as f:
        tdata = json.load(f)

    p33 = tdata["runtime_selection_thresholds"]["p33"]
    p67 = tdata["runtime_selection_thresholds"]["p67"]

    n = len(frames)
    half = n // 2
    print(f"Frames: {n}  (half A: 0-{half - 1}, half B: {half}-{n - 1})\n")

    m_full = measure_proportions(frames)
    m_a = measure_proportions(frames[:half])
    m_b = measure_proportions(frames[half:])

    print("Full-video measurement (compare against measure_skeleton's print):")
    for k in FIELDS:
        print(f"  {k:6}: {m_full[k]:6.1f}px")

    r_full, r_a, r_b = video_ratio(m_full), video_ratio(m_a), video_ratio(m_b)
    t_full, t_a, t_b = (tier_for(r_full, p33, p67),
                        tier_for(r_a, p33, p67),
                        tier_for(r_b, p33, p67))
    print(f"\nleg_to_spine  full={r_full:.3f} ({t_full})  "
          f"halfA={r_a:.3f} ({t_a})  halfB={r_b:.3f} ({t_b})")
    if not (t_full == t_a == t_b):
        print("  NOTE: the halves select DIFFERENT tiers. That is itself a")
        print("  stability finding worth reporting -- tier selection is not")
        print("  invariant to which frames were observed.")

    rows = []
    print(f"\n{'alpha':>6} {'fidelity':>10} {'stability':>10}   (both lower = better)")
    print("-" * 42)
    for alpha in ALPHAS:
        char_full = blend(tdata["templates"][t_full]["measure"], m_full, alpha)
        char_a = blend(tdata["templates"][t_a]["measure"], m_a, alpha)
        char_b = blend(tdata["templates"][t_b]["measure"], m_b, alpha)

        fidelity = prop_distance(char_full, m_full)
        stability = prop_distance(char_a, char_b)
        rows.append({"alpha": alpha, "fidelity": fidelity, "stability": stability,
                     "character": char_full})
        print(f"{alpha:6.2f} {fidelity:10.4f} {stability:10.4f}")

    f = np.array([r["fidelity"] for r in rows])
    s = np.array([r["stability"] for r in rows])
    fn = (f - f.min()) / (f.ptp() + 1e-12)
    sn = (s - s.min()) / (s.ptp() + 1e-12)
    best = int(np.argmin(fn + sn))
    print(f"\nEqual-weight balance point: alpha = {rows[best]['alpha']}")
    print("  (equal weighting is an assumption, not a result -- state it)")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, "ablation_E1_alpha.json")
    with open(out_json, "w") as fh:
        json.dump({
            "n_frames": n,
            "measure_full": m_full, "measure_half_a": m_a, "measure_half_b": m_b,
            "ratio_full": r_full, "ratio_half_a": r_a, "ratio_half_b": r_b,
            "tier_full": t_full, "tier_half_a": t_a, "tier_half_b": t_b,
            "rows": rows,
            "balance_alpha": rows[best]["alpha"],
        }, fh, indent=2)
    print(f"\nSaved {out_json}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].plot(f, s, "o-", color="steelblue")
    for r in rows:
        ax[0].annotate(f"α={r['alpha']}", (r["fidelity"], r["stability"]),
                       textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax[0].set_xlabel("fidelity cost  (distance to this dog's measurement)")
    ax[0].set_ylabel("instability  (difference between halves)")
    ax[0].set_title("E1: fidelity vs stability trade-off\nlower-left is better on both")
    ax[0].grid(alpha=0.3)

    alphas = [r["alpha"] for r in rows]
    ax[1].plot(alphas, f, "o-", label="fidelity cost", color="indianred")
    ax[1].plot(alphas, s, "s-", label="instability", color="seagreen")
    ax[1].axvline(rows[best]["alpha"], color="gray", ls="--", lw=1,
                  label=f"balance α={rows[best]['alpha']}")
    ax[1].set_xlabel("α (individualisation strength)")
    ax[1].set_ylabel("normalised proportion distance")
    ax[1].set_title("E1: both costs against α")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "ablation_E1_alpha.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT_DIR = "outputs"
TIER_NAMES = ["small", "medium", "large"]


EAR_ANGLE = {"floppy": -75.0, "semi_erect": -20.0, "erect": 55.0}

TIER_DEFAULT_APPEARANCE = {
    "small": {
        "ear_type": "erect", "ear_len": 0.13, "ear_width": 0.048,
        "tail_len": 0.26, "tail_curl": 0.45,
        "muzzle_len": 0.38, "coat": "medium",
    },
    "medium": {
        "ear_type": "floppy", "ear_len": 0.20, "ear_width": 0.054,
        "tail_len": 0.31, "tail_curl": 0.20,
        "muzzle_len": 0.43, "coat": "short",
    },
    "large": {
        "ear_type": "semi_erect", "ear_len": 0.15, "ear_width": 0.052,
        "tail_len": 0.36, "tail_curl": 0.10,
        "muzzle_len": 0.65, "coat": "short",
    },
}

BREED_APPEARANCE = {
    "basset":            {"ear_type": "floppy", "ear_len": 0.22, "ear_width": 0.060,
                          "tail_len": 0.30, "tail_curl": 0.05, "muzzle_len": 0.45, "coat": "short"},
    "pekingese":         {"ear_type": "floppy", "ear_len": 0.16, "ear_width": 0.050,
                          "tail_len": 0.22, "tail_curl": 0.85, "muzzle_len": 0.18, "coat": "long"},
    "pekinese":          {"ear_type": "floppy", "ear_len": 0.16, "ear_width": 0.050,
                          "tail_len": 0.22, "tail_curl": 0.85, "muzzle_len": 0.18, "coat": "long"},
    "chihuahua":         {"ear_type": "erect", "ear_len": 0.13, "ear_width": 0.055,
                          "tail_len": 0.26, "tail_curl": 0.40, "muzzle_len": 0.30, "coat": "short"},
    "cardigan":          {"ear_type": "erect", "ear_len": 0.13, "ear_width": 0.055,
                          "tail_len": 0.32, "tail_curl": 0.15, "muzzle_len": 0.40, "coat": "medium"},
    "pembroke":          {"ear_type": "erect", "ear_len": 0.12, "ear_width": 0.052,
                          "tail_len": 0.08, "tail_curl": 0.10, "muzzle_len": 0.40, "coat": "medium"},
    "beagle":            {"ear_type": "floppy", "ear_len": 0.2, "ear_width": 0.055,
                          "tail_len": 0.30, "tail_curl": 0.25, "muzzle_len": 0.42, "coat": "short"},
    "cocker_spaniel":    {"ear_type": "floppy", "ear_len": 0.20, "ear_width": 0.058,
                          "tail_len": 0.24, "tail_curl": 0.10, "muzzle_len": 0.38, "coat": "long"},
    "labrador":          {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.048,
                          "tail_len": 0.34, "tail_curl": 0.08, "muzzle_len": 0.46, "coat": "short"},
    "borzoi":            {"ear_type": "semi_erect", "ear_len": 0.09, "ear_width": 0.035,
                          "tail_len": 0.42, "tail_curl": 0.20, "muzzle_len": 0.72, "coat": "long"},
    "great_dane":        {"ear_type": "semi_erect", "ear_len": 0.13, "ear_width": 0.050,
                          "tail_len": 0.38, "tail_curl": 0.05, "muzzle_len": 0.62, "coat": "short"},
    "german_shepherd":   {"ear_type": "erect", "ear_len": 0.16, "ear_width": 0.058,
                          "tail_len": 0.38, "tail_curl": 0.15, "muzzle_len": 0.58, "coat": "medium"},
    "afghan_hound":      {"ear_type": "floppy", "ear_len": 0.22, "ear_width": 0.050,
                          "tail_len": 0.36, "tail_curl": 0.35, "muzzle_len": 0.68, "coat": "long"},
    "whippet":           {"ear_type": "semi_erect", "ear_len": 0.08, "ear_width": 0.032,
                          "tail_len": 0.42, "tail_curl": 0.15, "muzzle_len": 0.64, "coat": "short"},
    "pug":               {"ear_type": "floppy", "ear_len": 0.08, "ear_width": 0.042,
                          "tail_len": 0.14, "tail_curl": 0.95, "muzzle_len": 0.16, "coat": "short"},
    "siberian_husky":    {"ear_type": "erect", "ear_len": 0.13, "ear_width": 0.055,
                          "tail_len": 0.36, "tail_curl": 0.55, "muzzle_len": 0.50, "coat": "long"},
    "golden_retriever":  {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.050,
                          "tail_len": 0.36, "tail_curl": 0.10, "muzzle_len": 0.48, "coat": "long"},
    "boxer":             {"ear_type": "semi_erect", "ear_len": 0.11, "ear_width": 0.048,
                          "tail_len": 0.18, "tail_curl": 0.10, "muzzle_len": 0.24, "coat": "short"},
    "rottweiler":        {"ear_type": "floppy", "ear_len": 0.16, "ear_width": 0.050,
                          "tail_len": 0.20, "tail_curl": 0.08, "muzzle_len": 0.44, "coat": "short"},
    "doberman":          {"ear_type": "erect", "ear_len": 0.14, "ear_width": 0.045,
                          "tail_len": 0.16, "tail_curl": 0.05, "muzzle_len": 0.60, "coat": "short"},
    "samoyed":           {"ear_type": "erect", "ear_len": 0.11, "ear_width": 0.050,
                          "tail_len": 0.34, "tail_curl": 0.90, "muzzle_len": 0.46, "coat": "long"},
    "chow":              {"ear_type": "erect", "ear_len": 0.09, "ear_width": 0.048,
                          "tail_len": 0.28, "tail_curl": 0.88, "muzzle_len": 0.32, "coat": "long"},
    "shih-tzu":          {"ear_type": "floppy", "ear_len": 0.13, "ear_width": 0.050,
                          "tail_len": 0.24, "tail_curl": 0.80, "muzzle_len": 0.18, "coat": "long"},
    "maltese":           {"ear_type": "floppy", "ear_len": 0.11, "ear_width": 0.045,
                          "tail_len": 0.26, "tail_curl": 0.75, "muzzle_len": 0.28, "coat": "long"},
    "pomeranian":        {"ear_type": "erect", "ear_len": 0.09, "ear_width": 0.042,
                          "tail_len": 0.26, "tail_curl": 0.90, "muzzle_len": 0.26, "coat": "long"},
    "border_collie":     {"ear_type": "semi_erect", "ear_len": 0.13, "ear_width": 0.052,
                          "tail_len": 0.34, "tail_curl": 0.20, "muzzle_len": 0.50, "coat": "long"},
    "greyhound":         {"ear_type": "semi_erect", "ear_len": 0.09, "ear_width": 0.034,
                          "tail_len": 0.44, "tail_curl": 0.15, "muzzle_len": 0.66, "coat": "short"},
    "saint_bernard":     {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.058,
                          "tail_len": 0.34, "tail_curl": 0.10, "muzzle_len": 0.40, "coat": "long"},
    "french_bulldog":    {"ear_type": "erect", "ear_len": 0.12, "ear_width": 0.062,
                          "tail_len": 0.08, "tail_curl": 0.30, "muzzle_len": 0.16, "coat": "short"},
    "poodle":            {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.052,
                          "tail_len": 0.24, "tail_curl": 0.15, "muzzle_len": 0.54, "coat": "long"},
}


BREED_FAMILY_APPEARANCE = [
    ("retriever", {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.052,
                    "tail_len": 0.35, "tail_curl": 0.10, "muzzle_len": 0.47, "coat": "medium"}),
    ("spaniel",   {"ear_type": "floppy", "ear_len": 0.19, "ear_width": 0.056,
                    "tail_len": 0.26, "tail_curl": 0.12, "muzzle_len": 0.38, "coat": "long"}),
    ("setter",    {"ear_type": "floppy", "ear_len": 0.16, "ear_width": 0.052,
                    "tail_len": 0.36, "tail_curl": 0.10, "muzzle_len": 0.56, "coat": "long"}),
    ("wolfhound", {"ear_type": "semi_erect", "ear_len": 0.10, "ear_width": 0.038,
                    "tail_len": 0.42, "tail_curl": 0.15, "muzzle_len": 0.68, "coat": "long"}),
    ("deerhound", {"ear_type": "semi_erect", "ear_len": 0.10, "ear_width": 0.038,
                    "tail_len": 0.42, "tail_curl": 0.15, "muzzle_len": 0.68, "coat": "long"}),
    ("hound",     {"ear_type": "floppy", "ear_len": 0.17, "ear_width": 0.055,
                    "tail_len": 0.32, "tail_curl": 0.20, "muzzle_len": 0.50, "coat": "short"}),
    ("terrier",   {"ear_type": "semi_erect", "ear_len": 0.10, "ear_width": 0.044,
                    "tail_len": 0.22, "tail_curl": 0.15, "muzzle_len": 0.36, "coat": "medium"}),
    ("sheepdog",  {"ear_type": "semi_erect", "ear_len": 0.12, "ear_width": 0.050,
                    "tail_len": 0.34, "tail_curl": 0.20, "muzzle_len": 0.48, "coat": "long"}),
    ("collie",    {"ear_type": "semi_erect", "ear_len": 0.13, "ear_width": 0.052,
                    "tail_len": 0.34, "tail_curl": 0.20, "muzzle_len": 0.50, "coat": "long"}),
    ("mastiff",   {"ear_type": "floppy", "ear_len": 0.12, "ear_width": 0.052,
                    "tail_len": 0.28, "tail_curl": 0.15, "muzzle_len": 0.26, "coat": "short"}),
    ("bulldog",   {"ear_type": "semi_erect", "ear_len": 0.10, "ear_width": 0.055,
                    "tail_len": 0.10, "tail_curl": 0.35, "muzzle_len": 0.18, "coat": "short"}),
    ("schnauzer", {"ear_type": "semi_erect", "ear_len": 0.11, "ear_width": 0.046,
                    "tail_len": 0.20, "tail_curl": 0.12, "muzzle_len": 0.42, "coat": "medium"}),
    ("poodle",    {"ear_type": "floppy", "ear_len": 0.15, "ear_width": 0.052,
                    "tail_len": 0.24, "tail_curl": 0.15, "muzzle_len": 0.54, "coat": "long"}),
]


def appearance_for(tier, breed_name=None):
    params = dict(TIER_DEFAULT_APPEARANCE[tier])
    if not breed_name:
        return params, f"tier_default:{tier}"

    key = breed_name.lower().replace(" ", "_").replace("-", "_")

    for breed_key, breed_params in BREED_APPEARANCE.items():
        if breed_key in key:
            params.update(breed_params)
            return params, f"breed_table:{breed_key}"

    for family_key, family_params in BREED_FAMILY_APPEARANCE:
        if family_key in key:
            params.update(family_params)
            return params, f"breed_family:{family_key}"

    return params, f"tier_default:{tier} (breed '{breed_name}' unmatched)"


def taper_quad(x0, y0, L, w0, w1, angle_deg):
    a = np.radians(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    nx, ny = -dy, dx
    base = np.array([x0, y0])
    end = base + np.array([dx, dy]) * L
    return np.array([base + np.array([nx, ny]) * w0,
                     base - np.array([nx, ny]) * w0,
                     end - np.array([nx, ny]) * w1,
                     end + np.array([nx, ny]) * w1]), end


def curved_tail(x0, y0, L, curl, ref, n=14):
    total_sweep = np.radians(20 + 180.0 * curl)
    pts, ang = [np.array([x0, y0])], np.radians(150.0)
    step = L / n
    for i in range(n):
        ang -= total_sweep / n
        pts.append(pts[-1] + np.array([np.cos(ang), np.sin(ang)]) * step)
    pts = np.array(pts)
    widths = np.linspace(ref * 0.026, ref * 0.012, len(pts))
    dirs = np.gradient(pts, axis=0)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    top = pts + normals * widths[:, None]
    bot = pts - normals * widths[:, None]
    return np.vstack([top, bot[::-1]])


def draw_dog(ax, measure, appear, ref, color, title):
    spine_len = measure["lower"] + measure["upper"]
    body_w = ref * 0.16
    body_poly, _ = taper_quad(0, 0, spine_len, body_w, ref * 0.13, 0)
    ax.add_patch(Polygon(body_poly, closed=True, facecolor=color,
                          edgecolor="black", linewidth=1.2, alpha=0.85, zorder=5))

    total_head = appear["muzzle_len"] * ref
    skull_len = total_head * 0.45
    muz_len = total_head * 0.55
    sh = ref * 0.10

    skull = np.array([
        (0.0, -sh * 0.85), (skull_len * 0.30, -sh * 1.00),
        (skull_len * 0.80, -sh * 0.90), (skull_len, -sh * 0.60),
        (skull_len, sh * 0.60), (skull_len * 0.75, sh * 0.95),
        (skull_len * 0.25, sh * 1.00), (0.0, sh * 0.90),
    ]) + np.array([spine_len, 0.0])
    ax.add_patch(Polygon(skull, closed=True, facecolor=color,
                          edgecolor="black", linewidth=1.0, alpha=0.9, zorder=6))

    muzzle_poly, _ = taper_quad(spine_len + skull_len, -sh * 0.15, muz_len,
                                  sh * 0.55, sh * 0.35, 0)
    ax.add_patch(Polygon(muzzle_poly, closed=True, facecolor=color,
                          edgecolor="black", linewidth=1.0, alpha=0.9, zorder=6))

    ear_x = spine_len + skull_len * 0.25
    ear_poly, _ = taper_quad(ear_x, sh * 0.80,
                              appear["ear_len"] * ref,
                              appear["ear_width"] * ref,
                              appear["ear_width"] * ref * 0.55,
                              EAR_ANGLE[appear["ear_type"]])
    ax.add_patch(Polygon(ear_poly, closed=True, facecolor=color,
                          edgecolor="black", linewidth=1.0, alpha=0.95, zorder=7))

    tail_poly = curved_tail(0, -ref * 0.02, appear["tail_len"] * ref,
                             appear["tail_curl"], ref)
    ax.add_patch(Polygon(tail_poly, closed=True, facecolor=color,
                          edgecolor="black", linewidth=1.0, alpha=0.85,
                          zorder=6 if appear["tail_curl"] > 0.5 else 3))

    for x, upper, lower, w in [
        (spine_len * 0.85, measure["hum"], measure["rad"], ref * 0.045),
        (spine_len * 0.15, measure["fem"], measure["tib"], ref * 0.050),
    ]:
        up_poly, joint = taper_quad(x, -body_w * 0.6, upper, w, w * 0.66, -90)
        ax.add_patch(Polygon(up_poly, closed=True, facecolor=color,
                              edgecolor="black", linewidth=1.0, alpha=0.75, zorder=4))
        lo_poly, _ = taper_quad(joint[0], joint[1], lower, w * 0.66, w * 0.40, -90)
        ax.add_patch(Polygon(lo_poly, closed=True, facecolor=color,
                              edgecolor="black", linewidth=1.0, alpha=0.75, zorder=4))

    ax.set_title(f"{title}\near={appear['ear_type']}  "
                 f"tail_curl={appear['tail_curl']:.2f}  coat={appear['coat']}",
                 fontsize=9)


def main():
    with open(os.path.join(OUT_DIR, "dog_templates.json"), "r") as f:
        data = json.load(f)
    ref = data["target_spine"]

    for tier in TIER_NAMES:
        params, source = appearance_for(tier)
        data["templates"][tier]["appearance"] = params
        data["templates"][tier]["appearance_source"] = source

    data["appearance_note"] = (
        "Appearance parameters (ear/tail/coat) are hand-authored knowledge, "
        "not measured from AP-10K -- its 17 keypoints contain no ear or "
        "tail-shape landmarks. muzzle_len is the exception: it is "
        "geometrically measurable, but step6_rig_v2.py's build_rig() "
        "currently clips all tiers to the same head length, so it is "
        "re-emitted here as an explicit appearance parameter."
    )
    data["ear_angle_map"] = EAR_ANGLE

    out_json = os.path.join(OUT_DIR, "dog_templates_v2.json")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {out_json}")

    for tier in TIER_NAMES:
        a = data["templates"][tier]["appearance"]
        print(f"[{tier:6}] ear={a['ear_type']:10} ear_len={a['ear_len']:.2f}  "
              f"tail_len={a['tail_len']:.2f} curl={a['tail_curl']:.2f}  "
              f"muzzle={a['muzzle_len']:.2f}  coat={a['coat']}")

    colors = {"small": "#D9784F", "medium": "#C99A2E", "large": "#3E7C5A"}
    demo_breeds = [("small", "Pekinese"), ("medium", "Beagle"), ("large", "German_shepherd")]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    max_spine = max(data["templates"][t]["measure"]["lower"]
                     + data["templates"][t]["measure"]["upper"] for t in TIER_NAMES)
    max_leg = max(data["templates"][t]["measure"]["fem"]
                   + data["templates"][t]["measure"]["tib"] for t in TIER_NAMES)

    for col, tier in enumerate(TIER_NAMES):
        measure = data["templates"][tier]["measure"]
        appear, _ = appearance_for(tier)
        draw_dog(axes[0, col], measure, appear, ref, colors[tier],
                 f"{tier.upper()} (tier default)")

        b_tier, breed = demo_breeds[col]
        b_appear, b_source = appearance_for(b_tier, breed)
        draw_dog(axes[1, col], data["templates"][b_tier]["measure"], b_appear, ref,
                 colors[b_tier], f"{breed} ({b_tier} tier)")

    for ax in axes.flat:
        ax.set_xlim(-max_spine * 0.45, max_spine * 1.75)
        ax.set_ylim(-max_leg * 1.2, max_leg * 0.9)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Top: tier-default appearance   |   Bottom: breed-table override\n"
                 "same skeletal proportions per column; only appearance params differ",
                 fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "appearance_preview.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

OUT_DIR = "outputs"
TIER_NAMES = ["small", "medium", "large"]
TIER_COLORS = {"small": "#D9784F", "medium": "#C99A2E", "large": "#3E7C5A"}


def taper_quad(x0, y0, L, w0, w1, angle_deg):
    a = np.radians(angle_deg)
    dx, dy = np.cos(a), np.sin(a)
    nx, ny = -dy, dx
    p0 = np.array([x0, y0]) + np.array([nx, ny]) * w0
    p1 = np.array([x0, y0]) - np.array([nx, ny]) * w0
    p2 = np.array([x0 + dx * L, y0 + dy * L]) - np.array([nx, ny]) * w1
    p3 = np.array([x0 + dx * L, y0 + dy * L]) + np.array([nx, ny]) * w1
    end = np.array([x0 + dx * L, y0 + dy * L])
    return np.array([p0, p1, p2, p3]), end


def draw_template(ax, measure, ref, color, label):
    spine_len = measure["lower"] + measure["upper"]
    neck = np.array([spine_len, 0.0])
    tail = np.array([0.0, 0.0])

    body_w0, body_w1 = ref * 0.16, ref * 0.13
    body_poly, _ = taper_quad(tail[0], tail[1], spine_len, body_w0, body_w1, 0)
    ax.add_patch(Polygon(body_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.2, alpha=0.85, zorder=5))

    head_len = max(measure["head"], ref * 0.08)
    head_poly, head_end = taper_quad(neck[0], neck[1], head_len, ref * 0.09, ref * 0.05, 0)
    ax.add_patch(Polygon(head_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=0.9, zorder=6))

    fx = spine_len * 0.85
    hum_top = np.array([fx, -body_w0 * 0.6])
    hum_poly, elbow = taper_quad(hum_top[0], hum_top[1], measure["hum"],
                                   ref * 0.045, ref * 0.030, -90)
    ax.add_patch(Polygon(hum_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=0.75, zorder=4))
    rad_poly, paw_f = taper_quad(elbow[0], elbow[1], measure["rad"],
                                   ref * 0.030, ref * 0.018, -90)
    ax.add_patch(Polygon(rad_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=0.75, zorder=4))

    hx = spine_len * 0.15
    fem_top = np.array([hx, -body_w0 * 0.6])
    fem_poly, knee = taper_quad(fem_top[0], fem_top[1], measure["fem"],
                                  ref * 0.050, ref * 0.032, -90)
    ax.add_patch(Polygon(fem_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=0.75, zorder=4))
    tib_poly, paw_h = taper_quad(knee[0], knee[1], measure["tib"],
                                   ref * 0.032, ref * 0.019, -90)
    ax.add_patch(Polygon(tib_poly, closed=True, facecolor=color, edgecolor="black",
                          linewidth=1.0, alpha=0.75, zorder=4))

    ax.set_title(f"{label}\nspine={spine_len:.0f}px  legs(f/h)="
                 f"{measure['hum']+measure['rad']:.0f}/{measure['fem']+measure['tib']:.0f}px",
                 fontsize=11)


def main():
    with open(os.path.join(OUT_DIR, "dog_templates.json"), "r") as f:
        data = json.load(f)
    ref = data["target_spine"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    max_leg = max(data["templates"][t]["measure"]["fem"] + data["templates"][t]["measure"]["tib"]
                   for t in TIER_NAMES)
    max_spine = max(data["templates"][t]["measure"]["lower"] + data["templates"][t]["measure"]["upper"]
                     for t in TIER_NAMES)

    for ax, tier in zip(axes, TIER_NAMES):
        measure = data["templates"][tier]["measure"]
        draw_template(ax, measure, ref, TIER_COLORS[tier], tier.upper())
        ax.set_xlim(-max_spine * 0.15, max_spine * 1.15)
        ax.set_ylim(-max_leg * 1.15, max_leg * 0.35)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.suptitle("Canonical body-proportion templates (static side view, standing pose)\n"
                 "same vertical/horizontal scale across all three panels",
                 fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "template_silhouettes.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()

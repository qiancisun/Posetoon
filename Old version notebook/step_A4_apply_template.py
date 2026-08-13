import os
import json

OUT_DIR = "outputs"
TEMPLATES_PATH = os.path.join(OUT_DIR, "dog_templates.json")


def tier_for(value, p33, p67):
    if value < p33:
        return "small"
    if value > p67:
        return "large"
    return "medium"


def derive_video_ratio(measure):
    front_total = measure["hum"] + measure["rad"]
    hind_total = measure["fem"] + measure["tib"]
    return 0.5 * (front_total + hind_total) / 200.0


def apply_template(measure, alpha=0.5, templates_path=TEMPLATES_PATH):
    with open(templates_path, "r") as f:
        data = json.load(f)

    p33 = data["runtime_selection_thresholds"]["p33"]
    p67 = data["runtime_selection_thresholds"]["p67"]

    video_ratio = derive_video_ratio(measure)
    tier = tier_for(video_ratio, p33, p67)
    template_measure = data["templates"][tier]["measure"]

    blended = {
        k: (1 - alpha) * template_measure[k] + alpha * measure[k]
        for k in template_measure
    }
    return tier, video_ratio, blended


if __name__ == "__main__":

    demo_measure = {
        "lower": 93.5,
        "upper": 93.5,
        "head": 93.2,
        "hum": 47.0,
        "rad": 34.6,
        "fem": 48.5,
        "tib": 44.4,
    }

    print("Demo measure dict (REPLACE with your own printed values):")
    for k, v in demo_measure.items():
        print(f"  {k:6}: {v:.1f}px")

    for alpha in [0.0, 0.5, 1.0]:
        tier, ratio, blended = apply_template(demo_measure, alpha=alpha)
        print(f"\nalpha={alpha}  video_ratio={ratio:.3f}  tier={tier}")
        for k, v in blended.items():
            print(f"  {k:6}: {v:.1f}px")

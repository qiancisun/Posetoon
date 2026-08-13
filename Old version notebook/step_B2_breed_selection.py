import os
import sys
import json
import numpy as np

OUT_DIR = "outputs"
TEMPLATES = os.path.join(OUT_DIR, "breed_templates.json")

DOG_IDX_START, DOG_IDX_END = 151, 269
SAMPLE_EVERY = 15
GEOM_TEMPERATURE = 0.015
CLASSIFIER_WEIGHT = 0.75

FIELDS = ["head", "hum", "rad", "fem", "tib"]

from breeds import fold_map

FAMILY_VOTES = fold_map()


def classifier_scores(video_path, template_names):
    import torch
    import cv2
    from PIL import Image
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    categories = weights.meta["categories"]
    preprocess = weights.transforms()

    cap = cv2.VideoCapture(video_path)
    accum = np.zeros(DOG_IDX_END - DOG_IDX_START)
    n, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % SAMPLE_EVERY == 0:
            batch = preprocess(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
            if torch.cuda.is_available():
                batch = batch.cuda()
            with torch.no_grad():
                logits = model(batch)[0]
            accum += torch.softmax(logits[DOG_IDX_START:DOG_IDX_END],
                                    dim=0).cpu().numpy()
            n += 1
        idx += 1
    cap.release()
    if n == 0:
        return None, None, 0
    accum /= n

    raw_top = sorted(
        [(categories[DOG_IDX_START + i], float(accum[i])) for i in range(len(accum))],
        key=lambda kv: -kv[1])[:5]

    votes = {t: 0.0 for t in template_names}
    unassigned = 0.0
    for i, p in enumerate(accum):
        name = categories[DOG_IDX_START + i].lower().replace(" ", "_").replace("-", "_")
        hit = None
        for tmpl, keys in FAMILY_VOTES.items():
            if tmpl in votes and any(k in name for k in keys):
                hit = tmpl
                break
        if hit:
            votes[hit] += float(p)
        else:
            unassigned += float(p)

    total = sum(votes.values())
    if total > 0:
        votes = {k: v / total for k, v in votes.items()}
    return votes, raw_top, unassigned


def geometry_scores(measure, templates, temperature=GEOM_TEMPERATURE):
    dists = {}
    for name, entry in templates.items():
        tm = entry["measure"]
        dists[name] = float(np.mean([abs(measure[f] - tm[f]) for f in FIELDS]) / 200.0)
    d = np.array([dists[n] for n in dists])
    s = np.exp(-d / temperature)
    s = s / s.sum()
    return {n: float(v) for n, v in zip(dists, s)}, dists


def gait_bias_note(measure, templates, geo_pick, cls_pick):
    def ratio(m):
        return 0.5 * ((m["hum"] + m["rad"]) + (m["fem"] + m["tib"])) / 200.0

    r_dog = ratio(measure)
    r_geo = ratio(templates[geo_pick]["measure"])
    r_cls = ratio(templates[cls_pick]["measure"])
    print(f"\n   leg-to-spine: measured {r_dog:.3f} | "
          f"{geo_pick} {r_geo:.3f} | {cls_pick} {r_cls:.3f}")
    if r_dog < r_cls - 0.05:
        print(f"   The measurement sits {r_cls - r_dog:.3f} below the "
              f"classifier's template.")
        print("   Templates come from standing dogs; a running subject's limbs")
        print("   are foreshortened, so geometry drifts toward shorter-legged")
        print("   breeds. Treat the geometry pick with caution on gait footage.")


def select(measure, video_path=None, classifier_weight=CLASSIFIER_WEIGHT):
    with open(TEMPLATES) as f:
        data = json.load(f)
    templates = data["templates"]
    names = list(templates)

    geo, dists = geometry_scores(measure, templates)
    print("Geometry (proportion distance -> score):")
    for n in sorted(dists, key=lambda k: dists[k]):
        print(f"  {n:17} dist={dists[n]:.4f}  score={geo[n]:.3f}")

    cls = None
    if video_path and os.path.exists(video_path):
        print("\nClassifier (ImageNet, folded onto the template set)...")
        cls, raw_top, unassigned = classifier_scores(video_path, names)
        if cls:
            print("  raw ImageNet top-5:")
            for nm, p in raw_top:
                print(f"    {nm:28} {p:.3f}")
            print(f"  probability not matching any template family: {unassigned:.3f}")
            print("  folded onto templates:")
            for n in sorted(cls, key=lambda k: -cls[k]):
                if cls[n] > 0.005:
                    print(f"    {n:17} {cls[n]:.3f}")

    if cls:
        combined = {n: (1 - classifier_weight) * geo[n] + classifier_weight * cls[n]
                    for n in names}
    else:
        combined = geo
        classifier_weight = 0.0

    pick = max(combined, key=combined.get)
    geo_pick = max(geo, key=geo.get)
    cls_pick = max(cls, key=cls.get) if cls else None
    agree = (cls_pick == geo_pick) if cls_pick else None

    print(f"\nGeometry picks   : {geo_pick}")
    if cls_pick:
        print(f"Classifier picks : {cls_pick}")
        print(f"Agreement        : {'YES' if agree else 'NO'}")
        if not agree:
            print("  Reported, not overridden: the weighted combination "
                  "decides. See the note printed below.")
    print(f"\n=> template: {pick}   (combined score {combined[pick]:.3f})")

    c_sorted = sorted(combined.values(), reverse=True)
    margin = (c_sorted[0] - c_sorted[1]) if len(c_sorted) > 1 else 1.0
    print(f"   margin over runner-up: {margin:.3f}"
          + ("   <- narrow; the pick is not well separated" if margin < 0.08 else ""))

    g_sorted = sorted(geo.values(), reverse=True)
    g_margin = g_sorted[0] - g_sorted[1] if len(g_sorted) > 1 else 1.0
    print(f"   geometry-only margin : {g_margin:.3f}"
          + ("   (body plans overlap; appearance is deciding)"
             if g_margin < 0.10 else ""))

    if cls and not agree:
        gait_bias_note(measure, templates, geo_pick, cls_pick)

    result = {
        "template": pick, "combined": combined,
        "geometry_scores": geo, "geometry_distances": dists, "geometry_pick": geo_pick,
        "classifier_scores": cls, "classifier_pick": cls_pick,
        "agreement": agree, "classifier_weight": classifier_weight,
        "margin": float(margin), "geometry_margin": float(g_margin),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "breed_selection.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT_DIR}/breed_selection.json")
    return result


if __name__ == "__main__":
    demo = {"lower": 93.5, "upper": 93.5, "head": 93.2,
            "hum": 47.0, "rad": 34.6, "fem": 48.5, "tib": 44.4}
    video = sys.argv[1] if len(sys.argv) > 1 else "dogvideo.mp4"
    select(demo, video)

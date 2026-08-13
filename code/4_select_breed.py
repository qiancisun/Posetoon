import os
import sys
import json
import numpy as np

OUT_DIR = "outputs"
TEMPLATES = os.path.join(OUT_DIR, "breed_templates.json")

DOG_IDX_START, DOG_IDX_END = 151, 269
SAMPLE_EVERY = 6
GEOM_TEMPERATURE = 0.015
CLASSIFIER_WEIGHT = 0.75

FIELDS = ["head", "hum", "rad", "fem", "tib"]

from breeds import fold_map

FAMILY_VOTES = fold_map()


def dog_boxes(stab_path=None, pad=0.35):
    stab_path = stab_path or STAB_PATH
    if not os.path.exists(stab_path):
        return {}
    try:
        with open(stab_path) as f:
            frames = json.load(f)
    except (ValueError, OSError):
        return {}
    boxes = {}
    for i, fr in enumerate(frames):
        kp = np.array(fr.get("keypoints", []), dtype=float)
        if kp.ndim != 2 or len(kp) < 5:
            continue
        good = kp[(kp[:, 0] > 0) & (kp[:, 1] > 0)]
        if len(good) < 5:
            continue
        x0, y0 = good.min(axis=0)
        x1, y1 = good.max(axis=0)
        w, h = x1 - x0, y1 - y0
        if w < 8 or h < 8:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(w, h) * (1.0 + 2 * pad)
        boxes[i] = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    return boxes


def classifier_scores(video_path, template_names, stab_path=None):
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

    boxes = dog_boxes(stab_path)
    if not boxes:
        print("  !! no keypoint track available -- classifying WHOLE FRAMES, "
              "which is unreliable when the dog is small or off-centre")

    cap = cv2.VideoCapture(video_path)
    accum = np.zeros(DOG_IDX_END - DOG_IDX_START)
    n, idx, n_cropped = 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % SAMPLE_EVERY == 0:
            img = frame[:, :, ::-1]
            box = boxes.get(idx)
            if box is not None:
                H, W = frame.shape[:2]
                x0 = int(max(0, min(W - 2, box[0])))
                y0 = int(max(0, min(H - 2, box[1])))
                x1 = int(max(x0 + 2, min(W, box[2])))
                y1 = int(max(y0 + 2, min(H, box[3])))
                if (x1 - x0) >= 32 and (y1 - y0) >= 32:
                    img = img[y0:y1, x0:x1]
                    n_cropped += 1
            pil = Image.fromarray(np.ascontiguousarray(img))
            batch = torch.stack([preprocess(pil),
                                 preprocess(pil.transpose(
                                     Image.FLIP_LEFT_RIGHT))])
            if torch.cuda.is_available():
                batch = batch.cuda()
            with torch.no_grad():
                logits = model(batch)
            probs = torch.softmax(logits[:, DOG_IDX_START:DOG_IDX_END],
                                  dim=1).mean(dim=0)
            accum += probs.cpu().numpy()
            n += 1
        idx += 1
    cap.release()
    if n:
        print(f"  classified {n} frame(s), {n_cropped} of them cropped to the "
              f"tracked dog"
              + ("" if n_cropped == n else
                 f"  ({n - n_cropped} fell back to the whole frame)"))
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


def select(measure, video_path=None, classifier_weight=CLASSIFIER_WEIGHT,
           stab_path=None):
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
        cls, raw_top, unassigned = classifier_scores(video_path, names,
                                                     stab_path=stab_path)
        if cls:
            print("  raw ImageNet top-5:")
            for nm, p in raw_top:
                print(f"    {nm:28} {p:.3f}")
            print(f"  probability not matching any template family: {unassigned:.3f}")
            print("  folded onto templates:")
            for n in sorted(cls, key=lambda k: -cls[k]):
                if cls[n] > 0.005:
                    print(f"    {n:17} {cls[n]:.3f}")

    geo_pick = max(geo, key=geo.get)
    cls_pick = max(cls, key=cls.get) if cls else None
    agree = (cls_pick == geo_pick) if cls_pick else None

    GAIT_BIAS_MIN_GAP = 0.05
    GAIT_BIAS_FULL_GAP = 0.20
    GAIT_BIAS_MAX_WEIGHT = 0.92
    GAIT_BIAS_MIN_CLS_CONF = 0.30

    def _ratio(m):
        return 0.5 * ((m["hum"] + m["rad"]) + (m["fem"] + m["tib"])) / 200.0

    if cls and cls_pick and not agree and cls[cls_pick] >= GAIT_BIAS_MIN_CLS_CONF:
        gap = _ratio(templates[cls_pick]["measure"]) - _ratio(templates[geo_pick]["measure"])
        if gap > GAIT_BIAS_MIN_GAP:
            t = min(1.0, (gap - GAIT_BIAS_MIN_GAP)
                    / max(GAIT_BIAS_FULL_GAP - GAIT_BIAS_MIN_GAP, 1e-6))
            new_w = classifier_weight + t * (GAIT_BIAS_MAX_WEIGHT - classifier_weight)
            if new_w > classifier_weight:
                print(f"\n  gait-bias discount: geometry picks {geo_pick}, which is "
                      f"{gap:.3f} shorter-legged than the classifier's {cls_pick}.")
                print(f"  That is the direction the standing-vs-running bias predicts, "
                      f"so it is not\n  independent evidence. Classifier weight "
                      f"{classifier_weight:.2f} -> {new_w:.2f} for this clip.")
                classifier_weight = float(new_w)

    if cls:
        combined = {n: (1 - classifier_weight) * geo[n] + classifier_weight * cls[n]
                    for n in names}
    else:
        combined = geo
        classifier_weight = 0.0

    pick = max(combined, key=combined.get)

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


KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
KP_INDEX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
STAB_PATH = "stabilised_keypoints.json"


def measure_from_keypoints(stab_path=STAB_PATH, target_spine=200.0):
    with open(stab_path) as f:
        stab = json.load(f)

    def kp(fr, name):
        return np.array(fr["keypoints"][KP_INDEX[name]], dtype=float)

    acc = {k: [] for k in ["lower", "upper", "head", "hum", "rad", "fem", "tib"]}
    for fr in stab:
        neck, tail = kp(fr, "Neck"), kp(fr, "root_of_tail")
        sl = float(np.linalg.norm(neck - tail))
        if sl < 1e-6:
            continue
        sc = target_spine / sl
        hips = (kp(fr, "L_Hip") + kp(fr, "R_Hip")) / 2
        pelvis = tail * 0.55 + hips * 0.45
        waist = (pelvis + neck) / 2
        acc["lower"].append(np.linalg.norm(waist - pelvis) * sc)
        acc["upper"].append(np.linalg.norm(neck - waist) * sc)
        acc["head"].append(np.linalg.norm(kp(fr, "Nose") - neck) * sc)
        acc["hum"].append(np.linalg.norm(kp(fr, "L_Elbow") - kp(fr, "L_Shoulder")) * sc)
        acc["rad"].append(np.linalg.norm(kp(fr, "L_F_Paw") - kp(fr, "L_Elbow")) * sc)
        acc["fem"].append(np.linalg.norm(kp(fr, "L_Knee") - kp(fr, "L_Hip")) * sc)
        acc["tib"].append(np.linalg.norm(kp(fr, "L_B_Paw") - kp(fr, "L_Knee")) * sc)

    m = {k: float(np.median(acc[k])) for k in ["lower", "upper", "head"]}
    m.update({k: float(np.percentile(acc[k], 75))
              for k in ["hum", "rad", "fem", "tib"]})
    print(f"Measured from {stab_path} ({len(stab)} frames), spine = 200:")
    print("  " + "  ".join(f"{k}={m[k]:.1f}" for k in
                           ["lower", "upper", "head", "hum", "rad", "fem", "tib"]))
    return m


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "dogvideo.mp4"
    stab = sys.argv[2] if len(sys.argv) > 2 else STAB_PATH
    if not os.path.exists(stab):
        print(f"{stab} not found -- run 2_repair_keypoints.py on this video first.")
        sys.exit(1)
    select(measure_from_keypoints(stab), video, stab_path=stab)

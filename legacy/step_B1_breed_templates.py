import os
import glob
import json
import numpy as np

from mmpose.apis import MMPoseInferencer

STANFORD_DIR = "images/Images"
OUT_DIR = "outputs"
TARGET_SPINE = 200.0
N_PER_BREED = 60
MIN_ACCEPTED = 12

SCORE_THRESHOLD = 0.35
MIN_SPINE_PX = 40.0
MIN_HORIZONTALITY = 0.70

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
K = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
FIELDS = ["lower", "upper", "head", "hum", "rad", "fem", "tib"]

from breeds import folder_map, appearance_of, validate as _validate_breeds

_validate_breeds()
BREEDS = folder_map()


def bone(xy, sc, a, b):
    ia, ib = K[a], K[b]
    if sc[ia] < SCORE_THRESHOLD or sc[ib] < SCORE_THRESHOLD:
        return None
    return float(np.linalg.norm(np.array(xy[ib]) - np.array(xy[ia])))


def measure_one(xy, sc):
    xy = np.asarray(xy, dtype=float)
    neck, tail = xy[K["Neck"]], xy[K["root_of_tail"]]
    if sc[K["Neck"]] < SCORE_THRESHOLD or sc[K["root_of_tail"]] < SCORE_THRESHOLD:
        return None, "spine"
    spine = float(np.linalg.norm(tail - neck))
    if spine < MIN_SPINE_PX:
        return None, "small"
    if abs(tail[0] - neck[0]) / spine < MIN_HORIZONTALITY:
        return None, "view"

    scale = TARGET_SPINE / spine

    def limb(up_a, up_b, lo_a, lo_b):
        u, l = bone(xy, sc, up_a, up_b), bone(xy, sc, lo_a, lo_b)
        return None if (u is None or l is None) else (u, l)

    front = [v for v in (limb("L_Shoulder", "L_Elbow", "L_Elbow", "L_F_Paw"),
                         limb("R_Shoulder", "R_Elbow", "R_Elbow", "R_F_Paw"))
             if v is not None]
    hind = [v for v in (limb("L_Hip", "L_Knee", "L_Knee", "L_B_Paw"),
                        limb("R_Hip", "R_Knee", "R_Knee", "R_B_Paw"))
            if v is not None]
    if not front:
        return None, "front"
    if not hind:
        return None, "hind"

    head = bone(xy, sc, "Nose", "Neck")
    if head is None:
        return None, "head"

    hips_ok = sc[K["L_Hip"]] >= SCORE_THRESHOLD and sc[K["R_Hip"]] >= SCORE_THRESHOLD
    hips = (xy[K["L_Hip"]] + xy[K["R_Hip"]]) / 2 if hips_ok else tail
    pelvis = tail * 0.55 + hips * 0.45
    waist = (pelvis + neck) / 2

    return {
        "lower": float(np.linalg.norm(waist - pelvis)) * scale,
        "upper": float(np.linalg.norm(neck - waist)) * scale,
        "head": head * scale,
        "hum": float(np.mean([f[0] for f in front])) * scale,
        "rad": float(np.mean([f[1] for f in front])) * scale,
        "fem": float(np.mean([h[0] for h in hind])) * scale,
        "tib": float(np.mean([h[1] for h in hind])) * scale,
    }, "ok"


def find_folder(substr):
    hits = glob.glob(os.path.join(STANFORD_DIR, f"*{substr}*"))
    return hits[0] if hits else None


def main():
    print("Loading MMPose inferencer...")
    inferencer = MMPoseInferencer("animal")
    print("Model loaded.\n")

    templates, skipped = {}, []
    for substr, name in BREEDS.items():
        folder = find_folder(substr)
        if folder is None:
            print(f"[skip] no folder matching '{substr}'")
            skipped.append((name, "folder not found"))
            continue

        paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))[:N_PER_BREED * 3]
        samples, rej = [], {}
        for p in paths:
            if len(samples) >= N_PER_BREED:
                break
            try:
                res = [r for r in inferencer(p, show=False, return_vis=False)]
                preds = res[0]["predictions"][0]
                if not preds:
                    continue
                xy, sc = preds[0]["keypoints"], preds[0]["keypoint_scores"]
            except Exception:
                continue
            m, why = measure_one(xy, sc)
            if m is None:
                rej[why] = rej.get(why, 0) + 1
                continue
            samples.append(m)

        if len(samples) < MIN_ACCEPTED:
            print(f"[{name:16}] only {len(samples)} usable (rejects {rej}) -- skipped")
            skipped.append((name, f"only {len(samples)} usable"))
            continue

        measure = {f: float(np.median([s[f] for s in samples])) for f in FIELDS}
        iqr = {f: float(np.percentile([s[f] for s in samples], 75)
                        - np.percentile([s[f] for s in samples], 25))
               for f in FIELDS}

        entry = {"measure": measure, "measure_raw": dict(measure),
                 "iqr": iqr, "n_samples": len(samples),
                 "folder": os.path.basename(folder)}
        ap = appearance_of(name)
        if ap is not None:
            src = f"breeds.py:{name}"
            entry["appearance"] = ap
            entry["appearance_source"] = src
            if not src.startswith("breed_table:"):
                print(f"    !! no breed entry for '{name}' -- fell back to "
                      f"{src}. This template will not carry that breed's "
                      f"muzzle, ears or tail. Add it to BREED_APPEARANCE.")

            authored = ap.get("muzzle_len")
            if authored:
                measured_ratio = measure["head"] / TARGET_SPINE
                entry["head_measured_ratio"] = round(measured_ratio, 3)
                entry["head_authored_ratio"] = authored
                entry["head_source"] = "authored (see note in step_B1)"
                measure["head"] = authored * TARGET_SPINE
        templates[name] = entry

        legs = measure["hum"] + measure["rad"], measure["fem"] + measure["tib"]
        hm = entry.get("head_measured_ratio")
        head_note = (f"head={measure['head']:6.1f} (measured {hm*TARGET_SPINE:.0f})"
                     if hm is not None else f"head={measure['head']:6.1f}")
        print(f"[{name:16}] n={len(samples):3}  front_leg={legs[0]:6.1f}  "
              f"hind_leg={legs[1]:6.1f}  {head_note}  "
              f"ratio={0.5*(legs[0]+legs[1])/TARGET_SPINE:.3f}")

    if not templates:
        print("\nNo templates built -- check STANFORD_DIR.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "target_spine": TARGET_SPINE,
        "source": "Stanford Dogs, MMPose inference (no GT keypoints exist "
                   "for this dataset; the deployed system is inference-only "
                   "too, so this is the matching domain)",
        "spine_split": "asymmetric, pelvis = tail*0.55 + hips*0.45, "
                        "matching measure_skeleton()",
        "view_filter": {"min_spine_px": MIN_SPINE_PX,
                         "min_horizontality": MIN_HORIZONTALITY,
                         "score_threshold": SCORE_THRESHOLD},
        "skipped": skipped,
        "templates": templates,
    }
    path = os.path.join(OUT_DIR, "breed_templates.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}  ({len(templates)} breeds)")

    names = list(templates)
    DISC = ["head", "hum", "rad", "fem", "tib"]
    print("\nPairwise proportion distance (mean |diff| over "
          "head/hum/rad/fem/tib, / 200):")
    print("      " + "".join(f"{n[:8]:>10}" for n in names))
    for a in names:
        row = f"{a[:8]:>6}"
        for b in names:
            d = np.mean([abs(templates[a]["measure"][f]
                             - templates[b]["measure"][f]) for f in DISC]) / TARGET_SPINE
            row += f"{d:10.3f}"
        print(row)
    print("\nOff-diagonal values near zero mean two templates are "
          "interchangeable and one should be dropped or replaced.")


if __name__ == "__main__":
    main()

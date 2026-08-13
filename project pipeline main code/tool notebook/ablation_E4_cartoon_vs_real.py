#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np

OUT = "outputs/evaluation"
OVERLAY_DIR = os.path.join(OUT, "E4_overlays")
DEMO_ROOT = os.path.join("outputs", "demos")
GRADES = "grades.txt"

CHAR_PANEL_W = 600
CAPTION_H = 45
SEAM_TRIM = 3
MIN_MASK_PX = 400
CANVAS = 256


def read_grades(path=GRADES):
    out = {}
    if not os.path.exists(path):
        return out
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = [x for x in line.split("  ") if x != ""]
        if len(p) >= 2 and p[0].strip() in ("A", "B", "C"):
            out[p[1].strip()] = p[0].strip()
    return out


def clips_for(grades_wanted):
    found = []
    for g in grades_wanted:
        for sub in (g, {"A": "A-best results"}.get(g, g), g + "_compat"):
            d = os.path.join(DEMO_ROOT, sub)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".mp4") and "_character" not in f:
                    found.append((g, os.path.join(d, f)))
            break
    return found


def panels(frame):
    h, w = frame.shape[:2]
    if w <= CHAR_PANEL_W + 100:
        return None, None
    src_w = (w - CHAR_PANEL_W) // 2
    src = frame[CAPTION_H:, :src_w]
    char = frame[CAPTION_H:, w - CHAR_PANEL_W + SEAM_TRIM:]
    return src, char


def character_mask(panel, tol=12):
    import cv2
    g = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.int16)
    border = np.concatenate([g[:, :6].ravel(), g[:, -6:].ravel(),
                             g[:6, :].ravel(), g[-6:, :].ravel()])
    bg = int(np.bincount(np.clip(border, 0, 255), minlength=256).argmax())

    ink = (np.abs(g - bg) > tol).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    h, w = ink.shape
    outside = np.zeros((h + 2, w + 2), np.uint8)
    free = (1 - ink).astype(np.uint8)
    cv2.floodFill(free, outside, (0, 0), 2)
    cv2.floodFill(free, outside, (w - 1, 0), 2)
    cv2.floodFill(free, outside, (0, h - 1), 2)
    cv2.floodFill(free, outside, (w - 1, h - 1), 2)
    filled = (free != 2)

    if filled.sum() < MIN_MASK_PX:
        filled = ink > 0
    return largest_blob(filled)


def largest_blob(mask):
    import cv2
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == idx


class DogSegmenter:
    def __init__(self):
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights, deeplabv3_resnet50)
        self.torch = torch
        self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        cats = self.weights.meta["categories"]
        self.dog = cats.index("dog")
        self.model = deeplabv3_resnet50(weights=self.weights).eval()
        self.tf = self.weights.transforms()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.dev)

    def __call__(self, panel_bgr):
        import cv2
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(img).unsqueeze(0).to(self.dev)
        with self.torch.inference_mode():
            out = self.model(x)["out"][0]
        pred = out.argmax(0).byte().cpu().numpy()
        m = (pred == self.dog)
        if m.sum() < MIN_MASK_PX:
            return None
        m = cv2.resize(m.astype(np.uint8), (panel_bgr.shape[1],
                                            panel_bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        return largest_blob(m)


def to_canvas(mask, n=CANVAS):
    import cv2
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = crop.shape
    scale = float(n) / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((n, n), np.uint8)
    y0, x0 = (n - nh) // 2, (n - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out > 0


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else float("nan")


def overlay(real, char, path):
    import cv2
    h, w = real.shape
    img = np.full((h, w, 3), 255, np.uint8)
    img[real & ~char] = (60, 60, 220)
    img[char & ~real] = (60, 190, 60)
    img[real & char] = (60, 200, 230)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", nargs="+", default=["A"])
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python is required.")
        return 1

    clips = clips_for(args.grades)
    if not clips:
        print("No delivered clips found under %s/" % DEMO_ROOT)
        return 1
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    print("Loading the segmentation model (downloads once, ~170 MB)...")
    try:
        seg = DogSegmenter()
    except Exception as exc:
        print("Could not load DeepLabV3: %s" % exc)
        print("torch and torchvision are needed; both are already used by the "
              "breed classifier.")
        return 1
    print("  running on %s, 'dog' is class %d\n" % (seg.dev, seg.dog))

    import cv2
    rows, all_scores = [], []
    for grade, path in clips:
        stem = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n <= 0:
            cap.release()
            print("  %-34s unreadable" % stem[:34])
            continue
        idx = np.linspace(n * 0.1, n * 0.9, args.frames).astype(int)

        scores, seg_fail, best = [], 0, (None, -1.0)
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            src, char = panels(frame)
            if src is None:
                continue
            cm = character_mask(char)
            if cm is None or cm.sum() < MIN_MASK_PX:
                continue
            rm = seg(src)
            if rm is None:
                seg_fail += 1
                continue
            ca, ra = to_canvas(cm), to_canvas(rm)
            if ca is None or ra is None:
                continue
            v = iou(ra, ca)
            scores.append(v)
            if v > best[1]:
                best = ((ra, ca), v)
        cap.release()

        if not scores:
            print("  %-34s no scorable frame (segmentation failed on %d)"
                  % (stem[:34], seg_fail))
            rows.append([stem, grade, "-", "-", "-", len(idx), seg_fail])
            continue

        if best[0] is not None:
            overlay(best[0][0], best[0][1],
                    os.path.join(OVERLAY_DIR, "%s.png" % stem))
        med = float(np.median(scores))
        all_scores.append(med)
        rows.append([stem, grade, "%.4f" % med, "%.4f" % np.min(scores),
                     "%.4f" % np.max(scores), len(scores), seg_fail])
        print("  %-34s IoU %.3f  (%d frames, %d segmentation failures)"
              % (stem[:34], med, len(scores), seg_fail))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "E4_1_cartoon_vs_real.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "grade", "IoU median", "IoU min", "IoU max",
                    "frames scored", "segmentation failures"])
        w.writerows(rows)

    scored = len(all_scores)
    total_fail = sum(r[6] for r in rows if isinstance(r[6], int))
    L = ["# E4 -- the character's outline against the real dog\n",
         "The only measurement in this project with no human judgement in "
         "it.\n",
         "## Method\n",
         "Both masks are taken from the delivered three-panel videos: the "
         "character from the flat-background panel by thresholding, the "
         "animal from the source panel with DeepLabV3 (VOC labels, class "
         "'dog'). Each is cropped to its bounding box and resampled to a "
         "common canvas without stretching, so the score compares shape "
         "and body proportion and ignores position and absolute size.\n",
         "Nothing in the pipeline is imported or re-run. This is why the "
         "measurement was possible after all: the rendered character was "
         "already on disk.\n",
         "## Result\n"]
    if scored:
        a = np.array(all_scores)
        L.append("- %d clip(s) scored; median IoU **%.3f** "
                 "(range %.3f-%.3f)." % (scored, np.median(a), a.min(), a.max()))
    else:
        L.append("- No clip could be scored.")
    L.append("- Frames where segmentation returned no dog: **%d**. Those are "
             "excluded rather than scored, and the exclusion rate is itself a "
             "result -- it says how often an off-the-shelf segmenter fails on "
             "this kind of footage." % total_fail)
    L.append("\n## Read the overlays before quoting the number\n")
    L.append("`E4_overlays/` has one image per clip: red is the real dog, "
             "green is the character, yellow is agreement. An earlier attempt "
             "at this measurement produced IoU 0.02-0.03 across the board and "
             "those figures were a bug, not a result -- the renderer mirrors "
             "the canvas when the dog faces left and the alignment did not. "
             "Numbers from misaligned masks look like numbers.")
    L.append("\n## Limits\n")
    L.append("- Bounding-box normalisation removes size, so a character of "
             "the right shape at the wrong scale scores well here. Size is "
             "covered separately by the proportion measurements in E1.")
    L.append("- The segmentation is a general-purpose model applied to dark "
             "coats, backlighting and cluttered backgrounds. Where it "
             "half-succeeds the mask is wrong in ways the IoU cannot flag; "
             "this is what the overlays are for.")
    L.append("- Frames within a clip come from one continuous shot and are "
             "not independent samples.")
    with open(os.path.join(OUT, "SUMMARY_E4.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    print("\n%d clip(s) scored, %d segmentation failures" % (scored, total_fail))
    if scored:
        print("median IoU across clips: %.3f" % float(np.median(all_scores)))
    print("\nwrote E4_1_cartoon_vs_real.csv + SUMMARY_E4.md to %s/" % OUT)
    print("LOOK AT %s/ before quoting any of it." % OVERLAY_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
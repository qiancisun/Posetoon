import os
import sys
import json
import numpy as np

OUT_DIR = "outputs"
STAB_PATH = "stabilised_keypoints.json"
DEFAULT_VIDEO = "dogvideo.mp4"

SAMPLE_FRAMES = [0, 40, 70, 110, 140, 180, 220, 260]
ROI_MARGIN = 0.35
ROI_BOTTOM_MARGIN = 0.06
KP_TOLERANCE_PX = 4
SCORE_THRESHOLD = 0.30
MORPH_KERNEL = 5

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]


def keypoint_roi(kp, scores, w, h):
    ok = np.asarray(scores) >= SCORE_THRESHOLD
    if ok.sum() < 4:
        return None, None
    pts = np.asarray(kp)[ok]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * ROI_MARGIN, bh * ROI_MARGIN
    my_bottom = bh * ROI_BOTTOM_MARGIN
    roi = (max(0, int(x0 - mx)), max(0, int(y0 - my)),
           min(w, int(x1 + mx)), min(h, int(y1 + my_bottom)))
    return roi, (x0, y0, x1, y1)


def segment_frame(frame, kp, scores):
    import cv2
    h, w = frame.shape[:2]
    roi, kbox = keypoint_roi(kp, scores, w, h)
    if roi is None:
        return None

    rx0, ry0, rx1, ry1 = roi
    sub = frame[ry0:ry1, rx0:rx1]
    if sub.size == 0:
        return None

    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    _thr, binimg = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, k)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE, k)

    n_lab, labels, stats, _cent = cv2.connectedComponentsWithStats(binimg, 8)

    ok = np.asarray(scores) >= SCORE_THRESHOLD
    pts = np.asarray(kp)[ok]
    local = pts - np.array([rx0, ry0])

    best_lab, best_hits = 0, -1
    for lab in range(1, n_lab):
        hits = 0
        for (lx, ly) in local:
            xi, yi = int(round(lx)), int(round(ly))
            if 0 <= yi < labels.shape[0] and 0 <= xi < labels.shape[1] \
               and labels[yi, xi] == lab:
                hits += 1
        if hits > best_hits:
            best_hits, best_lab = hits, lab
    if best_lab == 0:
        return None

    mask_local = (labels == best_lab).astype(np.uint8) * 255
    mask = np.zeros((h, w), np.uint8)
    mask[ry0:ry1, rx0:rx1] = mask_local

    kdil = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (KP_TOLERANCE_PX * 2 + 1, KP_TOLERANCE_PX * 2 + 1))
    mask_tol = cv2.dilate(mask, kdil)
    inside = 0
    for (px, py) in pts:
        xi, yi = int(round(px)), int(round(py))
        if 0 <= yi < h and 0 <= xi < w and mask_tol[yi, xi] > 0:
            inside += 1
    kp_inside = inside / max(1, len(pts))

    x0, y0, x1, y1 = kbox
    kp_h = max(1.0, y1 - y0)
    ys = np.where(mask.any(axis=1))[0]
    mask_bottom = ys.max() if ys.size else y1
    extent_below = max(0.0, (mask_bottom - y1)) / kp_h

    area = int((mask > 0).sum())
    fill_ratio = area / max(1.0, (x1 - x0) * (y1 - y0))

    return {
        "mask": mask, "roi": roi, "kbox": kbox,
        "kp_inside": float(kp_inside),
        "extent_below": float(extent_below),
        "fill_ratio": float(fill_ratio),
        "area_px": area,
        "n_components": int(n_lab - 1),
        "otsu_threshold": float(_thr),
    }


def main():
    import cv2

    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        return
    with open(STAB_PATH) as f:
        frames = json.load(f)

    cap = cv2.VideoCapture(video)
    results, panels = [], []

    for fi in SAMPLE_FRAMES:
        if fi >= len(frames):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        kp = frames[fi]["keypoints"]
        scores = frames[fi].get("scores", [1.0] * len(kp))

        r = segment_frame(frame, kp, scores)
        if r is None:
            print(f"frame {fi:3}: segmentation failed")
            continue

        rx0, ry0, rx1, ry1 = r["roi"]
        crop = frame[ry0:ry1, rx0:rx1]
        mcrop = r["mask"][ry0:ry1, rx0:rx1]

        overlay = crop.copy()
        overlay[mcrop > 0] = (0.45 * overlay[mcrop > 0]
                              + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        for (px, py), sc in zip(kp, scores):
            if sc >= SCORE_THRESHOLD:
                cv2.circle(overlay, (int(px - rx0), int(py - ry0)), 3, (0, 255, 0), -1)

        mvis = cv2.cvtColor(mcrop, cv2.COLOR_GRAY2BGR)
        panels.append((fi, np.hstack([crop, mvis, overlay])))

        results.append({k: v for k, v in r.items() if k != "mask"})
        results[-1]["frame"] = fi
        print(f"frame {fi:3}: kp_inside={r['kp_inside']:.2f}  "
              f"extent_below={r['extent_below']:+.2f}  "
              f"fill_ratio={r['fill_ratio']:.2f}  "
              f"components={r['n_components']:3}  otsu={r['otsu_threshold']:.0f}")
    cap.release()

    if not results:
        print("No frames segmented -- check paths and keypoint alignment.")
        return

    ki = np.array([r["kp_inside"] for r in results])
    eb = np.array([r["extent_below"] for r in results])
    fr = np.array([r["fill_ratio"] for r in results])

    print(f"\nAcross {len(results)} frames:")
    print(f"  kp_inside     median {np.median(ki):.2f}  min {ki.min():.2f}")
    print(f"  extent_below  median {np.median(eb):+.2f}  max {eb.max():+.2f}")
    print(f"  fill_ratio    median {np.median(fr):.2f}  range "
          f"{fr.min():.2f}-{fr.max():.2f}")

    print("\nVerdict:")
    verdict = []
    if np.median(ki) >= 0.90:
        verdict.append("keypoint containment is good -- the mask is on the dog")
    elif np.median(ki) >= 0.75:
        verdict.append("keypoint containment is marginal -- parts of the body "
                       "are being missed")
    else:
        verdict.append("keypoint containment is POOR -- the mask is not "
                       "reliably the dog")

    if np.median(eb) <= 0.10:
        verdict.append("no significant shadow absorption")
    elif np.median(eb) <= 0.30:
        verdict.append("some shadow may be attached below the paws")
    else:
        verdict.append("SHADOW IS LIKELY MERGED INTO THE MASK")

    verdict.append(f"fill ratio {np.median(fr):.2f} (descriptive only; "
                   f"a correct silhouette exceeds the joint bounding box)")

    for v in verdict:
        print(f"  - {v}")

    usable = np.median(ki) >= 0.90 and np.median(eb) <= 0.15
    print(f"\n=> {'Masks look usable: proceed to IoU (E4 Option A).' if usable else 'Masks NOT reliable enough. Use E4 Option B (joint-trajectory divergence) instead -- it needs no segmentation.'}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "e4_segmentation_stats.json"), "w") as f:
        json.dump({"video": video, "frames": results, "usable": bool(usable)},
                  f, indent=2)

    width = max(p.shape[1] for _fi, p in panels)
    stacked = []
    for fi, p in panels:
        pad = np.full((p.shape[0], width - p.shape[1], 3), 255, np.uint8)
        row = np.hstack([p, pad]) if pad.shape[1] else p
        cv2.putText(row, f"f{fi}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        stacked.append(row)
    sheet = np.vstack(stacked)
    cv2.putText(sheet, "original | mask | overlay(+keypoints)", (6, sheet.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out_png = os.path.join(OUT_DIR, "e4_segmentation_check.png")
    cv2.imwrite(out_png, sheet)
    print(f"Saved {out_png}")
    print(f"Saved {OUT_DIR}/e4_segmentation_stats.json")


if __name__ == "__main__":
    main()

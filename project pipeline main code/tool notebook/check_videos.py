import os
import sys
import csv
import shutil

import numpy as np

SAMPLE_FRAMES = 12
MIN_HORIZONTALITY = 0.70
MAX_OFF_PROFILE = 0.20
MIN_DETECTION_RATE = 0.75
MIN_FRAMES = 40
SCORE_THRESHOLD = 0.3
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "root_of_tail",
    "L_Shoulder", "L_Elbow", "L_F_Paw",
    "R_Shoulder", "R_Elbow", "R_F_Paw",
    "L_Hip", "L_Knee", "L_B_Paw",
    "R_Hip", "R_Knee", "R_B_Paw",
]
K = {n: i for i, n in enumerate(KEYPOINT_NAMES)}


def screen(video, inferencer):
    import cv2

    cap = cv2.VideoCapture(video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n_frames < 2:
        cap.release()
        return {"error": "unreadable"}

    idxs = np.linspace(0, n_frames - 1, min(SAMPLE_FRAMES, n_frames)).astype(int)
    horiz, n_dogs, detected = [], [], 0

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            res = [r for r in inferencer(rgb, show=False, return_vis=False)]
            preds = res[0]["predictions"][0]
        except Exception:
            continue
        n_dogs.append(len(preds))
        if not preds:
            continue
        detected += 1
        xy = np.asarray(preds[0]["keypoints"], dtype=float)
        sc = np.asarray(preds[0]["keypoint_scores"], dtype=float)
        if sc[K["Neck"]] < SCORE_THRESHOLD or sc[K["root_of_tail"]] < SCORE_THRESHOLD:
            continue
        neck, tail = xy[K["Neck"]], xy[K["root_of_tail"]]
        spine = float(np.linalg.norm(tail - neck))
        if spine > 1e-6:
            horiz.append(abs(tail[0] - neck[0]) / spine)
    cap.release()

    if not horiz:
        return {"error": "no usable pose in any sampled frame",
                "frames": n_frames, "fps": fps}

    return {
        "frames": n_frames,
        "fps": round(fps, 1),
        "seconds": round(n_frames / fps, 1),
        "horizontality": round(float(np.median(horiz)), 2),
        "horiz_min": round(float(np.min(horiz)), 2),
        "off_profile": round(float(np.mean(np.array(horiz) < MIN_HORIZONTALITY)), 2),
        "dogs": int(np.median(n_dogs)) if n_dogs else 0,
        "dogs_max": int(np.max(n_dogs)) if n_dogs else 0,
        "detection_rate": round(detected / len(idxs), 2),
    }


def verdict(r):
    if "error" in r:
        return "REJECT", r["error"]
    why = []
    if r["horizontality"] < MIN_HORIZONTALITY:
        why.append(f"not side-on (horizontality {r['horizontality']}, "
                   f"need {MIN_HORIZONTALITY})")
    elif r["off_profile"] > MAX_OFF_PROFILE:
        why.append(f"turns away from side-on in "
                   f"{int(r['off_profile']*100)}% of frames (median looks fine "
                   f"at {r['horizontality']}, but the rig collapses on those)")
    if r["dogs"] > 1:
        why.append(f"{r['dogs']} dogs in frame")
    if r["detection_rate"] < MIN_DETECTION_RATE:
        why.append(f"dog found in only {int(r['detection_rate']*100)}% of frames")
    if r["frames"] < MIN_FRAMES:
        why.append(f"{r['frames']} frames, need {MIN_FRAMES}")
    if why:
        return "REJECT", "; ".join(why)
    if r["dogs_max"] > 1:
        return "CHECK", (f"usually one dog but up to {r['dogs_max']} in some "
                         f"frames -- watch for the track jumping across")
    if r["horizontality"] < 0.85:
        return "CHECK", (f"borderline profile ({r['horizontality']}) -- "
                         f"usable, but a more side-on clip would retarget better")
    return "PASS", ""


def find_duplicates(folder, videos):
    import re
    import cv2

    groups = {}
    for v in videos:
        m = re.match(r"(\d{5,})", v)
        if m:
            groups.setdefault(m.group(1), []).append(v)

    dupes = {}
    for vid, names in groups.items():
        if len(names) < 2:
            continue
        sigs = {}
        for n in names:
            cap = cv2.VideoCapture(os.path.join(folder, n))
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                continue
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
            sigs[n] = (small.astype(float) - small.mean()) / (small.std() + 1e-6)
        keys = list(sigs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if float(np.corrcoef(sigs[keys[i]].ravel(),
                                     sigs[keys[j]].ravel())[0, 1]) > 0.95:
                    pair = sorted(
                        (keys[i], keys[j]),
                        key=lambda n: os.path.getsize(os.path.join(folder, n)))
                    dupes[pair[0]] = pair[1]
    return dupes


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_videos.py <folder-of-videos>")
        return
    folder = sys.argv[1]
    copy_to = None
    if "--copy" in sys.argv:
        copy_to = sys.argv[sys.argv.index("--copy") + 1]
        os.makedirs(copy_to, exist_ok=True)

    videos = sorted(f for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in VIDEO_EXT)
    if not videos:
        print(f"No videos in {folder}")
        return

    dupes = find_duplicates(folder, videos)
    if dupes:
        print(f"{len(dupes)} duplicate file(s) -- same clip at two "
              f"resolutions, keeping the larger:")
        for smaller, larger in dupes.items():
            print(f"   skip {smaller}\n     (same clip as {larger})")
        videos = [v for v in videos if v not in dupes]
        print()

    print("Loading MMPose...")
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer("animal")
    print(f"Screening {len(videos)} clip(s), {SAMPLE_FRAMES} frames each.\n")

    rows = []
    for name in videos:
        r = screen(os.path.join(folder, name), inferencer)
        v, why = verdict(r)
        rows.append({"video": name, "verdict": v, "reason": why, **r})
        line = f"{v:6} {name[:44]:44}"
        if "error" not in r:
            line += (f" horiz={r['horizontality']:.2f} off={r['off_profile']:.2f} "
                     f"dogs={r['dogs']} det={r['detection_rate']:.2f} "
                     f"{r['seconds']:.0f}s")
        print(line)
        if why:
            print(f"       {why}")
        if v == "PASS" and copy_to:
            shutil.copy(os.path.join(folder, name), os.path.join(copy_to, name))

    with open("video_screening.csv", "w", newline="") as f:
        keys = sorted({k for row in rows for k in row})
        w = csv.DictWriter(f, fieldnames=["video", "verdict", "reason"] +
                           [k for k in keys if k not in ("video", "verdict", "reason")])
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    n_check = sum(r["verdict"] == "CHECK" for r in rows)
    print(f"\n{n_pass} pass, {n_check} to check by eye, "
          f"{len(rows) - n_pass - n_check} rejected.  Table: video_screening.csv")
    if copy_to:
        print(f"Passing clips copied to {copy_to}/")
    print("\nStill worth doing by hand: scrub each survivor for cuts. A clip "
          "made of two shots passes every test here.")


if __name__ == "__main__":
    main()

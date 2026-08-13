import os
import sys
import csv
import json
import time
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_DIR = os.path.join("outputs", "batch")
TEMPLATES = os.path.join("outputs", "breed_templates.json")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

STAGES = [
    ("1_extract_keypoints.py", True,  "full_video_keypoints.json", "extract"),
    ("2_repair_keypoints.py",  True,  "stabilised_keypoints.json", "repair"),
    ("3_coat_colour.py",       False, "outputs/coat_palette.json", "coat"),
    ("4_select_breed.py",      False, "outputs/breed_selection.json", "breed"),
]


def last_line(r):
    for stream in (r.stderr, r.stdout):
        lines = [ln.strip() for ln in (stream or "").strip().split("\n") if ln.strip()]
        if lines:
            return lines[-1][:200]
    return f"exit {r.returncode}, no output"


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def run_one(video, workdir, log):
    os.makedirs(os.path.join(workdir, "outputs"), exist_ok=True)

    src = os.path.join(HERE, TEMPLATES)
    if not os.path.exists(src):
        return False, "setup", ("outputs/breed_templates.json not found -- run "
                                "0_build_breed_templates.py once first")
    shutil.copy(src, os.path.join(workdir, TEMPLATES))

    env = dict(os.environ, POSETOON_VIDEO=os.path.abspath(video))
    for script, is_marimo, output, label in STAGES:
        cmd = [sys.executable, os.path.join(HERE, script)]
        if not is_marimo:
            cmd.append(os.path.abspath(video))
        r = subprocess.run(cmd, cwd=workdir, env=env,
                           capture_output=True, text=True)
        log.write(f"\n===== {label} =====\n{r.stdout}\n{r.stderr}\n")
        if r.returncode != 0:
            return False, label, last_line(r)
        if not os.path.exists(os.path.join(workdir, output)):
            return False, label, f"finished without writing {output}"

    env["POSETOON_RUN_UPSTREAM"] = "0"
    r = subprocess.run([sys.executable, os.path.join(HERE, "posetoon_pipeline.py")],
                       cwd=workdir, env=env, capture_output=True, text=True)
    log.write(f"\n===== rig =====\n{r.stdout}\n{r.stderr}\n")
    if r.returncode != 0:
        return False, "rig", last_line(r)
    return True, "done", ""


def collect(video, workdir, ok, stage, err, seconds):
    q = read_json(os.path.join(workdir, "outputs", "video_quality.json"))
    sel = read_json(os.path.join(workdir, "outputs", "breed_selection.json"))
    desc = read_json(os.path.join(workdir, "outputs", "character_description.json"))
    coat = read_json(os.path.join(workdir, "outputs", "coat_palette.json"))
    return {
        "video": os.path.basename(video),
        "status": "ok" if ok else f"FAILED at {stage}",
        "error": err,
        "seconds": round(seconds),
        "frames": q.get("n_frames"),
        "fps": q.get("fps"),
        "dup_rate": round(q["dup_rate"], 3) if "dup_rate" in q else None,
        "dup_cadence": q.get("dup_cadence"),
        "sharpness": round(q["sharpness_median"], 1) if "sharpness_median" in q else None,
        "brightness": round(q["brightness_median"], 1) if "brightness_median" in q else None,
        "outliers_rejected": q.get("kinematic_outliers_rejected"),
        "quality_verdict": q.get("verdict"),
        "template": desc.get("template") or sel.get("template"),
        "template_requested": sel.get("template"),
        "geometry_pick": sel.get("geometry_pick"),
        "classifier_pick": sel.get("classifier_pick"),
        "sources_agree": sel.get("agreement"),
        "margin": round(sel["margin"], 3) if sel.get("margin") is not None else None,
        "alpha": desc.get("alpha"),
        "root_lift_px": desc.get("root_lift_px"),
        "spine_tail": desc.get("spine_tail_source"),
        "coat_hex": coat.get("base_hex"),
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "videos"
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}\nUsage: python run_all.py videos/")
        return

    videos = sorted(f for f in os.listdir(folder)
                    if os.path.splitext(f)[1].lower() in VIDEO_EXT)
    if not videos:
        print(f"No videos in {folder}")
        return

    os.makedirs(BATCH_DIR, exist_ok=True)
    print(f"{len(videos)} video(s) in {folder}\n")

    rows = []
    for i, name in enumerate(videos, 1):
        video = os.path.join(folder, name)
        stem = os.path.splitext(name)[0]
        workdir = os.path.join(BATCH_DIR, stem)
        os.makedirs(workdir, exist_ok=True)
        print(f"[{i}/{len(videos)}] {name}", flush=True)

        t0 = time.time()
        with open(os.path.join(workdir, "run.log"), "w") as log:
            ok, stage, err = run_one(video, workdir, log)
        row = collect(video, workdir, ok, stage, err, time.time() - t0)
        rows.append(row)

        if ok:
            print(f"        {row['template']}  dup={row['dup_rate']}  "
                  f"outliers={row['outliers_rejected']}  "
                  f"{row['seconds']}s", flush=True)
        else:
            print(f"        FAILED at {stage}: {err[:100]}", flush=True)
            print(f"        see {workdir}/run.log", flush=True)

    out = os.path.join("outputs", "batch_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{n_ok}/{len(rows)} completed.  Table: {out}")
    print(f"Renders: {BATCH_DIR}/<name>/outputs/<name>_rig.mp4")
    if n_ok < len(rows):
        print("Failures are rows in the table too -- what the pipeline "
              "rejects belongs in the limitations section.")

    agreed = [r for r in rows if r["sources_agree"] is True]
    if rows:
        print(f"\nclassifier and geometry agreed on "
              f"{len(agreed)}/{n_ok} of the successful runs")
        seen = {}
        for r in rows:
            if r["template"]:
                seen[r["template"]] = seen.get(r["template"], 0) + 1
        print("templates used: " + ", ".join(f"{k} x{v}" for k, v in
                                             sorted(seen.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()

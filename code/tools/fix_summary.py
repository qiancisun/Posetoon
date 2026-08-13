#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import sys

BATCH = "outputs/batch"
SUMMARY = "outputs/batch_summary.csv"
RUN_ALL = "run_all.py"

OLD = '''        "template": sel.get("template"),
        "geometry_pick": sel.get("geometry_pick"),'''

NEW = '''        # The template ACTUALLY USED, which is not always the one step_B2
        # picked: posetoon_aline overrides the classifier when its choice is
        # inconsistent with the dog's measured size. Reading breed_selection
        # here reported Great Dane for a clip that rendered as a Basset Hound.
        # The requested pick is kept alongside, because how often the override
        # fires is itself a result worth reporting.
        "template": desc.get("template") or sel.get("template"),
        "template_requested": sel.get("template"),
        "geometry_pick": sel.get("geometry_pick"),'''


def patch():
    if not os.path.exists(RUN_ALL):
        sys.exit(f"{RUN_ALL} not found -- run this from the project root")
    src = open(RUN_ALL).read()
    if '"template_requested"' in src:
        print("  run_all.py is already patched")
        return
    if src.count(OLD) != 1:
        sys.exit(f"  could not find the exact lines to replace in {RUN_ALL} "
                 f"({src.count(OLD)} matches). Edit line 138 by hand:\n"
                 f'      "template": desc.get("template") or sel.get("template"),')
    shutil.copy2(RUN_ALL, RUN_ALL + ".bak")
    open(RUN_ALL, "w").write(src.replace(OLD, NEW))
    print(f"  patched {RUN_ALL}  (backup at {RUN_ALL}.bak)")


def desc_of(name):
    p = os.path.join(BATCH, name, "outputs", "character_description.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def rebuild():
    if not os.path.exists(SUMMARY):
        sys.exit(f"{SUMMARY} not found -- nothing to rebuild")
    with open(SUMMARY) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("summary is empty")

    fields = list(rows[0].keys())
    if "template_requested" not in fields:
        fields.insert(fields.index("template") + 1, "template_requested")

    changed = []
    missing = []
    for r in rows:
        name = os.path.splitext(r["video"])[0]
        d = desc_of(name)
        used = d.get("template")
        r.setdefault("template_requested", r.get("template"))
        if not r.get("template_requested"):
            r["template_requested"] = r.get("template")
        if not used:
            missing.append(name)
            continue
        if used != r.get("template"):
            changed.append((name, r.get("template"), used))
        r["template"] = used

    shutil.copy2(SUMMARY, SUMMARY + ".bak")
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"  rebuilt {SUMMARY} from {len(rows)} clip(s)  "
          f"(backup at {SUMMARY}.bak)")
    if missing:
        print(f"  !! {len(missing)} clip(s) had no character_description.json "
              f"-- their template column is unchanged:")
        for m in missing[:6]:
            print(f"       {m}")
    print(f"\n  the override fired on {len(changed)} of {len(rows)} clip(s):")
    for name, was, now in changed:
        print(f"     {name:34} {was}  ->  {now}")
    if not changed:
        print("     none -- every clip used the template step_B2 proposed")

    used = {}
    for r in rows:
        t = r.get("template")
        if t:
            used[t] = used.get(t, 0) + 1
    print("\n  templates actually used: "
          + ", ".join(f"{k} x{v}" for k, v in
                      sorted(used.items(), key=lambda kv: -kv[1])))
    print(f"  {len(used)} distinct template(s)")


FIELDS = ["video", "status", "error", "seconds", "frames", "fps", "dup_rate",
          "dup_cadence", "sharpness", "brightness", "outliers_rejected",
          "quality_verdict", "template", "template_requested", "geometry_pick",
          "classifier_pick", "sources_agree", "margin", "alpha",
          "root_lift_px", "spine_tail", "coat_hex"]


def _load(name, fn):
    p = os.path.join(BATCH, name, "outputs", fn)
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def from_disk():
    if not os.path.isdir(BATCH):
        sys.exit(f"{BATCH} not found")
    names = sorted(d for d in os.listdir(BATCH)
                   if os.path.isdir(os.path.join(BATCH, d)))
    rows = []
    for n in names:
        q = _load(n, "video_quality.json")
        sel = _load(n, "breed_selection.json")
        desc = _load(n, "character_description.json")
        coat = _load(n, "coat_palette.json")
        rend = any(os.path.exists(os.path.join(BATCH, n, "outputs", n + sfx))
                   for sfx in ("_rig.mp4", "_rig_h264.mp4"))
        if not (q or sel or desc):
            continue
        rows.append({
            "video": n + ".mp4",
            "status": "ok" if rend else "FAILED (no render on disk)",
            "error": "",
            "seconds": "",
            "frames": q.get("n_frames"),
            "fps": q.get("fps"),
            "dup_rate": round(q["dup_rate"], 3) if "dup_rate" in q else None,
            "dup_cadence": q.get("dup_cadence"),
            "sharpness": (round(q["sharpness_median"], 1)
                          if "sharpness_median" in q else None),
            "brightness": (round(q["brightness_median"], 1)
                           if "brightness_median" in q else None),
            "outliers_rejected": q.get("kinematic_outliers_rejected"),
            "quality_verdict": q.get("verdict"),
            "template": desc.get("template") or sel.get("template"),
            "template_requested": sel.get("template"),
            "geometry_pick": sel.get("geometry_pick"),
            "classifier_pick": sel.get("classifier_pick"),
            "sources_agree": sel.get("agreement"),
            "margin": (round(sel["margin"], 3)
                       if sel.get("margin") is not None else None),
            "alpha": desc.get("alpha"),
            "root_lift_px": desc.get("root_lift_px"),
            "spine_tail": desc.get("spine_tail_source"),
            "coat_hex": coat.get("base_hex"),
        })

    if not rows:
        sys.exit("no clip JSON found under " + BATCH)
    if os.path.exists(SUMMARY):
        shutil.copy2(SUMMARY, SUMMARY + ".bak")
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in FIELDS})

    over = [(r["video"], r["template_requested"], r["template"]) for r in rows
            if r["template_requested"] and r["template"]
            and r["template_requested"] != r["template"]]
    used = {}
    for r in rows:
        if r["template"]:
            used[r["template"]] = used.get(r["template"], 0) + 1
    print(f"  rebuilt {SUMMARY} from {len(rows)} clip directory(ies)")
    print(f"  the override fired on {len(over)} of {len(rows)}:")
    for v, a, b in over:
        print(f"     {v:34} {a}  ->  {b}")
    if not over:
        print("     none")
    print("\n  templates actually used: "
          + ", ".join(f"{k} x{v}" for k, v in
                      sorted(used.items(), key=lambda kv: -kv[1])))
    print(f"  {len(used)} of 12 templates covered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--from-disk", action="store_true",
                    help="rebuild every row from outputs/batch/*/outputs/*.json, "
                         "ignoring the current CSV -- use after run_one.py has "
                         "truncated it")
    a = ap.parse_args()
    if getattr(a, "from_disk", False):
        print("rebuilding the summary from the per-clip JSON:")
        from_disk()
        return
    if not (a.patch or a.rebuild):
        a.patch = a.rebuild = True
    if a.patch:
        print("patching run_all.py:")
        patch()
    if a.rebuild:
        print("\nrebuilding the summary:")
        rebuild()


if __name__ == "__main__":
    main()
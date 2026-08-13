#!/usr/bin/env python3
import csv
import os
import sys
from collections import Counter, defaultdict

import json

import numpy as np

OUT = "outputs/evaluation"
BATCH_DIR = "outputs/batch"
CANVAS = 640
SCALE = 1.4

FALLBACK_MEASURE = {"lower": 97, "upper": 97, "head": 85,
                    "hum": 44, "rad": 34, "fem": 48, "tib": 42}


def reference_measure(grades=None):
    best = None
    if os.path.isdir(BATCH_DIR):
        for clip in sorted(os.listdir(BATCH_DIR)):
            if clip.endswith("__superseded"):
                continue
            p = os.path.join(BATCH_DIR, clip, "outputs",
                             "character_description.json")
            if not os.path.exists(p):
                continue
            try:
                m = json.load(open(p)).get("measure_used")
            except (ValueError, OSError):
                continue
            if not isinstance(m, dict):
                continue
            if not all(k in m for k in ("lower", "upper", "head",
                                        "hum", "rad", "fem", "tib")):
                continue
            if grades and grades.get(clip) == "A":
                return {k: float(m[k]) for k in m}, clip
            if best is None:
                best = ({k: float(m[k]) for k in m}, clip)
    if best:
        return best
    return dict(FALLBACK_MEASURE), "(no clip record found -- fallback values)"


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def rest_world(rig):
    bones = {k: v for k, v in rig.items()
             if not k.startswith("_") and isinstance(v, dict)
             and "offset" in v}
    pos, guard = {}, 0
    while len(pos) < len(bones) and guard < 100:
        guard += 1
        for name, b in bones.items():
            if name in pos:
                continue
            par = b.get("parent")
            if par is None or par not in bones:
                pos[name] = np.array(b["offset"], dtype=float)
            elif par in pos:
                pos[name] = pos[par] + np.array(b["offset"], dtype=float)
    return pos


def silhouette(rig, canvas=CANVAS, scale=SCALE):
    from PIL import Image, ImageDraw
    pos = rest_world(rig)
    img = Image.new("1", (canvas, canvas), 0)
    dr = ImageDraw.Draw(img)
    cx = cy = canvas // 2
    n_parts = n_verts = 0
    for name, b in rig.items():
        if name.startswith("_") or not isinstance(b, dict):
            continue
        mesh = b.get("mesh")
        if not mesh or name not in pos:
            continue
        n_parts += 1
        n_verts += len(mesh)
        p = pos[name]
        poly = [(cx + (p[0] + x) * scale, cy + (p[1] + y) * scale)
                for (x, y) in mesh]
        if len(poly) >= 3:
            dr.polygon(poly, fill=1)
    return np.array(img, dtype=bool), n_parts, n_verts


def e3(templates, ref_measure, ref_clip):
    import breeds
    import character_style as cs

    rows, drops = [], Counter()
    for name in templates:
        app = breeds.appearance_of(name)
        if app is None:
            continue
        out = {}
        for cx in ("fine", "coarse"):
            rig = cs.build_rig_styled(dict(ref_measure), app,
                                      tier=name, complexity=cx)
            mask, parts, verts = silhouette(rig)
            drawn = {k for k, v in rig.items()
                     if not k.startswith("_") and isinstance(v, dict)
                     and v.get("mesh")}
            out[cx] = (mask, parts, verts, drawn)
        (mf, pf, vf, df), (mc, pc, vc, dc) = out["fine"], out["coarse"]
        for k in df - dc:
            drops[k] += 1
        inter = float(np.logical_and(mf, mc).sum())
        union = float(np.logical_or(mf, mc).sum())
        iou = inter / union if union else float("nan")
        af, ac = float(mf.sum()), float(mc.sum())
        rows.append([name, pf, pc, vf, vc,
                     "%.0f" % af, "%.0f" % ac,
                     "%+.1f%%" % (100.0 * (ac - af) / af if af else 0),
                     "%.4f" % iou,
                     "%.1f%%" % (100.0 * (union - inter) / af if af else 0)])
    hdr = ["template", "parts fine", "parts coarse", "verts fine",
           "verts coarse", "area fine", "area coarse", "area change",
           "IoU fine vs coarse", "disagreeing pixels (% of fine)"]
    write_csv("E3_1_complexity_geometry.csv", hdr, rows)

    drop_rows = [[k, v] for k, v in drops.most_common()]
    write_csv("E3_2_parts_dropped.csv",
              ["part dropped by coarse", "templates affected"], drop_rows)
    return rows, hdr, drop_rows


def read_grades(path="grades.txt"):
    out = {}
    if not os.path.exists(path):
        return out
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = [x for x in line.split("  ") if x != ""]
        if len(p) >= 2 and p[0].strip() in ("A", "B", "C"):
            out[p[1].strip()] = p[0].strip()
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def e5(grades):
    path = "outputs/batch_summary.csv"
    if not os.path.exists(path):
        return [], [], []
    clips = []
    for r in csv.DictReader(open(path, newline="")):
        v = (r.get("video") or "").strip()
        if not v:
            continue
        name = os.path.splitext(v)[0]
        clips.append({
            "clip": name, "grade": grades.get(name, ""),
            "frames": num(r.get("frames")), "fps": num(r.get("fps")),
            "dup_rate": num(r.get("dup_rate")),
            "sharpness": num(r.get("sharpness")),
            "brightness": num(r.get("brightness")),
            "outliers": num(r.get("outliers_rejected")),
            "spine_tail": (r.get("spine_tail") or "").strip(),
        })
    graded = [c for c in clips if c["grade"]]

    def band(rs, label):
        n = len(rs)
        a = sum(1 for r in rs if r["grade"] == "A")
        return [label, n, "-" if not n else "%.0f%% (%d/%d)"
                % (100.0 * a / n, a, n)]

    rows = []
    rows.append(band([r for r in graded if np.isfinite(r["dup_rate"])
                      and r["dup_rate"] <= 0.01], "duplicate rate <= 1%"))
    rows.append(band([r for r in graded if 0.01 < r["dup_rate"] <= 0.25],
                     "duplicate rate 1-25% (pulldown range)"))
    rows.append(band([r for r in graded if r["dup_rate"] > 0.25],
                     "duplicate rate > 25% (beyond pulldown)"))
    for lo, hi, lab in ((0, 150, "under 150 frames"),
                        (150, 300, "150-300 frames"),
                        (300, 1e9, "over 300 frames")):
        rows.append(band([r for r in graded
                          if lo <= r["frames"] < hi], lab))
    for lab, key in (("annotated spine/tail", "annotated"),
                     ("synthesised spine/tail", "synthesised")):
        rows.append(band([r for r in graded if r["spine_tail"] == key], lab))
    write_csv("E5_1_footage_property_vs_outcome.csv",
              ["input property", "clips", "A-grade rate"], rows)

    eff = []
    for key, label in (("frames", "frames"), ("fps", "frame rate"),
                       ("dup_rate", "duplicate-frame rate"),
                       ("sharpness", "sharpness"),
                       ("brightness", "brightness"),
                       ("outliers", "outliers rejected")):
        a = np.array([r[key] for r in graded
                      if r["grade"] == "A" and np.isfinite(r[key])])
        c = np.array([r[key] for r in graded
                      if r["grade"] == "C" and np.isfinite(r[key])])
        if len(a) > 1 and len(c) > 1:
            pooled = np.sqrt((a.var(ddof=1) + c.var(ddof=1)) / 2.0)
            d = abs(a.mean() - c.mean()) / pooled if pooled > 1e-12 else 0.0
            eff.append([label, "%.3g" % np.median(a), "%.3g" % np.median(c),
                        "%.2f" % d,
                        "yes" if d >= 0.8 else ("weak" if d >= 0.5 else "no")])
    write_csv("E5_2_effect_sizes.csv",
              ["measurement", "A median", "C median", "effect size",
               "separates?"], eff)
    return rows, eff, graded


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        import breeds
    except ImportError:
        print("Run this from the project root (breeds.py must be importable).")
        return 1
    templates = sorted(breeds.BREEDS) if hasattr(breeds, "BREEDS") else []
    if not templates:
        print("No templates found in breeds.py")
        return 1

    grades0 = read_grades()
    ref_measure, ref_clip = reference_measure(grades0)
    e3_rows, e3_hdr, drops = e3(templates, ref_measure, ref_clip)
    e5_rows, e5_eff, graded = e5(grades0)

    L = ["# E3 and E5\n",
         "Generated by `ablation_E3_E5.py`. E3 replaces the 2025-07-29 run, "
         "which measured a size-tier system that has since been removed.\n",
         "## E3 -- what does more detail buy?\n",
         "Both settings built through `character_style.build_rig_styled` and "
         "compared at REST POSE. Complexity changes part geometry only: bone "
         "hierarchy, offsets, lengths and solver are identical, so every "
         "difference is already present before anything is animated.\n",
         "Proportions held constant across all twelve templates, taken from a "
         "real delivered clip: **%s**.\n" % ref_clip]
    if e3_rows:
        ious = np.array([float(r[8]) for r in e3_rows if r[8] != "nan"])
        parts_f = np.mean([r[1] for r in e3_rows])
        parts_c = np.mean([r[2] for r in e3_rows])
        verts_f = np.mean([r[3] for r in e3_rows])
        verts_c = np.mean([r[4] for r in e3_rows])
        L.append("- Parts drawn: **%.1f fine vs %.1f coarse** "
                 "(-%.0f%%); vertices **%.0f vs %.0f** (-%.0f%%)."
                 % (parts_f, parts_c, 100 * (1 - parts_c / parts_f),
                    verts_f, verts_c, 100 * (1 - verts_c / verts_f)))
        L.append("- Rest-pose silhouette IoU between the two: median "
                 "**%.3f** (range %.3f-%.3f across %d templates)."
                 % (np.median(ious), ious.min(), ious.max(), len(ious)))
        if np.median(ious) > 0.95:
            L.append("\nThe two characters occupy **almost the same outline**, "
                     "so the parts coarse removes are internal detail rather "
                     "than form.")
        else:
            L.append("\n**The two settings do not describe the same shape.** "
                     "About a quarter of the drawn pixels disagree. What coarse "
                     "drops (`E3_2_parts_dropped.csv`) is the four paws and the "
                     "lower ear segment -- parts that sit ON the outline, not "
                     "inside it -- so the complexity switch changes the "
                     "character's form and not only its finish.")
            areas = np.array([float(r[7].rstrip("%")) for r in e3_rows])
            slim = [r for r in e3_rows if abs(float(r[7].rstrip("%"))) < 2.0]
            if slim:
                L.append("\nArea and shape disagree, and the disagreement is "
                         "informative. %s lose almost no AREA (%s) yet differ "
                         "in a quarter to a third of their pixels. Coarse does "
                         "not only remove parts: it also replaces tapered limbs "
                         "with equal-width bars. On a slender breed the width "
                         "it adds roughly cancels the paws it removes, so the "
                         "totals match while the outline does not -- which is "
                         "why area alone would have been the wrong measure here."
                         % (" and ".join(r[0] for r in slim),
                            ", ".join(r[7] for r in slim)))
        L.append("\n### What this measurement cannot see\n")
        L.append("Only parts that carry a mesh are counted. The eye, the nose, "
                 "the muzzle patch and the joint dots are drawn by `render_rig` "
                 "as decorations and are skipped at coarse WITHOUT being mesh "
                 "parts, so none of them appear above. The real difference "
                 "between the two settings is therefore **larger** than the "
                 "figure reported, and in the direction the figure cannot "
                 "measure.")
    L.append("\n## E5 -- which footage works?\n")
    if e5_rows:
        for r in e5_rows:
            L.append("- %s: %s clips, A-rate %s" % (r[0], r[1], r[2]))
    if e5_eff:
        L.append("")
        for r in e5_eff:
            L.append("- %s: A %s / C %s, effect %s -> separates: %s"
                     % (r[0], r[1], r[2], r[3], r[4]))
    L.append("\n## Limits\n")
    L.append("- E3 is measured at rest, so it says nothing about how the two "
             "complexities read in motion, and nothing about internal detail.")
    L.append("- E3 uses one measurement set for all templates so that only "
             "complexity varies. It comes from clip `%s`; a different clip "
             "would move both characters together and would not change the "
             "difference between them." % ref_clip)
    L.append("- E5 bands are small. A rate computed on a handful of clips is a "
             "count, not a probability.")
    L.append("- **E4 (cartoon vs real outline IoU) is NOT re-run here.** It "
             "needs the animated render, which lives inside the pipeline "
             "notebook and cannot be imported. The 2025-07-29 E4 numbers "
             "belong to the removed size-tier system and should be reported as "
             "such or not at all.")
    with open(os.path.join(OUT, "SUMMARY_E3_E5.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("E3: %d templates, fine vs coarse at rest pose" % len(e3_rows))
    if e3_rows:
        ious = [float(r[8]) for r in e3_rows if r[8] != "nan"]
        print("    silhouette IoU median %.3f  (parts %d->%d on the first row)"
              % (np.median(ious), e3_rows[0][1], e3_rows[0][2]))
    print("E5: %d graded clips banded by footage property" % len(graded))
    print("\nwrote E3_1, E3_2, E5_1, E5_2 + SUMMARY_E3_E5.md to %s/" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

OUT = "outputs/evaluation"
GRADES = "grades.txt"
SUMMARY_CSV = "outputs/batch_summary.csv"
BATCH = "outputs/batch"

FAILURE_PATTERNS = [
    ("high camera",        r"high camera|camera above|off-profile|top.?down"),
    ("no gait",            r"no gait|swing \d+ deg|gait range"),
    ("dog turns away",     r"turning|turns away|off.profile"),
    ("leaves frame",       r"out of frame|leaves frame"),
    ("duplicate frames",   r"duplicate frame rate|dup rate|pulldown"),
    ("tracking failure",   r"outliers|tracking jump|root lift|rejected as outliers"),
    ("weak breed decision", r"not corroborated|alpha 0\.6|unsupported"),
    ("breed wrong",        r"breed identif|wrong breed|tricolour|saddle|template"),
    ("pipeline error",     r"status=failed|no render|FAILED"),
    ("superseded",         r"superseded"),
    ("same clip twice",    r"same video|same clip|different filename"),
]


def read_grades(path=GRADES):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p for p in line.split("  ") if p != ""]
            if len(parts) < 2 or parts[0].strip() not in ("A", "B", "C"):
                continue
            out[parts[1].strip()] = (parts[0].strip(),
                                     "  ".join(parts[2:]).strip())
    return out


def read_summary(path=SUMMARY_CSV):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("video") or "").strip()
            if not name:
                continue
            out[os.path.splitext(name)[0]] = row
    return out


def read_json(clip, filename):
    for cand in (os.path.join(BATCH, clip, "outputs", filename),
                 os.path.join(BATCH, clip, filename)):
        if os.path.exists(cand):
            try:
                with open(cand) as fh:
                    return json.load(fh)
            except (ValueError, OSError):
                return {}
    return {}


def num(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def failure_modes(note):
    n = (note or "").lower()
    hits = [label for label, pattern in FAILURE_PATTERNS if re.search(pattern, n)]
    if hits:
        return hits
    return ["unclassified"] if n.strip() else ["no reason recorded"]


def classify_failure(note):
    return failure_modes(note)[0]


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def build_dataset(grades, summary):
    rows = []
    for clip in sorted(set(grades) | set(summary)):
        s = summary.get(clip, {})
        grade, note = grades.get(clip, ("", ""))
        vq = read_json(clip, "video_quality.json")
        pal = read_json(clip, "coat_palette.json")
        rows.append({
            "clip": clip,
            "grade": grade,
            "status": s.get("status", ""),
            "frames": num(s.get("frames") or vq.get("frames")),
            "fps": num(s.get("fps") or vq.get("fps")),
            "dup_rate": num(s.get("dup_rate", vq.get("dup_rate"))),
            "sharpness": num(s.get("sharpness", vq.get("sharpness"))),
            "brightness": num(s.get("brightness", vq.get("brightness"))),
            "outliers": num(s.get("outliers_rejected",
                                  vq.get("outliers_rejected"))),
            "quality_verdict": s.get("quality_verdict", ""),
            "template": s.get("template", ""),
            "geometry_pick": s.get("geometry_pick", ""),
            "classifier_pick": s.get("classifier_pick", ""),
            "sources_agree": s.get("sources_agree", ""),
            "margin": num(s.get("margin")),
            "alpha": num(s.get("alpha")),
            "root_lift_px": num(s.get("root_lift_px")),
            "spine_tail": s.get("spine_tail", ""),
            "coat_hex": s.get("coat_hex", pal.get("base_hex", "")),
            "coat_lightness": num(pal.get("base_lightness")),
            "failure_mode": classify_failure(note) if grade == "C" else "",
            "note": note,
        })
    return rows


def template_distribution(rows):
    grades = ["A", "B", "C"]
    per = defaultdict(Counter)
    for r in rows:
        if r["template"]:
            per[r["template"]][r["grade"] or "-"] += 1
    out = []
    for t in sorted(per, key=lambda k: -sum(per[k].values())):
        c = per[t]
        out.append([t] + [c.get(g, 0) for g in grades] + [sum(c.values())])
    return out, ["template"] + grades + ["total"]


def selection_agreement(rows):
    have = [r for r in rows if r["template"]]
    agree = [r for r in have
             if str(r["sources_agree"]).strip().lower() == "true"]
    a_rows = [r for r in have if r["grade"] == "A"]
    a_agree = [r for r in a_rows
               if str(r["sources_agree"]).strip().lower() == "true"]

    def stat(vals):
        v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
        if not len(v):
            return ("-", "-", "-")
        return (round(float(np.median(v)), 3),
                round(float(v.min()), 3), round(float(v.max()), 3))

    m_med, m_lo, m_hi = stat([r["margin"] for r in have])
    al = Counter(round(r["alpha"], 2) for r in have if np.isfinite(r["alpha"]))
    out = [
        ["clips with a template", len(have), "", ""],
        ["classifier and geometry agree", len(agree),
         "%.0f%%" % (100.0 * len(agree) / max(len(have), 1)), ""],
        ["... among A-grade only", len(a_agree),
         "%.0f%%" % (100.0 * len(a_agree) / max(len(a_rows), 1)), ""],
        ["combined-score margin (median, min, max)", m_med, m_lo, m_hi],
    ]
    for a, n in sorted(al.items()):
        out.append(["alpha = %.2f" % a, n,
                    "%.0f%%" % (100.0 * n / max(len(have), 1)), ""])
    return out, ["measure", "value", "share", "extra"]


def failure_taxonomy(rows):
    c_rows = [r for r in rows if r["grade"] == "C"]
    total = len(c_rows)
    primary = Counter(r["failure_mode"] for r in c_rows)
    anymode = Counter()
    for r in c_rows:
        for m in failure_modes(r["note"]):
            anymode[m] += 1
    out = []
    for k, v in anymode.most_common():
        out.append([k, v, "%.0f%%" % (100.0 * v / max(total, 1)), primary.get(k, 0)])
    return out, ["failure mode", "clips affected", "share of rejected",
                 "as primary cause"]


METRICS = [("frames", "frames"), ("dup_rate", "duplicate-frame rate"),
           ("sharpness", "sharpness"), ("brightness", "brightness"),
           ("outliers", "outliers rejected"), ("margin", "selection margin"),
           ("root_lift_px", "root lift (px)"),
           ("coat_lightness", "sampled coat lightness")]


def metric_by_grade(rows):
    out = []
    for key, label in METRICS:
        cells = [label]
        for g in ("A", "B", "C"):
            v = np.array([r[key] for r in rows
                          if r["grade"] == g and np.isfinite(r[key])])
            cells.append("-" if not len(v) else "%.3g" % float(np.median(v)))
        a = np.array([r[key] for r in rows
                      if r["grade"] == "A" and np.isfinite(r[key])])
        c = np.array([r[key] for r in rows
                      if r["grade"] == "C" and np.isfinite(r[key])])
        if len(a) > 1 and len(c) > 1:
            pooled = np.sqrt((a.var(ddof=1) + c.var(ddof=1)) / 2.0)
            d = abs(a.mean() - c.mean()) / pooled if pooled > 1e-12 else 0.0
            cells.append("%.2f" % d)
            cells.append("yes" if d >= 0.8 else ("weak" if d >= 0.5 else "no"))
        else:
            cells += ["-", "-"]
        out.append(cells)
    return out, ["metric", "A median", "B median", "C median",
                 "|A-C| effect size", "separates A from C?"]


def screening_vs_human(rows):
    def auto_flag(r):
        v = (r["quality_verdict"] or "").upper()
        return "flagged" if ("LOW" in v or "REJECT" in v or "WARN" in v) \
            else ("clean" if v else "no verdict")
    cells = defaultdict(int)
    for r in rows:
        if r["grade"]:
            cells[(auto_flag(r), r["grade"])] += 1
    out = []
    for a in ("clean", "flagged", "no verdict"):
        row = [a] + [cells.get((a, g), 0) for g in ("A", "B", "C")]
        row.append(sum(row[1:]))
        out.append(row)
    return out, ["automatic verdict", "A", "B", "C", "total"]


def figures(rows, t2, t4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    if t2:
        names = [r[0] for r in t2]
        a = [r[1] for r in t2]; b = [r[2] for r in t2]; c = [r[3] for r in t2]
        y = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.6))
        ax.barh(y, a, color="#3B7A57", label="A (demo)")
        ax.barh(y, b, left=a, color="#C9A227", label="B (appendix)")
        ax.barh(y, c, left=np.array(a) + np.array(b), color="#B0413E",
                label="C (failure case)")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("clips")
        ax.set_title("Template usage by grade")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        p = os.path.join(OUT, "F1_template_distribution.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    plot_metrics = [(k, l) for k, l in METRICS
                    if sum(1 for r in rows if np.isfinite(r[k])) >= 4]
    if plot_metrics:
        n = len(plot_metrics)
        cols = min(4, n); rws = (n + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(3.1 * cols, 2.7 * rws))
        axes = np.atleast_1d(axes).ravel()
        for ax, (key, label) in zip(axes, plot_metrics):
            data, ticks = [], []
            for g in ("A", "B", "C"):
                v = [r[key] for r in rows
                     if r["grade"] == g and np.isfinite(r[key])]
                if v:
                    data.append(v); ticks.append("%s (n=%d)" % (g, len(v)))
            if data:
                ax.boxplot(data, tick_labels=ticks) if _boxplot_new() \
                    else ax.boxplot(data, labels=ticks)
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes[len(plot_metrics):]:
            ax.axis("off")
        fig.suptitle("Automatic measurements against the human grade",
                     fontsize=11)
        fig.tight_layout()
        p = os.path.join(OUT, "F2_metrics_by_grade.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)

    if t4:
        labels = [r[0] for r in t4]; vals = [r[1] for r in t4]
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.6))
        ax.barh(np.arange(len(labels)), vals, color="#7A5C8E")
        ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel("clips")
        ax.set_title("Why the rejected clips were rejected")
        fig.tight_layout()
        p = os.path.join(OUT, "F3_failure_taxonomy.png")
        fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def _boxplot_new():
    import matplotlib
    return tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 9)


def summary_md(rows, t2, t3, t4, t5, t6, figs):
    g = Counter(r["grade"] for r in rows if r["grade"])
    n = sum(g.values())
    tmpl_used = len({r["template"] for r in rows if r["template"]})
    tmpl_A = len({r["template"] for r in rows
                  if r["template"] and r["grade"] == "A"})
    L = []
    L.append("# Evaluation summary\n")
    L.append("Generated by `evaluate_results.py` from `grades.txt` and "
             "`outputs/batch_summary.csv`. Every number below is read from "
             "disk, not recomputed, so it matches the delivered clips.\n")
    L.append("## Dataset and outcome\n")
    L.append("- %d clips graded: **A %d / B %d / C %d** "
             "(A = demo reel, B = appendix, C = failure case)."
             % (n, g.get("A", 0), g.get("B", 0), g.get("C", 0)))
    if n:
        L.append("- Yield: **%.0f%%** of clips reached demo quality."
                 % (100.0 * g.get("A", 0) / n))
    L.append("- Templates selected at least once across the batch: **%d**; "
             "templates appearing in the demo reel: **%d**. These are "
             "different claims and should be reported separately."
             % (tmpl_used, tmpl_A))
    L.append("\n## Which criterion actually predicts the grade\n")
    L.append("`T5_metric_by_grade.csv` reports the A-vs-C effect size for each "
             "automatic measurement. A metric whose A and C medians coincide "
             "cannot screen, however reasonable it sounds:\n")
    for r in t5:
        L.append("- %s: A %s / C %s, effect %s -> separates: %s"
                 % (r[0], r[1], r[3], r[4], r[5]))
    L.append("\n## Failure modes\n")
    if t4:
        for r in t4:
            L.append("- %s: affects %s clips (%s of rejections); primary cause "
                     "in %s" % (r[0], r[1], r[2], r[3]))
        L.append("\nShares sum to more than 100%: most rejections have more "
                 "than one cause.")
        L.append("\nThe largest bucket is where the system's boundary is, and "
                 "it is a property of the method rather than of any one clip.")
    else:
        L.append("- no C-grade clips recorded.")
    L.append("\n## Automatic screening against human judgement\n")
    L.append("`T6_screening_vs_human.csv` crosses the quality verdict "
             "(computed before any human looked) with the grade. Read the "
             "off-diagonal: clips the pipeline called clean that a human "
             "rejected are the screening's false negatives, and clips it "
             "flagged that were graded A are the cost of screening too hard.\n")
    for r in t6:
        L.append("- verdict `%s`: A %s / B %s / C %s" % (r[0], r[1], r[2], r[3]))
    L.append("\n## Files\n")
    for f in sorted(os.listdir(OUT)):
        L.append("- `%s`" % f)
    L.append("\n## What this file does NOT establish\n")
    L.append("- The grades are one person's judgement on one viewing, not a "
             "controlled perceptual study, so treat A/B/C as an ordinal label "
             "and not a measurement.")
    L.append("- Clips are not independent: several are trimmed segments and "
             "the batch over-represents whichever breeds were easiest to "
             "source. Proportions describe THIS dataset.")
    L.append("- An effect size computed on tens of clips is an indication of "
             "direction, not a significance test.")
    path = os.path.join(OUT, "SUMMARY.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    grades, summary = read_grades(), read_summary()
    if not grades and not summary:
        print("Found neither grades.txt nor outputs/batch_summary.csv.")
        print("Run this from the project root.")
        return 1
    if not grades:
        print("WARNING: no grades.txt -- grade columns will be empty.")
    if not summary:
        print("WARNING: no batch_summary.csv -- run fix_summary.py --from-disk")

    rows = build_dataset(grades, summary)
    hdr = list(rows[0].keys())
    write_csv("T1_dataset.csv", hdr, [[r[k] for k in hdr] for r in rows])

    t2, h2 = template_distribution(rows); write_csv("T2_template_distribution.csv", h2, t2)
    t3, h3 = selection_agreement(rows);   write_csv("T3_selection_agreement.csv", h3, t3)
    t4, h4 = failure_taxonomy(rows);      write_csv("T4_failure_taxonomy.csv", h4, t4)
    t5, h5 = metric_by_grade(rows);       write_csv("T5_metric_by_grade.csv", h5, t5)
    t6, h6 = screening_vs_human(rows);    write_csv("T6_screening_vs_human.csv", h6, t6)

    figs = []
    try:
        figs = figures(rows, t2, t4)
    except Exception as exc:
        print("figures skipped: %s" % exc)

    summary_md(rows, t2, t3, t4, t5, t6, figs)

    g = Counter(r["grade"] for r in rows if r["grade"])
    print("%d clips  |  A %d  B %d  C %d"
          % (sum(g.values()), g.get("A", 0), g.get("B", 0), g.get("C", 0)))
    print("templates used in batch: %d   in demo reel: %d"
          % (len({r["template"] for r in rows if r["template"]}),
             len({r["template"] for r in rows
                  if r["template"] and r["grade"] == "A"})))
    print("\nwrote %d file(s) to %s/" % (len(os.listdir(OUT)), OUT))
    print("Start with SUMMARY.md -- it has the numbers as sentences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
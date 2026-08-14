#!/usr/bin/env python3
import csv, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = "outputs/batch_summary.csv"
GRADES = "outputs/demos/_grades.txt"
OUT_DIR = "outputs/evaluation"
if len(sys.argv) == 4:
    CSV, GRADES, OUT_DIR = sys.argv[1:4]

COLOUR = {"A": "#2E7D32", "B": "#F9A825", "C": "#C62828"}

def load_grades(path):
    grades = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = re.match(r"^([ABC])  (.+?)(?:  |$)", line)
            if m:
                grades[m.group(2).strip()] = m.group(1)
    return grades

def load_rows(path, grades):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        name = r["video"]
        if name.endswith(".mp4"):
            name = name[:-4]
        if name not in grades:
            print("WARNING: no grade for %s" % name)
            continue
        r["_name"] = name
        r["_grade"] = grades[name]
        out.append(r)
    return out

def short(name, n=22):
    return name if len(name) <= n else name[: n - 1] + "\u2026"

def fig_per_clip(rows, out_dir):
    data = []
    for r in rows:
        frames = int(r["frames"]); rej = int(r["outliers_rejected"])
        data.append((r["_name"], r["_grade"], rej / frames if frames else 0.0))
    data.sort(key=lambda t: -t[2])
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(range(len(data)), [d[2] for d in data],
           color=[COLOUR[d[1]] for d in data], edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([short(d[0]) for d in data], rotation=90, fontsize=7)
    ax.set_ylabel("Outlier rejections per frame")
    ax.set_xlabel("Clip (n = %d, sorted by rejection rate)" % len(data))
    ax.axhline(1.0, color="grey", ls="--", lw=0.8)
    ax.text(len(data) - 0.5, 1.02, "1 rejection per frame", ha="right", fontsize=7, color="grey")
    ax.set_title("Outlier rejection rate per clip, by manual grade")
    ax.legend(handles=[Patch(facecolor=COLOUR[g], edgecolor="black",
              label="%s  (n=%d)" % (g, sum(1 for d in data if d[1] == g))) for g in "ABC"],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "F4_outliers_per_clip.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)

def fig_duprate(rows, out_dir):
    data = [(r["_name"], r["_grade"], float(r["dup_rate"])) for r in rows]
    data.sort(key=lambda t: -t[2])
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(range(len(data)), [d[2] * 100 for d in data],
           color=[COLOUR[d[1]] for d in data], edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([short(d[0]) for d in data], rotation=90, fontsize=7)
    ax.set_ylabel("Duplicate frame rate (%)")
    ax.set_xlabel("Clip (n = %d, sorted)" % len(data))
    ax.set_title("Duplicate frame rate per clip, by manual grade")
    ax.legend(handles=[Patch(facecolor=COLOUR[g], edgecolor="black", label=g) for g in "ABC"],
              loc="upper right", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "F5_duprate_per_clip.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)

def fig_agreement(rows, out_dir):
    agree = [r for r in rows if r["sources_agree"].strip().lower() == "true"]
    dis = [r for r in rows if r["sources_agree"].strip().lower() != "true"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 2]})
    ax1.bar(["agree", "disagree"], [len(agree), len(dis)],
            color=["#2E7D32", "#C62828"], edgecolor="black")
    for i, v in enumerate([len(agree), len(dis)]):
        ax1.text(i, v + 0.4, "%d  (%.0f%%)" % (v, 100 * v / len(rows)), ha="center", fontsize=10)
    ax1.set_ylabel("Clips"); ax1.set_ylim(0, len(rows) * 0.9)
    ax1.set_title("Geometry vs classifier\n(n = %d)" % len(rows))
    data = sorted(((r["_name"], r["_grade"], float(r["margin"]),
                    r["sources_agree"].strip().lower() == "true") for r in rows), key=lambda t: t[2])
    ax2.bar(range(len(data)), [d[2] for d in data], color=[COLOUR[d[1]] for d in data],
            hatch=["" if d[3] else "//" for d in data], edgecolor="black", linewidth=0.4)
    ax2.set_xticks(range(len(data)))
    ax2.set_xticklabels([short(d[0], 18) for d in data], rotation=90, fontsize=6)
    ax2.set_ylabel("Decision margin")
    ax2.set_title("Template decision margin per clip  (hatched = estimators disagree)")
    ax2.legend(handles=[Patch(facecolor=COLOUR[g], edgecolor="black", label=g) for g in "ABC"],
               loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "F6_estimator_agreement.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)

def main():
    grades = load_grades(GRADES)
    rows = load_rows(CSV, grades)
    print("matched %d clips  (A/B/C = %d/%d/%d)" % (len(rows),
          sum(1 for r in rows if r["_grade"] == "A"),
          sum(1 for r in rows if r["_grade"] == "B"),
          sum(1 for r in rows if r["_grade"] == "C")))
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_per_clip(rows, OUT_DIR); fig_duprate(rows, OUT_DIR); fig_agreement(rows, OUT_DIR)

main()

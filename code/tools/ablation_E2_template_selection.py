#!/usr/bin/env python3
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

OUT = "outputs/evaluation"
GRADES = "grades.txt"
SUMMARY = "outputs/batch_summary.csv"

BREED_ERROR = re.compile(
    r"breed identif|wrong breed|should be|"
    r"tricolour|saddle|proportions.*wrong", re.I)


def read_grades(path=GRADES):
    out = {}
    if not os.path.exists(path):
        return out
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        p = [x for x in line.split("  ") if x != ""]
        if len(p) < 2 or p[0].strip() not in ("A", "B", "C"):
            continue
        out[p[1].strip()] = (p[0].strip(), "  ".join(p[2:]).strip())
    return out


def read_summary(path=SUMMARY):
    out = {}
    if not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path, newline="")):
        v = (row.get("video") or "").strip()
        if v:
            out[os.path.splitext(v)[0]] = row
    return out


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def rate(hits, total):
    return "-" if not total else "%.0f%% (%d/%d)" % (100.0 * hits / total, hits, total)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def build(grades, summary):
    rows = []
    for clip, s in summary.items():
        g, note = grades.get(clip, ("", ""))
        rows.append({
            "clip": clip, "grade": g, "note": note,
            "template": (s.get("template") or "").strip(),
            "geo": (s.get("geometry_pick") or "").strip(),
            "cls": (s.get("classifier_pick") or "").strip(),
            "agree": truthy(s.get("sources_agree")),
            "margin": num(s.get("margin")),
            "alpha": num(s.get("alpha")),
            "breed_error": bool(BREED_ERROR.search(note or "")),
        })
    return [r for r in rows if r["template"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    grades, summary = read_grades(), read_summary()
    if not summary:
        print("No outputs/batch_summary.csv. Run: python fix_summary.py --from-disk")
        return 1
    rows = build(grades, summary)
    if not rows:
        print("No clips with a template recorded.")
        return 1
    n = len(rows)
    graded = [r for r in rows if r["grade"]]

    agr = [r for r in rows if r["agree"]]
    dis = [r for r in rows if not r["agree"]]
    a_of = lambda rs: sum(1 for r in rs if r["grade"] == "A")
    t1 = [
        ["clips with a template", n, "", ""],
        ["classifier and geometry agree", len(agr),
         "%.0f%%" % (100.0 * len(agr) / n), ""],
        ["A-grade rate when they AGREE", "",
         rate(a_of(agr), len([r for r in agr if r["grade"]])), ""],
        ["A-grade rate when they DISAGREE", "",
         rate(a_of(dis), len([r for r in dis if r["grade"]])), ""],
    ]
    write_csv("E2_1_agreement.csv", ["measure", "count", "rate", ""], t1)

    src = Counter()
    for r in rows:
        if r["template"] == r["cls"] and r["template"] == r["geo"]:
            src["both (they agreed)"] += 1
        elif r["template"] == r["cls"]:
            src["classifier won"] += 1
        elif r["template"] == r["geo"]:
            src["geometry won"] += 1
        else:
            src["neither -- a third template"] += 1
    t2 = [[k, v, "%.0f%%" % (100.0 * v / n)] for k, v in src.most_common()]
    write_csv("E2_2_decision_source.csv", ["final template equals", "clips", "share"], t2)

    err = [r for r in rows if r["breed_error"]]
    ok = [r for r in rows if r["grade"] and not r["breed_error"]]

    def med(rs, key):
        v = np.array([r[key] for r in rs if np.isfinite(r[key])])
        return "-" if not len(v) else "%.3f" % float(np.median(v))

    t3 = [
        ["clips the grader called a breed error", len(err),
         "%.0f%% of graded" % (100.0 * len(err) / max(len(graded), 1)), ""],
        ["... of which the two sources DISagreed", sum(1 for r in err if not r["agree"]),
         rate(sum(1 for r in err if not r["agree"]), len(err)), ""],
        ["... disagreement rate among the rest", sum(1 for r in ok if not r["agree"]),
         rate(sum(1 for r in ok if not r["agree"]), len(ok)), ""],
        ["median margin, breed errors", med(err, "margin"), "", ""],
        ["median margin, the rest", med(ok, "margin"), "", ""],
        ["median alpha, breed errors", med(err, "alpha"), "", ""],
        ["median alpha, the rest", med(ok, "alpha"), "", ""],
    ]
    write_csv("E2_3_confidence_vs_error.csv",
              ["measure", "value", "rate", ""], t3)

    t4 = []
    fin = [r for r in rows if np.isfinite(r["margin"]) and r["grade"]]
    if fin:
        for q in (0.1, 0.25, 0.5, 0.75):
            thr = float(np.quantile([r["margin"] for r in fin], q))
            below = [r for r in fin if r["margin"] <= thr]
            above = [r for r in fin if r["margin"] > thr]
            t4.append(["margin <= %.3f (lowest %.0f%%)" % (thr, 100 * q),
                       len(below), rate(a_of(below), len(below)),
                       rate(a_of(above), len(above))])
    write_csv("E2_4_margin_threshold.csv",
              ["rule", "clips below", "A-rate below", "A-rate above"], t4)

    per = defaultdict(Counter)
    for r in rows:
        per[r["template"]][r["grade"] or "-"] += 1
    t5 = [[t] + [per[t].get(g, 0) for g in ("A", "B", "C")] + [sum(per[t].values())]
          for t in sorted(per, key=lambda k: -sum(per[k].values()))]
    write_csv("E2_5_template_coverage.csv",
              ["template", "A", "B", "C", "total"], t5)

    L = ["# E2 -- breed template selection\n",
         "Replaces the 2025-07-29 E2, which measured a small/medium/large tier "
         "system that no longer exists. Generated by "
         "`ablation_E2_template_selection.py` from the batch record.\n",
         "## What is measurable here\n",
         "There is no ground-truth breed label for these clips, so selection "
         "ACCURACY is not computed and is not claimed. Two weaker signals are "
         "used: disagreement between the two estimators, which is evidence "
         "that one of them is wrong; and the grader's explicit notes, which "
         "identify errors but never confirm successes and therefore bound the "
         "error rate from below only.\n",
         "## Results\n"]
    for r in t1:
        L.append("- %s: %s %s" % (r[0], r[1], r[2]))
    L.append("")
    for r in t2:
        L.append("- final template = %s: %s clips (%s)" % (r[0], r[1], r[2]))
    L.append("")
    for r in t3:
        L.append("- %s: %s %s" % (r[0], r[1], r[2]))
    L.append("\n## Does confidence predict the error?\n")
    MIN_FOR_A_DIRECTION = 8
    if err and len(err) < MIN_FOR_A_DIRECTION:
        e_dis = sum(1 for r in err if not r["agree"])
        L.append("**Not answerable from this data.** Only %d clip(s) carry a "
                 "written breed complaint, so the disagreement rate among them "
                 "(%d of %d) moves by 1/%d per clip and no direction can be "
                 "read from it. The comparison is printed in "
                 "E2_3_confidence_vs_error.csv for completeness and should be "
                 "reported as underpowered, not as a finding."
                 % (len(err), e_dis, len(err), len(err)))
        L.append("\nThe same caution applies to the agreement arm above: "
                 "%d clips had the two sources agree, so its A-rate is a "
                 "handful of clips and not a rate."
                 % len([r for r in rows if r["agree"]]))
    elif err and ok:
        e_dis = sum(1 for r in err if not r["agree"]) / len(err)
        o_dis = sum(1 for r in ok if not r["agree"]) / max(len(ok), 1)
        if e_dis - o_dis > 0.15:
            L.append("Disagreement is **more common** among the confirmed "
                     "errors (%.0f%% vs %.0f%%), so refusing or flagging on "
                     "disagreement would have caught some of them. It would "
                     "also reject many clips that were fine -- the cost is in "
                     "E2_4." % (100 * e_dis, 100 * o_dis))
        elif o_dis - e_dis > 0.15:
            L.append("Disagreement is **less** common among the confirmed "
                     "errors (%.0f%% vs %.0f%%). The signal points the wrong "
                     "way; do not screen on it." % (100 * e_dis, 100 * o_dis))
        else:
            L.append("Disagreement occurs at a similar rate among the "
                     "confirmed errors and everything else (%.0f%% vs %.0f%%). "
                     "**The system's own agreement signal does not predict its "
                     "breed errors**, so gating on it would reject good clips "
                     "without catching the bad ones. That is a negative result "
                     "and it is worth reporting: the signal is present, it is "
                     "printed, and it carries no information about this "
                     "failure." % (100 * e_dis, 100 * o_dis))
    else:
        L.append("Too few confirmed errors to compare.")
    third = [r for r in rows if r["template"] not in (r["geo"], r["cls"])]
    if third:
        L.append("\n## When the two sources disagree, the winner is sometimes "
                 "neither\n")
        L.append("The decision rule is a weighted sum of two score vectors. "
                 "Where the sources point far apart, the sum can peak on a "
                 "template that NEITHER proposed -- a compromise between two "
                 "categorical votes, which is not an operation that means "
                 "anything about a dog:\n")
        SHOW = 8
        for r in third[:SHOW]:
            L.append("- `%s`: geometry said %s, classifier said %s, the "
                     "combination chose **%s**" % (r["clip"], r["geo"],
                                                   r["cls"], r["template"]))
        if len(third) > SHOW:
            L.append("- ... and %d more; the full list is the rows of "
                     "outputs/batch_summary.csv where `template` matches "
                     "neither pick." % (len(third) - SHOW))
        L.append("\n%d of %d clips (%.0f%%) landed on a template neither "
                 "source proposed." % (len(third), len(rows),
                                       100.0 * len(third) / len(rows)))
        L.append("\nThis is a property of the rule, not of any clip, and it is "
                 "the strongest argument for refusing on low confidence rather "
                 "than always emitting a pick.")

    L.append("\n## Limits\n")
    L.append("- Errors are counted only where the grader wrote one down; a "
             "wrong pick that looked plausible is invisible here, so the true "
             "error rate is at least what is reported and probably higher.")
    L.append("- 42 clips over 12 templates is a handful per template; the "
             "per-template rows in E2_5 are descriptive, not comparative.")
    L.append("- Agreement between two estimators that share the same input "
             "video is not independence.")
    with open(os.path.join(OUT, "SUMMARY_E2.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("E2 over %d clips (%d graded, %d flagged as breed errors)"
          % (n, len(graded), len(err)))
    for r in t2:
        print("   final template = %-26s %s (%s)" % (r[0], r[1], r[2]))
    print("\nwrote E2_1..E2_5 + SUMMARY_E2.md to %s/" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
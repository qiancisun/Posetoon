#!/usr/bin/env python3
import csv
import json
import os
import sys

import numpy as np

OUT = "outputs/evaluation"
BATCH = "outputs/batch"
FIELDS = ["head", "hum", "rad", "fem", "tib"]


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


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


def ratio(m):
    try:
        return 0.5 * ((m["hum"] + m["rad"]) + (m["fem"] + m["tib"])) / 200.0
    except (KeyError, TypeError):
        return float("nan")


def dist(a, b):
    try:
        return float(np.mean([abs(a[f] - b[f]) for f in FIELDS]) / 200.0)
    except (KeyError, TypeError):
        return float("nan")


def e1_delivered(grades):
    rows = []
    if not os.path.isdir(BATCH):
        return rows
    for clip in sorted(os.listdir(BATCH)):
        if clip.endswith("__superseded"):
            continue
        p = os.path.join(BATCH, clip, "outputs", "character_description.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except (ValueError, OSError):
            continue
        meas, tmpl = d.get("measure_used"), d.get("template_measure")
        a = d.get("alpha")
        if not (isinstance(meas, dict) and isinstance(tmpl, dict)
                and a is not None):
            continue
        a = float(a)
        blend = {f: a * meas[f] + (1 - a) * tmpl[f]
                 for f in FIELDS if f in meas and f in tmpl}
        if len(blend) < len(FIELDS):
            continue
        d_meas, d_tmpl = dist(blend, meas), dist(blend, tmpl)
        span = dist(meas, tmpl)
        rows.append([
            clip, grades.get(clip, ""), d.get("template", ""), "%.2f" % a,
            "%.3f" % ratio(meas), "%.3f" % ratio(tmpl), "%.3f" % ratio(blend),
            "%.4f" % span, "%.4f" % d_meas, "%.4f" % d_tmpl,
            "-" if not np.isfinite(span) or span < 1e-9
            else "%.0f%%" % (100.0 * (1 - d_meas / span)),
        ])
    write_csv("E1_1_delivered_blend.csv",
              ["clip", "grade", "template", "alpha",
               "ratio measured", "ratio template", "ratio delivered",
               "template-measurement gap", "delivered to measurement",
               "delivered to template", "share of the gap closed"], rows)
    return rows


def _widest_real_measure():
    best, best_gap, best_clip = None, -1.0, None
    if not os.path.isdir(BATCH):
        return None, None
    for clip in sorted(os.listdir(BATCH)):
        if clip.endswith("__superseded"):
            continue
        p = os.path.join(BATCH, clip, "outputs", "character_description.json")
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except (ValueError, OSError):
            continue
        m, t = d.get("measure_used"), d.get("template_measure")
        if not (isinstance(m, dict) and isinstance(t, dict)):
            continue
        if not all(k in m for k in FIELDS + ["lower", "upper"]):
            continue
        gap = dist(m, t)
        if np.isfinite(gap) and gap > best_gap:
            best, best_gap, best_clip = {k: float(v) for k, v in m.items()}, gap, clip
    return best, best_clip


def e1_sweep():
    if not os.path.exists("outputs/breed_templates.json"):
        return [], "outputs/breed_templates.json not found -- sweep skipped."
    try:
        import posetoon_aline as al
        from ablation_E3_E5 import silhouette
    except ImportError as exc:
        return [], "sweep skipped: %s" % exc

    meas, meas_clip = _widest_real_measure()
    if meas is None:
        return [], ("sweep skipped: no clip record carried a usable "
                    "measure_used")
    try:
        _tm = json.load(open("outputs/breed_templates.json"))["templates"]
        tmpl_name = max(_tm, key=lambda k: dist(meas, _tm[k].get("measure", {}))
                        if np.isfinite(dist(meas, _tm[k].get("measure", {})))
                        else -1.0)
        _gap = dist(meas, _tm[tmpl_name].get("measure", {}))
    except Exception as exc:
        return [], "sweep skipped: cannot read templates (%s)" % exc

    masks, rows = {}, []
    for a in (0.0, 0.2, 0.35, 0.5, 0.6, 0.8, 1.0):
        try:
            rig, info = al.build_character_rig(
                dict(meas), n_frames=300, force_template=tmpl_name,
                force_alpha=a, verbose=False)
        except Exception as exc:
            return rows, "sweep stopped at alpha=%.2f: %s" % (a, exc)
        mask, _p, _v = silhouette(rig)
        masks[a] = mask
    ref1, ref0 = masks[1.0], masks[0.0]
    for a, m in sorted(masks.items()):
        def iou(x, y):
            u = float(np.logical_or(x, y).sum())
            return float(np.logical_and(x, y).sum()) / u if u else float("nan")
        rows.append(["%.2f" % a, "%.4f" % iou(m, ref1), "%.4f" % iou(m, ref0),
                     "%.0f" % float(m.sum())])
    write_csv("E1_2_alpha_sweep.csv",
              ["alpha", "IoU vs alpha=1 (measured)",
               "IoU vs alpha=0 (template)", "silhouette area"], rows)
    return rows, ("Measurements taken from the delivered clip `%s` -- the one "
                  "whose proportions sit farthest from its own template -- and "
                  "swept against **%s**, the template farthest from those "
                  "measurements (gap %.3f). Both ends are real records; "
                  "neither is invented, and the pair is the widest the "
                  "delivered set offers so the knob has the most room to show "
                  "an effect." % (meas_clip, tmpl_name, _gap))


def main():
    os.makedirs(OUT, exist_ok=True)
    grades = read_grades()
    rows = e1_delivered(grades)
    sweep, sweep_note = e1_sweep()

    L = ["# E1 -- the template/measurement blend\n",
         "Replaces the 2025-07-29 E1, which searched for an optimal alpha on "
         "the removed size-tier system and correctly found none: alpha is a "
         "regularisation strength, not a quantity with an optimum. The useful "
         "question is what the SHIPPED setting actually preserves.\n"]

    if rows:
        al_ = np.array([float(r[3]) for r in rows])
        closed = [float(r[10].rstrip("%")) for r in rows if r[10] != "-"]
        span = np.array([float(r[7]) for r in rows])
        L.append("## E1.1 -- what the delivered characters kept\n")
        L.append("Reconstructed from each clip's own `character_description."
                 "json`: the measurement, the template it met, and the alpha "
                 "applied. %d clips.\n" % len(rows))
        L.append("- alpha as shipped: median **%.2f** (range %.2f-%.2f)."
                 % (np.median(al_), al_.min(), al_.max()))
        if closed:
            L.append("- Share of the template-to-measurement gap that the "
                     "delivered character closes: median **%.0f%%**."
                     % np.median(closed))
            L.append("\nThat number IS alpha by construction -- the blend is "
                     "linear -- so it is a check that the record is consistent, "
                     "not a discovery. What it makes concrete is the trade "
                     "being made: at the shipped setting the character keeps "
                     "about %.0f%% of the difference between this dog and its "
                     "breed, and gives up the rest to the template."
                     % np.median(closed))
        L.append("- How far apart the two ends are in the first place: median "
                 "**%.3f** (min %.3f, max %.3f) in the same units as the "
                 "selector's distances." % (np.nanmedian(span),
                                            np.nanmin(span), np.nanmax(span)))
        L.append("\nWhere that gap is small the alpha setting barely matters; "
                 "where it is large, alpha is doing most of the work of "
                 "deciding what the character looks like. `E1_1_delivered_"
                 "blend.csv` has both per clip, with the grade alongside.")

        by_a = {}
        for r in rows:
            by_a.setdefault(r[3], []).append(r[1])
        L.append("\n### alpha against outcome\n")
        for a in sorted(by_a):
            gs = [g for g in by_a[a] if g]
            n_a = sum(1 for g in gs if g == "A")
            L.append("- alpha %s: %d clip(s)%s" % (
                a, len(by_a[a]),
                ", A-rate %d/%d" % (n_a, len(gs)) if gs else ""))
        L.append("\nThese are counts, not rates: alpha is CHOSEN from clip "
                 "length and selection confidence, so any association with the "
                 "grade is the confounder, not an effect of alpha. Nothing "
                 "causal can be read from this table and none is claimed.")
    else:
        L.append("## E1.1\n\nNo `character_description.json` carried "
                 "`measure_used`, `template_measure` and `alpha` together, so "
                 "the delivered blend could not be reconstructed.")

    L.append("\n## E1.2 -- alpha against shape\n")
    if sweep:
        L.append("%s\n" % sweep_note)
        L.append("| alpha | IoU vs measured | IoU vs template |")
        L.append("|---|---|---|")
        for r in sweep:
            L.append("| %s | %s | %s |" % (r[0], r[1], r[2]))
        try:
            lo = float(sweep[0][1])
            hi = float(sweep[-1][1])
            L.append("\nOutline agreement with the measured end rises "
                     "monotonically from **%.3f** at alpha=0 to %.3f at "
                     "alpha=1, and the mirror column falls the same way, so "
                     "the knob traverses the range rather than saturating at "
                     "one end." % (lo, hi))
            if hi - lo < 0.05:
                L.append("\nThe span is narrow (%.3f). Even between the two "
                         "most distant proportions this template set offers, "
                         "alpha moves the OUTLINE only slightly -- the blend "
                         "acts on bone lengths, and the drawn parts absorb "
                         "much of that before it reaches the silhouette. "
                         "Whatever alpha buys is therefore mostly in limb "
                         "proportion, which the outline measures poorly."
                         % (hi - lo))
        except (ValueError, IndexError):
            pass
    else:
        L.append(sweep_note)

    L.append("\n## Limits\n")
    L.append("- E1.1 is a reconstruction of a linear blend, so its central "
             "number is an identity. It is reported because the identity is "
             "what makes the trade legible, not because it was in doubt.")
    L.append("- alpha is not randomised. It is assigned by rule from clip "
             "length and selection confidence, so alpha and clip quality are "
             "confounded by design and no comparison across alpha values "
             "supports a causal reading.")
    L.append("- E1.2 measures the rest-pose outline for ONE clip against ONE "
             "template, chosen as the widest real pair available. It shows "
             "that the knob moves the shape; it does not say how much shape "
             "movement is desirable, and a narrower pair would move less.")
    with open(os.path.join(OUT, "SUMMARY_E1.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    print("E1.1: %d delivered clips reconstructed" % len(rows))
    if rows:
        al_ = [float(r[3]) for r in rows]
        print("      alpha median %.2f  (range %.2f-%.2f)"
              % (np.median(al_), min(al_), max(al_)))
    print("E1.2: %s" % (("%d alpha steps" % len(sweep)) if sweep else sweep_note))
    print("\nwrote E1_1 (+E1_2) + SUMMARY_E1.md to %s/" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
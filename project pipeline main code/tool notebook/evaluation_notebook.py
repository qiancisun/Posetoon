import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # ONE notebook for the whole evaluation. It DISPLAYS; it does not compute.
    #
    # The arithmetic stays in the four scripts, and this file reads what they
    # wrote to outputs/evaluation/. Two reasons for the split. Marimo requires
    # every variable to be defined in exactly one cell, so re-implementing the
    # tables here would collide with those modules on names like OUT and main.
    # More importantly a second copy of the arithmetic is a second thing that
    # can disagree with the CSVs on disk -- and the CSVs are what the thesis
    # will quote.
    #
    # Refresh everything, then reload this page:
    #
    #     python evaluate_results.py
    #     python ablation_E1_alpha.py
    #     python ablation_E2_template_selection.py
    #     python ablation_E3_E5.py
    #
    # If the batch summary looks wrong, rebuild it first -- run_one.py
    # overwrites it with the single clip it just ran:
    #
    #     python fix_summary.py --from-disk
    import csv
    import os
    import sys

    import marimo as mo

    sys.path.insert(0, os.getcwd())
    import evaluate_results as ev

    EVAL_DIR = "outputs/evaluation"

    mo.md(
        "# PoseToon — evaluation and ablation\n\n"
        "Every number below is read from the batch record on disk. Nothing "
        "here is recomputed, simulated or hand-entered.\n\n"
        "Sections: **audit** of the delivered batch, then **E1** (template / "
        "measurement blend), **E2** (which template was chosen), **E3** "
        "(character detail), **E5** (which footage works), **E4** (character "
        "against the real dog). "
        "The note at the end records why E4 was at first judged impossible."
    )
    return EVAL_DIR, csv, ev, mo, os


@app.cell
def _(EVAL_DIR, csv, mo, os):
    def show_csv(filename, title=None, note=None, page_size=20):
        """Render one results table, or say plainly that it is missing."""
        path = os.path.join(EVAL_DIR, filename)
        head = [mo.md("### %s" % (title or filename))]
        if note:
            head.append(mo.md(note))
        if not os.path.exists(path):
            head.append(mo.md("*Not generated yet — see the run commands at "
                              "the top of this notebook.*"))
            return mo.vstack(head)
        with open(path, newline="") as fh:
            rows_ = list(csv.DictReader(fh))
        if not rows_:
            head.append(mo.md("*(empty)*"))
            return mo.vstack(head)
        head.append(mo.ui.table(rows_, selection=None, page_size=page_size))
        return mo.vstack(head)

    def show_md(filename, fallback="*Not generated yet.*"):
        path = os.path.join(EVAL_DIR, filename)
        return mo.md(open(path).read()) if os.path.exists(path) \
            else mo.md(fallback)
    return show_csv, show_md


@app.cell
def _(EVAL_DIR, mo, os):
    # Regenerate anything missing, so the notebook stands on its own.
    #
    # The audit tables below are computed live from grades.txt and
    # batch_summary.csv. The ablation sections are not: they read the CSV and
    # SUMMARY files the four scripts write. Without this cell, opening the
    # notebook on a fresh copy of the project shows four empty sections and no
    # indication of why.
    #
    # The scripts are called rather than reimplemented. Their main() writes to
    # outputs/evaluation/ exactly as it does from the command line, so the
    # files this notebook reads are the same files the thesis quotes.
    def _ensure(marker, module):
        if os.path.exists(os.path.join(EVAL_DIR, marker)):
            return None
        try:
            import importlib
            importlib.import_module(module).main()
            return "generated %s" % module
        except Exception as exc:                              # noqa: BLE001
            return "could not run %s: %s" % (module, exc)

    _made = [m for m in (
        _ensure("T1_dataset.csv", "evaluate_results"),
        _ensure("E1_1_delivered_blend.csv", "ablation_E1_alpha"),
        _ensure("E2_1_agreement.csv", "ablation_E2_template_selection"),
        _ensure("E3_1_complexity_geometry.csv", "ablation_E3_E5"),
        _ensure("E4_1_cartoon_vs_real.csv", "ablation_E4_cartoon_vs_real"),
    ) if m]
    regen_note = mo.md("Regenerated: " + "; ".join("`%s`" % m for m in _made)) \
        if _made else mo.md("")
    regen_note
    return


@app.cell
def _(EVAL_DIR, ev, mo, os):
    _grades = ev.read_grades()
    _summary = ev.read_summary()
    rows = ev.build_dataset(_grades, _summary)

    _want = ["T1_dataset.csv", "E1_1_delivered_blend.csv",
             "E2_1_agreement.csv", "E3_1_complexity_geometry.csv",
             "E5_1_footage_property_vs_outcome.csv"]
    _absent = [f for f in _want
               if not os.path.exists(os.path.join(EVAL_DIR, f))]

    _msgs = []
    if not _grades:
        _msgs.append("`grades.txt` not found")
    if not _summary:
        _msgs.append("`outputs/batch_summary.csv` not found — run "
                     "`python fix_summary.py --from-disk`")
    if _absent:
        _msgs.append("missing results: " + ", ".join("`%s`" % f
                                                     for f in _absent))
    load_note = mo.md(("**Incomplete:** " + "; ".join(_msgs)) if _msgs
                      else "Loaded **%d** clips; all four experiments present."
                      % len(rows))
    load_note
    return (rows,)


@app.cell
def _(mo, rows):
    from collections import Counter as _C

    _g = _C(r["grade"] for r in rows if r["grade"])
    _n = sum(_g.values())
    _tb = len({r["template"] for r in rows if r["template"]})
    _ta = len({r["template"] for r in rows
               if r["template"] and r["grade"] == "A"})

    headline = mo.md(f"""
    ## Outcome

    | | |
    |---|---|
    | clips graded | **{_n}** |
    | A — demo reel | **{_g.get('A', 0)}** |
    | B — appendix | **{_g.get('B', 0)}** |
    | C — failure case | **{_g.get('C', 0)}** |
    | yield to demo quality | **{100.0 * _g.get('A', 0) / max(_n, 1):.0f}%** |
    | templates selected across the batch | **{_tb} of 12** |
    | templates appearing in the demo reel | **{_ta}** |

    The last two rows are different claims and should be reported separately.
    Full coverage of the template set is a property of the *batch*; the reel
    covers fewer because some clips carrying a template were rejected for
    reasons unrelated to that template.
    """)
    headline
    return


@app.cell
def _(mo, show_md):
    audit_summary = mo.vstack([
        mo.md("---\n# 1 — Audit of the delivered batch"),
        show_md("SUMMARY.md"),
    ])
    audit_summary
    return


@app.cell
def _(mo, show_csv):
    audit_tables = mo.vstack([
        show_csv("T5_metric_by_grade.csv",
                 "T5 — which automatic measurement predicts the grade",
                 "Effect size is |A − C| in pooled standard deviations. A "
                 "metric whose A and C medians sit on top of each other "
                 "cannot screen, however reasonable it sounds."),
        show_csv("T6_screening_vs_human.csv",
                 "T6 — automatic screening against human judgement",
                 "The quality verdict was computed before anyone watched the "
                 "clip. Read the off-diagonal. Note the *flagged/clean* split "
                 "is not a field the pipeline emits — it is derived by looking "
                 "for `LOW`/`REJECT`/`WARN` in the verdict string, so it is an "
                 "interpretation."),
        show_csv("T4_failure_taxonomy.csv",
                 "T4 — why clips were rejected",
                 "Multi-label, so shares sum to more than 100%. The buckets "
                 "are keyword matches on the grader's free text, not "
                 "measurements — that matching has been wrong once already, "
                 "folding *duplicate frame rate* into *same clip twice*."),
        show_csv("T2_template_distribution.csv",
                 "T2 — template usage by grade"),
    ])
    audit_tables
    return


@app.cell
def _(mo, rows):
    grade_filter = mo.ui.dropdown(
        options=["all", "A", "B", "C"], value="all", label="grade")
    per_clip_header = mo.vstack([
        mo.md("### T1 — per-clip record\n\nOne row per clip: the first place "
              "selection, quality and the human grade appear together. "
              "%d clips loaded." % len(rows)),
        grade_filter,
    ])
    per_clip_header
    return (grade_filter,)


@app.cell
def _(grade_filter, mo, rows):
    _sel = [r for r in rows
            if grade_filter.value == "all" or r["grade"] == grade_filter.value]
    t1_view = mo.ui.table(_sel, selection=None, page_size=15)
    t1_view
    return


@app.cell
def _(mo, show_csv, show_md):
    e1 = mo.vstack([
        mo.md("---\n# 2 — E1: the template / measurement blend"),
        show_md("SUMMARY_E1.md"),
        show_csv("E1_1_delivered_blend.csv",
                 "E1.1 — what each delivered character kept", page_size=15),
        show_csv("E1_2_alpha_sweep.csv",
                 "E1.2 — alpha against rest-pose shape"),
    ])
    e1
    return


@app.cell
def _(mo, show_csv, show_md):
    e2 = mo.vstack([
        mo.md("---\n# 3 — E2: which template was chosen"),
        show_md("SUMMARY_E2.md"),
        show_csv("E2_2_decision_source.csv",
                 "E2.2 — where the decision actually came from"),
        show_csv("E2_1_agreement.csv", "E2.1 — agreement between the sources"),
        show_csv("E2_3_confidence_vs_error.csv",
                 "E2.3 — confidence against confirmed errors"),
        show_csv("E2_4_margin_threshold.csv",
                 "E2.4 — margin swept as a screening threshold",
                 "Swept rather than quoted at one cut: a single flattering "
                 "threshold proves nothing about whether *any* cut separates."),
        show_csv("E2_5_template_coverage.csv", "E2.5 — coverage per template"),
    ])
    e2
    return


@app.cell
def _(mo, show_csv, show_md):
    e3e5 = mo.vstack([
        mo.md("---\n# 4 — E3: character detail, and E5: which footage works"),
        show_md("SUMMARY_E3_E5.md"),
        show_csv("E3_1_complexity_geometry.csv",
                 "E3.1 — fine against coarse, per template"),
        show_csv("E3_2_parts_dropped.csv",
                 "E3.2 — what coarse removes"),
        show_csv("E5_1_footage_property_vs_outcome.csv",
                 "E5.1 — input property against outcome"),
        show_csv("E5_2_effect_sizes.csv",
                 "E5.2 — footage measurements against the grade"),
    ])
    e3e5
    return


@app.cell
def _(mo, show_csv, show_md):
    e4 = mo.vstack([
        mo.md("---\n# 5 — E4: the character against the real dog"),
        show_md("SUMMARY_E4.md"),
        show_csv("E4_1_cartoon_vs_real.csv",
                 "E4 — outline agreement, per clip",
                 "Read the overlays below before quoting any of these. An "
                 "earlier version of this measurement returned IoU near 0.1 "
                 "for pale dogs because their bodies sit within a few levels "
                 "of the panel background, so only the outline passed the "
                 "threshold and the mask became a wire frame. The figure "
                 "looked like a poorly drawn character and was a badly taken "
                 "mask."),
    ])
    e4
    return


@app.cell
def _(EVAL_DIR, mo, os):
    _ov = os.path.join(EVAL_DIR, "E4_overlays")
    _sheet = os.path.join(EVAL_DIR, "E4_contact_sheet.png")
    _items = [mo.md("### E4 overlays — red is the real dog, green is the "
                    "character, yellow is agreement")]
    if os.path.exists(_sheet):
        _items.append(mo.image(_sheet))
    elif os.path.isdir(_ov):
        _items += [mo.image(os.path.join(_ov, f))
                   for f in sorted(os.listdir(_ov)) if f.endswith(".png")]
    else:
        _items.append(mo.md("*Not generated yet — run "
                            "`python ablation_E4_cartoon_vs_real.py`.*"))
    e4_overlays = mo.vstack(_items)
    e4_overlays
    return


@app.cell
def _(EVAL_DIR, mo, os):
    _imgs = [f for f in sorted(os.listdir(EVAL_DIR))
             if f.endswith(".png")] if os.path.isdir(EVAL_DIR) else []
    figs = mo.vstack(
        [mo.md("---\n# 6 — Figures")] +
        [mo.vstack([mo.md("**%s**" % f), mo.image(os.path.join(EVAL_DIR, f))])
         for f in _imgs]) if _imgs else mo.md(
        "---\n# 6 — Figures\n\nNone found — run `python evaluate_results.py`.")
    figs
    return


@app.cell
def _(mo):
    caveats = mo.md("""
    ---
    # 7 — What this evaluation does *not* establish

    - The grades are **one person's judgement on one viewing**. No second
      rater, no blinding, and the grader had worked on the clips. A/B/C is an
      ordinal label, not a measurement.
    - The failure buckets are **keyword matches on free text**.
    - The *flagged / clean* split is **derived from a string**, not emitted by
      the pipeline.
    - Clips are **not independent**: several are trimmed segments of the same
      source, and the batch over-represents whichever breeds were easiest to
      find. Proportions describe this dataset only.
    - **alpha is not randomised.** It is assigned by rule from clip length and
      selection confidence, so alpha and clip quality are confounded by
      design; no comparison across alpha values supports a causal reading.
    - Effect sizes over tens of clips indicate **direction, not significance**.

    ## On E4

    E4 was at first judged impossible: it needs the rendered character, and
    `render_rig` is defined inside a cell of the pipeline notebook where no
    external script can reach it. That reasoning was wrong about what E4
    needs — the rendered character is already on disk, in the right-hand
    panel of every delivered clip, so nothing has to be re-rendered.

    Its own limits are real and are listed with its results above:
    bounding-box normalisation removes absolute size, and the segmentation
    used for the real dog is a general-purpose model that fails outright on
    some coats — on one clip a white Samoyed was classified as *sheep*, with
    no dog pixels at all. Three of twenty clips were excluded after looking
    at the overlays, and that exclusion rate is part of the result rather
    than something tidied away.

    """)
    caveats
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import sys

BATCH = "outputs/batch"
SUMMARY = "outputs/batch_summary.csv"


def load_grades(path):
    rows = []
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run 'python make_reel.py "
                 f"--draft-grades' first")
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        grade, rest = line[0].upper(), line[1:].strip()
        if grade not in "ABC":
            continue
        if "  " in rest:
            name, note = rest.split("  ", 1)
        else:
            name, note = rest, ""
        rows.append((grade, name.strip(), note.strip()))
    return rows


def templates():
    out = {}
    if not os.path.exists(SUMMARY):
        return out
    for r in csv.DictReader(open(SUMMARY)):
        out[os.path.splitext(r["video"])[0]] = (r.get("template") or "").strip()
    return out


def render_of(name):
    d = os.path.join(BATCH, name, "outputs")
    for suffix in ("_rig_h264.mp4", "_rig.mp4"):
        p = os.path.join(d, name + suffix)
        if os.path.exists(p):
            return p
    return None


def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades.txt")
    ap.add_argument("--out", default="outputs/demos")
    ap.add_argument("--move", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = load_grades(a.grades)
    tmpl = templates()
    if not tmpl:
        print(f"  ({SUMMARY} not found -- filenames will have no template "
              f"prefix)")

    if not a.dry_run:
        for g in "ABC":
            d = os.path.join(a.out, g)
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

    counts = {"A": 0, "B": 0, "C": 0}
    per_template = {}
    missing, notes = [], {"A": [], "B": [], "C": []}

    for grade, name, note in rows:
        src = render_of(name)
        if src is None:
            missing.append(name)
            continue
        t = tmpl.get(name, "")
        base = os.path.basename(src)
        dst_name = f"{safe(t)}__{base}" if t else base
        dst = os.path.join(a.out, grade, dst_name)
        counts[grade] += 1
        if grade == "A" and t:
            per_template[t] = per_template.get(t, 0) + 1
        notes[grade].append((dst_name, note))
        if a.dry_run:
            continue
        if a.move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"A {counts['A']}   B {counts['B']}   C {counts['C']}"
          + ("   (dry run -- nothing written)" if a.dry_run else ""))
    if missing:
        print(f"\n  no render found for {len(missing)} clip(s):")
        for m in missing:
            print(f"     {m}")

    if per_template:
        print(f"\n  the {counts['A']} A clips cover "
              f"{len(per_template)} template(s):")
        for k, v in sorted(per_template.items(), key=lambda kv: -kv[1]):
            print(f"     {k:18} x{v}")

    if not a.dry_run:
        for g in "ABC":
            if not notes[g]:
                continue
            with open(os.path.join(a.out, g, "_notes.txt"), "w") as f:
                f.write(f"Grade {g} -- {len(notes[g])} clip(s)\n")
                f.write("Filenames are prefixed with the breed template the "
                        "rig actually used.\n\n")
                for n, note in sorted(notes[g]):
                    f.write(f"{n}\n")
                    if note:
                        f.write(f"    {note}\n")
        print(f"\n  written to {a.out}/A, /B and /C, one file per clip, "
              f"each folder with a _notes.txt")


if __name__ == "__main__":
    main()
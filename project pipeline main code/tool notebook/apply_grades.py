#!/usr/bin/env python3
import os
import shutil
import sys

GRADES = "grades.txt"

OVERRIDES = {
    "3644110-hd_1280_720_30fps":
        ("C", "high camera: dachshund shot from above, rig cannot represent it"),
    "18032366-hd_720_1280_30fps":
        ("C", "high camera: corgi shot from above"),
    "bg":
        ("C", "no gait: front limb swing 6 deg"),
    "12438625_1280_720_50fps":
        ("C", "no gait: front limb swing 9 deg"),

    "10380867-hd_720_1280_30fps":
        ("B", "breed identification wrong"),
    "MicrosoftTeams-video (3)":
        ("A", "graded by eye"),
    "MicrosoftTeams-video (7)":
        ("A", "graded by eye"),
    "blackbig":
        ("A", "graded by eye"),
    "chai":
        ("A", "graded by eye"),
    "smy":
        ("A", "graded by eye"),
}

HEADER = [
    "# Grades -- A = demo reel   B = appendix   C = failure case",
    "# grade, two spaces, clip name, two spaces, note.",
]


def parse(path):
    rows = []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("  ")
            parts = [p for p in parts if p != ""]
            if len(parts) < 2:
                continue
            grade = parts[0].strip()
            if grade not in ("A", "B", "C"):
                continue
            name = parts[1].strip()
            note = "  ".join(parts[2:]).strip()
            rows.append([grade, name, note])
    return rows


def main():
    write = "--write" in sys.argv
    if not os.path.exists(GRADES):
        print("no %s here -- run this from the project root" % GRADES)
        return 1

    rows = parse(GRADES)
    print("read %d graded clip(s) from %s\n" % (len(rows), GRADES))

    seen = set()
    changed = []
    for row in rows:
        name = row[1]
        if name in OVERRIDES:
            seen.add(name)
            new, why = OVERRIDES[name]
            if row[0] != new:
                changed.append((name, row[0], new, why))
                row[0] = new
                row[2] = why
            else:
                print("  already %s: %s" % (new, name))

    missing = [n for n in OVERRIDES if n not in seen]
    if missing:
        print("NOT FOUND in grades.txt -- check the exact spelling:")
        for n in missing:
            print("   %r" % n)
        print()

    if changed:
        print("changes:")
        for name, old, new, why in changed:
            print("  %s -> %s   %-38s %s" % (old, new, name, why))
    else:
        print("nothing to change.")

    counts = {g: sum(1 for r in rows if r[0] == g) for g in "ABC"}
    print("\nresult:  A=%d  B=%d  C=%d" % (counts["A"], counts["B"], counts["C"]))
    if counts["A"] < 20:
        print("  A is below 20 -- promote %d more before exporting."
              % (20 - counts["A"]))

    if not write:
        print("\nnothing written. Re-run with --write to apply.")
        return 0

    shutil.copyfile(GRADES, GRADES + ".bak")
    order = {"A": 0, "B": 1, "C": 2}
    rows.sort(key=lambda r: (order[r[0]], r[1].lower()))
    with open(GRADES, "w") as fh:
        for line in HEADER:
            fh.write(line + "\n")
        for g in "ABC":
            fh.write("\n# ---- %s  (%d clips) ----\n" % (g, counts[g]))
            for grade, name, note in rows:
                if grade == g:
                    fh.write("%s  %s%s\n" % (grade, name,
                                             ("  " + note) if note else ""))
    print("\nwrote %s   (previous version kept as %s.bak)" % (GRADES, GRADES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
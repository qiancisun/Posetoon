#!/usr/bin/env python3
import argparse
import os
import shutil

PROMOTE = {
    "15930157_720_1280_30fps":
        "Cardigan Corgi -- outlier figure counts joint-frames, not frames",
    "MicrosoftTeams-video (3)":
        "Samoyed -- same; the coat and outline read correctly",
    "10944870-hd_1920_1080_30fps":
        "German Shepherd -- saddle markings and proportions both correct",
    "4166357-hd_1280_720_60fps":
        "Siberian Husky -- erect ears and curled tail both read",
    "chai":
        "Shiba-type on the Samoyed template -- duplicate frames are removed "
        "before solving",
    "lachang":
        "Dachshund on the Basset Hound template after the size override -- "
        "short legs, long body, long ears",
    "12200647_720_1280_30fps":
        "Labrador -- off-profile 8% is within what the side-on solver handles",
    "16573252-hd_1920_1080_30fps":
        "Labrador -- outlier figure counts joint-frames, not frames",
}

DEMOTE = {
    "12753444_1280_720_24fps":
        "white dog: the torso coat sample landed on background, was detected "
        "as impossible for fur, and fell back to the Labrador template's "
        "black -- the opposite of the real coat",
    "10380867-hd_720_1280_30fps":
        "black dog on snow: same detection, fell back to the Great Dane "
        "template's fawn",
    "blackbig":
        "the dog jumps down from a box -- out-of-plane motion the 2D rig "
        "cannot represent",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades.txt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.grades):
        raise SystemExit(f"{a.grades} not found -- run "
                         f"'python make_reel.py --draft-grades' first")

    out, hits, counts = [], set(), {"A": 0, "B": 0, "C": 0}
    for raw in open(a.grades):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        grade, rest = line[0].upper(), line[1:].strip()
        name = rest.split("  ")[0].strip() if "  " in rest else rest.strip()

        if name in PROMOTE:
            grade, rest = "A", f"{name}  {PROMOTE[name]}"
            hits.add(name)
        elif name in DEMOTE:
            grade, rest = "C", f"{name}  {DEMOTE[name]}"
            hits.add(name)

        counts[grade] = counts.get(grade, 0) + 1
        out.append(f"{grade}  {rest}")

    missed = (set(PROMOTE) | set(DEMOTE)) - hits
    if missed:
        print("!! these clip names were not found in the file -- check the "
              "spelling against grades.txt:")
        for m in sorted(missed):
            print(f"     {m}")

    print(f"\nA={counts['A']}  B={counts['B']}  C={counts['C']}")
    if counts["A"] < 20:
        print(f"   still {20 - counts['A']} short of twenty. Open the contact "
              f"sheet and promote from B by eye.")
    else:
        print("   twenty or more in the demo reel.")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return
    shutil.copy2(a.grades, a.grades + ".bak")
    with open(a.grades, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {a.grades}  (backup at {a.grades}.bak)")
    print("next:  python make_reel.py --grades grades.txt")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

BATCH = "outputs/batch"


def already_done():
    if not os.path.isdir(BATCH):
        return []
    out = []
    for d in sorted(os.listdir(BATCH)):
        rend = os.path.join(BATCH, d, "outputs", d + "_rig.mp4")
        h264 = os.path.join(BATCH, d, "outputs", d + "_rig_h264.mp4")
        if os.path.exists(rend) or os.path.exists(h264):
            out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*", help="one or more clip paths")
    ap.add_argument("--list", action="store_true",
                    help="list clips already present in outputs/batch")
    ap.add_argument("--no-reencode", action="store_true")
    ap.add_argument("--open", action="store_true",
                    help="print the render path when finished")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if args.list:
        done = already_done()
        print(f"{len(done)} clip(s) in {BATCH}:")
        for d in done:
            print("   ", d)
        return
    if not args.videos:
        ap.error("give at least one clip, or --list")

    missing = [v for v in args.videos if not os.path.exists(v)]
    if missing:
        sys.exit("not found: " + ", ".join(missing))
    if not os.path.exists("run_all.py"):
        sys.exit("run this from the project root -- run_all.py is not here")

    tmp = tempfile.mkdtemp(prefix="posetoon_one_")
    try:
        for v in args.videos:
            dst = os.path.join(tmp, os.path.basename(v))
            try:
                os.symlink(os.path.abspath(v), dst)
            except (OSError, NotImplementedError):
                shutil.copy2(v, dst)

        print(f"running {len(args.videos)} clip(s) through run_all.py\n")
        rc = subprocess.call([args.python, "run_all.py", tmp])
        if rc != 0:
            print(f"\nrun_all.py exited with {rc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    names = [os.path.splitext(os.path.basename(v))[0] for v in args.videos]

    if not args.no_reencode and os.path.exists("reencode_h264.py"):
        for n in names:
            d = os.path.join(BATCH, n)
            if os.path.isdir(d):
                subprocess.call([args.python, "reencode_h264.py", "--dir", d])

    print()
    for n in names:
        for suffix in ("_rig_h264.mp4", "_rig.mp4"):
            p = os.path.join(BATCH, n, "outputs", n + suffix)
            if os.path.exists(p):
                print(f"  {p}")
                break
        else:
            print(f"  !! no render produced for {n} -- see "
                  f"{os.path.join(BATCH, n, 'run.log')}")

    done = already_done()
    print(f"\n{len(done)} clip(s) now in {BATCH}. "
          f"make_reel.py and reencode_h264.py see all of them.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def find_renders(root):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith("_rig.mp4") and not f.endswith("_rig_h264.mp4"):
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def codec_of(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name", "-of",
                        "csv=p=0", path], capture_output=True, text=True)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/batch")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite the original file instead of writing "
                         "<name>_rig_h264.mp4")
    ap.add_argument("--crf", type=int, default=20,
                    help="quality, lower is better; 18-23 is a sane range")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH -- install it or run this on a "
                 "machine that has it.")

    vids = find_renders(args.dir)
    if not vids:
        sys.exit(f"no *_rig.mp4 found under {args.dir}")

    print(f"{len(vids)} render(s) under {args.dir}\n")
    total_before = total_after = 0
    for i, src in enumerate(vids, 1):
        before = os.path.getsize(src)
        codec = codec_of(src)
        dst = src.replace("_rig.mp4", "_rig_h264.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src,
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{i}/{len(vids)}] FAILED {os.path.basename(src)}")
            print("    " + r.stderr.strip().splitlines()[-1])
            continue
        after = os.path.getsize(dst)
        total_before += before
        total_after += after
        print(f"[{i}/{len(vids)}] {os.path.basename(src):48} "
              f"{codec:6} {before/1e6:6.1f} MB -> h264 {after/1e6:5.1f} MB")
        if args.replace:
            os.replace(dst, src)

    if total_before:
        print(f"\ntotal {total_before/1e6:.0f} MB -> {total_after/1e6:.0f} MB "
              f"({total_after/total_before*100:.0f}%)")
    print("\nH.264 in an MP4 container plays in browsers, QuickTime, "
          "PowerPoint and Keynote.\nUse these for the demo and the viva.")


if __name__ == "__main__":
    main()
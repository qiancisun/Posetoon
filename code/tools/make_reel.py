#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

BATCH = "outputs/batch"
SUMMARY = "outputs/batch_summary.csv"
OUT_H, PAD = 480, 40


def load_summary():
    if not os.path.exists(SUMMARY):
        return {}
    rows = {}
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            rows[os.path.splitext(r["video"])[0]] = r
    return rows


def load_grades(path):
    out = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            grade, rest = line[0].upper(), line[1:].strip()
            if grade not in "ABC":
                print(f"  ignoring line without an A/B/C grade: {raw.rstrip()}")
                continue
            if "  " in rest:
                name, note = rest.split("  ", 1)
                out.append((grade, name.strip(), note.strip()))
            else:
                out.append((grade, rest, ""))
    return out


def render_path(name):
    for cand in (os.path.join(BATCH, name, "outputs", name + "_rig_h264.mp4"),
                 os.path.join(BATCH, name, "outputs", name + "_rig.mp4")):
        if os.path.exists(cand):
            return cand
    return None


def title_card(text_lines, width, path, seconds=2.0):
    filters = [f"color=c=0xF0F0F0:s={width}x{OUT_H}:d={seconds}"]
    draw = []
    y0 = OUT_H // 2 - 22 * len(text_lines) // 2
    for i, line in enumerate(text_lines):
        safe = line.replace("'", "").replace(":", "\\:").replace("%", "")
        size = 34 if i == 0 else 20
        draw.append(f"drawtext=text='{safe}':fontcolor=0x1E1E1E:fontsize={size}"
                    f":x=(w-text_w)/2:y={y0 + i * 34}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", filters[0],
           "-vf", ",".join(draw) if draw else "null",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-r", "30", path]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def normalise(src, dst, width):
    vf = (f"scale=-2:{OUT_H},pad={width}:{OUT_H}:(ow-iw)/2:0:color=0xF0F0F0")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-r", "30", "-an", dst]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def probe_width(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of",
                        "csv=p=0", path], capture_output=True, text=True)
    try:
        w, h = (int(x) for x in r.stdout.strip().split(","))
        return int(round(w * OUT_H / h))
    except ValueError:
        return 0


def build_reel(items, summary, out_path):
    widths = [probe_width(render_path(n)) for _g, n, _t in items]
    widths = [w for w in widths if w > 0]
    if not widths:
        print("  nothing to build")
        return False
    width = max(widths)
    width += width % 2

    tmp = tempfile.mkdtemp(prefix="reel_")
    parts = []
    for i, (grade, name, note) in enumerate(items):
        src = render_path(name)
        if src is None:
            print(f"  !! no render for {name}")
            continue
        r = summary.get(name, {})
        lines = [r.get("template", name)]
        if r:
            lines.append(f"template {r.get('template','?')}   "
                         f"alpha {r.get('alpha','?')}   "
                         f"root lift {r.get('root_lift_px','?')} px")
            lines.append(f"{r.get('frames','?')} frames   "
                         f"duplicate rate {r.get('dup_rate','?')}   "
                         f"outliers rejected {r.get('outliers_rejected','?')}")
        lines.append(f"[{grade}] {name}")
        if note:
            lines.append(note)
        card = os.path.join(tmp, f"{i:03d}_card.mp4")
        clip = os.path.join(tmp, f"{i:03d}_clip.mp4")
        if title_card(lines, width, card) and normalise(src, clip, width):
            parts += [card, clip]
        else:
            print(f"  !! failed to prepare {name}")

    if not parts:
        return False
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    ok = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat",
                         "-safe", "0", "-i", lst, "-c", "copy",
                         "-movflags", "+faststart", out_path],
                        capture_output=True, text=True).returncode == 0
    shutil.rmtree(tmp, ignore_errors=True)
    if ok:
        mb = os.path.getsize(out_path) / 1e6
        print(f"  saved {out_path}  ({len(parts)//2} clips, {width}x{OUT_H}, "
              f"{mb:.0f} MB)")
    return ok


def contact_sheet(summary, out_path, per_row=5):
    from PIL import Image, ImageDraw
    names = sorted(summary) or sorted(
        d for d in os.listdir(BATCH) if os.path.isdir(os.path.join(BATCH, d)))
    tmp = tempfile.mkdtemp(prefix="sheet_")
    tiles = []
    for n in names:
        src = render_path(n)
        if src is None:
            continue
        png = os.path.join(tmp, n.replace("/", "_") + ".png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src,
                        "-vf", "select=eq(n\\,30),scale=-1:200", "-vsync", "0",
                        "-frames:v", "1", png], capture_output=True)
        if os.path.exists(png):
            tiles.append((n, Image.open(png).convert("RGB")))
    if not tiles:
        print("  no frames extracted")
        return
    tw = max(t.width for _n, t in tiles)
    th = max(t.height for _n, t in tiles) + 22
    rows = -(-len(tiles) // per_row)
    sheet = Image.new("RGB", (per_row * tw, rows * th), (240, 240, 240))
    dr = ImageDraw.Draw(sheet)
    for k, (n, im) in enumerate(tiles):
        r, c = divmod(k, per_row)
        sheet.paste(im, (c * tw, r * th + 22))
        label = summary.get(n, {}).get("template", "")
        dr.text((c * tw + 4, r * th + 5), f"{n[:34]}   {label}",
                fill=(30, 30, 30))
    sheet.save(out_path)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  saved {out_path}  ({len(tiles)} clips, {per_row} per row)")


SCREENING = "video_screening.csv"


def load_screening():
    if not os.path.exists(SCREENING):
        return {}
    out = {}
    with open(SCREENING) as f:
        for r in csv.DictReader(f):
            key = None
            for k in ("video", "file", "filename", "name", "clip"):
                if r.get(k):
                    key = os.path.splitext(r[k])[0]
                    break
            if key is None:
                continue
            def num(*names):
                for n in names:
                    v = r.get(n)
                    if v not in (None, ""):
                        try:
                            return float(v)
                        except ValueError:
                            pass
                return None
            out[key] = {"horiz": num("horizontality", "horiz"),
                        "off": num("off_profile", "off")}
    return out


def draft_grades(summary, path):
    scr = load_screening()
    rows, counts = [], {"A": 0, "B": 0, "C": 0}
    for name in sorted(summary):
        r = summary[name]
        s = scr.get(name, {})

        def f(key, default=0.0):
            try:
                return float(r.get(key, default) or default)
            except ValueError:
                return default

        off = s.get("off")
        dup, frames = f("dup_rate"), f("frames", 1)
        alpha, lift = f("alpha", 0.35), f("root_lift_px")
        outl = f("outliers_rejected") / max(1.0, frames)
        status = (r.get("status") or "").lower()

        why = []
        if status and status not in ("ok", "done", "completed", "success"):
            why.append(f"pipeline reported status={status}")
        if off is not None and off > 0.20:
            why.append(f"off-profile in {off:.0%} of sampled frames")
        if dup > 0.15:
            why.append(f"duplicate frame rate {dup:.0%}")
        if frames < 60:
            why.append(f"only {frames:.0f} frames")
        if alpha > 0.40:
            why.append(f"breed decision not corroborated (alpha {alpha:.2f})")
        if lift > 20:
            why.append(f"root lift {lift:.0f}px -- measured limbs disagree "
                       f"with the template")
        if outl > 0.60:
            why.append(f"{outl:.0%} of frames rejected as outliers")

        soft = []
        if off is not None and 0.05 < off <= 0.20:
            soft.append(f"off-profile {off:.0%}")
        if 0.02 < dup <= 0.15:
            soft.append(f"duplicate frames {dup:.0%}")
        if 12 < lift <= 20:
            soft.append(f"root lift {lift:.0f}px")
        if 0.30 < outl <= 0.60:
            soft.append(f"outliers {outl:.0%}")

        if why:
            grade, note = "C", "; ".join(why)
        elif soft:
            grade, note = "B", "; ".join(soft)
        else:
            grade, note = "A", ""
        counts[grade] += 1
        rows.append((grade, name, note, r.get("template", "")))

    with open(path, "w") as f:
        f.write("# Draft grades -- WATCH THE CLIPS AND OVERRIDE THESE.\n")
        f.write("# grade, two spaces, clip name, two spaces, note.\n")
        f.write("# A = demo reel   B = appendix   C = failure case\n\n")
        for grade in "ABC":
            block = [x for x in rows if x[0] == grade]
            if not block:
                continue
            f.write(f"# ---- {grade}  ({len(block)} clips) ----\n")
            for _g, name, note, tmpl in block:
                line = f"{grade}  {name}"
                if note:
                    line += f"  {note}"
                elif tmpl:
                    line += f"  {tmpl}"
                f.write(line + "\n")
            f.write("\n")

    print(f"  saved {path}   A={counts['A']}  B={counts['B']}  "
          f"C={counts['C']}")
    if not scr:
        print(f"  ({SCREENING} not found -- off-profile was not available, so "
              f"the draft is missing its strongest signal)")
    if counts["A"] < 20:
        print(f"  !! only {counts['A']} clips reach A. The brief asks for "
              f"about 20 good demos.")
        print(f"     Promote from B by eye, or shoot for "
              f"{20 - counts['A']} more clips.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grades", default="grades.txt")
    ap.add_argument("--only", default=None, choices=["A", "B", "C"])
    ap.add_argument("--contact-only", action="store_true")
    ap.add_argument("--draft-grades", action="store_true",
                    help="propose A/B/C from the measured numbers and write "
                         "them to --grades, then stop")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not on PATH")

    summary = load_summary()
    print(f"batch summary: {len(summary)} clip(s)\n")

    if args.draft_grades:
        if not summary:
            sys.exit(f"{SUMMARY} not found -- run the batch first")
        print("draft grades:")
        draft_grades(summary, args.grades)
        return

    print("contact sheet:")
    contact_sheet(summary, "outputs/contact_sheet.png")
    if args.contact_only:
        return

    if not os.path.exists(args.grades):
        print(f"\nno {args.grades} yet -- write one line per clip:\n")
        for n in sorted(summary):
            print(f"A  {n}")
        print("\nchange the leading letter to B or C while reviewing, add a "
              "note after two spaces for the C clips, then re-run.")
        return

    graded = load_grades(args.grades)
    keep = [g for g in graded if args.only is None or g[0] == args.only]
    main_items = [g for g in keep if g[0] in "AB"]
    fail_items = [g for g in keep if g[0] == "C"]

    if main_items:
        print(f"\ndemo reel ({len(main_items)} clips):")
        build_reel(main_items, summary, "outputs/demo_reel.mp4")
    if fail_items:
        print(f"\nfailure-case reel ({len(fail_items)} clips):")
        build_reel(fail_items, summary, "outputs/failure_cases.mp4")


if __name__ == "__main__":
    main()
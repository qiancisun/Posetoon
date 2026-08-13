import hashlib
import json
import os
import re
import subprocess
import sys
import time

THEME = {
    "bg":        "#FFFFFF",
    "panel":     "#F4EDE3",
    "accent":    "#8A6A45",
    "accent_hi": "#75593A",
    "text":      "#332A20",
    "muted":     "#7A6B58",
}

THEME_GREEN = {
    "bg": "#FFFFFF",
    "panel": "#EDF4EE",
    "accent": "#3F7D58",
    "accent_hi": "#336547",
    "text": "#22302A",
    "muted": "#5C6F63",
}

THEME_CSS = """
<style>
  .stApp {{ background: {bg}; color: {text}; }}
  h1, h2, h3, h4 {{ color: {text}; letter-spacing: -.01em; }}
  h1 {{ font-weight: 650; }}
  .stCaption, [data-testid="stCaptionContainer"] {{ color: {muted}; }}
  .stButton > button {{
      background: {accent};
      color: #fff;
      border: 1px solid {accent};
      border-radius: 6px;
      font-weight: 550;
  }}
  .stButton > button:hover {{
      background: {accent_hi}; border-color: {accent_hi}; color: #fff;
  }}
  .stDownloadButton > button {{
      background: transparent; color: {accent};
      border: 1px solid {accent}; border-radius: 6px;
  }}
  .stDownloadButton > button:hover {{ background: {panel}; color: {accent_hi}; }}
  [data-testid="stFileUploaderDropzone"] {{
      background: {panel};
      border: 1px dashed {accent};
      border-radius: 8px;
  }}
  [data-testid="stMetricValue"] {{ color: {text}; font-size: 1.15rem; }}
  [data-testid="stMetricLabel"] {{ color: {muted}; }}
  hr {{ border-color: rgba(0,0,0,.08); }}
  a, a:visited {{ color: {accent}; }}
  [data-testid="stVerticalBlockBorderWrapper"] {{
      border-radius: 8px;
  }}
</style>
"""


def theme_css(theme=None):
    return THEME_CSS.format(**(theme or THEME))


PROJECT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(PROJECT, "videos")
BATCH_DIR = os.path.join(PROJECT, "outputs", "batch")
RUN_ONE = os.path.join(PROJECT, "run_one.py")
LOCK = os.path.join(PROJECT, "outputs", ".streamlit_run.lock")


def safe_stem(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = re.sub(r"[\s()\[\]{}'\"`$&|;<>*?!#\\/]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("._-")
    return stem or "clip"


def unique_stem(stem, video_dir=VIDEO_DIR, batch_dir=BATCH_DIR):
    def taken(s):
        if os.path.isdir(os.path.join(batch_dir, s)):
            return True
        if not os.path.isdir(video_dir):
            return False
        return any(os.path.splitext(f)[0] == s for f in os.listdir(video_dir))

    if not taken(stem):
        return stem
    n = 2
    while taken("%s_%d" % (stem, n)):
        n += 1
    return "%s_%d" % (stem, n)


def build_command(python_exe, video_path, run_one=RUN_ONE):
    return [python_exe, run_one, video_path]


def child_env(base=None):
    env = dict(base if base is not None else os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    return env


ROTATIONS = {"None": 0, "90 clockwise": 90, "90 anticlockwise": 270,
             "180": 180}


def probe_rotation(path):
    for args in (["-show_entries", "stream_side_data=rotation"],
                 ["-show_entries", "stream_tags=rotate"]):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0"] + args +
                ["-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return 0
        val = (out.stdout or "").strip().splitlines()
        if val:
            try:
                return int(round(float(val[0]))) % 360
            except ValueError:
                continue
    return 0


def preview_frame(path, extra_rotation=0, at=1.0):
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(at), "-i", path,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout:
        return None
    if not extra_rotation:
        return out.stdout
    try:
        import io

        from PIL import Image
        im = Image.open(io.BytesIO(out.stdout))
        im = im.rotate(-extra_rotation, expand=True)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return out.stdout


def normalise_upload(src, dst, extra_rotation=0):
    vf = []
    for _ in range(((extra_rotation % 360) // 90)):
        vf.append("transpose=1")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            "-metadata:s:v", "rotate=0", "-movflags", "+faststart", dst]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if r.returncode != 0 or not os.path.exists(dst):
        return False, (r.stderr or "ffmpeg failed").strip()[:400]
    return True, "ok"


def find_render(stem, batch_dir=BATCH_DIR):
    root = os.path.join(batch_dir, stem)
    if not os.path.isdir(root):
        return None
    prefer = os.path.join(root, "outputs", "%s_rig_h264.mp4" % stem)
    if os.path.exists(prefer):
        return prefer
    hits = []
    for dirpath, _dn, files in os.walk(root):
        for f in files:
            if f.endswith("_rig_h264.mp4"):
                hits.append((0, os.path.join(dirpath, f)))
            elif f.endswith("_rig.mp4"):
                hits.append((1, os.path.join(dirpath, f)))
    return min(hits)[1] if hits else None


def verify_render(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, "ffprobe not available — stream not verified."
    fmt = (out.stdout or "").strip()
    if out.returncode != 0 or not fmt or "yuv" not in fmt:
        return False, ("The render exists but its video stream will not "
                       "decode (pix_fmt %r). Re-encode with "
                       "`python reencode_h264.py`."
                       % (fmt or "unreadable"))
    return True, fmt


CHAR_PANEL_W = 600
SEAM_TRIM = 3


def character_only(three_panel, force=False):
    if not three_panel or not os.path.exists(three_panel):
        return None
    if three_panel.endswith(DERIVED_SUFFIX):
        return three_panel
    out = os.path.splitext(three_panel)[0] + DERIVED_SUFFIX
    if os.path.exists(out) and not force and \
            os.path.getmtime(out) >= os.path.getmtime(three_panel):
        return out
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "csv=p=0", three_panel],
            capture_output=True, text=True, timeout=30)
        width = int((probe.stdout or "0").strip() or 0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if width <= 0:
        return None
    w = CHAR_PANEL_W if width > CHAR_PANEL_W else max(2, (width // 3) & ~1)
    trim = SEAM_TRIM if width > CHAR_PANEL_W + SEAM_TRIM else 0
    cw = (w - trim) & ~1
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", three_panel,
             "-vf", "crop=%d:ih:iw-%d:0" % (cw, cw),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
             "-movflags", "+faststart", out],
            capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if r.returncode == 0 and os.path.exists(out) else None


def lock_held(path=LOCK, stale_after=3 * 3600):
    if not os.path.exists(path):
        return False
    try:
        if time.time() - os.path.getmtime(path) > stale_after:
            os.remove(path)
            return False
    except OSError:
        return False
    return True


def acquire_lock(path=LOCK):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("%d %f\n" % (os.getpid(), time.time()))


def release_lock(path=LOCK):
    try:
        os.remove(path)
    except OSError:
        pass


def stream_run(cmd, cwd=PROJECT, env=None):
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or child_env(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(proc.stdout.readline, ""):
        yield line.rstrip("\n")
    proc.stdout.close()
    yield proc.wait()


def preflight(python_exe):
    problems = []
    if not os.path.exists(RUN_ONE):
        problems.append("`run_one.py` not found — is this file in the project "
                        "root?")
    if not os.path.isdir(VIDEO_DIR):
        problems.append("`videos/` not found — the pipeline reads uploads "
                        "from there.")
    if not os.path.exists(python_exe):
        problems.append("Interpreter `%s` not found. Set it under Settings "
                        "at the bottom of the page — it must be the "
                        "environment with mmpose installed, not the system "
                        "python." % python_exe)
    return problems


def main():
    import streamlit as st

    st.set_page_config(page_title="PoseToon", layout="wide")
    st.markdown(theme_css(), unsafe_allow_html=True)
    st.title("PoseToon")
    st.caption("Upload a clip of a dog in profile — the pipeline tracks its "
               "skeleton, picks a breed template, samples the coat from the "
               "footage, and animates a 2D character from the motion.")

    default_py = os.environ.get("PY") or sys.executable
    python_exe = st.session_state.get("python_exe_input") or default_py

    if "stem" not in st.session_state:
        st.session_state.stem = None
    if "log" not in st.session_state:
        st.session_state.log = []

    problems = preflight(python_exe)
    for _p in problems:
        st.error(_p)

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Your video")
        up = st.file_uploader("", type=["mp4", "mov", "webm", "m4v"],
                              disabled=bool(problems),
                              label_visibility="collapsed")
        st.caption(
            "A dog moving side-on to the camera works best.  \n"
            "The whole dog should stay in frame, with nothing crossing in "
            "front of it.  \n"
            "A few seconds is enough."
        )

        go, rotation = False, 0
        if up is not None:
            os.makedirs(VIDEO_DIR, exist_ok=True)
            tmp = os.path.join(VIDEO_DIR, ".upload_preview.bin")
            data = up.getbuffer()
            if st.session_state.get("tmp_size") != len(data) or \
                    not os.path.exists(tmp):
                with open(tmp, "wb") as fh:
                    fh.write(data)
                st.session_state.tmp_size = len(data)
            st.session_state.tmp_upload = tmp

            choice = st.radio(
                "Orientation", list(ROTATIONS.keys()), horizontal=True,
                help="Some clips store the frames sideways. The pipeline reads "
                     "the pixels, not the container's rotation flag, so what "
                     "you see below is what it will see.")
            rotation = ROTATIONS[choice]

            png = preview_frame(tmp, rotation)
            if png:
                _image(st, png)
            else:
                st.video(up)

            flagged = probe_rotation(tmp)
            if flagged:
                st.caption("This clip carries a %d degree rotation flag. It is "
                           "applied automatically, so leave the control on "
                           "None unless the preview above still looks wrong."
                           % flagged)

            busy = lock_held()
            if busy:
                st.warning("Another run is in progress. Two at once corrupt "
                           "the encoded output.")
            go = st.button("Generate", type="primary",
                           use_container_width=True, disabled=busy)

    if go and up is not None:
        stem = unique_stem(safe_stem(up.name))
        saved = os.path.join(VIDEO_DIR, stem + ".mp4")
        os.makedirs(VIDEO_DIR, exist_ok=True)
        src = st.session_state.get("tmp_upload")
        if not src or not os.path.exists(src):
            with left:
                st.error("The upload was lost between steps. Choose the file "
                         "again.")
            return
        ok_norm, why = normalise_upload(src, saved, rotation)
        if not ok_norm:
            with left:
                st.error("Could not prepare the video: %s" % why)
            return

        acquire_lock()
        lines, code = [], None
        try:
            with left:
                with st.spinner("Generating — this takes a few minutes."):
                    for item in stream_run(build_command(python_exe, saved)):
                        if isinstance(item, int):
                            code = item
                            break
                        lines.append(item)
        except Exception as exc:
            code, lines = -1, lines + ["wrapper error: %s" % exc]
        finally:
            release_lock()
        st.session_state.stem = stem
        st.session_state.log = lines

    stem = st.session_state.stem
    render = find_render(stem) if stem else None

    with right:
        st.subheader("Rigged 2D character")
        if not render:
            st.container(border=True).markdown(
                "&nbsp;\n\nThe character will appear here.\n\n&nbsp;")
        else:
            ok, detail = verify_render(render)
            if ok is False:
                st.error(detail)
            elif ok is None:
                st.caption(detail)
            char = character_only(render)
            if char:
                st.video(char)
                with open(char, "rb") as fh:
                    st.download_button("Download", fh.read(),
                                       file_name=os.path.basename(char),
                                       mime="video/mp4",
                                       use_container_width=True)
            else:
                st.warning("Could not isolate the character panel; the "
                           "comparison below still holds the full render.")

    st.divider()
    st.subheader("Side by side")
    st.caption("Your video, the tracked skeleton, and the character, in step.")
    if render:
        st.video(render)
        with open(render, "rb") as fh:
            st.download_button("Download comparison", fh.read(),
                               file_name=os.path.basename(render),
                               mime="video/mp4")
    elif stem:
        st.error("No character was produced from that clip.")
        with st.expander("Details"):
            st.code("\n".join(st.session_state.log) or "(no output)",
                    language="text")
            st.caption("The clip's own log is at "
                       "`outputs/batch/%s/run.log`." % stem)
    else:
        st.container(border=True).markdown(
            "&nbsp;\n\nThe comparison will appear here after you "
            "generate.\n\n&nbsp;")

    _showcase(st)
    _settings(st, python_exe, problems)


def _rerun(st):
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


def _image(st, data):
    try:
        st.image(data, use_container_width=True)
    except TypeError:
        try:
            st.image(data, width="stretch")
        except TypeError:
            st.image(data)


def _settings(st, python_exe, problems):
    with st.expander("Settings", expanded=bool(problems)):
        st.text_input(
            "Python interpreter",
            value=st.session_state.get("python_exe", python_exe),
            key="python_exe_input",
            help="The environment with mmpose installed. run_one.py is "
                 "executed with it as a subprocess; no project file is "
                 "imported or modified.")
        st.caption("Change this only if generation fails with an import "
                   "error.")


DERIVED_SUFFIX = "_character.mp4"
POSTER_DIR = os.path.join(PROJECT, "outputs", ".app_cache",
                          "posters")
DEMO_DIR = os.path.join(PROJECT, "outputs", "demos", "A")
GRADES_FILE = os.path.join(PROJECT, "grades.txt")


def gallery_clips(n=None, demo_dir=DEMO_DIR, grades_file=GRADES_FILE,
                  batch_dir=BATCH_DIR):
    paths = []
    if os.path.isdir(demo_dir):
        paths = [os.path.join(demo_dir, f)
                 for f in sorted(os.listdir(demo_dir))
                 if f.endswith(".mp4") and not f.endswith(DERIVED_SUFFIX)]
    if not paths and os.path.exists(grades_file):
        for raw in open(grades_file):
            line = raw.rstrip("\n")
            if not line.startswith("A  "):
                continue
            parts = [x for x in line.split("  ") if x != ""]
            if len(parts) < 2:
                continue
            r = find_render(parts[1].strip(), batch_dir=batch_dir)
            if r:
                paths.append(r)
    return paths[:n] if n else paths


def poster_frame(path, at=1.0):
    try:
        os.makedirs(POSTER_DIR, exist_ok=True)
    except OSError:
        return None
    key = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:16]
    cache = os.path.join(POSTER_DIR, key + ".png")
    try:
        if os.path.exists(cache) and \
                os.path.getmtime(cache) >= os.path.getmtime(path):
            return cache
    except OSError:
        pass
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", str(at), "-i", path,
             "-frames:v", "1", cache],
            capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return cache if r.returncode == 0 and os.path.exists(cache) else None


def _showcase(st):
    paths = gallery_clips()
    if not paths:
        return
    st.divider()
    st.subheader("Showcase")
    st.caption("%d clips from the project's demo reel. Left to right in each: "
               "the source video, the tracked skeleton, the character."
               % len(paths))

    chosen = st.session_state.get("showcase_pick")
    if chosen and chosen in paths:
        st.video(chosen)
        if st.button("Close", key="showcase_close"):
            st.session_state.showcase_pick = None
            _rerun(st)

    cols = st.columns(4)
    for i, path in enumerate(paths):
        with cols[i % 4]:
            poster = poster_frame(path)
            if poster:
                _image(st, poster)
            if st.button("Play", key="showcase_%d" % i,
                         use_container_width=True):
                st.session_state.showcase_pick = path
                _rerun(st)


if __name__ == "__main__":
    main()

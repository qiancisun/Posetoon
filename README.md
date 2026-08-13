# PoseToon

A rigged 2D cartoon dog, animated from ordinary video of a real one.



---

## Overview

Given a short clip of a dog in profile, the pipeline tracks its skeleton,
measures its bone proportions, picks one of twelve breed templates and deforms
it partway toward those measurements, samples the coat colour from the footage,
and drives the character with the tracked motion.

The output is a three-panel comparison: the source video, the tracked skeleton,
and the character, playing in step.

Scope: one dog, side on, in a continuous shot. Footage shot from above is
outside what a flat rig can represent.

---

## Repository structure

```
project pipeline main code/     the pipeline
  tool notebook/                screening, grading, evaluation and ablation scripts
  results and data/             breed templates, grades, evaluation tables and figures
Old version notebook/           46 superseded versions, kept as a record of the work
Webpage code/                   a Streamlit front end
Demo video/                     the delivered clips, by grade
```

### project pipeline main code

The seven stages, run in order by `run_all.py`, plus the modules they import.

| | |
|---|---|
| `0_build_breed_templates.py` | builds the twelve breed templates from photographs |
| `1_extract_keypoints.py` | pose estimation |
| `2_repair_keypoints.py` | duplicate-frame retiming, outlier rejection, smoothing |
| `3_coat_colour.py` | samples the coat colour from the torso |
| `4_select_breed.py` | chooses a template from appearance and skeletal geometry |
| `posetoon_pipeline.py` | rigging, motion retargeting and rendering |
| `breeds.py` | the twelve breed definitions |
| `posetoon_aline.py` | template blending and appearance parameters |
| `character_style.py` | builds the character's part geometry |
| `breed_markings.py` | breed markings and coat retinting |
| `run_one.py` / `run_all.py` | one clip / a folder of clips |
| `reencode_h264.py` | transcodes the renders for playback |

### tool notebook

Screening (`check_videos.py`, `check_camera_elevation.py`, `find_clean_windows.py`),
grading (`promote_grades.py`, `sort_by_grade.py`, `apply_grades.py`), batch
helpers (`fix_summary.py`, `make_reel.py`, `scan_cells.py`), diagnostics
(`diagnose_coat.py`, `audit_clips.py`, `breed_gallery.py`), and the evaluation
and ablation scripts (`evaluate_results.py`, `ablation_E1_alpha.py`,
`ablation_E2_template_selection.py`, `ablation_E3_E5.py`,
`ablation_E4_cartoon_vs_real.py`).

### results and data

`breed_templates.json` — the twelve templates.
`grades.txt` — every clip's grade and the reason where it was rejected.
`batch_summary.csv` — per-clip template, blend factor, coat colour and quality measurements.
`evaluation/` — the evaluation and ablation tables, figures and summaries.
`evaluation_notebook.py` — a marimo notebook that displays all of it.

---

## Usage

Python 3.9 with MMPose, PyTorch, OpenCV and marimo.

```bash
python run_one.py videos/your_clip.mp4     # one clip
python run_all.py videos/                  # a folder
```

The web interface:

```bash
pip install streamlit
streamlit run "Webpage code/app_streamlit.py"
```

The evaluation, interactively:

```bash
marimo edit "project pipeline main code/tool notebook/evaluation_notebook.py"
```

---

## Datasets

Three public datasets are used. None are committed here — together they run to
several gigabytes.

| Dataset | Used for | Original source |
|---|---|---|
| AP-10K | animal pose annotations; the pose estimator was pretrained on it | https://github.com/AlexTheBad/AP-10K |
| Stanford Dogs | breed photographs, used to build the twelve templates | http://vision.stanford.edu/aditya86/ImageNetDogs/ |
| SyDog-Video | synthetic dog video annotations | https://cvssp.org/data/SyDogVideo/ |

The copies used here are mirrored in one archive, in the directory layout the
scripts expect:

https://drive.google.com/file/d/1F_U4w2fFzI20pf9DMg88siP4BOTmntfv/view?usp=sharing

---

## Note on pretrained models

No model was trained here. Pose estimation uses a network pretrained on AP-10K;
breed appearance uses an ImageNet-pretrained classifier without retraining. The
classifier chooses appearance only; body proportion is measured from geometry.

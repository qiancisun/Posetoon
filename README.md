# PoseToon

A rigged 2D cartoon dog, animated from ordinary video of a real one.

MSc project, Bournemouth University.

---

## Overview

Given a short clip of a dog **in profile**, the pipeline:

1. tracks the animal's skeleton with a pose estimator
2. measures its bone proportions
3. picks one of twelve breed templates and deforms it **partway** toward those measurements
4. samples the coat colour from the footage
5. drives the character with the tracked motion

The output is a three-panel comparison: the source video, the tracked skeleton,
and the character, playing in step.

The character is not a copy of the animal. The blend factor sits at 0.35 for
most clips, so roughly a third of its proportions come from the dog in the
video and the rest from its breed template. "Partway" is the accurate word and
the results are reported that way.

### Scope

- no fur, muscle or three-dimensional volume
- footage shot from above cannot be handled at all — a flat rig has nowhere to
  put limbs that foreshorten toward the spine
- one dog, side on, in a continuous shot

---

## Repository structure

```
project pipeline main code/     the pipeline: seven stages plus the modules they import
  tool notebook/                screening, grading, evaluation and ablation scripts
  results and data/             breed templates, grades, evaluation tables and figures
Old version notebook/           46 superseded versions, kept as a record of the work
Webpage code/                   a Streamlit front end
Demo video/                     the delivered clips, by grade
```

`run_all.py` gives every clip its own working directory, so no clip can pick up
another's intermediate files.

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

Datasets are not included here — see [DATA.md](DATA.md) for where to get them
and what was derived from them.

---

Evaluation tables, figures and written summaries are in
`results and data/evaluation/`. An interactive view:

```bash
marimo edit "project pipeline main code/tool notebook/evaluation_notebook.py"
```

---

## Datasets

None of the datasets are committed here — together they run to several
gigabytes, more than a repository should carry. All three are public and can be
obtained from their original sources:

| Dataset | Original source |
|---|---|
| AP-10K | https://github.com/AlexTheBad/AP-10K |
| Stanford Dogs | http://vision.stanford.edu/aditya86/ImageNetDogs/ |
| SyDog-Video | https://github.com/MoritzKappel/SyDog-Video |

The copies actually used here are also mirrored in a single archive, in the
directory layout the scripts expect (`ap10k/`, `sydogvideo/`, and the Stanford
Dogs `Images/` folder):

**https://drive.google.com/file/d/1F_U4w2fFzI20pf9DMg88siP4BOTmntfv/view?usp=sharing**

What each was used for is set out below.

### AP-10K

Pose annotations for animals, used for two things: the pose estimator this
project runs was pretrained on it, and its ground-truth keypoints were used to
study how much dog body proportion actually varies.

- Official: https://github.com/AlexTheBad/AP-10K
- What was taken from it: 1,129 dog annotations, of which 269 survived removing
  those with no annotated spine points and applying a side-on view filter
- What that established: leg-to-spine ratio varies by a factor of 2.37 between
  the 90th and 10th percentile, so body proportion is a real signal
- Note: this study set the thresholds for an earlier three-tier size system,
  which the twelve breed templates later replaced. It is background to the work
  rather than the basis of the delivered system.

### Stanford Dogs

Breed photographs, used to build the twelve breed templates.

- Official: http://vision.stanford.edu/aditya86/ImageNetDogs/
- What was taken from it: 52-60 photographs per breed, for twelve breeds
- Derived artefact committed to this repository:
  `results and data/breed_templates.json` — per-breed bone measurements,
  appearance parameters, sample count and interquartile range

### SyDog-Video

Synthetic dog video annotations, downloaded for evaluation but not used in the
delivered results.

- Official: https://github.com/MoritzKappel/SyDog-Video

### Video footage

The clips in `Demo video/` are results, not sources. Source footage came from
Pexels and Pixabay under their free licences, plus clips recorded by the author.
Screening used `check_videos.py`: side-on view, one dog, at least 40 usable
frames, no cuts.

### Rebuilding the templates

With Stanford Dogs downloaded and its path set inside the script:

```bash
python 0_build_breed_templates.py
```

This is the only stage that needs a dataset; everything else runs from a video.

---

The fix for these is animation technique and anatomical constraint, not better
tracking: an anatomically jointed neck and head, a tail driven by body
acceleration, and easing applied where the tracked signal is least trustworthy.

---

## Note on pretrained models

No model was trained here. Pose estimation uses a network pretrained on AP-10K;
breed appearance uses an ImageNet-pretrained classifier without retraining. 

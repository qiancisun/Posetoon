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

## Key findings

**The automatic quality screen does not predict quality.** Most clips a human
rejected had been called clean by the pipeline's own check. Sharpness,
brightness and duplicate-frame rate describe *encoding*, while clips were
rejected for *camera geometry* and *what the animal was doing*. A sharpness
figure is blind to a dog filmed from above by construction.

**Averaging two estimators is not a decision.** Where the classifier and the
geometry disagree, the weighted sum of their scores can peak on a template
neither proposed. On one clip geometry chose a short-legged breed at 0.373, the
classifier a long-legged one at 0.596, and the combination selected a third at
0.489 — almost exactly their arithmetic mean. At low confidence the right
response is to refuse, not to compromise.

**The failures that cost most did not crash.** They completed and returned a
plausible wrong answer: a unit-conversion error that trimmed the wrong segment
without complaint; a criterion that correctly reasoned about when the geometric
estimator would be unreliable, printed its conclusion, and was consumed by
nothing.

---

## Limitations

The first two groups are properties of the input rather than of the
implementation. The third is not, and is the more interesting of them. The
thesis records all of them in full.

**Data and projection.** Monocular 2D tracking cannot recover footage shot from
above, or a dog turning toward the camera. AP-10K annotates only the neck and
the root of the tail along the torso, so spinal flexion is inferred rather than
measured and the tail has no keypoints at all. Limb *width* is never measured,
only bone length. Coat colour comes from a single sampled base, which suits a
solid-coloured dog and cannot express a tricolour hound or a merle collie.

**Evaluation.** Grades are one person's judgement on one viewing, with no second
rater and no blinding. E4 excluded 3 of 20 clips after inspecting the overlays —
one where the segmenter classified a white Samoyed as *sheep*, two where the
mask picked up a second animal or a patch of background. That exclusion rate is
reported as part of the result rather than tidied away.

### Animation problems, not tracking problems

Several of the defects a viewer notices are not caused by imprecise tracking,
and more tracking accuracy would not remove them. They are places where the
character does not follow how animation is normally constructed.

- **Head turning is not represented.** The head is a rigid part on a rigid neck,
  so a dog looking around reads as the whole body rotating.
- **The tail does not respond to the body.** It lags the body axis, so when the
  body bounces the tail does not answer it, and when the body is still the tail
  is still too. Real tail motion carries follow-through and overlapping action;
  neither is modelled.
- **Occlusion produces visible instability.** Where one limb crosses another the
  tracked confidence drops and the drawn limb wavers. An animator would hold or
  ease through such a passage rather than follow the data frame by frame.
- **Animation principles are largely absent** — anticipation, follow-through,
  easing and weight are not part of the retargeting, which maps tracked angles
  onto the rig directly.

The fix for these is animation technique and anatomical constraint, not better
tracking: an anatomically jointed neck and head, a tail driven by body
acceleration, and easing applied where the tracked signal is least trustworthy.

---

## Note on pretrained models

No model was trained here. Pose estimation uses a network pretrained on AP-10K;
breed appearance uses an ImageNet-pretrained classifier without retraining. Both
are used within their competence — the classifier chooses *appearance* and is
explicitly not trusted for *measurement*, which comes from geometry instead.
What is claimed is the system and the design decisions in it.

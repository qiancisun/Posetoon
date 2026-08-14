# E4 -- the character's outline against the real dog

The only measurement in this project with no human judgement in it.

## Method

Both masks are taken from the delivered three-panel videos: the character from the flat-background panel by thresholding, the animal from the source panel with DeepLabV3 (VOC labels, class 'dog'). Each is cropped to its bounding box and resampled to a common canvas without stretching, so the score compares shape and body proportion and ignores position and absolute size.

Nothing in the pipeline is imported or re-run. This is why the measurement was possible after all: the rendered character was already on disk.

## Result

- 19 clip(s) scored; median IoU **0.415** (range 0.282-0.623).
- That figure is unfiltered. Two of the 19 were excluded after inspecting the
  overlays: on `12438625` the segmenter merged two dogs into a single region,
  and on `16170693_720_1280_24fps_cut` a large background area was labelled as
  dog. In both the union is inflated by a mask that is wrong, not by a
  character that is wrong. **The thesis reports the remaining 17 clips: median
  IoU 0.437, range 0.302-0.623.** Both numbers are kept here because the
  exclusions were made by eye, not by a criterion the script applies, and that
  distinction should be visible rather than tidied away.
- Frames where segmentation returned no dog: **13**. Those are excluded rather than scored, and the exclusion rate is itself a result -- it says how often an off-the-shelf segmenter fails on this kind of footage.

## Read the overlays before quoting the number

`E4_overlays/` has one image per clip: red is the real dog, green is the character, yellow is agreement. An earlier attempt at this measurement produced IoU 0.02-0.03 across the board and those figures were a bug, not a result -- the renderer mirrors the canvas when the dog faces left and the alignment did not. Numbers from misaligned masks look like numbers.

## Limits

- Bounding-box normalisation removes size, so a character of the right shape at the wrong scale scores well here. Size is covered separately by the proportion measurements in E1.
- The segmentation is a general-purpose model applied to dark coats, backlighting and cluttered backgrounds. Where it half-succeeds the mask is wrong in ways the IoU cannot flag; this is what the overlays are for.
- Frames within a clip come from one continuous shot and are not independent samples.

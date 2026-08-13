# E1 -- the template/measurement blend

Replaces the 2025-07-29 E1, which searched for an optimal alpha on the removed size-tier system and correctly found none: alpha is a regularisation strength, not a quantity with an optimum. The useful question is what the SHIPPED setting actually preserves.

## E1.1 -- what the delivered characters kept

Reconstructed from each clip's own `character_description.json`: the measurement, the template it met, and the alpha applied. 42 clips.

- alpha as shipped: median **0.35** (range 0.35-0.60).
- Share of the template-to-measurement gap that the delivered character closes: median **35%**.

That number IS alpha by construction -- the blend is linear -- so it is a check that the record is consistent, not a discovery. What it makes concrete is the trade being made: at the shipped setting the character keeps about 35% of the difference between this dog and its breed, and gives up the rest to the template.
- How far apart the two ends are in the first place: median **0.017** (min 0.006, max 0.058) in the same units as the selector's distances.

Where that gap is small the alpha setting barely matters; where it is large, alpha is doing most of the work of deciding what the character looks like. `E1_1_delivered_blend.csv` has both per clip, with the grade alongside.

### alpha against outcome

- alpha 0.35: 37 clip(s), A-rate 20/37
- alpha 0.60: 5 clip(s), A-rate 0/5

These are counts, not rates: alpha is CHOSEN from clip length and selection confidence, so any association with the grade is the confounder, not an effect of alpha. Nothing causal can be read from this table and none is claimed.

## E1.2 -- alpha against shape

Measurements taken from the delivered clip `8056259-hd_1280_720_25fps` -- the one whose proportions sit farthest from its own template -- and swept against **Pekingese**, the template farthest from those measurements (gap 0.118). Both ends are real records; neither is invented, and the pair is the widest the delivered set offers so the knob has the most room to show an effect.

| alpha | IoU vs measured | IoU vs template |
|---|---|---|
| 0.00 | 0.6918 | 1.0000 |
| 0.20 | 0.7648 | 0.9000 |
| 0.35 | 0.8183 | 0.8414 |
| 0.50 | 0.8635 | 0.7966 |
| 0.60 | 0.8918 | 0.7721 |
| 0.80 | 0.9457 | 0.7296 |
| 1.00 | 1.0000 | 0.6918 |

Outline agreement with the measured end rises monotonically from **0.692** at alpha=0 to 1.000 at alpha=1, and the mirror column falls the same way, so the knob traverses the range rather than saturating at one end.

## Limits

- E1.1 is a reconstruction of a linear blend, so its central number is an identity. It is reported because the identity is what makes the trade legible, not because it was in doubt.
- alpha is not randomised. It is assigned by rule from clip length and selection confidence, so alpha and clip quality are confounded by design and no comparison across alpha values supports a causal reading.
- E1.2 measures the rest-pose outline for ONE clip against ONE template, chosen as the widest real pair available. It shows that the knob moves the shape; it does not say how much shape movement is desirable, and a narrower pair would move less.

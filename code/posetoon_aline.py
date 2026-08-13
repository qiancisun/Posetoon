import os
import json

from character_style import build_rig_styled

BREED_TEMPLATES = "outputs/breed_templates.json"
TIER_TEMPLATES = "outputs/dog_templates_v2.json"
SELECTION = "outputs/breed_selection.json"
PALETTE = "outputs/coat_palette.json"

TARGET_SPINE = 200.0
FIELDS = ("lower", "upper", "head", "hum", "rad", "fem", "tib")

MIN_FRAMES_RELIABLE = 40
ALPHA_LONG = 0.35
ALPHA_SHORT = 0.2

MIN_CLASSIFIER_CONFIDENCE = 0.30
SPINE_UNDERMEASURED = {"Pug": 0.89, "Pekingese": 0.94}

MARKING_SPREAD_LO = 0.24
MARKING_SPREAD_HI = 0.52

MAX_GAIT_BIAS = 0.20

COAT_MIN_CHROMA = 0.07
HEAD_ALPHA_SCALE = 0.35
ALPHA_UNSUPPORTED = 0.60
ALPHA_BOTH_WEAK = 0.45


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


DISCRIMINATIVE = ("head", "hum", "rad", "fem", "tib")


def proportion_distance(measure, template_measure, scale=None):
    if scale is None:
        return sum(abs(measure[f] - template_measure[f])
                   for f in DISCRIMINATIVE) / len(DISCRIMINATIVE) / TARGET_SPINE
    return sum(abs(measure[f] - template_measure[f]) / scale[f]
               for f in DISCRIMINATIVE) / len(DISCRIMINATIVE)


def field_scales(templates):
    scale = {}
    for f in DISCRIMINATIVE:
        vals = [e["measure"][f] for e in templates.values()]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        sd = var ** 0.5
        scale[f] = sd if sd > 1e-6 else float("inf")
    return scale


def nearest_template(measure, templates):
    scale = field_scales(templates) if len(templates) > 1 else None
    d = {n: proportion_distance(measure, e["measure"], scale)
         for n, e in templates.items()}
    best = min(d, key=d.get)
    ordered = sorted(d.values())
    margin = (ordered[1] - ordered[0]) if len(ordered) > 1 else float("inf")
    return best, d, margin


COLOUR_MODE = "blend"
RETINT = 0.75


def build_character_rig(measure, n_frames, force_template=None, force_alpha=None,
                        complexity="fine", use_breed_colours=True,
                        colour_mode=None, verbose=True):
    data = _load(BREED_TEMPLATES)
    template_kind = "breed"
    if data is None:
        data = _load(TIER_TEMPLATES)
        template_kind = "size-tier (fallback)"
    if data is None:
        raise FileNotFoundError(
            "No template file found. Run step_B1_breed_templates.py to build "
            "outputs/breed_templates.json.")

    templates = data["templates"]

    geo_pick, distances, margin = nearest_template(measure, templates)
    sel = _load(SELECTION)
    if force_template:
        name, source = force_template, "forced (ablation)"
    elif sel and sel.get("template") in templates:
        name, source = sel["template"], "classifier + geometry (step_B2)"
    else:
        name, source = geo_pick, "geometry only (no step_B2 result)"

    confidence = None
    if sel:
        for k in ("confidence", "score", "combined_score", "margin"):
            v = sel.get(k)
            if isinstance(v, (int, float)):
                confidence, confidence_key = float(v), k
                break
        if confidence is None and isinstance(sel.get("classifier"), dict):
            for k in ("confidence", "score", "prob"):
                v = sel["classifier"].get(k)
                if isinstance(v, (int, float)):
                    confidence, confidence_key = float(v), f"classifier.{k}"
                    break
    if confidence is None:
        confidence_key = None

    agreed = sel.get("agreement") if sel else None

    def _leg_ratio(tname):
        m = templates[tname]["measure"]
        return 0.5 * ((m["hum"] + m["rad"]) + (m["fem"] + m["tib"])) / TARGET_SPINE

    _override = False
    if sel and not force_template and name in templates and geo_pick in templates:
        _r_sel0, _r_geo0 = _leg_ratio(name), _leg_ratio(geo_pick)
        if _r_sel0 - _r_geo0 > MAX_GAIT_BIAS:
            if verbose:
                print(f"        OVERRIDE: the classifier picked {name} "
                      f"(leg-to-spine {_r_sel0:.3f}) but this dog measures "
                      f"nearer {_r_geo0:.3f} ({geo_pick}).")
                print(f"                  The standing-vs-running bias, "
                      f"quantified at about {MAX_GAIT_BIAS:.2f}, cannot account "
                      f"for a gap of {_r_sel0 - _r_geo0:.3f};")
                print(f"                  the classifier is wrong about size. "
                      f"Using {geo_pick}.")
            name = geo_pick
            source = (f"geometry -- the classifier's pick was inconsistent "
                      f"with this dog's measured proportions")
            _override = True

    if _override:
        supported = True
        support_why = ("classifier overridden on size; the geometric pick is "
                       "the one consistent with what was measured")
    elif force_template:
        supported, support_why = True, "forced -- not judged"
    elif not sel:
        supported = (geo_pick == name)
        support_why = "geometry only, no classifier to corroborate it"
    elif agreed is True:
        supported, support_why = True, "classifier and geometry agree"
    elif confidence is not None and confidence >= MIN_CLASSIFIER_CONFIDENCE:
        supported = True
        support_why = (f"classifier confident ({confidence:.2f} >= "
                       f"{MIN_CLASSIFIER_CONFIDENCE}) though geometry differs")
    else:
        _r_geo, _r_sel = _leg_ratio(geo_pick), _leg_ratio(name)

        if _r_sel - _r_geo > MAX_GAIT_BIAS:
            if verbose:
                print(f"        OVERRIDE: classifier picked {name} "
                      f"({_r_sel:.3f}) but this dog measures {_r_geo:.3f}-ish "
                      f"({geo_pick}).")
                print(f"                  A gap of {_r_sel - _r_geo:.3f} is far "
                      f"beyond the {MAX_GAIT_BIAS:.2f} the standing-vs-running "
                      f"bias can explain, so the")
                print(f"                  classifier is wrong about size. "
                      f"Using {geo_pick}.")
            name = geo_pick
            tmpl_source = f"geometry (classifier's {sel.get('template')} "
            tmpl_source += "was inconsistent with the measurement)"
            source = tmpl_source
            supported = True
            support_why = (f"classifier overridden: its pick was "
                           f"{_r_sel - _r_geo:.3f} longer-legged than measured, "
                           f"beyond any gait bias")
        elif _r_geo < _r_sel:
            supported = True
            support_why = (f"geometry disagrees but picks a shorter-legged "
                           f"template ({geo_pick} {_r_geo:.3f} vs {name} "
                           f"{_r_sel:.3f}) -- the direction the known "
                           f"standing-vs-running bias predicts, so it is not "
                           f"counted against the classifier")
        else:
            supported = False
            support_why = (f"classifier and geometry disagree, and geometry "
                           f"picks a LONGER-legged template ({geo_pick} "
                           f"{_r_geo:.3f} vs {name} {_r_sel:.3f}) which the "
                           f"gait bias cannot explain"
                           + (f"; confidence is only {confidence:.2f}"
                              if confidence is not None else
                              "; no confidence was recorded"))

    short = n_frames < MIN_FRAMES_RELIABLE
    if force_alpha is not None:
        alpha, alpha_why = float(force_alpha), "forced"
    elif supported and not short:
        alpha, alpha_why = ALPHA_LONG, f"{n_frames} frames, breed decision supported"
    elif supported and short:
        alpha, alpha_why = ALPHA_SHORT, (f"{n_frames} frames < {MIN_FRAMES_RELIABLE}"
                                          f", lean on the supported template")
    elif short:
        alpha, alpha_why = ALPHA_BOTH_WEAK, ("short clip AND unsupported breed "
                                              "decision -- neither source is reliable")
    else:
        alpha, alpha_why = ALPHA_UNSUPPORTED, ("breed decision not corroborated -- "
                                                "leaning back on this dog's own "
                                                "measurements")

    entry = templates[name]

    if name in SPINE_UNDERMEASURED:
        _k = SPINE_UNDERMEASURED[name]
        entry = {**entry, "measure": {**entry["measure"]}}
        for _f in ("hum", "rad", "fem", "tib"):
            entry["measure"][_f] *= _k
        if verbose:
            _m = entry["measure"]
            _r = 0.5 * ((_m["hum"] + _m["rad"])
                        + (_m["fem"] + _m["tib"])) / TARGET_SPINE
            print(f"        {name} template limbs x{_k:.2f} -- a curled tail "
                  f"and short neck under-measure the spine, which inflates "
                  f"every ratio built on it (leg-to-spine now {_r:.3f})")

    alpha_head = alpha * HEAD_ALPHA_SCALE
    blended = {k: (1 - (alpha_head if k == "head" else alpha))
               * entry["measure"][k]
               + (alpha_head if k == "head" else alpha) * measure[k]
               for k in entry["measure"]}
    _front = blended["hum"] + blended["rad"]
    _hind = blended["fem"] + blended["tib"]
    if _front > 1e-6 and _hind > 1e-6:
        _target = 0.5 * (_front + _hind)
        _kf, _kh = _target / _front, _target / _hind
        blended["hum"] *= _kf
        blended["rad"] *= _kf
        blended["fem"] *= _kh
        blended["tib"] *= _kh
        if verbose and abs(_front - _hind) / _target > 0.04:
            print(f"        levelled front/hind: {_front:.1f}/{_hind:.1f}px "
                  f"-> {_target:.1f}/{_target:.1f} "
                  f"(17-keypoint chains are not anatomically matched)")

    appearance = dict(entry.get("appearance", {}))

    try:
        import breeds as _breeds_mod
        _live = _breeds_mod.appearance_of(name)
        if _live:
            _added = [k for k, v in _live.items()
                      if appearance.get(k) is None]
            for k in _added:
                appearance[k] = _live[k]
            if _added and verbose:
                print(f"        appearance fields taken from breeds.py "
                      f"(absent from breed_templates.json): "
                      f"{', '.join(sorted(_added))}")
    except ImportError:
        pass
    if not appearance:
        raise ValueError(
            f"Template '{name}' has no appearance parameters. step_B1 fills "
            f"these from the breed table; check that "
            f"step_A5_appearance_params.py was importable when it ran.")

    mode = colour_mode or COLOUR_MODE
    if not use_breed_colours:
        mode = "sampled"
    sampled = _load(PALETTE)
    sampled_hex = sampled.get("base_hex") if sampled else None

    _marking_strength = 1.0
    _meas = (sampled or {}).get("measured") or {}
    _parts = [v for v in _meas.values()
              if isinstance(v, str) and v.startswith("#")]
    if len(_parts) >= 3:
        def _lum(hx):
            h = hx.lstrip("#")
            try:
                r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            except ValueError:
                return None
            return 0.5 * (max(r, g, b) + min(r, g, b))
        _ls = [x for x in (_lum(v) for v in _parts) if x is not None]
        if len(_ls) >= 3:
            _spread = max(_ls) - min(_ls)
            _marking_strength = float(min(1.0, max(0.0,
                (_spread - MARKING_SPREAD_LO)
                / (MARKING_SPREAD_HI - MARKING_SPREAD_LO))))
            if verbose:
                print(f"        coat spread across {len(_ls)} sampled parts: "
                      f"{_spread:.2f} -> markings at "
                      f"{_marking_strength * 100:.0f}% strength"
                      + ("  (this dog measures solid, so the template's "
                         "markings are suppressed)" if _marking_strength < 0.35
                         else ""))

    if sampled_hex:
        _h = sampled_hex.lstrip("#")
        try:
            _r, _g, _b = (int(_h[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            _r = _g = _b = 0
        _chroma = (max(_r, _g, _b) - min(_r, _g, _b)) / 255.0
        _mx, _mn = max(_r, _g, _b) / 255.0, min(_r, _g, _b) / 255.0
        if _chroma > 1e-6:
            _rn, _gn, _bn = _r / 255.0, _g / 255.0, _b / 255.0
            if _mx == _rn:
                _hue = (60 * ((_gn - _bn) / (_mx - _mn))) % 360
            elif _mx == _gn:
                _hue = 60 * ((_bn - _rn) / (_mx - _mn)) + 120
            else:
                _hue = 60 * ((_rn - _gn) / (_mx - _mn)) + 240
        else:
            _hue = 0.0
        _lum = 0.5 * (_mx + _mn)
        if (_chroma >= COAT_MIN_CHROMA
                and 0.12 < _lum < 0.88
                and not (_hue <= 60.0 or _hue >= 340.0)):
            print(f"        NOTE: sampled coat {sampled_hex} sits at hue "
                  f"{_hue:.0f} deg (chroma {_chroma:.2f}), outside the range "
                  f"real coats occupy.")
            print(f"              Keeping its LIGHTNESS, which is what such a "
                  f"sample reports well; the hue is dropped downstream by "
                  f"breed_markings.")

    palette, colour_src = None, "default greys"
    if mode in ("breed", "blend"):
        try:
            from breed_markings import breed_palette, BREED_COLOURS
            known = False
            try:
                from breeds import resolve as _resolve_breed
                known = _resolve_breed(name) is not None
            except ImportError:
                pass
            if known or name in BREED_COLOURS:
                if mode == "blend" and sampled_hex:
                    palette = breed_palette(name, sampled_hex=sampled_hex,
                                            use_sampled=True, retint=RETINT,
                                            lightness_match=1.0,
                                            marking_strength=_marking_strength)
                    colour_src = (f"{name} markings retinted toward the "
                                  f"measured coat {sampled_hex}")
                else:
                    palette = breed_palette(name)
                    colour_src = f"breed-typical markings ({name})"
                    if mode == "blend":
                        colour_src += "  (no coat_palette.json to retint with)"
            else:
                colour_src = f"no markings defined for '{name}'"
        except ImportError:
            colour_src = "breed_markings.py not importable"
    if palette is None:
        _why = colour_src if colour_src.startswith("no markings") \
            or colour_src.startswith("breed_markings") else None
        if sampled:
            palette = sampled["palette"]
            colour_src = "sampled from video" + (f"  ({_why})" if _why else "")

    rig = build_rig_styled(blended, appearance, tier=name, ref=TARGET_SPINE,
                           complexity=complexity, palette=palette)

    hind_used = blended["fem"] + blended["tib"]
    hind_measured = measure["fem"] + measure["tib"]
    root_lift = max(0.0, hind_used - hind_measured)

    info = {
        "template": name, "template_kind": template_kind,
        "selection_source": source, "geometry_pick": geo_pick,
        "geometry_distances": distances, "geometry_margin": margin,
        "classifier_agreed": agreed,
        "selection_confidence": confidence,
        "selection_confidence_key": confidence_key,
        "selection_supported": supported,
        "selection_support_reason": support_why,
        "alpha": alpha, "alpha_head": round(alpha_head, 3),
        "alpha_reason": alpha_why,
        "complexity": complexity, "colour_source": colour_src,
        "appearance": appearance,
        "root_lift_px": round(root_lift, 1),
        "measure_used": {k: round(blended[k], 1) for k in FIELDS},
        "template_measure": {k: round(entry["measure"][k], 1) for k in FIELDS},
    }

    if verbose:
        print(f"A-line: template = {name}   [{template_kind}]")
        print(f"        selected by {source}"
              + (f"   (geometry alone would pick {geo_pick})"
                 if geo_pick != name and not force_template else ""))
        if not supported and not force_template:
            print(f"        WARNING: {support_why}.")
            print(f"                 The template is a guess; alpha raised to "
                  f"{alpha} so the character follows this dog's own "
                  f"proportions more than the breed's.")
        elif sel and agreed is False and not force_template:
            print(f"        NOTE: {support_why}")
        if margin < 0.008:
            print(f"        NOTE: geometry margin {margin:.4f} is narrow; this "
                  f"dog sits between templates")
        print(f"        alpha={alpha} (head {alpha_head:.2f})  [{alpha_why}]"
              f"   complexity={complexity}")
        print(f"        ear={appearance['ear_type']}  "
              f"neck={rig['neck']['length']:.0f}px  "
              f"head={rig['head']['length']:.0f}px")
        print(f"        colour: {colour_src}")
        if root_lift > 1.0:
            print(f"        root lift {root_lift:.0f}px  (legs are "
                  f"{hind_used - hind_measured:+.0f}px vs measured; without this "
                  f"the feet would sink)")

    return rig, info


if __name__ == "__main__":
    demo = {"lower": 93.5, "upper": 93.5, "head": 93.2,
            "hum": 47.0, "rad": 34.6, "fem": 48.5, "tib": 44.4}
    print("--- 279 frames, breed colours ---")
    build_character_rig(demo, n_frames=279)
    print("\n--- 20 frames (alpha should drop) ---")
    build_character_rig(demo, n_frames=20)
    print("\n--- sampled coat colour instead of breed colours ---")
    build_character_rig(demo, n_frames=279, use_breed_colours=False)

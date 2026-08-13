import colorsys

BREED_MARKINGS = {
    "German Shepherd": {
        "body": "dark", "neck": "dark", "tail": "dark",
        "head": "base", "muzzle": "dark", "ear_upper": "base", "ear_lower": "dark",
        "near_upper": "base", "near_mid": "base", "near_lower": "base",
        "far_upper": "base", "far_mid": "base", "far_lower": "base",
    },
    "Pekingese": {
        "body": "base", "neck": "base", "tail": "base",
        "head": "base", "muzzle": "dark", "ear_upper": "dark", "ear_lower": "dark",
        "near_upper": "base", "near_mid": "base", "near_lower": "light",
        "far_upper": "base", "far_mid": "base", "far_lower": "light",
    },
    "Beagle": {
        "body": "dark", "neck": "base", "tail": "light",
        "head": "base", "muzzle": "light", "ear_upper": "base", "ear_lower": "base",
        "near_upper": "light", "near_mid": "light", "near_lower": "light",
        "far_upper": "light", "far_mid": "light", "far_lower": "light",
    },
    "Basset Hound": {
        "body": "dark", "neck": "base", "tail": "light",
        "head": "base", "muzzle": "light", "ear_upper": "base", "ear_lower": "base",
        "near_upper": "light", "near_mid": "light", "near_lower": "light",
        "far_upper": "light", "far_mid": "light", "far_lower": "light",
    },
    "Cardigan Corgi": {
        "body": "base", "neck": "light", "tail": "base",
        "head": "base", "muzzle": "light", "ear_upper": "base", "ear_lower": "base",
        "near_upper": "light", "near_mid": "light", "near_lower": "light",
        "far_upper": "light", "far_mid": "light", "far_lower": "light",
    },
    "Pug": {
        "body": "base", "neck": "base", "tail": "base",
        "head": "base", "muzzle": "dark", "ear_upper": "dark", "ear_lower": "dark",
        "near_upper": "base", "near_mid": "base", "near_lower": "base",
        "far_upper": "base", "far_mid": "base", "far_lower": "base",
    },
    "Great Dane": {
        "body": "base", "neck": "base", "tail": "base",
        "head": "base", "muzzle": "dark", "ear_upper": "dark", "ear_lower": "dark",
        "near_upper": "base", "near_mid": "base", "near_lower": "base",
        "far_upper": "base", "far_mid": "base", "far_lower": "base",
    },
    "Whippet": {
        "body": "base", "neck": "base", "tail": "base",
        "head": "base", "muzzle": "light", "ear_upper": "base", "ear_lower": "base",
        "near_upper": "base", "near_mid": "base", "near_lower": "light",
        "far_upper": "base", "far_mid": "base", "far_lower": "light",
    },
    "Labrador": {},
}

BREED_COLOURS = {
    "German Shepherd": {"base": "#B0793C", "dark": "#2A2320", "light": "#D6BE9C"},
    "Pekingese":       {"base": "#D8B87A", "dark": "#2B241D", "light": "#EFE2C6"},
    "Beagle":          {"base": "#C08A4A", "dark": "#2E2A26", "light": "#EFE8DC"},
    "Basset Hound":    {"base": "#C48B4E", "dark": "#33291F", "light": "#F0EAE0"},
    "Cardigan Corgi":  {"base": "#B87A3E", "dark": "#3A2F26", "light": "#F2EDE4"},
    "Pug":             {"base": "#D9BC85", "dark": "#241F1B", "light": "#E9DCC0"},
    "Great Dane":      {"base": "#C9A268", "dark": "#2B2622", "light": "#DCC9A6"},
    "Whippet":         {"base": "#C9BBA8", "dark": "#4A4239", "light": "#F0EBE3"},
    "Labrador":        {"base": "#22201E", "dark": "#141312", "light": "#3C3835"},
}

PALETTE_PARTS = ["body", "neck", "head", "muzzle", "ear_upper", "ear_lower",
                 "tail", "near_upper", "near_mid", "near_lower",
                 "far_upper", "far_mid", "far_lower"]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*[max(0, min(255, int(round(v)))) for v in rgb])


def _hls(hex_colour):
    r, g, b = [v / 255 for v in _hex_to_rgb(hex_colour)]
    return colorsys.rgb_to_hls(r, g, b)


def _retint_set(colours, target_hex, amount, body_role="base", l_amount=None,
                used_roles=None):
    th, tl, ts = _hls(target_hex)

    _rgb_t = _hex_to_rgb(target_hex)
    _chroma = (max(_rgb_t) - min(_rgb_t)) / 255.0
    _hue_conf = min(1.0, max(0.0, (_chroma - 0.08) / (0.25 - 0.08)))

    _sat_damp = 1.0
    if _chroma > COAT_MAX_CHROMA:
        _sat_damp = COAT_MAX_CHROMA / _chroma

    _hue_deg = _hls(target_hex)[0] * 360.0
    _dist = 0.0 if (_hue_deg <= 60.0 or _hue_deg >= 340.0) else min(
        abs(_hue_deg - 60.0), abs(_hue_deg - 340.0))
    _band_conf = max(0.0, 1.0 - _dist / 25.0)

    amount = amount * _hue_conf * _band_conf * _sat_damp

    OTHER_HUE_SHARE = 0.25

    anchor_role = body_role if body_role in colours else "base"
    if anchor_role not in colours:
        anchor_role = next(iter(colours))
    ah, al, asat = _hls(colours[anchor_role])

    l_amt = amount if l_amount is None else float(l_amount)
    d_l = (tl - al) * l_amt

    _use = set(used_roles or colours.keys()) | {body_role}
    _ls = [_hls(v)[1] for k, v in colours.items() if k in _use] or \
          [_hls(v)[1] for v in colours.values()]
    _lo, _hi = min(_ls), max(_ls)
    _anchor_new = al + d_l
    _up_room = ROLE_L_CEIL - _anchor_new
    _dn_room = _anchor_new - ROLE_L_FLOOR
    _up_need = _hi - al
    _dn_need = al - _lo
    _up_scale = min(1.0, _up_room / _up_need) if _up_need > 1e-9 else 1.0
    _dn_scale = min(1.0, _dn_room / _dn_need) if _dn_need > 1e-9 else 1.0
    _compressed = (_up_scale < 0.999) or (_dn_scale < 0.999)

    def _place(l):
        d = l - al
        return _anchor_new + d * (_up_scale if d >= 0 else _dn_scale)

    out = {}
    for role, hexv in colours.items():
        h, l, s = _hls(hexv)
        _dh = ((th - h) + 0.5) % 1.0 - 0.5

        _far = abs(_dh) > HUE_SNAP_ARC and amount >= HUE_SNAP_MIN
        if role == anchor_role:
            h2 = th if _far else (h + _dh * amount) % 1.0
            s2 = s + (ts - s) * amount
            l2 = _anchor_new
        else:
            h2 = th if _far else (h + _dh * amount * OTHER_HUE_SHARE) % 1.0
            s2 = s + (ts - s) * amount * OTHER_HUE_SHARE
            l2 = _place(l)
        out[role] = _rgb_to_hex([c * 255 for c in
                                 colorsys.hls_to_rgb(h2, max(ROLE_L_FLOOR, min(ROLE_L_CEIL, l2)), s2)])
    return out


CANVAS_BG = "#F0F0F0"

MIN_BG_GAP = 0.28

COAT_MAX_CHROMA = 0.32

HUE_SNAP_ARC = 0.28

HUE_SNAP_MIN = 0.20

DARK_LINE_FLOOR = 0.06


def _set_lightness(hex_colour, l_target):
    h, l, s = _hls(hex_colour)
    return _rgb_to_hex([c * 255 for c in
                        colorsys.hls_to_rgb(h, max(0.0, min(1.0, l_target)), s)])


def _mix(a_hex, b_hex, t):
    a, b = _hex_to_rgb(a_hex), _hex_to_rgb(b_hex)
    return _rgb_to_hex([a[i] + (b[i] - a[i]) * t for i in range(3)])


def _shift(hex_colour, dl):
    r, g, b = [v / 255 for v in _hex_to_rgb(hex_colour)]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    target = l + dl
    if target < 0.06 or target > 0.94:
        target = l - dl
    l2 = max(0.06, min(0.94, target))
    return _rgb_to_hex([c * 255 for c in colorsys.hls_to_rgb(h, l2, s)])


def _registry(breed):
    try:
        from breeds import resolve, markings_of, colours_of
        name = resolve(breed)
        if name:
            return name, markings_of(name), colours_of(name)
    except ImportError:
        pass
    return breed, None, None


ROLE_L_FLOOR = 0.03
ROLE_L_CEIL = 0.97


def breed_palette(breed, sampled_hex=None, use_sampled=False, retint=0.75,
                  bg_hex=CANVAS_BG, lightness_match=None,
                  marking_strength=1.0):
    breed, reg_mark, reg_col = _registry(breed)
    colours = dict(reg_col or BREED_COLOURS.get(breed, {"base": "#4A4A4A",
                                                         "dark": "#262626",
                                                         "light": "#7A7A7A"}))
    markings_for_role = reg_mark if reg_mark is not None else BREED_MARKINGS.get(breed, {})
    if use_sampled and sampled_hex:
        colours = _retint_set(colours, sampled_hex, retint,
                              body_role=markings_for_role.get("body", "base"),
                              l_amount=lightness_match,
                              used_roles=set(markings_for_role.values()))

    markings = reg_mark if reg_mark is not None else BREED_MARKINGS.get(breed, {})

    markings = dict(markings)
    if "ear_lower" not in markings and "ear_upper" in markings:
        markings["ear_lower"] = markings["ear_upper"]
    elif "ear_upper" not in markings and "ear_lower" in markings:
        markings["ear_upper"] = markings["ear_lower"]

    _ms = max(0.0, min(1.0, float(marking_strength)))
    palette = {}
    _anchor_key = markings.get("body", "base")
    for part in PALETTE_PARTS:
        _c = colours[markings.get(part, "base")]
        if _ms < 0.999:
            _c = _mix(colours.get(_anchor_key, _c), _c, _ms)
        palette[part] = _c

    if palette["ear_upper"] == palette["head"]:
        _hl = _hls(palette["head"])[1]
        _dl = -0.10 if _hl > 0.30 else 0.12
        for _part in ("ear_upper", "ear_lower"):
            palette[_part] = _set_lightness(palette[_part],
                                            max(0.03, min(0.97, _hl + _dl)))

    bg_l = _hls(bg_hex)[1]

    _near_l = {p: _hls(h)[1] for p, h in palette.items()}
    for _p in ("far_upper", "far_mid", "far_lower"):
        _l = _near_l[_p]
        _step = 0.13 if _l < 0.55 else -0.13
        palette[_p] = _set_lightness(palette[_p],
                                     max(0.02, min(0.98, _l + _step)))

    lightness = {p: colorsys.rgb_to_hls(*[v / 255 for v in _hex_to_rgb(h)])[1]
                 for p, h in palette.items()}
    darkest = min(palette, key=lambda p: lightness[p])
    lightest = max(palette, key=lambda p: lightness[p])

    _main = [lightness[k] for k in ("body", "neck", "head",
                                    "near_upper", "near_lower")
             if k in lightness]
    _main.sort()
    _bulk = _main[len(_main) // 2] if _main else lightness[darkest]

    _dark_line = lightness[darkest] - 0.14
    if _bulk >= 0.28:
        l_out = max(0.04, min(_dark_line, bg_l - MIN_BG_GAP))
        palette["outline"] = _set_lightness(palette[darkest], l_out)
        palette["outline_dark"] = _set_lightness(palette[darkest],
                                                 max(0.03, l_out - 0.05))
    else:
        l_out = min(0.80, lightness[lightest] + 0.22)
        palette["outline"] = _set_lightness(palette[lightest], l_out)
        palette["outline_dark"] = _set_lightness(palette[lightest],
                                                 max(0.06, l_out - 0.08))
    return palette


def legibility_report(palette, bg_hex=CANVAS_BG):
    bg_l = _hls(bg_hex)[1]
    return {p: (h, round(abs(_hls(h)[1] - bg_l), 3)) for p, h in palette.items()}


if __name__ == "__main__":
    print(f"{'breed':17} {'body':9} {'head':9} {'muzzle':9} {'legs':9} {'ear':9}")
    print("-" * 70)
    for b in BREED_COLOURS:
        p = breed_palette(b)
        print(f"{b:17} {p['body']:9} {p['head']:9} {p['muzzle']:9} "
              f"{p['near_lower']:9} {p['ear_lower']:9}")

    print("\nSame breed, retinted toward a black coat sampled from video "
          "(#1B2424):")
    for b in ["German Shepherd", "Beagle", "Pekingese"]:
        p = breed_palette(b, sampled_hex="#1B2424", use_sampled=True)
        print(f"  {b:17} body={p['body']}  head={p['head']}  "
              f"legs={p['near_lower']}   <- pattern kept, hue moved")

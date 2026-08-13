import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import json
    import cv2
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    KEYPOINT_NAMES = [
        'L_Eye', 'R_Eye', 'Nose', 'Neck', 'root_of_tail',
        'L_Shoulder', 'L_Elbow', 'L_F_Paw',
        'R_Shoulder', 'R_Elbow', 'R_F_Paw',
        'L_Hip', 'L_Knee', 'L_B_Paw',
        'R_Hip', 'R_Knee', 'R_B_Paw',
    ]
    KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

    with open('stabilised_keypoints.json', 'r') as f:
        stab_kps = json.load(f)
    with open('full_video_keypoints.json', 'r') as f:
        raw_kps = json.load(f)

    def get_kp(frame_data, name):
        return np.array(frame_data['keypoints'][KP_INDEX[name]], dtype=float)

    def get_score(frame_data, name):
        return frame_data['scores'][KP_INDEX[name]]

    with open('spine_tail_tracks.json', 'r') as f:
        spine_tail = json.load(f)
    spine_mid_track = np.array(spine_tail['spine_mid'], dtype=float)
    tail_tip_track = np.array(spine_tail['tail_tip'], dtype=float)

    print(f"Loaded {len(stab_kps)} stabilised frames, {len(raw_kps)} raw frames, "
          f"spine/tail tracks {len(spine_mid_track)} frames")
    return (
        Image,
        ImageDraw,
        KEYPOINT_NAMES,
        cv2,
        get_kp,
        get_score,
        np,
        plt,
        raw_kps,
        spine_mid_track,
        stab_kps,
        tail_tip_track,
    )


@app.cell
def _(get_kp, np, stab_kps):
    def measure_skeleton(stab_list, target_spine=200.0):
        acc = {k: [] for k in ['lower', 'upper', 'head', 'hum', 'rad', 'fem', 'tib']}
        for fr in stab_list:
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            sl = np.linalg.norm(neck - tail)
            if sl < 1e-6:
                continue
            sc = target_spine / sl
            hips = (get_kp(fr, 'L_Hip') + get_kp(fr, 'R_Hip')) / 2
            pelvis = tail * 0.55 + hips * 0.45
            waist = (pelvis + neck) / 2
            acc['lower'].append(np.linalg.norm(waist - pelvis) * sc)
            acc['upper'].append(np.linalg.norm(neck - waist) * sc)
            acc['head'].append(np.linalg.norm(get_kp(fr, 'Nose') - neck) * sc)
            acc['hum'].append(np.linalg.norm(get_kp(fr, 'L_Elbow') - get_kp(fr, 'L_Shoulder')) * sc)
            acc['rad'].append(np.linalg.norm(get_kp(fr, 'L_F_Paw') - get_kp(fr, 'L_Elbow')) * sc)
            acc['fem'].append(np.linalg.norm(get_kp(fr, 'L_Knee') - get_kp(fr, 'L_Hip')) * sc)
            acc['tib'].append(np.linalg.norm(get_kp(fr, 'L_B_Paw') - get_kp(fr, 'L_Knee')) * sc)
        m = {}
        for k in ['lower', 'upper', 'head']:
            m[k] = float(np.median(acc[k]))
        for k in ['hum', 'rad', 'fem', 'tib']:
            m[k] = float(np.percentile(acc[k], 75))
        print("measured proportions (spine = 200):")
        for k in ['lower', 'upper', 'head', 'hum', 'rad', 'fem', 'tib']:
            print(f"  {k:6}: {m[k]:6.1f}px")
        return m

    measure = measure_skeleton(stab_kps)
    return (measure,)


@app.cell
def _(measure, np, stab_kps):
    def build_rig(measure, ref=200.0):
        LIMB_ROOT_SPREAD = 0.022
        L_low, L_up = measure['lower'], measure['upper']
        L_spine_total = L_low + L_up
        L_seg = (L_spine_total / 4.0) * 0.82
        L_neck = ref * 0.13
        L_head = float(np.clip(measure['head'] - L_neck, ref * 0.18, ref * 0.24))
        L_hum, L_rad = measure['hum'], measure['rad']
        L_fem, L_tib = measure['fem'], measure['tib']
        L_paw_f, L_paw_h = ref * 0.052, ref * 0.056
        L_t1, L_t2 = ref * 0.17, ref * 0.14
        L_e1, L_e2 = ref * 0.10, ref * 0.11

        def taper(L, w0, w1, back=0.0):
            return [(-back, -w0), (L, -w1), (L, w1), (-back, w0)]

        def paw_shape(L, w):
            return [(-L*0.34, -w*0.95), (L*0.70, -w*0.85), (L, -w*0.45),
                    (L, w*0.55), (L*0.55, w*0.95), (-L*0.34, w*0.90)]

        rig = {}

        def bone(name, parent, offset, length, mesh, color, outline=None, layer=0):
            rig[name] = {'parent': parent, 'offset': np.array(offset, dtype=float),
                         'length': length, 'mesh': mesh, 'color': color,
                         'outline': outline, 'layer': layer}

        bone('root', None, (0, 0), 0.0, None, None, layer=0)

        for i in range(1, 5):
            parent = 'root' if i == 1 else f'spine{i-1}'
            off = (0, 0) if i == 1 else (L_seg, 0)
            bone(f'spine{i}', parent, off, L_seg, None, color='#3D3D3D', layer=30 + i)

        bone('neck', 'spine4', (L_seg, 0), L_neck,
             mesh=[(-ref*0.03, -ref*0.082), (L_neck, -ref*0.090),
                   (L_neck, ref*0.078), (-ref*0.03, ref*0.074)],
             color='#424242', layer=36)
        bone('head', 'neck', (L_neck, 0), L_head,
             mesh=[(-ref*0.02, -L_head*0.48), (L_head*0.30, -L_head*0.58),
                   (L_head*0.70, -L_head*0.54), (L_head*0.98, -L_head*0.32),
                   (L_head*1.14, -L_head*0.02), (L_head*1.08, L_head*0.24),
                   (L_head*0.66, L_head*0.44), (L_head*0.26, L_head*0.40),
                   (-ref*0.02, L_head*0.36)],
             color='#4A4A4A', outline='#222222', layer=37)

        bone('ear_upper', 'head', (L_head*0.30, -L_head*0.36), L_e1,
             taper(L_e1, ref*0.030, ref*0.036), color='#363636', outline='#222222', layer=38)
        bone('ear_lower', 'ear_upper', (L_e1, 0), L_e2,
             taper(L_e2, ref*0.036, ref*0.020, back=ref*0.010), color='#2E2E2E', outline='#1A1A1A', layer=39)


        TAIL_ROOT_BACK = 0.03
        bone('tail_1', 'spine1', (-ref*TAIL_ROOT_BACK, ref*0.02), L_t1,
                 taper(L_t1, ref*0.028, ref*0.020, back=ref*0.075),
                 color='#2A2A2A', layer=20)

        bone('tail_2', 'tail_1', (L_t1, 0), L_t2,
             taper(L_t2, ref*0.020, ref*0.010, back=ref*0.010), color='#2A2A2A', layer=21)

        bone('shoulder', 'spine4', (0, 0), 0.0, None, None, layer=0)
        bone('hip', 'spine1', (0, 0), 0.0, None, None, layer=0)

        for side, base, cu, cm, cl in [('L', 40, '#444444', '#3C3C3C', '#2A2A2A'),
                                       ('R', 10, '#666666', '#5E5E5E', '#4E4E4E')]:
            _lat = ref * LIMB_ROOT_SPREAD * (1.0 if side == 'L' else -1.0)
            bone(f'humerus_{side}', 'shoulder', (_lat, _lat * 0.35), L_hum,
                 taper(L_hum, ref*0.044, ref*0.028, back=ref*0.045), color=cu, layer=base+1)
            bone(f'radius_{side}', f'humerus_{side}', (L_hum, 0), L_rad,
                 taper(L_rad, ref*0.028, ref*0.018, back=ref*0.026), color=cm, layer=base+2)
            bone(f'paw_front_{side}', f'radius_{side}', (L_rad, 0), L_paw_f,
                 paw_shape(L_paw_f, ref*0.021), color=cl, outline='#1A1A1A', layer=base+3)
            bone(f'femur_{side}', 'hip', (_lat, _lat * 0.35), L_fem,
                 taper(L_fem, ref*0.048, ref*0.029, back=ref*0.045), color=cu, layer=base+4)
            bone(f'tibia_{side}', f'femur_{side}', (L_fem, 0), L_tib,
                 taper(L_tib, ref*0.029, ref*0.018, back=ref*0.027), color=cm, layer=base+5)
            bone(f'paw_hind_{side}', f'tibia_{side}', (L_tib, 0), L_paw_h,
                 paw_shape(L_paw_h, ref*0.021), color=cl, outline='#1A1A1A', layer=base+6)

        rig['_spine_meta'] = {'L_seg': L_seg,
                              'w_up': ref * 0.125, 'w_dn': ref * 0.150,
                              'w_up_end': ref * 0.075, 'w_dn_end': ref * 0.090}
        return rig

    FORCE_TEMPLATE = None
    FORCE_ALPHA = None
    COMPLEXITY = 'fine'
    USE_BREED_COLOURS = True

    from posetoon_aline import build_character_rig

    rig, aline_info = build_character_rig(
        measure,
        n_frames=len(stab_kps),
        force_template=FORCE_TEMPLATE,
        force_alpha=FORCE_ALPHA,
        complexity=COMPLEXITY,
        use_breed_colours=USE_BREED_COLOURS,
    )

    _ap = aline_info['appearance']
    _EAR_WORDS = {'floppy': 'drop ears', 'semi_erect': 'semi-erect ears',
                  'erect': 'erect ears'}
    _curl = ('a tightly curled tail' if _ap['tail_curl'] > 0.6 else
             'a gently curved tail' if _ap['tail_curl'] > 0.25 else
             'a straight tail')
    CHARACTER_DESCRIPTION = (
        f"{aline_info['template']}-type dog "
        f"(template chosen by {aline_info['selection_source']}); "
        f"{_EAR_WORDS[_ap['ear_type']]} and {_curl}; "
        f"{_ap['coat']} coat; colour: {aline_info['colour_source']}."
    )
    print("\nCharacter description:")
    print(f"  {CHARACTER_DESCRIPTION}")
    print(f"  template={aline_info['template']} "
          f"alpha={aline_info.get('alpha')} "
          f"root_lift={aline_info.get('root_lift_px'):.1f}px")
    _mb = aline_info.get('blended', aline_info.get('measure'))
    if isinstance(_mb, dict):
        print("  blended bone lengths: "
              + "  ".join(f"{k}={float(v):.0f}" for k, v in _mb.items()))

    import json as _j
    with open('outputs/character_description.json', 'w') as _f:
        _j.dump({'description': CHARACTER_DESCRIPTION, **aline_info,
                 'spine_tail_source': 'annotated'}, _f, indent=2, default=str)
    print(f"Rig built: {len([k for k in rig if not k.startswith('_')])} bones "
          f"(4-segment spine, dynamic body ribbon)")
    return aline_info, rig


@app.cell
def _(get_kp, np, stab_kps):
    def compute_root_motion(stab_list, target_spine=200.0, detrend_win=145, bounce_gain=0.6):
        pel = []
        for fr in stab_list:
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            sl = np.linalg.norm(tail - neck)
            if sl < 1e-6:
                pel.append(pel[-1] if pel else np.zeros(2)); continue
            sc = target_spine / sl
            hips = (get_kp(fr, 'L_Hip') + get_kp(fr, 'R_Hip')) / 2
            pel.append((tail * 0.55 + hips * 0.45) * sc)
        pel = np.array(pel)
        half = detrend_win // 2
        base = np.convolve(np.pad(pel[:, 1], (half, half), mode='edge'),
                           np.ones(detrend_win)/detrend_win, 'valid')[:len(pel)]
        y_gait = (pel[:, 1] - base) * bounce_gain
        _cap = target_spine * 0.09
        _n_clip = int(np.sum(np.abs(y_gait) > _cap))
        y_gait = np.clip(y_gait, -_cap, _cap)
        if _n_clip:
            print(f"  bounce clipped on {_n_clip} frames at +-{_cap:.0f}px "
                  f"(slow drift, not gait)")
        print(f"gait bounce range : {np.ptp(y_gait):.1f}px "
              f"(detrend window {detrend_win} frames vs ~72-frame stride)")
        return y_gait

    def _one_euro_local(x, min_cutoff=0.35, beta=0.002):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        out[0] = x[0]
        prev = x[0]
        dprev = 0.0
        for _i in range(1, len(x)):
            d = x[_i] - prev
            ds = dprev + 0.1 * (d - dprev)
            cut = min_cutoff + beta * abs(ds)
            alpha = cut / (cut + 1.0)
            out[_i] = prev + alpha * (x[_i] - prev)
            prev = out[_i]
            dprev = ds
        return out

    _ry_raw = compute_root_motion(stab_kps)
    _jerk_raw = float(np.max(np.abs(np.diff(_ry_raw, 2))))
    root_y_series = _one_euro_local(_ry_raw)
    print("root_y smoothed: jerk %.2f -> %.2f px/frame^2, amplitude %.1f -> %.1fpx"
          % (_jerk_raw, float(np.max(np.abs(np.diff(root_y_series, 2)))),
             float(np.ptp(_ry_raw)), float(np.ptp(root_y_series))))
    return (root_y_series,)


@app.cell
def _(facing_series, get_kp, get_score, np, stab_kps):
    def one_euro(x, min_cutoff=1.4, beta=0.008):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x); out[0] = x[0]; xp = x[0]; dxp = 0.0
        for i in range(1, len(x)):
            dx = x[i] - xp
            dxs = dxp + 0.1 * (dx - dxp)
            cutoff = min_cutoff + beta * abs(dxs)
            a = cutoff / (cutoff + 1.0)
            out[i] = xp + a * (x[i] - xp)
            xp = out[i]; dxp = dxs
        return out

    def interp_bad(track, bad):
        track = track.copy()
        good = ~bad
        if good.sum() < 2:
            return track
        idx = np.arange(len(track))
        track[bad] = np.interp(idx[bad], idx[good], track[good])
        return track

    LEG_DEF = {
        'L_front': ('L_Shoulder', 'L_Elbow', 'L_F_Paw', 'hum_L', 'rad_L'),
        'R_front': ('R_Shoulder', 'R_Elbow', 'R_F_Paw', 'hum_R', 'rad_R'),
        'L_hind':  ('L_Hip', 'L_Knee', 'L_B_Paw', 'fem_L', 'tib_L'),
        'R_hind':  ('R_Hip', 'R_Knee', 'R_B_Paw', 'fem_R', 'tib_R'),
    }
    CONF_THR = 0.35

    def build_leg_tracks(stab_list, facing_arr):
        n = len(stab_list)
        upper = {leg: np.zeros(n) for leg in LEG_DEF}
        lower = {leg: np.zeros(n) for leg in LEG_DEF}
        bad = {leg: np.zeros(n, dtype=bool) for leg in LEG_DEF}
        conf = {leg: np.zeros(n) for leg in LEG_DEF}

        for i, fr in enumerate(stab_list):
            facing = float(facing_arr[i])
            for leg, (rt, md, tp, _, _) in LEG_DEF.items():
                def a(p, q):
                    d = get_kp(fr, q) - get_kp(fr, p)
                    t = float(np.arctan2(d[1], d[0]))
                    return float(np.pi - t) if facing < 0 else t
                upper[leg][i] = a(rt, md)
                lower[leg][i] = a(md, tp)
                cmin = min(get_score(fr, rt), get_score(fr, md), get_score(fr, tp))
                bad[leg][i] = cmin < CONF_THR
                conf[leg][i] = cmin

        ACCEL_TARGET_DEG = 6.0
        CUTOFF_LADDER = [1.60, 1.20, 0.90, 0.70, 0.55, 0.45, 0.35, 0.28, 0.22]

        def fit_filter(series):
            for mc in CUTOFF_LADDER:
                out = one_euro(series, min_cutoff=mc, beta=0.005)
                a98 = float(np.percentile(np.abs(np.diff(out, 2)), 98)) * 180 / np.pi
                if a98 <= ACCEL_TARGET_DEG:
                    return out, mc, a98
            return out, CUTOFF_LADDER[-1], a98

        chosen = {}
        tracks = {}
        COLLAPSE_FRAC = 0.045
        COLLAPSE_ANGLE = 0.10
        spine_px = np.array([
            float(np.linalg.norm(get_kp(fr, 'root_of_tail') - get_kp(fr, 'Neck')))
            for fr in stab_kps])
        spine_px[spine_px < 1e-6] = np.nan

        n_collapse = {}
        for lname, rname, jl, jr in (
                ('L_front', 'R_front', 'L_F_Paw', 'R_F_Paw'),
                ('L_hind', 'R_hind', 'L_B_Paw', 'R_B_Paw')):
            if lname not in LEG_DEF or rname not in LEG_DEF:
                continue
            sep = np.array([
                float(np.linalg.norm(get_kp(stab_kps[i], jl) - get_kp(stab_kps[i], jr)))
                for i in range(n)]) / spine_px
            d_up = np.abs(np.arctan2(np.sin(upper[lname] - upper[rname]),
                                     np.cos(upper[lname] - upper[rname])))
            d_lo = np.abs(np.arctan2(np.sin(lower[lname] - lower[rname]),
                                     np.cos(lower[lname] - lower[rname])))
            merged = (sep < COLLAPSE_FRAC) & (d_up < COLLAPSE_ANGLE) \
                & (d_lo < COLLAPSE_ANGLE)
            far = np.where(conf[lname] < conf[rname], lname, rname)
            hit = {lname: 0, rname: 0}
            for i in np.where(merged)[0]:
                side = far[i]
                conf[side][i] = min(conf[side][i], CONF_THR * 0.5)
                bad[side][i] = True
                hit[side] += 1
            n_collapse[f"{lname[2:]}"] = dict(hit)
        print("  left/right collapse frames demoted (far side):", n_collapse)

        CONF_LO, CONF_HI = 0.28, 0.45
        MIN_ANCHOR_FRACTION = 0.65

        def conf_fill(v, c):
            thr = CONF_HI
            if float((c >= thr).mean()) < MIN_ANCHOR_FRACTION:
                thr = max(CONF_THR,
                          float(np.percentile(c, 100 * (1 - MIN_ANCHOR_FRACTION))))
            anchor = c >= thr
            if anchor.sum() < 2:
                return interp_bad(v, c < CONF_THR)
            i = np.arange(len(v))
            v_i = np.interp(i, i[anchor], v[anchor])
            wgt = np.clip((c - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)
            wgt = wgt * wgt * (3.0 - 2.0 * wgt)
            return wgt * v + (1.0 - wgt) * v_i

        for leg, (_, _, _, up_key, lo_key) in LEG_DEF.items():
            u = np.unwrap(upper[leg]); l = np.unwrap(lower[leg])
            u = conf_fill(u, conf[leg]); l = conf_fill(l, conf[leg])
            tracks[up_key], mc_u, a_u = fit_filter(u)
            tracks[lo_key], mc_l, a_l = fit_filter(l)
            chosen[up_key] = (mc_u, round(a_u, 2))
            chosen[lo_key] = (mc_l, round(a_l, 2))


        for leg, (_, _, _, up_key, lo_key) in LEG_DEF.items():
            rel = np.arctan2(np.sin(tracks[lo_key] - tracks[up_key]),
                             np.cos(tracks[lo_key] - tracks[up_key]))
            tracks[lo_key + '_rel'] = np.unwrap(rel)

        table = [{k: float(tracks[k][i]) for k in tracks} for i in range(n)]

        nbad = {leg: int(bad[leg].sum()) for leg in LEG_DEF}
        nmix = {leg: int(((conf[leg] > CONF_LO) & (conf[leg] < CONF_HI)).sum())
                for leg in LEG_DEF}
        print("leg tracks built. frames below CONF_THR per leg:", nbad)
        print("  frames in the blend band (partially replaced):", nmix)
        thrs = {}
        for leg in LEG_DEF:
            c = conf[leg]
            t = CONF_HI
            if float((c >= t).mean()) < MIN_ANCHOR_FRACTION:
                t = max(CONF_THR,
                        float(np.percentile(c, 100 * (1 - MIN_ANCHOR_FRACTION))))
            thrs[leg] = (round(t, 2), f"{float((c >= t).mean()):.0%}")
        print("  anchor threshold chosen per leg (threshold, % of frames kept):",
              thrs)
        print(f"  filter fitted per bone to p98 accel <= {ACCEL_TARGET_DEG} deg"
              f"/frame^2 (min_cutoff, achieved):")
        for k in sorted(chosen):
            mc, a = chosen[k]
            flag = "  <- hit the floor, still above target" if a > ACCEL_TARGET_DEG else ""
            print(f"     {k:7} cutoff {mc:.2f}  p98 {a:5.2f}d{flag}")
        return table

    leg_table = build_leg_tracks(stab_kps, facing_series)
    return leg_table, one_euro


@app.cell
def _(get_kp, np, stab_kps):
    def compute_facing(stab_list, hold=5):
        raw = np.array([1.0 if get_kp(fr, 'Nose')[0] >= get_kp(fr, 'root_of_tail')[0]
                        else -1.0 for fr in stab_list])
        out = raw.copy()
        cur = raw[0]; run = 0
        for i in range(len(raw)):
            if raw[i] != cur:
                run += 1
                if run >= hold:
                    cur = raw[i]; run = 0
            else:
                run = 0
            out[i] = cur
        flips = int((np.diff(out) != 0).sum())
        print(f"facing flips after hysteresis: {flips}")
        return out

    facing_series = compute_facing(stab_kps)
    return (facing_series,)


@app.cell
def _(
    KEYPOINT_NAMES,
    facing_series,
    get_kp,
    leg_table,
    np,
    one_euro,
    spine_mid_track,
    stab_kps,
    tail_tip_track,
):
    def wrap_angle(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    ELBOW_LIMIT = (-0.35, 1.60)
    STIFLE_LIMIT = (-2.00, 0.25)
    ARCH_FULL_SCALE = 0.25

    ARCH_LEAD_FRAMES = 0

    SPINE_ARCH_GAIN = 1.7
    SPINE_REST_FOLD = -0.11
    STRETCH_AMOUNT = 0.05

    def clamp_soft(rel, limits, softness=0.22):
        lo, hi = limits
        span = (hi - lo) * softness
        if span < 1e-6:
            return float(np.clip(rel, lo, hi))
        if rel > hi - span:
            return (hi - span) + span * float(np.tanh((rel - (hi - span)) / span))
        if rel < lo + span:
            return (lo + span) - span * float(np.tanh(((lo + span) - rel) / span))
        return rel

    def compute_tail_rel():
        rels = []
        for i, fr in enumerate(stab_kps):
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            d_body = tail - neck
            d_tail = np.asarray(tail_tip_track[i], dtype=float) - tail
            if np.linalg.norm(d_body) < 1e-6 or np.linalg.norm(d_tail) < 1e-6:
                rels.append(rels[-1] if rels else 0.0)
                continue
            r = float(np.arctan2(d_tail[1], d_tail[0]) - np.arctan2(d_body[1], d_body[0]))
            rels.append(wrap_angle(r))
        return np.array(rels)

    tail_rel_series = one_euro(np.unwrap(compute_tail_rel()), min_cutoff=0.5, beta=0.004)

    def _median_local(a_name, b_name):
        al, pe = [], []
        for i, fr in enumerate(stab_kps):
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            pt = (get_kp(fr, a_name) + get_kp(fr, b_name)) / 2.0
            ax = tail - neck
            L = float(np.linalg.norm(ax))
            if L < 1e-6:
                continue
            u = ax / L
            nvec = np.array([-u[1], u[0]])
            along = float((pt - neck) @ u) / L
            perp = float((pt - neck) @ nvec) / L
            if float(facing_series[i]) < 0:
                perp = -perp
            al.append(along)
            pe.append(perp)
        return float(np.median(al)), float(np.median(pe))

    SH_ALONG, SH_PERP = _median_local('L_Shoulder', 'R_Shoulder')
    HP_ALONG, HP_PERP = _median_local('L_Hip', 'R_Hip')
    print(f"limb attachments welded at body fractions: "
          f"shoulder along {SH_ALONG:+.3f} perp {SH_PERP:+.3f}, "
          f"hip along {HP_ALONG:+.3f} perp {HP_PERP:+.3f}")
    for _nm, _a, _p in (('shoulder', SH_ALONG, SH_PERP), ('hip', HP_ALONG, HP_PERP)):
        if _p > 0:
            print(f"  !! {_nm} perp {_p:+.3f} is on the BACK side of the spine; "
                  f"the limbs will attach above the body")
        if not (0.0 <= _a <= 1.0):
            print(f"  !! {_nm} along {_a:+.3f} falls outside neck..tail")

    def compute_limb_spread():
        out = np.zeros(len(stab_kps))
        for i, fr in enumerate(stab_kps):
            f = float(facing_series[i])
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            ax = tail - neck
            L = float(np.linalg.norm(ax))
            if L < 1e-6:
                continue
            u = ax / L
            vals = []
            for a, b in (('L_F_Paw', 'L_B_Paw'), ('R_F_Paw', 'R_B_Paw')):
                vals.append(float((get_kp(fr, a) - get_kp(fr, b)) @ u) / L)
            out[i] = float(np.mean(vals)) * (1.0 if f >= 0 else -1.0)
        return out

    _sp = one_euro(compute_limb_spread(), min_cutoff=0.30, beta=0.003)
    _sp = _sp - float(np.median(_sp))
    _spscale = float(np.percentile(np.abs(_sp), 95)) or 1.0
    stretch_series = np.clip(_sp / _spscale, -1.3, 1.3)
    print(f"torso length driver (limb spread, normalised): "
          f"{stretch_series.min():+.2f}..{stretch_series.max():+.2f}")

    USE_MEASURED_STRETCH = True
    MEASURED_STRETCH_GAIN = 0.35
    _sl_raw = np.array([
        float(np.linalg.norm(get_kp(fr, 'root_of_tail') - get_kp(fr, 'Neck')))
        for fr in stab_kps])

    def _movmed(a, win=145):
        out = np.empty_like(a)
        h = win // 2
        for i_mm in range(len(a)):
            out[i_mm] = np.median(a[max(0, i_mm - h):i_mm + h + 1])
        return out

    _spine_slow = _movmed(_sl_raw)
    _sl_rel = _sl_raw / np.maximum(_spine_slow, 1e-6) - 1.0
    _sl_rel = one_euro(_sl_rel, min_cutoff=0.5, beta=0.003)
    _agree = float(np.corrcoef(_sl_rel, _sp)[0, 1])
    print(f"measured torso stretch (detrended spine length): "
          f"{_sl_rel.min():+.3f}..{_sl_rel.max():+.3f} "
          f"(fractional); corr with limb-spread proxy {_agree:+.2f}")
    BEND_SHARE = 0.75
    CHORD_BEND_WEIGHT = 0.7
    chord_bend_series = np.zeros(len(stab_kps))
    if USE_MEASURED_STRETCH:
        BEND_KNEE = 0.06
        _u_k = np.clip(-_sl_rel / BEND_KNEE, 0.0, 1.0)
        _share_k = 3.0 * _u_k ** 2 - 2.0 * _u_k ** 3
        _contr = -_share_k * np.maximum(-_sl_rel, 0.0)
        _r_bend = np.clip(1.0 + BEND_SHARE * _contr, 0.30, 1.0)
        _x = np.sqrt(6.0 * (1.0 - _r_bend))
        for _nw in range(4):
            _xs = np.maximum(_x, 1e-9)
            _s = np.sin(_xs) / _xs
            _ds = (_xs * np.cos(_xs) - np.sin(_xs)) / _xs ** 2
            _step = np.where(np.abs(_ds) > 1e-12,
                             (_s - _r_bend) / np.where(np.abs(_ds) > 1e-12,
                                                       _ds, 1.0), 0.0)
            _x = np.clip(_x - _step, 0.0, np.pi - 1e-6)
        chord_bend_series = np.clip(2.0 * _x, 0.0, 1.4)

        HOLLOW_GAIN = 1.6
        _u_h = np.clip(_sl_rel / BEND_KNEE, 0.0, 1.0)
        _share_h = 3.0 * _u_h ** 2 - 2.0 * _u_h ** 3
        _hollow = -HOLLOW_GAIN * _share_h * np.maximum(_sl_rel, 0.0)
        chord_bend_series = chord_bend_series + _hollow
        print(f"chord channel now signed: hollow reaches "
              f"{np.degrees(_hollow.min()):.1f} deg on the extension side "
              f"(roach side unchanged)")

        BEND_SMOOTH_SIGMA = 3.0
        if BEND_SMOOTH_SIGMA > 0:
            _kr = int(np.ceil(3 * BEND_SMOOTH_SIGMA))
            _kx = np.arange(-_kr, _kr + 1, dtype=float)
            _kern = np.exp(-0.5 * (_kx / BEND_SMOOTH_SIGMA) ** 2)
            _kern /= _kern.sum()
            _pad = np.pad(chord_bend_series, _kr, mode='edge')
            _bend_pre = chord_bend_series.copy()
            chord_bend_series = np.convolve(_pad, _kern, mode='valid')
            print(f"chord bend smoothed (sigma {BEND_SMOOTH_SIGMA:.0f}): "
                  f"per-frame change p98 "
                  f"{np.degrees(np.percentile(np.abs(np.diff(_bend_pre)), 98)):.2f}"
                  f" -> "
                  f"{np.degrees(np.percentile(np.abs(np.diff(chord_bend_series)), 98)):.2f}"
                  f" deg")

        CHORD_BEND_CAP_DEG = 45.0
        _cap = np.radians(CHORD_BEND_CAP_DEG)
        _pk_before = float(np.degrees(chord_bend_series.max()))
        chord_bend_series = _cap * np.tanh(chord_bend_series / _cap)
        print(f"   signed range after cap: {np.degrees(chord_bend_series.min()):+.1f}"
              f"..{np.degrees(chord_bend_series.max()):+.1f} deg")
        print(f"chord bend soft-capped at {CHORD_BEND_CAP_DEG:.0f} deg: peak "
              f"{_pk_before:.1f} -> {np.degrees(chord_bend_series.max()):.1f} deg")
        _sinc = np.where(_x > 1e-9, np.sin(np.maximum(_x, 1e-9))
                         / np.maximum(_x, 1e-9), 1.0)
        _resid_rel = (1.0 + _sl_rel) / _sinc - 1.0
        stretch_series = np.clip(
            _resid_rel * MEASURED_STRETCH_GAIN / STRETCH_AMOUNT, -2.4, 2.4)
        print(f"measured contraction split: chord bend "
              f"{np.degrees(chord_bend_series.min()):.0f}"
              f"..{np.degrees(chord_bend_series.max()):.0f} deg total turn "
              f"(BEND_SHARE {BEND_SHARE}); residual seg_scale range "
              f"{1 + stretch_series.min() * STRETCH_AMOUNT:.3f}"
              f"..{1 + stretch_series.max() * STRETCH_AMOUNT:.3f}")
        _d_bend = np.degrees(np.abs(np.diff(chord_bend_series)))
        _d_seg = np.abs(np.diff(1.0 + stretch_series * STRETCH_AMOUNT))
        print(f"per-frame change: bend p98 {np.percentile(_d_bend, 98):.2f} deg "
              f"(max {_d_bend.max():.2f}), body length p98 "
              f"{100 * np.percentile(_d_seg, 98):.2f}% (max {100 * _d_seg.max():.2f}%)")

    _prot = (np.pi / 2.0) - np.unwrap(
        np.array([leg_table[i]['hum_L'] for i in range(len(stab_kps))]))
    _prot = _prot - float(np.median(_prot))
    _pscale = float(np.percentile(np.abs(_prot), 95)) or 1.0
    chest_drive_series = np.clip(_prot / _pscale, -1.4, 1.4)
    print(f"chest drive (humerus protraction, normalised): "
          f"{chest_drive_series.min():+.2f}..{chest_drive_series.max():+.2f}")

    _retr = np.unwrap(
        np.array([leg_table[i]['fem_L'] for i in range(len(stab_kps))])
    ) - (np.pi / 2.0)
    _retr = _retr - float(np.median(_retr))
    _rscale = float(np.percentile(np.abs(_retr), 95)) or 1.0
    rump_drive_series = np.clip(_retr / _rscale, -1.4, 1.4)
    print(f"haunch drive (femur retraction, normalised): "
          f"{rump_drive_series.min():+.2f}..{rump_drive_series.max():+.2f}")

    _ANCHOR_CONF = 0.35

    def _anchor_series(a_name, b_name):
        ia = KEYPOINT_NAMES.index(a_name)
        ib = KEYPOINT_NAMES.index(b_name)
        pts, bad = [], []
        for fr in stab_kps:
            pts.append((get_kp(fr, a_name) + get_kp(fr, b_name)) / 2.0)
            bad.append(min(fr['scores'][ia], fr['scores'][ib]) < _ANCHOR_CONF)
        pts = np.asarray(pts, dtype=float)
        bad = np.asarray(bad, dtype=bool)

        def fill(v):
            v = v.copy()
            good = ~bad
            if good.sum() >= 2:
                i = np.arange(len(v))
                v[bad] = np.interp(i[bad], i[good], v[good])
            return v

        out = np.empty_like(pts)
        for c in (0, 1):
            out[:, c] = one_euro(fill(pts[:, c]), min_cutoff=0.45, beta=0.004)
        return out, int(bad.sum())

    shoulder_series, _nb_sh = _anchor_series('L_Shoulder', 'R_Shoulder')
    hip_series, _nb_hp = _anchor_series('L_Hip', 'R_Hip')
    print(f"anchors smoothed to limb standard "
          f"(low-confidence frames interpolated: shoulder {_nb_sh}, hip {_nb_hp})")

    ARCH_SMOOTH_SIGMA = 6.0

    def _gauss_smooth(a, sigma):
        r = int(np.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        pad = np.pad(a, ((r, r), (0, 0)), mode='edge')
        return np.stack([np.convolve(pad[:, c], k, mode='valid')
                         for c in range(a.shape[1])], axis=1)

    _smt_raw = np.asarray(spine_mid_track, dtype=float)

    STRIDE = 72
    PHASE_TOL = 2.0
    PHASE_PULL = 0.75

    def _phase_correct(track):
        n = len(track)
        out = track.copy()
        dev_all, fixed = [], []
        for i in range(n):
            peers = [i + k * STRIDE for k in (-2, -1, 1, 2)]
            peers = [j for j in peers if 0 <= j < n]
            if len(peers) < 2:
                continue
            med = np.median(track[peers], axis=0)
            dev_all.append(np.linalg.norm(track[i] - med))
        if not dev_all:
            return out, []
        scale = float(np.median(dev_all)) + 1e-9
        for i in range(n):
            peers = [i + k * STRIDE for k in (-2, -1, 1, 2)]
            peers = [j for j in peers if 0 <= j < n]
            if len(peers) < 2:
                continue
            if not (any(j < i for j in peers) and any(j > i for j in peers)):
                continue
            med = np.median(track[peers], axis=0)
            d = float(np.linalg.norm(track[i] - med))
            if d > PHASE_TOL * scale:
                out[i] = track[i] + (med - track[i]) * PHASE_PULL
                fixed.append((i, d / scale))
        return out, fixed

    _smt_pc, _fixed = _phase_correct(_smt_raw)
    if _fixed:
        _worst = sorted(_fixed, key=lambda kv: -kv[1])[:8]
        print(f"spine track: {len(_fixed)} frames disagree with their own gait "
              f"phase and were pulled {PHASE_PULL:.0%} toward it")
        print("   worst: " + ", ".join(f"f{i}({d:.1f}x)" for i, d in _worst))
    else:
        print("spine track: no frame disagrees with its own gait phase")

    spine_mid_used = _gauss_smooth(_smt_pc, ARCH_SMOOTH_SIGMA)

    _step_mm = np.linalg.norm(np.diff(_smt_raw, axis=0), axis=1)
    _moving = np.where(_step_mm > 1e-6)[0]
    HELD_FROM = int(_moving.max()) + 2 if _moving.size else len(_smt_raw)
    arch_trust_series = np.ones(len(_smt_raw))
    _n_held = len(_smt_raw) - HELD_FROM
    if _n_held >= 2:
        _ramp = np.arange(1, _n_held + 1) / _n_held
        arch_trust_series[HELD_FROM:] = 0.5 * (1.0 + np.cos(np.pi * _ramp))
        print(f"annotation held tail detected: frames {HELD_FROM}..{len(_smt_raw) - 1} "
              f"({_n_held} frames) carry no observation; arch tapered to 0 over them")
    else:
        print("annotation covers every frame; no held tail to taper")
    _obs = arch_trust_series > 0.999
    _jerk_before = float(np.max(np.abs(np.diff(_smt_raw[:, 1], 2))))
    _jerk_after = float(np.max(np.abs(np.diff(spine_mid_used[:, 1], 2))))
    print(f"spine track smoothed (sigma {ARCH_SMOOTH_SIGMA:.0f}): peak second "
          f"difference {_jerk_before:.2f} -> {_jerk_after:.2f}px/frame^2")

    _root_ref_al = get_kp(stab_kps[0], 'root_of_tail')

    def _rs_al(q_al, facing_al):
        q_al = np.asarray(q_al, dtype=float)
        return (np.array([2 * _root_ref_al[0] - q_al[0], q_al[1]])
                if facing_al < 0 else q_al)

    def _arch_of(track_al):
        out_al = np.zeros(len(stab_kps))
        for i_al, fr_al in enumerate(stab_kps):
            f_al = float(facing_series[i_al])
            neck_al = _rs_al(get_kp(fr_al, 'Neck'), f_al)
            tail_al = _rs_al(get_kp(fr_al, 'root_of_tail'), f_al)
            mid_al = _rs_al(track_al[i_al], f_al)
            ch_al = tail_al - neck_al
            cl_al = float(np.linalg.norm(ch_al))
            if cl_al < 1e-6:
                continue
            u_al = ch_al / cl_al
            out_al[i_al] = float((mid_al - neck_al)
                                 @ np.array([-u_al[1], u_al[0]])) / (cl_al * 0.5)
        return out_al

    _n_flip = int(np.sum(np.asarray(facing_series) < 0))
    print(f"facing: {_n_flip}/{len(facing_series)} frames mirrored "
          f"(dog runs left) -- arch measured in canonical space")

    ALIGN_TO_SPREAD = True
    _spread_al = np.full(len(stab_kps), np.nan)
    for _i_al, _fr_al in enumerate(stab_kps):
        _neck_al = get_kp(_fr_al, 'Neck')
        _ax_al = get_kp(_fr_al, 'root_of_tail') - _neck_al
        _ln_al = float(np.linalg.norm(_ax_al))
        if _ln_al < 1e-6:
            continue
        _u2_al = _ax_al / _ln_al
        _pr_al = [float((get_kp(_fr_al, _n_al) - _neck_al) @ _u2_al)
                  for _n_al in ('L_F_Paw', 'R_F_Paw', 'L_B_Paw', 'R_B_Paw')]
        _spread_al[_i_al] = (max(_pr_al) - min(_pr_al)) / _ln_al
    _spread_al = np.where(np.isnan(_spread_al),
                          np.nanmean(_spread_al), _spread_al)
    _gather_al = -(_spread_al - float(np.mean(_spread_al)))

    ANNOT_ALIGN_AUTO = True
    if ANNOT_ALIGN_AUTO:
        _ann_dc = _arch_of(spine_mid_used)
        _ann_dc = _ann_dc - float(np.mean(_ann_dc))
        _chd_dc = (_gather_al if ALIGN_TO_SPREAD
                   else chord_bend_series - float(np.mean(chord_bend_series)))
        _ref_name = 'limb gathering' if ALIGN_TO_SPREAD else 'chord bend'
        _chd_dc0 = chord_bend_series - float(np.mean(chord_bend_series))
        _r_chd = float(np.corrcoef(_chd_dc0, _gather_al)[0, 1])
        _l_chd = max(range(-25, 26), key=lambda L: float(
            np.corrcoef(np.roll(_chd_dc0, L), _gather_al)[0, 1]))
        print(f"reference check: chord bend vs limb gathering corr "
              f"{_r_chd:+.2f} at lag 0, best "
              f"{float(np.corrcoef(np.roll(_chd_dc0, _l_chd), _gather_al)[0, 1]):+.2f}"
              f" at lag {_l_chd:+d} (0 means the chord channel is on time)")
        _lag_best, _r_best = 0, -2.0
        for _L in range(-30, 31):
            _r_al = float(np.corrcoef(np.roll(_ann_dc, _L), _chd_dc)[0, 1])
            if _r_al > _r_best:
                _lag_best, _r_best = _L, _r_al
        _r_zero = float(np.corrcoef(_ann_dc, _chd_dc)[0, 1])
        print(f"annotation vs {_ref_name}: corr {_r_zero:+.2f} at "
              f"lag 0, best {_r_best:+.2f} at lag {_lag_best:+d}")
        if abs(_lag_best) >= 3 and _r_best > _r_zero + 0.15:
            ARCH_LEAD_FRAMES = -int(_lag_best)
            print(f"   => annotation re-timed by {ARCH_LEAD_FRAMES:+d} frames so "
                  f"the two flexion channels reinforce instead of cancel")
        else:
            print("   => channels already aligned; no re-timing applied")

    if ARCH_LEAD_FRAMES:
        n_sh = int(ARCH_LEAD_FRAMES)
        if n_sh > 0:
            spine_mid_used = np.concatenate([
                spine_mid_used[n_sh:],
                np.repeat(spine_mid_used[-1:], n_sh, axis=0)], axis=0)
            print(f"arch advanced {n_sh} frames; last {n_sh} frames hold the "
                  f"final observed value")
        else:
            k_sh = -n_sh
            spine_mid_used = np.concatenate([
                np.repeat(spine_mid_used[:1], k_sh, axis=0),
                spine_mid_used[:-k_sh]], axis=0)
            print(f"arch delayed {k_sh} frames; first {k_sh} frames hold the "
                  f"first observed value")

    def _arch_raw():
        out = np.zeros(len(stab_kps))
        for i, fr in enumerate(stab_kps):
            neck = get_kp(fr, 'Neck'); tail = get_kp(fr, 'root_of_tail')
            ch = tail - neck
            cl = float(np.linalg.norm(ch))
            if cl < 1e-6:
                continue
            u = ch / cl
            out[i] = float((np.asarray(spine_mid_used[i]) - neck)
                           @ np.array([-u[1], u[0]])) / (cl * 0.5)
        return out

    _arr = _arch_raw()
    arch_audit_raw = _arch_of(spine_mid_used)
    _pos = _arr[_arr > 0]
    _neg = -_arr[_arr < 0]
    ARCH_LIMIT_POS = float(np.percentile(_pos, 82.0)) if _pos.size else 0.3
    ARCH_LIMIT_NEG = float(np.percentile(_neg, 95.0)) if _neg.size else 0.3
    print(f"arch limits: roach p82 = {ARCH_LIMIT_POS:.3f} "
          f"(max {_pos.max() if _pos.size else 0:.3f}), "
          f"sag p95 = {ARCH_LIMIT_NEG:.3f} "
          f"(max {_neg.max() if _neg.size else 0:.3f})")

    ANNOT_TRUST_SCALE = 0.25
    BEND_FULL_SCALE = (ARCH_LIMIT_POS * SPINE_ARCH_GAIN * ANNOT_TRUST_SCALE
                       + CHORD_BEND_WEIGHT
                       * float(np.percentile(chord_bend_series[_obs], 95)) / 2.25)
    BEND_FULL_SCALE = max(BEND_FULL_SCALE, 1e-3)
    print(f"combined bend full-scale = {BEND_FULL_SCALE:.3f} rad "
          f"(annotation {ARCH_LIMIT_POS * SPINE_ARCH_GAIN * ANNOT_TRUST_SCALE:.3f} + chord "
          f"{CHORD_BEND_WEIGHT * float(np.percentile(chord_bend_series[_obs], 95)) / 2.25:.3f})")

    EAR_DRIVE, EAR_STIFFNESS, EAR_DAMPING, EAR_MAX = 1.4, 0.22, 0.80, 0.40
    EAR_DRAG = 0.34

    def compute_head_angles():
        out = []
        for fr in stab_kps:
            d = get_kp(fr, 'Nose') - get_kp(fr, 'Neck')
            out.append(float(np.arctan2(d[1], d[0]))
                       if np.linalg.norm(d) > 1e-6 else 0.0)
        return np.unwrap(np.array(out, dtype=float))

    def compute_ear_lag(head_ang):
        acc = np.gradient(np.gradient(head_ang))
        lag = np.zeros(len(head_ang)); x = v = 0.0
        for i in range(len(head_ang)):
            v = (v - EAR_STIFFNESS * x - EAR_DRIVE * acc[i]) * EAR_DAMPING
            x += v
            lag[i] = x
        return np.clip(lag, -EAR_MAX, EAR_MAX)

    def compute_limb_speed():
        spd = np.zeros(len(stab_kps))
        for c in ('fem_L', 'fem_R', 'hum_L', 'hum_R'):
            a = np.unwrap(np.array([leg_table[i][c] for i in range(len(stab_kps))]))
            spd += np.abs(np.gradient(a))
        return one_euro(spd / 4.0, min_cutoff=0.6, beta=0.005)

    TAIL_WHIP, TAIL_WHIP_STIFF, TAIL_WHIP_DAMP, TAIL_WHIP_MAX = 1.6, 0.25, 0.82, 0.35

    def compute_tail_whip(base_ang):
        acc = np.gradient(np.gradient(base_ang))
        out = np.zeros(len(base_ang)); x = v = 0.0
        for i in range(len(base_ang)):
            v = (v - TAIL_WHIP_STIFF * x - TAIL_WHIP * acc[i]) * TAIL_WHIP_DAMP
            x += v
            out[i] = x
        return np.clip(out, -TAIL_WHIP_MAX, TAIL_WHIP_MAX)

    tail_whip_series = compute_tail_whip(tail_rel_series)
    print(f"tail whip: +-{np.abs(tail_whip_series).max()*180/np.pi:.1f} deg")

    ear_lag_series = compute_ear_lag(
        one_euro(compute_head_angles(), min_cutoff=0.8, beta=0.01))
    _ls = compute_limb_speed()
    ear_drag_series = np.clip(_ls / max(float(np.percentile(_ls, 90)), 1e-6),
                              0.0, 1.0) * EAR_DRAG
    print(f"ear: lag +-{np.abs(ear_lag_series).max()*180/np.pi:.1f} deg, "
          f"drag {np.degrees(ear_drag_series.min()):.1f}.."
          f"{np.degrees(ear_drag_series.max()):.1f} deg")

    PAW_LEVER_LIMIT = 0.8
    PAW_SMOOTH_SIGMA = 2.0
    _PAW_REST_P, _PAW_FLATTEN_P = 0.45, 0.65

    def _paw_offset_series(low_key, up_key, limits):
        n_p = len(stab_kps)
        a_p = np.array([leg_table[i][up_key]
                        + clamp_soft(leg_table[i][low_key + '_rel'], limits)
                        for i in range(n_p)])
        aw_p = np.arctan2(np.sin(a_p), np.cos(a_p))
        stance_p = np.clip(np.cos(aw_p - np.pi / 2), 0.0, 1.0) ** 2
        flat_p = np.arctan2(np.sin(-aw_p), np.cos(-aw_p))
        if PAW_LEVER_LIMIT:
            flat_p = PAW_LEVER_LIMIT * np.tanh(flat_p / PAW_LEVER_LIMIT)
        off_p = _PAW_REST_P + _PAW_FLATTEN_P * stance_p * (flat_p - _PAW_REST_P)
        if PAW_SMOOTH_SIGMA > 0:
            kr_p = int(np.ceil(3 * PAW_SMOOTH_SIGMA))
            kx_p = np.arange(-kr_p, kr_p + 1, dtype=float)
            k_p = np.exp(-0.5 * (kx_p / PAW_SMOOTH_SIGMA) ** 2)
            k_p /= k_p.sum()
            off_p = np.convolve(np.pad(off_p, kr_p, mode='edge'), k_p, mode='valid')
        return off_p

    paw_offsets = {
        'rad_L': _paw_offset_series('rad_L', 'hum_L', ELBOW_LIMIT),
        'rad_R': _paw_offset_series('rad_R', 'hum_R', ELBOW_LIMIT),
        'tib_L': _paw_offset_series('tib_L', 'fem_L', STIFLE_LIMIT),
        'tib_R': _paw_offset_series('tib_R', 'fem_R', STIFLE_LIMIT),
    }
    for _k_p, _v_p in paw_offsets.items():
        print(f"paw offset {_k_p}: range "
              f"{np.degrees(_v_p.min()):+6.1f}..{np.degrees(_v_p.max()):+6.1f} deg, "
              f"per-frame p98 {np.degrees(np.percentile(np.abs(np.diff(_v_p)), 98)):5.2f} deg")

    def solve_pose(frame_data, rig, idx, root_y=0.0, tail_state=None,
                   fold_ref=0.0, target_spine=200.0, root_anchor=(300, 235)):
        kp = {n: get_kp(frame_data, n) for n in KEYPOINT_NAMES}
        neck_p, tail_p = kp['Neck'], kp['root_of_tail']
        spine_len = np.linalg.norm(tail_p - neck_p)
        if spine_len < 1e-6:
            return None
        SLOW_SCALE_NORM = True
        _sl_ref = (float(_spine_slow[idx]) if SLOW_SCALE_NORM
                   else float(spine_len))
        if _sl_ref < 1e-6:
            _sl_ref = float(spine_len)
        sc = target_spine / _sl_ref
        facing = float(facing_series[idx])

        hips = (kp['L_Hip'] + kp['R_Hip']) / 2
        shs = (kp['L_Shoulder'] + kp['R_Shoulder']) / 2
        pelvis = tail_p * 0.55 + hips * 0.45
        root_pos = np.array([root_anchor[0], root_anchor[1] + root_y], dtype=float)

        def rs(p):
            q = root_pos + (np.asarray(p, dtype=float) - pelvis) * sc
            if facing < 0:
                q = np.array([2.0 * root_pos[0] - q[0], q[1]])
            return q

        def ang(a, b):
            d = np.asarray(b) - np.asarray(a)
            return float(np.arctan2(d[1], d[0]))

        neck_c = rs(neck_p); tail_c = rs(tail_p)
        backmid_c = rs(spine_mid_used[idx])
        chord = tail_c - neck_c
        chord_len = np.linalg.norm(chord)
        if chord_len > 1e-6:
            cd = chord / chord_len
            cn = np.array([-cd[1], cd[0]])
            perp = float((backmid_c - neck_c) @ cn)
            _a = perp / (chord_len * 0.5)
            _lim = ARCH_LIMIT_POS if _a > 0 else ARCH_LIMIT_NEG
            _knee = _lim * 0.62
            if abs(_a) > _knee:
                _room = _lim - _knee
                _a = np.sign(_a) * (_knee + _room * np.tanh((abs(_a) - _knee) / _room))
            ANNOT_TRUST = 0.25
            arch = (float(_a) * float(arch_trust_series[idx]) * ANNOT_TRUST)
        else:
            arch = 0.0

        seg_scale = 1.0 + float(stretch_series[idx]) * STRETCH_AMOUNT

        total_bend = (SPINE_REST_FOLD + arch * SPINE_ARCH_GAIN
                      + CHORD_BEND_WEIGHT
                      * float(chord_bend_series[idx]) / 2.25)

        bend_norm = float(np.clip((total_bend - SPINE_REST_FOLD)
                                  / BEND_FULL_SCALE, -1.0, 1.0))

        w = {'root': 0.0}

        raw_dir = ang(rs(pelvis), rs(neck_p))
        base_dir = raw_dir * 0.85


        tilt = (np.array([-2.3, -0.5, 0.6, 2.2]) * (total_bend * 0.5)
                / max(seg_scale, 0.7))
        w['spine1'] = base_dir + tilt[0]
        w['spine2'] = base_dir + tilt[1]
        w['spine3'] = base_dir + tilt[2]
        w['spine4'] = base_dir + tilt[3]
        w['neck'] = ang(rs(neck_p), rs(kp['Nose']))
        w['head'] = w['neck']
        _el = float(ear_lag_series[idx]) * facing
        _ed = float(ear_drag_series[idx])
        w['ear_upper'] = w['head'] + rig['_style']['ear_droop'] + _el + _ed
        w['ear_lower'] = w['ear_upper'] + 0.25 + _el * 0.55 + _ed * 0.4
        w['shoulder'] = w['spine4']
        w['hip'] = w['spine1']

        PAW_REST = 0.45
        PAW_FLATTEN = 0.65

        def paw_angle(a_lower):
            a_w = float(np.arctan2(np.sin(a_lower), np.cos(a_lower)))
            stance = float(np.clip(np.cos(a_w - np.pi / 2), 0.0, 1.0)) ** 2
            to_flat = float(np.arctan2(np.sin(-a_w), np.cos(-a_w)))
            return a_lower + PAW_REST + PAW_FLATTEN * stance * (to_flat - PAW_REST)

        t = leg_table[idx]
        for s, up_f, lo_f, up_h, lo_h in [
                ('L', 'hum_L', 'rad_L', 'fem_L', 'tib_L'),
                ('R', 'hum_R', 'rad_R', 'fem_R', 'tib_R')]:
            a_hum = t[up_f]
            a_rad = a_hum + clamp_soft(t[lo_f + '_rel'], ELBOW_LIMIT)
            a_fem = t[up_h]
            a_tib = a_fem + clamp_soft(t[lo_h + '_rel'], STIFLE_LIMIT)
            w[f'humerus_{s}'] = a_hum
            w[f'radius_{s}'] = a_rad
            w[f'paw_front_{s}'] = a_rad + float(paw_offsets['rad_' + s][idx])
            w[f'femur_{s}'] = a_fem
            w[f'tibia_{s}'] = a_tib
            w[f'paw_hind_{s}'] = a_tib + float(paw_offsets['tib_' + s][idx])


        TAIL_FOLLOW_RUMP = 1.0
        rel = tail_rel_series[idx] * facing
        tail_base = base_dir + TAIL_FOLLOW_RUMP * (w['spine1'] - base_dir)
        w['tail_1'] = wrap_angle(tail_base + np.pi + rel)
        w['tail_2'] = (w['tail_1'] + rig['_style']['tail_curl_rel']
                       + float(tail_whip_series[idx]) * facing)

        local = {'root': 0.0}
        for name, b in rig.items():
            if name.startswith('_') or name == 'root':
                continue
            local[name] = wrap_angle(w[name] - w.get(b['parent'], 0.0))

        arch_norm = np.clip(arch / ARCH_FULL_SCALE, -1.0, 1.0)
        L_seg = rig['_spine_meta']['L_seg'] * seg_scale

        _base_len = rig['_spine_meta']['L_seg'] * 4.0
        _extra = (L_seg * 4.0) - _base_len
        _fwd = np.array([np.cos(base_dir), np.sin(base_dir)])
        CHAIN_REAR_SHARE = 0.40
        _chain_start = root_pos - _fwd * (_extra * CHAIN_REAR_SHARE)

        verts = [_chain_start.copy()]
        pos = _chain_start.copy()
        for i in range(1, 5):
            a = w[f'spine{i}']
            pos = pos + np.array([L_seg*np.cos(a), L_seg*np.sin(a)])
            verts.append(pos.copy())
        spine_verts = np.array(verts)

        def edge_anchor(joint_world, seg_start_pos, seg_ang):
            c, s = np.cos(-seg_ang), np.sin(-seg_ang)
            d = np.asarray(joint_world) - seg_start_pos
            lx = d[0]*c - d[1]*s
            ly = d[0]*s + d[1]*c
            wdn = rig['_spine_meta']['w_dn']
            lx = clamp_soft(lx, (-L_seg*0.7, L_seg*0.7), softness=0.35)
            ly = clamp_soft(ly, (-wdn*4.0, wdn*0.92), softness=0.25)
            cc, ss = np.cos(seg_ang), np.sin(seg_ang)
            return seg_start_pos + np.array([lx*cc - ly*ss, lx*ss + ly*cc])

        def _body_point(along, perp):
            a = tail_c - neck_c
            L = float(np.linalg.norm(a))
            if L < 1e-6:
                return neck_c
            u = a / L
            return neck_c + u * (along * L) + np.array([-u[1], u[0]]) * (perp * L)

        def _chain_point(along, perp):
            pts_cp = spine_verts[::-1].astype(float)
            seg_cp = np.diff(pts_cp, axis=0)
            segl_cp = np.linalg.norm(seg_cp, axis=1)
            total_cp = float(segl_cp.sum())
            if total_cp < 1e-6:
                return spine_verts[2].copy()
            target_cp = float(np.clip(along, 0.0, 1.0)) * total_cp
            acc_cp = 0.0
            for k_cp in range(len(seg_cp)):
                if acc_cp + segl_cp[k_cp] >= target_cp or k_cp == len(seg_cp) - 1:
                    t_cp = (target_cp - acc_cp) / max(segl_cp[k_cp], 1e-9)
                    base_cp = pts_cp[k_cp] + seg_cp[k_cp] * t_cp
                    u_cp = seg_cp[k_cp] / max(segl_cp[k_cp], 1e-9)
                    n_cp = np.array([-u_cp[1], u_cp[0]])
                    return base_cp + n_cp * (perp * total_cp)
                acc_cp += segl_cp[k_cp]
            return spine_verts[2].copy()

        pose_chest_drive = float(chest_drive_series[idx])
        pose_rump_drive = float(rump_drive_series[idx])

        SH_SLIDE = 0.030
        HP_SLIDE = 0.020
        _cd_a = float(np.clip(pose_chest_drive, -1.2, 1.2))
        _rd_a = float(np.clip(pose_rump_drive, -1.2, 1.2))
        _sh_along = float(np.clip(SH_ALONG - SH_SLIDE * _cd_a, 0.02, 0.98))
        _hp_along = float(np.clip(HP_ALONG + HP_SLIDE * _rd_a, 0.02, 0.98))
        anchors = {
            'shoulder': edge_anchor(_chain_point(_sh_along, SH_PERP),
                                    spine_verts[3], w['spine4']),
            'hip':      edge_anchor(_chain_point(_hp_along, HP_PERP),
                                    spine_verts[0], w['spine1']),
        }
        return {'root_pos': root_pos, 'scale': sc, 'local': local,
                'facing': facing, 'anchors': anchors, 'arch': arch,
                'chest_drive': float(chest_drive_series[idx]),
                'rump_drive': float(rump_drive_series[idx]),
            'shoulder_anchor': np.asarray(anchors['shoulder'], dtype=float),
            'spine_verts': spine_verts, 'seg_scale': seg_scale,
            'bend_norm': bend_norm,
                'world_leg': {
                    'humerus_L': w['humerus_L'], 'humerus_R': w['humerus_R'],
                    'femur_L': w['femur_L'], 'femur_R': w['femur_R']}}

    def calibrate_spine_fold(stab_list, rig):
        arches, scales = [], []
        for i, fr in enumerate(stab_list):
            p = solve_pose(fr, rig, i, root_y=0.0)
            if p is not None:
                arches.append(p['arch']); scales.append(p['seg_scale'])
        print(f"arch range: [{np.min(arches):.2f}, {np.max(arches):.2f}]")
        print(f"body stretch: [{np.min(scales):.2f}, {np.max(scales):.2f}] "
              f"({100*(np.max(scales)-np.min(scales)):.0f}% length change)")
        return 0.0
    return (
        ARCH_LEAD_FRAMES,
        ELBOW_LIMIT,
        SPINE_ARCH_GAIN,
        SPINE_REST_FOLD,
        STIFLE_LIMIT,
        arch_audit_raw,
        arch_trust_series,
        calibrate_spine_fold,
        solve_pose,
        wrap_angle,
    )


@app.cell
def _(np, rig, solve_pose, stab_kps):
    def compute_world_transforms(rig, pose):
        world = {'root': {'pos': np.array(pose['root_pos'], dtype=float), 'angle': 0.0}}
        local = pose['local']
        anchors = pose.get('anchors', {})
        seg_scale = pose.get('seg_scale', 1.0)
        resolved = {'root'}
        pending = [n for n in rig if n != 'root' and not n.startswith('_')]

        while pending:
            progressed = False
            for name in list(pending):
                b = rig[name]
                if b['parent'] not in resolved:
                    continue
                pw = world[b['parent']]
                ox, oy = float(b['offset'][0]), float(b['offset'][1])
                if name == 'neck' or name.startswith('spine'):
                    ox *= seg_scale
                c, s = np.cos(pw['angle']), np.sin(pw['angle'])
                pos = pw['pos'] + np.array([ox*c - oy*s, ox*s + oy*c])
                angle = pw['angle'] + local.get(name, 0.0)
                if name.startswith('spine') and 'spine_verts' in pose:
                    pos = np.asarray(pose['spine_verts'][int(name[-1]) - 1],
                                     dtype=float)
                if name in anchors:
                    pos = np.array(anchors[name], dtype=float)
                world[name] = {'pos': pos, 'angle': angle}
                resolved.add(name)
                pending.remove(name)
                progressed = True
            if not progressed:
                print("WARNING unresolved:", pending)
                break
        return world

    _w = compute_world_transforms(rig, solve_pose(stab_kps[0], rig, 0))
    print("world transform OK, bones:", len(_w))
    return (compute_world_transforms,)


@app.cell
def _(Image, ImageDraw, np, rig):
    JOINT_STYLE = {
        'neck': (0.022, '#4A4A4A', '#222222', 2), 'head': (0.016, '#4A4A4A', '#222222', 1),
        'humerus_L': (0.040, '#505050', '#222222', 2), 'radius_L': (0.027, '#4A4A4A', '#222222', 2),
        'femur_L': (0.038, '#505050', '#222222', 2), 'tibia_L': (0.027, '#4A4A4A', '#222222', 2),
        'humerus_R': (0.036, '#5E5E5E', '#454545', 1), 'radius_R': (0.025, '#5A5A5A', '#454545', 1),
        'femur_R': (0.034, '#5E5E5E', '#454545', 1), 'tibia_R': (0.025, '#5A5A5A', '#454545', 1),
    }

    def body_ribbon(pose):
        raw = pose['spine_verts'].copy()
        meta = rig['_spine_meta']
        w_up_k = np.array([0.112, 0.128, 0.132, 0.128, 0.112]) * 200.0
        w_dn_k = np.array([0.135, 0.110, 0.120, 0.158, 0.150]) * 200.0

        RIBBON_SAMPLES = 17

        def _catmull(P, n_out):
            P = np.asarray(P, dtype=float)
            ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
            segs = len(P) - 1
            out = np.empty((n_out, 2))
            for k in range(n_out):
                g = k / (n_out - 1) * segs
                i = min(int(g), segs - 1)
                t = g - i
                p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
                out[k] = 0.5 * ((2 * p1) + (-p0 + p2) * t
                                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                                + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
            return out

        verts = _catmull(raw, RIBBON_SAMPLES)
        _u = np.linspace(0, len(raw) - 1, RIBBON_SAMPLES)
        w_up = np.interp(_u, np.arange(len(raw)), w_up_k)
        w_dn = np.interp(_u, np.arange(len(raw)), w_dn_k)

        ARCH_DOME = 0.35
        BELLY_TUCK = 0.30
        LOIN_U = 0.38
        LOIN_SIGMA = 0.28
        _bn = float(np.clip(pose.get('bend_norm', 0.0), -1.0, 1.0))
        _t_rb = np.linspace(0.0, 1.0, RIBBON_SAMPLES)
        _bump = np.exp(-0.5 * ((_t_rb - LOIN_U) / LOIN_SIGMA) ** 2)
        w_up = w_up * np.maximum(1.0 + ARCH_DOME * _bn * _bump, 0.65)
        w_dn = w_dn * np.maximum(1.0 - BELLY_TUCK * _bn * _bump, 0.70)

        dirs = np.zeros_like(verts)
        for i in range(len(verts)):
            if i == 0:
                d = verts[1] - verts[0]
            elif i == len(verts) - 1:
                d = verts[-1] - verts[-2]
            else:
                d = verts[i+1] - verts[i-1]
            n = np.linalg.norm(d)
            dirs[i] = d / n if n > 1e-6 else np.array([1.0, 0.0])
        normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

        top = [verts[i] - normals[i] * w_up[i] for i in range(len(verts))]
        bot = [verts[i] + normals[i] * w_dn[i] for i in range(len(verts))]


        _ss = float(pose.get('seg_scale', 1.0))
        neck_ext = verts[-1] + dirs[-1] * (meta['L_seg'] * _ss * 0.15)

        CHEST_BREATHE = 0.12
        RUMP_BREATHE = 0.10
        _cd = float(np.clip(pose.get('chest_drive', 0.0), -1.2, 1.2))
        _rd = float(np.clip(pose.get('rump_drive', 0.0), -1.2, 1.2))
        _n_rb = len(verts)
        _breathe_span = 5
        for k_rb in range(_n_rb - _breathe_span, _n_rb):
            w_rb = (k_rb - (_n_rb - 1 - _breathe_span)) / _breathe_span
            bot[k_rb] = bot[k_rb] + dirs[k_rb] * (
                _cd * CHEST_BREATHE * meta['L_seg'] * _ss * w_rb)
        for k_rb in range(0, _breathe_span):
            w_rb = 1.0 - k_rb / _breathe_span
            bot[k_rb] = bot[k_rb] - dirs[k_rb] * (
                _rd * RUMP_BREATHE * meta['L_seg'] * _ss * w_rb)
        neck_ext = neck_ext + dirs[-1] * (
            _cd * CHEST_BREATHE * meta['L_seg'] * _ss)

        back = -dirs[0]
        rump_top = top[0]
        rump_bot = bot[0]
        rump_mid = (rump_top + rump_bot) / 2
        bulge = meta['L_seg'] * _ss * (0.25 + 0.5 * RUMP_BREATHE * _rd)
        rump_c = rump_mid + back * bulge

        poly = []
        poly.append(tuple(rump_top))
        for i in range(len(verts)):
            poly.append(tuple(top[i]))
        poly.append(tuple(neck_ext - normals[-1] * w_up[-1]))
        poly.append(tuple(neck_ext + normals[-1] * w_dn[-1]))
        for i in range(len(verts) - 1, -1, -1):
            poly.append(tuple(bot[i]))
        poly.append(tuple(rump_c))
        return poly

    def render_rig(rig, world, pose=None, facing=1.0, canvas_size=(600, 400), ref=200.0):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        d = ImageDraw.Draw(canvas)

        def to_world(mesh, wt):
            c, s = np.cos(wt['angle']), np.sin(wt['angle']); px, py = wt['pos']
            return [(int(x*c - y*s + px), int(x*s + y*c + py)) for (x, y) in mesh]

        def disc(pos, r, fill, outline='#222222', width=1):
            r = int(r)
            d.ellipse([int(pos[0])-r, int(pos[1])-r, int(pos[0])+r, int(pos[1])+r],
                      fill=fill, outline=outline, width=width)

        if pose is not None and 'spine_verts' in pose:
            poly = body_ribbon(pose)
            _pal = rig.get('_style', {}).get('palette', {})
            d.polygon([(int(x), int(y)) for (x, y) in poly],
                      fill=_pal.get('body', '#3D3D3D'),
                      outline=_pal.get('outline', '#222222'))

        drawable = [(n, b) for n, b in rig.items()
                    if not n.startswith('_') and b.get('mesh') is not None]
        for name, b in sorted(drawable, key=lambda kv: kv[1]['layer']):
            if name not in world:
                continue
            poly = to_world(b['mesh'], world[name])
            if b['outline']:
                d.polygon(poly, fill=b['color'], outline=b['outline'])
            else:
                d.polygon(poly, fill=b['color'])

        if 'head' in world:
            hw = world['head']; hl = rig['head']['length']
            c, s = np.cos(hw['angle']), np.sin(hw['angle'])
            def hp(x, y):
                return np.array([x*c - y*s + hw['pos'][0], x*s + y*c + hw['pos'][1]])
            hh = rig['_style']['h_head'] if '_style' in rig else hl*0.48
            _pal2 = rig.get('_style', {}).get('palette', {})
            mz = hp(hl*0.98, hh*0.125); mr = hh*0.54
            d.ellipse([int(mz[0]-mr), int(mz[1]-mr*0.68), int(mz[0]+mr), int(mz[1]+mr*0.68)],
                      fill=_pal2.get('muzzle', '#5A5A5A'), outline='#333333', width=1)
            nsp = hp(hl*1.16, hh*0.083)
            d.ellipse([int(nsp[0])-4, int(nsp[1])-3, int(nsp[0])+4, int(nsp[1])+3], fill='#1A1A1A')
            eye = hp(hl*0.58, -hh*0.33)
            _er = max(3, int(hh*0.217))
            disc(eye, _er, 'white', '#222222', 1); disc(eye, max(2, int(_er*0.4)), '#1A1A1A', None, 0)

        for name, (r, fill, outline, wdt) in JOINT_STYLE.items():
            if name in world:
                disc(world[name]['pos'], ref*r, fill, outline, wdt)

        if facing < 0:
            canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        return canvas
    return (render_rig,)


@app.cell
def _(
    ELBOW_LIMIT,
    STIFLE_LIMIT,
    aline_info,
    calibrate_spine_fold,
    compute_world_transforms,
    np,
    plt,
    render_rig,
    rig,
    root_y_series,
    solve_pose,
    stab_kps,
    wrap_angle,
):
    spine_fold_ref = calibrate_spine_fold(stab_kps, rig)

    def acceptance_checks():
        n = len(stab_kps)
        wl = {k: [] for k in ['humerus_L', 'humerus_R', 'femur_L', 'femur_R']}
        rel_elbow = {'L': [], 'R': []}
        rel_stifle = {'L': [], 'R': []}
        for i in range(n):
            p = solve_pose(stab_kps[i], rig, i, root_y=root_y_series[i] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            for k in wl:
                wl[k].append(p['world_leg'][k])
            for s in ('L', 'R'):
                rel_elbow[s].append(wrap_angle(p['local'][f'radius_{s}']))
                rel_stifle[s].append(wrap_angle(p['local'][f'tibia_{s}']))

        def lag(a, b):
            a = np.unwrap(np.array(a)); a -= a.mean()
            b = np.unwrap(np.array(b)); b -= b.mean()
            return int(np.argmax(np.correlate(b, a, mode='full')) - (len(a) - 1))
        LF = wl['humerus_L']
        print("=== spec #6  PHASE (reference: RF +9, LH -25, RH -24) ===")
        print(f"  RF (humerus_R) lag = {lag(LF, wl['humerus_R']):+3d}")
        print(f"  LH (femur_L)   lag = {lag(LF, wl['femur_L']):+3d}")
        print(f"  RH (femur_R)   lag = {lag(LF, wl['femur_R']):+3d}")

        def sat(vals, lim, name):
            v = np.array(vals); lo, hi = lim
            near = np.mean((v < lo + 0.05) | (v > hi - 0.05)) * 100
            print(f"  {name:10} range=[{v.min():+.2f},{v.max():+.2f}] "
                  f"limit=[{lo:+.2f},{hi:+.2f}]  %at-limit={near:4.1f}%")
        print("=== spec #8  JOINT LIMIT SATURATION (want %at-limit low) ===")
        for s in ('L', 'R'):
            sat(rel_elbow[s], ELBOW_LIMIT, f'elbow_{s}')
        for s in ('L', 'R'):
            sat(rel_stifle[s], STIFLE_LIMIT, f'stifle_{s}')

        print("=== motion continuity (jerky = |dangle|>8deg/frame) ===")
        for k in wl:
            v = np.abs(np.diff(np.degrees(np.unwrap(np.array(wl[k])))))
            print(f"  {k:10} jerky={int((v>8).sum()):3d}  max={v.max():5.1f}")

    def preview(idx=0):
        p = solve_pose(stab_kps[idx], rig, idx, root_y=root_y_series[idx] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
        w = compute_world_transforms(rig, p)

        img = render_rig(rig, w, pose=p, facing=p['facing'])
        plt.figure(figsize=(9, 7)); plt.imshow(img); plt.axis('off')
        plt.title(f'rig v2 - frame {idx}')
        plt.savefig(f'v2_frame_{idx}.png', dpi=150, bbox_inches='tight'); plt.show()


    acceptance_checks()
    preview(270)
    return (spine_fold_ref,)


@app.cell
def _(
    aline_info,
    compute_world_transforms,
    cv2,
    np,
    raw_kps,
    render_rig,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def generate_video(vid_path='dogvideo.mp4', out_path='rig_v2.mp4',
                       fps=30, panel_h=400, anim_w=600):
        SKEL = [(0,2),(1,2),(2,3),(3,5),(3,8),(5,6),(6,7),(8,9),(9,10),
                (3,4),(4,11),(4,14),(11,12),(12,13),(14,15),(15,16)]
        cap = cv2.VideoCapture(vid_path)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = panel_h / oh; pw = int(ow * scale)
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (pw*2 + anim_w, panel_h))
        written = 0
        for i, fr in enumerate(stab_kps):
            ok, orig = cap.read()
            if not ok:
                break
            p1 = cv2.resize(orig, (pw, panel_h)); p2 = p1.copy()
            kps = raw_kps[i]['keypoints']; scs = raw_kps[i]['scores']
            for a, b in SKEL:
                if scs[a] > 0.3 and scs[b] > 0.3:
                    cv2.line(p2, (int(kps[a][0]*scale), int(kps[a][1]*scale)),
                             (int(kps[b][0]*scale), int(kps[b][1]*scale)), (0,0,255), 2)
            for kp, sco in zip(kps, scs):
                if sco > 0.3:
                    cv2.circle(p2, (int(kp[0]*scale), int(kp[1]*scale)), 3, (0,255,0), -1)
            pose = solve_pose(fr, rig, i, root_y=root_y_series[i] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            if pose is None:
                p3 = np.full((panel_h, anim_w, 3), 240, np.uint8)
            else:
                w = compute_world_transforms(rig, pose)

                img = render_rig(rig, w, pose=pose, facing=pose['facing'], canvas_size=(anim_w, panel_h))
                p3 = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
            cv2.putText(p1, 'Original', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(p2, 'Tracked skeleton', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(p3, 'Rigged 2D character', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,50), 2)
            out.write(np.hstack([p1, p2, p3])); written += 1
            if i % 50 == 0:
                print(f"  frame {i}/{len(stab_kps)}")
        cap.release(); out.release()
        print(f"Saved {written} frames to {out_path}")

    generate_video()
    generate_video(out_path='rig_v7.mp4')
    return


@app.cell
def _(
    aline_info,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    p = solve_pose(stab_kps[60], rig, 60, root_y=root_y_series[60] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
    print("arch:", p['arch'])
    print("spine1-4 local angles (deg):", [round(np.degrees(p['local'][f'spine{i}']),1) for i in range(1,5)])
    return


@app.cell
def _(
    aline_info,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    p60 = solve_pose(stab_kps[60], rig, 60, root_y=root_y_series[60] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)

    verts = p60['spine_verts'].copy()
    print("facing:", p60['facing'])
    if p60['facing'] < 0:
        verts[:, 0] = 2.0 * 300.0 - verts[:, 0]
    print("verts after flip:")
    for i, v in enumerate(verts):
        print(f"  vert{i}: ({v[0]:.1f}, {v[1]:.1f})")

    meta = rig['_spine_meta']
    print("meta:", meta)

    w_up = np.linspace(meta['w_up'], meta['w_up_end'], len(verts))
    w_dn = np.linspace(meta['w_dn'], meta['w_dn_end'], len(verts))
    print("w_up:", w_up)
    print("w_dn:", w_dn)

    dirs = np.zeros_like(verts)
    for i in range(len(verts)):
        if i == 0:
            d = verts[1] - verts[0]
        elif i == len(verts) - 1:
            d = verts[-1] - verts[-2]
        else:
            d = verts[i+1] - verts[i-1]
        nrm = np.linalg.norm(d)
        dirs[i] = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0])
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)

    top = [verts[i] - normals[i] * w_up[i] for i in range(len(verts))]
    bot = [verts[i] + normals[i] * w_dn[i] for i in range(len(verts))]
    print("top verts:", [(round(p[0]),round(p[1])) for p in top])
    print("bot verts:", [(round(p[0]),round(p[1])) for p in bot])
    return


@app.cell
def _(
    aline_info,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def diag_tail():
        angs = []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            angs.append(np.degrees(pp['local']['tail_1']))
        angs = np.array(angs)
        print(f"tail_1 local angle: range=[{angs.min():.0f}, {angs.max():.0f}], ptp={np.ptp(angs):.0f} deg")

    diag_tail()
    return


@app.cell
def _(
    aline_info,
    compute_world_transforms,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def diag_tail_world():
        angs = []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            ww = compute_world_transforms(rig, pp)
            angs.append(np.degrees(ww['tail_1']['angle']))
        angs = np.array(angs)
        au = np.unwrap(np.radians(angs))
        au = np.degrees(au)
        print(f"tail_1 WORLD angle: ptp(unwrapped)={np.ptp(au):.0f} deg, "
              f"raw range=[{angs.min():.0f},{angs.max():.0f}]")

    diag_tail_world()
    return


@app.cell
def _(
    aline_info,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def diag_body_tilt():
        tilts, ys = [], []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            v = pp['spine_verts']
            tilt = np.degrees(np.arctan2(v[-1][1]-v[0][1], v[-1][0]-v[0][0]))
            tilts.append(tilt)
            ys.append(root_y_series[k])
        tilts = np.array(tilts); ys = np.array(ys)
        print(f"body tilt: ptp={np.ptp(tilts):.0f} deg, range=[{tilts.min():.0f},{tilts.max():.0f}]")
        print(f"root_y bounce: ptp={np.ptp(ys):.0f} px, range=[{ys.min():.0f},{ys.max():.0f}]")

    diag_body_tilt()
    return


@app.cell
def _(
    aline_info,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def diag_arch_dynamics():
        arches = []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            arches.append(pp['arch'])
        a = np.array(arches)
        print(f"arch: mean={a.mean():.3f}, range=[{a.min():.3f},{a.max():.3f}], ptp={np.ptp(a):.3f}")
        print(f"frames with arch>0.2 (strongly arched): {int((a>0.2).sum())}/{len(a)}")
        print(f"frames with arch<0.05 (nearly flat): {int((a<0.05).sum())}/{len(a)}")

    diag_arch_dynamics()
    return


@app.cell
def _():
    return


@app.cell
def _(
    ARCH_LEAD_FRAMES,
    SPINE_ARCH_GAIN,
    SPINE_REST_FOLD,
    facing_series,
    get_kp,
    np,
    spine_mid_track,
    stab_kps,
):
    def arch_audit():
        mid = np.asarray(spine_mid_track, dtype=float)
        if ARCH_LEAD_FRAMES:
            n_sh = int(ARCH_LEAD_FRAMES)
            if n_sh > 0:
                mid = np.concatenate([mid[n_sh:],
                                      np.repeat(mid[-1:], n_sh, axis=0)], axis=0)
                print(f"(auditing the track advanced by {n_sh} frames)")
            else:
                k_sh_a = -n_sh
                mid = np.concatenate([np.repeat(mid[:1], k_sh_a, axis=0),
                                      mid[:-k_sh_a]], axis=0)
                print(f"(auditing the track delayed by {k_sh_a} frames)")

        root_ref = get_kp(stab_kps[0], 'root_of_tail')

        def rs(q, facing):
            q = np.asarray(q, dtype=float)
            return np.array([2 * root_ref[0] - q[0], q[1]]) if facing < 0 else q

        arch = np.zeros(len(stab_kps))
        for i, fr in enumerate(stab_kps):
            f = float(facing_series[i])
            neck = rs(get_kp(fr, 'Neck'), f)
            tail = rs(get_kp(fr, 'root_of_tail'), f)
            mid_i = rs(mid[i], f)
            chord = tail - neck
            cl = float(np.linalg.norm(chord))
            if cl < 1e-6:
                continue
            cd = chord / cl
            cn = np.array([-cd[1], cd[0]])
            arch[i] = float((mid_i - neck) @ cn) / (cl * 0.5)

        print(f"facing: {np.sum(np.asarray(facing_series) < 0)}/{len(facing_series)}"
              f" frames flipped (dog runs left)")
        print(f"\narch in solve_pose's space: min {arch.min():+.3f}  "
              f"max {arch.max():+.3f}  mean {arch.mean():+.3f}")

        k = int(np.argmax(np.abs(arch)))
        f = float(facing_series[k])
        neck = rs(get_kp(stab_kps[k], 'Neck'), f)
        tail = rs(get_kp(stab_kps[k], 'root_of_tail'), f)
        mid_k = rs(mid[k], f)
        chord_y = neck[1] + (tail[1] - neck[1]) * 0.5
        above = mid_k[1] < chord_y
        sign_of_up = np.sign(arch[k]) if above else -np.sign(arch[k])
        print(f"  at frame {k} (|arch| largest) the mid-back is "
              f"{'ABOVE' if above else 'BELOW'} the chord and arch = {arch[k]:+.3f}")
        print(f"  => arch {'>' if sign_of_up > 0 else '<'} 0 means ARCHED UP "
              f"(roach); the opposite sign means SAGGING (sway)")

        tb = SPINE_REST_FOLD + arch * SPINE_ARCH_GAIN
        print(f"\ntotal_bend = {SPINE_REST_FOLD:+.2f} + arch * {SPINE_ARCH_GAIN}"
              f"  ->  min {tb.min():+.2f}  max {tb.max():+.2f}  mean {tb.mean():+.2f}")
        cross = np.mean(np.sign(tb) != np.sign(tb.mean()))
        print(f"  {cross:.0%} of frames fall on the opposite side of straight "
              f"from the average pose")

        print("\nframes 160-190 (the passage reported as sagging when it should arch):")
        for i in range(160, min(191, len(arch)), 3):
            bar = '#' * int(abs(arch[i]) * 40)
            print(f"   {i:3}: arch {arch[i]:+.3f}  total_bend {tb[i]:+.2f}  {bar}")

        print("\nlast 12 frames (the frame reported as over-curved):")
        tail_span = arch[-12:]
        if float(np.ptp(tail_span)) < 0.06:
            print("   NOTE: arch barely moves across these frames. Annotation")
            print("   keyframes are 15 frames apart, so the clip's final frames")
            print("   sit at or beyond the last one and are held rather than")
            print("   observed. Treat the ending as weakly constrained.")
        for i in range(max(0, len(arch) - 12), len(arch)):
            bar = '#' * int(abs(arch[i]) * 40)
            print(f"   {i:3}: arch {arch[i]:+.3f}  total_bend {tb[i]:+.2f}  {bar}")
        return arch

    arch_audited = arch_audit()
    return (arch_audited,)


@app.cell
def _(arch_audited, facing_series, get_kp, np, stab_kps):
    def timing_audit():
        root_ref0 = get_kp(stab_kps[0], 'root_of_tail')

        def rs_x(q, facing):
            return (2 * root_ref0[0] - q[0]) if facing < 0 else q[0]

        spread = np.full(len(stab_kps), np.nan)
        for i, fr in enumerate(stab_kps):
            f = float(facing_series[i])
            vals = []
            for a, b in (('L_F_Paw', 'L_B_Paw'), ('R_F_Paw', 'R_B_Paw')):
                vals.append(rs_x(get_kp(fr, a), f) - rs_x(get_kp(fr, b), f))
            spread[i] = float(np.mean(vals))
        k = np.ones(5) / 5.0
        spread = np.convolve(spread, k, mode='same')
        gather = -spread

        a = np.asarray(arch_audited, dtype=float)
        lo, hi = 20, len(a) - 20
        x = gather[lo:hi] - np.nanmean(gather[lo:hi])
        y = a[lo:hi] - a[lo:hi].mean()

        best_lag, best_r = 0, 0.0
        for lag in range(-36, 37):
            if lag >= 0:
                u, v = x[:len(x) - lag], y[lag:]
            else:
                u, v = x[-lag:], y[:len(y) + lag]
            if u.size < 40:
                continue
            d = np.sqrt(np.nansum(u * u) * np.nansum(v * v))
            r = float(np.nansum(u * v) / d) if d > 1e-12 else 0.0
            if r > best_r:
                best_r, best_lag = r, lag

        g = gather[lo:hi]
        cross = int(np.sum(np.diff(np.sign(g - np.nanmean(g))) != 0))
        cycles = cross / 2.0
        print(f"gather signal: {cycles:.1f} cycles over {hi-lo} frames "
              f"= {(hi-lo)/max(cycles,1e-6):.0f} frames per cycle")
        print(f"  (stride measured elsewhere is ~72 frames; roughly half that")
        print(f"   would mean the metric is still frequency-doubled)")

        print(f"\ngathering vs arch: peak correlation {best_r:+.2f} at lag "
              f"{best_lag:+d} frames")
        print("  lag > 0 : the arch happens LATER than the gather")
        print("  lag < 0 : the arch happens EARLIER than the gather")
        if best_r < 0.35:
            print("\n  Correlation is weak. The two are mechanically linked in a")
            print("  gallop, so a weak result points at the annotation rather")
            print("  than at a timing offset -- shifting it would not help.")
        else:
            print(f"\n  => set ARCH_LEAD_FRAMES = {best_lag:+d} to align them.")
            print("     (positive makes the spine act earlier, which cancels a")
            print("      positive lag).")

        print(f"\narch percentiles (for deciding whether the clip end is an outlier):")
        for q in (2, 5, 25, 50, 75, 95, 98):
            print(f"   p{q:<3} {np.percentile(a, q):+.3f}")
        print(f"   max  {a.max():+.3f}  <- reached only in the extrapolated tail")
        return best_lag, best_r

    timing_result = timing_audit()
    return


@app.cell
def _(
    aline_info,
    compute_world_transforms,
    leg_table,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    def twitch_locator():
        names = ['humerus_L', 'radius_L', 'paw_front_L',
                 'humerus_R', 'radius_R', 'paw_front_R',
                 'femur_L', 'tibia_L', 'femur_R', 'tibia_R',
                 'spine1', 'spine4', 'neck']
        ang = {n: [] for n in names}
        pos = {n: [] for n in names}
        ok = []
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, i,
                              root_y=root_y_series[i] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            if pose is None:
                ok.append(False)
                for n in names:
                    ang[n].append(np.nan); pos[n].append([np.nan, np.nan])
                continue
            ok.append(True)
            w = compute_world_transforms(rig, pose)
            for n in names:
                if n in w:
                    ang[n].append(float(w[n]['angle']))
                    pos[n].append(np.asarray(w[n]['pos'], dtype=float))
                else:
                    ang[n].append(np.nan); pos[n].append([np.nan, np.nan])

        n_fail = int(len(ok) - sum(ok))
        print(f"frames solved: {sum(ok)}/{len(ok)}"
              + (f"   <- {n_fail} FAILED, solve_pose returned None" if n_fail else ""))
        if n_fail:
            bad = [i for i, o in enumerate(ok) if not o]
            print(f"   failed frames: {bad[:20]}")
            print("   A failed frame is skipped by the renderer, so the character")
            print("   holds its previous pose and then snaps -- that alone would")
            print("   look exactly like a dropped frame.")

        print(f"\n{'bone':14} {'ang/frame':>10} {'worst':>7} {'pos/frame':>10} {'worst':>7}")
        print("-" * 54)
        report = {}
        for n in names:
            a = np.unwrap(np.array(ang[n], dtype=float))
            da = np.abs(np.diff(a)) * 180 / np.pi
            p = np.array(pos[n], dtype=float)
            dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
            report[n] = (da, dp)
            print(f"{n:14} {np.nanmedian(da):9.2f}d {np.nanmax(da):6.1f}d "
                  f"{np.nanmedian(dp):9.2f}px {np.nanmax(dp):6.1f}px")

        print("\nframes where any front-leg bone moves far beyond its own norm:")
        flagged = {}
        for n in ('humerus_L', 'radius_L', 'humerus_R', 'radius_R'):
            da, dp = report[n]
            for arr, unit, thr in ((da, 'deg', 6.0), (dp, 'px', 6.0)):
                med = np.nanmedian(arr)
                mad = np.nanmedian(np.abs(arr - med)) + 1e-9
                for f in np.where(arr > med + thr * mad)[0]:
                    flagged.setdefault(int(f) + 1, []).append(
                        f"{n} {arr[f]:.1f}{unit}")
        if not flagged:
            print("   none -- the front legs are smooth in the solved output, so")
            print("   the twitch is being introduced after solving (rendering,")
            print("   layer order, or the paw mesh) rather than in the pose.")
        else:
            for f in sorted(flagged)[:25]:
                print(f"   frame {f:3}: " + ", ".join(flagged[f]))
            print(f"   ({len(flagged)} frames flagged in total)")

        print("\njoint limit contact (clamp active = motion flattened):")
        ELBOW = (-0.35, 1.60); STIFLE = (-2.00, 0.25)
        for key, lim, lbl in (('rad_L_rel', ELBOW, 'elbow L'),
                              ('rad_R_rel', ELBOW, 'elbow R'),
                              ('tib_L_rel', STIFLE, 'stifle L'),
                              ('tib_R_rel', STIFLE, 'stifle R')):
            if key not in leg_table[0]:
                print(f"   {lbl:9} (no {key} in leg_table)")
                continue
            v = np.array([leg_table[i][key] for i in range(len(stab_kps))])
            over = np.mean((v < lim[0]) | (v > lim[1]))
            print(f"   {lbl:9} range {v.min():+.2f}..{v.max():+.2f}  "
                  f"limits {lim[0]:+.2f}..{lim[1]:+.2f}  "
                  f"{over:.0%} of frames outside")
        return report

    twitch_report = twitch_locator()
    return


@app.cell
def _(KEYPOINT_NAMES, leg_table, np, stab_kps):
    def twitch_cause():
        LEGS = {
            'hum_L': ('L_Shoulder', 'L_Elbow'), 'rad_L': ('L_Elbow', 'L_F_Paw'),
            'hum_R': ('R_Shoulder', 'R_Elbow'), 'rad_R': ('R_Elbow', 'R_F_Paw'),
            'fem_L': ('L_Hip', 'L_Knee'),       'tib_L': ('L_Knee', 'L_B_Paw'),
            'fem_R': ('R_Hip', 'R_Knee'),       'tib_R': ('R_Knee', 'R_B_Paw'),
        }
        CONF = 0.35
        ELBOW = (-0.35, 1.60); STIFLE = (-2.00, 0.25)
        n = len(stab_kps)

        conf = {}
        for k, (a, b) in LEGS.items():
            ia, ib = KEYPOINT_NAMES.index(a), KEYPOINT_NAMES.index(b)
            conf[k] = np.array([min(stab_kps[i]['scores'][ia],
                                    stab_kps[i]['scores'][ib]) for i in range(n)])

        print(f"{'bone':7} {'accel med':>10} {'accel max':>10} {'ratio':>7} "
              f"{'lowconf':>8}")
        print("-" * 48)
        acc = {}
        for k in LEGS:
            a = np.unwrap(np.array([leg_table[i][k] for i in range(n)]))
            d2 = np.abs(np.diff(a, 2)) * 180 / np.pi
            acc[k] = d2
            med = float(np.median(d2))
            print(f"{k:7} {med:9.2f}d {d2.max():9.2f}d "
                  f"{d2.max()/max(med,1e-6):6.0f}x {int((conf[k] < CONF).sum()):7}")

        print("\nworst 12 acceleration events, with cause:")
        events = []
        for k in LEGS:
            d2 = acc[k]
            med = float(np.median(d2))
            mad = float(np.median(np.abs(d2 - med))) + 1e-9
            for f in np.where(d2 > med + 8 * mad)[0]:
                events.append((float(d2[f]), k, int(f) + 1))
        events.sort(reverse=True)

        if not events:
            print("   none above threshold -- the angle tracks are smooth, so")
            print("   the visible twitch is coming from position, not rotation.")
        for mag, k, f in events[:12]:
            causes = []
            lo = max(0, f - 2); hi = min(n, f + 3)
            if (conf[k][lo:hi] < CONF).any():
                causes.append(f"LOW CONF (min {conf[k][lo:hi].min():.2f})")
            rel_key = k.replace('rad', 'rad').replace('tib', 'tib') + '_rel'
            if rel_key in leg_table[0]:
                lim = ELBOW if 'rad' in k else STIFLE
                v = np.array([leg_table[i][rel_key] for i in range(lo, hi)])
                if ((v < lim[0]) | (v > lim[1])).any():
                    causes.append(f"AT LIMIT ({v.min():+.2f}..{v.max():+.2f})")
            print(f"   frame {f:3} {k:7} {mag:6.1f} deg/frm^2   "
                  + (" + ".join(causes) if causes else "neither -- tracked data"))

        print(f"\n{len(events)} events above threshold in total")
        print("\nreading it:")
        print("  mostly LOW CONF  -> raise smoothing on that leg, or widen CONF_THR")
        print("  mostly AT LIMIT  -> the limits are too tight for this gait")
        print("  mostly neither   -> tracked data is spiky; smooth harder")
        return acc, conf

    twitch_cause_report = twitch_cause()
    return


@app.cell
def _(KEYPOINT_NAMES, leg_table, np, stab_kps):
    def overlap_limit_audit():
        n = len(stab_kps)
        ELBOW = (-0.35, 1.60); STIFLE = (-2.00, 0.25)

        def kp(i, name):
            return np.asarray(stab_kps[i]['keypoints'][KEYPOINT_NAMES.index(name)],
                              dtype=float)

        spine = np.array([np.linalg.norm(kp(i, 'root_of_tail') - kp(i, 'Neck'))
                          for i in range(n)])
        spine[spine < 1e-6] = np.nan

        print("A. front/hind paw proximity (same side), as a fraction of spine:")
        overlap = {}
        for side, fp, bp in (('L', 'L_F_Paw', 'L_B_Paw'), ('R', 'R_F_Paw', 'R_B_Paw')):
            d = np.array([np.linalg.norm(kp(i, fp) - kp(i, bp)) for i in range(n)]) / spine
            close = d < 0.10
            overlap[side] = close
            print(f"   {side}: min {np.nanmin(d):.3f}  "
                  f"frames below 0.10 = {int(close.sum())}  "
                  f"below 0.06 = {int((d < 0.06).sum())}")

        print("\nB. joint-limit excursion:")
        outside = {}
        for key, lim, lbl in (('rad_L_rel', ELBOW, 'elbow L'), ('rad_R_rel', ELBOW, 'elbow R'),
                              ('tib_L_rel', STIFLE, 'stifle L'), ('tib_R_rel', STIFLE, 'stifle R')):
            if key not in leg_table[0]:
                continue
            v = np.array([leg_table[i][key] for i in range(n)])
            o = (v < lim[0]) | (v > lim[1])
            outside[key] = o
            print(f"   {lbl:9} range {v.min():+.2f}..{v.max():+.2f}  "
                  f"limit {lim[0]:+.2f}..{lim[1]:+.2f}  outside {o.mean():.0%}  "
                  f"suggested {np.percentile(v, 0.5):+.2f}..{np.percentile(v, 99.5):+.2f}")

        print("\nDo the biggest accelerations land on those frames?")
        for key in ('hum_L', 'rad_L', 'hum_R', 'rad_R', 'fem_L', 'tib_L', 'fem_R', 'tib_R'):
            a = np.unwrap(np.array([leg_table[i][key] for i in range(n)]))
            d2 = np.abs(np.diff(a, 2)) * 180 / np.pi
            med = float(np.median(d2)); mad = float(np.median(np.abs(d2 - med))) + 1e-9
            bad = np.where(d2 > med + 8 * mad)[0] + 1
            if bad.size == 0:
                continue
            side = 'L' if key.endswith('_L') else 'R'
            rel = key.replace('rad', 'rad').replace('tib', 'tib') + '_rel'
            n_ov = sum(1 for f in bad if f < n and overlap[side][max(0, f - 2):f + 3].any())
            n_lim = 0
            if rel in outside:
                n_lim = sum(1 for f in bad if f < n and outside[rel][max(0, f - 2):f + 3].any())
            print(f"   {key:6} {bad.size:3} spikes -> {n_ov:3} near a paw overlap, "
                  f"{n_lim:3} near a limit")
        return overlap, outside

    overlap_audit = overlap_limit_audit()
    return


@app.cell
def _(np, root_y_series):
    _r = np.asarray(root_y_series, dtype=float)
    _cap = 18.0
    _n_at = int(np.sum(np.abs(np.abs(_r) - _cap) < 0.01))
    cap_check = "frames sitting exactly at the +-%.0fpx cap: %d / %d (%.0f%%) range %.1f..%.1f" % (_cap, _n_at, len(_r), 100.0 * _n_at / len(_r), _r.min(), _r.max())
    print(cap_check)
    cap_check
    return


@app.cell
def _(KEYPOINT_NAMES, facing_series, leg_table, np, stab_kps):
    _n = len(stab_kps)
    _KI = {nm: i for i, nm in enumerate(KEYPOINT_NAMES)}

    _aliases = {
        "L_Shoulder": ["L_Shoulder", "L_Front_Shoulder", "L_F_Shoulder", "Shoulder_L"],
        "L_Elbow": ["L_Elbow", "L_Front_Elbow", "L_F_Elbow", "Elbow_L"],
        "R_Shoulder": ["R_Shoulder", "R_Front_Shoulder", "R_F_Shoulder", "Shoulder_R"],
        "R_Elbow": ["R_Elbow", "R_Front_Elbow", "R_F_Elbow", "Elbow_R"],
        "L_Hip": ["L_Hip", "L_Hind_Hip", "L_B_Hip", "Hip_L"],
        "L_Knee": ["L_Knee", "L_Stifle", "L_B_Knee", "Stifle_L"],
        "R_Hip": ["R_Hip", "R_Hind_Hip", "R_B_Hip", "Hip_R"],
        "R_Knee": ["R_Knee", "R_Stifle", "R_B_Knee", "Stifle_R"],
        "L_F_Paw": ["L_F_Paw", "L_Front_Paw", "Paw_FL"],
        "R_F_Paw": ["R_F_Paw", "R_Front_Paw", "Paw_FR"],
        "L_B_Paw": ["L_B_Paw", "L_Hind_Paw", "L_Back_Paw", "Paw_HL"],
        "R_B_Paw": ["R_B_Paw", "R_Hind_Paw", "R_Back_Paw", "Paw_HR"],
    }

    def _get_kp_index(name):
        if name in _KI:
            return _KI[name]
        if name in _aliases:
            for alias in _aliases[name]:
                if alias in _KI:
                    return _KI[alias]
        raise KeyError(f"Keypoint '{name}' not found in KEYPOINT_NAMES")

    _P = np.array(
        [[fr["keypoints"][_KI[nm]] for nm in KEYPOINT_NAMES] for fr in stab_kps],
        dtype=float,
    )

    _DEF = {
        "hum_L": ("L_Shoulder", "L_Elbow"),
        "rad_L": ("L_Elbow", "L_F_Paw"),
        "hum_R": ("R_Shoulder", "R_Elbow"),
        "rad_R": ("R_Elbow", "R_F_Paw"),
        "fem_L": ("L_Hip", "L_Knee"),
        "tib_L": ("L_Knee", "L_B_Paw"),
        "fem_R": ("R_Hip", "R_Knee"),
        "tib_R": ("R_Knee", "R_B_Paw"),
    }

    _lines = ["FIDELITY: character motion vs raw tracked motion", ""]
    _lines.append(
        "  %-7s %9s %9s %8s %7s %7s"
        % ("bone", "raw range", "rig range", "kept", "corr", "lag")
    )

    _keep = []
    for _k in ("hum_L", "rad_L", "hum_R", "rad_R", "fem_L", "tib_L", "fem_R", "tib_R"):
        if _k not in leg_table[0]:
            continue

        _a_name, _b_name = _DEF[_k]
        _a_idx = _get_kp_index(_a_name)
        _b_idx = _get_kp_index(_b_name)

        _d = _P[:, _b_idx] - _P[:, _a_idx]
        _raw = np.arctan2(_d[:, 1], _d[:, 0])

        _fl = np.asarray(facing_series, dtype=float) < 0
        _raw = np.where(_fl, np.pi - _raw, _raw)
        _raw = np.unwrap(_raw)

        _rig = np.unwrap(np.array([leg_table[_i][_k] for _i in range(_n)]))

        _r0 = _raw - _raw.mean()
        _r1 = _rig - _rig.mean()

        _kept = 100.0 * np.ptp(_rig) / max(np.ptp(_raw), 1e-9)
        _keep.append(_kept)

        _best_r = 0.0
        _best_l = 0

        for _lg in range(-10, 11):
            if _lg >= 0:
                _x, _y = _r0[: _n - _lg], _r1[_lg:]
            else:
                _x, _y = _r0[-_lg:], _r1[: _n + _lg]

            _dn = np.sqrt((_x * _x).sum() * (_y * _y).sum())
            _rr = float((_x * _y).sum() / _dn) if _dn > 1e-12 else 0.0

            if _rr > _best_r:
                _best_r, _best_l = _rr, _lg

        _lines.append(
            "  %-7s %8.1fd %8.1fd %7.0f%% %7.2f %+6d"
            % (
                _k,
                np.degrees(np.ptp(_raw)),
                np.degrees(np.ptp(_rig)),
                _kept,
                _best_r,
                _best_l,
            )
        )

    _lines.append("")
    _lines.append(("  mean amplitude kept: %.0f%%" % np.mean(_keep)) if _keep else "  mean amplitude kept: 0%")
    _lines.append("")
    _lines.append("  reading it:")
    _lines.append("    kept  = how much of the real swing survived filtering (100% = none lost)")
    _lines.append("    corr  = shape agreement with the tracked motion (1.00 = identical)")
    _lines.append("    lag   = frames the rig trails the data (0 = in step)")

    fidelity_report = "\n".join(_lines)
    print(fidelity_report)
    return


@app.cell
def _(compute_world_transforms, np, rig, solve_pose, stab_kps):
    neck_gap_jd, tail_gap_jd, sh_local_jd = [], [], []
    for i_jd in range(len(stab_kps)):
        p_jd = solve_pose(stab_kps[i_jd], rig, i_jd)
        if p_jd is None:
            continue
        w_jd = compute_world_transforms(rig, p_jd)
        sv_jd = np.asarray(p_jd['spine_verts'])
        neck_gap_jd.append(float(np.linalg.norm(w_jd['neck']['pos'] - sv_jd[4])))
        tail_gap_jd.append(float(np.linalg.norm(w_jd['tail_1']['pos'] - sv_jd[0])))
        a_jd = w_jd['spine4']['angle']
        c_jd, s_jd = np.cos(-a_jd), np.sin(-a_jd)
        d_jd = np.asarray(p_jd['anchors']['shoulder']) - sv_jd[3]
        sh_local_jd.append((d_jd[0] * c_jd - d_jd[1] * s_jd,
                            d_jd[0] * s_jd + d_jd[1] * c_jd))
    sh_local_jd = np.asarray(sh_local_jd)
    for nm_jd, g_jd in (('neck-to-ribbon-front', np.asarray(neck_gap_jd)),
                        ('tailroot-to-rump-vertex', np.asarray(tail_gap_jd))):
        print(f"{nm_jd:26} mean {g_jd.mean():6.2f}px  std {g_jd.std():5.2f}  "
              f"p2p {g_jd.max() - g_jd.min():5.2f}px")
    print(f"{'shoulder anchor (local)':26} std x {sh_local_jd[:, 0].std():5.2f}  "
          f"std y {sh_local_jd[:, 1].std():5.2f}px  "
          f"(constant = welded; compare against the pre-fix file)")
    return


@app.cell
def _(arch_audit_raw, arch_trust_series, np, rig, solve_pose, stab_kps):
    bends_aa, heights_aa, norms_aa, scales_aa = [], [], [], []
    for i_aa in range(len(stab_kps)):
        p_aa = solve_pose(stab_kps[i_aa], rig, i_aa)
        if p_aa is None:
            continue
        sv_aa = np.asarray(p_aa['spine_verts'])
        chord_aa = sv_aa[4] - sv_aa[0]
        cl_aa = float(np.linalg.norm(chord_aa))
        if cl_aa < 1e-6:
            continue
        cn_aa = np.array([-chord_aa[1], chord_aa[0]]) / cl_aa
        heights_aa.append(float((sv_aa[2] - sv_aa[0]) @ cn_aa))
        seg_aa = np.diff(sv_aa, axis=0)
        ang_aa = np.arctan2(seg_aa[:, 1], seg_aa[:, 0])
        bends_aa.append(float(np.sum(np.abs(np.diff(np.unwrap(ang_aa))))))
        norms_aa.append(float(p_aa.get('bend_norm', 0.0)))
        scales_aa.append(float(p_aa['seg_scale']))
    bends_aa = np.asarray(bends_aa)
    heights_aa = np.asarray(heights_aa)
    norms_aa = np.asarray(norms_aa)
    scales_aa = np.asarray(scales_aa)
    k_gather = int(np.argmax(norms_aa))
    k_extend = int(np.argmin(norms_aa))
    print(f"combined spine turn: {np.degrees(bends_aa.min()):5.1f}"
          f"..{np.degrees(bends_aa.max()):5.1f} deg "
          f"(median {np.degrees(np.median(bends_aa)):.1f})")
    print(f"drawn arc height   : {np.abs(heights_aa).min():5.1f}"
          f"..{np.abs(heights_aa).max():5.1f} px; "
          f"at most-gathered frame {k_gather} = {heights_aa[k_gather]:+.1f}px "
          f"(seg_scale {scales_aa[k_gather]:.3f}), "
          f"at most-extended frame {k_extend} = {heights_aa[k_extend]:+.1f}px")
    sat_aa = int((np.abs(norms_aa) > 0.999).sum())
    print(f"bend_norm (dome driver): {norms_aa.min():+.2f}..{norms_aa.max():+.2f}; "
          f"saturating on {sat_aa}/{len(norms_aa)} frames "
          f"({100 * sat_aa / len(norms_aa):.1f}%)")
    if sat_aa > 0.10 * len(norms_aa):
        print("   !! more than 10% of frames sit at the ceiling -- the dome is "
              "flat-topping and will read as one rigid roach; lower "
              "CHORD_BEND_WEIGHT rather than raising ARCH_DOME")
    else:
        print("   (below 10% -- no flat-topping; do NOT raise CHORD_BEND_WEIGHT "
              "further, the driver is already reaching its ceiling)")

    obs_aa = arch_trust_series[:len(heights_aa)] > 0.999
    ann_aa = np.asarray(arch_audit_raw)[:len(heights_aa)]
    ren_aa = -heights_aa
    if obs_aa.sum() > 30:
        x_aa = ren_aa[obs_aa] - ren_aa[obs_aa].mean()
        y_aa = ann_aa[obs_aa] - ann_aa[obs_aa].mean()
        lags_aa = range(-15, 16)
        best_aa = max(lags_aa, key=lambda L: float(
            np.corrcoef(np.roll(x_aa, L), y_aa)[0, 1]))
        r0_aa = float(np.corrcoef(x_aa, y_aa)[0, 1])
        rb_aa = float(np.corrcoef(np.roll(x_aa, best_aa), y_aa)[0, 1])
        print(f"rendered arc vs annotated back shape ({int(obs_aa.sum())} observed "
              f"frames): corr {r0_aa:+.2f} at lag 0, best {rb_aa:+.2f} at lag "
              f"{best_aa:+d}")
        if abs(best_aa) >= 3 and rb_aa > r0_aa + 0.15:
            print(f"   !! the character arches {abs(best_aa)} frames "
                  f"{'early' if best_aa > 0 else 'late'}; that is a TIMING "
                  f"problem, not an amplitude one -- changing ARCH_DOME will "
                  f"not fix it. With ANNOT_ALIGN_AUTO on, check the "
                  f"alignment print above first")
        else:
            print("   (timing agrees with the annotation; any remaining "
                  "complaint is amplitude, i.e. ARCH_DOME / BELLY_TUCK)")
    else:
        print("rendered arc vs annotation: too few observed frames to compare")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(compute_world_transforms, leg_table, np, rig, solve_pose, stab_kps):
    W_la = []
    for i_la in range(len(stab_kps)):
        p_la = solve_pose(stab_kps[i_la], rig, i_la)
        if p_la is None:
            W_la.append(None)
            continue
        W_la.append(compute_world_transforms(rig, p_la))
    ok_la = [w for w in W_la if w is not None]
    names_la = [n for n in ok_la[0] if not n.startswith('_')]

    rows_la = []
    for nm_la in names_la:
        ang_la = np.unwrap(np.array([w[nm_la]['angle'] for w in ok_la]))
        d_la = np.degrees(np.abs(np.diff(ang_la)))
        if d_la.size == 0:
            continue
        worst_la = np.argsort(d_la)[-3:][::-1] + 1
        rows_la.append((float(np.percentile(d_la, 98)), float(d_la.max()),
                        nm_la, worst_la.tolist()))
    rows_la.sort(reverse=True)
    print(f"{'bone':16} {'p98 deg/f':>10} {'max':>8}  worst frames")
    for p98_la, mx_la, nm_la, wf_la in rows_la[:12]:
        print(f"{nm_la:16} {p98_la:10.2f} {mx_la:8.2f}  {wf_la}")

    print()
    for key_la, lim_la, lbl_la in (('rad_L_rel', (-0.35, 1.60), 'elbow L'),
                                   ('rad_R_rel', (-0.35, 1.60), 'elbow R'),
                                   ('tib_L_rel', (-2.00, 0.25), 'stifle L'),
                                   ('tib_R_rel', (-2.00, 0.25), 'stifle R')):
        v_la = np.array([leg_table[i][key_la] for i in range(len(stab_kps))])
        span_la = (lim_la[1] - lim_la[0]) * 0.5
        n_soft = int(np.sum((v_la > lim_la[1] - span_la) |
                            (v_la < lim_la[0] + span_la)))
        n_out = int(np.sum((v_la > lim_la[1]) | (v_la < lim_la[0])))
        print(f"{lbl_la:9}: {100 * n_soft / len(v_la):5.1f}% of frames inside the "
              f"soft zone, {100 * n_out / len(v_la):5.1f}% beyond the limit")

    for s_la, low_la in (('L', 'rad_L'), ('R', 'rad_R'),
                         ('L', 'tib_L'), ('R', 'tib_R')):
        a_la = np.array([leg_table[i][low_la] for i in range(len(stab_kps))])
        aw_la = np.arctan2(np.sin(a_la), np.cos(a_la))
        st_la = np.clip(np.cos(aw_la - np.pi / 2), 0.0, 1.0) ** 2
        print(f"stance term on {low_la}: at the upper clip on "
              f"{100 * np.mean(st_la > 0.999):4.1f}% of frames, "
              f"p98 per-frame change {np.percentile(np.abs(np.diff(st_la)), 98):.3f}")
    return


@app.cell
def _(arch_audit_raw, get_kp, np, rig, solve_pose, stab_kps):
    spread_pa = np.full(len(stab_kps), np.nan)
    for i_pa, fr_pa in enumerate(stab_kps):
        neck_pa = get_kp(fr_pa, 'Neck')
        tail_pa = get_kp(fr_pa, 'root_of_tail')
        axis_pa = tail_pa - neck_pa
        len_pa = float(np.linalg.norm(axis_pa))
        if len_pa < 1e-6:
            continue
        u_pa = axis_pa / len_pa
        proj_pa = []
        for nm_pa in ('L_F_Paw', 'R_F_Paw', 'L_B_Paw', 'R_B_Paw'):
            proj_pa.append(float((get_kp(fr_pa, nm_pa) - neck_pa) @ u_pa))
        spread_pa[i_pa] = (max(proj_pa) - min(proj_pa)) / len_pa

    bend_pa, chord_pa = [], []
    for i_pa in range(len(stab_kps)):
        p_pa = solve_pose(stab_kps[i_pa], rig, i_pa)
        bend_pa.append(np.nan if p_pa is None
                       else float(p_pa.get('bend_norm', np.nan)))
    bend_pa = np.asarray(bend_pa)

    def _phase(x_pa, y_pa, label_pa):
        ok_pa = ~np.isnan(x_pa) & ~np.isnan(y_pa)
        a_pa = x_pa[ok_pa] - x_pa[ok_pa].mean()
        b_pa = y_pa[ok_pa] - y_pa[ok_pa].mean()
        r0_pa = float(np.corrcoef(a_pa, b_pa)[0, 1])
        best_pa = min(range(-25, 26), key=lambda L:
                      float(np.corrcoef(np.roll(a_pa, L), b_pa)[0, 1]))
        rb_pa = float(np.corrcoef(np.roll(a_pa, best_pa), b_pa)[0, 1])
        print(f"{label_pa:34} corr {r0_pa:+.2f} at lag 0; most anti-phase "
              f"{rb_pa:+.2f} at lag {best_pa:+d}")
        return r0_pa, best_pa

    print("ANTI-correlation with limb spread is what a correct gallop looks like")
    print("(back roaches as the limbs gather). Lag 0 should already be clearly")
    print("negative; a large best-lag means that channel is out of phase.\n")
    _phase(bend_pa, spread_pa, "combined flexion (what is drawn)")
    _phase(np.asarray(arch_audit_raw, dtype=float), spread_pa,
           "annotation channel alone")

    v_pa = spread_pa[~np.isnan(spread_pa)] - np.nanmean(spread_pa)
    f_pa = np.fft.rfftfreq(len(v_pa))
    P_pa = np.abs(np.fft.rfft(v_pa)) ** 2
    P_pa[f_pa < 1.0 / 60] = 0.0
    per_pa = 1.0 / f_pa[int(np.argmax(P_pa))]
    print(f"\nstride period {per_pa:.1f} frames, so half a cycle is "
          f"{per_pa / 2:.0f} frames -- a best-lag near that size means the "
          f"channel is inverted rather than merely late")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

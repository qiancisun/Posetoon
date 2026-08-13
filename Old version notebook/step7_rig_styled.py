import marimo

__generated_with = "0.21.1"
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

    try:
        with open('spine_tail_tracks.json', 'r') as f:
            spine_tail = json.load(f)
        spine_mid_track = np.array(spine_tail['spine_mid'], dtype=float)
        tail_tip_track = np.array(spine_tail['tail_tip'], dtype=float)
        SPINE_TAIL_SOURCE = 'annotated'
    except FileNotFoundError:
        SPINE_TAIL_SOURCE = 'synthesised'
        _neck_t = np.array([get_kp(fr, 'Neck') for fr in stab_kps])
        _tailr_t = np.array([get_kp(fr, 'root_of_tail') for fr in stab_kps])

        spine_mid_track = 0.5 * (_neck_t + _tailr_t)

        _axis_t = _tailr_t - _neck_t
        _torso_len = np.linalg.norm(_axis_t, axis=1, keepdims=True)
        _ang_t = np.unwrap(np.arctan2(_axis_t[:, 1], _axis_t[:, 0]))

        TAIL_LAG = 0.72
        TAIL_FRACTION = 0.30
        _lagged = np.empty_like(_ang_t)
        _lagged[0] = _ang_t[0]
        for _i in range(1, len(_ang_t)):
            _lagged[_i] = _lagged[_i - 1] + (1.0 - TAIL_LAG) * (_ang_t[_i] - _lagged[_i - 1])

        tail_tip_track = _tailr_t + np.stack(
            [np.cos(_lagged), np.sin(_lagged)], axis=1) * (_torso_len * TAIL_FRACTION)

    print(f"Loaded {len(stab_kps)} stabilised frames, {len(raw_kps)} raw frames, "
          f"spine/tail tracks {len(spine_mid_track)} frames [{SPINE_TAIL_SOURCE}]")
    if SPINE_TAIL_SOURCE == 'synthesised':
        print("  No spine_tail_tracks.json -- running the fallback path.")
        print("  Back will be RIGID (no arch) and the tail is a passive trail,")
        print("  not the real tail motion. Annotate for the high-quality path.")
    return (
        Image,
        ImageDraw,
        KEYPOINT_NAMES,
        SPINE_TAIL_SOURCE,
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
def _(SPINE_TAIL_SOURCE, measure, np, stab_kps):
    def build_rig(measure, ref=200.0):
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


        bone('tail_1', 'spine1', (-ref*0.07, ref*0.02), L_t1,
                 taper(L_t1, ref*0.028, ref*0.020), color='#2A2A2A', layer=20)

        bone('tail_2', 'tail_1', (L_t1, 0), L_t2,
             taper(L_t2, ref*0.020, ref*0.010, back=ref*0.010), color='#2A2A2A', layer=21)

        bone('shoulder', 'spine4', (0, 0), 0.0, None, None, layer=0)
        bone('hip', 'spine1', (0, 0), 0.0, None, None, layer=0)

        for side, base, cu, cm, cl in [('L', 40, '#444444', '#3C3C3C', '#2A2A2A'),
                                       ('R', 10, '#666666', '#5E5E5E', '#4E4E4E')]:
            bone(f'humerus_{side}', 'shoulder', (0, 0), L_hum,
                 taper(L_hum, ref*0.044, ref*0.028, back=ref*0.045), color=cu, layer=base+1)
            bone(f'radius_{side}', f'humerus_{side}', (L_hum, 0), L_rad,
                 taper(L_rad, ref*0.028, ref*0.018, back=ref*0.026), color=cm, layer=base+2)
            bone(f'paw_front_{side}', f'radius_{side}', (L_rad, 0), L_paw_f,
                 paw_shape(L_paw_f, ref*0.021), color=cl, outline='#1A1A1A', layer=base+3)
            bone(f'femur_{side}', 'hip', (0, 0), L_fem,
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

    import json as _j
    with open('outputs/character_description.json', 'w') as _f:
        _j.dump({'description': CHARACTER_DESCRIPTION, **aline_info,
                 'spine_tail_source': SPINE_TAIL_SOURCE}, _f, indent=2, default=str)
    print(f"Rig built: {len([k for k in rig if not k.startswith('_')])} bones "
          f"(4-segment spine, dynamic body ribbon)")
    return aline_info, rig


@app.cell
def _(get_kp, np, stab_kps):
    def compute_root_motion(stab_list, target_spine=200.0, detrend_win=31, bounce_gain=0.25):
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
        print(f"gait bounce range : {np.ptp(y_gait):.1f}px")
        return y_gait

    root_y_series = compute_root_motion(stab_kps)
    return (root_y_series,)


@app.cell
def _(get_kp, get_score, np, stab_kps):
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

    def build_leg_tracks(stab_list):
        n = len(stab_list)
        upper = {leg: np.zeros(n) for leg in LEG_DEF}
        lower = {leg: np.zeros(n) for leg in LEG_DEF}
        bad = {leg: np.zeros(n, dtype=bool) for leg in LEG_DEF}

        for i, fr in enumerate(stab_list):
            facing = 1.0 if get_kp(fr, 'Nose')[0] >= get_kp(fr, 'root_of_tail')[0] else -1.0
            for leg, (rt, md, tp, _, _) in LEG_DEF.items():
                def a(p, q):
                    d = get_kp(fr, q) - get_kp(fr, p)
                    t = float(np.arctan2(d[1], d[0]))
                    return float(np.pi - t) if facing < 0 else t
                upper[leg][i] = a(rt, md)
                lower[leg][i] = a(md, tp)
                cmin = min(get_score(fr, rt), get_score(fr, md), get_score(fr, tp))
                bad[leg][i] = cmin < CONF_THR

        FRONT_LEGS = {'L_front', 'R_front'}
        tracks = {}
        for leg, (_, _, _, up_key, lo_key) in LEG_DEF.items():
            u = np.unwrap(upper[leg]); l = np.unwrap(lower[leg])
            u = interp_bad(u, bad[leg]); l = interp_bad(l, bad[leg])
            if leg in FRONT_LEGS:
                tracks[up_key] = one_euro(u, min_cutoff=0.45, beta=0.004)
                tracks[lo_key] = one_euro(l, min_cutoff=0.45, beta=0.004)
            else:
                tracks[up_key] = one_euro(u, min_cutoff=1.4, beta=0.008)
                tracks[lo_key] = one_euro(l, min_cutoff=1.4, beta=0.008)


        table = [{k: float(tracks[k][i]) for k in tracks} for i in range(n)]
        nbad = {leg: int(bad[leg].sum()) for leg in LEG_DEF}
        print("leg tracks built. low-confidence frames filled per leg:", nbad)
        return table

    leg_table = build_leg_tracks(stab_kps)
    return (leg_table,)


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
    spine_mid_track,
    tail_tip_track,
):
    def wrap_angle(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))

    ELBOW_LIMIT = (-0.35, 1.60)
    STIFLE_LIMIT = (-2.00, 0.25)
    SPINE_ARCH_GAIN = 1.8
    SPINE_REST_FOLD = -0.05
    STRETCH_AMOUNT = 0.08

    def clamp_soft(rel, limits, softness=0.5):
        lo, hi = limits
        span = (hi - lo) * softness
        if span < 1e-6:
            return float(np.clip(rel, lo, hi))
        if rel > hi - span:
            return (hi - span) + span * float(np.tanh((rel - (hi - span)) / span))
        if rel < lo + span:
            return (lo + span) - span * float(np.tanh(((lo + span) - rel) / span))
        return rel

    def solve_pose(frame_data, rig, idx, root_y=0.0, tail_state=None,
                   fold_ref=0.0, target_spine=200.0, root_anchor=(300, 235)):
        kp = {n: get_kp(frame_data, n) for n in KEYPOINT_NAMES}
        neck_p, tail_p = kp['Neck'], kp['root_of_tail']
        spine_len = np.linalg.norm(tail_p - neck_p)
        if spine_len < 1e-6:
            return None
        sc = target_spine / spine_len
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
        backmid_c = rs(spine_mid_track[idx])
        chord = tail_c - neck_c
        chord_len = np.linalg.norm(chord)
        if chord_len > 1e-6:
            cd = chord / chord_len
            cn = np.array([-cd[1], cd[0]])
            perp = float((backmid_c - neck_c) @ cn)
            arch = perp / (chord_len * 0.5)
        else:
            arch = 0.0

        total_bend = SPINE_REST_FOLD + arch * SPINE_ARCH_GAIN

        w = {'root': 0.0}

        raw_dir = ang(rs(pelvis), rs(neck_p))
        base_dir = raw_dir * 0.15


        tilt = np.array([-1.5, -0.5, 0.5, 1.5]) * (total_bend * 0.5)
        w['spine1'] = base_dir + tilt[0]
        w['spine2'] = base_dir + tilt[1]
        w['spine3'] = base_dir + tilt[2]
        w['spine4'] = base_dir + tilt[3]
        w['neck'] = ang(rs(neck_p), rs(kp['Nose']))
        w['head'] = w['neck']
        w['ear_upper'] = w['head'] + rig['_style']['ear_droop']
        w['ear_lower'] = w['ear_upper'] + 0.25
        w['shoulder'] = w['spine4']
        w['hip'] = w['spine1']

        t = leg_table[idx]
        for s, up_f, lo_f, up_h, lo_h in [
                ('L', 'hum_L', 'rad_L', 'fem_L', 'tib_L'),
                ('R', 'hum_R', 'rad_R', 'fem_R', 'tib_R')]:
            a_hum = t[up_f]
            a_rad = a_hum + clamp_soft(wrap_angle(t[lo_f] - a_hum), ELBOW_LIMIT)
            a_fem = t[up_h]
            a_tib = a_fem + clamp_soft(wrap_angle(t[lo_h] - a_fem), STIFLE_LIMIT)
            w[f'humerus_{s}'] = a_hum
            w[f'radius_{s}'] = a_rad
            w[f'paw_front_{s}'] = a_rad
            w[f'femur_{s}'] = a_fem
            w[f'tibia_{s}'] = a_tib
            w[f'paw_hind_{s}'] = a_tib


        body_ax = ang(rs(neck_p), rs(tail_p))
        tail_dir = ang(rs(tail_p), rs(tail_tip_track[idx]))
        rel = wrap_angle(tail_dir - body_ax)
        w['tail_1'] = body_ax + rel
        w['tail_2'] = w['tail_1'] + rig['_style']['tail_curl_rel']

        local = {'root': 0.0}
        for name, b in rig.items():
            if name.startswith('_') or name == 'root':
                continue
            local[name] = wrap_angle(w[name] - w.get(b['parent'], 0.0))

        arch_norm = np.clip(arch / 0.15, -1.0, 1.0)
        seg_scale = 1.0 - arch_norm * STRETCH_AMOUNT
        L_seg = rig['_spine_meta']['L_seg'] * seg_scale

        verts = [root_pos.copy()]
        pos = root_pos.copy()
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
            lx = float(np.clip(lx, -L_seg*0.7, L_seg*0.7))
            ly = float(min(ly, wdn*0.92))
            cc, ss = np.cos(seg_ang), np.sin(seg_ang)
            return seg_start_pos + np.array([lx*cc - ly*ss, lx*ss + ly*cc])

        anchors = {
            'shoulder': edge_anchor(rs(shs), spine_verts[3], w['spine4']),
            'hip':      edge_anchor(rs(hips), spine_verts[0], w['spine1']),
        }
        return {'root_pos': root_pos, 'scale': sc, 'local': local,
                'facing': facing, 'anchors': anchors, 'arch': arch,
                'spine_verts': spine_verts, 'seg_scale': seg_scale,
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
        ELBOW_LIMIT,
        STIFLE_LIMIT,
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
                if name == 'neck':
                    ox *= seg_scale
                c, s = np.cos(pw['angle']), np.sin(pw['angle'])
                pos = pw['pos'] + np.array([ox*c - oy*s, ox*s + oy*c])
                angle = pw['angle'] + local.get(name, 0.0)
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
        verts = pose['spine_verts'].copy()
        meta = rig['_spine_meta']
        w_up = np.array([0.112, 0.128, 0.132, 0.128, 0.112]) * 200.0
        w_dn = np.array([0.135, 0.110, 0.120, 0.158, 0.150]) * 200.0

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

        neck_ext = verts[-1] + dirs[-1] * (meta['L_seg'] * 0.15)

        back = -dirs[0]
        rump_top = top[0]
        rump_bot = bot[0]
        rump_mid = (rump_top + rump_bot) / 2
        bulge = meta['L_seg'] * 0.25
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

        _coarse = rig.get('_style', {}).get('complexity') == 'coarse'

        if 'head' in world and not _coarse:
            hw = world['head']; hl = rig['head']['length']
            c, s = np.cos(hw['angle']), np.sin(hw['angle'])
            def hp(x, y):
                return np.array([x*c - y*s + hw['pos'][0], x*s + y*c + hw['pos'][1]])
            hh = rig['_style']['h_head'] if '_style' in rig else hl*0.48
            mz = hp(hl*0.98, hh*0.125); mr = hh*0.54
            d.ellipse([int(mz[0]-mr), int(mz[1]-mr*0.68), int(mz[0]+mr), int(mz[1]+mr*0.68)],
                      fill=rig.get('_style', {}).get('palette', {}).get('muzzle', '#5A5A5A'),
                      outline='#333333', width=1)
            nsp = hp(hl*1.16, hh*0.083)
            d.ellipse([int(nsp[0])-4, int(nsp[1])-3, int(nsp[0])+4, int(nsp[1])+3], fill='#1A1A1A')
            eye = hp(hl*0.58, -hh*0.33)
            _er = max(3, int(hh*0.217))
            disc(eye, _er, 'white', '#222222', 1)
            disc(eye, max(1, int(_er*0.4)), '#1A1A1A', None, 0)

        if not _coarse:
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
    preview(0)
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
    generate_video(out_path='rig_alpha035.mp4')
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
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
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
    Image,
    aline_info,
    compute_world_transforms,
    measure,
    np,
    render_rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    import json as _sj
    import os as _so
    from character_style import build_rig_styled as _brs

    SWEEP_DIR = 'outputs/sweep'
    SWEEP_FRAMES = [0, 70, 140]
    SWEEP_WRITE_VIDEO = False

    def _sweep_combos(template_names, auto_pick):
        rows = [(n, 0.0, 'fine') for n in template_names]
        rows += [(auto_pick, 1.0, 'fine'),
                 (auto_pick, 1.0, 'coarse')]
        return rows

    def sweep():
        _so.makedirs(SWEEP_DIR, exist_ok=True)
        from posetoon_aline import nearest_template as _nt
        _tpl_path = ('outputs/breed_templates.json'
                     if _so.path.exists('outputs/breed_templates.json')
                     else 'outputs/dog_templates_v2.json')
        with open(_tpl_path) as f:
            tpl = _sj.load(f)
        _sel = ('outputs/breed_selection.json'
                if _so.path.exists('outputs/breed_selection.json') else None)
        if _sel:
            with open(_sel) as f:
                _auto = _sj.load(f).get('template')
        if not _sel or _auto not in tpl['templates']:
            _auto, _, _ = _nt(measure, tpl['templates'])
        combos = _sweep_combos(list(tpl['templates']), _auto)
        print(f"Templates from {_tpl_path}  ({len(tpl['templates'])} entries), "
              f"auto-pick = {_auto}\n")

        def _pal_for(bname):
            try:
                from breed_markings import breed_palette, BREED_COLOURS
                if bname in BREED_COLOURS:
                    return breed_palette(bname)
            except ImportError:
                pass
            if _so.path.exists('outputs/coat_palette.json'):
                with open('outputs/coat_palette.json') as f:
                    return _sj.load(f)['palette']
            return None

        rows, summary = [], []
        for tier, alpha, complexity in combos:
            tm = tpl['templates'][tier]['measure']
            m = {k: (1 - alpha) * tm[k] + alpha * measure[k] for k in tm}
            ap = tpl['templates'][tier]['appearance']
            rg = _brs(m, ap, tier=tier, complexity=complexity,
                      palette=_pal_for(tier))

            tag = f"{tier.replace(chr(32), chr(95))}_a{int(round(alpha*100)):03d}_{complexity}"
            imgs = []
            for fi in SWEEP_FRAMES:
                if fi >= len(stab_kps):
                    continue
                _lift = max(0.0, (m['fem'] + m['tib'])
                            - (measure['fem'] + measure['tib']))
                pose = solve_pose(stab_kps[fi], rg, fi,
                                  root_y=root_y_series[fi] - _lift,
                                  fold_ref=spine_fold_ref)
                if pose is None:
                    continue
                w = compute_world_transforms(rg, pose)
                img = render_rig(rg, w, pose=pose, facing=pose['facing'],
                                 canvas_size=(560, 380))
                path = f"{SWEEP_DIR}/frame{fi:03d}_{tag}.png"
                img.convert('RGB').save(path)
                imgs.append(img.convert('RGB'))

            legs = m['hum'] + m['rad'], m['fem'] + m['tib']
            summary.append({
                'tag': tag, 'tier': tier, 'alpha': alpha, 'complexity': complexity,
                'front_leg_px': round(legs[0], 1), 'hind_leg_px': round(legs[1], 1),
                'head_px': round(rg['head']['length'], 1),
                'neck_px': round(rg['neck']['length'], 1),
                'ear_type': ap['ear_type'], 'ear_len': ap['ear_len'],
                'tail_len': ap['tail_len'], 'tail_curl': ap['tail_curl'],
                'build_factor': rg['_style']['build_factor'],
                'n_bones': len([k for k in rg if not k.startswith('_')]),
            })
            rows.append((tag, imgs))
            print(f"  {tag:26} front_leg={legs[0]:5.1f}  hind_leg={legs[1]:5.1f}  "
                  f"head={rg['head']['length']:5.1f}  ear={ap['ear_type']:10} "
                  f"bf={rg['_style']['build_factor']:.2f}  bones={summary[-1]['n_bones']}")

            if SWEEP_WRITE_VIDEO:
                _sweep_video(rg, f"{SWEEP_DIR}/rig_{tag}.mp4")

        if rows and rows[0][1]:
            cw, ch = rows[0][1][0].size
            sheet = Image.new('RGB', (cw * len(SWEEP_FRAMES), ch * len(rows)), 'white')
            for r, (_tag, imgs) in enumerate(rows):
                for c, im in enumerate(imgs):
                    sheet.paste(im, (c * cw, r * ch))
            sheet.save(f"{SWEEP_DIR}/contact_sheet.png")
            print(f"\nSaved {SWEEP_DIR}/contact_sheet.png "
                  f"({len(rows)} variants x {len(SWEEP_FRAMES)} frames)")

        with open(f"{SWEEP_DIR}/sweep_summary.json", 'w') as f:
            _sj.dump(summary, f, indent=2)
        print(f"Saved {SWEEP_DIR}/sweep_summary.json")
        print(f"\nRow order in the contact sheet:")
        for i, r in enumerate(summary):
            print(f"  row {i}: {r['tag']}")

    def _sweep_video(rg, out_path):
        import cv2 as _cv
        cap = _cv.VideoCapture('dogvideo.mp4')
        ow = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH)); oh = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT))
        ph = 400; scale = ph / oh; pw = int(ow * scale); aw = 600
        out = _cv.VideoWriter(out_path, _cv.VideoWriter_fourcc(*'mp4v'), 30, (pw + aw, ph))
        for i, fr in enumerate(stab_kps):
            ok, orig = cap.read()
            if not ok:
                break
            p1 = _cv.resize(orig, (pw, ph))
            pose = solve_pose(fr, rg, i, root_y=root_y_series[i] - aline_info['root_lift_px'], fold_ref=spine_fold_ref)
            if pose is None:
                p2 = np.full((ph, aw, 3), 240, np.uint8)
            else:
                w = compute_world_transforms(rg, pose)
                img = render_rig(rg, w, pose=pose, facing=pose['facing'], canvas_size=(aw, ph))
                p2 = _cv.cvtColor(np.array(img.convert('RGB')), _cv.COLOR_RGB2BGR)
            out.write(np.hstack([p1, p2]))
        cap.release(); out.release()
        print(f"    wrote {out_path}")

    sweep()
    return


@app.cell
def _(
    compute_world_transforms,
    measure,
    np,
    render_rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
):
    import json as _ij
    import os as _io
    import time as _it
    import cv2 as _ic
    from character_style import build_rig_styled as _ibrs

    IOU_FRAMES = list(range(0, min(len(stab_kps), 279), 12))
    IOU_SCORE_THR = 0.30
    IOU_ROI_MARGIN = 0.35
    IOU_ROI_BOTTOM = 0.06
    IOU_CANVAS = (560, 380)


    KPI = {n: i for i, n in enumerate([
        'L_Eye','R_Eye','Nose','Neck','root_of_tail','L_Shoulder','L_Elbow',
        'L_F_Paw','R_Shoulder','R_Elbow','R_F_Paw','L_Hip','L_Knee','L_B_Paw',
        'R_Hip','R_Knee','R_B_Paw'])}

    def _real_mask(frame, kp, scores):
        h, w = frame.shape[:2]
        ok = np.asarray(scores) >= IOU_SCORE_THR
        if ok.sum() < 4:
            return None
        pts = np.asarray(kp)[ok]
        x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        rx0 = max(0, int(x0 - bw * IOU_ROI_MARGIN))
        ry0 = max(0, int(y0 - bh * IOU_ROI_MARGIN))
        rx1 = min(w, int(x1 + bw * IOU_ROI_MARGIN))
        ry1 = min(h, int(y1 + bh * IOU_ROI_BOTTOM))
        sub = frame[ry0:ry1, rx0:rx1]
        if sub.size == 0:
            return None
        g = _ic.cvtColor(sub, _ic.COLOR_BGR2GRAY)
        _t, b = _ic.threshold(g, 0, 255, _ic.THRESH_BINARY_INV + _ic.THRESH_OTSU)
        k = _ic.getStructuringElement(_ic.MORPH_ELLIPSE, (5, 5))
        b = _ic.morphologyEx(b, _ic.MORPH_OPEN, k)
        b = _ic.morphologyEx(b, _ic.MORPH_CLOSE, k)
        n, lab, _st, _c = _ic.connectedComponentsWithStats(b, 8)
        loc = pts - np.array([rx0, ry0])
        best, hits_best = 0, -1
        for l in range(1, n):
            hits = sum(1 for (lx, ly) in loc
                       if 0 <= int(ly) < lab.shape[0] and 0 <= int(lx) < lab.shape[1]
                       and lab[int(ly), int(lx)] == l)
            if hits > hits_best:
                hits_best, best = hits, l
        if best == 0:
            return None
        m = np.zeros((h, w), np.uint8)
        m[ry0:ry1, rx0:rx1] = (lab == best).astype(np.uint8) * 255
        return m

    def _char_mask_and_time(rg, fi, lift=0.0):
        pose = solve_pose(stab_kps[fi], rg, fi,
                          root_y=root_y_series[fi] - lift,
                          fold_ref=spine_fold_ref)
        if pose is None:
            return None, None, None
        w = compute_world_transforms(rg, pose)
        t0 = _it.perf_counter()
        img = render_rig(rg, w, pose=pose, facing=pose['facing'], canvas_size=IOU_CANVAS)
        ms = (_it.perf_counter() - t0) * 1000.0
        rgba = np.asarray(img.convert('RGBA'))
        alpha = rgba[..., 3]
        if alpha.max() > 0 and alpha.min() < 255:
            m = (alpha > 8).astype(np.uint8) * 255
        else:
            a = rgba[..., :3].astype(int)
            bg = a[0, 0]
            m = (np.abs(a - bg).sum(axis=2) > 6).astype(np.uint8) * 255

        lm = {}
        for nm, wt in w.items():
            px, py = wt['pos']
            lm[nm] = ((IOU_CANVAS[0] - 1 - px) if pose['facing'] < 0 else px, py)
        return m, lm, ms

    JOINT_MAP = {
        'Neck': 'neck', 'root_of_tail': 'tail_1',
        'L_Shoulder': 'humerus_L', 'L_Elbow': 'radius_L', 'L_F_Paw': 'paw_front_L',
        'R_Shoulder': 'humerus_R', 'R_Elbow': 'radius_R', 'R_F_Paw': 'paw_front_R',
        'L_Hip': 'femur_L', 'L_Knee': 'tibia_L', 'L_B_Paw': 'paw_hind_L',
        'R_Hip': 'femur_R', 'R_Knee': 'tibia_R', 'R_B_Paw': 'paw_hind_R',
    }

    def _align(lm, kp, scores, mask, out_shape):
        src, dst = [], []
        for kname, bname in JOINT_MAP.items():
            i = KPI[kname]
            if scores[i] < IOU_SCORE_THR or bname not in lm:
                continue
            src.append(lm[bname]); dst.append(kp[i])
        if len(src) < 3:
            return None
        src = np.asarray(src, np.float32).reshape(-1, 1, 2)
        dst = np.asarray(dst, np.float32).reshape(-1, 1, 2)
        M, _inl = _ic.estimateAffinePartial2D(src, dst, method=_ic.LMEDS)
        if M is None:
            return None
        return _ic.warpAffine(mask, M, (out_shape[1], out_shape[0]),
                              flags=_ic.INTER_NEAREST)

    def run_iou():
        from posetoon_aline import nearest_template as _nt2
        _tp = ('outputs/breed_templates.json'
               if _io.path.exists('outputs/breed_templates.json')
               else 'outputs/dog_templates_v2.json')
        with open(_tp) as f:
            tpl = _ij.load(f)
        _sel2 = ('outputs/breed_selection.json'
                 if _io.path.exists('outputs/breed_selection.json') else None)
        _auto2 = None
        if _sel2:
            with open(_sel2) as f:
                _auto2 = _ij.load(f).get('template')
        if _auto2 not in tpl['templates']:
            _auto2, _, _ = _nt2(measure, tpl['templates'])
        IOU_VARIANTS = ([(n, 0.0, 'fine') for n in tpl['templates']]
                        + [(_auto2, 1.0, 'fine'), (_auto2, 1.0, 'coarse')])
        print(f"Templates from {_tp}, auto-pick = {_auto2}")
        def _pal(bname):
            try:
                from breed_markings import breed_palette, BREED_COLOURS
                if bname in BREED_COLOURS:
                    return breed_palette(bname)
            except ImportError:
                pass
            return None

        cap = _ic.VideoCapture('dogvideo.mp4')
        reals = {}
        for fi in IOU_FRAMES:
            cap.set(_ic.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            kp = stab_kps[fi]['keypoints']
            sc = stab_kps[fi].get('scores', [1.0] * len(kp))
            m = _real_mask(fr, kp, sc)
            if m is not None:
                reals[fi] = (m, np.asarray(kp, float), fr.shape[:2])
        cap.release()
        print(f"Real masks: {len(reals)}/{len(IOU_FRAMES)} frames\n")

        out = []
        for tier, alpha, comp in IOU_VARIANTS:
            tm = tpl['templates'][tier]['measure']
            m_ = {k: (1 - alpha) * tm[k] + alpha * measure[k] for k in tm}
            rg = _ibrs(m_, tpl['templates'][tier]['appearance'],
                       tier=tier, complexity=comp, palette=_pal(tier))
            _lift_v = max(0.0, (m_['fem'] + m_['tib'])
                          - (measure['fem'] + measure['tib']))
            for _wf in list(reals)[:3]:
                _char_mask_and_time(rg, _wf, _lift_v)

            ious, times = [], []
            for fi, (rmask, kp, shp) in reals.items():
                cmask, lm, ms = _char_mask_and_time(rg, fi, _lift_v)
                if cmask is None:
                    continue
                times.append(ms)
                sc_i = stab_kps[fi].get('scores', [1.0] * 17)
                warped = _align(lm, kp, sc_i, cmask, shp)
                if warped is None:
                    continue
                a = warped > 0; b = rmask > 0
                inter = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                if union > 0:
                    ious.append(inter / union)
                if fi == sorted(reals)[0]:
                    ov = np.zeros((shp[0], shp[1], 3), np.uint8)
                    ov[..., 2] = rmask
                    ov[..., 1] = warped
                    _safe = tier.replace(' ', '_')
                    _ic.imwrite(f'outputs/sweep/align_{_safe}_a{int(alpha*100):03d}'
                                f'_{comp}.png', ov)
            drawn = len([k for k, v in rg.items()
                         if not k.startswith('_') and v.get('mesh')])
            r = {'tier': tier, 'alpha': alpha, 'complexity': comp,
                 'iou_mean': float(np.mean(ious)) if ious else None,
                 'iou_median': float(np.median(ious)) if ious else None,
                 'iou_std': float(np.std(ious)) if ious else None,
                 'n_frames': len(ious),
                 'render_ms_median': float(np.median(times)) if times else None,
                 'render_ms_mean': float(np.mean(times)) if times else None,
                 'drawn_parts': drawn}
            out.append(r)
            print(f"  {tier:6} a={alpha:.1f} {comp:6}  "
                  f"IoU {r['iou_mean']:.3f} +/- {r['iou_std']:.3f}  "
                  f"n={r['n_frames']:3}  render {r['render_ms_median']:5.2f}ms  "
                  f"parts={drawn}")

        with open('outputs/e4_iou_results.json', 'w') as f:
            _ij.dump({'note': 'absolute IoU is a LOWER BOUND -- the real mask '
                              'retains a thin shadow band; relative comparison '
                              'is the valid reading',
                      'frames': IOU_FRAMES, 'results': out}, f, indent=2)
        print("\nSaved outputs/e4_iou_results.json")

        idx = {(r['tier'], r['alpha'], r['complexity']): r for r in out}
        pairs = [(t, a) for (t, a, c) in idx if c == 'coarse'
                 and (t, a, 'fine') in idx]
        if pairs:
            print("\nE3 matched-pair cost/benefit (same tier and alpha):")
            d_iou, d_ms = [], []
            for t, a in pairs:
                f_, c_ = idx[(t, a, 'fine')], idx[(t, a, 'coarse')]
                d_iou.append(f_['iou_mean'] - c_['iou_mean'])
                d_ms.append(f_['render_ms_median'] - c_['render_ms_median'])
                print(f"  {t:17} a={a:.1f}   IoU fine {f_['iou_mean']:.3f} vs "
                      f"coarse {c_['iou_mean']:.3f}  (diff {d_iou[-1]:+.3f}, "
                      f"pooled std ~{(f_['iou_std']+c_['iou_std'])/2:.3f})   "
                      f"render {f_['render_ms_median']:.2f} vs "
                      f"{c_['render_ms_median']:.2f}ms   "
                      f"parts {f_['drawn_parts']} vs {c_['drawn_parts']}")
            print(f"  mean IoU gain from detail: {np.mean(d_iou):+.3f}  "
                  f"(cost {np.mean(d_ms):+.2f}ms/frame)")
            print("  Read against the per-variant std: a gain far smaller than")
            print("  the std means detail buys no measurable shape fidelity.")

    run_iou()
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

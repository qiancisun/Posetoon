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

    # hand-annotated spine-mid + tail-tip tracks. AP-10K labels only Neck and
    # Root-of-Tail along the spine axis (nothing between) and nothing past the
    # tail root, so the real back-arch and tail motion are absent from the pose
    # data. These tracks (20 keyframes hand-read, interpolated + smoothed)
    # restore them, enabling data-driven spine flexion and tail animation.
    with open('spine_tail_tracks.json', 'r') as f:
        spine_tail = json.load(f)
    spine_mid_track = np.array(spine_tail['spine_mid'], dtype=float)   # (N,2)
    tail_tip_track = np.array(spine_tail['tail_tip'], dtype=float)     # (N,2)

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
        """Measure this dog's bone proportions. Torso -> median; limbs -> 75th
        percentile (a 2D projection can only shorten a bone, so the median
        under-estimates limb length)."""
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
def _(measure, np):
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

        # 4 spine segments -- skeleton only, no mesh (body drawn as one ribbon)
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


        # A tracked tail-root anchor prevents a visible torso-to-tail gap.
        bone('tail_root', 'root', (0, 0), 0.0, None, None, layer=0)
        bone('tail_1', 'tail_root', (0, 0), L_t1,
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

        # store base segment length + half-heights so the renderer can build the
        # body ribbon and the solver can stretch the chain
        rig['_spine_meta'] = {'L_seg': L_seg,
                              'w_up': ref * 0.125, 'w_dn': ref * 0.150,
                              'w_up_end': ref * 0.075, 'w_dn_end': ref * 0.090}
        return rig

    rig = build_rig(measure)
    print(f"Rig built: {len([k for k in rig if not k.startswith('_')])} bones "
          f"(4-segment spine, dynamic body ribbon)")
    return (rig,)


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
        """Linear-interpolate the entries flagged bad from the good ones."""
        track = track.copy()
        good = ~bad
        if good.sum() < 2:
            return track
        idx = np.arange(len(track))
        track[bad] = np.interp(idx[bad], idx[good], track[good])
        return track

    # leg definition: (root, mid, tip) keypoints and which two bone-angles
    LEG_DEF = {
        'L_front': ('L_Shoulder', 'L_Elbow', 'L_F_Paw', 'hum_L', 'rad_L'),
        'R_front': ('R_Shoulder', 'R_Elbow', 'R_F_Paw', 'hum_R', 'rad_R'),
        'L_hind':  ('L_Hip', 'L_Knee', 'L_B_Paw', 'fem_L', 'tib_L'),
        'R_hind':  ('R_Hip', 'R_Knee', 'R_B_Paw', 'fem_R', 'tib_R'),
    }
    CONF_THR = 0.35

    def build_leg_tracks(stab_list):
        n = len(stab_list)
        # raw upper/lower angle per leg + per-frame min confidence of that leg
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

        # per bone-angle: unwrap -> fill this leg's bad frames by time-interp -> smooth.
            # Front legs (humerus/radius) get stronger smoothing than hind: the front
            # bones are short (rad ~35px vs fem ~49px), so the same pixel of keypoint
            # noise becomes a larger angular spike through the short lever. Measured:
            # front humerus had 43-63 jerky frames vs 25-26 for the hind. A lower
            # min_cutoff damps the front harder without touching the already-smooth
            # hind legs.
        FRONT_LEGS = {'L_front', 'R_front'}
        tracks = {}
        for leg, (_, _, _, up_key, lo_key) in LEG_DEF.items():
            u = np.unwrap(upper[leg]); l = np.unwrap(lower[leg])
            u = interp_bad(u, bad[leg]); l = interp_bad(l, bad[leg])
            if leg in FRONT_LEGS:
                tracks[up_key] = one_euro(u, min_cutoff=0.45, beta=0.004)   # stronger
                tracks[lo_key] = one_euro(l, min_cutoff=0.45, beta=0.004)
            else:
                tracks[up_key] = one_euro(u, min_cutoff=1.4, beta=0.008)   # as before
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
        """Raw facing = sign(Nose.x - tail.x). Only flip after `hold` consecutive
        frames agree on the new direction, so a couple of noisy frames can't
        mirror the whole character."""
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
    # The hand-keyed spine-mid track starts bending about 8 frames late near frame 170.
    SPINE_TRACK_LEAD_FRAMES = 8

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

        # ---- data-driven arch from hand-tracked back-mid ----
        neck_c = rs(neck_p); tail_c = rs(tail_p)
        track_idx = min(idx + SPINE_TRACK_LEAD_FRAMES, len(spine_mid_track) - 1)
        backmid_c = rs(spine_mid_track[track_idx])
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
        base_dir = raw_dir * 0.15      # pull toward 0 (horizontal in canonical space)

    
        tilt = np.array([-1.5, -0.5, 0.5, 1.5]) * (total_bend * 0.5)
        w['spine1'] = base_dir + tilt[0]
        w['spine2'] = base_dir + tilt[1]
        w['spine3'] = base_dir + tilt[2]
        w['spine4'] = base_dir + tilt[3]
        w['neck'] = ang(rs(neck_p), rs(kp['Nose']))
        w['head'] = w['neck']
        # No ear landmarks are available, so use a small bounded head-pitch response.
        head_pitch = wrap_angle(w['head'] - base_dir)
        ear_sway = float(np.clip(-0.14 * head_pitch, -0.12, 0.12))
        w['ear_upper'] = w['head'] + 1.50 + ear_sway
        w['ear_lower'] = w['ear_upper'] + 0.25 + ear_sway * 0.45
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

   
        body_ax = ang(rs(neck_p), rs(tail_p))          # torso axis (points to rump)
        tail_dir = ang(rs(tail_p), rs(tail_tip_track[idx]))
        rel = wrap_angle(tail_dir - body_ax)           # tail angle relative to body
        w['tail_root'] = body_ax
        w['tail_1'] = body_ax + rel
        w['tail_2'] = w['tail_1'] + 0.25
    
        local = {'root': 0.0}
        for name, b in rig.items():
            if name.startswith('_') or name == 'root':
                continue
            local[name] = wrap_angle(w[name] - w.get(b['parent'], 0.0))

        # ---- spine stretch, linked to arch (arched=shorter, flat=longer) ----
        arch_norm = np.clip(arch / 0.15, -1.0, 1.0)
        seg_scale = 1.0 - arch_norm * STRETCH_AMOUNT
        L_seg = rig['_spine_meta']['L_seg'] * seg_scale

        # ---- build the 4-segment spine vertex chain in canonical space ----
        verts = [root_pos.copy()]
        pos = root_pos.copy()
        for i in range(1, 5):
            a = w[f'spine{i}']
            pos = pos + np.array([L_seg*np.cos(a), L_seg*np.sin(a)])
            verts.append(pos.copy())
        spine_verts = np.array(verts)          # (5,2)

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
            'tail_root': rs(tail_p),
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
        """
        Compose each bone's world transform from its parent. Two special cases:
          * anchor bones ('shoulder','hip') have their world POSITION overwritten
            with the tracked joint location (limbs stay welded to the torso).
          * the neck's offset is scaled by seg_scale so it lengthens/shortens with
            the body: in the extension phase the spine lengthens and the neck
            reaches forward with it, so the forward reach is shared by spine+neck
            instead of the torso mesh ballooning toward a static neck. The head
            hangs off the lengthened neck end, so it moves forward without
            deforming.
        """
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
        """
        One closed polygon for the torso along the 4-segment spine.
        Deep-chest / tucked-loin silhouette (w_dn shaped front-to-back), a gentle
        back line (w_up), extended past the shoulder to cover the neck root, and
        a rounded rump made by a single modest rearward bulge point that stays
        attached to the body (no detached wedge). Drawn in canonical facing-right
        space; the final canvas flip in render_rig mirrors it with the limbs.
        """
        verts = pose['spine_verts'].copy()          # (5,2) rump(0) -> shoulder(4)
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

        # rounded rump: one modest point centred behind the rump, only a little
        # way out, so the rear edge is a gentle curve that stays attached.
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

        # --- body ribbon first (lowest layer) ---
        if pose is not None and 'spine_verts' in pose:
            poly = body_ribbon(pose)
            d.polygon([(int(x), int(y)) for (x, y) in poly],
                      fill='#3D3D3D', outline='#222222')

        # --- remaining meshed bones, skipping non-bone keys & mesh-less spines ---
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
            mz = hp(hl*0.98, hl*0.06); mr = hl*0.26
            d.ellipse([int(mz[0]-mr), int(mz[1]-mr*0.68), int(mz[0]+mr), int(mz[1]+mr*0.68)],
                      fill='#5A5A5A', outline='#333333', width=1)
            nsp = hp(hl*1.16, hl*0.04)
            d.ellipse([int(nsp[0])-4, int(nsp[1])-3, int(nsp[0])+4, int(nsp[1])+3], fill='#1A1A1A')
            eye = hp(hl*0.58, -hl*0.16); disc(eye, 5, 'white', '#222222', 1); disc(eye, 2, '#1A1A1A', None, 0)

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
            p = solve_pose(stab_kps[i], rig, i, root_y=root_y_series[i], fold_ref=spine_fold_ref)
            for k in wl:
                wl[k].append(p['world_leg'][k])
            for s in ('L', 'R'):
                rel_elbow[s].append(wrap_angle(p['local'][f'radius_{s}']))
                rel_stifle[s].append(wrap_angle(p['local'][f'tibia_{s}']))

        # ---- spec #6: phase lags vs LF, must reproduce reference gait ----
        def lag(a, b):
            a = np.unwrap(np.array(a)); a -= a.mean()
            b = np.unwrap(np.array(b)); b -= b.mean()
            return int(np.argmax(np.correlate(b, a, mode='full')) - (len(a) - 1))
        LF = wl['humerus_L']
        print("=== spec #6  PHASE (reference: RF +9, LH -25, RH -24) ===")
        print(f"  RF (humerus_R) lag = {lag(LF, wl['humerus_R']):+3d}")
        print(f"  LH (femur_L)   lag = {lag(LF, wl['femur_L']):+3d}")
        print(f"  RH (femur_R)   lag = {lag(LF, wl['femur_R']):+3d}")

        # ---- spec #8: joints must not sit pinned at a limit ----
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

        # ---- jerkiness, for continuity ----
        print("=== motion continuity (jerky = |dangle|>8deg/frame) ===")
        for k in wl:
            v = np.abs(np.diff(np.degrees(np.unwrap(np.array(wl[k])))))
            print(f"  {k:10} jerky={int((v>8).sum()):3d}  max={v.max():5.1f}")

    def preview(idx=0):
        p = solve_pose(stab_kps[idx], rig, idx, root_y=root_y_series[idx], fold_ref=spine_fold_ref)
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
            pose = solve_pose(fr, rig, i, root_y=root_y_series[i], fold_ref=spine_fold_ref)
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
    return


@app.cell
def _(np, rig, root_y_series, solve_pose, spine_fold_ref, stab_kps):
    p = solve_pose(stab_kps[60], rig, 60, root_y=root_y_series[60], fold_ref=spine_fold_ref)
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
def _(np, rig, root_y_series, solve_pose, spine_fold_ref, stab_kps):
    p60 = solve_pose(stab_kps[60], rig, 60, root_y=root_y_series[60], fold_ref=spine_fold_ref)

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
def _(np, rig, root_y_series, solve_pose, spine_fold_ref, stab_kps):
    def diag_tail():
        angs = []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k], fold_ref=spine_fold_ref)
            angs.append(np.degrees(pp['local']['tail_1']))
        angs = np.array(angs)
        print(f"tail_1 local angle: range=[{angs.min():.0f}, {angs.max():.0f}], ptp={np.ptp(angs):.0f} deg")

    diag_tail()
    return


@app.cell
def _(
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
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k], fold_ref=spine_fold_ref)
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
def _(np, rig, root_y_series, solve_pose, spine_fold_ref, stab_kps):
    def diag_body_tilt():
        tilts, ys = [], []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k], fold_ref=spine_fold_ref)
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
def _(np, rig, root_y_series, solve_pose, spine_fold_ref, stab_kps):
    def diag_arch_dynamics():
        arches = []
        for k in range(len(stab_kps)):
            pp = solve_pose(stab_kps[k], rig, k, root_y=root_y_series[k], fold_ref=spine_fold_ref)
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


if __name__ == "__main__":
    app.run()

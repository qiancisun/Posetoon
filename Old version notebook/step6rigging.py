import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    return


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

    print(f"Loaded {len(stab_kps)} stabilised frames, {len(raw_kps)} raw frames")
    return (
        Image,
        ImageDraw,
        KEYPOINT_NAMES,
        KP_INDEX,
        cv2,
        get_kp,
        np,
        plt,
        raw_kps,
        stab_kps,
    )


@app.cell
def _(get_kp, np, stab_kps):
    def measure_skeleton(stab_list, target_spine=200.0, ref=200.0):
        acc = {k: [] for k in ['lower', 'upper', 'head',
                               'hum', 'rad', 'fem', 'tib']}
        off = {'sh': [], 'hip': []}

        def local_off(point, origin, tip, sc, facing):
            u = tip - origin
            n = np.linalg.norm(u)
            if n < 1e-6:
                return None
            u = u / n
            v = np.array([-u[1], u[0]])
            d = point - origin
            return np.array([float(d @ u) * sc,
                             float(d @ v) * sc * facing])

        for fr in stab_list:
            neck = get_kp(fr, 'Neck')
            tail = get_kp(fr, 'root_of_tail')
            sl = np.linalg.norm(neck - tail)
            if sl < 1e-6:
                continue
            sc = target_spine / sl
            facing = 1.0 if get_kp(fr, 'Nose')[0] >= tail[0] else -1.0

            hips = (get_kp(fr, 'L_Hip') + get_kp(fr, 'R_Hip')) / 2
            shs = (get_kp(fr, 'L_Shoulder') + get_kp(fr, 'R_Shoulder')) / 2
            pelvis = tail * 0.55 + hips * 0.45
            waist = (pelvis + neck) / 2
            waist_adj = waist + np.array(
                [0.0, (shs[1] + hips[1]) / 2 - waist[1]]) * 0.35

            acc['lower'].append(np.linalg.norm(waist - pelvis) * sc)
            acc['upper'].append(np.linalg.norm(neck - waist) * sc)
            acc['head'].append(np.linalg.norm(get_kp(fr, 'Nose') - neck) * sc)

            acc['hum'].append(np.linalg.norm(
                get_kp(fr, 'L_Elbow') - get_kp(fr, 'L_Shoulder')) * sc)
            acc['rad'].append(np.linalg.norm(
                get_kp(fr, 'L_F_Paw') - get_kp(fr, 'L_Elbow')) * sc)
            acc['fem'].append(np.linalg.norm(
                get_kp(fr, 'L_Knee') - get_kp(fr, 'L_Hip')) * sc)
            acc['tib'].append(np.linalg.norm(
                get_kp(fr, 'L_B_Paw') - get_kp(fr, 'L_Knee')) * sc)

            o = local_off(shs, waist_adj, neck, sc, facing)
            if o is not None:
                off['sh'].append(o)
            o = local_off(hips, pelvis, waist_adj, sc, facing)
            if o is not None:
                off['hip'].append(o)

        m = {}
        for k in ['lower', 'upper', 'head']:
            m[k] = float(np.median(acc[k]))
        for k in ['hum', 'rad', 'fem', 'tib']:
            m[k] = float(np.percentile(acc[k], 75))

        sh_raw = np.median(np.array(off['sh']), axis=0)
        hip_raw = np.median(np.array(off['hip']), axis=0)
        SH_MAX_Y = ref * 0.28
        HIP_MAX_Y = ref * 0.130
        m['sh_off'] = np.array([sh_raw[0],
                                float(np.clip(sh_raw[1], -SH_MAX_Y, SH_MAX_Y))])
        m['hip_off'] = np.array([hip_raw[0],
                                 float(np.clip(hip_raw[1], -HIP_MAX_Y, HIP_MAX_Y))])

        m['sh_off_intercept'] = m['sh_off']
        m['sh_off_slope'] = np.zeros(2)
        m['hip_off_intercept'] = m['hip_off']
        m['hip_off_slope'] = np.zeros(2)

        print("measured proportions (spine = 200):")
        for k in ['lower', 'upper', 'head', 'hum', 'rad', 'fem', 'tib']:
            print(f"  {k:6}: {m[k]:6.1f}px")
        print(f"  front leg total : {m['hum'] + m['rad']:6.1f}px")
        print(f"  hind  leg total : {m['fem'] + m['tib']:6.1f}px")
        print(f"  shoulder offset : ({m['sh_off'][0]:6.1f},{m['sh_off'][1]:6.1f})")
        print(f"  hip offset      : ({m['hip_off'][0]:6.1f},{m['hip_off'][1]:6.1f})")
        return m


    def build_rig(measure, ref=200.0):
        L_low = measure['lower']
        L_up = measure['upper']
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

        def bone(name, parent, offset, length, mesh, color,
                 outline=None, layer=0):
            rig[name] = {'parent': parent,
                         'offset': np.array(offset, dtype=float),
                         'length': length, 'mesh': mesh, 'color': color,
                         'outline': outline, 'layer': layer}

        bone('root', None, (0, 0), 0.0, None, None, layer=0)

        bone('spine_lower', 'root', (0, 0), L_low,
             mesh=[(-ref*0.16, -ref*0.068), (-ref*0.05, -ref*0.118),
                   (L_low*0.55, -ref*0.124), (L_low + ref*0.16, -ref*0.116),
                   (L_low + ref*0.16, ref*0.128), (L_low*0.55, ref*0.168),
                   (-ref*0.02, ref*0.152), (-ref*0.14, ref*0.066)],
             color='#3D3D3D', outline='#222222', layer=30)

        bone('spine_upper', 'spine_lower', (L_low, 0), L_up,
             mesh=[(-ref*0.12, -ref*0.060), (-ref*0.02, -ref*0.116),
                   (L_up*0.42, -ref*0.130), (L_up*0.82, -ref*0.114),
                   (L_up, -ref*0.070), (L_up, ref*0.100),
                   (L_up*0.70, ref*0.188), (L_up*0.28, ref*0.180),
                   (-ref*0.02, ref*0.128), (-ref*0.12, ref*0.070)],
             color='#3D3D3D', outline='#222222', layer=31)

        bone('neck', 'spine_upper', (L_up, 0), L_neck,
             mesh=[(-ref*0.03, -ref*0.082), (L_neck, -ref*0.090),
                   (L_neck, ref*0.078), (-ref*0.03, ref*0.074)],
             color='#424242', layer=32)

        bone('head', 'neck', (L_neck, 0), L_head,
             mesh=[(-ref*0.02, -L_head*0.48), (L_head*0.30, -L_head*0.58),
                   (L_head*0.70, -L_head*0.54), (L_head*0.98, -L_head*0.32),
                   (L_head*1.14, -L_head*0.02), (L_head*1.08, L_head*0.24),
                   (L_head*0.66, L_head*0.44), (L_head*0.26, L_head*0.40),
                   (-ref*0.02, L_head*0.36)],
             color='#4A4A4A', outline='#222222', layer=33)

        bone('ear_upper', 'head', (L_head*0.30, -L_head*0.36), L_e1,
             taper(L_e1, ref*0.030, ref*0.036),
             color='#363636', outline='#222222', layer=34)
        bone('ear_lower', 'ear_upper', (L_e1, 0), L_e2,
             taper(L_e2, ref*0.036, ref*0.020, back=ref*0.010),
             color='#2E2E2E', outline='#1A1A1A', layer=35)

        bone('tail_1', 'root', (-ref*0.12, -ref*0.040), L_t1,
             taper(L_t1, ref*0.028, ref*0.020),
             color='#2A2A2A', layer=20)
        bone('tail_2', 'tail_1', (L_t1, 0), L_t2,
             taper(L_t2, ref*0.020, ref*0.010, back=ref*0.010),
             color='#2A2A2A', layer=21)

        for side, base, cu, cm, cl in [('L', 40, '#444444', '#3C3C3C', '#2A2A2A'),
                                       ('R', 10, '#666666', '#5E5E5E', '#4E4E4E')]:
            bone(f'humerus_{side}', 'spine_upper', measure['sh_off'], L_hum,
                 taper(L_hum, ref*0.044, ref*0.028, back=ref*0.045),
                 color=cu, layer=base+1)
            bone(f'radius_{side}', f'humerus_{side}', (L_hum, 0), L_rad,
                 taper(L_rad, ref*0.028, ref*0.018, back=ref*0.026),
                 color=cm, layer=base+2)
            bone(f'paw_front_{side}', f'radius_{side}', (L_rad, 0), L_paw_f,
                 paw_shape(L_paw_f, ref*0.021), color=cl,
                 outline='#1A1A1A', layer=base+3)

            bone(f'femur_{side}', 'spine_lower', measure['hip_off'], L_fem,
                 taper(L_fem, ref*0.048, ref*0.029, back=ref*0.045),
                 color=cu, layer=base+4)
            bone(f'tibia_{side}', f'femur_{side}', (L_fem, 0), L_tib,
                 taper(L_tib, ref*0.029, ref*0.018, back=ref*0.027),
                 color=cm, layer=base+5)
            bone(f'paw_hind_{side}', f'tibia_{side}', (L_tib, 0), L_paw_h,
                 paw_shape(L_paw_h, ref*0.021), color=cl,
                 outline='#1A1A1A', layer=base+6)

        return rig


    measure = measure_skeleton(stab_kps)
    rig = build_rig(measure)
    calib = {
        'sh_off_intercept': measure['sh_off_intercept'],
        'sh_off_slope': measure['sh_off_slope'],
        'hip_off_intercept': measure['hip_off_intercept'],
        'hip_off_slope': measure['hip_off_slope'],
    }
    print(f"\nRig built: {len(rig)} bones")
    return calib, rig


@app.cell
def _():
    return


@app.cell
def _(KEYPOINT_NAMES, calib, get_kp, np, rig, stab_kps):
    def compute_root_motion(stab_list, target_spine=200.0,
                            detrend_win=31, bounce_gain=0.55):
        pel = []
        for fr in stab_list:
            neck = get_kp(fr, 'Neck')
            tail = get_kp(fr, 'root_of_tail')
            sl = np.linalg.norm(tail - neck)
            if sl < 1e-6:
                pel.append(pel[-1] if pel else np.zeros(2))
                continue
            sc = target_spine / sl
            hips = (get_kp(fr, 'L_Hip') + get_kp(fr, 'R_Hip')) / 2
            pel.append((tail * 0.55 + hips * 0.45) * sc)
        pel = np.array(pel)

        def moving_avg(x, win):
            half = win // 2
            padded = np.pad(x, (half, half), mode='edge')
            return np.convolve(padded, np.ones(win) / win, mode='valid')[:len(x)]

        y_base = moving_avg(pel[:, 1], detrend_win)
        x_base = moving_avg(pel[:, 0], detrend_win)
        y_gait = (pel[:, 1] - y_base) * bounce_gain
        detrended = np.stack([pel[:, 0] - x_base, pel[:, 1] - y_base], axis=1)
        vel = np.gradient(detrended, axis=0)

        print(f"gait bounce range : {np.ptp(y_gait):.1f}px "
              f"({np.ptp(y_gait)/target_spine*100:.1f}% of spine)")
        return y_gait, vel


    def estimate_stride_period(stab_list, lo=6, hi=90, verbose=True):
        sig = np.zeros(len(stab_list))
        for name_a, name_b in [('L_Shoulder', 'L_Elbow'), ('L_Hip', 'L_Knee')]:
            s = []
            for fr in stab_list:
                d = get_kp(fr, name_b) - get_kp(fr, name_a)
                s.append(np.arctan2(d[1], d[0]))
            s = np.unwrap(np.array(s))
            s = s - np.convolve(np.pad(s, (20, 20), mode='edge'),
                                np.ones(41) / 41, 'valid')[:len(s)]
            sig += s / (s.std() + 1e-9)

        ac = np.correlate(sig, sig, mode='full')[len(sig) - 1:]
        ac = ac / ac[0]
        hi = min(hi, len(ac) - 2)

        neg = np.nonzero(ac[:hi] < 0)[0]
        start = int(neg[0]) if len(neg) else lo
        period = None
        for L in range(max(start, lo), hi - 1):
            if ac[L] > ac[L - 1] and ac[L] >= ac[L + 1] and ac[L] > 0.05:
                period = L
                break
        if period is None:
            period = int(np.argmax(ac[max(start, lo):hi]) + max(start, lo))

        if verbose:
            print(f"stride period : {period} frames ({30.0/period:.2f} Hz), "
                  f"far-side limbs delayed by {period//2} frames")
            print("  autocorrelation:", "  ".join(
                f"{L}:{ac[L]:+.2f}" for L in range(6, min(hi, 46), 4)))
        return period


    class TailDynamics:
        def __init__(self, stiffness=0.20, damping=0.82,
                     drag=0.055, lift=0.075, lag=0.60, limit=0.45):
            self.stiffness, self.damping = stiffness, damping
            self.drag, self.lift, self.lag, self.limit = drag, lift, lag, limit
            self.reset()

        def reset(self):
            self.a1 = self.v1 = self.a2 = self.v2 = 0.0

        def step(self, vel):
            t1 = float(np.clip(-vel[0]*self.drag + vel[1]*self.lift,
                               -self.limit, self.limit))
            self.v1 = (self.v1 + (t1 - self.a1) * self.stiffness) * self.damping
            self.a1 += self.v1
            t2 = self.a1 * self.lag
            self.v2 = (self.v2 + (t2 - self.a2) * self.stiffness) * self.damping
            self.a2 += self.v2
            return self.a1, self.a2


    def wrap_angle(a):
        return float(np.arctan2(np.sin(a), np.cos(a)))


    PAW_FOLLOW = 0.0
    PAW_LIMIT = (-1.75, 0.30)
    SPINE_FOLD_GAIN = 2.4
    SPINE_REST_FOLD = -0.14

    ELBOW_LIMIT = (-0.30, 1.45)
    STIFLE_LIMIT = (-1.90, 0.15)


    def clamp_joint(child_world, parent_world, limits, softness=0.35):
        lo, hi = limits
        rel = wrap_angle(child_world - parent_world)
        span = (hi - lo) * softness
        if span < 1e-6:
            rel = float(np.clip(rel, lo, hi))
        elif rel > hi - span:
            rel = (hi - span) + span * float(np.tanh((rel - (hi - span)) / span))
        elif rel < lo + span:
            rel = (lo + span) - span * float(np.tanh(((lo + span) - rel) / span))
        return parent_world + rel


    def near_limb_angles(frame_data, facing):
        def a(p, q):
            d = get_kp(frame_data, q) - get_kp(frame_data, p)
            t = float(np.arctan2(d[1], d[0]))
            return float(np.pi - t) if facing < 0 else t

        return {
            'hum': a('L_Shoulder', 'L_Elbow'),
            'rad': a('L_Elbow', 'L_F_Paw'),
            'fem': a('L_Hip', 'L_Knee'),
            'tib': a('L_Knee', 'L_B_Paw'),
        }


    def precompute_smooth_leg_angles(stab_list, min_cutoff=0.6, beta=0.004):
        raw = {'hum': [], 'rad': [], 'fem': [], 'tib': []}
        for fr in stab_list:
            tail_p = get_kp(fr, 'root_of_tail')
            facing = 1.0 if get_kp(fr, 'Nose')[0] >= tail_p[0] else -1.0
            ang = near_limb_angles(fr, facing)
            for k in raw:
                raw[k].append(ang[k])

        def one_euro(series, min_cutoff, beta):
            x = np.array(series, dtype=float)
            out = np.zeros_like(x)
            out[0] = x[0]
            x_prev = x[0]
            dx_prev = 0.0
            for i in range(1, len(x)):
                dx = x[i] - x_prev
                dx_smooth = dx_prev + 0.1 * (dx - dx_prev)
                cutoff = min_cutoff + beta * abs(dx_smooth)
                alpha = cutoff / (cutoff + 1.0)
                out[i] = x_prev + alpha * (x[i] - x_prev)
                x_prev = out[i]
                dx_prev = dx_smooth
            return out

        smoothed = {}
        for k in raw:
            unwrapped = np.unwrap(np.array(raw[k]))
            smoothed[k] = one_euro(unwrapped, min_cutoff, beta)

        n = len(stab_list)
        table = []
        for i in range(n):
            table.append({k: float(smoothed[k][i]) for k in smoothed})

        pad = n
        far_table = []
        for i in range(-pad, n):
            j = min(max(i, 0), n - 1)
            far_table.append({k: float(smoothed[k][j]) for k in smoothed})

        return table, far_table, pad


    def solve_pose(frame_data, rig, root_y=0.0, vel=None, tail_sim=None,
                   fold_ref=0.0, far_frame=None,
                   fold_gain=SPINE_FOLD_GAIN, rest_fold=SPINE_REST_FOLD,
                   target_spine=200.0, root_anchor=(300, 235), calib=None,
                   frame_idx=None, far_idx=None, smooth_table=None,
                   far_table=None, far_pad=0):
        kp = {n: get_kp(frame_data, n) for n in KEYPOINT_NAMES}
        neck_p, tail_p = kp['Neck'], kp['root_of_tail']
        spine_len = np.linalg.norm(tail_p - neck_p)
        if spine_len < 1e-6:
            return None
        sc = target_spine / spine_len
        facing = 1.0 if kp['Nose'][0] >= tail_p[0] else -1.0

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

        waist = pelvis * 0.5 + neck_p * 0.5
        waist_adj = waist + np.array([0.0, (shs[1] + hips[1]) / 2 - waist[1]]) * 0.35

        w = {'root': 0.0}

        a_lower = ang(rs(pelvis), rs(waist_adj))
        a_upper = ang(rs(waist_adj), rs(neck_p))
        raw_fold = wrap_angle(a_upper - a_lower)
        fold = rest_fold + (raw_fold - fold_ref) * fold_gain
        mid = a_lower + raw_fold * 0.5
        w['spine_lower'] = mid - fold * 0.5
        w['spine_upper'] = mid + fold * 0.5

        if calib is not None:
            sh_dyn = calib['sh_off_intercept'] + calib['sh_off_slope'] * raw_fold
            hip_dyn = calib['hip_off_intercept'] + calib['hip_off_slope'] * raw_fold
            offset_override = {'humerus_L': sh_dyn, 'humerus_R': sh_dyn,
                               'femur_L': hip_dyn, 'femur_R': hip_dyn}
        else:
            offset_override = {}

        w['neck'] = ang(rs(neck_p), rs(kp['Nose']))
        w['head'] = w['neck']
        w['ear_upper'] = w['head'] + 1.50
        w['ear_lower'] = w['ear_upper'] + 0.25

        if smooth_table is not None and frame_idx is not None:
            near = smooth_table[frame_idx]
            if far_table is not None and far_idx is not None:
                far = far_table[far_idx + far_pad]
            else:
                far = near
        else:
            near = near_limb_angles(frame_data, facing)
            far = near_limb_angles(far_frame, facing) if far_frame is not None else near

        for side, src in (('L', near), ('R', far)):
            a_hum = src['hum']
            a_rad = clamp_joint(src['rad'], a_hum, ELBOW_LIMIT)
            a_fem = src['fem']
            a_tib = clamp_joint(src['tib'], a_fem, STIFLE_LIMIT, softness=0.45)

            w[f'humerus_{side}'] = a_hum
            w[f'radius_{side}'] = a_rad
            w[f'femur_{side}'] = a_fem
            w[f'tibia_{side}'] = a_tib
            w[f'paw_front_{side}'] = clamp_joint(
                wrap_angle(a_rad) * PAW_FOLLOW, a_rad, PAW_LIMIT)
            w[f'paw_hind_{side}'] = clamp_joint(
                wrap_angle(a_tib) * PAW_FOLLOW, a_tib, PAW_LIMIT)

        tail_base = ang(rs(neck_p), rs(tail_p))
        d1, d2 = tail_sim.step(vel) if (tail_sim is not None and vel is not None) else (0.0, 0.0)
        w['tail_1'] = tail_base + 0.40 + d1
        w['tail_2'] = w['tail_1'] + 0.30 + d2

        local = {'root': 0.0}
        for name, b in rig.items():
            if name != 'root':
                local[name] = wrap_angle(w[name] - w.get(b['parent'], 0.0))

        return {'root_pos': root_pos, 'scale': sc,
                'local': local, 'facing': facing,
                'offset_override': offset_override}


    def calibrate_spine_fold(stab_list, rig, root_y, vel):
        folds = []
        for i, fr in enumerate(stab_list):
            p = solve_pose(fr, rig, root_y=root_y[i], vel=vel[i], tail_sim=None,
                           fold_ref=0.0, fold_gain=1.0, rest_fold=0.0, calib=calib)
            if p is not None:
                folds.append(p['local']['spine_upper'])
        ref = float(np.mean(folds))
        print(f"waist fold reference : {np.degrees(ref):.1f} deg")
        return ref


    root_y_series, vel_series = compute_root_motion(stab_kps)
    stride_period = estimate_stride_period(stab_kps)
    half_stride = max(1, stride_period // 2)
    spine_fold_ref = calibrate_spine_fold(stab_kps, rig, root_y_series, vel_series)
    smooth_leg_table, far_leg_table, far_pad = precompute_smooth_leg_angles(stab_kps)


    def far_frame_for(i):
        return stab_kps[(i - half_stride) % len(stab_kps)]


    def verify_phase_shift():
        sim = TailDynamics()
        L, R = [], []
        for i, fr in enumerate(stab_kps):
            p = solve_pose(fr, rig, root_y=root_y_series[i], vel=vel_series[i],
                           tail_sim=sim, fold_ref=spine_fold_ref,
                           far_frame=far_frame_for(i), calib=calib,
                           frame_idx=i, far_idx=i - half_stride,
                           smooth_table=smooth_leg_table,
                           far_table=far_leg_table, far_pad=far_pad)
            if p is not None:
                L.append(p['local']['femur_L'])
                R.append(p['local']['femur_R'])
        L = np.unwrap(np.array(L)); R = np.unwrap(np.array(R))
        sep = np.degrees(np.abs(L - R))
        print(f"L/R hind separation : mean={sep.mean():5.1f} max={sep.max():5.1f} deg")
        if sep.mean() < 8:
            print("  WARNING both sides are nearly identical -- "
                  "the stride estimate is probably too short")


    def check_motion():
        sim = TailDynamics()
        keys = ['spine_upper', 'radius_L', 'tibia_L', 'radius_R', 'tibia_R',
                'paw_front_L', 'paw_hind_L', 'tail_1']
        rec = {k: [] for k in keys}
        for i, fr in enumerate(stab_kps):
            p = solve_pose(fr, rig, root_y=root_y_series[i], vel=vel_series[i],
                           tail_sim=sim, fold_ref=spine_fold_ref,
                           far_frame=far_frame_for(i), calib=calib,
                           frame_idx=i, far_idx=i - half_stride,
                           smooth_table=smooth_leg_table,
                           far_table=far_leg_table, far_pad=far_pad)
            if p is None:
                continue
            for k in keys:
                rec[k].append(p['local'][k])
        for k in keys:
            v = np.degrees(np.unwrap(np.array(rec[k])))
            v = v - np.round(v.mean() / 360.0) * 360.0
            print(f"  {k:13} mean={v.mean():7.1f}  range={np.ptp(v):6.1f}  "
                  f"[{v.min():7.1f}, {v.max():7.1f}] deg")


    verify_phase_shift()
    check_motion()
    return (
        TailDynamics,
        far_frame_for,
        far_leg_table,
        far_pad,
        half_stride,
        near_limb_angles,
        root_y_series,
        smooth_leg_table,
        solve_pose,
        spine_fold_ref,
        vel_series,
    )


@app.cell
def _():
    return


@app.cell
def _(
    Image,
    ImageDraw,
    TailDynamics,
    calib,
    np,
    plt,
    rig,
    root_y_series,
    solve_pose,
    stab_kps,
    vel_series,
):
    def compute_world_transforms(rig, pose):
        world = {'root': {'pos': np.array(pose['root_pos'], dtype=float),
                          'angle': 0.0}}
        local = pose['local']
        override_map = pose.get('offset_override', {}) or {}
        resolved = {'root'}
        pending = [n for n in rig if n != 'root']

        while pending:
            progressed = False
            for name in list(pending):
                b = rig[name]
                if b['parent'] not in resolved:
                    continue
                pw = world[b['parent']]
                override = override_map.get(name)
                if override is not None:
                    ox, oy = float(override[0]), float(override[1])
                else:
                    ox, oy = float(b['offset'][0]), float(b['offset'][1])
                c, s = np.cos(pw['angle']), np.sin(pw['angle'])
                world[name] = {
                    'pos': pw['pos'] + np.array([ox * c - oy * s,
                                                 ox * s + oy * c]),
                    'angle': pw['angle'] + local.get(name, 0.0),
                }
                resolved.add(name)
                pending.remove(name)
                progressed = True
            if not progressed:
                print("WARNING unresolved bones:", pending)
                break
        return world


    JOINT_STYLE = {
        'spine_upper':  (0.024, '#4A4A4A', '#222222', 2),
        'neck':         (0.022, '#4A4A4A', '#222222', 2),
        'head':         (0.016, '#4A4A4A', '#222222', 1),
        'tail_1':       (0.018, '#3A3A3A', '#222222', 1),
        'tail_2':       (0.012, '#4A4A4A', '#333333', 1),
        'ear_upper':    (0.013, '#4A4A4A', '#222222', 1),
        'ear_lower':    (0.013, '#4A4A4A', '#222222', 1),
        'humerus_L':    (0.040, '#505050', '#222222', 2),
        'radius_L':     (0.027, '#4A4A4A', '#222222', 2),
        'paw_front_L':  (0.017, '#3A3A3A', '#1A1A1A', 1),
        'femur_L':      (0.038, '#505050', '#222222', 2),
        'tibia_L':      (0.027, '#4A4A4A', '#222222', 2),
        'paw_hind_L':   (0.017, '#3A3A3A', '#1A1A1A', 1),
        'humerus_R':    (0.036, '#5E5E5E', '#454545', 1),
        'radius_R':     (0.025, '#5A5A5A', '#454545', 1),
        'paw_front_R':  (0.015, '#565656', '#3A3A3A', 1),
        'femur_R':      (0.034, '#5E5E5E', '#454545', 1),
        'tibia_R':      (0.025, '#5A5A5A', '#454545', 1),
        'paw_hind_R':   (0.015, '#565656', '#3A3A3A', 1),
    }
    END_STYLE = {}

    def render_rig(rig, world, facing=1.0, canvas_size=(600, 400),
                   ref=200.0, show_root=False):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        d = ImageDraw.Draw(canvas)

        def to_world(mesh, wt):
            c, s = np.cos(wt['angle']), np.sin(wt['angle'])
            px, py = wt['pos']
            return [(int(x * c - y * s + px), int(x * s + y * c + py))
                    for (x, y) in mesh]

        def disc(pos, r, fill, outline='#222222', width=1):
            r = int(r)
            d.ellipse([int(pos[0]) - r, int(pos[1]) - r,
                       int(pos[0]) + r, int(pos[1]) + r],
                      fill=fill, outline=outline, width=width)

        for name, b in sorted(rig.items(), key=lambda kv: kv[1]['layer']):
            if b['mesh'] is None or name not in world:
                continue
            poly = to_world(b['mesh'], world[name])
            if b['outline']:
                d.polygon(poly, fill=b['color'], outline=b['outline'])
            else:
                d.polygon(poly, fill=b['color'])

        if 'head' in world:
            hw = world['head']
            hl = rig['head']['length']
            c, s = np.cos(hw['angle']), np.sin(hw['angle'])

            def head_pt(x, y):
                return np.array([x * c - y * s + hw['pos'][0],
                                 x * s + y * c + hw['pos'][1]])

            mz = head_pt(hl * 0.98, hl * 0.06)
            mr = hl * 0.26
            d.ellipse([int(mz[0] - mr), int(mz[1] - mr * 0.68),
                       int(mz[0] + mr), int(mz[1] + mr * 0.68)],
                      fill='#5A5A5A', outline='#333333', width=1)
            nsp = head_pt(hl * 1.16, hl * 0.04)
            d.ellipse([int(nsp[0]) - 4, int(nsp[1]) - 3,
                       int(nsp[0]) + 4, int(nsp[1]) + 3], fill='#1A1A1A')
            eye = head_pt(hl * 0.58, -hl * 0.16)
            disc(eye, 5, 'white', '#222222', 1)
            disc(eye, 2, '#1A1A1A', None, 0)

        for name, (r, fill, outline, wdt) in JOINT_STYLE.items():
            if name in world:
                disc(world[name]['pos'], ref * r, fill, outline, wdt)

        for name, (r, fill, outline, wdt) in END_STYLE.items():
            if name not in world:
                continue
            wt = world[name]
            L = rig[name]['length']
            endp = wt['pos'] + np.array([np.cos(wt['angle']),
                                         np.sin(wt['angle'])]) * L
            disc(endp, ref * r, fill, outline, wdt)

        if show_root and 'root' in world:
            disc(world['root']['pos'], 6, '#D03030', '#701010', 2)

        if facing < 0:
            canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
        return canvas


    def test_rig_frame(idx=0, show_root=True):
        sim = TailDynamics()
        for k in range(max(0, idx - 25), idx):
            solve_pose(stab_kps[k], rig, root_y=root_y_series[k],
                       vel=vel_series[k], tail_sim=sim, calib=calib)
        pose = solve_pose(stab_kps[idx], rig, root_y=root_y_series[idx],
                          vel=vel_series[idx], tail_sim=sim, calib=calib)
        if pose is None:
            print("bad frame")
            return
        world = compute_world_transforms(rig, pose)
        img = render_rig(rig, world, facing=pose['facing'], show_root=show_root)
        plt.figure(figsize=(9, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Rigged - frame {idx} (FK limbs + pastern)')
        plt.savefig(f'rig_frame_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"root {pose['root_pos']}, facing {pose['facing']:+.0f}")

    test_rig_frame(60)
    return compute_world_transforms, render_rig


@app.cell
def _():
    return


@app.cell
def _(
    Image,
    ImageDraw,
    KEYPOINT_NAMES,
    TailDynamics,
    calib,
    compute_world_transforms,
    far_frame_for,
    get_kp,
    np,
    plt,
    render_rig,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def rig_space_keypoints(frame_data, target_spine=200.0,
                            root_anchor=(300, 235), root_y=0.0):
        kp = {n: get_kp(frame_data, n) for n in KEYPOINT_NAMES}
        neck_p, tail_p = kp['Neck'], kp['root_of_tail']
        spine_len = np.linalg.norm(tail_p - neck_p)
        if spine_len < 1e-6:
            return None, 1.0
        sc = target_spine / spine_len
        facing = 1.0 if kp['Nose'][0] >= tail_p[0] else -1.0
        hips = (kp['L_Hip'] + kp['R_Hip']) / 2
        pelvis = tail_p * 0.55 + hips * 0.45
        root_pos = np.array([root_anchor[0], root_anchor[1] + root_y], dtype=float)

        out = {}
        for n, p in kp.items():
            q = root_pos + (p - pelvis) * sc
            if facing < 0:
                q = np.array([2.0 * root_pos[0] - q[0], q[1]])
            out[n] = q
        return out, facing


    JOINT_MAP = [
        ('neck',        'Neck'),
        ('humerus_L',   'L_Shoulder'),
        ('radius_L',    'L_Elbow'),
        ('paw_front_L', 'L_F_Paw'),
        ('femur_L',     'L_Hip'),
        ('tibia_L',     'L_Knee'),
        ('paw_hind_L',  'L_B_Paw'),
        ('tail_1',      'root_of_tail'),
    ]

    OVERLAY_BONES = [
        ('Neck', 'L_Shoulder'), ('L_Shoulder', 'L_Elbow'), ('L_Elbow', 'L_F_Paw'),
        ('Neck', 'root_of_tail'), ('root_of_tail', 'L_Hip'),
        ('L_Hip', 'L_Knee'), ('L_Knee', 'L_B_Paw'), ('Neck', 'Nose'),
    ]


    def _solved_pose(i, sim):
        return solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                          vel=vel_series[i], tail_sim=sim,
                          fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                          calib=calib)


    def joint_error_report():
        sim = TailDynamics()
        errs = {b: [] for b, _ in JOINT_MAP}
        for i in range(len(stab_kps)):
            pose = _solved_pose(i, sim)
            if pose is None:
                continue
            world = compute_world_transforms(rig, pose)
            kps, _ = rig_space_keypoints(stab_kps[i], root_y=root_y_series[i])
            if kps is None:
                continue
            for b, k in JOINT_MAP:
                if b in world:
                    errs[b].append(float(np.linalg.norm(world[b]['pos'] - kps[k])))

        print("joint placement error (canonical px, spine = 200):")
        means = []
        for b, k in JOINT_MAP:
            v = np.array(errs[b])
            means.append(v.mean())
            flag = '   <-- rig problem' if v.mean() > 18 else ''
            print(f"  {b:12} vs {k:14} mean={v.mean():6.1f} "
                  f"p90={np.percentile(v, 90):6.1f}{flag}")
        print(f"  {'OVERALL':12} {'':17} mean={np.mean(means):6.1f}")


    def jitter_report():
        sim = TailDynamics()
        keys = ['spine_upper', 'humerus_L', 'radius_L', 'femur_L', 'tibia_L',
                'paw_front_L', 'paw_hind_L', 'neck', 'tail_1']
        rec = {k: [] for k in keys}
        for i in range(len(stab_kps)):
            pose = _solved_pose(i, sim)
            if pose is None:
                continue
            for k in keys:
                rec[k].append(pose['local'][k])
        print("per-frame angular change (deg):")
        for k in keys:
            v = np.degrees(np.unwrap(np.array(rec[k])))
            d = np.abs(np.diff(v))
            flag = '   <-- jitter' if d.max() > 25 else ''
            print(f"  {k:13} mean={d.mean():5.2f}  p95={np.percentile(d, 95):6.2f}  "
                  f"max={d.max():6.2f}{flag}")


    def render_diagnostic(idx):
        sim = TailDynamics()
        for k in range(max(0, idx - 30), idx):
            _solved_pose(k, sim)
        pose = _solved_pose(idx, sim)
        if pose is None:
            print("bad frame")
            return
        world = compute_world_transforms(rig, pose)

        img = render_rig(rig, world, facing=1.0, canvas_size=(600, 400))
        kps, _ = rig_space_keypoints(stab_kps[idx], root_y=root_y_series[idx])

        if kps is not None:
            d = ImageDraw.Draw(img)
            for a, b in OVERLAY_BONES:
                d.line([tuple(kps[a].astype(int)), tuple(kps[b].astype(int))],
                       fill=(220, 40, 40, 255), width=2)
            for bone_name, kp_name in JOINT_MAP:
                p = kps[kp_name].astype(int)
                if bone_name in world:
                    q = world[bone_name]['pos'].astype(int)
                    d.line([tuple(p), tuple(q)], fill=(255, 150, 0, 255), width=2)
                d.ellipse([p[0]-4, p[1]-4, p[0]+4, p[1]+4],
                          fill=(20, 200, 60, 255), outline=(0, 90, 20, 255))

        if pose['facing'] < 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        plt.figure(figsize=(11, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Frame {idx} - green = tracked keypoint, '
                  f'orange = offset to rig joint')
        plt.savefig(f'diag_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()


    def contact_sheet(start=50, step=12, n=6):
        sim = TailDynamics()
        frames = []
        for i in range(len(stab_kps)):
            pose = _solved_pose(i, sim)
            if start <= i < start + step * n and (i - start) % step == 0 and pose:
                world = compute_world_transforms(rig, pose)
                frames.append((i, render_rig(rig, world, facing=pose['facing'],
                                             canvas_size=(600, 400))))
        cols = 3
        rows = (len(frames) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.8 * rows))
        flat = np.array(axes).ravel()
        for ax, (i, im) in zip(flat, frames):
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(f'frame {i}', fontsize=11)
        for ax in flat[len(frames):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('contact_sheet.png', dpi=130, bbox_inches='tight')
        plt.show()


    joint_error_report()
    contact_sheet()
    jitter_report()
    render_diagnostic(60)
    render_diagnostic(50)
    return JOINT_MAP, rig_space_keypoints


@app.cell
def _(KP_INDEX, stab_kps):
    scores_elbow = [fr['scores'][KP_INDEX['L_Elbow']] for fr in stab_kps]
    scores_paw   = [fr['scores'][KP_INDEX['L_F_Paw']] for fr in stab_kps]
    print("L_Elbow low-confidence (<0.3) frames:", sum(s < 0.3 for s in scores_elbow), "/", len(scores_elbow))
    print("L_F_Paw low-confidence (<0.3) frames:", sum(s < 0.3 for s in scores_paw), "/", len(scores_paw))
    return


@app.cell
def _(
    KEYPOINT_NAMES,
    TailDynamics,
    compute_world_transforms,
    far_frame_for,
    get_kp,
    np,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    sim = TailDynamics()
    errs_c = []
    for i in range(len(stab_kps)):
        fr = stab_kps[i]
        pose = solve_pose(fr, rig, root_y=root_y_series[i], vel=vel_series[i],
                           tail_sim=sim, fold_ref=spine_fold_ref, far_frame=far_frame_for(i))
        if pose is None:
            continue
        world = compute_world_transforms(rig, pose)

        kp = {n: get_kp(fr, n) for n in KEYPOINT_NAMES}
        neck_p, tail_p = kp['Neck'], kp['root_of_tail']
        spine_len = np.linalg.norm(tail_p - neck_p)
        if spine_len < 1e-6:
            continue
        sc = 200.0 / spine_len
        facing = 1.0 if kp['Nose'][0] >= tail_p[0] else -1.0
        hips = (kp['L_Hip'] + kp['R_Hip']) / 2
        pelvis = tail_p * 0.55 + hips * 0.45
        waist = pelvis * 0.5 + neck_p * 0.5
        shs = (kp['L_Shoulder'] + kp['R_Shoulder']) / 2
        waist_adj = waist + np.array([0.0, (shs[1] + hips[1]) / 2 - waist[1]]) * 0.35

        root_pos = np.array([300, 235 + root_y_series[i]], dtype=float)
        def rs(p):
            q = root_pos + (np.asarray(p, dtype=float) - pelvis) * sc
            if facing < 0:
                q = np.array([2.0 * root_pos[0] - q[0], q[1]])
            return q

        waist_canonical = rs(waist_adj)
        errs_c.append(np.linalg.norm(world['spine_upper']['pos'] - waist_canonical))

    errs_c = np.array(errs_c)
    print("[DIAG-C] spine_upper pivot vs tracked waist: mean=", errs_c.mean(), " max=", errs_c.max())
    return


@app.cell
def _(
    JOINT_MAP,
    TailDynamics,
    compute_world_transforms,
    far_frame_for,
    np,
    rig,
    rig_space_keypoints,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def joint_error_report_test(fold_gain_test):
        sim = TailDynamics()
        errs = {b: [] for b, _ in JOINT_MAP}
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              fold_gain=fold_gain_test)
            if pose is None:
                continue
            world = compute_world_transforms(rig, pose)
            kps, _ = rig_space_keypoints(stab_kps[i], root_y=root_y_series[i])
            if kps is None:
                continue
            for b, k in JOINT_MAP:
                if b in world:
                    errs[b].append(float(np.linalg.norm(world[b]['pos'] - kps[k])))

        print(f"[DIAG-E] fold_gain={fold_gain_test}:")
        for b, k in JOINT_MAP:
            v = np.array(errs[b])
            print(f"  {b:12} mean={v.mean():6.1f}")

    joint_error_report_test(1.0)
    return


@app.cell
def _(
    TailDynamics,
    compute_world_transforms,
    far_frame_for,
    get_kp,
    np,
    rig,
    rig_space_keypoints,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def diagnose_shoulder_error():
        sim = TailDynamics()
        errs, folds, x_off_list = [], [], []
        for i in range(len(stab_kps)):
            fr = stab_kps[i]
            neck = get_kp(fr, 'Neck')
            tail = get_kp(fr, 'root_of_tail')
            sl = np.linalg.norm(neck - tail)
            if sl < 1e-6:
                continue
            hips = (get_kp(fr, 'L_Hip') + get_kp(fr, 'R_Hip')) / 2
            shs = (get_kp(fr, 'L_Shoulder') + get_kp(fr, 'R_Shoulder')) / 2
            pelvis = tail * 0.55 + hips * 0.45
            waist = pelvis * 0.5 + neck * 0.5
            waist_adj = waist + np.array([0.0, (shs[1] + hips[1]) / 2 - waist[1]]) * 0.35
            sc = 200.0 / sl

            pose = solve_pose(fr, rig, root_y=root_y_series[i], vel=vel_series[i],
                              tail_sim=sim, fold_ref=spine_fold_ref,
                              far_frame=far_frame_for(i), fold_gain=1.0)
            if pose is None:
                continue
            world = compute_world_transforms(rig, pose)
            kps, _ = rig_space_keypoints(fr, root_y=root_y_series[i])
            if kps is None:
                continue

            err = float(np.linalg.norm(world['humerus_L']['pos'] - kps['L_Shoulder']))
            a_lower = np.arctan2(*(waist_adj - pelvis)[::-1])
            a_upper = np.arctan2(*(neck - waist_adj)[::-1])
            raw_fold = float(np.arctan2(np.sin(a_upper - a_lower), np.cos(a_upper - a_lower)))

            errs.append(err)
            folds.append(raw_fold)

        errs = np.array(errs); folds = np.array(folds)
        corr = np.corrcoef(errs, np.abs(folds - np.median(folds)))[0, 1]
        print(f"[DIAG-F] humerus_L error: mean={errs.mean():.1f} std={errs.std():.1f}")
        print(f"[DIAG-F] correlation(error, |spine bend deviation|) = {corr:.2f}")

    diagnose_shoulder_error()
    return


@app.cell
def _(
    Image,
    ImageDraw,
    TailDynamics,
    calib,
    compute_world_transforms,
    far_frame_for,
    get_kp,
    np,
    plt,
    render_rig,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def detect_stance(paw_name, speed_percentile=45, min_run=6, smooth_win=5):
        pos = np.array([get_kp(fr, paw_name) for fr in stab_kps])
        vel = np.gradient(pos, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        half = smooth_win // 2
        padded = np.pad(speed, (half, half), mode='edge')
        speed_smooth = np.convolve(
            padded, np.ones(smooth_win) / smooth_win, mode='valid')[:len(speed)]
        thresh = np.percentile(speed_smooth, speed_percentile)
        is_stance = speed_smooth < thresh

        runs = []
        i = 0
        while i < len(is_stance):
            if is_stance[i]:
                j = i
                while j < len(is_stance) and is_stance[j]:
                    j += 1
                if j - i >= min_run:
                    runs.append((i, j))
                i = j
            else:
                i += 1
        return runs


    def compute_unlocked_paw_xy(bone_name):
        sim = TailDynamics()
        xs = [None] * len(stab_kps)
        ys = [None] * len(stab_kps)
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib)
            if pose is None:
                continue
            world = compute_world_transforms(rig, pose)
            if bone_name in world:
                xs[i] = float(world[bone_name]['pos'][0])
                ys[i] = float(world[bone_name]['pos'][1])
        return xs, ys


    def build_lock_anchors(runs, xs_per_frame, ys_per_frame):
        anchors = []
        for (start, end) in runs:
            xvals = [xs_per_frame[i] for i in range(start, end) if xs_per_frame[i] is not None]
            yvals = [ys_per_frame[i] for i in range(start, end) if ys_per_frame[i] is not None]
            if xvals and yvals:
                anchors.append({'start': start, 'end': end,
                                'x': float(np.median(xvals)),
                                'y': float(np.median(yvals))})
        return anchors


    def apply_foot_lock(world, frame_idx, bone_name, anchors, blend=4):
        if bone_name not in world:
            return
        for a in anchors:
            s, e = a['start'], a['end']
            if s <= frame_idx < e:
                run_len = e - s
                ramp = min(blend, run_len // 2)
                w = 1.0 if ramp <= 0 else min(
                    1.0, (frame_idx - s) / ramp, (e - frame_idx) / ramp)
                cur = world[bone_name]['pos']
                world[bone_name]['pos'] = np.array(
                    [cur[0] * (1 - w) + a['x'] * w, cur[1]])
                return


    front_runs = detect_stance('L_F_Paw')
    hind_runs = detect_stance('L_B_Paw')

    front_x_unlocked, front_y_unlocked = compute_unlocked_paw_xy('paw_front_L')
    hind_x_unlocked, hind_y_unlocked = compute_unlocked_paw_xy('paw_hind_L')

    front_anchors = build_lock_anchors(front_runs, front_x_unlocked, front_y_unlocked)
    hind_anchors = build_lock_anchors(hind_runs, hind_x_unlocked, hind_y_unlocked)

    front_ground_y = float(np.median([a['y'] for a in front_anchors])) if front_anchors else None
    hind_ground_y = float(np.median([a['y'] for a in hind_anchors])) if hind_anchors else None

    print(f"front paw stance runs: {len(front_runs)}, ground_y={front_ground_y:.1f}")
    print(f"hind  paw stance runs: {len(hind_runs)}, ground_y={hind_ground_y:.1f}")


    def draw_ground_lines(img):
        d = ImageDraw.Draw(img)
        w, h = img.size
        if front_ground_y is not None:
            y = int(front_ground_y)
            d.line([(0, y), (w, y)], fill=(220, 60, 60, 140), width=1)
        if hind_ground_y is not None:
            y = int(hind_ground_y)
            d.line([(0, y), (w, y)], fill=(60, 60, 220, 140), width=1)


    def render_pose_with_lock(pose, idx, canvas_size=(600, 400)):
        if pose is None:
            return None
        world = compute_world_transforms(rig, pose)
        apply_foot_lock(world, idx, 'paw_front_L', front_anchors)
        apply_foot_lock(world, idx, 'paw_hind_L', hind_anchors)
        img = render_rig(rig, world, facing=1.0, canvas_size=canvas_size)
        draw_ground_lines(img)
        if pose['facing'] < 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


    def preview_foot_lock(idx=60):
        sim = TailDynamics()
        for k in range(max(0, idx - 30), idx):
            solve_pose(stab_kps[k], rig, root_y=root_y_series[k],
                      vel=vel_series[k], tail_sim=sim,
                      fold_ref=spine_fold_ref, far_frame=far_frame_for(k), calib=calib)
        pose = solve_pose(stab_kps[idx], rig, root_y=root_y_series[idx],
                          vel=vel_series[idx], tail_sim=sim,
                          fold_ref=spine_fold_ref, far_frame=far_frame_for(idx), calib=calib)
        img = render_pose_with_lock(pose, idx)
        if img is None:
            print("bad frame")
            return
        plt.figure(figsize=(9, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Foot-locked render - frame {idx}')
        plt.savefig(f'footlock_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()


    def contact_sheet_locked(start=50, step=12, n=6):
        sim = TailDynamics()
        frames = []
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib)
            if start <= i < start + step * n and (i - start) % step == 0 and pose:
                frames.append((i, render_pose_with_lock(pose, i)))
        cols = 3
        rows = (len(frames) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.8 * rows))
        flat = np.array(axes).ravel()
        for ax, (i, im) in zip(flat, frames):
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(f'frame {i}', fontsize=11)
        for ax in flat[len(frames):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig('contact_sheet_locked.png', dpi=130, bbox_inches='tight')
        plt.show()


    preview_foot_lock(60)
    return (
        front_anchors,
        front_runs,
        front_x_unlocked,
        hind_anchors,
        hind_runs,
        hind_x_unlocked,
        render_pose_with_lock,
    )


@app.cell
def _(
    front_anchors,
    front_runs,
    front_x_unlocked,
    hind_anchors,
    hind_runs,
    hind_x_unlocked,
    np,
):
    def measure_slide_v2(runs, xs_unlocked, anchors):
        before, after = [], []
        for (s, e) in runs:
            vals_before = [xs_unlocked[i] for i in range(s, e) if xs_unlocked[i] is not None]
            if vals_before:
                before.append(np.ptp(vals_before))
            a = next((a for a in anchors if a['start'] == s and a['end'] == e), None)
            if a is None:
                continue
            vals_after = []
            for i in range(s, e):
                run_len = e - s
                ramp = min(4, run_len // 2)
                w = 1.0 if ramp <= 0 else min(1.0, (i - s) / ramp, (e - i) / ramp)
                if xs_unlocked[i] is not None:
                    vals_after.append(xs_unlocked[i] * (1 - w) + a['x'] * w)
            if vals_after:
                after.append(np.ptp(vals_after))
        return np.mean(before), np.mean(after)

    fb, fa = measure_slide_v2(front_runs, front_x_unlocked, front_anchors)
    hb, ha = measure_slide_v2(hind_runs, hind_x_unlocked, hind_anchors)
    print(f"front paw slide: before={fb:.1f} after={fa:.1f}")
    print(f"hind  paw slide: before={hb:.1f} after={ha:.1f}")
    return


@app.cell
def _(front_runs, hind_runs, np):
    def run_length_stats(runs, label):
        lens = np.array([e - s for (s, e) in runs])
        print(f"{label}: n={len(lens)}, mean_len={lens.mean():.1f}, "
              f"min={lens.min()}, max={lens.max()}, "
              f"pct_at_blend_or_shorter={100*np.mean(lens <= 8):.0f}%")

    run_length_stats(front_runs, "front runs")
    run_length_stats(hind_runs, "hind runs")
    return


@app.cell
def _():
    return


@app.cell
def _(
    TailDynamics,
    calib,
    cv2,
    far_frame_for,
    far_leg_table,
    far_pad,
    half_stride,
    np,
    raw_kps,
    render_pose_with_lock,
    rig,
    root_y_series,
    smooth_leg_table,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def generate_rig_video(vid_path, raw_list, stab_list, rig,
                           root_y, vel, out_path='rig_three_panel_smooth_v1.mp4',
                           fps=30, panel_h=400, anim_w=600):
        SKEL_IDX = [
            (0, 2), (1, 2), (2, 3), (3, 5), (3, 8),
            (5, 6), (6, 7), (8, 9), (9, 10),
            (3, 4), (4, 11), (4, 14),
            (11, 12), (12, 13), (14, 15), (15, 16),
        ]

        cap = cv2.VideoCapture(vid_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = panel_h / orig_h
        panel_w = int(orig_w * scale)
        total_w = panel_w * 2 + anim_w

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (total_w, panel_h))

        sim = TailDynamics()
        written = 0

        for i, frame_data in enumerate(stab_list):
            ok, orig = cap.read()
            if not ok:
                break

            p1 = cv2.resize(orig, (panel_w, panel_h))
            p2 = p1.copy()

            kps = raw_list[i]['keypoints']
            scs = raw_list[i]['scores']
            for a, b in SKEL_IDX:
                if scs[a] > 0.3 and scs[b] > 0.3:
                    pa = (int(kps[a][0] * scale), int(kps[a][1] * scale))
                    pb = (int(kps[b][0] * scale), int(kps[b][1] * scale))
                    cv2.line(p2, pa, pb, (0, 0, 255), 2)
            for kp, s in zip(kps, scs):
                if s > 0.3:
                    cv2.circle(p2, (int(kp[0] * scale), int(kp[1] * scale)),
                               3, (0, 255, 0), -1)

            pose = solve_pose(frame_data, rig, root_y=root_y[i], vel=vel[i],
                              tail_sim=sim, fold_ref=spine_fold_ref,
                              far_frame=far_frame_for(i), calib=calib,
                              frame_idx=i, far_idx=i - half_stride,
                              smooth_table=smooth_leg_table,
                              far_table=far_leg_table, far_pad=far_pad)
            img = render_pose_with_lock(pose, i, canvas_size=(anim_w, panel_h))
            if img is None:
                p3 = np.full((panel_h, anim_w, 3), 240, dtype=np.uint8)
            else:
                p3 = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

            cv2.putText(p1, 'Original', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(p2, 'Tracked skeleton', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(p3, 'Rigged 2D character (smoothed + foot-locked)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

            out.write(np.hstack([p1, p2, p3]))
            written += 1
            if i % 50 == 0:
                print(f"  frame {i}/{len(stab_list)}")

        cap.release()
        out.release()
        print(f"Saved {written} frames to {out_path}")


    generate_rig_video('dogvideo.mp4', raw_kps, stab_kps, rig,
                       root_y_series, vel_series)
    return


@app.cell
def _():
    return


@app.cell
def _(KP_INDEX, get_kp, np, stab_kps):
    def diagnose_hind_leg_gaps():
        hind_kps = ['L_Hip', 'L_Knee', 'L_B_Paw', 'R_Hip', 'R_Knee', 'R_B_Paw']
        n = len(stab_kps)

        tail_pos = np.array([get_kp(stab_kps[i], 'root_of_tail') for i in range(n)])
        body_speed = np.linalg.norm(np.gradient(tail_pos, axis=0), axis=1)
        body_moving = body_speed > np.percentile(body_speed, 40)

        print("=== (1) Confidence gaps (score < 0.3) ===")
        for name in hind_kps:
            idx = KP_INDEX[name]
            scores = np.array([stab_kps[i]['scores'][idx] for i in range(n)])
            low = scores < 0.3
            low_frames = np.nonzero(low)[0]
            print(f"  {name:10}: {low.sum():3d}/{n} low-conf frames"
                  + (f"  e.g. {low_frames[:8].tolist()}" if low.sum() else ""))

        print("\n=== (2) Motion-freeze (paw nearly still while body moves) ===")
        for name in ['L_B_Paw', 'R_B_Paw']:
            pos = np.array([get_kp(stab_kps[i], name) for i in range(n)])
            paw_speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1)
            frozen = (paw_speed < 0.5) & body_moving
            frozen_frames = np.nonzero(frozen)[0]
            runs = []
            if len(frozen_frames):
                s = frozen_frames[0]; prev = frozen_frames[0]
                for f in frozen_frames[1:]:
                    if f == prev + 1:
                        prev = f
                    else:
                        if prev - s >= 2:
                            runs.append((s, prev))
                        s = f; prev = f
                if prev - s >= 2:
                    runs.append((s, prev))
            print(f"  {name:10}: {frozen.sum():3d} frozen frames, "
                  f"{len(runs)} stuck runs (>=3 frames): {runs[:10]}")

        print("\n=== (3) Per-frame hind paw displacement distribution ===")
        for name in ['L_B_Paw', 'R_B_Paw']:
            pos = np.array([get_kp(stab_kps[i], name) for i in range(n)])
            disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
            print(f"  {name:10}: median step={np.median(disp):.2f}px  "
                  f"p10={np.percentile(disp,10):.2f}  p90={np.percentile(disp,90):.2f}  "
                  f"max={disp.max():.2f}")

    diagnose_hind_leg_gaps()
    return


@app.cell
def _(
    TailDynamics,
    calib,
    far_frame_for,
    np,
    plt,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def diagnose_leg_angle_continuity():
        sim = TailDynamics()
        fem_L, tib_L, fem_R, tib_R = [], [], [], []
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib)
            if pose is None:
                for L in (fem_L, tib_L, fem_R, tib_R):
                    L.append(np.nan)
                continue
            fem_L.append(pose['local']['femur_L'])
            tib_L.append(pose['local']['tibia_L'])
            fem_R.append(pose['local']['femur_R'])
            tib_R.append(pose['local']['tibia_R'])

        fem_L = np.degrees(np.unwrap(np.array(fem_L)))
        tib_L = np.degrees(np.unwrap(np.array(tib_L)))
        fem_R = np.degrees(np.unwrap(np.array(fem_R)))
        tib_R = np.degrees(np.unwrap(np.array(tib_R)))

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        axes[0].plot(fem_L, 'b-', label='femur_L (data-driven)', linewidth=1.5)
        axes[0].plot(fem_R, 'r-', label='femur_R (delayed copy)', linewidth=1.5, alpha=0.8)
        axes[0].set_title('Femur angle over time'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(tib_L, 'b-', label='tibia_L (data-driven)', linewidth=1.5)
        axes[1].plot(tib_R, 'r-', label='tibia_R (delayed copy)', linewidth=1.5, alpha=0.8)
        axes[1].set_title('Tibia angle over time'); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('leg_angle_continuity.png', dpi=130, bbox_inches='tight')
        plt.show()

        print("Plateaus (angular velocity < 0.5 deg/frame for >=3 frames):")
        for name, series in [('femur_L', fem_L), ('tibia_L', tib_L),
                             ('femur_R', fem_R), ('tibia_R', tib_R)]:
            vel = np.abs(np.diff(series))
            flat = vel < 0.5
            runs = []
            i = 0
            while i < len(flat):
                if flat[i]:
                    j = i
                    while j < len(flat) and flat[j]:
                        j += 1
                    if j - i >= 3:
                        runs.append((i, j))
                    i = j
                else:
                    i += 1
            print(f"  {name:10}: {len(runs)} plateaus {runs[:8]}")

    diagnose_leg_angle_continuity()
    return


@app.cell
def _(
    TailDynamics,
    calib,
    far_frame_for,
    np,
    plt,
    rig,
    root_y_series,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def diagnose_lr_phase_conflict():
        sim = TailDynamics()
        fem_L, fem_R, tib_L, tib_R = [], [], [], []
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib)
            if pose is None:
                for L in (fem_L, fem_R, tib_L, tib_R):
                    L.append(np.nan)
                continue
            fem_L.append(pose['local']['femur_L'])
            fem_R.append(pose['local']['femur_R'])
            tib_L.append(pose['local']['tibia_L'])
            tib_R.append(pose['local']['tibia_R'])

        def vel_deg(series):
            return np.diff(np.degrees(np.unwrap(np.array(series))))

        vfl, vfr = vel_deg(fem_L), vel_deg(fem_R)
        vtl, vtr = vel_deg(tib_L), vel_deg(tib_R)

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        axes[0].plot(vfl, 'b-', label='femur_L angular velocity', alpha=0.8)
        axes[0].plot(vfr, 'r-', label='femur_R angular velocity', alpha=0.8)
        axes[0].axhline(0, color='k', linewidth=0.5)
        axes[0].set_title('Femur angular velocity (deg/frame)'); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(vtl, 'b-', label='tibia_L angular velocity', alpha=0.8)
        axes[1].plot(vtr, 'r-', label='tibia_R angular velocity', alpha=0.8)
        axes[1].axhline(0, color='k', linewidth=0.5)
        axes[1].set_title('Tibia angular velocity (deg/frame)'); axes[1].legend(); axes[1].grid(alpha=0.3)
        axes[1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('lr_velocity_conflict.png', dpi=130, bbox_inches='tight')
        plt.show()

        print("Frames with |angular velocity| > 8 deg/frame (visible jerk):")
        for name, v in [('femur_L', vfl), ('femur_R', vfr),
                        ('tibia_L', vtl), ('tibia_R', vtr)]:
            print(f"  {name:10}: {np.sum(np.abs(v) > 8):3d} jerky frames, "
                  f"max |vel|={np.abs(v).max():.1f} deg/frame")

    diagnose_lr_phase_conflict()
    return


@app.cell
def _(
    TailDynamics,
    calib,
    far_frame_for,
    far_leg_table,
    far_pad,
    half_stride,
    np,
    rig,
    root_y_series,
    smooth_leg_table,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def verify_smoothing_v2():
        sim = TailDynamics()
        tracks = {'femur_L': [], 'femur_R': [], 'tibia_L': [], 'tibia_R': []}
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib,
                              frame_idx=i, far_idx=i - half_stride,
                              smooth_table=smooth_leg_table,
                              far_table=far_leg_table, far_pad=far_pad)
            if pose is None:
                continue
            for k in tracks:
                tracks[k].append(pose['local'][k])
        print("After wrap-seam fix -- jerky frames (|angular velocity| > 8 deg/frame):")
        for k, series in tracks.items():
            v = np.abs(np.diff(np.degrees(np.unwrap(np.array(series)))))
            print(f"  {k:10}: {np.sum(v > 8):3d} jerky frames, max |vel|={v.max():.1f} deg/frame")

    verify_smoothing_v2()
    return


@app.cell
def _(
    TailDynamics,
    calib,
    far_frame_for,
    far_leg_table,
    far_pad,
    half_stride,
    np,
    plt,
    rig,
    root_y_series,
    smooth_leg_table,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def diagnose_gait_phase():
        sim = TailDynamics()
        ang = {'humerus_L': [], 'humerus_R': [], 'femur_L': [], 'femur_R': []}
        for i in range(len(stab_kps)):
            pose = solve_pose(stab_kps[i], rig, root_y=root_y_series[i],
                              vel=vel_series[i], tail_sim=sim,
                              fold_ref=spine_fold_ref, far_frame=far_frame_for(i),
                              calib=calib,
                              frame_idx=i, far_idx=i - half_stride,
                              smooth_table=smooth_leg_table,
                              far_table=far_leg_table, far_pad=far_pad)
            if pose is None:
                continue
            for k in ang:
                ang[k].append(pose['local'][k])

        fig, ax = plt.subplots(figsize=(16, 6))
        for k, c in [('humerus_L', 'blue'), ('humerus_R', 'cyan'),
                     ('femur_L', 'red'), ('femur_R', 'orange')]:
            series = np.degrees(np.unwrap(np.array(ang[k])))
            series = series - series.mean()
            ax.plot(series, color=c, label=k, linewidth=1.5, alpha=0.8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title('All four upper-limb angles (mean-centred) -- inter-limb phase')
        ax.set_xlabel('Frame'); ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(50, 200)
        plt.tight_layout()
        plt.savefig('gait_phase.png', dpi=130, bbox_inches='tight')
        plt.show()

        ref = np.degrees(np.unwrap(np.array(ang['humerus_L'])))
        ref = ref - ref.mean()
        print("Phase lag relative to humerus_L (frames, +ve = lags behind):")
        for k in ['humerus_R', 'femur_L', 'femur_R']:
            s = np.degrees(np.unwrap(np.array(ang[k]))); s = s - s.mean()
            corr = np.correlate(s, ref, mode='full')
            lag = np.argmax(corr) - (len(ref) - 1)
            print(f"  {k:10}: lag = {lag:+d} frames  (half_stride={half_stride})")

    diagnose_gait_phase()
    return


@app.cell
def _(get_kp, half_stride, np, plt, stab_kps):
    def diagnose_raw_gait():
        front_ang, hind_ang = [], []
        for fr in stab_kps:
            f = get_kp(fr, 'L_F_Paw') - get_kp(fr, 'L_Shoulder')
            h = get_kp(fr, 'L_B_Paw') - get_kp(fr, 'L_Hip')
            front_ang.append(np.arctan2(f[1], f[0]))
            hind_ang.append(np.arctan2(h[1], h[0]))
        front = np.degrees(np.unwrap(np.array(front_ang))); front -= front.mean()
        hind = np.degrees(np.unwrap(np.array(hind_ang))); hind -= hind.mean()

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(front, 'b-', label='front leg (L_Shoulder->L_F_Paw)', linewidth=1.5)
        ax.plot(hind, 'r-', label='hind leg (L_Hip->L_B_Paw)', linewidth=1.5)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_xlim(50, 200); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title('RAW tracked near-side front vs hind swing -- the real gait')
        ax.set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('raw_gait.png', dpi=130, bbox_inches='tight')
        plt.show()

        corr = np.correlate(hind, front, mode='full')
        lag = np.argmax(corr) - (len(front) - 1)
        print(f"raw hind lags front by {lag} frames (stride~{2*half_stride})")

    diagnose_raw_gait()
    return


@app.cell
def _(
    TailDynamics,
    calib,
    far_frame_for,
    far_leg_table,
    far_pad,
    get_kp,
    half_stride,
    near_limb_angles,
    np,
    rig,
    root_y_series,
    smooth_leg_table,
    solve_pose,
    spine_fold_ref,
    stab_kps,
    vel_series,
):
    def diagnose_hind_amplitude():
        raw_fem, smooth_fem, rig_fem = [], [], []
        sim = TailDynamics()
        for i in range(len(stab_kps)):
            fr = stab_kps[i]
            facing = 1.0 if get_kp(fr, 'Nose')[0] >= get_kp(fr, 'root_of_tail')[0] else -1.0
            raw_fem.append(near_limb_angles(fr, facing)['fem'])
            smooth_fem.append(smooth_leg_table[i]['fem'])
            pose = solve_pose(fr, rig, root_y=root_y_series[i], vel=vel_series[i],
                              tail_sim=sim, fold_ref=spine_fold_ref,
                              far_frame=far_frame_for(i), calib=calib,
                              frame_idx=i, far_idx=i - half_stride,
                              smooth_table=smooth_leg_table,
                              far_table=far_leg_table, far_pad=far_pad)
            rig_fem.append(pose['local']['femur_L'] if pose else np.nan)

        for name, s in [('raw tracked', raw_fem), ('smoothed', smooth_fem), ('final rig (clamped)', rig_fem)]:
            v = np.degrees(np.unwrap(np.array(s)))
            print(f"  {name:22}: range={np.ptp(v):6.1f} deg  [{v.min():.0f}, {v.max():.0f}]")

    diagnose_hind_amplitude()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

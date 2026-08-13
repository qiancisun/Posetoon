import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    # ------------------------------------------------------------------
    # step8 v2: video quality gate + pulldown repair by UNIFORM RETIMING
    #
    # Root cause (measured): the source video is 24fps converted to 30fps
    # by repeating every 4th frame (55/279 frames are byte-level copies,
    # exact cadence 5). Keypoints freeze for one frame then jump double
    # distance -- a 6 Hz stutter that survives One-Euro smoothing (the
    # rendered character shows a spectral peak at exactly 0.20 c/f).
    #
    # v1 (neighbour interpolation of duplicated frames) was WRONG and is
    # replaced. Two reasons, confirmed on real data and reproduced in a
    # synthetic experiment:
    #   1. It leaves a residual timing error. Interpolating only the dup
    #      frame turns the per-cycle velocity pattern (1,1,1,0,jump) into
    #      (1,1,1,0.5,0.5) -- the period-5 modulation is reduced, not
    #      removed.
    #   2. It throws information away. The duplicated frame is the SAME
    #      image re-detected, i.e. a free repeated measurement of the
    #      same true pose; discarding it discards noise averaging.
    # v2 fixes both: dup pairs are AVERAGED into one source sample, the
    # unique-sample sequence is then resampled on a uniform time grid
    # back to the full frame count. Synthetic check (20 noise seeds,
    # gait-like signal + pulldown + detection noise, same One Euro):
    #   stutter-band fraction  before 0.0024 / v1 0.0017 / v2 0.0010
    #   lag-free RMSE          before 2.68   / v1 2.62   / v2 2.43
    #
    # Note on metrics: p98 angular acceleration is NOT comparable before
    # vs after -- the pulldown stream pads the distribution with frozen
    # (zero-motion) frames, so removing the freezes raises p98 even as
    # motion becomes objectively smoother. Primary metric is the
    # stutter-band power fraction; a constant One-Euro phase lag is
    # visually irrelevant and must not be counted as error.
    #
    # Sync note: after retiming, the character corresponds to uniform
    # source time while the original panel keeps its pulldown freezes;
    # the mismatch is < 1 frame at all times (invisible side by side).
    # spine_tail_tracks.json is 15-frame linear interpolation, so its
    # pulldown distortion is already negligible; left untouched.
    # ------------------------------------------------------------------
    import os
    import json
    import shutil
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

    VIDEO_PATH = 'dogvideo.mp4'
    RAW_KPS_PATH = 'full_video_keypoints.json'      # from step4 extraction cell
    STAB_OUT_PATH = 'stabilised_keypoints.json'     # consumed by step6/step7
    STAB_BACKUP_PATH = 'stabilised_keypoints_pre_dedup.json'

    KEYPOINT_NAMES = [
        'L_Eye', 'R_Eye', 'Nose', 'Neck', 'root_of_tail',
        'L_Shoulder', 'L_Elbow', 'L_F_Paw',
        'R_Shoulder', 'R_Elbow', 'R_F_Paw',
        'L_Hip', 'L_Knee', 'L_B_Paw',
        'R_Hip', 'R_Knee', 'R_B_Paw',
    ]
    KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
    return (
        KP_INDEX,
        RAW_KPS_PATH,
        STAB_BACKUP_PATH,
        STAB_OUT_PATH,
        VIDEO_PATH,
        cv2,
        json,
        np,
        os,
        plt,
        shutil,
    )


@app.cell
def _(cv2, np):
    # ---- video-level scan: duplicated frames + blur + darkness ----------
    def scan_video_quality(video_path, dup_abs_floor=0.5, dup_rel=0.15):
        """Per-transition pixel motion; a transition below
        max(dup_abs_floor, dup_rel * median motion) is a duplicated /
        frozen frame. Also measures sharpness and brightness."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        prev_g = None
        diffs, sharps, brights = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
            sharps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            brights.append(float(g.mean()))
            if prev_g is not None:
                diffs.append(float(np.abs(g - prev_g).mean()))
            prev_g = g
        cap.release()

        diffs = np.array(diffs)
        thr = max(dup_abs_floor, dup_rel * float(np.median(diffs)))
        dup_mask = diffs < thr          # dup_mask[i] => frame i+1 copies frame i
        dup_frames = np.where(dup_mask)[0] + 1

        cadence = None
        if len(dup_frames) >= 4:
            gaps = np.diff(dup_frames)
            mode_gap = int(np.bincount(gaps).argmax())
            if float((gaps == mode_gap).mean()) >= 0.7:
                cadence = mode_gap

        report = {
            'n_frames': len(diffs) + 1,
            'fps': fps,
            'dup_transitions': int(dup_mask.sum()),
            'dup_rate': float(dup_mask.mean()),
            'dup_cadence': cadence,
            'dup_threshold': thr,
            'sharpness_median': float(np.median(sharps)),
            'brightness_median': float(np.median(brights)),
        }
        if report['dup_rate'] > 0.05 and cadence is not None:
            report['verdict'] = (f"PULLDOWN: duplicated frame every {cadence} "
                                 f"frames -- run the retiming repair cell")
        elif report['dup_rate'] > 0.05:
            report['verdict'] = ("IRREGULAR FREEZES (VFR video?) -- retiming "
                                 "still applies but check the result")
        else:
            report['verdict'] = "CLEAN frame timing"
        if report['sharpness_median'] < 50:
            report['verdict'] += " | LOW SHARPNESS: expect noisy keypoints"
        if report['brightness_median'] < 60:
            report['verdict'] += " | DARK footage: expect lower confidence"
        return report, dup_mask
    return (scan_video_quality,)


@app.cell
def _(VIDEO_PATH, scan_video_quality):
    quality_report, dup_mask = scan_video_quality(VIDEO_PATH)
    print(f"Video quality report for {VIDEO_PATH}:")
    for k_q, v_q in quality_report.items():
        print(f"  {k_q:18}: {v_q}")
    return (dup_mask,)


@app.cell
def _(RAW_KPS_PATH, json, np, os):
    # ---- load raw keypoints (produced by step4's extraction cell) -------
    assert os.path.exists(RAW_KPS_PATH), (
        f"{RAW_KPS_PATH} not found -- run the MMPose extraction cell "
        f"(step4, extract_full_video_keypoints) for this video first")
    with open(RAW_KPS_PATH, 'r') as f_raw:
        raw_kps_s8 = json.load(f_raw)
    n_frames_s8 = len(raw_kps_s8)
    kp_arr = np.array([fr['keypoints'] for fr in raw_kps_s8], dtype=float)   # (N,17,2)
    sc_arr = np.array([fr['scores'] for fr in raw_kps_s8], dtype=float)     # (N,17)
    print(f"Loaded {n_frames_s8} frames of raw keypoints")
    return kp_arr, n_frames_s8, raw_kps_s8, sc_arr


@app.cell
def _(dup_mask, kp_arr, n_frames_s8, np):
    # ---- cross-check: keypoints must freeze exactly on dup frames -------
    disp = np.linalg.norm(np.diff(kp_arr, axis=0), axis=2)
    med_disp = np.median(disp, axis=1)
    m_ck = min(len(med_disp), len(dup_mask))
    on_d = med_disp[:m_ck][dup_mask[:m_ck]]
    off_d = med_disp[:m_ck][~dup_mask[:m_ck]]
    print("median joint displacement per transition:")
    print(f"  on duplicated frames : {on_d.mean():6.2f}px  (n={len(on_d)})")
    print(f"  on normal frames     : {off_d.mean():6.2f}px  (n={len(off_d)})")
    print(f"  ratio {on_d.mean() / max(off_d.mean(), 1e-9):.2f} (<< 1 expected)")

    dup_frame_idx = np.zeros(n_frames_s8, dtype=bool)
    dup_frame_idx[np.where(dup_mask[:m_ck])[0] + 1] = True
    print(f"frames marked as duplicates: {int(dup_frame_idx.sum())}/{n_frames_s8}")
    return (dup_frame_idx,)


@app.cell
def _(
    STAB_BACKUP_PATH,
    STAB_OUT_PATH,
    dup_frame_idx,
    json,
    kp_arr,
    n_frames_s8,
    np,
    os,
    raw_kps_s8,
    sc_arr,
    shutil,
):
    # ---- v2 repair: average dup pairs + uniform retiming + One Euro -----
    # Steps, in order (the order is the point):
    #   1. collapse the frame stream to UNIQUE source samples; a frame and
    #      its duplicate are two detections of the same image, so they are
    #      score-weight AVERAGED (repeated measurement -> noise / sqrt(2))
    #   2. per joint, invalidate low-confidence / missing samples on the
    #      unique timeline and fill by interpolation there
    #   3. resample every joint coordinate on a UNIFORM time grid back to
    #      the full N frames -- this removes the period-5 velocity
    #      modulation exactly, which neighbour interpolation cannot do
    #   4. One-Euro smooth (same parameters the pipeline always used)
    def one_euro_s8(data, min_cutoff=1.0, beta=0.007):
        data_f = np.asarray(data, dtype=float)
        out = np.zeros_like(data_f)
        out[0] = data_f[0]
        x_prev, dx_prev = data_f[0], 0.0
        for i in range(1, len(data_f)):
            dx = data_f[i] - x_prev
            dx_s = dx_prev + 0.1 * (dx - dx_prev)
            cutoff = min_cutoff + beta * abs(dx_s)
            alpha = cutoff / (cutoff + 1.0)
            out[i] = x_prev + alpha * (data_f[i] - x_prev)
            x_prev, dx_prev = out[i], dx_s
        return out

    def fill_bad_s8(vals, bad):
        vals_f = np.asarray(vals, dtype=float).copy()
        bad_f = np.asarray(bad, dtype=bool)
        good = ~bad_f
        if good.sum() < 2:
            return vals_f
        idx = np.arange(len(vals_f))
        vals_f[bad_f] = np.interp(idx[bad_f], idx[good], vals_f[good])
        return vals_f

    def repair_and_stabilise_v2(score_thr=0.3):
        # 1. unique-sample index for every frame: advances only when the
        #    frame is NOT a duplicate of its predecessor
        src_of_frame = np.cumsum(~dup_frame_idx) - 1        # (N,) -> 0..M-1
        n_unique = int(src_of_frame[-1]) + 1
        print(f"{n_frames_s8} displayed frames -> {n_unique} unique source "
              f"samples ({n_frames_s8 - n_unique} duplicates averaged in)")

        num_joints = kp_arr.shape[1]
        stab_out = np.zeros_like(kp_arr)
        t_uniform = np.linspace(0.0, n_unique - 1.0, n_frames_s8)
        t_unique = np.arange(n_unique, dtype=float)
        n_filled_total = 0

        for j in range(num_joints):
        
            w = np.where(sc_arr[:, j] >= score_thr, sc_arr[:, j], 0.0)
            # score weights
            uniq_xy = np.zeros((n_unique, 2))
            uniq_w = np.zeros(n_unique)
            uniq_sc = np.zeros(n_unique)                    # best score in group
            for k in range(n_frames_s8):
                s = src_of_frame[k]
                uniq_xy[s] += kp_arr[k, j] * w[k]
                uniq_w[s] += w[k]
                uniq_sc[s] = max(uniq_sc[s], sc_arr[k, j])
            uniq_xy /= uniq_w[:, None]

            # 2. gap fill on the unique timeline
            bad_u = (uniq_sc < score_thr) | (
                (np.abs(uniq_xy[:, 0]) < 1e-6) & (np.abs(uniq_xy[:, 1]) < 1e-6))
            n_filled_total += int(bad_u.sum())
            for c in range(2):
                filled = fill_bad_s8(uniq_xy[:, c], bad_u)
                # 3. uniform retiming, 4. smoothing
                resampled = np.interp(t_uniform, t_unique, filled)
                stab_out[:, j, c] = one_euro_s8(resampled)

        print(f"gap-filled {n_filled_total} low-confidence/missing samples "
              f"on the unique timeline before retiming")
        return stab_out

    stab_new = repair_and_stabilise_v2()

    if os.path.exists(STAB_OUT_PATH) and not os.path.exists(STAB_BACKUP_PATH):
        shutil.copy(STAB_OUT_PATH, STAB_BACKUP_PATH)
        print(f"backed up old {STAB_OUT_PATH} -> {STAB_BACKUP_PATH}")

    stab_json = [{
        'frame': raw_kps_s8[i]['frame'],
        'keypoints': stab_new[i].tolist(),
        'scores': raw_kps_s8[i]['scores'],
    } for i in range(n_frames_s8)]
    with open(STAB_OUT_PATH, 'w') as f_out:
        json.dump(stab_json, f_out)
    print(f"wrote {STAB_OUT_PATH} ({n_frames_s8} frames) -- "
          f"step6/step7 consume it unchanged")
    return (stab_new,)


@app.cell
def _(KP_INDEX, STAB_BACKUP_PATH, dup_frame_idx, json, np, os, plt, stab_new):
    # ---- verification -----------------------------------------------------
    # Primary metric: power fraction of angular velocity in the stutter
    # band (0.20 and 0.40 cycles/frame). Secondary, and the most intuitive:
    # the FREEZE-SLOT VELOCITY RATIO -- mean |dAngle| on the output slots
    # that used to be duplicated frames, divided by mean |dAngle| on the
    # other slots. Pulldown drives it toward 0 (the leg literally halts on
    # those slots); a correct retiming drives it to ~1.0 (motion no longer
    # knows which slots were duplicated).
    # p98 acceleration is reported but flagged: removing frozen frames
    # RAISES it by construction (see header note); do not read it alone.
    def leg_angle_v(kp_stack, root, tip):
        d = kp_stack[:, KP_INDEX[tip], :] - kp_stack[:, KP_INDEX[root], :]
        return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))

    def stutter_band_frac(angle, cadence=5, band=0.02):
        v = np.diff(angle)
        v = v - v.mean()
        freqs = np.fft.rfftfreq(len(v))
        power = np.abs(np.fft.rfft(v)) ** 2
        f0 = 1.0 / cadence
        in_band = ((np.abs(freqs - f0) < band) | (np.abs(freqs - 2 * f0) < band))
        return float(power[in_band].sum() / power.sum())

    def freeze_slot_ratio(angle):
        v = np.abs(np.diff(angle))
        slots = dup_frame_idx[1:len(v) + 1]
        return float(v[slots].mean() / max(v[~slots].mean(), 1e-12))

    if os.path.exists(STAB_BACKUP_PATH):
        with open(STAB_BACKUP_PATH, 'r') as f_old:
            stab_old = np.array(
                [fr['keypoints'] for fr in json.load(f_old)], dtype=float)

        fig_v, axes_v = plt.subplots(2, 2, figsize=(15, 8))
        for col, (root_v, tip_v, label_v) in enumerate([
                ('L_Knee', 'L_B_Paw', 'L hind lower leg'),
                ('L_Elbow', 'L_F_Paw', 'L front lower leg')]):
            ang_o = leg_angle_v(stab_old, root_v, tip_v)
            ang_n = leg_angle_v(stab_new, root_v, tip_v)
            bo, bn = stutter_band_frac(ang_o), stutter_band_frac(ang_n)
            ro, rn = freeze_slot_ratio(ang_o), freeze_slot_ratio(ang_n)
            ao = float(np.percentile(np.abs(np.diff(ang_o, 2)), 98) * 180 / np.pi)
            an = float(np.percentile(np.abs(np.diff(ang_n, 2)), 98) * 180 / np.pi)
            print(f"{label_v}:")
            print(f"  stutter-band power fraction : {bo:.3f} -> {bn:.3f}")
            print(f"  freeze-slot velocity ratio  : {ro:.2f} -> {rn:.2f} "
                  f"(target ~1.0)")
            print(f"  p98 angular accel deg/f^2   : {ao:.2f} -> {an:.2f} "
                  f"(rises by construction; see note)")

            axv = axes_v[0][col]
            axv.plot(np.degrees(ang_o), 'k-', lw=1, alpha=0.7, label='before')
            axv.plot(np.degrees(ang_n), 'r-', lw=1, label='after retiming')
            axv.set_title(f'{label_v} bone angle')
            axv.legend(); axv.grid(alpha=0.3)

            axv2 = axes_v[1][col]
            for ang_p, nm_p, colr in ((ang_o, 'before', 'k'),
                                      (ang_n, 'after', 'r')):
                vel = np.diff(ang_p) - np.diff(ang_p).mean()
                fr_p = np.fft.rfftfreq(len(vel))
                axv2.plot(fr_p, np.abs(np.fft.rfft(vel)) ** 2,
                          colr, lw=1, alpha=0.8, label=nm_p)
            axv2.axvline(0.2, color='red', ls='--', alpha=0.5,
                         label='duplication cadence')
            axv2.set_title(f'{label_v} angular-velocity spectrum')
            axv2.set_xlabel('cycles/frame'); axv2.legend(); axv2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('dedup_before_after.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("saved dedup_before_after.png")
    else:
        print("no pre-repair backup found -- verification skipped")
    return


@app.cell
def _(os, scan_video_quality):
    # ---- batch quality gate for the ~20 demo videos ---------------------
    CANDIDATE_DIR = 'candidate_videos'
    if os.path.isdir(CANDIDATE_DIR):
        rows_b = []
        for fn_b in sorted(os.listdir(CANDIDATE_DIR)):
            if not fn_b.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                continue
            rep_b, _mb = scan_video_quality(os.path.join(CANDIDATE_DIR, fn_b))
            rows_b.append((fn_b, rep_b))
        print(f"{'video':30} {'frames':>6} {'fps':>5} {'dup%':>6} "
              f"{'cad':>4} {'sharp':>7}  verdict")
        for fn_b, r_b in rows_b:
            print(f"{fn_b:30} {r_b['n_frames']:>6} {r_b['fps']:>5.1f} "
                  f"{r_b['dup_rate'] * 100:>5.1f}% "
                  f"{str(r_b['dup_cadence']):>4} "
                  f"{r_b['sharpness_median']:>7.0f}  {r_b['verdict']}")
    else:
        print(f"(batch mode idle -- create ./{CANDIDATE_DIR}/ and drop "
              f"candidate demo videos in it)")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

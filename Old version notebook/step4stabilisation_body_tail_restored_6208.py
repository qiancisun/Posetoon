import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    from mmpose.apis import MMPoseInferencer
    import cv2
    import numpy as np
    import json
    import matplotlib.pyplot as plt

    inferencer_s4 = MMPoseInferencer('animal')
    print("Model loaded")
    return cv2, inferencer_s4, json, np, plt


@app.cell
def _(cv2, inferencer_s4, json):
    def extract_full_video_keypoints(vid_path, infer):
        cap_s4 = cv2.VideoCapture(vid_path)
        all_frame_kps = []
        frame_idx_s4 = 0
        total = int(cap_s4.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing {total} frames...")

        while True:
            ret_s4, frame_s4 = cap_s4.read()
            if not ret_s4:
                break

            rgb_s4 = cv2.cvtColor(frame_s4, cv2.COLOR_BGR2RGB)
            result_s4 = [r for r in infer(rgb_s4, show=False, return_vis=False)]

            if len(result_s4[0]['predictions'][0]) > 0:
                kps_s4 = result_s4[0]['predictions'][0][0]['keypoints']
                scores_s4 = result_s4[0]['predictions'][0][0]['keypoint_scores']
            else:
                kps_s4 = [[0.0, 0.0]] * 17
                scores_s4 = [0.0] * 17

            all_frame_kps.append({
                'frame': frame_idx_s4,
                'keypoints': kps_s4,
                'scores': scores_s4
            })
            frame_idx_s4 += 1

            if frame_idx_s4 % 30 == 0:
                print(f"  Processed {frame_idx_s4}/{total} frames")

        cap_s4.release()
        print(f"Done! Total frames: {len(all_frame_kps)}")
        return all_frame_kps

    full_video_kps = extract_full_video_keypoints('dogvideo.mp4', inferencer_s4)

    # save
    with open('full_video_keypoints.json', 'w') as f_s4:
        json.dump(full_video_kps, f_s4)
    print("Saved to full_video_keypoints.json")
    return (full_video_kps,)


@app.cell
def _(full_video_kps, np, plt):
    from scipy.signal import savgol_filter as sgf

    def moving_average_fn(data, window=5):
        result = np.array(data, dtype=float)
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            result[i] = np.mean(data[start:end])
        return result

    def savitzky_golay_fn(data, window=7, poly=2):
        data_arr = np.array(data, dtype=float)
        if len(data_arr) < window:
            return data_arr
        return sgf(data_arr, window, poly)

    def kalman_filter_fn(data, process_noise=1e-3, measurement_noise=0.1):
        data_arr = np.array(data, dtype=float)
        n = len(data_arr)
        result = np.zeros(n)
        x = data_arr[0]
        p = 1.0
        for i in range(n):
            p = p + process_noise
            k = p / (p + measurement_noise)
            x = x + k * (data_arr[i] - x)
            p = (1 - k) * p
            result[i] = x
        return result

    def one_euro_filter_fn(data, min_cutoff=1.0, beta=0.007):
        data_arr = np.array(data, dtype=float)
        result = np.zeros_like(data_arr)
        result[0] = data_arr[0]
        x_prev = data_arr[0]
        dx_prev = 0.0
        for i in range(1, len(data_arr)):
            dx = data_arr[i] - x_prev
            dx_smooth = dx_prev + 0.1 * (dx - dx_prev)
            cutoff = min_cutoff + beta * abs(dx_smooth)
            alpha = cutoff / (cutoff + 1.0)
            result[i] = x_prev + alpha * (data_arr[i] - x_prev)
            x_prev = result[i]
            dx_prev = dx_smooth
        return result

    def compare_stabilisation_methods_fn(kps_list, joint_idx=3, joint_name='Neck'):
        frames_c4 = [k['frame'] for k in kps_list]
        raw_x_c4 = np.array([k['keypoints'][joint_idx][0] for k in kps_list])
        raw_y_c4 = np.array([k['keypoints'][joint_idx][1] for k in kps_list])

        methods_c4 = {
            'Raw': (raw_x_c4, raw_y_c4),
            'Moving Average': (moving_average_fn(raw_x_c4), moving_average_fn(raw_y_c4)),
            'Savitzky-Golay': (savitzky_golay_fn(raw_x_c4), savitzky_golay_fn(raw_y_c4)),
            'Kalman': (kalman_filter_fn(raw_x_c4), kalman_filter_fn(raw_y_c4)),
            'One Euro': (one_euro_filter_fn(raw_x_c4), one_euro_filter_fn(raw_y_c4)),
        }

        colors_c4 = {'Raw': 'black', 'Moving Average': 'blue',
                     'Savitzky-Golay': 'green', 'Kalman': 'orange', 'One Euro': 'red'}

        fig_c4, axes_c4 = plt.subplots(2, 1, figsize=(14, 8))

        for name_c4, (x_c4, y_c4) in methods_c4.items():
            lw_c4 = 2.5 if name_c4 == 'Raw' else 1.5
            axes_c4[0].plot(frames_c4, x_c4, label=name_c4,
                            color=colors_c4[name_c4], linewidth=lw_c4, alpha=0.8)
            axes_c4[1].plot(frames_c4, y_c4, label=name_c4,
                            color=colors_c4[name_c4], linewidth=lw_c4, alpha=0.8)

        axes_c4[0].set_title(f'{joint_name} X position - all methods comparison')
        axes_c4[0].set_xlabel('Frame')
        axes_c4[0].set_ylabel('X (pixels)')
        axes_c4[0].legend()
        axes_c4[0].grid(True, alpha=0.3)

        axes_c4[1].set_title(f'{joint_name} Y position - all methods comparison')
        axes_c4[1].set_xlabel('Frame')
        axes_c4[1].set_ylabel('Y (pixels)')
        axes_c4[1].legend()
        axes_c4[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('stabilisation_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nJitter comparison (std dev of frame-to-frame difference):")
        print(f"{'Method':<20} {'X jitter':>10} {'Y jitter':>10}")
        print("-" * 42)
        for name_c4, (x_c4, y_c4) in methods_c4.items():
            x_jitter_c4 = np.std(np.diff(x_c4))
            y_jitter_c4 = np.std(np.diff(y_c4))
            print(f"{name_c4:<20} {x_jitter_c4:>10.3f} {y_jitter_c4:>10.3f}")

        return methods_c4

    stabilised_methods = compare_stabilisation_methods_fn(full_video_kps, joint_idx=3, joint_name='Neck')
    return (one_euro_filter_fn,)


@app.cell
def _(full_video_kps, json, np, one_euro_filter_fn):
    def apply_stabilisation_all_joints(kps_list, filter_fn, score_thr=0.3):
        num_joints = 17
        all_x = [[k['keypoints'][j][0] for k in kps_list] for j in range(num_joints)]
        all_y = [[k['keypoints'][j][1] for k in kps_list] for j in range(num_joints)]

        # Mask missing / unreliable detections BEFORE smoothing. Detector
        # failures are stored as (0,0) with score 0; feeding those zeros into
        # the filter drags every trajectory toward the image corner and then
        # snaps back (visible as limbs "flying away"). Interpolate across the
        # gaps first, then apply the filter.
        def fill_gaps(vals, bad):
            vals = np.asarray(vals, dtype=float)
            bad = np.asarray(bad, dtype=bool)
            good = ~bad
            if good.sum() < 2:
                return vals
            idx = np.arange(len(vals))
            vals[bad] = np.interp(idx[bad], idx[good], vals[good])
            return vals

        smoothed_x, smoothed_y = [], []
        stabilised_kps = []
        nfilled = 0
        for j in range(num_joints):
            bad = [(k['scores'][j] < score_thr or
                    (abs(k['keypoints'][j][0]) < 1e-6 and abs(k['keypoints'][j][1]) < 1e-6))
                   for k in kps_list]
            nfilled += int(sum(bad))
            smoothed_x.append(filter_fn(fill_gaps(all_x[j], bad)))
            smoothed_y.append(filter_fn(fill_gaps(all_y[j], bad)))
        print(f"interpolated {nfilled} missing/low-confidence joint positions before smoothing")

        for i, frame_data in enumerate(kps_list):
            new_kps = [[smoothed_x[j][i], smoothed_y[j][i]] for j in range(num_joints)]
            stabilised_kps.append({
                'frame': frame_data['frame'],
                'keypoints': new_kps,
                'scores': frame_data['scores']
            })

        with open('stabilised_keypoints.json', 'w') as f_stab:
            json.dump(stabilised_kps, f_stab)

        print(f"Stabilised keypoints saved: {len(stabilised_kps)} frames, {num_joints} joints")
        return stabilised_kps

    stabilised_kps = apply_stabilisation_all_joints(full_video_kps, one_euro_filter_fn)
    return (stabilised_kps,)


@app.cell
def _(full_video_kps, np, plt, stabilised_kps):
    def plot_before_after(raw_kps, stab_kps, joint_idx=3, joint_name='Neck'):
        frames_ba = [k['frame'] for k in raw_kps]
        raw_x_ba = np.array([k['keypoints'][joint_idx][0] for k in raw_kps])
        stab_x_ba = np.array([k['keypoints'][joint_idx][0] for k in stab_kps])

        raw_jitter = np.std(np.diff(raw_x_ba))
        stab_jitter = np.std(np.diff(stab_x_ba))
        improvement = (1 - stab_jitter/raw_jitter) * 100

        fig_ba, ax_ba = plt.subplots(figsize=(14, 5))
        ax_ba.plot(frames_ba, raw_x_ba, 'k-', linewidth=1.5, alpha=0.7, label=f'Raw (jitter={raw_jitter:.3f})')
        ax_ba.plot(frames_ba, stab_x_ba, 'r-', linewidth=1.5, alpha=0.9, label=f'One Euro Filter (jitter={stab_jitter:.3f})')
        ax_ba.set_title(f'{joint_name} X position: Before vs After Stabilisation\nJitter reduced by {improvement:.1f}%')
        ax_ba.set_xlabel('Frame')
        ax_ba.set_ylabel('X (pixels)')
        ax_ba.legend()
        ax_ba.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('before_after_stabilisation.png', dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Jitter reduced by {improvement:.1f}%")

    plot_before_after(full_video_kps, stabilised_kps)
    return


@app.cell
def _(full_video_kps, np, stabilised_kps):
    def check_bone_length_consistency(kps_list, label='Raw'):
        bone_pairs = [(3,5), (3,8), (5,6), (6,7), (8,9), (9,10),
                      (3,4), (11,12), (12,13), (14,15), (15,16)]

        variances = []
        for (p1, p2) in bone_pairs:
            lengths = []
            for frame in kps_list:
                kp1 = np.array(frame['keypoints'][p1])
                kp2 = np.array(frame['keypoints'][p2])
                length = np.linalg.norm(kp2 - kp1)
                lengths.append(length)
            variances.append(np.std(lengths))

        avg_variance = np.mean(variances)
        print(f"{label}: Average bone length std = {avg_variance:.3f}")
        return avg_variance

    raw_var = check_bone_length_consistency(full_video_kps, 'Raw')
    stab_var = check_bone_length_consistency(stabilised_kps, 'One Euro Filter')
    print(f"\nBone length consistency improved by {(1 - stab_var/raw_var)*100:.1f}%")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

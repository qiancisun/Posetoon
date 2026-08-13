import marimo


__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import json
    import cv2
    from mmpose.apis import MMPoseInferencer

    VIDEO = os.environ.get('POSETOON_VIDEO', 'dogvideo.mp4')
    OUT = 'full_video_keypoints.json'

    if not os.path.exists(VIDEO):
        raise FileNotFoundError(f"{VIDEO} not found")

    print(f"extracting from {VIDEO}")
    inferencer = MMPoseInferencer('animal')
    print("model loaded")
    return MMPoseInferencer, OUT, VIDEO, cv2, inferencer, json, os


@app.cell
def _(OUT, VIDEO, cv2, inferencer, json):
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

    full_video_kps = extract_full_video_keypoints(VIDEO, inferencer)

    with open(OUT, 'w') as f_s4:
        json.dump(full_video_kps, f_s4)

    n_missing = sum(1 for fr in full_video_kps if max(fr['scores']) == 0.0)
    print(f"Saved {OUT}: {len(full_video_kps)} frames, "
          f"{n_missing} with no detection")
    if n_missing > 0.1 * len(full_video_kps):
        print(f"  WARNING: no dog found in {100*n_missing/len(full_video_kps):.0f}% "
              f"of frames. Stage 2 will interpolate, but this clip may not be "
              f"usable -- check that the dog stays in shot.")
    return (full_video_kps,)


if __name__ == "__main__":
    app.run()

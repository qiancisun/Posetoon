import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch
    import json

    with open('bone_ratios.json', 'r') as fin:
        bone_ratios = json.load(fin)

    SPINE_LENGTH = 200

    def build_skeleton_positions(spine_length=200):
        s = spine_length
        r = bone_ratios

        positions = {}

        positions['Neck']         = np.array([0.0, 0.0])
        positions['root_of_tail'] = np.array([s, 0.0])

        head_len = r['Nose_to_Neck'] * s
        positions['Nose']  = np.array([-head_len, 0.0])
        positions['L_Eye'] = np.array([-head_len * 0.8, -head_len * 0.4])
        positions['R_Eye'] = np.array([-head_len * 0.8, head_len * 0.4])

        shoulder_x = s * 0.15
        front_upper = r['L_Shoulder_to_L_Elbow'] * s
        front_lower = r['L_Elbow_to_L_F_Paw'] * s
        positions['L_Shoulder'] = np.array([shoulder_x, 0.0])
        positions['L_Elbow']    = np.array([shoulder_x, front_upper])
        positions['L_F_Paw']    = np.array([shoulder_x, front_upper + front_lower])

        hip_x = s * 0.85
        back_upper = r['L_Hip_to_L_Knee'] * s
        back_lower = r['L_Knee_to_L_B_Paw'] * s
        positions['L_Hip']   = np.array([hip_x, 0.0])
        positions['L_Knee']  = np.array([hip_x, back_upper])
        positions['L_B_Paw'] = np.array([hip_x, back_upper + back_lower])

        positions['Tail'] = np.array([s * 1.15, -s * 0.2])

        return positions

    positions = build_skeleton_positions(SPINE_LENGTH)
    print("Skeleton positions built!")
    for name, pos in positions.items():
        print(f"  {name:15}: ({pos[0]:.1f}, {pos[1]:.1f})")
    return np, plt, positions


@app.cell
def _(plt, positions):
    def draw_character_skeleton(positions):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_facecolor('#F5F5F5')

        connections = [
            ('L_Eye', 'Nose'),
            ('R_Eye', 'Nose'),
            ('Nose', 'Neck'),
            ('Neck', 'root_of_tail'),
            ('root_of_tail', 'Tail'),
            ('Neck', 'L_Shoulder'),
            ('L_Shoulder', 'L_Elbow'),
            ('L_Elbow', 'L_F_Paw'),
            ('Neck', 'L_Hip'),
            ('L_Hip', 'L_Knee'),
            ('L_Knee', 'L_B_Paw'),
        ]

        for (p1, p2) in connections:
            x = [positions[p1][0], positions[p2][0]]
            y = [positions[p1][1], positions[p2][1]]
            ax.plot(x, y, 'b-', linewidth=3, alpha=0.8)

        colors = {
            'Nose': 'red', 'L_Eye': 'orange', 'R_Eye': 'orange',
            'Neck': 'green', 'root_of_tail': 'green',
            'L_Shoulder': 'blue', 'L_Elbow': 'blue', 'L_F_Paw': 'cyan',
            'L_Hip': 'purple', 'L_Knee': 'purple', 'L_B_Paw': 'pink',
            'Tail': 'brown'
        }

        for name, pos in positions.items():
            color = colors.get(name, 'gray')
            ax.plot(pos[0], pos[1], 'o', color=color, markersize=12, zorder=5)
            ax.text(pos[0]+5, pos[1]-8, name, fontsize=8, color='black')

        ax.set_xlim(-150, 300)
        ax.set_ylim(-100, 150)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_title('2D Dog Character Skeleton (Side View)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('character_skeleton.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved to character_skeleton.png")

    draw_character_skeleton(positions)
    return


@app.cell
def _():
    RIGGING = {
        'body': {
            'joint': 'Neck',
            'end_joint': 'root_of_tail',
            'anchor': (0.0, 0.5),
            'image': 'assets/body.png'
        },
        'head': {
            'joint': 'Nose',
            'end_joint': 'Neck',
            'anchor': (1.0, 0.5),
            'image': 'assets/head.png'
        },
        'front_leg_upper': {
            'joint': 'L_Shoulder',
            'end_joint': 'L_Elbow',
            'anchor': (0.5, 0.0),
            'image': 'assets/front_leg_upper.png'
        },
        'front_leg_lower': {
            'joint': 'L_Elbow',
            'end_joint': 'L_F_Paw',
            'anchor': (0.5, 0.0),
            'image': 'assets/front_leg_lower.png'
        },
        'back_leg_upper': {
            'joint': 'L_Hip',
            'end_joint': 'L_Knee',
            'anchor': (0.5, 0.0),
            'image': 'assets/back_leg_upper.png'
        },
        'back_leg_lower': {
            'joint': 'L_Knee',
            'end_joint': 'L_B_Paw',
            'anchor': (0.5, 0.0),
            'image': 'assets/back_leg_lower.png'
        },
        'tail': {
            'joint': 'root_of_tail',
            'end_joint': 'Tail',
            'anchor': (0.0, 0.5),
            'image': 'assets/tail.png'
        }
    }

    print("Rigging structure defined!")
    print(f"Total parts: {len(RIGGING)}")
    for part_name, part_info in RIGGING.items():
        print(f"  {part_name:20}: joint={part_info['joint']:15} → {part_info['end_joint']}")
    return


@app.cell
def _():
    import os
    from PIL import Image, ImageDraw

    os.makedirs('assets', exist_ok=True)

    def create_placeholder_parts():
        parts = {
            'body':             (180, 80, '#8B6914', 'oval torso'),
            'head':             (90,  80, '#D4A017', 'round head'),
            'front_leg_upper':  (30,  80, '#8B6914', 'front leg upper'),
            'front_leg_lower':  (25,  70, '#7A5C10', 'front leg lower'),
            'back_leg_upper':   (35,  85, '#8B6914', 'back leg upper'),
            'back_leg_lower':   (28,  75, '#7A5C10', 'back leg lower'),
            'tail':             (80,  25, '#8B6914', 'tail'),
        }

        for part_name, (w, h, color, desc) in parts.items():
            img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            if part_name in ('head', 'body', 'tail'):
                draw.ellipse([2, 2, w-2, h-2], fill=color, outline='#5C3D00', width=2)
            else:
                draw.rounded_rectangle([2, 2, w-2, h-2], radius=8,
                                       fill=color, outline='#5C3D00', width=2)

            img.save(f'assets/{part_name}.png')
            print(f"Created assets/{part_name}.png ({w}x{h}) - {desc}")

    create_placeholder_parts()
    print("\nAll placeholder parts created!")
    return (Image,)


@app.cell
def _(Image, np, plt, positions):
    RIGGING_V2 = {
        'body': {
            'joint': 'Neck',
            'end_joint': 'root_of_tail',
            'anchor': (0.0, 0.5),
            'image': 'assets/body.png'
        },
        'head': {
            'joint': 'Neck',
            'end_joint': 'Nose',
            'anchor': (1.0, 0.5),
            'image': 'assets/head.png'
        },
        'front_leg_upper': {
            'joint': 'L_Shoulder',
            'end_joint': 'L_Elbow',
            'anchor': (0.5, 0.0),
            'image': 'assets/front_leg_upper.png'
        },
        'front_leg_lower': {
            'joint': 'L_Elbow',
            'end_joint': 'L_F_Paw',
            'anchor': (0.5, 0.0),
            'image': 'assets/front_leg_lower.png'
        },
        'back_leg_upper': {
            'joint': 'L_Hip',
            'end_joint': 'L_Knee',
            'anchor': (0.5, 0.0),
            'image': 'assets/back_leg_upper.png'
        },
        'back_leg_lower': {
            'joint': 'L_Knee',
            'end_joint': 'L_B_Paw',
            'anchor': (0.5, 0.0),
            'image': 'assets/back_leg_lower.png'
        },
        'tail': {
            'joint': 'root_of_tail',
            'end_joint': 'Tail',
            'anchor': (0.0, 0.5),
            'image': 'assets/tail.png'
        }
    }

    def render_character_v4(positions_dict, rigging_dict, canvas_size=(600, 400)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))

        render_order = ['tail', 'back_leg_upper', 'back_leg_lower',
                        'body', 'front_leg_upper', 'front_leg_lower', 'head']

        all_x = [p[0] for p in positions_dict.values()]
        all_y = [p[1] for p in positions_dict.values()]
        offset_x = int(canvas_size[0] // 2 - (max(all_x) + min(all_x)) / 2)
        offset_y = int(canvas_size[1] // 2 - (max(all_y) + min(all_y)) / 2)

        for part_name in render_order:
            if part_name not in rigging_dict:
                continue

            part = rigging_dict[part_name]
            joint_pos = positions_dict[part['joint']]
            end_pos = positions_dict[part['end_joint']]

            dx = end_pos[0] - joint_pos[0]
            dy = end_pos[1] - joint_pos[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            part_img = Image.open(part['image']).convert('RGBA')
            rotated = part_img.rotate(angle, expand=True)

            paste_x = int(joint_pos[0] + offset_x - rotated.width * part['anchor'][0])
            paste_y = int(joint_pos[1] + offset_y - rotated.height * part['anchor'][1])

            canvas.paste(rotated, (paste_x, paste_y), rotated)

        return canvas

    character_v4 = render_character_v4(positions, RIGGING_V2)
    plt.figure(figsize=(8, 6))
    plt.imshow(character_v4)
    plt.axis('off')
    plt.title('2D Dog Character - v4')
    plt.savefig('character_render_v4.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved!")
    return


@app.cell
def _():
    import cv2
    from mmpose.apis import MMPoseInferencer

    video_path = 'dogvideo.mp4'
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    print(f"Video info:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    print(f"  Resolution: {width}x{height}")
    return MMPoseInferencer, cv2


@app.cell
def _(MMPoseInferencer):
    inferencer_vid = MMPoseInferencer('animal')
    print("Model loaded!")
    return (inferencer_vid,)


@app.cell
def _(cv2, inferencer_vid):
    def extract_video_keypoints_v2(vid_path, infer, sample_every=3):
        cap_v2 = cv2.VideoCapture(vid_path)
        frame_kps_list = []
        frame_idx_v2 = 0

        while True:
            ret_v2, frame_v2 = cap_v2.read()
            if not ret_v2:
                break

            if frame_idx_v2 % sample_every == 0:
                rgb_frame_v2 = cv2.cvtColor(frame_v2, cv2.COLOR_BGR2RGB)
                result_gen_v2 = infer(rgb_frame_v2, show=False, return_vis=False)
                result_v2 = [r for r in result_gen_v2]

                if len(result_v2[0]['predictions'][0]) > 0:
                    kps_v2 = result_v2[0]['predictions'][0][0]['keypoints']
                    scores_v2 = result_v2[0]['predictions'][0][0]['keypoint_scores']
                    frame_kps_list.append({
                        'frame': frame_idx_v2,
                        'keypoints': kps_v2,
                        'scores': scores_v2
                    })

            frame_idx_v2 += 1

        cap_v2.release()
        return frame_kps_list

    video_kps = extract_video_keypoints_v2('dogvideo.mp4', inferencer_vid, sample_every=3)
    print(f"Extracted keypoints from {len(video_kps)} frames")
    print(f"Sample frame 0 keypoints: {len(video_kps[0]['keypoints'])} joints")
    return (video_kps,)


@app.cell
def _(plt, video_kps):
    def plot_video_keypoints_trajectory(kps_list, joint_idx=3, joint_name='Neck'):
        frames = [k['frame'] for k in kps_list]
        x_vals = [k['keypoints'][joint_idx][0] for k in kps_list]
        y_vals = [k['keypoints'][joint_idx][1] for k in kps_list]

        fig_traj, axes_traj = plt.subplots(2, 1, figsize=(12, 6))

        axes_traj[0].plot(frames, x_vals, 'b-', linewidth=1.5)
        axes_traj[0].set_title(f'{joint_name} X position over frames (raw)')
        axes_traj[0].set_xlabel('Frame')
        axes_traj[0].set_ylabel('X (pixels)')
        axes_traj[0].grid(True, alpha=0.3)

        axes_traj[1].plot(frames, y_vals, 'r-', linewidth=1.5)
        axes_traj[1].set_title(f'{joint_name} Y position over frames (raw)')
        axes_traj[1].set_xlabel('Frame')
        axes_traj[1].set_ylabel('Y (pixels)')
        axes_traj[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('keypoint_trajectory_raw.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved to keypoint_trajectory_raw.png")

    plot_video_keypoints_trajectory(video_kps, joint_idx=3, joint_name='Neck')
    return


@app.cell
def _(cv2, plt, video_kps):
    def visualize_video_skeleton(vid_path, kps_list, output_path='video_skeleton_preview.png'):
        cap_vis = cv2.VideoCapture(vid_path)

        sample_frames_idx = [0, len(kps_list)//2, len(kps_list)-1]

        fig_vid, axes_vid = plt.subplots(1, 3, figsize=(15, 5))

        skeleton_connections = [
            (0,2),(1,2),(2,3),(3,5),(3,8),(5,6),(6,7),
            (8,9),(9,10),(3,4),(4,11),(4,14),
            (11,12),(12,13),(14,15),(15,16)
        ]

        for plot_idx, frame_data_idx in enumerate(sample_frames_idx):
            frame_data = kps_list[frame_data_idx]
            frame_num = frame_data['frame']

            cap_vis.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret_vis, frame_vis = cap_vis.read()

            if ret_vis:
                rgb_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
                axes_vid[plot_idx].imshow(rgb_vis)

                kps_vis2 = frame_data['keypoints']
                sc_vis2 = frame_data['scores']

                for (p1, p2) in skeleton_connections:
                    if sc_vis2[p1] > 0.3 and sc_vis2[p2] > 0.3:
                        axes_vid[plot_idx].plot(
                            [kps_vis2[p1][0], kps_vis2[p2][0]],
                            [kps_vis2[p1][1], kps_vis2[p2][1]],
                            'b-', linewidth=2
                        )

                for i, (kp_v, sc_v) in enumerate(zip(kps_vis2, sc_vis2)):
                    if sc_v > 0.3:
                        axes_vid[plot_idx].plot(kp_v[0], kp_v[1], 'ro', markersize=5)

                axes_vid[plot_idx].set_title(f'Frame {frame_num}')
                axes_vid[plot_idx].axis('off')

        cap_vis.release()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {output_path}")

    visualize_video_skeleton('dogvideo.mp4', video_kps)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

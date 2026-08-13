import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import numpy as np
    import json
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import os

    with open('full_video_keypoints.json', 'r') as f_r:
        raw_kps = json.load(f_r)

    with open('stabilised_keypoints.json', 'r') as f_s:
        stab_kps = json.load(f_s)

    with open('bone_ratios.json', 'r') as f_b:
        bone_ratios = json.load(f_b)

    print(f"Raw keypoints: {len(raw_kps)} frames")
    print(f"Stabilised keypoints: {len(stab_kps)} frames")
    print(f"Bone ratios loaded: {len(bone_ratios)} bones")
    return Image, ImageDraw, cv2, np, os, plt, raw_kps, stab_kps


@app.cell
def _(np):
    KEYPOINT_NAMES = [
        'L_Eye', 'R_Eye', 'Nose', 'Neck', 'root_of_tail',
        'L_Shoulder', 'L_Elbow', 'L_F_Paw',
        'R_Shoulder', 'R_Elbow', 'R_F_Paw',
        'L_Hip', 'L_Knee', 'L_B_Paw',
        'R_Hip', 'R_Knee', 'R_B_Paw'
    ]

    SKELETON = [
        ('Neck', 'root_of_tail'),
        ('Neck', 'Nose'),
        ('Nose', 'L_Eye'),
        ('Nose', 'R_Eye'),
        ('Neck', 'L_Shoulder'),
        ('L_Shoulder', 'L_Elbow'),
        ('L_Elbow', 'L_F_Paw'),
        ('Neck', 'R_Shoulder'),
        ('R_Shoulder', 'R_Elbow'),
        ('R_Elbow', 'R_F_Paw'),
        ('root_of_tail', 'L_Hip'),
        ('L_Hip', 'L_Knee'),
        ('L_Knee', 'L_B_Paw'),
        ('root_of_tail', 'R_Hip'),
        ('R_Hip', 'R_Knee'),
        ('R_Knee', 'R_B_Paw'),
    ]

    def get_kp(frame_data, name):
        idx = KEYPOINT_NAMES.index(name)
        return np.array(frame_data['keypoints'][idx])

    def get_score(frame_data, name):
        idx = KEYPOINT_NAMES.index(name)
        return frame_data['scores'][idx]

    def normalise_skeleton(frame_data, target_spine_length=200):
        neck = get_kp(frame_data, 'Neck')
        tail = get_kp(frame_data, 'root_of_tail')
        spine_len = np.linalg.norm(tail - neck)

        if spine_len < 1e-6:
            return None, None

        scale = target_spine_length / spine_len
        centre = neck

        normalised = {}
        for name in KEYPOINT_NAMES:
            kp = get_kp(frame_data, name)
            normalised[name] = (kp - centre) * scale

        return normalised, scale

    print("Skeleton and retargeting functions defined")
    print(f"Keypoints: {len(KEYPOINT_NAMES)}")
    print(f"Bones: {len(SKELETON)}")
    return KEYPOINT_NAMES, SKELETON, get_kp, normalise_skeleton


@app.cell
def _(Image, ImageDraw, SKELETON, normalise_skeleton, plt, stab_kps):
    def render_frame(normalised_kps, canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw = ImageDraw.Draw(canvas)

        ox, oy = offset

        for (p1, p2) in SKELETON:
            if p1 in normalised_kps and p2 in normalised_kps:
                x1 = int(normalised_kps[p1][0] + ox)
                y1 = int(normalised_kps[p1][1] + oy)
                x2 = int(normalised_kps[p2][0] + ox)
                y2 = int(normalised_kps[p2][1] + oy)
                draw.line([(x1, y1), (x2, y2)], fill=(50, 100, 200, 255), width=4)

        joint_colors = {
            'Neck': (0, 200, 0),
            'root_of_tail': (0, 200, 0),
            'Nose': (255, 0, 0),
            'L_Eye': (255, 165, 0),
            'R_Eye': (255, 165, 0),
            'L_Shoulder': (100, 100, 255),
            'R_Shoulder': (100, 100, 255),
            'L_Hip': (200, 0, 200),
            'R_Hip': (200, 0, 200),
        }

        for name, kp in normalised_kps.items():
            x = int(kp[0] + ox)
            y = int(kp[1] + oy)
            color = joint_colors.get(name, (150, 150, 150))
            r = 6
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

        return canvas

    test_norm, test_scale = normalise_skeleton(stab_kps[0])

    if test_norm:
        test_frame = render_frame(test_norm)
        plt.figure(figsize=(8, 6))
        plt.imshow(test_frame)
        plt.axis('off')
        plt.title('Retargeted skeleton - Frame 0')
        plt.savefig('retarget_test_frame0.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Frame 0 rendered successfully!")
    else:
        print("Failed to normalise skeleton")
    return (test_norm,)


@app.cell
def _(Image, np, os, plt, test_norm):
    def render_character_frame(normalised_kps, canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        ox, oy = offset

        parts = [
            ('tail',            'root_of_tail', 'L_Hip',      (0.0, 0.5)),
            ('back_leg_upper',  'L_Hip',        'L_Knee',     (0.5, 0.0)),
            ('back_leg_lower',  'L_Knee',       'L_B_Paw',    (0.5, 0.0)),
            ('body',            'Neck',         'root_of_tail',(0.0, 0.5)),
            ('front_leg_upper', 'L_Shoulder',   'L_Elbow',    (0.5, 0.0)),
            ('front_leg_lower', 'L_Elbow',      'L_F_Paw',    (0.5, 0.0)),
            ('head',            'Neck',         'Nose',        (1.0, 0.5)),
        ]

        for part_name, j1, j2, anchor in parts:
            if j1 not in normalised_kps or j2 not in normalised_kps:
                continue

            p1 = normalised_kps[j1]
            p2 = normalised_kps[j2]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = -np.degrees(np.arctan2(dy, dx))

            img_path = f'assets/{part_name}.png'
            if not os.path.exists(img_path):
                continue

            part_img = Image.open(img_path).convert('RGBA')
            rotated = part_img.rotate(angle, expand=True)

            paste_x = int(p1[0] + ox - rotated.width * anchor[0])
            paste_y = int(p1[1] + oy - rotated.height * anchor[1])

            canvas.paste(rotated, (paste_x, paste_y), rotated)

        return canvas

    char_frame0 = render_character_frame(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_frame0)
    plt.axis('off')
    plt.title('2D Character - Frame 0')
    plt.savefig('character_frame0.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Character rendered")
    return


@app.cell
def _():
    return


@app.cell
def _(Image, ImageDraw, plt, test_norm):
    def render_character_v2(normalised_kps, canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw = ImageDraw.Draw(canvas)
        ox, oy = offset

        def pt(name):
            kp = normalised_kps[name]
            return (int(kp[0] + ox), int(kp[1] + oy))

        draw.line([pt('Neck'), pt('root_of_tail')], fill='#8B6914', width=28)

        draw.line([pt('Neck'), pt('Nose')], fill='#D4A017', width=22)
        nx, ny = pt('Nose')
        draw.ellipse([nx-20, ny-20, nx+20, ny+20], fill='#D4A017')

        draw.ellipse([pt('L_Eye')[0]-6, pt('L_Eye')[1]-6,
                      pt('L_Eye')[0]+6, pt('L_Eye')[1]+6], fill='black')
        draw.ellipse([pt('R_Eye')[0]-6, pt('R_Eye')[1]-6,
                      pt('R_Eye')[0]+6, pt('R_Eye')[1]+6], fill='black')

        draw.line([pt('L_Shoulder'), pt('L_Elbow')], fill='#7A5C10', width=14)
        draw.line([pt('L_Elbow'), pt('L_F_Paw')], fill='#6B4F0E', width=11)

        draw.line([pt('L_Hip'), pt('L_Knee')], fill='#7A5C10', width=14)
        draw.line([pt('L_Knee'), pt('L_B_Paw')], fill='#6B4F0E', width=11)

        draw.line([pt('root_of_tail'), pt('R_Hip')], fill='#8B6914', width=10)

        return canvas

    char_v2 = render_character_v2(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v2)
    plt.axis('off')
    plt.title('2D Character - Direct Draw')
    plt.savefig('character_v2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return (render_character_v2,)


@app.cell
def _(cv2, normalise_skeleton, np, render_character_v2, stab_kps):
    def generate_animation_video(stab_kps_list, output_path='animation.mp4',
                                  canvas_size=(500, 400), fps=30):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, canvas_size)

        success_frames = 0

        for i, frame_data in enumerate(stab_kps_list):
            norm_kps, scale = normalise_skeleton(frame_data)

            if norm_kps is None:
                continue

            char_img = render_character_v2(norm_kps, canvas_size=canvas_size)
            frame_bgr = cv2.cvtColor(np.array(char_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            success_frames += 1

            if i % 50 == 0:
                print(f"Rendered {i}/{len(stab_kps_list)} frames")

        out.release()
        print(f"Video saved to {output_path}")
        print(f"Total frames rendered: {success_frames}")

    generate_animation_video(stab_kps, output_path='animation.mp4', fps=30)
    return


@app.cell
def _(cv2, normalise_skeleton, np, render_character_v2, stab_kps):
    def generate_side_by_side(original_video_path, stab_kps_list,
                               output_path='side_by_side.mp4', fps=30):
        cap_sb = cv2.VideoCapture(original_video_path)
        orig_w = int(cap_sb.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_sb.get(cv2.CAP_PROP_FRAME_HEIGHT))

        anim_w, anim_h = 500, 400

        scale_orig = anim_h / orig_h
        new_orig_w = int(orig_w * scale_orig)

        total_w = new_orig_w + anim_w
        total_h = anim_h

        fourcc_sb = cv2.VideoWriter_fourcc(*'mp4v')
        out_sb = cv2.VideoWriter(output_path, fourcc_sb, fps, (total_w, total_h))

        for i, frame_data in enumerate(stab_kps_list):
            ret_sb, orig_frame = cap_sb.read()
            if not ret_sb:
                break

            orig_resized = cv2.resize(orig_frame, (new_orig_w, anim_h))

            norm_kps_sb, _ = normalise_skeleton(frame_data)
            if norm_kps_sb is None:
                anim_frame = np.ones((anim_h, anim_w, 3), dtype=np.uint8) * 240
            else:
                char_img_sb = render_character_v2(norm_kps_sb, canvas_size=(anim_w, anim_h))
                anim_frame = cv2.cvtColor(np.array(char_img_sb.convert('RGB')), cv2.COLOR_RGB2BGR)

            cv2.putText(orig_resized, 'Original', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(anim_frame, '2D Animation', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,50), 2)

            combined = np.hstack([orig_resized, anim_frame])
            out_sb.write(combined)

            if i % 50 == 0:
                print(f"Processing {i}/{len(stab_kps_list)} frames")

        cap_sb.release()
        out_sb.release()
        print(f"Side-by-side video saved to {output_path}")

    generate_side_by_side('dogvideo.mp4', stab_kps, output_path='side_by_side.mp4')
    return


@app.cell
def _(cv2, normalise_skeleton, plt, render_character_v2, stab_kps):
    def save_preview_frames(stab_kps_list, original_video, output_path='preview_frames.png'):
        cap_pv = cv2.VideoCapture(original_video)
        sample_indices = [0, len(stab_kps_list)//3, len(stab_kps_list)*2//3, len(stab_kps_list)-1]

        fig_pv, axes_pv = plt.subplots(2, 4, figsize=(20, 8))

        for col, idx in enumerate(sample_indices):
            frame_data_pv = stab_kps_list[idx]
            cap_pv.set(cv2.CAP_PROP_POS_FRAMES, frame_data_pv['frame'])
            ret_pv, orig_pv = cap_pv.read()

            if ret_pv:
                axes_pv[0][col].imshow(cv2.cvtColor(orig_pv, cv2.COLOR_BGR2RGB))
                axes_pv[0][col].set_title(f'Original Frame {frame_data_pv["frame"]}')
                axes_pv[0][col].axis('off')

            norm_pv, _ = normalise_skeleton(frame_data_pv)
            if norm_pv:
                char_pv = render_character_v2(norm_pv)
                axes_pv[1][col].imshow(char_pv)
                axes_pv[1][col].set_title(f'2D Animation Frame {frame_data_pv["frame"]}')
                axes_pv[1][col].axis('off')

        cap_pv.release()
        plt.suptitle('Original vs 2D Animation - Frame Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {output_path}")

    save_preview_frames(stab_kps, 'dogvideo.mp4')
    return


@app.cell
def _(cv2, normalise_skeleton, np, raw_kps, render_character_v2, stab_kps):
    def generate_three_panel(original_video_path, raw_kps_list, stab_kps_list,
                              output_path='three_panel.mp4', fps=30):
        SKELETON_IDX = [
            (0,2),(1,2),(2,3),(3,5),(3,8),(5,6),(6,7),
            (8,9),(9,10),(3,4),(4,11),(4,14),
            (11,12),(12,13),(14,15),(15,16)
        ]

        cap_tp = cv2.VideoCapture(original_video_path)
        orig_w = int(cap_tp.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap_tp.get(cv2.CAP_PROP_FRAME_HEIGHT))

        panel_h = 400
        scale_tp = panel_h / orig_h
        panel_w_orig = int(orig_w * scale_tp)
        panel_w_anim = 500

        total_w = panel_w_orig * 2 + panel_w_anim
        total_h = panel_h

        fourcc_tp = cv2.VideoWriter_fourcc(*'mp4v')
        out_tp = cv2.VideoWriter(output_path, fourcc_tp, fps, (total_w, total_h))

        for i, stab_frame in enumerate(stab_kps_list):
            ret_tp, orig_frame = cap_tp.read()
            if not ret_tp:
                break

            panel1 = cv2.resize(orig_frame, (panel_w_orig, panel_h))

            panel2 = panel1.copy()
            raw_frame = raw_kps_list[i]
            kps_raw = raw_frame['keypoints']
            sc_raw = raw_frame['scores']
            for (p1, p2) in SKELETON_IDX:
                if sc_raw[p1] > 0.3 and sc_raw[p2] > 0.3:
                    pt1 = (int(kps_raw[p1][0]*scale_tp), int(kps_raw[p1][1]*scale_tp))
                    pt2 = (int(kps_raw[p2][0]*scale_tp), int(kps_raw[p2][1]*scale_tp))
                    cv2.line(panel2, pt1, pt2, (0,0,255), 2)
            for j, (kp, sc) in enumerate(zip(kps_raw, sc_raw)):
                if sc > 0.3:
                    cv2.circle(panel2, (int(kp[0]*scale_tp), int(kp[1]*scale_tp)), 3, (0,255,0), -1)

            norm_tp, _ = normalise_skeleton(stab_frame)
            if norm_tp:
                char_tp = render_character_v2(norm_tp, canvas_size=(panel_w_anim, panel_h))
                panel3 = cv2.cvtColor(np.array(char_tp.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                panel3 = np.ones((panel_h, panel_w_anim, 3), dtype=np.uint8) * 240

            cv2.putText(panel1, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel2, 'Skeleton', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel3, '2D Character', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)

            combined = np.hstack([panel1, panel2, panel3])
            out_tp.write(combined)

            if i % 50 == 0:
                print(f"Processing {i}/{len(stab_kps_list)} frames")

        cap_tp.release()
        out_tp.release()
        print(f"Three-panel video saved to {output_path}")

    generate_three_panel('dogvideo.mp4', raw_kps, stab_kps, output_path='three_panel.mp4')
    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v3(normalised_kps, canvas_size=(500, 400), offset=(200, 180),
                             body_color='#8B6914', head_color='#D4A017'):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset

        def pt_c(name):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox), int(kp_c[1] + oy))

        def mid_pt(name1, name2):
            p1 = normalised_kps[name1]
            p2 = normalised_kps[name2]
            return (int((p1[0]+p2[0])/2 + ox), int((p1[1]+p2[1])/2 + oy))

        neck = pt_c('Neck')
        tail = pt_c('root_of_tail')
        nose = pt_c('Nose')
        l_eye = pt_c('L_Eye')
        r_eye = pt_c('R_Eye')

        tail_end = pt_c('R_Hip')
        draw_c.line([tail, tail_end], fill='#6B4F0E', width=8)
        draw_c.ellipse([tail_end[0]-6, tail_end[1]-6, tail_end[0]+6, tail_end[1]+6], fill='#6B4F0E')

        l_hip = pt_c('L_Hip')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill='#6B4F0E', width=16)
        draw_c.line([l_knee, l_bpaw], fill='#5C3D00', width=12)
        draw_c.ellipse([l_knee[0]-8, l_knee[1]-8, l_knee[0]+8, l_knee[1]+8], fill='#6B4F0E')
        draw_c.ellipse([l_bpaw[0]-7, l_bpaw[1]-7, l_bpaw[0]+7, l_bpaw[1]+7], fill='#5C3D00')

        body_cx = (neck[0] + tail[0]) // 2
        body_cy = (neck[1] + tail[1]) // 2
        body_half_w = int(np.linalg.norm(np.array(tail) - np.array(neck)) / 2 * 1.1)
        body_half_h = int(body_half_w * 0.4)

        dx_body = tail[0] - neck[0]
        dy_body = tail[1] - neck[1]
        body_angle = -np.degrees(np.arctan2(dy_body, dx_body))

        body_img = Image.new('RGBA', (body_half_w*2, body_half_h*2), (0,0,0,0))
        body_draw = ImageDraw.Draw(body_img)
        body_draw.ellipse([4, 4, body_half_w*2-4, body_half_h*2-4], fill=body_color, outline='#5C3D00', width=2)
        body_rotated = body_img.rotate(body_angle, expand=True)
        bx = body_cx - body_rotated.width // 2
        by = body_cy - body_rotated.height // 2
        canvas.paste(body_rotated, (bx, by), body_rotated)
        draw_c = ImageDraw.Draw(canvas)

        l_shoulder = pt_c('L_Shoulder')
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill='#7A5C10', width=16)
        draw_c.line([l_elbow, l_fpaw], fill='#5C3D00', width=12)
        draw_c.ellipse([l_shoulder[0]-9, l_shoulder[1]-9, l_shoulder[0]+9, l_shoulder[1]+9], fill=body_color)
        draw_c.ellipse([l_elbow[0]-8, l_elbow[1]-8, l_elbow[0]+8, l_elbow[1]+8], fill='#7A5C10')
        draw_c.ellipse([l_fpaw[0]-7, l_fpaw[1]-7, l_fpaw[0]+7, l_fpaw[1]+7], fill='#5C3D00')

        head_cx = (nose[0] + neck[0]) // 2
        head_cy = (nose[1] + neck[1]) // 2
        head_r = int(np.linalg.norm(np.array(nose) - np.array(neck)) / 2 * 1.2)

        draw_c.ellipse([head_cx-head_r, head_cy-head_r, head_cx+head_r, head_cy+head_r],
                        fill=head_color, outline='#5C3D00', width=2)

        ear_offset_x = int(head_r * 0.5)
        ear_offset_y = int(head_r * 0.9)
        ear_r = int(head_r * 0.45)
        draw_c.ellipse([head_cx-ear_offset_x-ear_r, head_cy-ear_offset_y-ear_r,
                        head_cx-ear_offset_x+ear_r, head_cy-ear_offset_y+ear_r],
                        fill='#A07818', outline='#5C3D00', width=1)
        draw_c.ellipse([head_cx+ear_offset_x-ear_r, head_cy-ear_offset_y-ear_r,
                        head_cx+ear_offset_x+ear_r, head_cy-ear_offset_y+ear_r],
                        fill='#A07818', outline='#5C3D00', width=1)

        draw_c.ellipse([l_eye[0]-5, l_eye[1]-5, l_eye[0]+5, l_eye[1]+5], fill='white', outline='black', width=1)
        draw_c.ellipse([l_eye[0]-2, l_eye[1]-2, l_eye[0]+2, l_eye[1]+2], fill='black')
        draw_c.ellipse([r_eye[0]-5, r_eye[1]-5, r_eye[0]+5, r_eye[1]+5], fill='white', outline='black', width=1)
        draw_c.ellipse([r_eye[0]-2, r_eye[1]-2, r_eye[0]+2, r_eye[1]+2], fill='black')

        draw_c.ellipse([nose[0]-6, nose[1]-4, nose[0]+6, nose[1]+4], fill='#2C1A00')

        return canvas

    char_v3_test = render_character_v3(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v3_test)
    plt.axis('off')
    plt.title('Upgraded 2D Cartoon Dog')
    plt.savefig('character_v3_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")

    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v6(normalised_kps, canvas_size=(500, 400), offset=(200, 180), body_color='#8B6914', head_color='#D4A017'):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        def pt_c(name):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox), int(kp_c[1] + oy))
        neck = pt_c('Neck')
        tail = pt_c('root_of_tail')
        nose = pt_c('Nose')
        spine_len = np.linalg.norm(np.array(tail) - np.array(neck))
        head_r = int(spine_len * 0.16)
        ear_r = int(head_r * 0.35)
        tail_tip_x = tail[0] + int(spine_len * 0.18)
        tail_tip_y = tail[1] - int(spine_len * 0.15)
        draw_c.line([tail, (tail_tip_x, tail_tip_y)], fill='#6B4F0E', width=6)
        draw_c.ellipse([tail_tip_x-3, tail_tip_y-3, tail_tip_x+3, tail_tip_y+3], fill='#6B4F0E')
        r_hip_pt = pt_c('R_Hip')
        r_knee = pt_c('R_Knee')
        r_bpaw = pt_c('R_B_Paw')
        draw_c.line([r_hip_pt, r_knee], fill='#7A5C10', width=10)
        draw_c.line([r_knee, r_bpaw], fill='#6B4F0E', width=8)
        draw_c.ellipse([r_knee[0]-5, r_knee[1]-5, r_knee[0]+5, r_knee[1]+5], fill='#7A5C10')
        draw_c.ellipse([r_bpaw[0]-4, r_bpaw[1]-4, r_bpaw[0]+4, r_bpaw[1]+4], fill='#6B4F0E')
        r_shoulder = pt_c('R_Shoulder')
        r_elbow = pt_c('R_Elbow')
        r_fpaw = pt_c('R_F_Paw')
        draw_c.line([r_shoulder, r_elbow], fill='#7A5C10', width=10)
        draw_c.line([r_elbow, r_fpaw], fill='#6B4F0E', width=8)
        draw_c.ellipse([r_elbow[0]-5, r_elbow[1]-5, r_elbow[0]+5, r_elbow[1]+5], fill='#7A5C10')
        draw_c.ellipse([r_fpaw[0]-4, r_fpaw[1]-4, r_fpaw[0]+4, r_fpaw[1]+4], fill='#6B4F0E')
        body_cx = (neck[0] + tail[0]) // 2
        body_cy = (neck[1] + tail[1]) // 2
        body_half_w = int(spine_len / 2 * 1.05)
        body_half_h = int(spine_len * 0.18)
        dx_body = tail[0] - neck[0]
        dy_body = tail[1] - neck[1]
        body_angle = -np.degrees(np.arctan2(dy_body, dx_body))
        body_img = Image.new('RGBA', (body_half_w*2, body_half_h*2), (0,0,0,0))
        body_draw = ImageDraw.Draw(body_img)
        body_draw.ellipse([2, 2, body_half_w*2-2, body_half_h*2-2], fill=body_color, outline='#5C3D00', width=2)
        body_rotated = body_img.rotate(body_angle, expand=True)
        canvas.paste(body_rotated, (body_cx - body_rotated.width//2, body_cy - body_rotated.height//2), body_rotated)
        draw_c = ImageDraw.Draw(canvas)
        l_hip = pt_c('L_Hip')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill='#6B4F0E', width=12)
        draw_c.line([l_knee, l_bpaw], fill='#5C3D00', width=10)
        draw_c.ellipse([l_hip[0]-6, l_hip[1]-6, l_hip[0]+6, l_hip[1]+6], fill=body_color)
        draw_c.ellipse([l_knee[0]-5, l_knee[1]-5, l_knee[0]+5, l_knee[1]+5], fill='#6B4F0E')
        draw_c.ellipse([l_bpaw[0]-4, l_bpaw[1]-4, l_bpaw[0]+4, l_bpaw[1]+4], fill='#5C3D00')
        l_shoulder = pt_c('L_Shoulder')
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill='#7A5C10', width=12)
        draw_c.line([l_elbow, l_fpaw], fill='#5C3D00', width=10)
        draw_c.ellipse([l_shoulder[0]-6, l_shoulder[1]-6, l_shoulder[0]+6, l_shoulder[1]+6], fill=body_color)
        draw_c.ellipse([l_elbow[0]-5, l_elbow[1]-5, l_elbow[0]+5, l_elbow[1]+5], fill='#7A5C10')
        draw_c.ellipse([l_fpaw[0]-4, l_fpaw[1]-4, l_fpaw[0]+4, l_fpaw[1]+4], fill='#5C3D00')
        head_dir_x = nose[0] - neck[0]
        head_dir_y = nose[1] - neck[1]
        head_dist = np.linalg.norm([head_dir_x, head_dir_y])
        if head_dist > 0:
            head_dir_x /= head_dist
            head_dir_y /= head_dist
        head_cx = neck[0] + int(head_dir_x * head_r * 1.2)
        head_cy = neck[1] + int(head_dir_y * head_r * 1.2)
        draw_c.ellipse([head_cx-head_r, head_cy-head_r, head_cx+head_r, head_cy+head_r], fill=head_color, outline='#5C3D00', width=2)
        ear_cx = head_cx
        ear_cy = head_cy - int(head_r * 0.85)
        draw_c.ellipse([ear_cx-ear_r, ear_cy-ear_r*2, ear_cx+ear_r, ear_cy], fill='#A07818', outline='#5C3D00', width=1)
        snout_cx = head_cx + int(head_dir_x * head_r * 0.75)
        snout_cy = head_cy + int(head_dir_y * head_r * 0.75) + int(head_r * 0.1)
        snout_rx = int(head_r * 0.45)
        snout_ry = int(head_r * 0.3)
        draw_c.ellipse([snout_cx-snout_rx, snout_cy-snout_ry, snout_cx+snout_rx, snout_cy+snout_ry], fill='#C4941A', outline='#5C3D00', width=1)
        nose_x = snout_cx + int(head_dir_x * snout_rx * 0.5)
        nose_y = snout_cy
        draw_c.ellipse([nose_x-3, nose_y-3, nose_x+3, nose_y+3], fill='#2C1A00')
        eye_cx = head_cx + int(head_dir_x * head_r * 0.25)
        eye_cy = head_cy - int(head_r * 0.15)
        draw_c.ellipse([eye_cx-4, eye_cy-4, eye_cx+4, eye_cy+4], fill='white', outline='black', width=1)
        draw_c.ellipse([eye_cx-2, eye_cy-2, eye_cx+2, eye_cy+2], fill='black')
        return canvas

    char_v6_test = render_character_v6(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v6_test)
    plt.axis('off')
    plt.title('Cartoon Dog v6 - Side View')
    plt.savefig('character_v6_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v7(normalised_kps, canvas_size=(500, 400), offset=(200, 180), body_color='#8B6914', head_color='#D4A017'):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        def pt_c(name):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox), int(kp_c[1] + oy))
        neck = pt_c('Neck')
        tail = pt_c('root_of_tail')
        nose = pt_c('Nose')
        spine_len = np.linalg.norm(np.array(tail) - np.array(neck))
        head_r = int(spine_len * 0.16)
        ear_r = int(head_r * 0.35)
        tail_tip_x = tail[0] + int(spine_len * 0.18)
        tail_tip_y = tail[1] - int(spine_len * 0.15)
        draw_c.line([tail, (tail_tip_x, tail_tip_y)], fill='#6B4F0E', width=6)
        draw_c.ellipse([tail_tip_x-3, tail_tip_y-3, tail_tip_x+3, tail_tip_y+3], fill='#6B4F0E')
        l_hip = pt_c('L_Hip')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill='#6B4F0E', width=12)
        draw_c.line([l_knee, l_bpaw], fill='#5C3D00', width=10)
        draw_c.ellipse([l_hip[0]-6, l_hip[1]-6, l_hip[0]+6, l_hip[1]+6], fill=body_color)
        draw_c.ellipse([l_knee[0]-5, l_knee[1]-5, l_knee[0]+5, l_knee[1]+5], fill='#6B4F0E')
        draw_c.ellipse([l_bpaw[0]-4, l_bpaw[1]-4, l_bpaw[0]+4, l_bpaw[1]+4], fill='#5C3D00')
        body_cx = (neck[0] + tail[0]) // 2
        body_cy = (neck[1] + tail[1]) // 2
        body_half_w = int(spine_len / 2 * 1.05)
        body_half_h = int(spine_len * 0.18)
        dx_body = tail[0] - neck[0]
        dy_body = tail[1] - neck[1]
        body_angle = -np.degrees(np.arctan2(dy_body, dx_body))
        body_img = Image.new('RGBA', (body_half_w*2, body_half_h*2), (0,0,0,0))
        body_draw = ImageDraw.Draw(body_img)
        body_draw.ellipse([2, 2, body_half_w*2-2, body_half_h*2-2], fill=body_color, outline='#5C3D00', width=2)
        body_rotated = body_img.rotate(body_angle, expand=True)
        canvas.paste(body_rotated, (body_cx - body_rotated.width//2, body_cy - body_rotated.height//2), body_rotated)
        draw_c = ImageDraw.Draw(canvas)
        l_shoulder = pt_c('L_Shoulder')
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill='#7A5C10', width=12)
        draw_c.line([l_elbow, l_fpaw], fill='#5C3D00', width=10)
        draw_c.ellipse([l_shoulder[0]-6, l_shoulder[1]-6, l_shoulder[0]+6, l_shoulder[1]+6], fill=body_color)
        draw_c.ellipse([l_elbow[0]-5, l_elbow[1]-5, l_elbow[0]+5, l_elbow[1]+5], fill='#7A5C10')
        draw_c.ellipse([l_fpaw[0]-4, l_fpaw[1]-4, l_fpaw[0]+4, l_fpaw[1]+4], fill='#5C3D00')
        head_dir_x = nose[0] - neck[0]
        head_dir_y = nose[1] - neck[1]
        head_dist = np.linalg.norm([head_dir_x, head_dir_y])
        if head_dist > 0:
            head_dir_x /= head_dist
            head_dir_y /= head_dist
        head_cx = neck[0] + int(head_dir_x * head_r * 1.2)
        head_cy = neck[1] + int(head_dir_y * head_r * 1.2)
        draw_c.ellipse([head_cx-head_r, head_cy-head_r, head_cx+head_r, head_cy+head_r], fill=head_color, outline='#5C3D00', width=2)
        ear_cx = head_cx
        ear_cy = head_cy - int(head_r * 0.85)
        draw_c.ellipse([ear_cx-ear_r, ear_cy-ear_r*2, ear_cx+ear_r, ear_cy], fill='#A07818', outline='#5C3D00', width=1)
        snout_cx = head_cx + int(head_dir_x * head_r * 0.75)
        snout_cy = head_cy + int(head_dir_y * head_r * 0.75) + int(head_r * 0.1)
        snout_rx = int(head_r * 0.45)
        snout_ry = int(head_r * 0.3)
        draw_c.ellipse([snout_cx-snout_rx, snout_cy-snout_ry, snout_cx+snout_rx, snout_cy+snout_ry], fill='#C4941A', outline='#5C3D00', width=1)
        nose_x = snout_cx + int(head_dir_x * snout_rx * 0.5)
        nose_y = snout_cy
        draw_c.ellipse([nose_x-3, nose_y-3, nose_x+3, nose_y+3], fill='#2C1A00')
        eye_cx = head_cx + int(head_dir_x * head_r * 0.25)
        eye_cy = head_cy - int(head_r * 0.15)
        draw_c.ellipse([eye_cx-4, eye_cy-4, eye_cx+4, eye_cy+4], fill='white', outline='black', width=1)
        draw_c.ellipse([eye_cx-2, eye_cy-2, eye_cx+2, eye_cy+2], fill='black')
        return canvas

    char_v7_test = render_character_v7(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v7_test)
    plt.axis('off')
    plt.title('Cartoon Dog v7 - Side View Single Legs')
    plt.savefig('character_v7_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return (render_character_v7,)


@app.cell
def _(cv2, normalise_skeleton, np, raw_kps, render_character_v7, stab_kps):
    def generate_three_panel_v5(vid_path_tp5, raw_list_tp5, stab_list_tp5, out_path_tp5='three_panel_v5.mp4', fps_tp5=30):
        SKEL_IDX_V5 = [(0,2),(1,2),(2,3),(3,5),(3,8),(5,6),(6,7),(8,9),(9,10),(3,4),(4,11),(4,14),(11,12),(12,13),(14,15),(15,16)]
        cap_tp5 = cv2.VideoCapture(vid_path_tp5)
        orig_w5 = int(cap_tp5.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h5 = int(cap_tp5.get(cv2.CAP_PROP_FRAME_HEIGHT))
        panel_h5 = 400
        scale_tp5 = panel_h5 / orig_h5
        panel_w_orig5 = int(orig_w5 * scale_tp5)
        panel_w_anim5 = 500
        total_w5 = panel_w_orig5 * 2 + panel_w_anim5
        fourcc_tp5 = cv2.VideoWriter_fourcc(*'mp4v')
        out_tp5 = cv2.VideoWriter(out_path_tp5, fourcc_tp5, fps_tp5, (total_w5, panel_h5))
        for i, stab_frame5 in enumerate(stab_list_tp5):
            ret_tp5, orig_frame5 = cap_tp5.read()
            if not ret_tp5:
                break
            panel1_v5 = cv2.resize(orig_frame5, (panel_w_orig5, panel_h5))
            panel2_v5 = panel1_v5.copy()
            raw_frame5 = raw_list_tp5[i]
            kps_raw5 = raw_frame5['keypoints']
            sc_raw5 = raw_frame5['scores']
            for (p1, p2) in SKEL_IDX_V5:
                if sc_raw5[p1] > 0.3 and sc_raw5[p2] > 0.3:
                    pt1_v5 = (int(kps_raw5[p1][0]*scale_tp5), int(kps_raw5[p1][1]*scale_tp5))
                    pt2_v5 = (int(kps_raw5[p2][0]*scale_tp5), int(kps_raw5[p2][1]*scale_tp5))
                    cv2.line(panel2_v5, pt1_v5, pt2_v5, (0,0,255), 2)
            for j5, (kp5, sc5) in enumerate(zip(kps_raw5, sc_raw5)):
                if sc5 > 0.3:
                    cv2.circle(panel2_v5, (int(kp5[0]*scale_tp5), int(kp5[1]*scale_tp5)), 3, (0,255,0), -1)
            norm_tp5, _ = normalise_skeleton(stab_frame5)
            if norm_tp5:
                char_tp5 = render_character_v7(norm_tp5, canvas_size=(panel_w_anim5, panel_h5))
                panel3_v5 = cv2.cvtColor(np.array(char_tp5.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                panel3_v5 = np.ones((panel_h5, panel_w_anim5, 3), dtype=np.uint8) * 240
            cv2.putText(panel1_v5, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel2_v5, 'Skeleton', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel3_v5, '2D Character', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
            combined_v5 = np.hstack([panel1_v5, panel2_v5, panel3_v5])
            out_tp5.write(combined_v5)
            if i % 50 == 0:
                print(f"Processing {i}/{len(stab_list_tp5)} frames")
        cap_tp5.release()
        out_tp5.release()
        print(f"Saved to {out_path_tp5}")

    generate_three_panel_v5('dogvideo.mp4', raw_kps, stab_kps)
    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v10(normalised_kps, canvas_size=(500, 400), offset=(200, 180), body_color='#3D3D3D', belly_color='#5A5A5A', head_color='#4A4A4A', leg_color='#333333'):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        r_leg_shift = 6
        def pt_c(name, x_off=0, y_off=0):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox + x_off), int(kp_c[1] + oy + y_off))
        neck = pt_c('Neck')
        tail_pt = pt_c('root_of_tail')
        nose = pt_c('Nose')
        spine_len = np.linalg.norm(np.array(tail_pt) - np.array(neck))
        head_r = int(spine_len * 0.16)
        body_half_h = int(spine_len * 0.18)
        joint_r = 4
        paw_r = 5
        head_dir_x = nose[0] - neck[0]
        head_dir_y = nose[1] - neck[1]
        head_dist = np.linalg.norm([head_dir_x, head_dir_y])
        if head_dist > 0:
            head_dir_x /= head_dist
            head_dir_y /= head_dist
        tail_tip_x = tail_pt[0] + int(spine_len * 0.15)
        tail_tip_y = tail_pt[1] + int(spine_len * 0.05)
        tail_mid_x = tail_pt[0] + int(spine_len * 0.08)
        tail_mid_y = tail_pt[1] - int(spine_len * 0.05)
        draw_c.line([tail_pt, (tail_mid_x, tail_mid_y)], fill='#2A2A2A', width=7)
        draw_c.line([(tail_mid_x, tail_mid_y), (tail_tip_x, tail_tip_y)], fill='#2A2A2A', width=5)
        draw_c.ellipse([tail_tip_x-3, tail_tip_y-3, tail_tip_x+3, tail_tip_y+3], fill='#2A2A2A')
        r_hip = pt_c('R_Hip', r_leg_shift, 0)
        r_knee = pt_c('R_Knee', r_leg_shift, 0)
        r_bpaw = pt_c('R_B_Paw', r_leg_shift, 0)
        draw_c.line([r_hip, r_knee], fill='#555555', width=11)
        draw_c.line([r_knee, r_bpaw], fill='#555555', width=8)
        draw_c.ellipse([r_hip[0]-joint_r, r_hip[1]-joint_r, r_hip[0]+joint_r, r_hip[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_knee[0]-joint_r, r_knee[1]-joint_r, r_knee[0]+joint_r, r_knee[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_bpaw[0]-paw_r, r_bpaw[1]-paw_r, r_bpaw[0]+paw_r, r_bpaw[1]+paw_r], fill='#444444', outline='#222222', width=1)
        r_shoulder = pt_c('R_Shoulder', r_leg_shift, 0)
        r_elbow = pt_c('R_Elbow', r_leg_shift, 0)
        r_fpaw = pt_c('R_F_Paw', r_leg_shift, 0)
        draw_c.line([r_shoulder, r_elbow], fill='#555555', width=11)
        draw_c.line([r_elbow, r_fpaw], fill='#555555', width=8)
        draw_c.ellipse([r_shoulder[0]-joint_r, r_shoulder[1]-joint_r, r_shoulder[0]+joint_r, r_shoulder[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_elbow[0]-joint_r, r_elbow[1]-joint_r, r_elbow[0]+joint_r, r_elbow[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_fpaw[0]-paw_r, r_fpaw[1]-paw_r, r_fpaw[0]+paw_r, r_fpaw[1]+paw_r], fill='#444444', outline='#222222', width=1)
        body_cx = (neck[0] + tail_pt[0]) // 2
        body_cy = (neck[1] + tail_pt[1]) // 2
        body_half_w = int(spine_len / 2 * 1.05)
        dx_body = tail_pt[0] - neck[0]
        dy_body = tail_pt[1] - neck[1]
        body_angle = -np.degrees(np.arctan2(dy_body, dx_body))
        body_w = body_half_w * 2
        body_h = body_half_h * 2
        body_img = Image.new('RGBA', (body_w, body_h), (0, 0, 0, 0))
        body_drw = ImageDraw.Draw(body_img)
        body_drw.ellipse([2, 2, body_w-2, body_h-2], fill=body_color, outline='#222222', width=2)
        chest_w = int(body_w * 0.45)
        chest_h = int(body_h * 0.85)
        body_drw.ellipse([2, body_h//2-chest_h//2, chest_w, body_h//2+chest_h//2], fill='#4A4A4A')
        belly_top = int(body_h * 0.55)
        body_drw.ellipse([int(body_w*0.15), belly_top, int(body_w*0.85), body_h-2], fill=belly_color)
        body_rotated = body_img.rotate(body_angle, expand=True)
        canvas.paste(body_rotated, (body_cx - body_rotated.width//2, body_cy - body_rotated.height//2), body_rotated)
        draw_c = ImageDraw.Draw(canvas)
        l_hip = pt_c('L_Hip')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill=leg_color, width=13)
        draw_c.line([l_knee, l_bpaw], fill=leg_color, width=10)
        draw_c.ellipse([l_hip[0]-joint_r, l_hip[1]-joint_r, l_hip[0]+joint_r, l_hip[1]+joint_r], fill=body_color)
        draw_c.ellipse([l_knee[0]-joint_r, l_knee[1]-joint_r, l_knee[0]+joint_r, l_knee[1]+joint_r], fill='#4A4A4A')
        draw_c.ellipse([l_bpaw[0]-paw_r, l_bpaw[1]-paw_r, l_bpaw[0]+paw_r, l_bpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        l_shoulder = pt_c('L_Shoulder')
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill=leg_color, width=13)
        draw_c.line([l_elbow, l_fpaw], fill=leg_color, width=10)
        draw_c.ellipse([l_shoulder[0]-joint_r, l_shoulder[1]-joint_r, l_shoulder[0]+joint_r, l_shoulder[1]+joint_r], fill=body_color)
        draw_c.ellipse([l_elbow[0]-joint_r, l_elbow[1]-joint_r, l_elbow[0]+joint_r, l_elbow[1]+joint_r], fill='#4A4A4A')
        draw_c.ellipse([l_fpaw[0]-paw_r, l_fpaw[1]-paw_r, l_fpaw[0]+paw_r, l_fpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        head_cx = neck[0] + int(head_dir_x * head_r * 1.2)
        head_cy = neck[1] + int(head_dir_y * head_r * 1.2)
        draw_c.ellipse([head_cx-head_r, head_cy-head_r, head_cx+head_r, head_cy+head_r], fill=head_color, outline='#222222', width=2)
        ear_cx = head_cx + int(head_dir_x * head_r * 0.1)
        ear_cy = head_cy - int(head_r * 0.5)
        ear_w = int(head_r * 0.5)
        ear_h = int(head_r * 0.9)
        ear_pts = [(ear_cx, ear_cy - int(ear_h*0.3)), (ear_cx - ear_w, ear_cy + ear_h), (ear_cx + int(ear_w*0.3), ear_cy + ear_h)]
        draw_c.polygon(ear_pts, fill='#3A3A3A', outline='#222222')
        snout_cx = head_cx + int(head_dir_x * head_r * 0.75)
        snout_cy = head_cy + int(head_dir_y * head_r * 0.75) + int(head_r * 0.15)
        snout_rx = int(head_r * 0.5)
        snout_ry = int(head_r * 0.35)
        draw_c.ellipse([snout_cx-snout_rx, snout_cy-snout_ry, snout_cx+snout_rx, snout_cy+snout_ry], fill='#5A5A5A', outline='#333333', width=1)
        nose_x = snout_cx + int(head_dir_x * snout_rx * 0.4)
        nose_y = snout_cy - int(snout_ry * 0.15)
        draw_c.ellipse([nose_x-5, nose_y-3, nose_x+5, nose_y+3], fill='#1A1A1A')
        mouth_start = (snout_cx, snout_cy + int(snout_ry * 0.2))
        mouth_end = (snout_cx + int(snout_rx * 0.5), snout_cy + int(snout_ry * 0.5))
        draw_c.line([mouth_start, mouth_end], fill='#333333', width=1)
        eye_cx = head_cx + int(head_dir_x * head_r * 0.3)
        eye_cy = head_cy - int(head_r * 0.15)
        draw_c.ellipse([eye_cx-5, eye_cy-5, eye_cx+5, eye_cy+5], fill='white', outline='#222222', width=1)
        draw_c.ellipse([eye_cx-2, eye_cy-2, eye_cx+2, eye_cy+2], fill='#1A1A1A')
        draw_c.ellipse([eye_cx, eye_cy-3, eye_cx+2, eye_cy-1], fill='white')
        draw_c.ellipse([neck[0]-joint_r, neck[1]-joint_r, neck[0]+joint_r, neck[1]+joint_r], fill='#444444')
        draw_c.ellipse([tail_pt[0]-joint_r, tail_pt[1]-joint_r, tail_pt[0]+joint_r, tail_pt[1]+joint_r], fill='#444444')
        return canvas

    char_v10_test = render_character_v10(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v10_test)
    plt.axis('off')
    plt.title('Cartoon Dog v10 - Black Lab Style')
    plt.savefig('character_v10_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v14(normalised_kps, canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        r_leg_shift = 6
        def pt_c(name, x_off=0, y_off=0):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox + x_off), int(kp_c[1] + oy + y_off))
        neck = pt_c('Neck')
        tail_pt = pt_c('root_of_tail')
        nose = pt_c('Nose')
        l_shoulder = pt_c('L_Shoulder')
        l_hip = pt_c('L_Hip')
        r_shoulder = pt_c('R_Shoulder', r_leg_shift, 0)
        r_hip_pt = pt_c('R_Hip', r_leg_shift, 0)
        spine_len = np.linalg.norm(np.array(tail_pt) - np.array(neck))
        head_r = int(spine_len * 0.16)
        joint_r = 5
        paw_r = 5
        hx = nose[0] - neck[0]
        hy = nose[1] - neck[1]
        hd = np.linalg.norm([hx, hy])
        if hd > 0:
            hx /= hd
            hy /= hd
        px = -hy
        py = hx
        s = spine_len
        back_rise = int(s * 0.08)
        tail_tip_x = tail_pt[0] + int(s * 0.15)
        tail_tip_y = tail_pt[1] + int(s * 0.02)
        tail_mid_x = tail_pt[0] + int(s * 0.08)
        tail_mid_y = tail_pt[1] - int(s * 0.08)
        draw_c.line([tail_pt, (tail_mid_x, tail_mid_y)], fill='#2A2A2A', width=6)
        draw_c.line([(tail_mid_x, tail_mid_y), (tail_tip_x, tail_tip_y)], fill='#2A2A2A', width=4)
        r_knee = pt_c('R_Knee', r_leg_shift, 0)
        r_bpaw = pt_c('R_B_Paw', r_leg_shift, 0)
        draw_c.line([r_hip_pt, r_knee], fill='#4A4A4A', width=10)
        draw_c.line([r_knee, r_bpaw], fill='#4A4A4A', width=7)
        draw_c.ellipse([r_hip_pt[0]-joint_r, r_hip_pt[1]-joint_r, r_hip_pt[0]+joint_r, r_hip_pt[1]+joint_r], fill='#555555')
        draw_c.ellipse([r_knee[0]-joint_r, r_knee[1]-joint_r, r_knee[0]+joint_r, r_knee[1]+joint_r], fill='#555555')
        draw_c.ellipse([r_bpaw[0]-paw_r, r_bpaw[1]-paw_r, r_bpaw[0]+paw_r, r_bpaw[1]+paw_r], fill='#3A3A3A')
        r_elbow = pt_c('R_Elbow', r_leg_shift, 0)
        r_fpaw = pt_c('R_F_Paw', r_leg_shift, 0)
        draw_c.line([r_shoulder, r_elbow], fill='#4A4A4A', width=10)
        draw_c.line([r_elbow, r_fpaw], fill='#4A4A4A', width=7)
        draw_c.ellipse([r_shoulder[0]-joint_r, r_shoulder[1]-joint_r, r_shoulder[0]+joint_r, r_shoulder[1]+joint_r], fill='#555555')
        draw_c.ellipse([r_elbow[0]-joint_r, r_elbow[1]-joint_r, r_elbow[0]+joint_r, r_elbow[1]+joint_r], fill='#555555')
        draw_c.ellipse([r_fpaw[0]-paw_r, r_fpaw[1]-paw_r, r_fpaw[0]+paw_r, r_fpaw[1]+paw_r], fill='#3A3A3A')
        body_pts = [
            (neck[0], neck[1] - back_rise),
            (int((neck[0]+tail_pt[0])*0.4), int((neck[1]+tail_pt[1])*0.5) - int(back_rise*1.2)),
            (int((neck[0]+tail_pt[0])*0.5), int((neck[1]+tail_pt[1])*0.5) - int(back_rise*1.0)),
            (tail_pt[0], tail_pt[1] - int(back_rise*0.3)),
            (l_hip[0], l_hip[1]),
            (int((l_hip[0]+l_shoulder[0])*0.5), max(l_hip[1], l_shoulder[1]) + int(s*0.05)),
            (l_shoulder[0], l_shoulder[1]),
            (neck[0], neck[1] + int(s*0.1)),
        ]
        draw_c.polygon(body_pts, fill='#3D3D3D', outline='#222222')
        belly_y_offset = int(s * 0.03)
        belly_pts = [
            (l_shoulder[0], l_shoulder[1] - belly_y_offset),
            (int((l_shoulder[0]+l_hip[0])*0.5), max(l_shoulder[1], l_hip[1]) + belly_y_offset),
            (l_hip[0], l_hip[1] - belly_y_offset),
        ]
        draw_c.polygon(belly_pts, fill='#4A4A4A')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill='#333333', width=12)
        draw_c.line([l_knee, l_bpaw], fill='#333333', width=9)
        draw_c.ellipse([l_hip[0]-joint_r, l_hip[1]-joint_r, l_hip[0]+joint_r, l_hip[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_knee[0]-joint_r, l_knee[1]-joint_r, l_knee[0]+joint_r, l_knee[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_bpaw[0]-paw_r, l_bpaw[1]-paw_r, l_bpaw[0]+paw_r, l_bpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill='#333333', width=12)
        draw_c.line([l_elbow, l_fpaw], fill='#333333', width=9)
        draw_c.ellipse([l_shoulder[0]-joint_r, l_shoulder[1]-joint_r, l_shoulder[0]+joint_r, l_shoulder[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_elbow[0]-joint_r, l_elbow[1]-joint_r, l_elbow[0]+joint_r, l_elbow[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_fpaw[0]-paw_r, l_fpaw[1]-paw_r, l_fpaw[0]+paw_r, l_fpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        hw = int(head_r * 1.3)
        hh = int(head_r * 1.0)
        neck_w = int(head_r * 0.6)
        head_polygon = [
            (neck[0] + int(px * neck_w), neck[1] + int(py * neck_w)),
            (neck[0] + int(hx * hw) + int(px * hh), neck[1] + int(hy * hw) + int(py * hh)),
            (neck[0] + int(hx * hw * 1.4) + int(px * hh * 0.5), neck[1] + int(hy * hw * 1.4) + int(py * hh * 0.5)),
            (neck[0] + int(hx * hw * 1.5), neck[1] + int(hy * hw * 1.5)),
            (neck[0] + int(hx * hw * 1.4) - int(px * hh * 0.5), neck[1] + int(hy * hw * 1.4) - int(py * hh * 0.5)),
            (neck[0] + int(hx * hw * 0.5) - int(px * hh * 0.6), neck[1] + int(hy * hw * 0.5) - int(py * hh * 0.6)),
            (neck[0] - int(px * neck_w), neck[1] - int(py * neck_w)),
        ]
        draw_c.polygon(head_polygon, fill='#4A4A4A', outline='#222222')
        mz_cx = neck[0] + int(hx * hw * 1.35)
        mz_cy = neck[1] + int(hy * hw * 1.35) - int(px * hh * 0.15)
        mz_rx = int(head_r * 0.4)
        mz_ry = int(head_r * 0.28)
        draw_c.ellipse([mz_cx-mz_rx, mz_cy-mz_ry, mz_cx+mz_rx, mz_cy+mz_ry], fill='#5A5A5A', outline='#333333', width=1)
        ns_x = mz_cx + int(hx * mz_rx * 0.5)
        ns_y = mz_cy + int(hy * mz_rx * 0.5)
        draw_c.ellipse([ns_x-4, ns_y-3, ns_x+4, ns_y+3], fill='#1A1A1A')
        ear_bx = neck[0] + int(hx * hw * 0.4) + int(px * hh * 0.9)
        ear_by = neck[1] + int(hy * hw * 0.4) + int(py * hh * 0.9)
        ear_tip_x = ear_bx + int(hx * head_r * 0.5) + int(px * head_r * 0.2)
        ear_tip_y = ear_by + int(hy * head_r * 0.5) + int(py * head_r * 0.2)
        ear_end_x = ear_bx + int(hx * head_r * 0.6) - int(px * head_r * 0.3)
        ear_end_y = ear_by + int(hy * head_r * 0.6) - int(py * head_r * 0.3)
        draw_c.polygon([(ear_bx, ear_by), (ear_tip_x, ear_tip_y), (ear_end_x, ear_end_y)], fill='#363636', outline='#222222')
        ey_cx = neck[0] + int(hx * hw * 0.7) + int(px * hh * 0.4)
        ey_cy = neck[1] + int(hy * hw * 0.7) + int(py * hh * 0.4)
        draw_c.ellipse([ey_cx-5, ey_cy-5, ey_cx+5, ey_cy+5], fill='white', outline='#222222', width=1)
        draw_c.ellipse([ey_cx-2, ey_cy-2, ey_cx+2, ey_cy+2], fill='#1A1A1A')
        draw_c.ellipse([neck[0]-joint_r, neck[1]-joint_r, neck[0]+joint_r, neck[1]+joint_r], fill='#444444')
        draw_c.ellipse([tail_pt[0]-joint_r, tail_pt[1]-joint_r, tail_pt[0]+joint_r, tail_pt[1]+joint_r], fill='#444444')
        return canvas

    char_v14_test = render_character_v14(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v14_test)
    plt.axis('off')
    plt.title('Cartoon Dog v14')
    plt.savefig('character_v14_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return


@app.cell
def _(Image, ImageDraw, np, plt, test_norm):
    def render_character_v21(normalised_kps, canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        r_shift = 6
        def pt_c(name, x_off=0, y_off=0):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox + x_off), int(kp_c[1] + oy + y_off))
        neck = pt_c('Neck')
        tail_pt = pt_c('root_of_tail')
        nose = pt_c('Nose')
        l_shoulder = pt_c('L_Shoulder')
        l_hip = pt_c('L_Hip')
        spine_len = np.linalg.norm(np.array(tail_pt) - np.array(neck))
        head_r = int(spine_len * 0.16)
        joint_r = 5
        paw_r = 5
        hx = nose[0] - neck[0]
        hy = nose[1] - neck[1]
        hd = np.linalg.norm([hx, hy])
        if hd > 0:
            hx /= hd
            hy /= hd
        px = -hy
        py = hx
        spine_dx = tail_pt[0] - neck[0]
        spine_dy = tail_pt[1] - neck[1]
        back_rise = int(spine_len * 0.06)
        tail_tip_x = tail_pt[0] + int(spine_len * 0.15)
        tail_tip_y = tail_pt[1] + int(spine_len * 0.02)
        tail_mid_x = tail_pt[0] + int(spine_len * 0.08)
        tail_mid_y = tail_pt[1] - int(spine_len * 0.08)
        draw_c.line([tail_pt, (tail_mid_x, tail_mid_y)], fill='#2A2A2A', width=6)
        draw_c.line([(tail_mid_x, tail_mid_y), (tail_tip_x, tail_tip_y)], fill='#2A2A2A', width=4)
        r_shoulder = pt_c('R_Shoulder', r_shift, 0)
        r_elbow = pt_c('R_Elbow', r_shift, 0)
        r_fpaw = pt_c('R_F_Paw', r_shift, 0)
        draw_c.line([r_shoulder, r_elbow], fill='#555555', width=9)
        draw_c.line([r_elbow, r_fpaw], fill='#555555', width=7)
        draw_c.ellipse([r_shoulder[0]-joint_r, r_shoulder[1]-joint_r, r_shoulder[0]+joint_r, r_shoulder[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_elbow[0]-joint_r, r_elbow[1]-joint_r, r_elbow[0]+joint_r, r_elbow[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_fpaw[0]-paw_r, r_fpaw[1]-paw_r, r_fpaw[0]+paw_r, r_fpaw[1]+paw_r], fill='#4A4A4A')
        r_hip_pt = pt_c('R_Hip', r_shift, 0)
        r_knee = pt_c('R_Knee', r_shift, 0)
        r_bpaw = pt_c('R_B_Paw', r_shift, 0)
        draw_c.line([r_hip_pt, r_knee], fill='#555555', width=9)
        draw_c.line([r_knee, r_bpaw], fill='#555555', width=7)
        draw_c.ellipse([r_hip_pt[0]-joint_r, r_hip_pt[1]-joint_r, r_hip_pt[0]+joint_r, r_hip_pt[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_knee[0]-joint_r, r_knee[1]-joint_r, r_knee[0]+joint_r, r_knee[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_bpaw[0]-paw_r, r_bpaw[1]-paw_r, r_bpaw[0]+paw_r, r_bpaw[1]+paw_r], fill='#4A4A4A')
        mid_belly_x = (l_shoulder[0] + l_hip[0]) // 2
        mid_belly_y = (l_shoulder[1] + l_hip[1]) // 2 - int(spine_len * 0.02)
        body_pts = [
            (neck[0], neck[1] - back_rise),
            (neck[0] + int(spine_dx*0.3), neck[1] + int(spine_dy*0.3) - int(back_rise*1.1)),
            (neck[0] + int(spine_dx*0.6), neck[1] + int(spine_dy*0.6) - int(back_rise*0.8)),
            (tail_pt[0], tail_pt[1] - int(back_rise*0.2)),
            l_hip,
            (mid_belly_x, mid_belly_y),
            l_shoulder,
            (neck[0], neck[1] + int(spine_len * 0.04)),
        ]
        draw_c.polygon(body_pts, fill='#3D3D3D', outline='#222222')
        hw = int(head_r * 1.3)
        hh = int(head_r * 1.0)
        neck_w = int(head_r * 0.6)
        head_polygon = [
            (neck[0] + int(px * neck_w), neck[1] + int(py * neck_w)),
            (neck[0] + int(hx * hw * 0.5) + int(px * hh), neck[1] + int(hy * hw * 0.5) + int(py * hh)),
            (neck[0] + int(hx * hw) + int(px * hh * 0.8), neck[1] + int(hy * hw) + int(py * hh * 0.8)),
            (neck[0] + int(hx * hw * 1.4) + int(px * hh * 0.3), neck[1] + int(hy * hw * 1.4) + int(py * hh * 0.3)),
            (neck[0] + int(hx * hw * 1.5), neck[1] + int(hy * hw * 1.5)),
            (neck[0] + int(hx * hw * 1.4) - int(px * hh * 0.4), neck[1] + int(hy * hw * 1.4) - int(py * hh * 0.4)),
            (neck[0] + int(hx * hw * 0.8) - int(px * hh * 0.6), neck[1] + int(hy * hw * 0.8) - int(py * hh * 0.6)),
            (neck[0] + int(hx * hw * 0.3) - int(px * hh * 0.5), neck[1] + int(hy * hw * 0.3) - int(py * hh * 0.5)),
            (neck[0] - int(px * neck_w), neck[1] - int(py * neck_w)),
        ]
        draw_c.polygon(head_polygon, fill='#4A4A4A', outline='#222222')
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([l_shoulder, l_elbow], fill='#333333', width=12)
        draw_c.line([l_elbow, l_fpaw], fill='#333333', width=9)
        draw_c.ellipse([l_shoulder[0]-joint_r-1, l_shoulder[1]-joint_r-1, l_shoulder[0]+joint_r+1, l_shoulder[1]+joint_r+1], fill='#3D3D3D')
        draw_c.ellipse([l_elbow[0]-joint_r, l_elbow[1]-joint_r, l_elbow[0]+joint_r, l_elbow[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_fpaw[0]-paw_r, l_fpaw[1]-paw_r, l_fpaw[0]+paw_r, l_fpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([l_hip, l_knee], fill='#333333', width=12)
        draw_c.line([l_knee, l_bpaw], fill='#333333', width=9)
        draw_c.ellipse([l_hip[0]-joint_r-1, l_hip[1]-joint_r-1, l_hip[0]+joint_r+1, l_hip[1]+joint_r+1], fill='#3D3D3D')
        draw_c.ellipse([l_knee[0]-joint_r, l_knee[1]-joint_r, l_knee[0]+joint_r, l_knee[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_bpaw[0]-paw_r, l_bpaw[1]-paw_r, l_bpaw[0]+paw_r, l_bpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)
        mz_cx = neck[0] + int(hx * hw * 1.35)
        mz_cy = neck[1] + int(hy * hw * 1.35) - int(px * hh * 0.15)
        mz_rx = int(head_r * 0.4)
        mz_ry = int(head_r * 0.28)
        draw_c.ellipse([mz_cx-mz_rx, mz_cy-mz_ry, mz_cx+mz_rx, mz_cy+mz_ry], fill='#5A5A5A', outline='#333333', width=1)
        ns_x = mz_cx + int(hx * mz_rx * 0.5)
        ns_y = mz_cy + int(hy * mz_rx * 0.5)
        draw_c.ellipse([ns_x-4, ns_y-3, ns_x+4, ns_y+3], fill='#1A1A1A')
        ear_bx = neck[0] + int(hx * hw * 0.4) + int(px * hh * 0.9)
        ear_by = neck[1] + int(hy * hw * 0.4) + int(py * hh * 0.9)
        ear_tip_x = ear_bx + int(hx * head_r * 0.5) + int(px * head_r * 0.3)
        ear_tip_y = ear_by + int(hy * head_r * 0.5) + int(py * head_r * 0.3)
        ear_end_x = ear_bx + int(hx * head_r * 0.7) - int(px * head_r * 0.2)
        ear_end_y = ear_by + int(hy * head_r * 0.7) - int(py * head_r * 0.2)
        draw_c.polygon([(ear_bx, ear_by), (ear_tip_x, ear_tip_y), (ear_end_x, ear_end_y)], fill='#363636', outline='#222222')
        ey_cx = neck[0] + int(hx * hw * 0.7) + int(px * hh * 0.4)
        ey_cy = neck[1] + int(hy * hw * 0.7) + int(py * hh * 0.4)
        draw_c.ellipse([ey_cx-5, ey_cy-5, ey_cx+5, ey_cy+5], fill='white', outline='#222222', width=1)
        draw_c.ellipse([ey_cx-2, ey_cy-2, ey_cx+2, ey_cy+2], fill='#1A1A1A')
        draw_c.ellipse([neck[0]-joint_r, neck[1]-joint_r, neck[0]+joint_r, neck[1]+joint_r], fill='#444444')
        draw_c.ellipse([tail_pt[0]-joint_r, tail_pt[1]-joint_r, tail_pt[0]+joint_r, tail_pt[1]+joint_r], fill='#444444')
        return canvas

    char_v21_test = render_character_v21(test_norm)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v21_test)
    plt.axis('off')
    plt.title('Cartoon Dog v21 - v14 Head Style')
    plt.savefig('character_v21.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return (render_character_v21,)


@app.cell
def _(cv2, normalise_skeleton, np, raw_kps, render_character_v21, stab_kps):
    def generate_three_panel_v8(vid_path_tp8, raw_list_tp8, stab_list_tp8, out_path_tp8='three_panel_v8.mp4', fps_tp8=30):
        SKEL_IDX_V8 = [(0,2),(1,2),(2,3),(3,5),(3,8),(5,6),(6,7),(8,9),(9,10),(3,4),(4,11),(4,14),(11,12),(12,13),(14,15),(15,16)]
        cap_tp8 = cv2.VideoCapture(vid_path_tp8)
        orig_w8 = int(cap_tp8.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h8 = int(cap_tp8.get(cv2.CAP_PROP_FRAME_HEIGHT))
        panel_h8 = 400
        scale_tp8 = panel_h8 / orig_h8
        panel_w_orig8 = int(orig_w8 * scale_tp8)
        panel_w_anim8 = 500
        total_w8 = panel_w_orig8 * 2 + panel_w_anim8
        fourcc_tp8 = cv2.VideoWriter_fourcc(*'mp4v')
        out_tp8 = cv2.VideoWriter(out_path_tp8, fourcc_tp8, fps_tp8, (total_w8, panel_h8))
        for i, stab_frame8 in enumerate(stab_list_tp8):
            ret_tp8, orig_frame8 = cap_tp8.read()
            if not ret_tp8:
                break
            panel1_v8 = cv2.resize(orig_frame8, (panel_w_orig8, panel_h8))
            panel2_v8 = panel1_v8.copy()
            raw_frame8 = raw_list_tp8[i]
            kps_raw8 = raw_frame8['keypoints']
            sc_raw8 = raw_frame8['scores']
            for (p1, p2) in SKEL_IDX_V8:
                if sc_raw8[p1] > 0.3 and sc_raw8[p2] > 0.3:
                    pt1_v8 = (int(kps_raw8[p1][0]*scale_tp8), int(kps_raw8[p1][1]*scale_tp8))
                    pt2_v8 = (int(kps_raw8[p2][0]*scale_tp8), int(kps_raw8[p2][1]*scale_tp8))
                    cv2.line(panel2_v8, pt1_v8, pt2_v8, (0,0,255), 2)
            for j8, (kp8, sc8) in enumerate(zip(kps_raw8, sc_raw8)):
                if sc8 > 0.3:
                    cv2.circle(panel2_v8, (int(kp8[0]*scale_tp8), int(kp8[1]*scale_tp8)), 3, (0,255,0), -1)
            norm_tp8, _ = normalise_skeleton(stab_frame8)
            if norm_tp8:
                char_tp8 = render_character_v21(norm_tp8, canvas_size=(panel_w_anim8, panel_h8))
                panel3_v8 = cv2.cvtColor(np.array(char_tp8.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                panel3_v8 = np.ones((panel_h8, panel_w_anim8, 3), dtype=np.uint8) * 240
            cv2.putText(panel1_v8, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel2_v8, 'Skeleton', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel3_v8, '2D Character', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
            combined_v8 = np.hstack([panel1_v8, panel2_v8, panel3_v8])
            out_tp8.write(combined_v8)
            if i % 50 == 0:
                print(f"Processing {i}/{len(stab_list_tp8)} frames")
        cap_tp8.release()
        out_tp8.release()
        print(f"Saved to {out_path_tp8}")

    generate_three_panel_v8('dogvideo.mp4', raw_kps, stab_kps)
    return


@app.cell
def _(get_kp, np, stab_kps):
    all_spine_lengths = []
    for frame in stab_kps:
        s_neck = get_kp(frame, 'Neck')
        s_tail = get_kp(frame, 'root_of_tail')
        all_spine_lengths.append(np.linalg.norm(s_tail - s_neck))

    median_spine = np.median(all_spine_lengths)
    print(f"Spine lengths: min={min(all_spine_lengths):.1f}, median={median_spine:.1f}, max={max(all_spine_lengths):.1f}")

    return (median_spine,)


@app.cell
def _(KEYPOINT_NAMES, get_kp, np):
    def normalise_skeleton_v3(frame_data, median_spine_len, target_spine=200):
        neck = get_kp(frame_data, 'Neck')
        tail = get_kp(frame_data, 'root_of_tail')
        actual_spine = np.linalg.norm(tail - neck)

        if actual_spine < 1e-6:
            return None, None, 1.0

        scale = target_spine / actual_spine
        centre = neck

        normalised = {}
        for name in KEYPOINT_NAMES:
            kp = get_kp(frame_data, name)
            normalised[name] = (kp - centre) * scale

        stretch_ratio = actual_spine / median_spine_len

        return normalised, scale, stretch_ratio

    print("normalise_skeleton_v3 defined")
    return (normalise_skeleton_v3,)


@app.cell
def _(np, stab_kps):
    def smooth_leg_keypoints(stab_list, window=5, conf_threshold=0.5):
        leg_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        filled = []
        for i in range(len(stab_list)):
            frame = {
                'frame': stab_list[i]['frame'],
                'keypoints': [list(kp) for kp in stab_list[i]['keypoints']],
                'scores': list(stab_list[i]['scores'])
            }
            if i > 0:
                for idx in leg_indices:
                    if frame['scores'][idx] < conf_threshold:
                        frame['keypoints'][idx] = list(filled[i-1]['keypoints'][idx])
            filled.append(frame)

        smoothed = []
        half = window // 2
        for i in range(len(filled)):
            frame = {
                'frame': filled[i]['frame'],
                'keypoints': [list(kp) for kp in filled[i]['keypoints']],
                'scores': list(filled[i]['scores'])
            }
            for idx in leg_indices:
                xs, ys = [], []
                for j in range(max(0, i - half), min(len(filled), i + half + 1)):
                    xs.append(filled[j]['keypoints'][idx][0])
                    ys.append(filled[j]['keypoints'][idx][1])
                frame['keypoints'][idx] = [np.mean(xs), np.mean(ys)]
            smoothed.append(frame)
        print(f"Smoothed {len(smoothed)} frames, window={window}, conf fallback<{conf_threshold}")
        return smoothed

    smooth_kps = smooth_leg_keypoints(stab_kps, window=5)
    print("Done")
    return (smooth_kps,)


@app.cell
def _(
    Image,
    ImageDraw,
    median_spine,
    normalise_skeleton_v3,
    np,
    plt,
    smooth_kps,
):
    def render_character_v27(normalised_kps, raw_frame_data=None,
                             stretch_ratio=1.0,
                             canvas_size=(500, 400), offset=(200, 180)):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        r_shift = 6

        if raw_frame_data:
            raw_neck_y = raw_frame_data['keypoints'][3][1]
            bounce = int((raw_neck_y - 280) * 0.3)
            oy = oy + bounce

        def pt_c(name, x_off=0, y_off=0):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox + x_off), int(kp_c[1] + oy + y_off))

        neck = pt_c('Neck')
        tail_pt = pt_c('root_of_tail')
        nose = pt_c('Nose')
        l_shoulder_real = pt_c('L_Shoulder')
        l_hip_real = pt_c('L_Hip')

        spine_len = np.linalg.norm(np.array(tail_pt) - np.array(neck))
        if spine_len < 1e-6:
            return canvas

        spine_dx = tail_pt[0] - neck[0]
        spine_dy = tail_pt[1] - neck[1]
        head_r = int(spine_len * 0.16)
        joint_r = 5
        paw_r = 5

        hx = nose[0] - neck[0]
        hy = nose[1] - neck[1]
        hd = np.linalg.norm([hx, hy])
        if hd > 0:
            hx /= hd
            hy /= hd
        px = -hy
        py = hx

        sr = np.clip(stretch_ratio, 0.75, 1.35)
        base_body_drop = int(spine_len * 0.17 / sr)
        back_rise = int(spine_len * 0.06 / sr)

        sh_pos = 0.12 - (sr - 1.0) * 0.06
        hp_pos = 0.88 + (sr - 1.0) * 0.06

        blend = 0.55
        fixed_shoulder = (
            neck[0] + int(spine_dx * sh_pos),
            neck[1] + int(spine_dy * sh_pos) + base_body_drop
        )
        fixed_hip = (
            neck[0] + int(spine_dx * hp_pos),
            neck[1] + int(spine_dy * hp_pos) + base_body_drop
        )
        body_shoulder = (
            int(fixed_shoulder[0] * (1 - blend) + l_shoulder_real[0] * blend),
            int(fixed_shoulder[1] * (1 - blend) + l_shoulder_real[1] * blend)
        )
        body_hip = (
            int(fixed_hip[0] * (1 - blend) + l_hip_real[0] * blend),
            int(fixed_hip[1] * (1 - blend) + l_hip_real[1] * blend)
        )

        spine_mid_y = neck[1] + spine_dy * 0.5
        avg_limb_y = (l_shoulder_real[1] + l_hip_real[1]) / 2
        spine_curve = avg_limb_y - spine_mid_y
        curve_adjust = int(np.clip(spine_curve * -0.15, -spine_len * 0.04, spine_len * 0.04))

        body_mid = (
            (body_shoulder[0] + body_hip[0]) // 2,
            (body_shoulder[1] + body_hip[1]) // 2 - int(spine_len * 0.02)
        )

        tail_tip_x = tail_pt[0] + int(spine_len * 0.15)
        tail_tip_y = tail_pt[1] + int(spine_len * 0.02)
        tail_mid_x = tail_pt[0] + int(spine_len * 0.08)
        tail_mid_y = tail_pt[1] - int(spine_len * 0.08)
        draw_c.line([tail_pt, (tail_mid_x, tail_mid_y)], fill='#2A2A2A', width=6)
        draw_c.line([(tail_mid_x, tail_mid_y), (tail_tip_x, tail_tip_y)], fill='#2A2A2A', width=4)

        r_elbow = pt_c('R_Elbow', r_shift, 0)
        r_fpaw = pt_c('R_F_Paw', r_shift, 0)
        r_shoulder_draw = (body_shoulder[0] + r_shift, body_shoulder[1])
        draw_c.line([r_shoulder_draw, r_elbow], fill='#555555', width=9)
        draw_c.line([r_elbow, r_fpaw], fill='#555555', width=7)
        draw_c.ellipse([r_shoulder_draw[0]-joint_r, r_shoulder_draw[1]-joint_r,
                        r_shoulder_draw[0]+joint_r, r_shoulder_draw[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_elbow[0]-joint_r, r_elbow[1]-joint_r,
                        r_elbow[0]+joint_r, r_elbow[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_fpaw[0]-paw_r, r_fpaw[1]-paw_r,
                        r_fpaw[0]+paw_r, r_fpaw[1]+paw_r], fill='#4A4A4A')

        r_knee = pt_c('R_Knee', r_shift, 0)
        r_bpaw = pt_c('R_B_Paw', r_shift, 0)
        r_hip_draw = (body_hip[0] + r_shift, body_hip[1])
        draw_c.line([r_hip_draw, r_knee], fill='#555555', width=9)
        draw_c.line([r_knee, r_bpaw], fill='#555555', width=7)
        draw_c.ellipse([r_hip_draw[0]-joint_r, r_hip_draw[1]-joint_r,
                        r_hip_draw[0]+joint_r, r_hip_draw[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_knee[0]-joint_r, r_knee[1]-joint_r,
                        r_knee[0]+joint_r, r_knee[1]+joint_r], fill='#666666')
        draw_c.ellipse([r_bpaw[0]-paw_r, r_bpaw[1]-paw_r,
                        r_bpaw[0]+paw_r, r_bpaw[1]+paw_r], fill='#4A4A4A')

        br1 = back_rise + curve_adjust
        br2 = int(br1 * 1.1)
        br3 = int(br1 * 0.8)
        br4 = int(br1 * 0.2)
        body_pts = [
            (neck[0], neck[1] - br1),
            (neck[0] + int(spine_dx * 0.2), neck[1] + int(spine_dy * 0.2) - br2),
            (neck[0] + int(spine_dx * 0.4), neck[1] + int(spine_dy * 0.4) - int(br2 * 1.05)),
            (neck[0] + int(spine_dx * 0.6), neck[1] + int(spine_dy * 0.6) - br3),
            (neck[0] + int(spine_dx * 0.8), neck[1] + int(spine_dy * 0.8) - br4),
            (tail_pt[0], tail_pt[1] - int(br4 * 0.5)),
            body_hip,
            body_mid,
            body_shoulder,
            (neck[0], neck[1] + int(spine_len * 0.04)),
        ]
        draw_c.polygon(body_pts, fill='#3D3D3D', outline='#222222')

        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        draw_c.line([body_shoulder, l_elbow], fill='#333333', width=12)
        draw_c.line([l_elbow, l_fpaw], fill='#333333', width=9)
        draw_c.ellipse([body_shoulder[0]-joint_r-1, body_shoulder[1]-joint_r-1,
                        body_shoulder[0]+joint_r+1, body_shoulder[1]+joint_r+1], fill='#3D3D3D')
        draw_c.ellipse([l_elbow[0]-joint_r, l_elbow[1]-joint_r,
                        l_elbow[0]+joint_r, l_elbow[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_fpaw[0]-paw_r, l_fpaw[1]-paw_r,
                        l_fpaw[0]+paw_r, l_fpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)

        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')
        draw_c.line([body_hip, l_knee], fill='#333333', width=12)
        draw_c.line([l_knee, l_bpaw], fill='#333333', width=9)
        draw_c.ellipse([body_hip[0]-joint_r-1, body_hip[1]-joint_r-1,
                        body_hip[0]+joint_r+1, body_hip[1]+joint_r+1], fill='#3D3D3D')
        draw_c.ellipse([l_knee[0]-joint_r, l_knee[1]-joint_r,
                        l_knee[0]+joint_r, l_knee[1]+joint_r], fill='#3D3D3D')
        draw_c.ellipse([l_bpaw[0]-paw_r, l_bpaw[1]-paw_r,
                        l_bpaw[0]+paw_r, l_bpaw[1]+paw_r], fill='#2A2A2A', outline='#111111', width=1)

        hw = int(head_r * 1.3)
        hh = int(head_r * 1.0)
        neck_w = int(head_r * 0.6)
        head_polygon = [
            (neck[0] + int(px * neck_w), neck[1] + int(py * neck_w)),
            (neck[0] + int(hx * hw * 0.5) + int(px * hh), neck[1] + int(hy * hw * 0.5) + int(py * hh)),
            (neck[0] + int(hx * hw) + int(px * hh * 0.8), neck[1] + int(hy * hw) + int(py * hh * 0.8)),
            (neck[0] + int(hx * hw * 1.4) + int(px * hh * 0.3), neck[1] + int(hy * hw * 1.4) + int(py * hh * 0.3)),
            (neck[0] + int(hx * hw * 1.5), neck[1] + int(hy * hw * 1.5)),
            (neck[0] + int(hx * hw * 1.4) - int(px * hh * 0.4), neck[1] + int(hy * hw * 1.4) - int(py * hh * 0.4)),
            (neck[0] + int(hx * hw * 0.8) - int(px * hh * 0.6), neck[1] + int(hy * hw * 0.8) - int(py * hh * 0.6)),
            (neck[0] + int(hx * hw * 0.3) - int(px * hh * 0.5), neck[1] + int(hy * hw * 0.3) - int(py * hh * 0.5)),
            (neck[0] - int(px * neck_w), neck[1] - int(py * neck_w)),
        ]
        draw_c.polygon(head_polygon, fill='#4A4A4A', outline='#222222')

        mz_cx = neck[0] + int(hx * hw * 1.35)
        mz_cy = neck[1] + int(hy * hw * 1.35) - int(px * hh * 0.15)
        mz_rx = int(head_r * 0.4)
        mz_ry = int(head_r * 0.28)
        draw_c.ellipse([mz_cx-mz_rx, mz_cy-mz_ry, mz_cx+mz_rx, mz_cy+mz_ry],
                       fill='#5A5A5A', outline='#333333', width=1)
        ns_x = mz_cx + int(hx * mz_rx * 0.5)
        ns_y = mz_cy + int(hy * mz_rx * 0.5)
        draw_c.ellipse([ns_x-4, ns_y-3, ns_x+4, ns_y+3], fill='#1A1A1A')

        ear_bx = neck[0] + int(hx * hw * 0.4) + int(px * hh * 0.9)
        ear_by = neck[1] + int(hy * hw * 0.4) + int(py * hh * 0.9)
        ear_tip_x = ear_bx + int(hx * head_r * 0.5) + int(px * head_r * 0.3)
        ear_tip_y = ear_by + int(hy * head_r * 0.5) + int(py * head_r * 0.3)
        ear_end_x = ear_bx + int(hx * head_r * 0.7) - int(px * head_r * 0.2)
        ear_end_y = ear_by + int(hy * head_r * 0.7) - int(py * head_r * 0.2)
        draw_c.polygon([(ear_bx, ear_by), (ear_tip_x, ear_tip_y), (ear_end_x, ear_end_y)],
                       fill='#363636', outline='#222222')

        ey_cx = neck[0] + int(hx * hw * 0.7) + int(px * hh * 0.4)
        ey_cy = neck[1] + int(hy * hw * 0.7) + int(py * hh * 0.4)
        draw_c.ellipse([ey_cx-5, ey_cy-5, ey_cx+5, ey_cy+5],
                       fill='white', outline='#222222', width=1)
        draw_c.ellipse([ey_cx-2, ey_cy-2, ey_cx+2, ey_cy+2], fill='#1A1A1A')

        draw_c.ellipse([neck[0]-joint_r, neck[1]-joint_r,
                        neck[0]+joint_r, neck[1]+joint_r], fill='#444444')
        draw_c.ellipse([tail_pt[0]-joint_r, tail_pt[1]-joint_r,
                        tail_pt[0]+joint_r, tail_pt[1]+joint_r], fill='#444444')

        return canvas


    test_norm_v3, _, test_sr = normalise_skeleton_v3(smooth_kps[0], median_spine)
    char_v27_test = render_character_v27(test_norm_v3, stretch_ratio=test_sr)
    plt.figure(figsize=(8, 6))
    plt.imshow(char_v27_test)
    plt.axis('off')
    plt.title(f'v27 - Curved Spine, stretch={test_sr:.2f}')
    plt.savefig('character_v27.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Done")
    return (render_character_v27,)


@app.cell
def _(
    cv2,
    median_spine,
    normalise_skeleton_v3,
    np,
    raw_kps,
    render_character_v27,
    smooth_kps,
):
    def generate_three_panel_v13(vid_path, raw_list, smooth_list, median_spine_len,
                                  out_path='three_panel_v13.mp4', fps=30):
        SKEL_IDX = [
            (0,2),(1,2),(2,3),(3,5),(3,8),
            (5,6),(6,7),(8,9),(9,10),
            (3,4),(4,11),(4,14),
            (11,12),(12,13),(14,15),(15,16)
        ]
        cap = cv2.VideoCapture(vid_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        panel_h = 400
        scale = panel_h / orig_h
        panel_w_orig = int(orig_w * scale)
        panel_w_anim = 500
        total_w = panel_w_orig * 2 + panel_w_anim
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (total_w, panel_h))
        for i, smooth_frame in enumerate(smooth_list):
            ret, orig_frame = cap.read()
            if not ret:
                break
            panel1 = cv2.resize(orig_frame, (panel_w_orig, panel_h))
            panel2 = panel1.copy()
            raw_frame = raw_list[i]
            kps_raw = raw_frame['keypoints']
            sc_raw = raw_frame['scores']
            for (p1, p2) in SKEL_IDX:
                if sc_raw[p1] > 0.3 and sc_raw[p2] > 0.3:
                    pt1 = (int(kps_raw[p1][0]*scale), int(kps_raw[p1][1]*scale))
                    pt2 = (int(kps_raw[p2][0]*scale), int(kps_raw[p2][1]*scale))
                    cv2.line(panel2, pt1, pt2, (0,0,255), 2)
            for kp, sc in zip(kps_raw, sc_raw):
                if sc > 0.3:
                    cv2.circle(panel2, (int(kp[0]*scale), int(kp[1]*scale)), 3, (0,255,0), -1)
            norm, _, sr = normalise_skeleton_v3(smooth_frame, median_spine_len)
            if norm:
                char_img = render_character_v27(norm, raw_frame_data=smooth_frame,
                                                stretch_ratio=sr,
                                                canvas_size=(panel_w_anim, panel_h))
                panel3 = cv2.cvtColor(np.array(char_img.convert('RGB')), cv2.COLOR_RGB2BGR)
            else:
                panel3 = np.ones((panel_h, panel_w_anim, 3), dtype=np.uint8) * 240
            cv2.putText(panel1, 'Original', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel2, 'Skeleton', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(panel3, '2D Character', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
            combined = np.hstack([panel1, panel2, panel3])
            out.write(combined)
            if i % 50 == 0:
                print(f"Processing {i}/{len(smooth_list)} frames")
        cap.release()
        out.release()
        print(f"Saved to {out_path}")

    generate_three_panel_v13('dogvideo.mp4', raw_kps, smooth_kps, median_spine)
    return


@app.cell
def _(median_spine, normalise_skeleton_v3, np, smooth_kps):
    import math
    test_n, _, _ = normalise_skeleton_v3(smooth_kps[0], median_spine)

    def angle_debug(p_from, p_to):
        d = np.array(test_n[p_to]) - np.array(test_n[p_from])
        ang = np.degrees(np.arctan2(d[1], d[0]))
        dist = np.linalg.norm(d)
        print(f"{p_from}→{p_to}: angle={ang:.1f}°, dist={dist:.1f}px")

    angle_debug('L_Shoulder', 'L_Elbow')
    angle_debug('L_Elbow', 'L_F_Paw')
    angle_debug('L_Hip', 'L_Knee')
    angle_debug('L_Knee', 'L_B_Paw')
    angle_debug('R_Shoulder', 'R_Elbow')
    angle_debug('R_Hip', 'R_Knee')
    print(f"spine_len in normalised: {np.linalg.norm(np.array(test_n['root_of_tail']) - np.array(test_n['Neck'])):.1f}px")

    return


@app.cell
def _(KEYPOINT_NAMES, get_kp, np):
    def normalise_skeleton_v4(frame_data, median_spine_len, target_spine=200, stretch_amount=0.3):
        neck_kp = get_kp(frame_data, 'Neck')
        tail_kp = get_kp(frame_data, 'root_of_tail')
        actual_spine = np.linalg.norm(tail_kp - neck_kp)

        if actual_spine < 1e-6:
            return None, None, 1.0

        scale = target_spine / actual_spine
        centre = neck_kp

        sr = actual_spine / median_spine_len
        x_stretch = 1.0 + (sr - 1.0) * stretch_amount

        normalised = {}
        for name in KEYPOINT_NAMES:
            kp = get_kp(frame_data, name)
            norm_pos = (kp - centre) * scale
            normalised[name] = np.array([norm_pos[0] * x_stretch, norm_pos[1]])

        return normalised, scale, sr

    print("normalise_skeleton_v4 defined - stretch_amount=0.3")
    return (normalise_skeleton_v4,)


@app.cell
def _(
    Image,
    ImageDraw,
    bone_ratios_selected,
    dog_template,
    median_spine,
    normalise_skeleton_v4,
    np,
    smooth_kps,
):
    def render_character_v28(normalised_kps, raw_frame_data=None,
                             canvas_size=(500, 400), offset=(200, 180),
                             bone_ratios=None):
        canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
        draw_c = ImageDraw.Draw(canvas)
        ox, oy = offset
        r_shift = 6


        bounce = 0
        if raw_frame_data:
            raw_neck_y = raw_frame_data['keypoints'][3][1]
            raw_hip_y_b = (raw_frame_data['keypoints'][11][1] + raw_frame_data['keypoints'][14][1]) / 2
            bounce = int(((raw_neck_y + raw_hip_y_b) / 2 - 280) * 0.35)
            oy = oy + bounce

        def pt_c(name, x_off=0, y_off=0):
            kp_c = normalised_kps[name]
            return (int(kp_c[0] + ox + x_off), int(kp_c[1] + oy + y_off))

        def pt_arr(name):
            kp_c = normalised_kps[name]
            return np.array([kp_c[0] + ox, kp_c[1] + oy])

        neck = pt_arr('Neck')
        tail_pt = pt_arr('root_of_tail')
        nose = pt_arr('Nose')
        l_sh = pt_arr('L_Shoulder'); r_sh = pt_arr('R_Shoulder')
        l_hip = pt_arr('L_Hip'); r_hip = pt_arr('R_Hip')
        avg_sh = (l_sh + r_sh) / 2
        avg_hip = (l_hip + r_hip) / 2

        spine_vec = tail_pt - neck
        spine_len = np.linalg.norm(spine_vec)
        if spine_len < 1e-6:
            return canvas

        ref_spine = 200.0

        if bone_ratios is not None:
            leg_ratio = bone_ratios.get('L_Hip_to_L_Knee', 0.26)
            body_scale = np.clip(leg_ratio / 0.26, 0.7, 1.4)
        else:
            body_scale = 1.0

        head_r = int(ref_spine * 0.16)
        sh_r = int(ref_spine * 0.045)
        hip_r = int(ref_spine * 0.04)
        elbow_r = int(ref_spine * 0.03)
        knee_r = int(ref_spine * 0.03)
        paw_r = int(ref_spine * 0.025)
        neck_r = int(ref_spine * 0.025)
        tail_r = int(ref_spine * 0.02)

        hx = nose[0] - neck[0]; hy = nose[1] - neck[1]
        hd = np.linalg.norm([hx, hy])
        if hd > 0: hx /= hd; hy /= hd
        px_h = -hy; py_h = hx

        blend_sh = 0.5
        blend_hp = 0.7
        body_drop = ref_spine * 0.12
        fixed_sh = neck + spine_vec * 0.15 + np.array([0, body_drop])
        fixed_hp = neck + spine_vec * 0.85 + np.array([0, body_drop])
        body_shoulder = tuple((fixed_sh * (1-blend_sh) + avg_sh * blend_sh).astype(int))
        body_hip_pt = tuple((fixed_hp * (1-blend_hp) + avg_hip * blend_hp).astype(int))

        def draw_limb(p1, p2, w_top, w_bot, fill_color):
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            length = np.linalg.norm([dx, dy])
            if length < 1: return
            px_l = -dy / length; py_l = dx / length
            poly = [
                (int(p1[0]+px_l*w_top), int(p1[1]+py_l*w_top)),
                (int(p1[0]-px_l*w_top), int(p1[1]-py_l*w_top)),
                (int(p2[0]-px_l*w_bot), int(p2[1]-py_l*w_bot)),
                (int(p2[0]+px_l*w_bot), int(p2[1]+py_l*w_bot)),
            ]
            draw_c.polygon(poly, fill=fill_color)

        upper_w = int(ref_spine * 0.05)
        lower_w = int(ref_spine * 0.032)
        shin_w = int(ref_spine * 0.025)
        paw_w = int(ref_spine * 0.018)

        tail_i = tuple(tail_pt.astype(int))
        tail_base_dx = spine_vec[0] / spine_len
        tail_base_dy = spine_vec[1] / spine_len
        tail_up_x = -tail_base_dy; tail_up_y = tail_base_dx
        tail_angle_1 = 0.0; tail_angle_2 = 0.0
        if raw_frame_data:
            kps = raw_frame_data['keypoints']
            raw_hip_y = (kps[11][1] + kps[14][1]) / 2
            raw_knee_y = (kps[12][1] + kps[15][1]) / 2
            raw_neck_y_t = kps[3][1]
            raw_tail_y = kps[4][1]
            tail_angle_1 = np.clip((raw_hip_y - (raw_neck_y_t + raw_tail_y) / 2) * 0.04, -1.0, 1.0)
            tail_angle_2 = np.clip((raw_knee_y - raw_hip_y) * 0.025 + bounce * 0.04, -0.8, 0.8)

        seg1_len = ref_spine * 0.16
        tail_mid_x = int(tail_pt[0] + (tail_base_dx*0.4 + tail_up_x*(0.6+tail_angle_1)) * seg1_len)
        tail_mid_y = int(tail_pt[1] + (tail_base_dy*0.4 + tail_up_y*(0.6+tail_angle_1)) * seg1_len)
        seg2_dx = tail_mid_x - tail_pt[0]; seg2_dy = tail_mid_y - tail_pt[1]
        seg2_len_v = np.linalg.norm([seg2_dx, seg2_dy])
        if seg2_len_v > 0:
            seg2_ux = seg2_dx/seg2_len_v; seg2_uy = seg2_dy/seg2_len_v
        else:
            seg2_ux = tail_base_dx; seg2_uy = tail_base_dy
        seg2_px = -seg2_uy; seg2_py = seg2_ux
        seg2_len = ref_spine * 0.13
        tail_tip_x = int(tail_mid_x + (seg2_ux*0.7 + seg2_px*tail_angle_2) * seg2_len)
        tail_tip_y = int(tail_mid_y + (seg2_uy*0.7 + seg2_py*tail_angle_2) * seg2_len)
        draw_c.line([tail_i, (tail_mid_x, tail_mid_y)], fill='#2A2A2A', width=6)
        draw_c.line([(tail_mid_x, tail_mid_y), (tail_tip_x, tail_tip_y)], fill='#2A2A2A', width=4)
        tail_jr = int(ref_spine * 0.012)
        draw_c.ellipse([tail_mid_x-tail_jr, tail_mid_y-tail_jr,
                        tail_mid_x+tail_jr, tail_mid_y+tail_jr],
                       fill='#4A4A4A', outline='#333333', width=1)

        r_elbow = pt_c('R_Elbow', r_shift, 0)
        r_fpaw = pt_c('R_F_Paw', r_shift, 0)
        r_knee = pt_c('R_Knee', r_shift, 0)
        r_bpaw = pt_c('R_B_Paw', r_shift, 0)
        r_sh_draw = (body_shoulder[0] + r_shift, body_shoulder[1])
        r_hip_draw = (body_hip_pt[0] + r_shift, body_hip_pt[1])
        l_elbow = pt_c('L_Elbow')
        l_fpaw = pt_c('L_F_Paw')
        l_knee = pt_c('L_Knee')
        l_bpaw = pt_c('L_B_Paw')

        draw_limb(r_sh_draw, r_elbow, upper_w-2, lower_w-1, '#666666')
        draw_limb(r_elbow, r_fpaw, shin_w-1, paw_w, '#5A5A5A')
        draw_limb(r_hip_draw, r_knee, upper_w-1, lower_w-1, '#666666')
        draw_limb(r_knee, r_bpaw, shin_w-1, paw_w, '#5A5A5A')
        for pt_j, r_j, col in [
            (r_sh_draw, sh_r, '#707070'), (r_elbow, elbow_r, '#666666'),
            (r_fpaw, paw_r, '#555555'), (r_hip_draw, hip_r, '#707070'),
            (r_knee, knee_r, '#666666'), (r_bpaw, paw_r, '#555555'),
        ]:
            draw_c.ellipse([pt_j[0]-r_j, pt_j[1]-r_j, pt_j[0]+r_j, pt_j[1]+r_j],
                           fill=col, outline='#444444', width=1)


        sb = 0.25
        p0 = neck
        p1 = neck + spine_vec * 0.25 + (avg_sh - (neck + spine_vec * 0.25)) * sb
        p2_straight = neck + spine_vec * 0.50
        p2 = p2_straight + (((avg_sh + avg_hip) / 2) - p2_straight) * sb
        p3 = neck + spine_vec * 0.75 + (avg_hip - (neck + spine_vec * 0.75)) * sb
        p4 = tail_pt
        spine_pts = [p0, p1, p2, p3, p4]

        base_w_b = ref_spine * 0.12 * body_scale
        widths_top = [base_w_b*0.3, base_w_b*0.9, base_w_b*0.65, base_w_b*0.8, base_w_b*0.05]
        widths_bot = [base_w_b*0.4, base_w_b*1.5, base_w_b*1.2, base_w_b*1.25, base_w_b*0.05]
        top_edge_b = []; bot_edge_b = []
        for i, sp in enumerate(spine_pts):
            if i == 0: tang = spine_pts[1] - spine_pts[0]
            elif i == len(spine_pts)-1: tang = spine_pts[-1] - spine_pts[-2]
            else: tang = spine_pts[i+1] - spine_pts[i-1]
            tl = np.linalg.norm(tang)
            if tl > 0: tang = tang / tl
            perp = np.array([-tang[1], tang[0]])
            top_edge_b.append(sp - perp * widths_top[i])
            bot_edge_b.append(sp + perp * widths_bot[i])
        from scipy.interpolate import CubicSpline
        ts_b = np.array([0, 0.25, 0.5, 0.75, 1.0])
        t_fine_b = np.linspace(0, 1, 40)
        tx = CubicSpline(ts_b, [p[0] for p in top_edge_b])(t_fine_b)
        ty = CubicSpline(ts_b, [p[1] for p in top_edge_b])(t_fine_b)
        bx = CubicSpline(ts_b, [p[0] for p in bot_edge_b])(t_fine_b)
        by = CubicSpline(ts_b, [p[1] for p in bot_edge_b])(t_fine_b)
        body_poly = ([(int(tx[i]), int(ty[i])) for i in range(len(t_fine_b))] +
                     [(int(bx[i]), int(by[i])) for i in range(len(t_fine_b)-1, -1, -1)])
        draw_c.polygon(body_poly, fill='#3D3D3D', outline='#222222')

        draw_limb(body_shoulder, l_elbow, upper_w, lower_w, '#444444')
        draw_limb(l_elbow, l_fpaw, shin_w, paw_w, '#383838')
        draw_limb(body_hip_pt, l_knee, upper_w+1, lower_w, '#444444')
        draw_limb(l_knee, l_bpaw, shin_w, paw_w, '#383838')
        for pt_j, r_j, col, ow in [
            (body_shoulder, sh_r, '#505050', 2), (l_elbow, elbow_r, '#4A4A4A', 2),
            (l_fpaw, paw_r, '#2A2A2A', 1), (body_hip_pt, hip_r, '#505050', 2),
            (l_knee, knee_r, '#4A4A4A', 2), (l_bpaw, paw_r, '#2A2A2A', 1),
        ]:
            draw_c.ellipse([pt_j[0]-r_j, pt_j[1]-r_j, pt_j[0]+r_j, pt_j[1]+r_j],
                           fill=col, outline='#222222', width=ow)

        draw_c.ellipse([int(neck[0])-neck_r, int(neck[1])-neck_r,
                        int(neck[0])+neck_r, int(neck[1])+neck_r],
                       fill='#4A4A4A', outline='#222222', width=2)
        draw_c.ellipse([int(tail_pt[0])-tail_r, int(tail_pt[1])-tail_r,
                        int(tail_pt[0])+tail_r, int(tail_pt[1])+tail_r],
                       fill='#4A4A4A', outline='#222222', width=1)

        hw = int(head_r * 1.3); hh = int(head_r * 1.0)
        neck_thick = int(head_r * 0.65)
        head_attach = neck + np.array([hx*hw*0.25, hy*hw*0.25])
        neck_polygon = [
            (int(neck[0]-px_h*neck_thick*0.8), int(neck[1]-py_h*neck_thick*0.8)),
            (int(neck[0]+px_h*neck_thick), int(neck[1]+py_h*neck_thick)),
            (int(head_attach[0]+px_h*neck_thick*1.1), int(head_attach[1]+py_h*neck_thick*1.1)),
            (int(head_attach[0]-px_h*neck_thick*0.7), int(head_attach[1]-py_h*neck_thick*0.7)),
        ]
        draw_c.polygon(neck_polygon, fill='#424242')
        draw_c.ellipse([int(neck[0])-neck_r, int(neck[1])-neck_r,
                        int(neck[0])+neck_r, int(neck[1])+neck_r],
                       fill='#4A4A4A', outline='#222222', width=2)

        head_polygon = [
            (int(neck[0]+px_h*neck_thick*0.9), int(neck[1]+py_h*neck_thick*0.9)),
            (int(neck[0]+hx*hw*0.4+px_h*hh*0.75), int(neck[1]+hy*hw*0.4+py_h*hh*0.75)),
            (int(neck[0]+hx*hw*0.8+px_h*hh*0.75), int(neck[1]+hy*hw*0.8+py_h*hh*0.75)),
            (int(neck[0]+hx*hw*1.2+px_h*hh*0.5), int(neck[1]+hy*hw*1.2+py_h*hh*0.5)),
            (int(neck[0]+hx*hw*1.5), int(neck[1]+hy*hw*1.5)),
            (int(neck[0]+hx*hw*1.4-px_h*hh*0.4), int(neck[1]+hy*hw*1.4-py_h*hh*0.4)),
            (int(neck[0]+hx*hw*0.8-px_h*hh*0.6), int(neck[1]+hy*hw*0.8-py_h*hh*0.6)),
            (int(neck[0]+hx*hw*0.3-px_h*hh*0.45), int(neck[1]+hy*hw*0.3-py_h*hh*0.45)),
            (int(neck[0]-px_h*neck_thick*0.6), int(neck[1]-py_h*neck_thick*0.6)),
        ]
        draw_c.polygon(head_polygon, fill='#4A4A4A', outline='#222222')

        mz_cx = int(neck[0]+hx*hw*1.35); mz_cy = int(neck[1]+hy*hw*1.35-px_h*hh*0.15)
        mz_rx = int(head_r*0.4); mz_ry = int(head_r*0.28)
        draw_c.ellipse([mz_cx-mz_rx, mz_cy-mz_ry, mz_cx+mz_rx, mz_cy+mz_ry],
                       fill='#5A5A5A', outline='#333333', width=1)
        ns_x = int(mz_cx+hx*mz_rx*0.5); ns_y = int(mz_cy+hy*mz_rx*0.5)
        draw_c.ellipse([ns_x-4, ns_y-3, ns_x+4, ns_y+3], fill='#1A1A1A')

        ear_flop = bounce * 0.12
        ear_base = neck + np.array([hx*hw*0.45+px_h*hh*0.72, hy*hw*0.45+py_h*hh*0.72])
        ear_len = int(head_r*1.1); ear_w = int(head_r*0.55)
        droop = np.array([-px_h*0.3, -py_h*0.3+1.0])
        droop = droop / (np.linalg.norm(droop)+1e-6)
        ear_mid = ear_base + droop*ear_len*0.45 + np.array([0, ear_flop*0.4])
        ear_tip = ear_mid + droop*ear_len*0.55 + np.array([hx*ear_w*0.05, ear_flop*0.6])
        upper_ear = [
            (int(ear_base[0]+hx*ear_w*0.2), int(ear_base[1]+hy*ear_w*0.2)),
            (int(ear_mid[0]+hx*ear_w*0.25), int(ear_mid[1]+hy*ear_w*0.25)),
            (int(ear_mid[0]-hx*ear_w*0.2), int(ear_mid[1]-hy*ear_w*0.2)),
            (int(ear_base[0]-hx*ear_w*0.1), int(ear_base[1]-hy*ear_w*0.1)),
        ]
        draw_c.polygon(upper_ear, fill='#363636', outline='#222222')
        lower_ear = [
            (int(ear_mid[0]+hx*ear_w*0.25), int(ear_mid[1]+hy*ear_w*0.25)),
            (int(ear_tip[0]+hx*ear_w*0.15), int(ear_tip[1]+hy*ear_w*0.15)),
            (int(ear_tip[0]), int(ear_tip[1])),
            (int(ear_tip[0]-hx*ear_w*0.15), int(ear_tip[1]-hy*ear_w*0.15)),
            (int(ear_mid[0]-hx*ear_w*0.2), int(ear_mid[1]-hy*ear_w*0.2)),
        ]
        draw_c.polygon(lower_ear, fill='#2E2E2E', outline='#1A1A1A')
        ear_jr = int(ref_spine*0.015)
        for ep in [ear_base, ear_mid]:
            draw_c.ellipse([int(ep[0])-ear_jr, int(ep[1])-ear_jr,
                            int(ep[0])+ear_jr, int(ep[1])+ear_jr],
                           fill='#4A4A4A', outline='#222222', width=1)

        ey_cx = int(neck[0]+hx*hw*0.85+px_h*hh*0.25)
        ey_cy = int(neck[1]+hy*hw*0.85+py_h*hh*0.25)
        draw_c.ellipse([ey_cx-5, ey_cy-5, ey_cx+5, ey_cy+5],
                       fill='white', outline='#222222', width=1)
        draw_c.ellipse([ey_cx-2, ey_cy-2, ey_cx+2, ey_cy+2], fill='#1A1A1A')

        return canvas


    def test_v28_breed():
        norm, _, _ = normalise_skeleton_v4(smooth_kps[0], median_spine)
        img = render_character_v28(norm, bone_ratios=bone_ratios_selected)
        import matplotlib.pyplot as plt_br
        body_sc = float(np.clip(bone_ratios_selected.get('L_Hip_to_L_Knee', 0.26)/0.26, 0.7, 1.4))
        plt_br.figure(figsize=(8, 6))
        plt_br.imshow(img)
        plt_br.axis('off')
        plt_br.title(f'v28 breed - {dog_template} (body_scale={body_sc:.2f})')
        plt_br.savefig('character_v28_breed.png', dpi=150, bbox_inches='tight')
        plt_br.show()
        print(f"Done - {dog_template}, body_scale={body_sc:.2f}")

    test_v28_breed()
    return (render_character_v28,)


@app.cell
def _(
    bone_ratios_selected,
    cv2,
    median_spine,
    normalise_skeleton_v4,
    np,
    raw_kps,
    render_character_v28,
    smooth_kps,
):
    def run_v16_video():
        def gen_v16(vid_path, raw_list, smooth_list, median_spine_len,
                    out_path='three_panel_v16.mp4', fps=30):
            SKEL_IDX = [
                (0,2),(1,2),(2,3),(3,5),(3,8),
                (5,6),(6,7),(8,9),(9,10),
                (3,4),(4,11),(4,14),
                (11,12),(12,13),(14,15),(15,16)
            ]
            cap = cv2.VideoCapture(vid_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            panel_h = 400
            scale = panel_h / orig_h
            panel_w_orig = int(orig_w * scale)
            panel_w_anim = 500
            total_w = panel_w_orig * 2 + panel_w_anim
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (total_w, panel_h))
            for i, smooth_frame in enumerate(smooth_list):
                ret, orig_frame = cap.read()
                if not ret:
                    break
                panel1 = cv2.resize(orig_frame, (panel_w_orig, panel_h))
                panel2 = panel1.copy()
                raw_frame = raw_list[i]
                kps_raw = raw_frame['keypoints']
                sc_raw = raw_frame['scores']
                for (p1, p2) in SKEL_IDX:
                    if sc_raw[p1] > 0.3 and sc_raw[p2] > 0.3:
                        pt1 = (int(kps_raw[p1][0]*scale), int(kps_raw[p1][1]*scale))
                        pt2 = (int(kps_raw[p2][0]*scale), int(kps_raw[p2][1]*scale))
                        cv2.line(panel2, pt1, pt2, (0,0,255), 2)
                for kp, sc in zip(kps_raw, sc_raw):
                    if sc > 0.3:
                        cv2.circle(panel2, (int(kp[0]*scale), int(kp[1]*scale)), 3, (0,255,0), -1)
                norm, _, _ = normalise_skeleton_v4(smooth_frame, median_spine_len)
                if norm:
                    char_img = render_character_v28(norm, raw_frame_data=smooth_frame,
                                                    canvas_size=(panel_w_anim, panel_h),
                                                    bone_ratios=bone_ratios_selected)
                    panel3 = cv2.cvtColor(np.array(char_img.convert('RGB')), cv2.COLOR_RGB2BGR)
                else:
                    panel3 = np.ones((panel_h, panel_w_anim, 3), dtype=np.uint8) * 240
                cv2.putText(panel1, 'Original', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(panel2, 'Skeleton', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(panel3, '2D Character', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
                combined = np.hstack([panel1, panel2, panel3])
                out.write(combined)
                if i % 50 == 0:
                    print(f"Processing {i}/{len(smooth_list)} frames")
            cap.release()
            out.release()
            print(f"Saved to {out_path}")

        gen_v16('dogvideo.mp4', raw_kps, smooth_kps, median_spine)

    run_v16_video()
    return


@app.cell
def _():
    return


@app.cell
def _(get_kp, np, stab_kps):
    def select_dog_template(stab_kps_list, n_frames=10):
        import json as json_tmpl

        leg_ratios = []
        for frame in stab_kps_list[:n_frames]:
            neck = get_kp(frame, 'Neck')
            tail = get_kp(frame, 'root_of_tail')
            l_hip = get_kp(frame, 'L_Hip')
            l_knee = get_kp(frame, 'L_Knee')

            spine_len = np.linalg.norm(tail - neck)
            leg_len = np.linalg.norm(l_knee - l_hip)

            if spine_len > 1e-6:
                leg_ratios.append(leg_len / spine_len)

        avg_ratio = np.mean(leg_ratios) if leg_ratios else 0.0
        print(f"Average leg/spine ratio: {avg_ratio:.3f}")

        if avg_ratio > 0.28:
            template = 'large'
            ratios_file = 'bone_ratios_large.json'
        else:
            template = 'small'
            ratios_file = 'bone_ratios_small.json'

        with open(ratios_file, 'r') as ft:
            bone_ratios_loaded = json_tmpl.load(ft)

        print(f"Selected template: {template} dog ({ratios_file})")
        return template, bone_ratios_loaded


    dog_template, bone_ratios_selected = select_dog_template(stab_kps)
    print(f"\nBone ratios loaded for {dog_template} dog:")
    for k, v in bone_ratios_selected.items():
        print(f"  {k}: {v:.2f}")
    return bone_ratios_selected, dog_template


@app.cell
def _(bone_ratios_selected):
    def build_character_parts(bone_ratios=None):
        ref = 200.0

        if bone_ratios is not None:
            upper_leg_f = bone_ratios.get('L_Shoulder_to_L_Elbow', 0.31) * ref
            lower_leg_f = bone_ratios.get('L_Elbow_to_L_F_Paw', 0.15) * ref
            upper_leg_h = bone_ratios.get('L_Hip_to_L_Knee', 0.16) * ref
            lower_leg_h = bone_ratios.get('L_Knee_to_L_B_Paw', 0.17) * ref
            head_scale = ref * 0.20
        else:
            upper_leg_f = 0.31 * ref
            lower_leg_f = 0.15 * ref
            upper_leg_h = 0.16 * ref
            lower_leg_h = 0.17 * ref
            head_scale = ref * 0.20


        upper_leg_f = max(upper_leg_f, ref * 0.26)
        lower_leg_f = max(lower_leg_f, ref * 0.24)
        upper_leg_h = max(upper_leg_h, ref * 0.28)
        lower_leg_h = max(lower_leg_h, ref * 0.24)

        def limb_shape(length, w_top, w_bot):
            return [
                (0, -w_top), (length, -w_bot),
                (length, w_bot), (0, w_top),
            ]

        parts = {}

        chest_len = ref * 0.58
        parts['chest'] = {
            'polygon': [
                (0, -ref*0.055),
                (chest_len*0.35, -ref*0.115),
                (chest_len*0.75, -ref*0.105),
                (chest_len, -ref*0.09),
                (chest_len, ref*0.13),
                (chest_len*0.7, ref*0.175),
                (chest_len*0.3, ref*0.185),
                (0, ref*0.06),
            ],
            'color': '#3D3D3D', 'outline': '#222222',
        }

        hip_len = ref * 0.55
        parts['hindq'] = {
            'polygon': [
                (0, -ref*0.10),
                (hip_len*0.4, -ref*0.105),
                (hip_len*0.8, -ref*0.08),
                (hip_len, -ref*0.02),
                (hip_len, ref*0.02),
                (hip_len*0.75, ref*0.145),
                (hip_len*0.3, ref*0.17),
                (0, ref*0.16),
            ],
            'color': '#3D3D3D', 'outline': '#222222',
        }

        parts['upper_leg_front'] = {
            'polygon': limb_shape(upper_leg_f, ref*0.05, ref*0.032),
            'color': '#444444', 'outline': None,
            'length': upper_leg_f,
        }
        parts['lower_leg_front'] = {
            'polygon': limb_shape(lower_leg_f, ref*0.028, ref*0.018),
            'color': '#383838', 'outline': None,
            'length': lower_leg_f,
        }
        parts['upper_leg_hind'] = {
            'polygon': limb_shape(upper_leg_h, ref*0.055, ref*0.032),
            'color': '#444444', 'outline': None,
            'length': upper_leg_h,
        }
        parts['lower_leg_hind'] = {
            'polygon': limb_shape(lower_leg_h, ref*0.028, ref*0.018),
            'color': '#383838', 'outline': None,
            'length': lower_leg_h,
        }

        for src, dst in [('upper_leg_front', 'upper_leg_front_bg'),
                         ('lower_leg_front', 'lower_leg_front_bg'),
                         ('upper_leg_hind', 'upper_leg_hind_bg'),
                         ('lower_leg_hind', 'lower_leg_hind_bg')]:
            parts[dst] = dict(parts[src])
            parts[dst]['color'] = '#666666' if 'upper' in src else '#5A5A5A'

        tail_seg1 = ref * 0.16
        tail_seg2 = ref * 0.13
        parts['tail_seg1'] = {
            'polygon': limb_shape(tail_seg1, ref*0.016, ref*0.012),
            'color': '#2A2A2A', 'outline': None,
            'length': tail_seg1,
        }
        parts['tail_seg2'] = {
            'polygon': limb_shape(tail_seg2, ref*0.012, ref*0.007),
            'color': '#2A2A2A', 'outline': None,
            'length': tail_seg2,
        }

        hs = head_scale
        parts['head'] = {
            'polygon': [
                (0, -hs*0.55),
                (hs*0.35, -hs*0.72),
                (hs*0.75, -hs*0.70),
                (hs*1.1, -hs*0.45),
                (hs*1.35, -hs*0.05),
                (hs*1.28, hs*0.30),
                (hs*0.75, hs*0.52),
                (hs*0.3, hs*0.45),
                (0, hs*0.42),
            ],
            'color': '#4A4A4A', 'outline': '#222222',
            'scale': hs,
        }

        parts['neck_conn'] = {
            'polygon': [
                (0, -ref*0.09),
                (ref*0.12, -ref*0.105),
                (ref*0.12, ref*0.075),
                (0, ref*0.065),
            ],
            'color': '#424242', 'outline': None,
        }

        ear_l1 = hs * 0.5
        ear_l2 = hs * 0.55
        parts['ear_seg1'] = {
            'polygon': limb_shape(ear_l1, hs*0.16, hs*0.19),
            'color': '#363636', 'outline': '#222222',
            'length': ear_l1,
        }
        parts['ear_seg2'] = {
            'polygon': limb_shape(ear_l2, hs*0.19, hs*0.10),
            'color': '#2E2E2E', 'outline': '#1A1A1A',
            'length': ear_l2,
        }

        return parts


    character_parts = build_character_parts(bone_ratios=bone_ratios_selected)
    print(f"Character parts built: {len(character_parts)} parts")
    for name in character_parts:
        print(f"  {name}")
    return


@app.cell
def _(Image, ImageDraw, np, plt, stab_kps):
    def run_rigged_render_v4():
        def select_dog_template(avg_ratio):
            if avg_ratio > 0.22:
                template = 'large'
                ratios_file = 'bone_ratios_large.json'
            elif avg_ratio > 0.16:
                template = 'medium'
                ratios_file = 'bone_ratios_medium.json'
            else:
                template = 'small'
                ratios_file = 'bone_ratios_small.json'
            return template, ratios_file

        def limb_shape(length, w_start, w_end):
            return [
                (0, -w_start / 2),
                (length, -w_end / 2),
                (length, w_end / 2),
                (0, w_start / 2)
            ]


        def build_character_parts(template_ratios, ref=200.0):
            parts = {}

            upper_leg_f = template_ratios.get('upper_leg_front', ref * 0.51)
            lower_leg_f = template_ratios.get('lower_leg_front', ref * 0.30)
            upper_leg_h = template_ratios.get('upper_leg_hind', ref * 0.51)
            lower_leg_h = template_ratios.get('lower_leg_hind', ref * 0.30)

            chest_len = ref * 0.58 * 0.68

            parts['chest'] = {
                'polygon': [
                    (0, -ref * 0.12),
                    (chest_len * 0.4, -ref * 0.12),
                    (chest_len, -ref * 0.08),
                    (chest_len, ref * 0.17),
                    (chest_len * 0.7, ref * 0.225),
                    (chest_len * 0.3, ref * 0.235),
                    (0, ref * 0.12)
                ],
                'color': '#3D3D3D'
            }

            hindq_len = ref * 0.55 * 0.68
            parts['hindq'] = {
                'polygon': [
                    (0, -ref * 0.11),
                    (hindq_len, -ref * 0.09),
                    (hindq_len, ref * 0.11),
                    (0, ref * 0.14)
                ],
                'color': '#363636'
            }

            parts['upper_leg_front'] = {'polygon': limb_shape(upper_leg_f, ref * 0.042, ref * 0.026), 'color': '#4A4A4A'}
            parts['lower_leg_front'] = {'polygon': limb_shape(lower_leg_f, ref * 0.026, ref * 0.020), 'color': '#333333'}
            parts['upper_leg_front_bg'] = {'polygon': limb_shape(upper_leg_f, ref * 0.042, ref * 0.026), 'color': '#3A3A3A'}
            parts['lower_leg_front_bg'] = {'polygon': limb_shape(lower_leg_f, ref * 0.026, ref * 0.020), 'color': '#2A2A2A'}

            parts['upper_leg_hind'] = {'polygon': limb_shape(upper_leg_h, ref * 0.046, ref * 0.026), 'color': '#4A4A4A'}
            parts['lower_leg_hind'] = {'polygon': limb_shape(lower_leg_h, ref * 0.026, ref * 0.020), 'color': '#333333'}
            parts['upper_leg_hind_bg'] = {'polygon': limb_shape(upper_leg_h, ref * 0.046, ref * 0.026), 'color': '#3A3A3A'}
            parts['lower_leg_hind_bg'] = {'polygon': limb_shape(lower_leg_h, ref * 0.026, ref * 0.020), 'color': '#2A2A2A'}

            parts['upper_leg_front']['length'] = upper_leg_f
            parts['lower_leg_front']['length'] = lower_leg_f
            parts['upper_leg_hind']['length'] = upper_leg_h
            parts['lower_leg_hind']['length'] = lower_leg_h
            parts['upper_leg_front_bg']['length'] = upper_leg_f
            parts['lower_leg_front_bg']['length'] = lower_leg_f
            parts['upper_leg_hind_bg']['length'] = upper_leg_h
            parts['lower_leg_hind_bg']['length'] = lower_leg_h

            parts['tail_seg1'] = {'polygon': limb_shape(ref * 0.18, ref * 0.025, ref * 0.018), 'length': ref * 0.18, 'color': '#2C2C2C'}
            parts['tail_seg2'] = {'polygon': limb_shape(ref * 0.16, ref * 0.018, ref * 0.010), 'length': ref * 0.16, 'color': '#222222'}
            parts['head'] = {'polygon': [(-10, -15), (25, -10), (35, 10), (10, 20), (-15, 10)], 'color': '#424242', 'scale': ref * 0.20}
            parts['neck_conn'] = {'polygon': [(0, -10), (20, -5), (15, 15), (0, 10)], 'color': '#3D3D3D'}
            parts['ear_seg1'] = {'polygon': limb_shape(ref * 0.08, ref * 0.020, ref * 0.016), 'length': ref * 0.08, 'color': '#2A2A2A'}
            parts['ear_seg2'] = {'polygon': limb_shape(ref * 0.07, ref * 0.016, ref * 0.010), 'length': ref * 0.07, 'color': '#202020'}

            return parts


        def render_character_rigged(normalised_kps, parts, raw_frame_data=None, canvas_size=(500, 400), offset=(200, 180)):
            canvas = Image.new('RGBA', canvas_size, (240, 240, 240, 255))
            draw_c = ImageDraw.Draw(canvas)
            ox, oy = offset
            ref = 200.0
            r_shift = 6

            bounce = 0
            if raw_frame_data:
                kps_raw = raw_frame_data['keypoints']
                raw_neck_y = kps_raw[3][1]
                raw_hip_y_b = (kps_raw[11][1] + kps_raw[14][1]) / 2
                bounce = int(((raw_neck_y + raw_hip_y_b) / 2 - 280) * 0.35)
                oy = oy + bounce

            def pt(name):
                kp = normalised_kps[name]
                return np.array([kp[0] + ox, kp[1] + oy])

            def angle_of(p_from, p_to):
                d = pt(p_to) - pt(p_from)
                return np.arctan2(d[1], d[0])

            def transform(polygon, angle, origin):
                c, s = np.cos(angle), np.sin(angle)
                out = []
                for (x, y) in polygon:
                    rx = x * c - y * s + origin[0]
                    ry = x * s + y * c + origin[1]
                    out.append((int(rx), int(ry)))
                return out

            def draw_part(part, angle, origin):
                poly = transform(part['polygon'], angle, origin)
                outline = part.get('outline')
                if outline:
                    draw_c.polygon(poly, fill=part['color'], outline=outline)
                else:
                    draw_c.polygon(poly, fill=part['color'])

            def end_of(part, angle, origin):
                L = part['length']
                return origin + np.array([np.cos(angle), np.sin(angle)]) * L

            def joint(pos, r, fill, outline='#222222', ow=2):
                draw_c.ellipse([int(pos[0])-r, int(pos[1])-r,
                                int(pos[0])+r, int(pos[1])+r],
                               fill=fill, outline=outline, width=ow)

            neck = pt('Neck')
            tail_root = pt('root_of_tail')
            nose = pt('Nose')
            avg_sh = (pt('L_Shoulder') + pt('R_Shoulder')) / 2
            avg_hip = (pt('L_Hip') + pt('R_Hip')) / 2
            spine_dir = tail_root - neck
            spine_len_now = np.linalg.norm(spine_dir)
            if spine_len_now < 1e-6:
                return canvas

            chest_end_t = 0.68
            hindq_start_t = 0.32
            hindq_start = neck + spine_dir * hindq_start_t

            chest_target = neck + spine_dir * chest_end_t
            sh_pull = (avg_sh[1] - chest_target[1]) * 0.2
            chest_target_adj = chest_target + np.array([0, sh_pull])
            ang_chest = np.arctan2(chest_target_adj[1] - neck[1], chest_target_adj[0] - neck[0])

            hip_pull = (avg_hip[1] - (hindq_start[1] + tail_root[1]) / 2) * 0.2
            tail_adj = tail_root + np.array([0, hip_pull * 0.4])
            ang_hindq = np.arctan2(tail_adj[1] - hindq_start[1], tail_adj[0] - hindq_start[0])

            sh_anchor = neck + np.array([np.cos(ang_chest), np.sin(ang_chest)]) * (ref * 0.58 * 0.45) \
                        + np.array([-np.sin(ang_chest), np.cos(ang_chest)]) * (ref * 0.14)
            hp_anchor = hindq_start + np.array([np.cos(ang_hindq), np.sin(ang_hindq)]) * (ref * 0.55 * 0.40) \
                        + np.array([-np.sin(ang_hindq), np.cos(ang_hindq)]) * (ref * 0.13)

            ang_ulf = angle_of('L_Shoulder', 'L_Elbow')
            ang_llf = angle_of('L_Elbow', 'L_F_Paw')
            ang_ulh = angle_of('L_Hip', 'L_Knee')
            ang_ulf_r = angle_of('R_Shoulder', 'R_Elbow')
            ang_llf_r = angle_of('R_Elbow', 'R_F_Paw')
            ang_ulh_r = angle_of('R_Hip', 'R_Knee')

            def clamp_shin(ang, ang_thigh):
                rel = np.arctan2(np.sin(ang - ang_thigh), np.cos(ang - ang_thigh))
                rel = np.clip(rel, -1.75, 1.75)
                return ang_thigh + rel

            ang_llh = clamp_shin(angle_of('L_Knee', 'L_B_Paw'), ang_ulh)
            ang_llh_r = clamp_shin(angle_of('R_Knee', 'R_B_Paw'), ang_ulh_r)
            ang_head = np.arctan2(nose[1] - neck[1], nose[0] - neck[0])

            tail_wag = 0.0
            if raw_frame_data:
                kps_t = raw_frame_data['keypoints']
                raw_hip_y_t = (kps_t[11][1] + kps_t[14][1]) / 2
                raw_mid_y = (kps_t[3][1] + kps_t[4][1]) / 2
                tail_wag = np.clip((raw_hip_y_t - raw_mid_y) * 0.012, -0.5, 0.5)
            tail_a1 = ang_hindq - 0.3 + tail_wag
            tail_a2_rel = 0.35 + bounce * 0.03

            ear_a1 = 1.45 + bounce * 0.010
            ear_a2 = ear_a1 + 0.20 + bounce * 0.012

            r_off = np.array([r_shift, 0])

            r_sh_a = sh_anchor + r_off
            r_elbow_fk = end_of(parts['upper_leg_front_bg'], ang_ulf_r, r_sh_a)
            r_fpaw_fk = end_of(parts['lower_leg_front_bg'], ang_llf_r, r_elbow_fk)
            draw_part(parts['upper_leg_front_bg'], ang_ulf_r, r_sh_a)
            draw_part(parts['lower_leg_front_bg'], ang_llf_r, r_elbow_fk)

            r_hp_a = hp_anchor + r_off
            r_knee_fk = end_of(parts['upper_leg_hind_bg'], ang_ulh_r, r_hp_a)
            r_bpaw_fk = end_of(parts['lower_leg_hind_bg'], ang_llh_r, r_knee_fk)
            draw_part(parts['upper_leg_hind_bg'], ang_ulh_r, r_hp_a)
            draw_part(parts['lower_leg_hind_bg'], ang_llh_r, r_knee_fk)

            joint(r_sh_a, int(ref*0.030), '#5E5E5E', '#454545', 1)
            joint(r_elbow_fk, int(ref*0.022), '#5A5A5A', '#454545', 1)
            joint(r_fpaw_fk, int(ref*0.018), '#4E4E4E', '#3A3A3A', 1)
            joint(r_hp_a, int(ref*0.028), '#5E5E5E', '#454545', 1)
            joint(r_knee_fk, int(ref*0.022), '#5A5A5A', '#454545', 1)
            joint(r_bpaw_fk, int(ref*0.018), '#4E4E4E', '#3A3A3A', 1)

            draw_part(parts['hindq'], ang_hindq, hindq_start)
            draw_part(parts['chest'], ang_chest, neck)

            tail_anchor = hindq_start + np.array([np.cos(ang_hindq), np.sin(ang_hindq)]) * (ref * 0.55 * 0.90)
            tail_mid = end_of(parts['tail_seg1'], tail_a1, tail_anchor)
            draw_part(parts['tail_seg1'], tail_a1, tail_anchor)
            draw_part(parts['tail_seg2'], tail_a1 + tail_a2_rel * 0.4, tail_mid)
            joint(tail_anchor, int(ref*0.018), '#3A3A3A', '#222222', 1)
            joint(tail_mid, int(ref*0.012), '#4A4A4A', '#333333', 1)

            draw_part(parts['neck_conn'], ang_head, neck)
            draw_part(parts['head'], ang_head, neck)

            hs = parts['head'].get('scale', ref * 0.20)
            hx, hy = np.cos(ang_head), np.sin(ang_head)
            px_h, py_h = -hy, hx

            mz = neck + np.array([hx, hy]) * hs * 1.15 - np.array([px_h, py_h]) * hs * 0.05
            draw_c.ellipse([int(mz[0]-hs*0.28), int(mz[1]-hs*0.20),
                            int(mz[0]+hs*0.28), int(mz[1]+hs*0.20)],
                           fill='#5A5A5A', outline='#333333', width=1)
            ns = mz + np.array([hx, hy]) * hs * 0.18
            draw_c.ellipse([int(ns[0]-4), int(ns[1]-3), int(ns[0]+4), int(ns[1]+3)],
                           fill='#1A1A1A')
            ey = neck + np.array([hx, hy]) * hs * 0.62 + np.array([px_h, py_h]) * hs * 0.18
            draw_c.ellipse([int(ey[0]-5), int(ey[1]-5), int(ey[0]+5), int(ey[1]+5)],
                           fill='white', outline='#222222', width=1)
            draw_c.ellipse([int(ey[0]-2), int(ey[1]-2), int(ey[0]+2), int(ey[1]+2)],
                           fill='#1A1A1A')

            ear_base = neck + np.array([hx, hy]) * hs * 0.15 + np.array([px_h, py_h]) * hs * 0.55
            ear_mid = end_of(parts['ear_seg1'], ear_a1, ear_base)
            draw_part(parts['ear_seg1'], ear_a1, ear_base)
            draw_part(parts['ear_seg2'], ear_a2, ear_mid)
            joint(ear_base, int(ref*0.014), '#4A4A4A', '#222222', 1)
            joint(ear_mid, int(ref*0.014), '#4A4A4A', '#222222', 1)

            l_elbow_fk = end_of(parts['upper_leg_front'], ang_ulf, sh_anchor)
            l_fpaw_fk = end_of(parts['lower_leg_front'], ang_llf, l_elbow_fk)
            draw_part(parts['upper_leg_front'], ang_ulf, sh_anchor)
            draw_part(parts['lower_leg_front'], ang_llf, l_elbow_fk)

            l_knee_fk = end_of(parts['upper_leg_hind'], ang_ulh, hp_anchor)
            l_bpaw_fk = end_of(parts['lower_leg_hind'], ang_llh, l_knee_fk)
            draw_part(parts['upper_leg_hind'], ang_ulh, hp_anchor)
            draw_part(parts['lower_leg_hind'], ang_llh, l_knee_fk)

            joint(sh_anchor, int(ref*0.034), '#505050')
            joint(l_elbow_fk, int(ref*0.024), '#4A4A4A')
            joint(l_fpaw_fk, int(ref*0.020), '#2A2A2A', '#111111', 1)

            joint(hp_anchor, int(ref*0.030), '#505050')
            joint(l_knee_fk, int(ref*0.024), '#4A4A4A')
            joint(l_bpaw_fk, int(ref*0.020), '#2A2A2A', '#111111', 1)

            joint(neck, int(ref*0.025), '#4A4A4A')

            return canvas


        avg_ratio = 0.233
        dog_template, ratios_file = select_dog_template(avg_ratio)

        large_ratios = {
            'upper_leg_front': 200.0 * 0.51,
            'lower_leg_front': 200.0 * 0.30,
            'upper_leg_hind': 200.0 * 0.51,
            'lower_leg_hind': 200.0 * 0.30
        }

        character_parts = build_character_parts(large_ratios)

        frame_0 = stab_kps[0]

        neck_kp = np.array(frame_0['keypoints'][3])
        tail_kp = np.array(frame_0['keypoints'][4])
        s_len = np.linalg.norm(tail_kp - neck_kp)
        scale = 200.0 / s_len if s_len > 0 else 1.0

        kp_names = [
            'L_Eye', 'R_Eye', 'Nose', 'Neck', 'root_of_tail',
            'L_Shoulder', 'L_Elbow', 'L_F_Paw', 'R_Shoulder', 'R_Elbow', 'R_F_Paw',
            'L_Hip', 'L_Knee', 'L_B_Paw', 'R_Hip', 'R_Knee', 'R_B_Paw'
        ]

        norm_kps = {name: (np.array(frame_0['keypoints'][i]) - neck_kp) * scale for i, name in enumerate(kp_names)}

        img = render_character_rigged(norm_kps, character_parts, raw_frame_data=frame_0)

        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Part-based rigged v4 - {dog_template} template')
        plt.savefig('character_rigged_v4.png', dpi=150, bbox_inches='tight')
        plt.show()

    run_rigged_render_v4()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

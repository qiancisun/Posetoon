import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    from mmpose.apis import MMPoseInferencer
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np


    inferencer = MMPoseInferencer('animal')
    print('Model loaded!')
    return inferencer, np, plt


@app.cell
def _():
    import os
    import json

    ap10k_img_dir = 'data/ap10k/data'
    ap10k_ann_file = 'data/ap10k/annotations/ap10k-train-split1.json'

    with open(ap10k_ann_file, 'r') as f:
        ann_data = json.load(f)

    categories = ann_data['categories']
    print("Categories:")
    for cat in categories:
        print(f"  id:{cat['id']} name:{cat['name']}")

    print(f"\nTotal images: {len(ann_data['images'])}")
    print(f"Total annotations: {len(ann_data['annotations'])}")
    return ann_data, ap10k_img_dir, os


@app.cell
def _(ann_data, ap10k_img_dir, inferencer, os):
    dog_annotations = [ann for ann in ann_data['annotations'] if ann['category_id'] == 8]
    print(f"Dog annotations: {len(dog_annotations)}")

    img_id_to_filename = {img['id']: img['file_name'] for img in ann_data['images']}

    sample_dog_anns = dog_annotations[:3]

    dog_keypoints_list = []

    for ann in sample_dog_anns:
        img_filename = img_id_to_filename[ann['image_id']]
        img_path = os.path.join(ap10k_img_dir, img_filename)

        if os.path.exists(img_path):
            result_generator = inferencer(img_path, show=False, return_vis=False)
            results = [r for r in result_generator]

            if len(results[0]['predictions'][0]) > 0:
                kps = results[0]['predictions'][0][0]['keypoints']
                scores = results[0]['predictions'][0][0]['keypoint_scores']
                dog_keypoints_list.append({
                    'img': img_path,
                    'keypoints': kps,
                    'scores': scores
                })
                print(f"{img_filename}: {len(kps)} keypoints extracted")
        else:
            print(f"{img_filename} not found")

    print(f"\nTotal dogs processed: {len(dog_keypoints_list)}")
    return dog_annotations, dog_keypoints_list, img_id_to_filename


@app.cell
def _(dog_keypoints_list, np):
    KEYPOINT_NAMES = [
        'L_Eye',
        'R_Eye',
        'Nose',
        'Neck',
        'root_of_tail',
        'L_Shoulder',
        'L_Elbow',
        'L_F_Paw',
        'R_Shoulder',
        'R_Elbow',
        'R_F_Paw',
        'L_Hip',
        'L_Knee',
        'L_B_Paw',
        'R_Hip',
        'R_Knee',
        'R_B_Paw',
    ]

    SKELETON = [
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 5),
        (3, 8),
        (5, 6),
        (6, 7),
        (8, 9),
        (9, 10),
        (3, 4),
        (4, 11),
        (4, 14),
        (11, 12),
        (12, 13),
        (14, 15),
        (15, 16),
    ]

    def calc_bone_lengths(keypoints, scores, threshold=0.3):
        lengths = {}
        for (p1, p2) in SKELETON:
            if scores[p1] > threshold and scores[p2] > threshold:
                kp1 = np.array(keypoints[p1])
                kp2 = np.array(keypoints[p2])
                length = np.linalg.norm(kp2 - kp1)
                lengths[(p1, p2)] = length
        return lengths

    all_lengths = []
    for dog in dog_keypoints_list:
        lengths = calc_bone_lengths(dog['keypoints'], dog['scores'])
        all_lengths.append(lengths)

    print("Bone lengths (normalised to spine=1.0):")
    for bone in SKELETON:
        bone_lens = [l[bone] for l in all_lengths if bone in l]
        if bone_lens:
            avg = np.mean(bone_lens)
            spine_lens = [l[(3,4)] for l in all_lengths if (3,4) in l]
            spine_avg = np.mean(spine_lens)
            normalised = avg / spine_avg
            print(f"  {KEYPOINT_NAMES[bone[0]]} to {KEYPOINT_NAMES[bone[1]]}: {normalised:.2f}")
    return KEYPOINT_NAMES, SKELETON, calc_bone_lengths


@app.cell
def _(
    KEYPOINT_NAMES,
    SKELETON,
    ap10k_img_dir,
    calc_bone_lengths,
    dog_annotations,
    img_id_to_filename,
    inferencer,
    np,
    os,
):
    sample_size = 50
    sample_dog_anns_50 = dog_annotations[:sample_size]

    all_lengths_50 = []

    for ann_50 in sample_dog_anns_50:
        img_filename_50 = img_id_to_filename[ann_50['image_id']]
        img_path_50 = os.path.join(ap10k_img_dir, img_filename_50)

        if os.path.exists(img_path_50):
            result_gen_50 = inferencer(img_path_50, show=False, return_vis=False)
            results_50 = [r for r in result_gen_50]

            if len(results_50[0]['predictions'][0]) > 0:
                kps_50 = results_50[0]['predictions'][0][0]['keypoints']
                scores_50 = results_50[0]['predictions'][0][0]['keypoint_scores']
                lengths_50 = calc_bone_lengths(kps_50, scores_50, threshold=0.5)
                if (3, 4) in lengths_50:
                    all_lengths_50.append(lengths_50)

    print(f"Valid samples: {len(all_lengths_50)}")

    print("\nAverage bone lengths (normalised to spine=1.0):")
    bone_ratios = {}
    for bone_50 in SKELETON:
        bone_lens_50 = [l[bone_50] for l in all_lengths_50 if bone_50 in l]
        spine_lens_50 = [l[(3,4)] for l in all_lengths_50 if (3,4) in l]
        if bone_lens_50 and spine_lens_50:
            avg_50 = np.mean(bone_lens_50)
            spine_avg_50 = np.mean(spine_lens_50)
            ratio_50 = avg_50 / spine_avg_50
            bone_ratios[bone_50] = ratio_50
            print(f"  {KEYPOINT_NAMES[bone_50[0]]:15} to {KEYPOINT_NAMES[bone_50[1]]:15}: {ratio_50:.2f}")

    print("\nThese ratios will be used to design the 2D cartoon character proportions.")
    return (bone_ratios,)


@app.cell
def _(KEYPOINT_NAMES, SKELETON, dog_keypoints_list, plt):
    import matplotlib.patches as mpatches

    def visualize_skeletons(dog_list, keypoint_names, skeleton):
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))

        for idx, dog_item in enumerate(dog_list):
            ax = axes[idx]
            img_vis = plt.imread(dog_item['img'])
            ax.imshow(img_vis)

            kps_vis = dog_item['keypoints']
            sc_vis = dog_item['scores']

            for (p1, p2) in skeleton:
                if sc_vis[p1] > 0.3 and sc_vis[p2] > 0.3:
                    ax.plot([kps_vis[p1][0], kps_vis[p2][0]],
                            [kps_vis[p1][1], kps_vis[p2][1]],
                            'b-', linewidth=2, alpha=0.7)

            for i, (kp_vis, score_vis) in enumerate(zip(kps_vis, sc_vis)):
                if score_vis > 0.3:
                    ax.plot(kp_vis[0], kp_vis[1], 'ro', markersize=6)
                    ax.text(kp_vis[0]+3, kp_vis[1]+3, keypoint_names[i],
                           color='yellow', fontsize=6)

            ax.set_title(f'Dog {idx+1}')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('skeleton_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Saved to skeleton_visualization.png")

    visualize_skeletons(dog_keypoints_list, KEYPOINT_NAMES, SKELETON)
    return


@app.cell
def _(
    KEYPOINT_NAMES,
    SKELETON,
    ap10k_img_dir,
    calc_bone_lengths,
    dog_annotations,
    img_id_to_filename,
    inferencer,
    np,
    os,
):
    def is_running_pose(keypoints, scores, threshold=0.3, stride_ratio=0.8):
        kp = keypoints
        sc = scores
        needed = [3, 4, 7, 13]
        if not all(sc[i] > threshold for i in needed):
            needed_r = [3, 4, 10, 16]
            if not all(sc[i] > threshold for i in needed_r):
                return False
            fpaw = np.array(kp[10]); bpaw = np.array(kp[16])
        else:
            fpaw = np.array(kp[7]); bpaw = np.array(kp[13])

        neck = np.array(kp[3]); tail = np.array(kp[4])
        spine_len = np.linalg.norm(tail - neck)
        if spine_len < 1e-6:
            return False

        paw_spread = abs(fpaw[0] - bpaw[0])
        return (paw_spread / spine_len) > stride_ratio


    sample_size_run = 200
    sample_dog_anns_run = dog_annotations[:sample_size_run]

    all_lengths_run = []
    running_count = 0

    for ann_run in sample_dog_anns_run:
        img_filename_run = img_id_to_filename[ann_run['image_id']]
        img_path_run = os.path.join(ap10k_img_dir, img_filename_run)

        if not os.path.exists(img_path_run):
            continue

        result_gen_run = inferencer(img_path_run, show=False, return_vis=False)
        results_run = [r for r in result_gen_run]

        if len(results_run[0]['predictions'][0]) == 0:
            continue

        kps_run = results_run[0]['predictions'][0][0]['keypoints']
        scores_run = results_run[0]['predictions'][0][0]['keypoint_scores']

        if not is_running_pose(kps_run, scores_run):
            continue

        lengths_run = calc_bone_lengths(kps_run, scores_run, threshold=0.4)
        if (3, 4) in lengths_run:
            all_lengths_run.append(lengths_run)
            running_count += 1

        if running_count >= 30:
            break

    print(f"Running pose samples found: {running_count}")

    bone_ratios_run = {}
    for bone_r in SKELETON:
        bone_lens_r = [l[bone_r] for l in all_lengths_run if bone_r in l]
        spine_lens_r = [l[(3,4)] for l in all_lengths_run if (3,4) in l]
        if bone_lens_r and spine_lens_r:
            avg_r = np.mean(bone_lens_r)
            spine_avg_r = np.mean(spine_lens_r)
            bone_ratios_run[bone_r] = avg_r / spine_avg_r
            print(f"  {KEYPOINT_NAMES[bone_r[0]]:15} to {KEYPOINT_NAMES[bone_r[1]]:15}: {bone_ratios_run[bone_r]:.2f}")

    ratios_save_run = {
        f"{KEYPOINT_NAMES[k[0]]}_to_{KEYPOINT_NAMES[k[1]]}": float(v)
        for k, v in bone_ratios_run.items()
    }
    with open('bone_ratios.json', 'w') as f_run:
        import json as json_run
        json_run.dump(ratios_save_run, f_run, indent=2)
    print("Saved running pose bone_ratios.json")
    return (all_lengths_run,)


@app.cell
def _(KEYPOINT_NAMES, bone_ratios):
    def save_bone_ratios(ratios, keypoint_names):
        ratios_save = {
            f"{keypoint_names[k[0]]}_to_{keypoint_names[k[1]]}": float(v)
            for k, v in ratios.items()
        }
        with open('bone_ratios.json', 'w') as fout:
            import json as json_module
            json_module.dump(ratios_save, fout, indent=2)
        print("Saved to bone_ratios.json")
        print(ratios_save)

    save_bone_ratios(bone_ratios, KEYPOINT_NAMES)
    return


@app.cell
def _(KEYPOINT_NAMES, SKELETON, all_lengths_run, np):
    def classify_dog_sizes():
        import json as json_save2

        large_dog_lengths = []
        small_dog_lengths = []

        for lengths_item in all_lengths_run:
            if (11, 12) in lengths_item and (3, 4) in lengths_item:
                leg_ratio = lengths_item[(11, 12)] / lengths_item[(3, 4)]
                if leg_ratio > 0.28:
                    large_dog_lengths.append(lengths_item)
                else:
                    small_dog_lengths.append(lengths_item)

        print(f"Large dog samples: {len(large_dog_lengths)}")
        print(f"Small dog samples: {len(small_dog_lengths)}")

        def calc_ratios(lengths_list):
            ratios = {}
            for bone_item in SKELETON:
                bone_lens = [l[bone_item] for l in lengths_list if bone_item in l]
                spine_lens = [l[(3,4)] for l in lengths_list if (3,4) in l]
                if bone_lens and spine_lens:
                    ratios[bone_item] = np.mean(bone_lens) / np.mean(spine_lens)
            return ratios

        ratios_large = calc_ratios(large_dog_lengths)
        ratios_small = calc_ratios(small_dog_lengths)

        print("\nLarge dog ratios:")
        for bone_item, ratio in ratios_large.items():
            print(f"  {KEYPOINT_NAMES[bone_item[0]]:15} to {KEYPOINT_NAMES[bone_item[1]]:15}: {ratio:.2f}")

        print("\nSmall dog ratios:")
        for bone_item, ratio in ratios_small.items():
            print(f"  {KEYPOINT_NAMES[bone_item[0]]:15} to {KEYPOINT_NAMES[bone_item[1]]:15}: {ratio:.2f}")

        large_save = {
            f"{KEYPOINT_NAMES[k[0]]}_to_{KEYPOINT_NAMES[k[1]]}": float(v)
            for k, v in ratios_large.items()
        }
        with open('bone_ratios_large.json', 'w') as fout:
            json_save2.dump(large_save, fout, indent=2)
        print("\nSaved bone_ratios_large.json")

        small_save = {
            f"{KEYPOINT_NAMES[k[0]]}_to_{KEYPOINT_NAMES[k[1]]}": float(v)
            for k, v in ratios_small.items()
        }
        with open('bone_ratios_small.json', 'w') as fout2:
            json_save2.dump(small_save, fout2, indent=2)
        print("Saved bone_ratios_small.json")

    classify_dog_sizes()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

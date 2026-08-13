import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import torch
    import mmcv
    import mmpose
    import cv2
    import numpy as np
    print('torch:', torch.__version__)
    print('mmcv:', mmcv.__version__)
    print('mmpose:', mmpose.__version__)
    print('cv2:', cv2.__version__)
    print('numpy:', np.__version__)
    print('CUDA:', torch.cuda.is_available())
    return


@app.cell
def _():
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('animal')
    print('Model loaded!')
    return (inferencer,)


@app.cell
def _():
    import urllib.request
    url = "https://images.dog.ceo/breeds/labrador/n02099712_4323.jpg"
    urllib.request.urlretrieve(url, "test_dog.jpg")
    print("Image downloaded!")
    return


@app.cell
def _(inferencer):
    result_generator = inferencer('test_dog.jpg', show=False, return_vis=True)
    results = [r for r in result_generator]
    print("Keypoints extracted!")
    print("Dogs detected:", len(results[0]['predictions'][0]))
    if len(results[0]['predictions'][0]) > 0:
        keypoints = results[0]['predictions'][0][0]['keypoints']
        print("Keypoints shape:", len(keypoints), "joints")
        print("First keypoint (x, y):", keypoints[0])
    return (results,)


@app.cell
def _(results):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread('test_dog.jpg')
    plt.figure(figsize=(10, 8))
    plt.imshow(img)

    kps = results[0]['predictions'][0][0]['keypoints']
    scores_vis = results[0]['predictions'][0][0]['keypoint_scores']

    for i, (kp, score) in enumerate(zip(kps, scores_vis)):
        if score > 0.3:
            plt.plot(kp[0], kp[1], 'ro', markersize=8)
            plt.text(kp[0]+3, kp[1]+3, str(i), color='yellow', fontsize=8)

    plt.title('Dog Keypoints Detection')
    plt.axis('off')
    plt.savefig('dog_keypoints.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to dog_keypoints.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

import sys
import os
import numpy as np

DOG_IDX_START, DOG_IDX_END = 151, 269
MIN_CONFIDENCE = 0.15
VIDEO_SAMPLE_EVERY = 15


def _load_model():
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    categories = weights.meta["categories"]
    preprocess = weights.transforms()
    return model, preprocess, categories


def _predict_probs(model, preprocess, pil_img):
    import torch
    batch = preprocess(pil_img).unsqueeze(0)
    if torch.cuda.is_available():
        batch = batch.cuda()
    with torch.no_grad():
        logits = model(batch)[0]
    dog_logits = logits[DOG_IDX_START:DOG_IDX_END]
    return torch.softmax(dog_logits, dim=0).cpu().numpy()


def classify_breed(image_path, min_confidence=MIN_CONFIDENCE):
    from PIL import Image
    model, preprocess, categories = _load_model()
    img = Image.open(image_path).convert("RGB")
    probs = _predict_probs(model, preprocess, img)
    best = int(np.argmax(probs))
    conf = float(probs[best])
    name = categories[DOG_IDX_START + best]
    return (name if conf >= min_confidence else None), conf, name


def classify_breed_video(video_path, sample_every=VIDEO_SAMPLE_EVERY,
                          min_confidence=MIN_CONFIDENCE):
    import cv2
    from PIL import Image
    model, preprocess, categories = _load_model()

    cap = cv2.VideoCapture(video_path)
    accum = np.zeros(DOG_IDX_END - DOG_IDX_START, dtype=float)
    n_used, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            pil = Image.fromarray(frame[:, :, ::-1])
            accum += _predict_probs(model, preprocess, pil)
            n_used += 1
        idx += 1
    cap.release()

    if n_used == 0:
        return None, 0.0, None
    accum /= n_used
    best = int(np.argmax(accum))
    conf = float(accum[best])
    name = categories[DOG_IDX_START + best]

    order = np.argsort(accum)[::-1][:5]
    print(f"Sampled {n_used} frames. Top-5 aggregated:")
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {categories[DOG_IDX_START + int(i)]:28} {accum[int(i)]:.3f}")

    return (name if conf >= min_confidence else None), conf, name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python step_A5b_breed_classifier.py <image_or_video>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Not found: {path}")
        sys.exit(1)

    is_video = os.path.splitext(path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv"}
    if is_video:
        breed, conf, raw_best = classify_breed_video(path)
    else:
        breed, conf, raw_best = classify_breed(path)

    print(f"\nBest guess : {raw_best}")
    print(f"Confidence : {conf:.3f}  (threshold {MIN_CONFIDENCE})")
    if breed is None:
        print("=> Below threshold. Appearance will fall back to the tier default,")
        print("   which is the intended safe behaviour, not a failure.")
    else:
        print(f"=> Accepted. Pass breed='{breed}' to appearance_for(tier, breed).")

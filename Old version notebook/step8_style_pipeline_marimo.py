# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "opencv-python",
#     "numpy",
#     "scikit-learn",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import os
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    return KMeans, KNeighborsClassifier, StandardScaler, cv2, np, os


@app.cell
def _(KMeans, cv2, os):
    def extract_dominant_color(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixels = img.reshape(-1, 3)

        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(pixels)

        return kmeans.cluster_centers_[0]

    return (extract_dominant_color,)


@app.cell
def _(np):
    def compute_bone_ratios(keypoints):
        head_to_body = np.linalg.norm(keypoints['head'] - keypoints['spine'])
        leg_length = np.linalg.norm(keypoints['paw'] - keypoints['spine'])
        return np.array([head_to_body, leg_length])

    return (compute_bone_ratios,)


@app.cell
def _(extract_dominant_color, np):
    def build_style_vector(image_path, bone_ratios):
        color = extract_dominant_color(image_path)
        return np.concatenate([bone_ratios, color])

    return (build_style_vector,)


@app.cell
def _(KNeighborsClassifier, StandardScaler, np):
    X_train = np.array([
        [0.3, 0.5, 120, 100, 90],
        [0.5, 0.7, 150, 130, 110],
        [0.8, 1.0, 180, 160, 140]
    ])
    y_train = ['small', 'medium', 'large']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_scaled, y_train)
    return model, scaler


@app.cell
def _(model, scaler):
    def predict_dog_type(style_vector):
        style_scaled = scaler.transform([style_vector])
        return model.predict(style_scaled)[0]

    return (predict_dog_type,)


@app.cell
def _(build_style_vector, compute_bone_ratios, np, predict_dog_type):
    image_path = "test_dog.jpg"

    keypoints = {
        'head': np.array([0, 1]),
        'spine': np.array([0, 0]),
        'paw': np.array([0, -1])
    }

    bone_ratios = compute_bone_ratios(keypoints)
    style_vector = build_style_vector(image_path, bone_ratios)

    dog_type = predict_dog_type(style_vector)

    print("Predicted dog type:", dog_type)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

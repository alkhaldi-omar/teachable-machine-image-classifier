"""Load the Teachable Machine Keras model and predict the class of an input image.

Usage:
    python predict.py path/to/image.png
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "model", "labels.txt")
IMAGE_SIZE = (224, 224)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        # Each line looks like "0 Circle"; drop the leading index.
        return [line.strip().split(" ", 1)[1] for line in f if line.strip()]


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB")
    # Same preprocessing Teachable Machine uses: center-crop to a square, resize to 224x224,
    # then scale pixels from [0, 255] to [-1, 1].
    image = ImageOps.fit(image, IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    normalized = (array / 127.5) - 1
    return np.expand_dims(normalized, axis=0)


def predict(model, class_names, image_path):
    probabilities = model.predict(preprocess(image_path), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index]), probabilities


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a Teachable Machine model.")
    parser.add_argument("image", help="Path to the input image")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    np.set_printoptions(suppress=True)
    model = load_model(MODEL_PATH, compile=False)
    class_names = load_labels(LABELS_PATH)

    label, confidence, probabilities = predict(model, class_names, args.image)

    print(f"Image      : {os.path.basename(args.image)}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("All classes:")
    for name, p in zip(class_names, probabilities):
        print(f"  - {name:<10} {p * 100:6.2f}%")


if __name__ == "__main__":
    main()

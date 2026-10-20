"""Evaluate the model on a folder of labeled test images.

Expected layout (one sub-folder per class, named exactly like labels.txt):
    test_images/Circle/*.png
    test_images/Triangle/*.png

Usage:
    python evaluate.py --data test_images
"""
import argparse
import os

import numpy as np
from tensorflow.keras.models import load_model

from predict import LABELS_PATH, MODEL_PATH, load_labels, preprocess

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test_images")
    args = parser.parse_args()

    model = load_model(MODEL_PATH, compile=False)
    class_names = load_labels(LABELS_PATH)
    n = len(class_names)
    confusion = np.zeros((n, n), dtype=int)

    for true_index, name in enumerate(class_names):
        folder = os.path.join(args.data, name)
        if not os.path.isdir(folder):
            continue
        for file in sorted(os.listdir(folder)):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            probabilities = model.predict(preprocess(os.path.join(folder, file)), verbose=0)[0]
            confusion[true_index, int(np.argmax(probabilities))] += 1

    total = confusion.sum()
    correct = np.trace(confusion)
    print(f"Test images : {total}")
    print(f"Accuracy    : {correct / total * 100:.2f}% ({correct}/{total})\n")

    print("Accuracy per class:")
    for i, name in enumerate(class_names):
        count = confusion[i].sum()
        if count:
            print(f"  - {name:<10} {confusion[i, i] / count * 100:6.2f}% ({confusion[i, i]}/{count})")

    width = max(len(c) for c in class_names) + 2
    print("\nConfusion matrix (rows = true class, columns = predicted):")
    print(" " * width + "".join(f"{c:>{width}}" for c in class_names))
    for i, name in enumerate(class_names):
        print(f"{name:<{width}}" + "".join(f"{v:>{width}}" for v in confusion[i]))


if __name__ == "__main__":
    main()

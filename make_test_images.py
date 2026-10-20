"""Generate new, unseen circle/triangle images for testing the model.

The training images were generated in the browser with random colors, sizes, positions
and backgrounds. This script draws a fresh set the same way so we can evaluate the model
on images it has never seen.

Usage:
    python make_test_images.py --per-class 25
"""
import argparse
import colorsys
import math
import os
import random

from PIL import Image, ImageDraw

SIZE = 224
CLASSES = ["Circle", "Triangle"]


def hsl(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h / 360, l / 100, s / 100)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_shape(shape, rng):
    background = hsl(rng.uniform(0, 360), rng.uniform(0, 40), rng.uniform(80, 98))
    image = Image.new("RGB", (SIZE, SIZE), background)
    draw = ImageDraw.Draw(image, "RGBA")

    for _ in range(6):
        x, y = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
        w, h = rng.uniform(5, 40), rng.uniform(5, 40)
        draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, int(rng.uniform(0.02, 0.08) * 255)))

    r = rng.uniform(35, 85)
    cx, cy = rng.uniform(r + 5, SIZE - 5 - r), rng.uniform(r + 5, SIZE - 5 - r)
    color = hsl(rng.uniform(0, 360), rng.uniform(50, 100), rng.uniform(25, 60))

    if shape == "Circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        rot = rng.uniform(0, 2 * math.pi)
        points = [(cx + r * math.cos(rot + k * 2 * math.pi / 3), cy + r * math.sin(rot + k * 2 * math.pi / 3)) for k in range(3)]
        draw.polygon(points, fill=color)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=25)
    parser.add_argument("--out", default="test_images")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for shape in CLASSES:
        folder = os.path.join(args.out, shape)
        os.makedirs(folder, exist_ok=True)
        for i in range(args.per_class):
            draw_shape(shape, rng).save(os.path.join(folder, f"{shape.lower()}_{i:02d}.png"))
    print(f"Saved {args.per_class * len(CLASSES)} images to '{args.out}/'")


if __name__ == "__main__":
    main()

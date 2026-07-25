"""
dataset_prep.py

Two modes:

1) REAL DATA (recommended for actual accuracy / your submission):
   Download the "Chest X-Ray Images (Pneumonia)" dataset from Kaggle:
   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   Unzip it so you get:
       data/chest_xray/train/NORMAL
       data/chest_xray/train/PNEUMONIA
       data/chest_xray/val/NORMAL
       data/chest_xray/val/PNEUMONIA
       data/chest_xray/test/NORMAL
       data/chest_xray/test/PNEUMONIA
   Then just run train.py, it uses torchvision.datasets.ImageFolder on this
   structure automatically.

2) DEMO / SYNTHETIC DATA (no download needed, lets you test the ENTIRE
   pipeline end-to-end in minutes):
   Run this script directly. It generates small synthetic grayscale
   "X-ray-like" images with random blobs so the code, training loop,
   Grad-CAM, API, DB and frontend can all be verified to work without
   needing the real ~2GB dataset.

   python dataset_prep.py --synthetic
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _make_fake_xray(path: str, disease: bool, size: int = 224):
    """Creates a plausible-looking synthetic chest X-ray style image."""
    img = Image.new("L", (size, size), color=random.randint(15, 40))
    draw = ImageDraw.Draw(img)

    # Fake ribcage-ish arcs
    for i in range(6):
        y = 30 + i * 25
        draw.arc([20, y, size - 20, y + 180], start=200, end=340,
                  fill=random.randint(80, 130), width=2)

    # Fake lung fields (two blobs)
    draw.ellipse([40, 60, 100, 190], fill=random.randint(40, 70))
    draw.ellipse([124, 60, 184, 190], fill=random.randint(40, 70))

    if disease:
        # Add cloudy opacity patches to simulate pneumonia infiltrates
        for _ in range(random.randint(3, 6)):
            x = random.randint(50, 170)
            y = random.randint(70, 180)
            r = random.randint(8, 20)
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=random.randint(140, 200))

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    noise = np.random.normal(0, 8, (size, size))
    arr = np.clip(np.array(img).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").convert("RGB").save(path)


def generate_synthetic_dataset(root: str = "data/chest_xray", n_per_split=(200, 40, 40)):
    splits = ["train", "val", "test"]
    for split, n in zip(splits, n_per_split):
        for cls, is_disease in [("NORMAL", False), ("PNEUMONIA", True)]:
            out_dir = os.path.join(root, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                _make_fake_xray(os.path.join(out_dir, f"{cls.lower()}_{i}.png"), is_disease)
        print(f"[dataset_prep] {split}: {n*2} images generated")

    print(f"[dataset_prep] Synthetic dataset ready at ./{root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                         help="Generate a small synthetic dataset for pipeline testing")
    parser.add_argument("--root", default="data/chest_xray")
    args = parser.parse_args()

    if args.synthetic:
        generate_synthetic_dataset(args.root)
    else:
        print("Pass --synthetic to generate a demo dataset, or place the real "
              "Kaggle chest_xray folder under ./data/chest_xray")

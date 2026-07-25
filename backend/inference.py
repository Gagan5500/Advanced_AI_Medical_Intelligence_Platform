"""
inference.py

Loads the trained model once at startup and exposes a single function,
`predict_and_explain`, that:
  1. Preprocesses the uploaded image
  2. Runs the classifier
  3. Runs Grad-CAM to produce a heatmap overlay
  4. Saves the overlay to disk
  5. Returns prediction, confidence and the overlay path

If no trained model checkpoint exists yet (fresh clone, before running
train.py), this module falls back to an untrained model so the API still
boots and can be smoke-tested end-to-end -- predictions just won't be
meaningful until you train on real/synthetic data.
"""

import os
import sys
import uuid

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))

from model_def import build_model, CLASS_NAMES  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from config import MODEL_PATH, GRADCAM_DIR  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_model():
    model = build_model(pretrained=not os.path.exists(MODEL_PATH))
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[inference] Loaded trained weights from {MODEL_PATH}")
    else:
        print(f"[inference] WARNING: no checkpoint found at {MODEL_PATH}. "
              f"Run model/train.py first. Using untrained ImageNet backbone for now.")
    model.to(DEVICE)
    model.eval()
    return model


_model = _load_model()
_gradcam = GradCAM(_model, target_layer=_model.layer4[-1])


def predict_and_explain(image_bytes: bytes, orig_filename: str):
    pil_img = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB")
    resized_rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))

    input_tensor = _preprocess(pil_img).unsqueeze(0).to(DEVICE)

    heatmap, class_idx, confidence = _gradcam.generate(input_tensor)
    overlay = overlay_heatmap(resized_rgb, heatmap)

    out_name = f"{uuid.uuid4().hex}_{os.path.splitext(orig_filename)[0]}.png"
    out_path = os.path.join(GRADCAM_DIR, out_name)
    Image.fromarray(overlay).save(out_path)

    return {
        "predicted_class": CLASS_NAMES[class_idx],
        "confidence": float(confidence),
        "gradcam_filename": out_name,
        "gradcam_path": out_path,
    }

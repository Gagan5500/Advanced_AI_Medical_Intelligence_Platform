"""
gradcam.py

Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for the
ResNet18 classifier, so predictions are explainable: it highlights which
regions of the X-ray most influenced the model's decision.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: the trained torch model (in eval mode)
        target_layer: the conv layer to hook (for ResNet18 this is model.layer4[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: shape (1, 3, H, W), already normalized
        class_idx: which class to explain. If None, uses the predicted class.
        Returns: (heatmap [H,W] in 0..1, predicted_class_idx, confidence)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]      # (C, h, w)
        activations = self.activations[0]  # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))

        return cam, class_idx, confidence


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    original_rgb: HxWx3 uint8 image
    heatmap: HxW float array in [0,1]
    Returns HxWx3 uint8 image with heatmap overlaid.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlay = (colored.astype(np.float32) * alpha + original_rgb.astype(np.float32) * (1 - alpha))
    return np.clip(overlay, 0, 255).astype(np.uint8)

"""Vision explainability via Grad-CAM for ResNet-style backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from materials_ai.logging_utils import get_logger
from materials_ai.utils import ensure_dir

logger = get_logger(__name__)


class GradCAM:
    """Minimal Grad-CAM for the last convolutional block of ResNet."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module: Any, _inp: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module: Any, _gin: Any, gout: tuple[torch.Tensor, ...]) -> None:
        self.gradients = gout[0].detach()

    def close(self) -> None:
        for h in self._handles:
            h.remove()

    def __call__(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor | None = None,
        text: list[str] | None = None,
    ) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        image = image.unsqueeze(0) if image.ndim == 3 else image
        image = image.requires_grad_(True)
        kwargs: dict[str, Any] = {"image": image}
        if getattr(self.model, "use_tabular", False):
            if tabular is None:
                raise ValueError("Grad-CAM requires tabular input for this model")
            kwargs["tabular"] = tabular.unsqueeze(0) if tabular.ndim == 1 else tabular
        if getattr(self.model, "use_text", False):
            kwargs["text"] = text if text is not None else [""]
        out = self.model(**kwargs)
        if out.ndim == 0:
            out = out.unsqueeze(0)
        out.sum().backward()
        assert self.activations is not None and self.gradients is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def _find_resnet_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    vision = getattr(model, "vision", None)
    if vision is None:
        return None
    encoder = getattr(vision, "encoder", None)
    if encoder is None:
        return None
    if hasattr(encoder, "layer4"):
        return encoder.layer4
    return None


def save_gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image_path: str | Path,
    out_path: str | Path,
    tabular: torch.Tensor | None = None,
    text: list[str] | None = None,
) -> Path | None:
    """Save Grad-CAM overlay next to the original structure image."""
    layer = _find_resnet_layer(model)
    if layer is None:
        logger.warning("Grad-CAM skipped: ResNet layer4 not found")
        return None

    model.eval()
    device = next(model.parameters()).device
    cam_helper = GradCAM(model, layer)
    try:
        tab = tabular.to(device) if tabular is not None else None
        heatmap = cam_helper(image_tensor.to(device), tabular=tab, text=text)
    finally:
        cam_helper.close()

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    original = Image.open(original_image_path).convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    orig_np = np.asarray(original).astype(float) / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title("Structure image")
    axes[0].axis("off")
    axes[1].imshow(orig_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger

log = get_logger("extractor")


class FeatureExtractor(nn.Module):
    """
    DINOv2 Feature Extractor module.
    Extracts concatenated CLS and average-pooled patch tokens from DINOv2.
    """

    def __init__(self, model_name: str = "dinov2_vitl14_reg") -> None:
        """
        Initializes the extractor with a frozen DINOv2 model.

        Args:
            model_name: Name of the DINOv2 model to load from PyTorch Hub.
        """
        super().__init__()
        log.info("Initializing FeatureExtractor", model=model_name)

        # Load model from PyTorch Hub
        self.model = cast(
            nn.Module, torch.hub.load("facebookresearch/dinov2", model_name)
        )
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract normalized features.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224).

        Returns:
            L2-normalized feature vector of shape (B, 2048).
        """
        with torch.no_grad():
            # Extract features from DINOv2
            # forward_features returns a dict for models with registers
            features: Any = cast(Any, self.model).forward_features(x)

            cls_token: torch.Tensor = features["x_norm_clstoken"]  # [B, 1024]
            # [B, 256, 1024]
            patch_tokens: torch.Tensor = features["x_norm_patchtokens"]

            # Average pool spatial patch tokens across the spatial dimension
            avg_patch: torch.Tensor = torch.mean(patch_tokens, dim=1)  # [B, 1024]

            # Concatenate CLS token with pooled patch tokens
            # [B, 2048]
            combined: torch.Tensor = torch.cat([cls_token, avg_patch], dim=1)

            # Apply L2-normalization
            normalized: torch.Tensor = F.normalize(combined, p=2, dim=1)

            return normalized

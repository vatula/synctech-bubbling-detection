import os
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.loader import BubblingDataset
from src.data.transforms import get_train_transforms
from src.utils.logger import get_logger, setup_logger

setup_logger()
log = get_logger(__name__)


def visualize_batch() -> None:
    """
    Saves a batch of augmented images for manual verification of specular highlights.
    """
    output_dir = Path("notebooks/augmentation_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    nominal_dir = "resources/assignment/hard-negatives-bubbling"
    bubbling_dir = "resources/assignment/train-bubbling"

    if not os.path.exists(nominal_dir) or not os.path.exists(bubbling_dir):
        log.warning("Data directories not found. Skipping visualization.")
        return

    transforms = get_train_transforms(image_size=224)
    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=transforms,
    )

    log.info("Saving augmented samples", output_dir=str(output_dir))

    for i in range(min(5, len(dataset))):
        image_tensor, label = dataset[i]
        # Convert tensor back to image
        # Image is normalized: (image - mean) / std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(output_dir / f"sample_{i}_label_{label}.png"), image_bgr)

    log.info("Augmented samples saved successfully")


if __name__ == "__main__":
    visualize_batch()

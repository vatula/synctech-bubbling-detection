import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch

from src.data.loader import BubblingDataset
from src.data.transforms import get_train_transforms
from src.utils.logger import get_logger, setup_logger

setup_logger()
log = get_logger(__name__)


def test_transform_constraints() -> None:
    """
    Asserts that destructive transforms are NOT present in the training pipeline.
    Specifically checks for Blur, Elastic Warp, and Inversion.
    """
    transforms = get_train_transforms()

    # Flatten the transforms list if there are any nested Compose
    def get_all_transforms(transform: object) -> list[object]:
        if isinstance(transform, A.Compose):
            all_t: list[object] = []
            for t in transform.transforms:
                all_t.extend(get_all_transforms(t))
            return all_t
        return [transform]

    all_transforms = get_all_transforms(transforms)
    transform_names = [t.__class__.__name__ for t in all_transforms]

    # PROHIBITED TRANSFORMS
    prohibited = {
        "GaussianBlur",
        "MotionBlur",
        "Blur",
        "MedianBlur",
        "ElasticTransform",
        "InvertImg",
        "OpticalDistortion",
        "GridDistortion",
    }

    for name in transform_names:
        if name in prohibited:
            msg = f"Destructive transform '{name}' detected in training pipeline!"
            log.error(msg)
            raise AssertionError(msg)

    log.info("Transform constraints assertion", status="PASSED")


def test_transform_output_shape() -> None:
    """
    Verifies the output shape and type of the augmentation pipeline.
    """
    transforms = get_train_transforms(image_size=224)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)

    augmented = transforms(image=dummy_image)
    image = augmented["image"]

    if not isinstance(image, torch.Tensor):
        msg = f"Output should be a torch.Tensor, got {type(image)}"
        log.error(msg)
        raise AssertionError(msg)

    if image.shape != (3, 224, 224):
        msg = f"Expected shape (3, 224, 224), got {image.shape}"
        log.error(msg)
        raise AssertionError(msg)

    log.info("Transform output shape assertion", status="PASSED")


def test_visualize_samples() -> None:
    """
    Saves a batch of augmented images for manual verification of specular highlights.
    This fulfills the requirement for host-side validation.
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

    log.info("Saving augmented samples for host validation", output_dir=str(output_dir))

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

        sample_path = output_dir / f"sample_{i}_label_{label}.png"
        cv2.imwrite(str(sample_path), image_bgr)

    log.info("Augmented samples saved successfully", status="PASSED")


if __name__ == "__main__":
    test_transform_constraints()
    test_transform_output_shape()
    test_visualize_samples()

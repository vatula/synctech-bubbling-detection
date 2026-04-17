import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the strict D4 + conservative jitter augmentation pipeline.
    Preserves specular highlight topologies by avoiding destructive transforms.

    Args:
        image_size: Target size for resizing.

    Returns:
        Albumentations Compose object.
    """
    log.info("Configuring training transforms", image_size=image_size)

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            # D4 Group: 8 symmetries (Rot90 + Flips)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # Conservative jitter
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.5,
            ),
            # Normalize to ImageNet stats for DINOv2
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    """
    Returns the basic inference transforms (Resize + Normalization).

    Args:
        image_size: Target size for resizing.

    Returns:
        Albumentations Compose object.
    """
    log.info("Configuring inference transforms", image_size=image_size)

    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

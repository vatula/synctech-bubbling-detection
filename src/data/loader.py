from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

log = get_logger(__name__)


class Transform(Protocol):
    def __call__(self, *, image: np.ndarray, **kwargs: Any) -> dict[str, Any]: ...  # noqa: ANN401


class BubblingDataset(Dataset[tuple[torch.Tensor, int]]):
    """
    Dataset for Bubbling defect detection.
    Handles binary labeling: 0 for Nominal (Hard Negatives), 1 for Bubbling.
    """

    def __init__(
        self,
        nominal_dir: Path | str,
        bubbling_dir: Path | str,
        transform: "Transform | None" = None,
    ) -> None:
        """
        Initializes the dataset.

        Args:
            nominal_dir: Path to directory with nominal (normal) images.
            bubbling_dir: Path to directory with bubbling images.
            transform: Albumentations transform pipeline.
        """
        self.nominal_path = Path(nominal_dir)
        self.bubbling_path = Path(bubbling_dir)
        self.transform = transform

        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        # Load Nominal (Label 0)
        nominal_images = sorted(self.nominal_path.glob("*"))
        self.image_paths.extend(nominal_images)
        self.labels.extend([0] * len(nominal_images))

        # Load Bubbling (Label 1)
        bubbling_images = sorted(self.bubbling_path.glob("*"))
        self.image_paths.extend(bubbling_images)
        self.labels.extend([1] * len(bubbling_images))

        log.info(
            "Dataset initialized",
            nominal_count=len(nominal_images),
            bubbling_count=len(bubbling_images),
            total=len(self.image_paths),
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = str(self.image_paths[idx])
        # cv2 loads as BGR
        image = cv2.imread(image_path)
        if image is None:
            log.error("Failed to load image", path=image_path)
            raise FileNotFoundError(f"Could not load image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            # Albumentations returns various types based on transform
            image_raw = augmented["image"]
            if isinstance(image_raw, np.ndarray | torch.Tensor):
                image = image_raw
            else:
                msg = f"Unexpected image type from transform: {type(image_raw)}"
                raise TypeError(msg)

        label = self.labels[idx]

        if isinstance(image, np.ndarray):
            # Fallback if no ToTensorV2 in transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

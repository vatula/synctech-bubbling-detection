from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

logger = structlog.get_logger()

# Label constants exist to make class semantics explicit at callsites and logs.
# Expected range: binary labels only (0 or 1).
# Impact if changed: downstream classifier training and metrics labeling
# become incorrect.
NOMINAL_LABEL = 0
BUBBLING_LABEL = 1


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
        self.labels.extend([NOMINAL_LABEL] * len(nominal_images))

        # Load Bubbling (Label 1)
        bubbling_images = sorted(self.bubbling_path.glob("*"))
        self.image_paths.extend(bubbling_images)
        self.labels.extend([BUBBLING_LABEL] * len(bubbling_images))

        logger.info(
            "dataset_initialized",
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
            logger.error("image_load_failed", path=image_path)
            raise FileNotFoundError(f"Could not load image at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_raw: object
        if self.transform is None:
            image_raw = image
        else:
            augmented = self.transform(image=image)
            image_raw = augmented["image"]

        if not isinstance(image_raw, torch.Tensor):
            payload = {
                "event": "dataset_transform_contract_violation",
                "index": idx,
                "image_path": image_path,
                "received_type": type(image_raw).__name__,
            }
            logger.error(**payload)
            raise RuntimeError(str(payload))

        image_tensor = image_raw

        label = self.labels[idx]

        return image_tensor, label

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

import src.data.loader as loader_module
from src.data.loader import BubblingDataset

# Keeps test fixture generation lightweight while still representing valid RGB images.
# Expected range: positive image dimensions accepted by OpenCV and Albumentations.
# Impact if changed: modifies tensor shape expectations in deterministic contract tests.
TEST_IMAGE_SIZE = 32


class InvalidTransform:
    def __call__(self, *, image: np.ndarray, **_: object) -> dict[str, object]:
        return {"image": image}


def _write_image(path: Path) -> None:
    image = np.zeros((TEST_IMAGE_SIZE, TEST_IMAGE_SIZE, 3), dtype=np.uint8)
    image[:, :, 0] = 16
    image[:, :, 1] = 64
    image[:, :, 2] = 128
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def _build_dataset_root(tmp_path: Path) -> tuple[Path, Path]:
    nominal_dir = tmp_path / "nominal"
    bubbling_dir = tmp_path / "bubbling"
    nominal_dir.mkdir(parents=True)
    bubbling_dir.mkdir(parents=True)
    _write_image(nominal_dir / "nominal_0.png")
    _write_image(bubbling_dir / "bubbling_0.png")
    return nominal_dir, bubbling_dir


def test_contract_valid_transform_returns_tensor(tmp_path: Path) -> None:
    nominal_dir, bubbling_dir = _build_dataset_root(tmp_path)
    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=A.Compose([ToTensorV2()]),
    )

    image_tensor, label = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE)
    assert label in {loader_module.NOMINAL_LABEL, loader_module.BUBBLING_LABEL}


def test_contract_invalid_transform_raises_runtime_error(tmp_path: Path) -> None:
    nominal_dir, bubbling_dir = _build_dataset_root(tmp_path)
    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=InvalidTransform(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _ = dataset[0]

    payload = literal_eval(str(exc_info.value))
    assert payload["event"] == "dataset_transform_contract_violation"
    assert payload["index"] == 0
    assert payload["image_path"].endswith("nominal_0.png")
    assert payload["received_type"] == "ndarray"


def test_dataset_initialization_emits_class_count_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nominal_dir, bubbling_dir = _build_dataset_root(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []

    class _FakeLogger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append((event, kwargs))

        def error(self, event: str, **kwargs: object) -> None:
            _ = (event, kwargs)

    monkeypatch.setattr(loader_module, "logger", _FakeLogger())

    _ = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=A.Compose([ToTensorV2()]),
    )

    assert len(events) == 1
    event_name, payload = events[0]
    assert event_name == "dataset_initialized"
    assert payload["nominal_count"] == 1
    assert payload["bubbling_count"] == 1
    assert payload["total"] == 2


def test_repeated_indexing_is_deterministic_for_type_shape_and_label(
    tmp_path: Path,
) -> None:
    nominal_dir, bubbling_dir = _build_dataset_root(tmp_path)
    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=A.Compose([ToTensorV2()]),
    )

    torch.manual_seed(42)
    first_image, first_label = dataset[0]
    second_image, second_label = dataset[0]

    assert isinstance(first_image, torch.Tensor)
    assert isinstance(second_image, torch.Tensor)
    assert first_image.dtype == second_image.dtype
    assert first_image.shape == second_image.shape
    assert torch.equal(first_image, second_image)
    assert first_label == second_label

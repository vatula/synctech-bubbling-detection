from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from src.models.localization import AnomalyLocalizer


def test_localizer_resolves_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _DummyModel:
        backbone = "dinov2_vitl14"

        def to(self, _device: object) -> "_DummyModel":
            return self

        def eval(self) -> "_DummyModel":
            return self

    class _DummyEngine:
        def __init__(self, **_kwargs: object) -> None:
            pass

    checkpoint = tmp_path / "weights" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("", encoding="utf-8")

    def _load_from_checkpoint(_path: object, **_kwargs: object) -> _DummyModel:
        return _DummyModel()

    monkeypatch.setattr(
        "src.models.localization.Dinomaly.load_from_checkpoint",
        _load_from_checkpoint,
    )
    monkeypatch.setattr("src.models.localization.Engine", _DummyEngine)

    localizer = AnomalyLocalizer(checkpoint_path=checkpoint)
    assert localizer.checkpoint_path == checkpoint.resolve()


def test_process_images_matches_single_image_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeEngine:
        def predict(self, **_kwargs: object) -> list[object]:
            batch = SimpleNamespace(
                anomaly_map=torch.tensor(
                    [[[0.1, 0.8], [0.2, 0.9]]],
                    dtype=torch.float32,
                ),
                pred_score=torch.tensor([0.8], dtype=torch.float32),
            )
            return [batch]

    localizer = cast(Any, object.__new__(AnomalyLocalizer))
    localizer.engine = _FakeEngine()
    localizer.model = object()
    localizer.predict_batch_size = 8
    localizer.predict_num_workers = 0
    localizer.device = torch.device("cpu")

    def _create_predict_dataloader(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(
        localizer,
        "_create_predict_dataloader",
        _create_predict_dataloader,
    )

    single_output = localizer.process_image("dummy.png", threshold=0.5)
    batch_output = localizer.process_images(
        ["dummy.png"],
        threshold=0.5,
        batch_size=1,
        num_workers=0,
    )[0]

    assert np.array_equal(single_output["mask"], batch_output["mask"])
    assert np.array_equal(single_output["heatmap"], batch_output["heatmap"])
    assert single_output["boxes"] == batch_output["boxes"]
    assert single_output["score"] == batch_output["score"]


def test_overlay_results_draws_scaled_boxes() -> None:
    localizer = object.__new__(AnomalyLocalizer)
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    heatmap = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)

    overlay = localizer.overlay_results(
        image=image_rgb,
        heatmap=heatmap,
        _mask=mask,
        boxes=[[10, 10, 20, 20]],
    )

    assert overlay.shape == image_rgb.shape
    assert np.array_equal(overlay[20, 20], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(overlay[40, 40], np.array([0, 255, 0], dtype=np.uint8))


def test_overlay_results_clips_out_of_bounds_boxes() -> None:
    localizer = object.__new__(AnomalyLocalizer)
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    heatmap = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)

    overlay = localizer.overlay_results(
        image=image_rgb,
        heatmap=heatmap,
        _mask=mask,
        boxes=[[-5, -5, 120, 60]],
    )

    assert np.array_equal(overlay[0, 0], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(overlay[99, 199], np.array([0, 255, 0], dtype=np.uint8))

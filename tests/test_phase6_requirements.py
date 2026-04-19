from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.models.classifier import BubblingClassifier
from src.pipeline import MIGraphXEngine

_IMAGE_SUFFIXES: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)
_LATENCY_MEAN_THRESHOLD_US = 5_000.0
_LATENCY_P95_THRESHOLD_US = 10_000.0


def _count_dataset_images(dataset_root: Path) -> int:
    return sum(
        1
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


class _SpyLinearModel:
    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SpyLinearModel:
        _ = (X, y)
        self.fit_calls += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = float(X[0, 0])
        return np.array([1 if score >= 0.0 else 0], dtype=np.int64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.array([float(X[0, 0])], dtype=np.float64)


class _SpyReportGenerator:
    def __init__(self) -> None:
        self.call_count = 0
        self.sample_count = 0

    def generate_report(
        self,
        metrics: Mapping[str, float],
        decision_scores: np.ndarray,
        ground_truth: np.ndarray,
        output_path: str | Path = "pipeline_metrics_report.md",
    ) -> str:
        _ = (metrics, output_path)
        self.call_count += 1
        assert decision_scores.shape == ground_truth.shape
        self.sample_count = int(decision_scores.size)
        return "ok"


def _assert_latency_thresholds(profile: Mapping[str, Any]) -> None:
    artifacts = profile.get("artifacts")
    assert isinstance(artifacts, list)
    assert artifacts

    for artifact in artifacts:
        assert isinstance(artifact, Mapping)
        mean_us = artifact.get("latency_us_mean")
        p95_us = artifact.get("latency_us_p95")
        assert isinstance(mean_us, int | float)
        assert isinstance(p95_us, int | float)
        assert float(mean_us) <= _LATENCY_MEAN_THRESHOLD_US
        assert float(p95_us) <= _LATENCY_P95_THRESHOLD_US


def test_loocv_iterations_match_dataset_sample_count() -> None:
    sample_count = _count_dataset_images(Path("resources/assignment"))
    assert sample_count > 1

    features = np.column_stack(
        (
            np.linspace(-1.0, 1.0, sample_count, dtype=np.float64),
            np.linspace(1.0, -1.0, sample_count, dtype=np.float64),
        )
    )
    labels = np.array([index % 2 for index in range(sample_count)], dtype=np.int64)

    report_generator = _SpyReportGenerator()
    classifier = BubblingClassifier(report_generator=report_generator)
    spy_model = _SpyLinearModel()
    cast(Any, classifier).model = spy_model

    _ = classifier.run_loocv(features=features, labels=labels)

    assert spy_model.fit_calls == sample_count
    assert report_generator.call_count == 1
    assert report_generator.sample_count == sample_count


def test_latency_threshold_guard_accepts_valid_profile() -> None:
    profile = {
        "artifacts": [
            {"latency_us_mean": 1_500.0, "latency_us_p95": 2_700.0},
            {"latency_us_mean": 2_450.0, "latency_us_p95": 3_900.0},
        ]
    }

    _assert_latency_thresholds(profile=profile)


def test_latency_threshold_guard_rejects_profile_violation() -> None:
    profile = {
        "artifacts": [
            {
                "latency_us_mean": _LATENCY_MEAN_THRESHOLD_US + 1.0,
                "latency_us_p95": 3_400.0,
            }
        ]
    }

    with pytest.raises(AssertionError):
        _assert_latency_thresholds(profile=profile)


def test_latency_profile_artifact_respects_budget_if_available() -> None:
    profile_path = Path("results/phase6/migraphx/latency_profile.json")
    if not profile_path.exists():
        pytest.skip("MIGraphX latency profile is not available in this environment")

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    _assert_latency_thresholds(profile=payload)


class _ProgramWithMissingInputs:
    def get_parameter_shapes(self) -> dict[str, object]:
        return {"input_0": object(), "input_1": object()}


def test_pipeline_engine_raises_on_missing_required_inputs() -> None:
    engine = object.__new__(MIGraphXEngine)
    object.__setattr__(engine, "name", "fault_tolerance_probe")
    object.__setattr__(engine, "output_names", ("output",))
    object.__setattr__(engine, "_program", _ProgramWithMissingInputs())

    with pytest.raises(KeyError, match="Missing required input 'input_0'"):
        _ = engine.infer(inputs={})

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.models.classifier import BubblingClassifier
from src.utils.visualizer import StaticModelCardGenerator


def _valid_metrics() -> Mapping[str, float]:
    return {
        "accuracy": 0.94,
        "auroc": 0.95,
        "precision": 0.93,
        "recall": 0.96,
    }


def test_visualizer_rejects_missing_metric_key() -> None:
    generator = StaticModelCardGenerator()
    invalid_metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.7,
    }

    with pytest.raises(ValueError, match="Missing required metrics"):
        generator.generate_report(
            metrics=invalid_metrics,
            decision_scores=np.array([0.1, 0.9], dtype=np.float64),
            ground_truth=np.array([0, 1], dtype=np.int64),
        )


def test_visualizer_rejects_score_label_length_mismatch() -> None:
    generator = StaticModelCardGenerator()

    with pytest.raises(ValueError, match="length mismatch"):
        generator.generate_report(
            metrics=_valid_metrics(),
            decision_scores=np.array([0.2, 0.4, 0.9], dtype=np.float64),
            ground_truth=np.array([0, 1], dtype=np.int64),
        )


def test_visualizer_generates_report_with_embedded_images(tmp_path: Path) -> None:
    generator = StaticModelCardGenerator()
    output_path = tmp_path / "pipeline_metrics_report.md"

    report = generator.generate_report(
        metrics=_valid_metrics(),
        decision_scores=np.array([0.1, 0.3, 0.8, 1.2], dtype=np.float64),
        ground_truth=np.array([0, 0, 1, 1], dtype=np.int64),
        output_path=output_path,
    )

    assert "Accuracy:" in report
    assert report.count("data:image/png;base64,") == 2
    assert output_path.exists()


def test_visualizer_repeated_generation_does_not_leak_figures(tmp_path: Path) -> None:
    generator = StaticModelCardGenerator()
    baseline_open_figures = len(plt.get_fignums())

    for index in range(8):
        output_path = tmp_path / f"report_{index}.md"
        generator.generate_report(
            metrics=_valid_metrics(),
            decision_scores=np.array([0.2, 0.5, 0.9, 1.1], dtype=np.float64),
            ground_truth=np.array([0, 0, 1, 1], dtype=np.int64),
            output_path=output_path,
        )

    assert len(plt.get_fignums()) == baseline_open_figures


def test_classifier_loocv_triggers_single_report_generation() -> None:
    class _SpyReportGenerator:
        def __init__(self) -> None:
            self.call_count = 0

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
            return "ok"

    spy_generator = _SpyReportGenerator()
    classifier = BubblingClassifier(report_generator=spy_generator)

    features = np.array(
        [
            [-2.0, -1.0],
            [-1.5, -1.2],
            [1.2, 1.6],
            [1.8, 2.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    _ = classifier.run_loocv(features, labels)

    assert spy_generator.call_count == 1

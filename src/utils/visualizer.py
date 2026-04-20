from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.figure import Figure

logger = structlog.get_logger()

# Metrics are fixed to the LOOCV report contract consumed by the pipeline.
# Source provenance: classifier validation outputs.
# Impact if changed: report rendering and validation expectations break.
REQUIRED_METRIC_KEYS = ("accuracy", "auroc", "precision", "recall")

# Binary classes are fixed by dataset semantics (0 nominal, 1 bubbling).
# Source provenance: `src/data/loader.py` labeling contract.
# Impact if changed: strip-plot class partition becomes invalid.
NEGATIVE_CLASS_LABEL = 0
POSITIVE_CLASS_LABEL = 1


class StaticModelCardGenerator:
    def _validate_metrics(self, metrics: Mapping[str, float]) -> None:
        missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
        if missing:
            msg = f"Missing required metrics: {missing}"
            raise ValueError(msg)

        for key in REQUIRED_METRIC_KEYS:
            value = float(metrics[key])
            if not np.isfinite(value):
                msg = f"Metric '{key}' must be finite"
                raise ValueError(msg)

    def _validate_scores_and_labels(
        self,
        decision_scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        if decision_scores.ndim != 1 or ground_truth.ndim != 1:
            msg = "Decision scores and ground truth must be 1D arrays"
            raise ValueError(msg)

        if decision_scores.shape[0] != ground_truth.shape[0]:
            msg = (
                "Decision scores and ground truth length mismatch: "
                f"{decision_scores.shape[0]} != {ground_truth.shape[0]}"
            )
            raise ValueError(msg)

        if decision_scores.size == 0:
            msg = "Decision scores and ground truth must be non-empty"
            raise ValueError(msg)

        labels = set(np.unique(ground_truth).tolist())
        if not labels.issubset({NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL}):
            msg = "Ground truth labels must be binary (0 or 1)"
            raise ValueError(msg)

    def _figure_to_base64(self, figure: Figure) -> str:
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            buffer.close()
            plt.close(figure)

    def _build_decision_score_strip_plot(
        self,
        decision_scores: np.ndarray,
        ground_truth: np.ndarray,
    ) -> str:
        figure, axis = plt.subplots(figsize=(10, 2.6))

        nominal_scores = decision_scores[ground_truth == NEGATIVE_CLASS_LABEL]
        bubbling_scores = decision_scores[ground_truth == POSITIVE_CLASS_LABEL]

        nominal_y = np.full_like(nominal_scores, 0.0, dtype=np.float64)
        bubbling_y = np.full_like(bubbling_scores, 1.0, dtype=np.float64)

        axis.scatter(
            nominal_scores,
            nominal_y,
            c="#1f77b4",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            s=45,
            label="Nominal (0)",
        )
        axis.scatter(
            bubbling_scores,
            bubbling_y,
            c="#d62728",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            s=45,
            label="Bubbling (1)",
        )

        axis.set_yticks([0.0, 1.0])
        axis.set_yticklabels(["Nominal", "Bubbling"])
        axis.set_xlabel("LinearSVC Decision Score")
        axis.set_title("Decision Score Strip Plot by Class")
        axis.grid(alpha=0.25, linestyle="--", axis="x")
        axis.legend(loc="upper right")

        return self._figure_to_base64(figure)

    def _build_metrics_bar_chart(self, metrics: Mapping[str, float]) -> str:
        figure, axis = plt.subplots(figsize=(7.5, 4.0))

        keys = list(REQUIRED_METRIC_KEYS)
        values = [float(metrics[key]) for key in keys]
        bars = axis.bar(
            keys,
            values,
            color=["#2ca02c", "#9467bd", "#17becf", "#ff7f0e"],
        )

        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Score")
        axis.set_title("LOOCV Classification Metrics")
        axis.grid(axis="y", linestyle="--", alpha=0.3)

        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        return self._figure_to_base64(figure)

    def generate_report(
        self,
        metrics: Mapping[str, float],
        decision_scores: np.ndarray,
        ground_truth: np.ndarray,
        output_path: str | Path = "pipeline_metrics_report.md",
    ) -> str:
        self._validate_metrics(metrics)
        self._validate_scores_and_labels(decision_scores, ground_truth)

        strip_plot = self._build_decision_score_strip_plot(
            decision_scores,
            ground_truth,
        )
        metrics_plot = self._build_metrics_bar_chart(metrics)

        report_body = "\n".join(
            [
                "### Pipeline Metrics Report",
                "",
                "#### Scalar Metrics",
                f"- Accuracy: `{float(metrics['accuracy']):.4f}`",
                f"- AUROC: `{float(metrics['auroc']):.4f}`",
                f"- Precision: `{float(metrics['precision']):.4f}`",
                f"- Recall: `{float(metrics['recall']):.4f}`",
                "",
                "#### Decision Score Strip Plot",
                (
                    '<img src="data:image/png;base64,'
                    f'{strip_plot}" alt="Decision Score Strip Plot" />'
                ),
                "",
                "#### Classification Metrics Bar Chart",
                (
                    '<img src="data:image/png;base64,'
                    f'{metrics_plot}" alt="Classification Metrics Bar Chart" />'
                ),
            ]
        )

        destination = Path(output_path)
        destination.write_text(report_body, encoding="utf-8")
        logger.info(
            "static_model_card_generated",
            output_path=str(destination),
            decision_score_count=int(decision_scores.shape[0]),
        )
        return report_body

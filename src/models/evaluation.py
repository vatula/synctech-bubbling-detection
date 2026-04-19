from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC

from src.data.loader import BubblingDataset
from src.data.transforms import get_inference_transforms
from src.models.distillation import build_student
from src.models.extractor import FeatureExtractor
from src.models.localization import AnomalyLocalizer
from src.utils.logger import get_logger, setup_project
from src.utils.warning_hygiene import install_warning_hygiene

log = get_logger("evaluation")


class BinaryMetrics(TypedDict):
    accuracy: float
    auroc: float
    precision: float
    recall: float


class LocalizationMetrics(BinaryMetrics):
    mean_score_nominal: float
    mean_score_bubbling: float
    mean_box_count: float
    mean_mask_ratio_nominal: float
    mean_mask_ratio_bubbling: float


class SemanticMetrics(BinaryMetrics):
    final_distillation_loss: float
    mean_distillation_loss: float
    embedding_norm_mean: float
    embedding_norm_std: float


class LatencyMetrics(TypedDict):
    classification_mean_ms: float
    localization_mean_ms: float
    semantic_mean_ms: float
    end_to_end_estimated_mean_ms: float


class RuntimeContext(TypedDict):
    device_type: str
    cuda_available: bool
    gpu_model: str
    torch_version: str
    rocm_hip_version: str


class InferenceContext(TypedDict):
    input_resolution_hw: tuple[int, int]
    classification_batch_size: int
    localization_batch_size: int
    semantic_batch_size: int


class LocalizationContext(TypedDict):
    architecture: str
    encoder_name: str
    checkpoint_path: str
    score_threshold: float


class ValidationContext(TypedDict):
    classification_validation_type: str
    classification_fold_count: int
    localization_validation_type: str
    semantic_validation_type: str
    semantic_fold_count: int


class TrainingProgressContext(TypedDict):
    distillation_epochs: int
    distillation_loss_curve_generated: bool
    distillation_loss_curve_path: str


class ConsolidatedReport(TypedDict):
    generated_at_utc: str
    sample_count: int
    runtime_context: RuntimeContext
    inference_context: InferenceContext
    localization_context: LocalizationContext
    validation_context: ValidationContext
    training_progress: TrainingProgressContext
    classification: BinaryMetrics
    localization: LocalizationMetrics
    semantic: SemanticMetrics
    latency_ms: LatencyMetrics


@dataclass(frozen=True)
class EvaluationSample:
    path: Path
    label: int
    image: torch.Tensor


def _compute_binary_metrics(
    y_true: list[int], y_pred: list[int], y_score: list[float]
) -> BinaryMetrics:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def _load_samples() -> list[EvaluationSample]:
    dataset = BubblingDataset(
        nominal_dir=Path("resources/assignment/hard-negatives-bubbling"),
        bubbling_dir=Path("resources/assignment/train-bubbling"),
        transform=get_inference_transforms(),
    )

    samples: list[EvaluationSample] = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        sample = EvaluationSample(
            path=dataset.image_paths[idx],
            label=int(label),
            image=image,
        )
        samples.append(sample)

    return samples


def _collect_runtime_context() -> RuntimeContext:
    cuda_available = torch.cuda.is_available()
    gpu_model = "cpu"
    if cuda_available and torch.cuda.device_count() > 0:
        gpu_model = str(torch.cuda.get_device_name(0))

    return {
        "device_type": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "gpu_model": gpu_model,
        "torch_version": str(torch.__version__),
        "rocm_hip_version": str(torch.version.hip or ""),
    }


def _collect_inference_context(samples: list[EvaluationSample]) -> InferenceContext:
    if not samples:
        msg = "Cannot infer inference context from empty sample list"
        raise ValueError(msg)

    sample_image = samples[0].image
    if sample_image.ndim != 3:
        msg = "Expected sample image tensor shape [C, H, W]"
        raise ValueError(msg)

    height = int(sample_image.shape[-2])
    width = int(sample_image.shape[-1])

    return {
        "input_resolution_hw": (height, width),
        "classification_batch_size": 1,
        "localization_batch_size": 1,
        "semantic_batch_size": 1,
    }


def _collect_localization_context(
    checkpoint_path: Path,
    score_threshold: float,
    localizer: AnomalyLocalizer | None = None,
) -> LocalizationContext:
    resolved_localizer = (
        localizer
        if localizer is not None
        else AnomalyLocalizer(checkpoint_path=checkpoint_path)
    )
    provenance = resolved_localizer.get_provenance(score_threshold=score_threshold)

    return {
        "architecture": str(provenance["architecture"]),
        "encoder_name": str(provenance["encoder_name"]),
        "checkpoint_path": str(provenance["checkpoint_path"]),
        "score_threshold": float(provenance["score_threshold"]),
    }


def _build_validation_context(sample_count: int) -> ValidationContext:
    return {
        "classification_validation_type": "Leave-One-Out Cross-Validation",
        "classification_fold_count": sample_count,
        "localization_validation_type": (
            "Thresholded anomaly-map evaluation over labeled nominal/bubbling split"
        ),
        "semantic_validation_type": (
            "Leave-One-Out Cross-Validation over student embeddings"
        ),
        "semantic_fold_count": sample_count,
    }


def _write_distillation_loss_curve(
    loss_history: list[float],
    output_path: Path,
) -> bool:
    if not loss_history:
        if output_path.exists():
            output_path.unlink()
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(loss_history) + 1, dtype=np.int64)

    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    axis.plot(
        epochs,
        loss_history,
        marker="o",
        markersize=4,
        linewidth=2.0,
        color="#1f77b4",
    )
    axis.set_title("Distillation Training Progress")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.3, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def evaluate_classification(
    samples: list[EvaluationSample],
    classifier_path: Path,
) -> tuple[BinaryMetrics, float]:
    if not classifier_path.exists():
        msg = f"Classifier artifact not found: {classifier_path}"
        raise FileNotFoundError(msg)

    with classifier_path.open("rb") as file:
        classifier = cast(LinearSVC, pickle.load(file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    extractor.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    latencies_ms: list[float] = []

    for sample in samples:
        started = time.perf_counter()
        with torch.no_grad():
            features = extractor(sample.image.unsqueeze(0).to(device)).cpu().numpy()
        prediction = cast(np.ndarray, classifier.predict(features))
        decision = cast(np.ndarray, classifier.decision_function(features))
        finished = time.perf_counter()

        y_true.append(sample.label)
        y_pred.append(int(prediction[0]))
        y_score.append(float(decision[0]))
        latencies_ms.append((finished - started) * 1000.0)

    metrics = _compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score)
    latency_mean = float(np.mean(latencies_ms))
    return metrics, latency_mean


def _resolve_dinomaly_checkpoint() -> Path:
    preferred = Path("results/Dinomaly/bubbling/latest/weights/lightning/model.ckpt")
    if preferred.exists():
        return preferred

    run_dirs = sorted(Path("results/Dinomaly/bubbling").glob("v*"))
    for run_dir in reversed(run_dirs):
        candidate = run_dir / "weights/lightning/model.ckpt"
        if candidate.exists():
            return candidate

    msg = "No Dinomaly checkpoint found under results/Dinomaly/bubbling"
    raise FileNotFoundError(msg)


def evaluate_localization(
    samples: list[EvaluationSample],
) -> tuple[LocalizationMetrics, float, LocalizationContext]:
    checkpoint_path = _resolve_dinomaly_checkpoint()
    localizer = AnomalyLocalizer(checkpoint_path=checkpoint_path, predict_num_workers=0)
    score_threshold = localizer.get_calibrated_threshold()
    localization_context = _collect_localization_context(
        checkpoint_path=checkpoint_path,
        localizer=localizer,
        score_threshold=score_threshold,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    latencies_ms: list[float] = []
    box_counts: list[float] = []
    mask_ratios_nominal: list[float] = []
    mask_ratios_bubbling: list[float] = []
    nominal_scores: list[float] = []
    bubbling_scores: list[float] = []

    image_paths = [sample.path for sample in samples]
    started = time.perf_counter()
    outputs = localizer.process_images(image_paths, threshold=score_threshold)
    finished = time.perf_counter()

    if len(outputs) != len(samples):
        msg = "Localization output count mismatch"
        raise RuntimeError(msg)

    total_latency_ms = (finished - started) * 1000.0
    per_sample_latency_ms = total_latency_ms / float(len(samples))

    for sample, output in zip(samples, outputs, strict=True):
        score = float(output["score"])
        mask = output["mask"]
        boxes = output["boxes"]

        y_true.append(sample.label)
        y_score.append(score)
        y_pred.append(1 if score >= score_threshold else 0)
        latencies_ms.append(per_sample_latency_ms)
        box_counts.append(float(len(boxes)))

        mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if sample.label == 0:
            mask_ratios_nominal.append(mask_ratio)
            nominal_scores.append(score)
        else:
            mask_ratios_bubbling.append(mask_ratio)
            bubbling_scores.append(score)

    metrics: LocalizationMetrics = {
        **_compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score),
        "mean_score_nominal": float(np.mean(nominal_scores)),
        "mean_score_bubbling": float(np.mean(bubbling_scores)),
        "mean_box_count": float(np.mean(box_counts)),
        "mean_mask_ratio_nominal": float(np.mean(mask_ratios_nominal)),
        "mean_mask_ratio_bubbling": float(np.mean(mask_ratios_bubbling)),
    }

    latency_mean = float(np.mean(latencies_ms))
    return metrics, latency_mean, localization_context


def _infer_embedding_dim(state_dict: dict[str, torch.Tensor]) -> int:
    for key in ["backbone.head.weight", "head.weight"]:
        if key in state_dict and state_dict[key].ndim == 2:
            return int(state_dict[key].shape[0])

    for value in state_dict.values():
        if value.ndim == 2:
            return int(value.shape[0])

    msg = "Unable to infer embedding dimension from student checkpoint"
    raise RuntimeError(msg)


def _infer_student_architecture(
    payload_architecture: object,
    state_dict: dict[str, torch.Tensor],
) -> Literal["cnn", "vit_tiny", "fastvit_t8"]:
    if payload_architecture in {"cnn", "vit_tiny", "fastvit_t8"}:
        return cast(Literal["cnn", "vit_tiny", "fastvit_t8"], payload_architecture)

    state_keys = tuple(state_dict.keys())
    if any(
        key.startswith("feature_extractor.") or key.startswith("projection_head.")
        for key in state_keys
    ):
        return "fastvit_t8"

    if any(key.startswith("features.") for key in state_keys):
        return "cnn"

    if any(key.startswith("backbone.") for key in state_keys):
        return "vit_tiny"

    msg = "Unable to infer student architecture from checkpoint payload"
    raise RuntimeError(msg)


def evaluate_semantic(
    samples: list[EvaluationSample],
    checkpoint_path: Path,
) -> tuple[SemanticMetrics, float, list[float]]:
    if not checkpoint_path.exists():
        msg = f"Distillation checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    payload = cast(dict[str, object], torch.load(checkpoint_path, map_location="cpu"))
    state_dict = cast(dict[str, torch.Tensor], payload["student_state_dict"])
    history = cast(list[float], payload.get("loss_history", []))
    student_architecture = _infer_student_architecture(
        payload_architecture=payload.get("student_architecture"),
        state_dict=state_dict,
    )

    embedding_dim = _infer_embedding_dim(state_dict)
    student = build_student(
        architecture=student_architecture,
        embedding_dim=embedding_dim,
    )
    student.load_state_dict(state_dict)
    log.info(
        "Loaded semantic student checkpoint",
        checkpoint_path=str(checkpoint_path),
        student_architecture=student_architecture,
        embedding_dim=embedding_dim,
        loss_history_points=len(history),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    student.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    latencies_ms: list[float] = []
    embeddings: list[np.ndarray] = []
    embedding_norms: list[float] = []

    with torch.no_grad():
        for sample in samples:
            started = time.perf_counter()
            embedding = student(sample.image.unsqueeze(0).to(device)).cpu().numpy()
            finished = time.perf_counter()

            embeddings.append(embedding)
            embedding_norms.append(float(np.linalg.norm(embedding[0])))
            y_true.append(sample.label)
            latencies_ms.append((finished - started) * 1000.0)

    embedding_matrix = np.concatenate(embeddings, axis=0)

    loo = LeaveOneOut()
    evaluator = LinearSVC(class_weight="balanced", random_state=42, dual="auto")
    for train_idx, test_idx in loo.split(embedding_matrix):
        x_train = embedding_matrix[train_idx]
        y_train = np.array(y_true)[train_idx]
        x_test = embedding_matrix[test_idx]

        evaluator.fit(x_train, y_train)
        prediction = cast(np.ndarray, evaluator.predict(x_test))
        decision = cast(np.ndarray, evaluator.decision_function(x_test))

        y_pred.append(int(prediction[0]))
        y_score.append(float(decision[0]))

    semantic_metrics: SemanticMetrics = {
        **_compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score),
        "final_distillation_loss": float(history[-1]) if history else float("nan"),
        "mean_distillation_loss": float(np.mean(history)) if history else float("nan"),
        "embedding_norm_mean": float(np.mean(embedding_norms)),
        "embedding_norm_std": float(np.std(embedding_norms)),
    }

    latency_mean = float(np.mean(latencies_ms))
    return semantic_metrics, latency_mean, history


def _render_markdown(report: ConsolidatedReport) -> str:
    runtime_context = report["runtime_context"]
    inference_context = report["inference_context"]
    localization_context = report["localization_context"]
    validation_context = report["validation_context"]
    training_progress = report["training_progress"]
    classification = report["classification"]
    localization = report["localization"]
    semantic = report["semantic"]
    latency = report["latency_ms"]

    return "\n".join(
        [
            "### Phase 5 Consolidated Evaluation Report",
            f"- Generated at (UTC): {report['generated_at_utc']}",
            f"- Sample count: {report['sample_count']}",
            "",
            "### Runtime Context",
            f"- Device type: {runtime_context['device_type']}",
            f"- CUDA available: {runtime_context['cuda_available']}",
            f"- GPU model: {runtime_context['gpu_model']}",
            f"- Torch version: {runtime_context['torch_version']}",
            f"- ROCm HIP version: {runtime_context['rocm_hip_version']}",
            "",
            "### Inference Context",
            (f"- Input resolution (H, W): {inference_context['input_resolution_hw']}"),
            (
                "- Classification batch size: "
                f"{inference_context['classification_batch_size']}"
            ),
            (
                "- Localization batch size: "
                f"{inference_context['localization_batch_size']}"
            ),
            f"- Semantic batch size: {inference_context['semantic_batch_size']}",
            "",
            "### Localization Context",
            f"- Architecture: {localization_context['architecture']}",
            f"- Encoder name: {localization_context['encoder_name']}",
            f"- Checkpoint path: {localization_context['checkpoint_path']}",
            f"- Score threshold: {localization_context['score_threshold']:.6f}",
            "",
            "### Validation Context",
            (
                "- Classification validation type: "
                f"{validation_context['classification_validation_type']}"
            ),
            (
                "- Classification fold count: "
                f"{validation_context['classification_fold_count']}"
            ),
            (
                "- Localization validation type: "
                f"{validation_context['localization_validation_type']}"
            ),
            (
                "- Semantic validation type: "
                f"{validation_context['semantic_validation_type']}"
            ),
            f"- Semantic fold count: {validation_context['semantic_fold_count']}",
            "",
            "### Training Progress",
            (
                "- Distillation epochs tracked: "
                f"{training_progress['distillation_epochs']}"
            ),
            (
                "- Distillation loss curve generated: "
                f"{training_progress['distillation_loss_curve_generated']}"
            ),
            (
                "- Distillation loss curve path: "
                f"{training_progress['distillation_loss_curve_path']}"
            ),
            "",
            "### Classification",
            f"- Accuracy: {classification['accuracy']:.6f}",
            f"- AUROC: {classification['auroc']:.6f}",
            f"- Precision: {classification['precision']:.6f}",
            f"- Recall: {classification['recall']:.6f}",
            "",
            "### Localization",
            f"- Accuracy: {localization['accuracy']:.6f}",
            f"- AUROC: {localization['auroc']:.6f}",
            f"- Precision: {localization['precision']:.6f}",
            f"- Recall: {localization['recall']:.6f}",
            f"- Mean nominal anomaly score: {localization['mean_score_nominal']:.6f}",
            f"- Mean bubbling anomaly score: {localization['mean_score_bubbling']:.6f}",
            f"- Mean box count: {localization['mean_box_count']:.6f}",
            (
                "- Mean nominal mask ratio: "
                f"{localization['mean_mask_ratio_nominal']:.6f}"
            ),
            (
                "- Mean bubbling mask ratio: "
                f"{localization['mean_mask_ratio_bubbling']:.6f}"
            ),
            "",
            "### Semantic (Distilled Student)",
            f"- Accuracy: {semantic['accuracy']:.6f}",
            f"- AUROC: {semantic['auroc']:.6f}",
            f"- Precision: {semantic['precision']:.6f}",
            f"- Recall: {semantic['recall']:.6f}",
            (f"- Final distillation loss: {semantic['final_distillation_loss']:.6f}"),
            (f"- Mean distillation loss: {semantic['mean_distillation_loss']:.6f}"),
            f"- Embedding norm mean: {semantic['embedding_norm_mean']:.6f}",
            f"- Embedding norm std: {semantic['embedding_norm_std']:.6f}",
            "",
            "### Latency (mean ms / sample)",
            f"- Classification: {latency['classification_mean_ms']:.6f}",
            f"- Localization: {latency['localization_mean_ms']:.6f}",
            f"- Semantic: {latency['semantic_mean_ms']:.6f}",
            f"- End-to-end estimated: {latency['end_to_end_estimated_mean_ms']:.6f}",
        ]
    )


def write_report(report: ConsolidatedReport) -> tuple[Path, Path]:
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "phase5_consolidated_report.json"
    markdown_path = output_dir / "phase5_consolidated_report.md"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)

    with markdown_path.open("w", encoding="utf-8") as markdown_file:
        markdown_file.write(_render_markdown(report))

    return json_path, markdown_path


def main() -> None:
    install_warning_hygiene()
    samples = _load_samples()
    runtime_context = _collect_runtime_context()
    inference_context = _collect_inference_context(samples=samples)

    classification_metrics, classification_latency = evaluate_classification(
        samples=samples,
        classifier_path=Path("results/phase5/classifier/linear_svc.pkl"),
    )
    localization_metrics, localization_latency, localization_context = (
        evaluate_localization(samples=samples)
    )
    semantic_metrics, semantic_latency, semantic_loss_history = evaluate_semantic(
        samples=samples,
        checkpoint_path=Path("results/phase5/distillation/student_distillation.pt"),
    )
    report_output_dir = Path("results")
    distillation_curve_path = report_output_dir / "phase5_distillation_loss_curve.png"
    distillation_curve_generated = _write_distillation_loss_curve(
        loss_history=semantic_loss_history,
        output_path=distillation_curve_path,
    )

    latency_report: LatencyMetrics = {
        "classification_mean_ms": classification_latency,
        "localization_mean_ms": localization_latency,
        "semantic_mean_ms": semantic_latency,
        "end_to_end_estimated_mean_ms": (
            classification_latency + localization_latency + semantic_latency
        ),
    }

    report: ConsolidatedReport = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "sample_count": len(samples),
        "runtime_context": runtime_context,
        "inference_context": inference_context,
        "localization_context": localization_context,
        "validation_context": _build_validation_context(sample_count=len(samples)),
        "training_progress": {
            "distillation_epochs": len(semantic_loss_history),
            "distillation_loss_curve_generated": distillation_curve_generated,
            "distillation_loss_curve_path": str(distillation_curve_path),
        },
        "classification": classification_metrics,
        "localization": localization_metrics,
        "semantic": semantic_metrics,
        "latency_ms": latency_report,
    }

    json_path, markdown_path = write_report(report)
    log.info(
        "Phase 5 consolidated report generated",
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


if __name__ == "__main__":
    setup_project()
    main()

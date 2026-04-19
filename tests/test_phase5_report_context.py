from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from src.utils.image_size import DEFAULT_IMAGE_SIZE


def _install_evaluation_import_stubs() -> None:
    loader_module = ModuleType("src.data.loader")

    class BubblingDataset: ...

    loader_module.__dict__["BubblingDataset"] = BubblingDataset
    sys.modules.setdefault("src.data.loader", loader_module)

    transforms_module = ModuleType("src.data.transforms")

    def get_inference_transforms() -> object:
        return object()

    transforms_module.__dict__["get_inference_transforms"] = get_inference_transforms
    sys.modules.setdefault("src.data.transforms", transforms_module)

    distillation_module = ModuleType("src.models.distillation")

    def build_student(*_: object, **__: object) -> object:
        return object()

    distillation_module.__dict__["build_student"] = build_student
    sys.modules.setdefault("src.models.distillation", distillation_module)

    extractor_module = ModuleType("src.models.extractor")

    class FeatureExtractor: ...

    extractor_module.__dict__["FeatureExtractor"] = FeatureExtractor
    sys.modules.setdefault("src.models.extractor", extractor_module)

    localization_module = ModuleType("src.models.localization")

    class AnomalyLocalizer: ...

    localization_module.__dict__["AnomalyLocalizer"] = AnomalyLocalizer
    sys.modules.setdefault("src.models.localization", localization_module)


_install_evaluation_import_stubs()

evaluation_module = importlib.import_module("src.models.evaluation")
write_report = cast(
    Callable[[dict[str, Any]], tuple[Path, Path]],
    evaluation_module.write_report,
)


def _build_report() -> dict[str, Any]:
    return {
        "generated_at_utc": "2026-04-18T00:00:00+00:00",
        "sample_count": 34,
        "runtime_context": {
            "device_type": "cuda",
            "cuda_available": True,
            "gpu_model": "AMD Radeon Graphics",
            "torch_version": "2.11.0+rocm7.2",
            "rocm_hip_version": "7.2",
        },
        "inference_context": {
            "input_resolution_hw": (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            "classification_batch_size": 1,
            "localization_batch_size": 1,
            "semantic_batch_size": 1,
        },
        "localization_context": {
            "architecture": "anomalib.models.Dinomaly",
            "encoder_name": "dinov2reg_vit_large_14",
            "checkpoint_path": (
                "results/Dinomaly/bubbling/latest/weights/lightning/model.ckpt"
            ),
            "score_threshold": 0.5,
        },
        "classification": {
            "accuracy": 0.5,
            "auroc": 0.6,
            "precision": 0.4,
            "recall": 0.3,
        },
        "localization": {
            "accuracy": 0.5,
            "auroc": 0.7,
            "precision": 0.4,
            "recall": 0.6,
            "mean_score_nominal": 0.1,
            "mean_score_bubbling": 0.9,
            "mean_box_count": 1.2,
            "mean_mask_ratio_nominal": 0.01,
            "mean_mask_ratio_bubbling": 0.12,
        },
        "semantic": {
            "accuracy": 0.4,
            "auroc": 0.5,
            "precision": 0.3,
            "recall": 0.2,
            "final_distillation_loss": 0.9,
            "mean_distillation_loss": 1.1,
            "embedding_norm_mean": 0.4,
            "embedding_norm_std": 0.05,
        },
        "latency_ms": {
            "classification_mean_ms": 1.0,
            "localization_mean_ms": 2.0,
            "semantic_mean_ms": 3.0,
            "end_to_end_estimated_mean_ms": 6.0,
        },
    }


class _ImageStub:
    ndim = 3
    shape = (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)


class _SampleStub:
    def __init__(self, image: object) -> None:
        self.image = image


def test_runtime_context_contains_hardware_identity() -> None:
    runtime_collector = cast(
        Callable[[], dict[str, Any]],
        evaluation_module.__dict__["_collect_runtime_context"],
    )
    runtime_context = runtime_collector()
    assert runtime_context["device_type"] in {"cuda", "cpu"}
    assert runtime_context["gpu_model"]
    assert runtime_context["torch_version"]


def test_inference_context_has_resolution_and_batch_sizes() -> None:
    inference_collector = cast(
        Callable[[list[object]], dict[str, Any]],
        evaluation_module.__dict__["_collect_inference_context"],
    )
    sample = _SampleStub(image=cast(Any, _ImageStub()))
    inference_context = inference_collector([sample])

    assert inference_context["input_resolution_hw"] == (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )
    assert inference_context["classification_batch_size"] == 1
    assert inference_context["localization_batch_size"] == 1
    assert inference_context["semantic_batch_size"] == 1


def test_report_outputs_include_provenance_blocks(tmp_path: Path) -> None:
    report = _build_report()
    previous_dir = Path.cwd()
    try:
        os.chdir(tmp_path)
        json_path, markdown_path = write_report(report)
        json_path = json_path.resolve()
        markdown_path = markdown_path.resolve()
    finally:
        os.chdir(previous_dir)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["runtime_context"]["gpu_model"]
    assert tuple(payload["inference_context"]["input_resolution_hw"]) == (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )
    assert payload["inference_context"]["classification_batch_size"] == 1
    assert payload["inference_context"]["localization_batch_size"] == 1
    assert payload["inference_context"]["semantic_batch_size"] == 1
    assert payload["localization_context"]["architecture"] == "anomalib.models.Dinomaly"

    assert "### Runtime Context" in markdown
    assert "### Inference Context" in markdown
    assert "### Localization Context" in markdown


def test_render_markdown_raises_when_context_blocks_missing() -> None:
    markdown_renderer = cast(
        Callable[[dict[str, Any]], str],
        evaluation_module.__dict__["_render_markdown"],
    )
    incomplete = dict(_build_report())
    incomplete.pop("runtime_context")
    incomplete.pop("localization_context")

    try:
        markdown_renderer(incomplete)
    except KeyError:
        return

    raise AssertionError("Expected KeyError when required context blocks are missing")

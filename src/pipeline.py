from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import torch

from src.data.transforms import get_inference_transforms
from src.models.localization import AnomalyLocalizer
from src.utils.image_size import resolve_image_size
from src.utils.logger import get_logger, setup_project

try:
    import migraphx  # type: ignore
except ImportError:
    migraphx = None

log = get_logger("pipeline")


class ClassificationPayload(TypedDict):
    label: str
    label_id: int
    decision_score: float


class LocalizationPayload(TypedDict):
    anomaly_score: float
    boxes: list[list[int]]


class SemanticPayload(TypedDict):
    embedding_norm: float
    semantic_text: str


class UnifiedPayload(TypedDict):
    image_path: str
    classification: ClassificationPayload
    localization: LocalizationPayload
    semantic: SemanticPayload


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


def _select_output(
    outputs: Mapping[str, np.ndarray],
    candidates: Sequence[str],
) -> np.ndarray:
    for name in candidates:
        value = outputs.get(name)
        if value is not None:
            return value

    if outputs:
        return next(iter(outputs.values()))

    msg = "No outputs were produced by the compiled engine"
    raise RuntimeError(msg)


class MIGraphXEngine:
    def __init__(
        self,
        program_path: Path,
        output_names: Sequence[str],
        name: str,
    ) -> None:
        self.name = name
        self.program_path = program_path.expanduser().resolve()
        self.output_names = tuple(output_names)
        self._module = self._require_module()
        self._program = self._load_program()

    @staticmethod
    def _require_module() -> object:
        if migraphx is None:
            msg = "migraphx Python module is not available"
            raise RuntimeError(msg)
        return migraphx

    def _load_program(self) -> object:
        if not self.program_path.exists():
            msg = f"Compiled MIGraphX program not found: {self.program_path}"
            raise FileNotFoundError(msg)

        load_program = cast(
            Callable[[str], object] | None,
            getattr(self._module, "load", None),
        )
        if not callable(load_program):
            msg = "migraphx.load is unavailable"
            raise RuntimeError(msg)

        program = load_program(str(self.program_path))
        log.info("Loaded MIGraphX program", name=self.name, path=str(self.program_path))
        return program

    def _to_argument(self, tensor: np.ndarray) -> object:
        argument_ctor = cast(
            Callable[[np.ndarray], object] | None,
            getattr(self._module, "argument", None),
        )
        if not callable(argument_ctor):
            msg = "migraphx.argument is unavailable"
            raise RuntimeError(msg)

        contiguous = np.ascontiguousarray(tensor.astype(np.float32))
        return argument_ctor(contiguous)

    @staticmethod
    def _to_numpy(argument: object) -> np.ndarray:
        maybe_tolist = cast(
            Callable[[], object] | None,
            getattr(argument, "tolist", None),
        )
        if callable(maybe_tolist):
            return np.asarray(maybe_tolist(), dtype=np.float32)

        return np.asarray(argument, dtype=np.float32)

    @staticmethod
    def _normalize_outputs(raw_outputs: object) -> list[object]:
        if isinstance(raw_outputs, list):
            return raw_outputs
        if isinstance(raw_outputs, tuple):
            return list(raw_outputs)
        return [raw_outputs]

    def infer(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        get_shapes = cast(
            Callable[[], Mapping[str, object]] | None,
            getattr(self._program, "get_parameter_shapes", None),
        )
        if not callable(get_shapes):
            msg = f"Program {self.name} does not expose get_parameter_shapes()"
            raise RuntimeError(msg)

        parameter_shapes = dict(get_shapes())
        if not parameter_shapes:
            msg = f"Program {self.name} has no declared parameter shapes"
            raise RuntimeError(msg)

        bound_inputs: dict[str, object] = {}
        input_values = tuple(inputs.values())
        for parameter_name in parameter_shapes:
            if parameter_name in inputs:
                value = inputs[parameter_name]
            elif len(input_values) == 1:
                value = input_values[0]
            else:
                msg = (
                    f"Missing required input '{parameter_name}' for program {self.name}"
                )
                raise KeyError(msg)
            bound_inputs[parameter_name] = self._to_argument(value)

        run_program = cast(
            Callable[[Mapping[str, object]], object] | None,
            getattr(self._program, "run", None),
        )
        if not callable(run_program):
            msg = f"Program {self.name} does not expose run()"
            raise RuntimeError(msg)

        raw_outputs = run_program(bound_inputs)
        normalized = self._normalize_outputs(raw_outputs=raw_outputs)

        output_dict: dict[str, np.ndarray] = {}
        for index, output_value in enumerate(normalized):
            if index < len(self.output_names):
                output_name = self.output_names[index]
            else:
                output_name = f"output_{index}"
            output_dict[output_name] = self._to_numpy(output_value)

        return output_dict


class UnifiedCompiledPipeline:
    def __init__(
        self,
        compiled_dir: str | Path = "results/phase6/migraphx",
        image_size: int | None = None,
        anomaly_threshold: float = 0.5,
        dinomaly_checkpoint: str | Path | None = None,
    ) -> None:
        self.compiled_dir = Path(compiled_dir).expanduser().resolve()
        self.image_size = resolve_image_size(image_size)
        self.anomaly_threshold = anomaly_threshold
        self.transform = get_inference_transforms(image_size=self.image_size)

        self.feature_engine = MIGraphXEngine(
            program_path=self.compiled_dir / "dinov2_feature_extractor.mxr",
            output_names=("features",),
            name="dinov2_feature_extractor",
        )
        self.svm_engine = MIGraphXEngine(
            program_path=self.compiled_dir / "linear_svc_head.mxr",
            output_names=("decision_scores", "predicted_labels"),
            name="linear_svc_head",
        )
        self.student_engine = MIGraphXEngine(
            program_path=self.compiled_dir / "student_semantic.mxr",
            output_names=("embeddings",),
            name="student_semantic",
        )

        self.dinomaly_engine = self._resolve_dinomaly_engine()
        self.dinomaly_localizer = self._resolve_dinomaly_localizer(
            checkpoint_path=dinomaly_checkpoint
        )

    def _resolve_dinomaly_engine(self) -> MIGraphXEngine | None:
        candidates = sorted(self.compiled_dir.rglob("*dinomaly*.mxr"))
        if not candidates:
            log.warning(
                "No compiled Dinomaly MIGraphX program found; using checkpoint fallback"
            )
            return None

        return MIGraphXEngine(
            program_path=candidates[0],
            output_names=("anomaly_map", "anomaly_score"),
            name="dinomaly",
        )

    def _resolve_dinomaly_localizer(
        self,
        checkpoint_path: str | Path | None,
    ) -> AnomalyLocalizer | None:
        if self.dinomaly_engine is not None:
            return None

        resolved_checkpoint = (
            Path(checkpoint_path).expanduser().resolve()
            if checkpoint_path is not None
            else _resolve_dinomaly_checkpoint()
        )
        return AnomalyLocalizer(
            checkpoint_path=resolved_checkpoint,
            predict_batch_size=1,
            predict_num_workers=0,
        )

    def _load_image(self, image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            msg = f"Failed to read image: {image_path}"
            raise FileNotFoundError(msg)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=rgb)
        image_tensor = cast(torch.Tensor, transformed["image"])
        return rgb, image_tensor

    def _predict_classification(
        self,
        image_tensor: torch.Tensor,
    ) -> ClassificationPayload:
        image_batch = image_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
        feature_outputs = self.feature_engine.infer(inputs={"images": image_batch})
        features = _select_output(feature_outputs, ("features",))

        svm_outputs = self.svm_engine.infer(inputs={"features": features})
        decision = float(
            _select_output(svm_outputs, ("decision_scores",)).reshape(-1)[0]
        )
        label_id = int(
            _select_output(svm_outputs, ("predicted_labels",)).reshape(-1)[0]
        )
        label = "bubbling" if label_id == 1 else "nominal"

        return {
            "label": label,
            "label_id": label_id,
            "decision_score": decision,
        }

    def _extract_boxes(
        self,
        anomaly_map: np.ndarray,
        target_width: int,
        target_height: int,
    ) -> list[list[int]]:
        mask = (anomaly_map >= self.anomaly_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes: list[list[int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            raw_boxes.append([x, y, x + w, y + h])

        source_height, source_width = anomaly_map.shape
        return self._scale_boxes_to_target_size(
            boxes=raw_boxes,
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
        )

    @staticmethod
    def _scale_boxes_to_target_size(
        boxes: list[list[int]],
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> list[list[int]]:
        if source_width <= 0 or source_height <= 0:
            msg = "Source dimensions must be positive"
            raise ValueError(msg)
        if target_width <= 0 or target_height <= 0:
            msg = "Target dimensions must be positive"
            raise ValueError(msg)

        scale_x = target_width / source_width
        scale_y = target_height / source_height

        scaled_boxes: list[list[int]] = []
        for box in boxes:
            if len(box) != 4:
                msg = "Each box must contain exactly four values"
                raise ValueError(msg)

            x1 = int(round(box[0] * scale_x))
            y1 = int(round(box[1] * scale_y))
            x2 = int(round(box[2] * scale_x))
            y2 = int(round(box[3] * scale_y))

            x1 = min(max(x1, 0), target_width - 1)
            y1 = min(max(y1, 0), target_height - 1)
            x2 = min(max(x2, 0), target_width - 1)
            y2 = min(max(y2, 0), target_height - 1)

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            scaled_boxes.append([x1, y1, x2, y2])

        return scaled_boxes

    @staticmethod
    def _normalize_map(raw_map: np.ndarray) -> np.ndarray:
        squeezed = np.squeeze(raw_map)
        if squeezed.ndim == 3:
            squeezed = np.mean(squeezed, axis=0)
        if squeezed.ndim != 2:
            msg = "Unable to convert Dinomaly output to a 2D anomaly map"
            raise RuntimeError(msg)

        normalized = squeezed.astype(np.float32)
        max_value = float(np.max(normalized))
        min_value = float(np.min(normalized))
        if max_value > min_value:
            normalized = cast(
                np.ndarray,
                (normalized - min_value) / (max_value - min_value),
            )
        return cast(np.ndarray, normalized)

    def _predict_localization(
        self,
        image_path: Path,
        rgb_image: np.ndarray,
        image_tensor: torch.Tensor,
    ) -> LocalizationPayload:
        if self.dinomaly_engine is not None:
            image_batch = image_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
            outputs = self.dinomaly_engine.infer(inputs={"images": image_batch})
            anomaly_map = self._normalize_map(
                _select_output(outputs, ("anomaly_map", "pred_mask", "output_0"))
            )
            anomaly_score = float(np.max(anomaly_map))
            boxes = self._extract_boxes(
                anomaly_map=anomaly_map,
                target_width=int(rgb_image.shape[1]),
                target_height=int(rgb_image.shape[0]),
            )
            return {"anomaly_score": anomaly_score, "boxes": boxes}

        if self.dinomaly_localizer is None:
            msg = (
                "Neither compiled Dinomaly engine nor checkpoint localizer is available"
            )
            raise RuntimeError(msg)

        output = self.dinomaly_localizer.process_image(
            image_path=image_path,
            threshold=self.anomaly_threshold,
        )
        return {
            "anomaly_score": float(output["score"]),
            "boxes": output["boxes"],
        }

    def _predict_semantic(
        self,
        image_tensor: torch.Tensor,
        classification: ClassificationPayload,
        localization: LocalizationPayload,
    ) -> SemanticPayload:
        image_batch = image_tensor.unsqueeze(0).cpu().numpy().astype(np.float32)
        outputs = self.student_engine.infer(inputs={"images": image_batch})
        embeddings = _select_output(outputs, ("embeddings", "output_0"))
        embedding_vector = embeddings.reshape(embeddings.shape[0], -1)[0]
        embedding_norm = float(np.linalg.norm(embedding_vector))

        bubbling_flag = (
            classification["label_id"] == 1
            or localization["anomaly_score"] >= self.anomaly_threshold
        )
        semantic_text = (
            "Bubbling indicators detected; prioritize targeted inspection."
            if bubbling_flag
            else "Nominal surface characteristics dominate in the distilled embedding."
        )

        return {
            "embedding_norm": embedding_norm,
            "semantic_text": semantic_text,
        }

    def predict(self, image_path: str | Path) -> UnifiedPayload:
        resolved_path = Path(image_path).expanduser().resolve()
        rgb_image, image_tensor = self._load_image(resolved_path)

        classification = self._predict_classification(image_tensor=image_tensor)
        localization = self._predict_localization(
            image_path=resolved_path,
            rgb_image=rgb_image,
            image_tensor=image_tensor,
        )
        semantic = self._predict_semantic(
            image_tensor=image_tensor,
            classification=classification,
            localization=localization,
        )

        payload: UnifiedPayload = {
            "image_path": str(resolved_path),
            "classification": classification,
            "localization": localization,
            "semantic": semantic,
        }
        log.info("Unified pipeline inference complete", **payload)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Phase 6 pipeline over compiled MIGraphX engines"
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--compiled_dir",
        type=Path,
        default=Path("results/phase6/migraphx"),
        help="Directory with compiled MIGraphX programs",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("results/phase6/unified_payload.json"),
        help="Destination file for unified JSON payload",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Inference image size override",
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=0.5,
        help="Threshold for Dinomaly score and mask-to-box extraction",
    )
    parser.add_argument(
        "--dinomaly_checkpoint",
        type=Path,
        default=None,
        help="Optional fallback Dinomaly checkpoint when compiled engine is absent",
    )
    return parser.parse_args()


def main() -> None:
    setup_project()
    args = parse_args()
    pipeline = UnifiedCompiledPipeline(
        compiled_dir=args.compiled_dir,
        image_size=args.image_size,
        anomaly_threshold=args.anomaly_threshold,
        dinomaly_checkpoint=args.dinomaly_checkpoint,
    )
    payload = pipeline.predict(image_path=args.image)

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("Persisted unified pipeline payload", path=str(output_path))


if __name__ == "__main__":
    main()

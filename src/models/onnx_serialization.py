from __future__ import annotations

import argparse
import pickle
import subprocess
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
import torch.nn as nn
from sklearn.svm import LinearSVC

from src.models.distillation import build_student
from src.models.extractor import FeatureExtractor
from src.utils.image_size import resolve_image_size
from src.utils.logger import get_logger, setup_project

log = get_logger("onnx_serialization")

StudentArchitecture = Literal["cnn", "vit_tiny_patch16_224", "fastvit_t8"]


class ExportManifest(TypedDict):
    dinov2_onnx: str
    svm_onnx: str
    student_onnx: str
    dinomaly_export_root: str
    dinomaly_checkpoint: str


class LinearSVCPredictor(nn.Module):
    def __init__(self, coef: torch.Tensor, intercept: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("coef", coef)
        self.register_buffer("intercept", intercept)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coef = cast(torch.Tensor, self.coef)
        intercept = cast(torch.Tensor, self.intercept)
        decision_scores = torch.matmul(features, coef.transpose(0, 1)) + intercept
        flattened_scores = decision_scores.squeeze(1)
        predicted_labels = (flattened_scores > 0.0).to(dtype=torch.int64)
        return flattened_scores, predicted_labels


class CommandExecutionError(RuntimeError):
    def __init__(self, command: list[str], return_code: int, stderr: str) -> None:
        self.command = command
        self.return_code = return_code
        self.stderr = stderr
        super().__init__(
            f"Command failed with exit code {return_code}: {' '.join(command)}"
        )


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
) -> StudentArchitecture:
    if payload_architecture == "vit_tiny":
        return "vit_tiny_patch16_224"

    if payload_architecture in {"cnn", "vit_tiny_patch16_224", "fastvit_t8"}:
        return cast(StudentArchitecture, payload_architecture)

    state_keys = tuple(state_dict.keys())
    if any(
        key.startswith("feature_extractor.") or key.startswith("projection_head.")
        for key in state_keys
    ):
        return "fastvit_t8"

    if any(key.startswith("features.") for key in state_keys):
        return "cnn"

    if any(key.startswith("backbone.") for key in state_keys):
        return "vit_tiny_patch16_224"

    msg = "Unable to infer student architecture from checkpoint payload"
    raise RuntimeError(msg)


def _load_svm_predictor(classifier_path: Path) -> tuple[LinearSVCPredictor, int]:
    with classifier_path.open("rb") as model_file:
        fitted = pickle.load(model_file)

    if not isinstance(fitted, LinearSVC):
        msg = f"Expected LinearSVC artifact at {classifier_path}, got {type(fitted)}"
        raise TypeError(msg)

    coef = torch.tensor(fitted.coef_, dtype=torch.float32)
    intercept = torch.tensor(fitted.intercept_, dtype=torch.float32)
    if coef.ndim != 2 or coef.shape[0] != 1:
        msg = "Only binary LinearSVC export is supported"
        raise RuntimeError(msg)

    model = LinearSVCPredictor(coef=coef, intercept=intercept)
    model.eval()
    feature_dim = int(coef.shape[1])
    return model, feature_dim


def _load_student_model(
    checkpoint_path: Path,
) -> tuple[nn.Module, int, StudentArchitecture]:
    payload = cast(dict[str, object], torch.load(checkpoint_path, map_location="cpu"))
    state_dict = cast(dict[str, torch.Tensor], payload["student_state_dict"])
    architecture = _infer_student_architecture(
        payload_architecture=payload.get("student_architecture"),
        state_dict=state_dict,
    )
    embedding_dim = _infer_embedding_dim(state_dict)

    student = build_student(architecture=architecture, embedding_dim=embedding_dim)
    student.load_state_dict(state_dict)
    student.eval()

    return student, embedding_dim, architecture


def _export_dinov2_feature_extractor(
    output_path: Path,
    image_size: int,
    model_name: str,
    opset: int,
) -> None:
    extractor = FeatureExtractor(model_name=model_name)
    extractor.eval()

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        extractor,
        (dummy,),
        output_path,
        input_names=["images"],
        output_names=["features"],
        dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info(
        "Exported DINOv2 feature extractor to ONNX",
        path=str(output_path),
        model_name=model_name,
        image_size=image_size,
        opset=opset,
    )


def _export_svm_predictor(
    classifier_path: Path,
    output_path: Path,
    opset: int,
) -> None:
    predictor, feature_dim = _load_svm_predictor(classifier_path=classifier_path)
    dummy = torch.randn(1, feature_dim, dtype=torch.float32)

    torch.onnx.export(
        predictor,
        (dummy,),
        output_path,
        input_names=["features"],
        output_names=["decision_scores", "predicted_labels"],
        dynamic_axes={
            "features": {0: "batch"},
            "decision_scores": {0: "batch"},
            "predicted_labels": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info(
        "Exported fitted LinearSVC logic to ONNX",
        path=str(output_path),
        feature_dim=feature_dim,
        opset=opset,
    )


def _export_student_model(
    checkpoint_path: Path,
    output_path: Path,
    image_size: int,
    opset: int,
) -> None:
    student, embedding_dim, architecture = _load_student_model(
        checkpoint_path=checkpoint_path
    )

    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        student,
        (dummy,),
        output_path,
        input_names=["images"],
        output_names=["embeddings"],
        dynamic_axes={"images": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info(
        "Exported distilled student model to ONNX",
        path=str(output_path),
        architecture=architecture,
        embedding_dim=embedding_dim,
        image_size=image_size,
        opset=opset,
    )


def _run_command(command: list[str], *, log_failure: bool = True) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        log.info(
            "Command completed",
            command=" ".join(command),
            stdout=completed.stdout.strip(),
        )
        return

    stderr = completed.stderr.strip()
    if log_failure:
        log.warning(
            "Command failed",
            command=" ".join(command),
            return_code=completed.returncode,
            stderr=stderr,
        )
    raise CommandExecutionError(
        command=command,
        return_code=completed.returncode,
        stderr=stderr,
    )


def _build_dinomaly_export_command(
    config_path: Path,
    checkpoint_path: Path,
    export_root: Path,
    output_flag: str | None,
) -> list[str]:
    command = [
        "python",
        "-m",
        "src.utils.anomalib_entrypoint",
        "export",
        "--config",
        str(config_path),
        "--ckpt_path",
        str(checkpoint_path),
        "--export_type",
        "ONNX",
    ]
    if output_flag is not None:
        command.extend([output_flag, str(export_root)])
    return command


def _export_dinomaly_onnx(
    config_path: Path,
    checkpoint_path: Path,
    export_root: Path,
) -> None:
    export_root.mkdir(parents=True, exist_ok=True)
    # Newer anomalib CLI prefers `--export_root`, while some older builds
    # still accept `--output`. Prioritize the modern flag first.
    output_flag_candidates: list[str | None] = ["--export_root", "--output", None]

    last_error: CommandExecutionError | None = None
    for index, output_flag in enumerate(output_flag_candidates):
        command = _build_dinomaly_export_command(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            export_root=export_root,
            output_flag=output_flag,
        )
        try:
            _run_command(command=command, log_failure=False)
            log.info(
                "Exported Dinomaly ONNX graph",
                checkpoint_path=str(checkpoint_path),
                export_root=str(export_root),
                output_flag=output_flag if output_flag is not None else "<none>",
            )
            return
        except CommandExecutionError as error:
            last_error = error
            if index < len(output_flag_candidates) - 1:
                log.info(
                    (
                        "Dinomaly export command variant failed; "
                        "retrying compatibility variant"
                    ),
                    command=" ".join(command),
                    output_flag=output_flag if output_flag is not None else "<none>",
                    return_code=error.return_code,
                )

    if last_error is not None:
        log.warning(
            "Dinomaly export failed after all command variants",
            attempted_output_flags=[
                candidate if candidate is not None else "<none>"
                for candidate in output_flag_candidates
            ],
            last_command=" ".join(last_error.command),
            last_return_code=last_error.return_code,
            last_stderr=last_error.stderr,
        )
        raise last_error

    msg = "Dinomaly ONNX export failed unexpectedly"
    raise RuntimeError(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6.1 ONNX graph serialization for DINOv2/SVM/Student/Dinomaly"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/phase6/onnx"),
        help="Directory where serialized ONNX artifacts are written.",
    )
    parser.add_argument(
        "--classifier_artifact",
        type=Path,
        default=Path("results/phase5/classifier/linear_svc.pkl"),
        help="Path to fitted LinearSVC artifact.",
    )
    parser.add_argument(
        "--student_checkpoint",
        type=Path,
        default=Path("results/phase5/distillation/student_distillation.pt"),
        help="Path to distilled student checkpoint.",
    )
    parser.add_argument(
        "--dinomaly_config",
        type=Path,
        default=Path("dinomaly_config.yaml"),
        help="Dinomaly training/export config used for anomalib export.",
    )
    parser.add_argument(
        "--dinomaly_checkpoint",
        type=Path,
        default=None,
        help="Optional explicit Dinomaly checkpoint override.",
    )
    parser.add_argument(
        "--dinov2_model_name",
        type=str,
        default="dinov2_vitl14_reg",
        help="PyTorch Hub DINOv2 model identifier.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Square image side for ONNX tracing; uses project default when omitted.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version used for all exports.",
    )
    parser.add_argument(
        "--skip_dinomaly_export",
        action="store_true",
        help="Skip anomalib Dinomaly export call (debug mode).",
    )
    return parser.parse_args()


def export_phase6_onnx(args: argparse.Namespace) -> ExportManifest:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = args.classifier_artifact.expanduser().resolve()
    student_checkpoint_path = args.student_checkpoint.expanduser().resolve()
    config_path = args.dinomaly_config.expanduser().resolve()
    image_size = resolve_image_size(args.image_size)

    if args.dinomaly_checkpoint is not None:
        dinomaly_checkpoint = args.dinomaly_checkpoint.expanduser().resolve()
    else:
        dinomaly_checkpoint = _resolve_dinomaly_checkpoint().resolve()

    dinov2_onnx = output_dir / "dinov2_feature_extractor.onnx"
    svm_onnx = output_dir / "linear_svc_head.onnx"
    student_onnx = output_dir / "student_semantic.onnx"
    dinomaly_export_root = output_dir / "dinomaly"

    _export_dinov2_feature_extractor(
        output_path=dinov2_onnx,
        image_size=image_size,
        model_name=args.dinov2_model_name,
        opset=args.opset,
    )
    _export_svm_predictor(
        classifier_path=classifier_path,
        output_path=svm_onnx,
        opset=args.opset,
    )
    _export_student_model(
        checkpoint_path=student_checkpoint_path,
        output_path=student_onnx,
        image_size=image_size,
        opset=args.opset,
    )

    if args.skip_dinomaly_export:
        log.info(
            "Skipping Dinomaly ONNX export by request",
            checkpoint_path=str(dinomaly_checkpoint),
        )
    else:
        _export_dinomaly_onnx(
            config_path=config_path,
            checkpoint_path=dinomaly_checkpoint,
            export_root=dinomaly_export_root,
        )

    manifest: ExportManifest = {
        "dinov2_onnx": str(dinov2_onnx),
        "svm_onnx": str(svm_onnx),
        "student_onnx": str(student_onnx),
        "dinomaly_export_root": str(dinomaly_export_root),
        "dinomaly_checkpoint": str(dinomaly_checkpoint),
    }
    log.info("Phase 6.1 ONNX serialization completed", **manifest)
    return manifest


def main() -> None:
    setup_project()
    args = parse_args()
    export_phase6_onnx(args=args)


if __name__ == "__main__":
    main()

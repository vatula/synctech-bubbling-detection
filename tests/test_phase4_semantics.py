from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from src.models.distillation import (
    ContrastiveDistillationTrainer,
    QwenTeacherEncoder,
    TinyCNNStudent,
    contrastive_distillation_loss,
    save_distillation_checkpoint,
)
from src.models.qlora_config import (
    QLoRASettings,
    load_qlora_config,
    qlora_config_payload,
    qlora_settings_from_payload,
    save_qlora_config,
)
from src.models.vlm_json import extract_defect_json
from src.utils.image_size import DEFAULT_IMAGE_SIZE


class DummyTeacher:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode_images(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        if images.ndim != 4:
            msg = "Expected images with shape [B,C,H,W]"
            raise ValueError(msg)
        if len(prompts) != int(images.shape[0]):
            msg = "Prompts length must match batch size"
            raise ValueError(msg)

        pooled = images.mean(dim=(2, 3))
        repeats = (self.embedding_dim + pooled.shape[1] - 1) // pooled.shape[1]
        features = pooled.repeat(1, repeats)[:, : self.embedding_dim]
        return torch.nn.functional.normalize(features, dim=-1)

    @property
    def teacher_lora_lr(self) -> float:
        return 5e-5

    @property
    def lora_metadata(self) -> dict[str, Any]:
        return {}

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []


def assert_raises(
    expected_exception: type[Exception],
    fn: object,
    *args: object,
    match: str | None = None,
) -> None:
    callable_fn = fn
    if not callable(callable_fn):
        msg = "Provided function is not callable"
        raise TypeError(msg)

    try:
        callable_fn(*args)
    except expected_exception as err:
        if match is not None and match not in str(err):
            msg = f"Exception message does not include expected fragment: {match}"
            raise AssertionError(msg) from err
        return
    except Exception as err:
        msg = f"Unexpected exception type raised: {type(err).__name__}"
        raise AssertionError(msg) from err

    msg = f"Expected exception {expected_exception.__name__} was not raised"
    raise AssertionError(msg)


def test_extract_defect_json_valid_schema() -> None:
    raw = (
        "prefix "
        '{"coordinates": [[1, 2, 3, 4]], '
        '"defect_type": "bubbling", '
        '"reasoning": "surface blister"} '
        "suffix"
    )
    parsed = extract_defect_json(raw)

    assert parsed["coordinates"] == [[1, 2, 3, 4]]
    assert parsed["defect_type"] == "bubbling"
    assert parsed["reasoning"] == "surface blister"


def test_extract_defect_json_handles_fenced_flat_coordinates() -> None:
    raw = (
        "```json\n"
        '{"coordinates": [10, 20, 30, 40], '
        '"defect_type": "Bubbling", '
        '"reasoning": "Visible blister region"}'
        "\n```"
    )

    parsed = extract_defect_json(raw)
    assert parsed["coordinates"] == [[10, 20, 30, 40]]


def test_extract_defect_json_accepts_integral_float_coordinates() -> None:
    raw = (
        '{"coordinates": [[1.0, 2.0, 3.0, 4.0]], '
        '"defect_type": "bubbling", '
        '"reasoning": "surface blister"}'
    )

    parsed = extract_defect_json(raw)
    assert parsed["coordinates"] == [[1, 2, 3, 4]]


def test_extract_defect_json_rejects_invalid_keys() -> None:
    raw = '{"coordinates": [[1, 2, 3, 4]], "label": "bubbling", "reasoning": "x"}'
    assert_raises(ValueError, extract_defect_json, raw, match="Unexpected JSON keys")


def test_qlora_config_payload_required_values() -> None:
    payload = qlora_config_payload(settings=QLoRASettings())

    assert payload["r"] == 8
    assert payload["lora_alpha"] == 32
    assert payload["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_qlora_settings_from_payload_valid() -> None:
    payload = {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    settings = qlora_settings_from_payload(payload)
    assert settings.r == 16
    assert settings.lora_alpha == 64
    assert settings.lora_dropout == 0.1
    assert settings.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_qlora_settings_from_payload_missing_keys() -> None:
    payload = {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 64,
        # missing lora_dropout, bias, target_modules
    }
    assert_raises(
        ValueError, qlora_settings_from_payload, payload, match="Missing required"
    )


def test_qlora_settings_from_payload_invalid_target_modules() -> None:
    payload = {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj"],  # missing v_proj, o_proj
    }
    assert_raises(
        ValueError,
        qlora_settings_from_payload,
        payload,
        match="target_modules must include",
    )


def test_qlora_settings_from_payload_invalid_ranges() -> None:
    # invalid dropout
    payload = {
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 64,
        "lora_dropout": 1.1,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    assert_raises(
        ValueError, qlora_settings_from_payload, payload, match="lora_dropout"
    )

    # invalid r
    payload["lora_dropout"] = 0.1
    payload["r"] = 0
    assert_raises(ValueError, qlora_settings_from_payload, payload, match="r must be")


def test_load_qlora_config_valid(tmp_path: Path) -> None:
    config_path = tmp_path / "qlora.json"
    settings = QLoRASettings(r=16, lora_alpha=64)
    save_qlora_config(config_path, settings)

    loaded = load_qlora_config(config_path)
    assert loaded.r == 16
    assert loaded.lora_alpha == 64
    assert loaded.bias == "none"


def test_contrastive_loss_shape_mismatch_raises() -> None:
    student = torch.randn(2, 8)
    teacher = torch.randn(2, 16)

    assert_raises(
        ValueError,
        contrastive_distillation_loss,
        student,
        teacher,
        match="same shape",
    )


def test_distillation_train_step_runs_with_dummy_teacher() -> None:
    torch.manual_seed(7)
    images = torch.rand(2, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    teacher = DummyTeacher(embedding_dim=16)
    student = TinyCNNStudent(embedding_dim=16)
    trainer = ContrastiveDistillationTrainer(
        teacher=teacher,
        student=student,
        learning_rate=1e-3,
        device="cpu",
    )

    loss = trainer.train_step(images=images)
    assert loss > 0.0
    assert torch.isfinite(torch.tensor(loss))


def test_qwen_teacher_encoder_lora_wrapping() -> None:
    mock_loader = MagicMock()
    with (
        patch(
            "src.models.distillation._resolve_model_loader", return_value=mock_loader
        ),
        patch("src.models.distillation.get_peft_model") as mock_get_peft,
        patch("src.models.distillation.load_qlora_config"),
        patch("src.models.distillation.build_lora_config"),
        patch("src.models.distillation.AutoProcessor.from_pretrained"),
    ):
        mock_model = MagicMock()
        mock_loader.from_pretrained.return_value = mock_model

        # Enabled
        QwenTeacherEncoder(
            teacher_lora_enabled=True, teacher_lora_config_path=Path("dummy.json")
        )
        assert mock_get_peft.called

        # Disabled
        mock_get_peft.reset_mock()
        QwenTeacherEncoder(teacher_lora_enabled=False)
        assert not mock_get_peft.called


def test_qwen_teacher_encoder_trainable_parameters() -> None:
    mock_loader = MagicMock()
    mock_peft_model = MagicMock()
    with (
        patch(
            "src.models.distillation._resolve_model_loader", return_value=mock_loader
        ),
        patch("src.models.distillation.get_peft_model", return_value=mock_peft_model),
        patch("src.models.distillation.load_qlora_config"),
        patch("src.models.distillation.build_lora_config"),
        patch("src.models.distillation.AutoProcessor.from_pretrained"),
    ):
        # Mock model parameters
        param1 = MagicMock(spec=torch.nn.Parameter)
        param1.requires_grad = False
        param2 = MagicMock(spec=torch.nn.Parameter)
        param2.requires_grad = False
        mock_peft_model.named_parameters.return_value = [
            ("base_layer", param1),
            ("lora_layer", param2),
        ]
        mock_peft_model.parameters.return_value = [param1, param2]
        mock_loader.from_pretrained.return_value = MagicMock()

        # Trainable True
        encoder = QwenTeacherEncoder(
            teacher_lora_enabled=True,
            teacher_lora_config_path=Path("dummy.json"),
            teacher_lora_trainable=True,
        )
        assert len(encoder.trainable_parameters()) > 0

        # Trainable False
        encoder = QwenTeacherEncoder(
            teacher_lora_enabled=True,
            teacher_lora_config_path=Path("dummy.json"),
            teacher_lora_trainable=False,
        )
        assert len(encoder.trainable_parameters()) == 0

    class TrainableDummyTeacher(DummyTeacher):
        def __init__(self, embedding_dim: int, trainable: bool = True) -> None:
            super().__init__(embedding_dim)
            self.trainable = trainable
            self.param = torch.nn.Parameter(torch.randn(1, embedding_dim))

        def trainable_parameters(self) -> list[torch.nn.Parameter]:
            return [self.param] if self.trainable else []

        @property
        def teacher_lora_lr(self) -> float:
            return 5e-5

    def test_contrastive_trainer_optimizer_wiring() -> None:
        # Trainable teacher
        teacher = TrainableDummyTeacher(embedding_dim=16, trainable=True)
        student = TinyCNNStudent(embedding_dim=16)
        trainer = ContrastiveDistillationTrainer(teacher=teacher, student=student)

        # Initial param values
        initial_teacher_param = teacher.param.clone()

        images = torch.rand(2, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        trainer.train_step(images=images)

        # Check if teacher param changed
        assert not torch.equal(teacher.param, initial_teacher_param)

        # Non-trainable teacher
        teacher = TrainableDummyTeacher(embedding_dim=16, trainable=False)
        student = TinyCNNStudent(embedding_dim=16)
        trainer = ContrastiveDistillationTrainer(teacher=teacher, student=student)

        # Initial param values
        initial_teacher_param = teacher.param.clone()

        trainer.train_step(images=images)

        # Check if teacher param changed
        assert torch.equal(teacher.param, initial_teacher_param)

    def test_save_distillation_checkpoint_stores_metadata() -> None:
        mock_teacher = MagicMock()
        mock_teacher.lora_metadata = {
            "enabled": True,
            "trainable": True,
            "config_path": "test.json",
            "lr": 1e-4,
            "target_modules": ["q_proj", "k_proj"],
        }
        student = TinyCNNStudent(embedding_dim=16)
        history = [0.1]

        with patch("torch.save") as mock_save:
            save_distillation_checkpoint(student, mock_teacher, history, Path("."))
            args, _ = mock_save.call_args
            payload = args[0]
            assert payload["teacher_lora_metadata"] == mock_teacher.lora_metadata

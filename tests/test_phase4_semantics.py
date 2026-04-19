import torch

from src.models.distillation import (
    ContrastiveDistillationTrainer,
    TinyCNNStudent,
    contrastive_distillation_loss,
)
from src.models.qlora_config import QLoRASettings, qlora_config_payload
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


def test_distillation_train_step_rejects_prompt_mismatch() -> None:
    images = torch.rand(2, 3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    teacher = DummyTeacher(embedding_dim=16)
    student = TinyCNNStudent(embedding_dim=16)
    trainer = ContrastiveDistillationTrainer(
        teacher=teacher,
        student=student,
        learning_rate=1e-3,
        device="cpu",
    )

    assert_raises(
        ValueError,
        trainer.train_step,
        images,
        ["only one prompt"],
        match="Prompts length must match",
    )

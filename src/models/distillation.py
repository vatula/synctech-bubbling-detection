from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from src.data.loader import BubblingDataset
from src.data.transforms import get_train_transforms
from src.utils.logger import get_logger, setup_project

log = get_logger("distillation")


class TeacherEncoder(Protocol):
    def encode_images(
        self, images: torch.Tensor, prompts: list[str]
    ) -> torch.Tensor: ...


def _resolve_model_loader() -> type[Any]:
    import transformers

    candidate_names = [
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    for candidate in candidate_names:
        loader = getattr(transformers, candidate, None)
        if loader is not None:
            log.info("Resolved teacher loader", loader_class=candidate)
            return cast(type[Any], loader)

    msg = "No compatible transformers loader found for Qwen2.5-VL"
    raise RuntimeError(msg)


class QwenTeacherEncoder:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt_template: str = (
            "Analyze bubbling defects and summarize anomaly semantics for"
            " downstream student distillation."
        ),
    ) -> None:
        self.prompt_template = prompt_template
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        loader = _resolve_model_loader()
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = cast(Any, loader).from_pretrained(model_id, **model_kwargs)
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id)
        log.info(
            "Initialized QwenTeacherEncoder",
            model_id=model_id,
            device=str(self.device),
            dtype=str(dtype),
        )

    def _to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image = tensor.detach().cpu()
        if image.ndim != 3:
            msg = f"Expected image tensor with shape [C,H,W], got {tuple(image.shape)}"
            raise ValueError(msg)

        if float(image.min()) < 0.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean

        image = image.clamp(0.0, 1.0)
        return tvf.to_pil_image(image)

    def encode_images(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        if images.ndim != 4:
            msg = f"Expected images shape [B,C,H,W], got {tuple(images.shape)}"
            raise ValueError(msg)

        if images.shape[0] != len(prompts):
            msg = "Prompts length must match batch size"
            raise ValueError(msg)

        pil_images = [self._to_pil(img) for img in images]
        messages: list[list[dict[str, Any]]] = []
        for image, prompt in zip(pil_images, prompts, strict=True):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        prompt_texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        inputs = self.processor(
            text=prompt_texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        model_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            model_inputs[key] = (
                value.to(self.device) if isinstance(value, torch.Tensor) else value
            )

        with torch.no_grad():
            outputs = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            msg = "Teacher model did not return hidden states"
            raise RuntimeError(msg)

        last_hidden = hidden_states[-1]
        teacher_embeddings = F.normalize(last_hidden[:, -1, :], dim=-1)
        return teacher_embeddings.detach()


class TinyCNNStudent(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(images)
        return F.normalize(embeddings, dim=-1)


def build_student(
    architecture: Literal["cnn", "vit_tiny"],
    embedding_dim: int,
) -> nn.Module:
    if architecture == "cnn":
        return TinyCNNStudent(embedding_dim=embedding_dim)

    vit = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=embedding_dim,
    )

    class ViTTinyStudent(nn.Module):
        def __init__(self, backbone: nn.Module) -> None:
            super().__init__()
            self.backbone = backbone

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            logits = self.backbone(images)
            return F.normalize(logits, dim=-1)

    return ViTTinyStudent(vit)


def contrastive_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if student_embeddings.shape != teacher_embeddings.shape:
        msg = (
            "Student and teacher embeddings must have the same shape, "
            "got "
            f"{tuple(student_embeddings.shape)} and {tuple(teacher_embeddings.shape)}"
        )
        raise ValueError(msg)

    aligned_teacher = teacher_embeddings.to(
        device=student_embeddings.device,
        dtype=student_embeddings.dtype,
    )

    student = F.normalize(student_embeddings, dim=-1)
    teacher = F.normalize(aligned_teacher, dim=-1)

    logits = torch.matmul(student, teacher.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_student = F.cross_entropy(logits, labels)
    loss_teacher = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_student + loss_teacher)


class ContrastiveDistillationTrainer:
    def __init__(
        self,
        teacher: TeacherEncoder,
        student: nn.Module,
        learning_rate: float = 1e-4,
        temperature: float = 0.07,
        device: str | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.student.to(self.device)
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)

    def _default_prompts(self, batch_size: int) -> list[str]:
        return [
            "Describe bubbling defect semantics and visual anomaly patterns."
            for _ in range(batch_size)
        ]

    def train_step(
        self, images: torch.Tensor, prompts: list[str] | None = None
    ) -> float:
        self.student.train()
        images = images.to(self.device)
        batch_size = int(images.shape[0])
        effective_prompts = prompts or self._default_prompts(batch_size)

        if len(effective_prompts) != batch_size:
            msg = "Prompts length must match image batch size"
            raise ValueError(msg)

        with torch.no_grad():
            teacher_embeddings = self.teacher.encode_images(images, effective_prompts)

        self.optimizer.zero_grad(set_to_none=True)
        student_embeddings = self.student(images)

        loss = contrastive_distillation_loss(
            student_embeddings=student_embeddings,
            teacher_embeddings=teacher_embeddings,
            temperature=self.temperature,
        )
        loss.backward()
        self.optimizer.step()

        loss_value = float(loss.detach().item())
        log.info("Distillation train step", loss=loss_value, batch_size=batch_size)
        return loss_value

    def fit(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, int]],
        epochs: int,
    ) -> list[float]:
        history: list[float] = []
        for epoch in range(epochs):
            epoch_losses: list[float] = []
            for images, _labels in dataloader:
                epoch_losses.append(self.train_step(images=images))

            epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            history.append(epoch_loss)
            log.info("Distillation epoch completed", epoch=epoch + 1, loss=epoch_loss)

        return history


def save_distillation_checkpoint(
    student: nn.Module,
    history: list[float],
    output_dir: str | Path,
) -> Path:
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "student_distillation.pt"

    payload: dict[str, object] = {
        "student_state_dict": student.state_dict(),
        "loss_history": history,
        "epochs": len(history),
    }
    torch.save(payload, checkpoint_path)
    log.info(
        "Saved distillation checkpoint",
        path=str(checkpoint_path),
        epochs=len(history),
    )
    return checkpoint_path


def build_distillation_dataloader(
    nominal_dir: str | Path,
    bubbling_dir: str | Path,
    batch_size: int = 4,
    image_size: int = 224,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, int]]:
    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=get_train_transforms(image_size=image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def main() -> None:
    setup_project()

    dataloader = build_distillation_dataloader(
        nominal_dir="resources/assignment/hard-negatives-bubbling",
        bubbling_dir="resources/assignment/train-bubbling",
        batch_size=2,
    )

    teacher = QwenTeacherEncoder()

    warmup_images, _ = next(iter(dataloader))
    with torch.no_grad():
        warmup_embeddings = teacher.encode_images(
            warmup_images,
            ["Summarize bubbling defect semantics."] * int(warmup_images.shape[0]),
        )

    student = build_student(
        architecture="vit_tiny",
        embedding_dim=warmup_embeddings.shape[1],
    )
    trainer = ContrastiveDistillationTrainer(teacher=teacher, student=student)
    history = trainer.fit(dataloader=dataloader, epochs=1)
    save_distillation_checkpoint(
        student=student,
        history=history,
        output_dir="results/phase5/distillation",
    )


if __name__ == "__main__":
    main()

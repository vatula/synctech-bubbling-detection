from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.distillation import (
    REPARAM_COSINE_THRESHOLD,
    FastViTStudent,
    contrastive_distillation_loss,
)

# Uses FastViT-T8 default spatial resolution for compatibility with timm backbone.
# Expected range: valid model input resolution supported by the selected backbone.
# Impact if changed: backbone output shape and benchmark comparability may shift.
FASTVIT_TEST_IMAGE_SIZE = 224


def test_fastvit_student_matches_teacher_embedding_dimension() -> None:
    teacher_embedding_dim = 512
    model = FastViTStudent(
        teacher_embedding_dim=teacher_embedding_dim,
        image_size=FASTVIT_TEST_IMAGE_SIZE,
    )
    images = torch.randn(2, 3, FASTVIT_TEST_IMAGE_SIZE, FASTVIT_TEST_IMAGE_SIZE)

    with torch.no_grad():
        output = model(images)

    assert output.shape == (2, teacher_embedding_dim)


def test_reparameterization_preserves_output_similarity() -> None:
    model = FastViTStudent(
        teacher_embedding_dim=256,
        image_size=FASTVIT_TEST_IMAGE_SIZE,
    )
    model.eval()
    torch.manual_seed(7)
    images = torch.randn(2, 3, FASTVIT_TEST_IMAGE_SIZE, FASTVIT_TEST_IMAGE_SIZE)

    with torch.no_grad():
        before = model(images)
        model.apply_structural_reparameterization()
        after = model(images)

    cosine = F.cosine_similarity(before, after, dim=-1).mean().item()
    assert cosine >= REPARAM_COSINE_THRESHOLD


def test_training_step_smoke_has_finite_loss_and_gradients() -> None:
    model = FastViTStudent(
        teacher_embedding_dim=128,
        image_size=FASTVIT_TEST_IMAGE_SIZE,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    images = torch.randn(2, 3, FASTVIT_TEST_IMAGE_SIZE, FASTVIT_TEST_IMAGE_SIZE)
    student_embeddings = model(images)
    teacher_embeddings = F.normalize(torch.randn_like(student_embeddings), dim=-1)

    loss = contrastive_distillation_loss(student_embeddings, teacher_embeddings)
    assert torch.isfinite(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)

    optimizer.step()

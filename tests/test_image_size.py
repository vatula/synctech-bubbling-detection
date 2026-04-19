import pytest

from src.utils.image_size import (
    DEFAULT_IMAGE_SIZE,
    IMAGE_SIZE_ENV_KEY,
    MAX_IMAGE_SIZE,
    MIN_IMAGE_SIZE,
    get_default_image_size,
    resolve_image_size,
)


def test_default_image_size_within_bounds() -> None:
    assert MIN_IMAGE_SIZE <= DEFAULT_IMAGE_SIZE <= MAX_IMAGE_SIZE


def test_resolve_image_size_accepts_min_and_max() -> None:
    assert resolve_image_size(MIN_IMAGE_SIZE) == MIN_IMAGE_SIZE
    assert resolve_image_size(MAX_IMAGE_SIZE) == MAX_IMAGE_SIZE


def test_resolve_image_size_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="must be within"):
        resolve_image_size(MIN_IMAGE_SIZE - 1)

    with pytest.raises(ValueError, match="must be within"):
        resolve_image_size(MAX_IMAGE_SIZE + 1)


def test_get_default_image_size_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IMAGE_SIZE_ENV_KEY, str(MAX_IMAGE_SIZE))
    assert get_default_image_size() == MAX_IMAGE_SIZE


def test_get_default_image_size_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(IMAGE_SIZE_ENV_KEY, "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        get_default_image_size()


def test_get_default_image_size_rejects_out_of_range_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(IMAGE_SIZE_ENV_KEY, str(MAX_IMAGE_SIZE + 1))
    with pytest.raises(ValueError, match="must be within"):
        get_default_image_size()

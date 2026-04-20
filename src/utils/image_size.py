from __future__ import annotations

import os

from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_SIZE_ENV_KEY = "BUBBLING_IMAGE_SIZE"
MIN_IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 512
DEFAULT_IMAGE_SIZE = 224


def validate_image_size(image_size: int, *, source: str = "image_size") -> int:
    if image_size < MIN_IMAGE_SIZE or image_size > MAX_IMAGE_SIZE:
        msg = (
            f"{source} must be within [{MIN_IMAGE_SIZE}, {MAX_IMAGE_SIZE}], "
            f"got {image_size}"
        )
        raise ValueError(msg)
    return image_size


def get_default_image_size() -> int:
    raw_value = os.getenv(IMAGE_SIZE_ENV_KEY)
    if raw_value is None:
        return DEFAULT_IMAGE_SIZE

    try:
        parsed = int(raw_value)
    except ValueError as error:
        msg = f"{IMAGE_SIZE_ENV_KEY} must be an integer, got {raw_value!r}"
        raise ValueError(msg) from error

    resolved = validate_image_size(parsed, source=IMAGE_SIZE_ENV_KEY)
    log.info("Resolved image size from environment", image_size=resolved)
    return resolved


def resolve_image_size(image_size: int | None) -> int:
    if image_size is None:
        return get_default_image_size()
    return validate_image_size(image_size)

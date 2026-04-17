import numpy as np

from src.models.localization import AnomalyLocalizer


def test_overlay_results_draws_scaled_boxes() -> None:
    localizer = object.__new__(AnomalyLocalizer)
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    heatmap = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)

    overlay = localizer.overlay_results(
        image=image_rgb,
        heatmap=heatmap,
        _mask=mask,
        boxes=[[10, 10, 20, 20]],
    )

    assert overlay.shape == image_rgb.shape
    assert np.array_equal(overlay[20, 20], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(overlay[40, 40], np.array([0, 255, 0], dtype=np.uint8))


def test_overlay_results_clips_out_of_bounds_boxes() -> None:
    localizer = object.__new__(AnomalyLocalizer)
    image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    heatmap = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 100), dtype=np.uint8)

    overlay = localizer.overlay_results(
        image=image_rgb,
        heatmap=heatmap,
        _mask=mask,
        boxes=[[-5, -5, 120, 60]],
    )

    assert np.array_equal(overlay[0, 0], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(overlay[99, 199], np.array([0, 255, 0], dtype=np.uint8))

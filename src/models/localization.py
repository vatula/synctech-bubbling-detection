from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Dinomaly

from src.utils.logger import get_logger

log = get_logger("localization")


class AnomalyLocalizer:
    """
    Anomaly localization module using a trained Dinomaly model.
    Extracts heatmaps, masks, and bounding boxes.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ) -> None:
        """
        Initializes the localizer with a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.ckpt).
            device: Device to run inference on.
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(
            "Initializing AnomalyLocalizer",
            checkpoint=str(self.checkpoint_path),
            device=self.device,
        )

        # Load model and engine
        self.model = Dinomaly.load_from_checkpoint(
            self.checkpoint_path, weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        self._architecture = (
            f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
        )
        self._encoder_name = self._resolve_encoder_name()

        # Engine is needed for predict
        self.engine = Engine(devices=1 if self.device.type == "cuda" else 0)

    def _resolve_encoder_name(self) -> str:
        candidate_fields = ("backbone", "encoder_name", "encoder", "backbone_name")

        for field_name in candidate_fields:
            value = getattr(self.model, field_name, None)
            if isinstance(value, str) and value:
                return value

        hparams = getattr(self.model, "hparams", None)
        if isinstance(hparams, dict):
            for field_name in candidate_fields:
                value = hparams.get(field_name)
                if isinstance(value, str) and value:
                    return value
        elif hparams is not None:
            for field_name in candidate_fields:
                value = getattr(hparams, field_name, None)
                if isinstance(value, str) and value:
                    return value

        return self.model.__class__.__name__

    def get_provenance(self, score_threshold: float = 0.5) -> dict[str, str | float]:
        return {
            "architecture": self._architecture,
            "encoder_name": self._encoder_name,
            "checkpoint_path": str(self.checkpoint_path),
            "score_threshold": score_threshold,
        }

    def process_image(
        self, image_path: str | Path, threshold: float = 0.5
    ) -> dict[str, Any]:
        """
        Processes a single image and extracts localization data.

        Args:
            image_path: Path to the input image.
            threshold: Threshold for segmentation mask generation.

        Returns:
            Dictionary containing 'heatmap', 'mask', 'boxes', and 'score'.
        """
        # Engine.predict returns a list of results (usually one per batch)
        results = self.engine.predict(
            model=self.model,
            data_path=str(image_path),
            return_predictions=True,
            ckpt_path=str(self.checkpoint_path),
        )

        # results is typically a list of ImageBatch
        if not results:
            msg = "No results returned from prediction"
            raise RuntimeError(msg)

        predictions = cast(ImageBatch, results[0])

        # anomaly_map is typically (1, H, W)
        anomaly_map_tensor = predictions.anomaly_map
        if anomaly_map_tensor is None:
            msg = "Anomaly map is None"
            raise ValueError(msg)

        anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()  # (H, W)

        pred_score_tensor = predictions.pred_score
        if pred_score_tensor is None:
            msg = "Prediction score is None"
            raise ValueError(msg)

        pred_score = float(pred_score_tensor.item())

        normalized_map = np.clip(anomaly_map, 0.0, 1.0)

        # Normalize heatmap for visualization [0, 255]
        heatmap = cv2.applyColorMap(
            (normalized_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Generate binary mask
        mask = (normalized_map > threshold).astype(np.uint8) * 255

        # Extract bounding boxes from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])

        return {
            "heatmap": heatmap,
            "mask": mask,
            "boxes": boxes,
            "score": pred_score,
        }

    @staticmethod
    def _map_boxes_to_target_size(
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

    def overlay_results(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        _mask: np.ndarray,
        boxes: list[list[int]],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlays localization results onto the original image.

        Args:
            image: Original image in RGB.
            heatmap: Thermal heatmap in BGR.
            mask: Binary mask.
            boxes: List of [x1, y1, x2, y2] coordinates.
            alpha: Blending factor for heatmap.

        Returns:
            Overlayed image in RGB.
        """
        # Resize heatmap and mask to original image size if they differ
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        # Convert RGB image to BGR for opencv operations
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Blending heatmap
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_resized, alpha, 0)

        map_h, map_w = heatmap.shape[:2]
        mapped_boxes = self._map_boxes_to_target_size(
            boxes=boxes,
            source_width=map_w,
            source_height=map_h,
            target_width=w,
            target_height=h,
        )

        # Draw bounding boxes in image-space coordinates
        for box in mapped_boxes:
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Return as RGB
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def main() -> None:
    """
    Main entry point for Phase 3 localization verification.
    """
    from src.utils.logger import setup_project

    setup_project()

    # Find the latest checkpoint
    checkpoint_dir = Path("results/Dinomaly/bubbling/latest/weights/lightning")
    checkpoint_path = checkpoint_dir / "model.ckpt"

    if not checkpoint_path.exists():
        log.error("Checkpoint not found", path=str(checkpoint_path))
        return

    localizer = AnomalyLocalizer(checkpoint_path=checkpoint_path)

    # Process some images from the validation/test set
    test_dir = Path("resources/assignment/train-bubbling")
    output_dir = Path("results/localization_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = list(test_dir.glob("*.png"))[:5]
    for img_path in test_images:
        log.info("Processing image", path=str(img_path))

        # We need original image for overlaying
        image = cv2.imread(str(img_path))
        if image is None:
            log.error("Failed to load image", path=str(img_path))
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = localizer.process_image(img_path)

        overlay = localizer.overlay_results(
            image_rgb, results["heatmap"], results["mask"], results["boxes"]
        )

        # Save results
        base_name = img_path.stem
        cv2.imwrite(
            str(output_dir / f"{base_name}_overlay.png"),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(output_dir / f"{base_name}_heatmap.png"), results["heatmap"])
        log.info("Saved results for", image=base_name, score=results["score"])


if __name__ == "__main__":
    main()

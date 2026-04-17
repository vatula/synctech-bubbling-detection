from collections.abc import Sequence
from os import cpu_count
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np
import torch
from anomalib.data import ImageBatch, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Dinomaly
from torch.utils.data import ConcatDataset, DataLoader

from src.utils.logger import get_logger

log = get_logger("localization")


class LocalizationOutput(TypedDict):
    heatmap: np.ndarray
    mask: np.ndarray
    boxes: list[list[int]]
    score: float


class AnomalyLocalizer:
    """
    Anomaly localization module using a trained Dinomaly model.
    Extracts heatmaps, masks, and bounding boxes.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        predict_batch_size: int = 8,
        predict_num_workers: int | None = None,
    ) -> None:
        """
        Initializes the localizer with a trained checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint (.ckpt).
            device: Device to run inference on.
        """
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.predict_batch_size = predict_batch_size
        self.predict_num_workers = self._resolve_default_num_workers(
            configured_workers=predict_num_workers
        )
        log.info(
            "Initializing AnomalyLocalizer",
            checkpoint=str(self.checkpoint_path),
            device=self.device,
            predict_batch_size=self.predict_batch_size,
            predict_num_workers=self.predict_num_workers,
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
        self.engine = Engine(
            devices=1 if self.device.type == "cuda" else "auto",
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    @staticmethod
    def _resolve_default_num_workers(configured_workers: int | None) -> int:
        if configured_workers is not None:
            if configured_workers < 0:
                msg = "predict_num_workers must be non-negative"
                raise ValueError(msg)
            return configured_workers

        available = max(1, (cpu_count() or 1) - 1)
        return min(8, available)

    @staticmethod
    def _build_output(
        anomaly_map: np.ndarray, threshold: float, score: float
    ) -> LocalizationOutput:
        normalized_map = np.clip(anomaly_map, 0.0, 1.0)

        heatmap = cv2.applyColorMap(
            (normalized_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        mask = (normalized_map > threshold).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[list[int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])

        return {
            "heatmap": heatmap,
            "mask": mask,
            "boxes": boxes,
            "score": score,
        }

    def _parse_prediction_batch(
        self, predictions: ImageBatch, threshold: float
    ) -> list[LocalizationOutput]:
        anomaly_map_tensor = predictions.anomaly_map
        if anomaly_map_tensor is None:
            msg = "Anomaly map is None"
            raise ValueError(msg)

        pred_score_tensor = predictions.pred_score
        if pred_score_tensor is None:
            msg = "Prediction score is None"
            raise ValueError(msg)

        anomaly_map_tensor = anomaly_map_tensor.detach().cpu()
        pred_score_tensor = pred_score_tensor.detach().cpu()

        if anomaly_map_tensor.ndim == 2:
            anomaly_map_tensor = anomaly_map_tensor.unsqueeze(0)
        if pred_score_tensor.ndim == 0:
            pred_score_tensor = pred_score_tensor.unsqueeze(0)

        anomaly_maps = anomaly_map_tensor.numpy()
        pred_scores = pred_score_tensor.numpy()

        outputs: list[LocalizationOutput] = []
        for index in range(anomaly_maps.shape[0]):
            outputs.append(
                self._build_output(
                    anomaly_map=anomaly_maps[index],
                    threshold=threshold,
                    score=float(pred_scores[index]),
                )
            )

        return outputs

    def _create_predict_dataloader(
        self,
        image_paths: Sequence[Path],
        batch_size: int,
        num_workers: int,
    ) -> DataLoader[ImageBatch]:
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)
        if num_workers < 0:
            msg = "num_workers must be non-negative"
            raise ValueError(msg)

        datasets = [PredictDataset(path=image_path) for image_path in image_paths]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        collate_fn = datasets[0].collate_fn
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

    def process_images(
        self,
        image_paths: Sequence[str | Path],
        threshold: float = 0.5,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> list[LocalizationOutput]:
        resolved_paths = [
            Path(image_path).expanduser().resolve() for image_path in image_paths
        ]
        if not resolved_paths:
            msg = "image_paths cannot be empty"
            raise ValueError(msg)

        effective_batch_size = batch_size or self.predict_batch_size
        effective_num_workers = (
            self.predict_num_workers if num_workers is None else num_workers
        )

        dataloader = self._create_predict_dataloader(
            image_paths=resolved_paths,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
        )
        results = self.engine.predict(
            model=self.model,
            dataloaders=dataloader,
            return_predictions=True,
            ckpt_path=None,
        )

        if not results:
            msg = "No results returned from prediction"
            raise RuntimeError(msg)

        outputs: list[LocalizationOutput] = []
        for batch in results:
            outputs.extend(
                self._parse_prediction_batch(
                    predictions=cast(ImageBatch, batch),
                    threshold=threshold,
                )
            )

        if len(outputs) != len(resolved_paths):
            msg = "Prediction output count does not match input image count"
            raise RuntimeError(msg)

        return outputs

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
        self,
        image_path: str | Path,
        threshold: float = 0.5,
    ) -> LocalizationOutput:
        """
        Processes a single image and extracts localization data.

        Args:
            image_path: Path to the input image.
            threshold: Threshold for segmentation mask generation.

        Returns:
            Dictionary containing 'heatmap', 'mask', 'boxes', and 'score'.
        """
        return self.process_images(
            [image_path],
            threshold=threshold,
            batch_size=1,
            num_workers=0,
        )[0]

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

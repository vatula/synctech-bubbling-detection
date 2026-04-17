import json
import pickle
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC

from src.utils.logger import get_logger

log = get_logger("classifier")


class Metrics(TypedDict):
    """Dictionary for classification metrics."""

    accuracy: float
    auroc: float
    precision: float
    recall: float


class BubblingClassifier:
    """
    SVM Classifier for bubbling detection using LinearSVC and LOOCV.
    """

    def __init__(
        self, class_weight: str | dict[int, float] = "balanced", random_state: int = 42
    ) -> None:
        """
        Initializes the LinearSVC model.

        Args:
            class_weight: Weighting for classes (default 'balanced').
            random_state: Seed for reproducibility.
        """
        # dual="auto" is preferred in newer scikit-learn for robustness
        self.model = LinearSVC(
            class_weight=class_weight,
            random_state=random_state,
            dual="auto",
            max_iter=10000,
        )

    def run_loocv(self, features: np.ndarray, labels: np.ndarray) -> Metrics:
        """
        Runs Leave-One-Out Cross-Validation on the provided features and labels.

        Args:
            features: Feature matrix of shape (N, D).
            labels: Label vector of shape (N,).

        Returns:
            Dictionary of computed metrics.
        """
        loo = LeaveOneOut()
        y_true: list[int] = []
        y_pred: list[int] = []
        y_score: list[float] = []

        n_samples = len(labels)
        log.info("Starting Leave-One-Out Cross-Validation", iterations=n_samples)

        for train_index, test_index in loo.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            self.model.fit(X_train, y_train)

            # Predict label and get decision score for AUROC
            y_p = self.model.predict(X_test)
            y_s = self.model.decision_function(X_test)

            y_true.append(int(y_test[0]))
            y_pred.append(int(y_p[0]))
            # Decision function returns a 1D array for binary classification
            y_score.append(float(y_s[0]))

        # Compute metrics across all LOOCV iterations
        metrics: Metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auroc": float(roc_auc_score(y_true, y_score)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
        }

        log.info("LOOCV Validation Report", **metrics)
        return metrics

    def fit_full(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Fits the classifier on the full dataset for checkpoint persistence."""
        self.model.fit(features, labels)


def save_classifier_artifacts(
    classifier: BubblingClassifier,
    metrics: Metrics,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "linear_svc.pkl"
    metrics_path = artifact_dir / "loocv_metrics.json"

    with model_path.open("wb") as model_file:
        pickle.dump(classifier.model, model_file)

    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    log.info(
        "Saved classifier artifacts",
        model_path=str(model_path),
        metrics_path=str(metrics_path),
    )
    return model_path, metrics_path


def main() -> None:
    """
    Main entry point for running Phase 2 classification.
    """
    from src.data.loader import BubblingDataset
    from src.models.extractor import FeatureExtractor

    # Paths to data
    nominal_dir = Path("resources/assignment/hard-negatives-bubbling")
    bubbling_dir = Path("resources/assignment/train-bubbling")

    # Initialize Dataset and Extractor
    from src.data.transforms import get_inference_transforms

    dataset = BubblingDataset(
        nominal_dir=nominal_dir,
        bubbling_dir=bubbling_dir,
        transform=get_inference_transforms(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []

    log.info("Extracting features for all images")
    for i in range(len(dataset)):
        img, label = dataset[i]
        img_tensor = img.unsqueeze(0).to(device)

        features = extractor(img_tensor)
        all_features.append(features.cpu().numpy())
        all_labels.append(label)

    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)

    # Initialize and run LOOCV
    classifier = BubblingClassifier()
    metrics = classifier.run_loocv(X, y)
    classifier.fit_full(X, y)
    save_classifier_artifacts(
        classifier=classifier,
        metrics=metrics,
        output_dir="results/phase5/classifier",
    )


if __name__ == "__main__":
    from src.utils.logger import setup_project

    setup_project()
    main()

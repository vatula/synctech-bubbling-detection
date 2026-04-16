# Strategic Implementation Roadmap for Bubbling Defect Detection

## Executive Overview

This document serves as the authoritative, serialized execution plan for the AI engineering agent. It translates the high-level architectural synthesis into discrete, actionable implementation phases. The primary objective is to architect, train, and deploy a fault-tolerant, dual-pipeline visual inspection system (comprising SVM Binary Classification and Unsupervised Pixel-level Localization). Furthermore, the plan mandates the resolution of semantic reasoning gaps via the integration of a parameter-efficient Vision-Language Model (VLM) student-teacher distillation loop, culminating in hardware-optimized, bare-metal deployment.

## Phase 1: Core Infrastructure and Rigorous Dataset Engineering

**Objective:** Establish the foundational Python computational environments, algorithmically handle the severe 13/21 inverted class imbalance, and implement strictly controlled, non-destructive geometric augmentations to preserve specular highlight topologies.

* **Task 1.1: Environment and Dependency Initialization**
    * Initialize a strict, isolated virtual environment (venv or conda).
    * Install core dependencies via pip: `torch`, `torchvision`, `albumentations`, `scikit-learn`, `anomalib`, `onnx`, `tensorrt`, and `peft` (for QLoRA).
* **Task 1.2: Custom DataLoader Construction**
    * Develop a bespoke PyTorch `Dataset` class mapping natively to the provided directory structure.
    * Implement binary class labeling logic (0: Nominal Board, 1: Bubbling Defect).
* **Task 1.3: Non-Destructive Augmentation Pipeline Formulation**
    * Integrate the `Albumentations` library into the dataloader transform sequence.
    * Implement strictly constrained D4 dihedral geometric transformations (Explicitly: rigid rotations at 90, 180, 270 degrees, and horizontal/vertical flips).
    * Implement highly conservative brightness/contrast jitter (`limit=0.1`).
    * _Absolute Constraint:_ The agent must strictly prohibit and ensure the absence of `GaussianBlur`, `MotionBlur`, elastic warping, and color inversion transformations to mathematically protect the high-frequency specular highlight signatures.
    

## Phase 2: DINOv2 Feature Extraction and Hyperplane Optimization

**Objective:** Leverage the pre-trained ViT-L/14-registers foundation model to extract high-dimensional dense visual representations, subsequently fitting a statistically rigorous linear Support Vector Machine.

* **Task 2.1: Foundation Backbone Instantiation**
    * Load the `dinov2_vitl14_reg` model strictly via the `torch.hub.load` interface.
    * Transfer the model weights securely to the available CUDA device.
    * Enforce `.eval()` mode and strictly wrap all forward-pass inference scripts within a `with torch.no_grad():` context manager to permanently freeze the computational graph and eliminate gradient tracking VRAM overhead.
* **Task 2.2: High-Dimensional Feature Vector Engineering**
    * Iterate the augmented dataloader iteratively through the frozen DINOv2 backbone.
    * Extract the global `<CLS>` representation token.
    * Extract and mathematically average-pool the spatial sequence of patch tokens.
    * Concatenate the `<CLS>` token vector with the average-pooled patch token vector.
    * Apply rigorous $L2$-normalization to the final concatenated feature vector.
* **Task 2.3: Algorithmic Support Vector Classification**
    * Initialize the `sklearn.svm.LinearSVC` module.
    * Configure the parameter `class_weight='balanced'` to enforce inverse proportional weighting, mathematically countering the 13/21 dataset class imbalance and stabilizing the hyperplane.
* **Task 2.4: Leave-One-Out Cross-Validation (LOOCV) Execution**
    * Wrap the LinearSVC fitting and extraction process securely within an `sklearn.model_selection.LeaveOneOut` generator.
    * Execute exactly 34 independent training and validation loops.
    * Programmatically log terminal metrics: Aggregate Accuracy, AUROC, Precision, and Recall scores.

## Phase 3: Unsupervised Localization via Anomalib (Dinomaly Integration)

**Objective:** Implement the Dinomaly architecture within the Anomalib ecosystem to generate pixel-precise anomaly heatmaps utilizing strictly unsupervised learning on nominal data.

* **Task 3.1: YAML Configuration Engineering**
    * Generate a highly specific `dinomaly_config.yaml` file.
    * Define `model.class_path: anomalib.models.Dinomaly`.
    * Define `model.init_args.backbone: dinov2_vitl14`.
    * Define `model.init_args.attention_type: linear` to optimize throughput.
    * Define `model.init_args.dropout_rate: 0.1` to induce noise into the bottleneck.
    * Define `data.init_args.image_size: ` for standardizing embedding dimensions.
* **Task 3.2: Unsupervised Training Pipeline Execution**
    * Configure the datamodule pathing to map the 13 nominal images exclusively to the training subset.
    * Map the 21 anomalous bubbling images exclusively to the validation/test subset.
    * Execute the training routine via CLI: `anomalib train --config dinomaly_config.yaml`.
* **Task 3.3: Inference and Spatial Heatmap Generation**
    * Utilize the Lightning or Torch inferencer to process the test dataset.
    * Extract, save, and overlay the generated segmentation masks, bounding box coordinates, and thermal-scaled distance heatmaps onto the original imagery.

## Phase 4: Semantic Augmentation via VLM Distillation (Gap Resolution)

**Objective:** Address the identified lack of semantic reasoning and computational scalability in the original roadmap by fine-tuning Qwen2.5-VL and establishing a contrastive student-teacher knowledge transfer loop.

* **Task 4.1: QLoRA Parameter Configuration for Qwen2.5-VL**
    * Load the `Qwen/Qwen2.5-VL-3B-Instruct` model utilizing 4-bit normal float (`nf4`) quantization.
    * Configure the LoRA adapter parameters: `r=4`, `lora_alpha=16`.
    * Strictly target the self-attention matrices: `q_proj`, `k_proj`, `v_proj`.
* **Task 4.2: Structured Output Fine-Tuning**
    * Engineer a Pythonic distillation loop where the VLM (Teacher) infers across the full dataset to generate continuous soft labels and intermediate embedding representations.
    * Initialize a micro-scale CNN or ViT-Tiny (Student).
    * Train the Student model to regress toward the Teacher's soft representations utilizing a contrastive distillation loss function, establishing an edge-deployable, highly capable semantic model.

## Phase 5: Hardware Compilation and CI/CD Verification

**Objective:** Translate the high-level Python mathematical graphs into ultra-fast executable engines and automatically verify against all operational test constraints.

* **Task 5.1: ONNX Graph Serialization**
    * Trace and export the frozen DINOv2 backbone and the fitted SVM decision logic into the standard ONNX format, explicitly configuring dynamic batching axes.
    * Export the fully trained Dinomaly Anomalib model via the integrated CLI command: `anomalib export --export_type ONNX`.
* **Task 5.2: TensorRT FP16 Optimization Calibration**
    * Parse the serialized ONNX graphs utilizing the `trtexec` compiler.
    * Apply the `--fp16` flag to calibrate the network for half-precision execution, directly targeting Tensor Core matrix multiplication acceleration.
    * Benchmark and verify that inference latency falls within the mandated microsecond threshold.
* **Task 5.3: Requirement Verification and Assertion Validation**
    * Initialize and execute automated test suites strictly mapped against `TEST_REQUIREMENTS.md`.
    * Programmatically assert that the LOOCV loop executed exactly 34 times without deviation.
    * Programmatically assert that zero destructive augmentations (e.g., Gaussian blur) were applied to the generated training tensor batches.
    * Validate that the terminal diagnostic output correctly merges the binary classification, the pixel-level bounding boxes, and the JSON semantic description into a unified response payload.

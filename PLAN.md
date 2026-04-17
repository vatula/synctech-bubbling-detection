# Strategic Implementation Roadmap for Bubbling Defect Detection

## Executive Overview

This document serves as the authoritative, serialized execution plan for the AI engineering agent. It translates the high-level architectural synthesis into discrete, actionable implementation phases. The primary objective is to architect, train, and deploy a fault-tolerant, dual-pipeline visual inspection system (comprising SVM Binary Classification and Unsupervised Pixel-level Localization). Furthermore, the plan mandates the resolution of semantic reasoning gaps via the integration of a parameter-efficient Vision-Language Model (VLM) student-teacher distillation loop, culminating in hardware-optimized, bare-metal deployment.

## CRITICAL INSTRUCTION FOR AI AGENT:

* You must employ a **Micro-Experiment Methodology**. Before committing to writing full application logic or complex loops, you must write a tiny, isolated script to validate your assumptions (e.g., checking tensor shapes, validating GPU visibility, or testing a single inference pass). Furthermore, you must structure the application modularly. Do not write monolithic scripts. Separate data, models, and utilities into a reusable `src/` package.

## Phase 0: System Verification & Project Scaffolding [COMPLETED]

**Objective:** Establish the project structure and validate the ROCm/MIGraphX environment to prevent downstream hardware-acceleration blockers.

* **Task 0.1: Project Modularization Scaffolding**
    * Create the core repository structure: `src/data/`, `src/models/`, `src/utils/`, `tests/`, and `notebooks/` (for micro-experiments).
    * Initialize the strict Python 3.12 virtual environment via `uv` within the ROCm-optimized container (project Dockerfile).
* **Task 0.2: Micro-Experiment - ROCm & MIGraphX Validation**
    * _Deliverable:_ A script `verify_env.py`.
    * _Action:_ Assert `torch.cuda.is_available()` (mapped to HIP/ROCm).
    * _Action:_ Create a dummy linear PyTorch model, export it to ONNX (Opset 17 or 18), and compile it using `migraphx` to guarantee the compiler toolchain is functional before loading massive vision models.
* **Task 0.3: Hardware Optimization & Precision Tuning**
    * _Action:_ Perform micro-experiments to benchmark `torch.set_float32_matmul_precision` on AMD Matrix Cores.
    * _Implementation:_ Enforce `high` precision via `src/utils/env.py` and `setup_project()` to utilize Tensor/Matrix Cores and eliminate hardware-utilization warnings.

## Phase 1: Core Infrastructure and Modular Dataset Engineering [COMPLETED]

**Objective:** Build the foundational shared dataloaders and enforce the strict augmentation constraints required to preserve specular highlight topologies.

* **Task 1.1: Shared Dataloader Module (`src/data/loader.py`)**
    * Develop a PyTorch `Dataset` class handling the 13/21 inverted class imbalance.
    * Implement binary class labeling logic (0: Nominal, 1: Bubbling).
* **Task 1.2: Strict Augmentation Pipeline (`src/data/transforms.py`)**
    * Integrate `Albumentations`.
    * Implement strictly constrained D4 dihedral geometric transformations (90, 180, 270-degree rigid rotations, plus horizontal/vertical flips).
    * Implement conservative brightness/contrast jitter (`limit=0.1`).
* **Task 1.3: Micro-Experiment & Unit Testing**
   * _Deliverable:_ `tests/test_augmentations.py`
   * _Action:_ Write assertions to programmatically guarantee the absolute absence of `GaussianBlur`, `MotionBlur`, elastic warping, and color inversion within the transform composition.
   * _Action:_ Visualize one batch of augmented data to ensure specular highlights remain mathematically intact.

## Phase 2: DINOv2 Feature Extraction and Hyperplane Optimization [COMPLETED]

**Objective:** Leverage the frozen ViT-L/14-registers model for feature extraction and fit a cross-validated Support Vector Machine.

* **Task 2.1: Micro-Experiment - DINOv2 Shape Verification**
    * _Action:_ Load `dinov2_vitl14_reg` via PyTorch Hub to the ROCm GPU. Pass a single dummy tensor `(1,3,224,224)` and verify the output shapes of the `<CLS>` token and spatial patch tokens.
* **Task 2.2: Extraction Module (`src/models/extractor.py`)**
    * Implement the forward pass securely inside `with torch.no_grad():` and `.eval()` mode.
    * Concatenate the global `<CLS>` token with the average-pooled spatial patch tokens.
    * Apply $L2$-normalization to the final concatenated feature vector.
* **Task 2.3: SVM Classification & LOOCV (`src/models/classifier.py`)**
    * Initialize `sklearn.svm.LinearSVC(class_weight='balanced')`.
    * Implement the `LeaveOneOut` cross-validation loop (exactly 34 iterations).
    * _Deliverable:_ Log terminal metrics (Accuracy, AUROC, Precision, Recall) to a structured validation report.

## Phase 3: Unsupervised Localization via Anomalib (Dinomaly Integration)

**Objective:** Generate pixel-precise anomaly heatmaps utilizing unsupervised learning on nominal data.

* **Task 3.1: Micro-Experiment - Anomalib Data Ingestion**
    * _Action:_ Create the `dinomaly_config.yaml`.
    * _Action:_ Run a dry-run of the datamodule to ensure Anomalib correctly isolates the 13 normal images for training and routes the 21 bubbling images to validation.
* **Task 3.2: Unsupervised Training Pipeline Execution**
    * Configure the YAML: `model.class_path: anomalib.models.Dinomaly`, `backbone: dinov2_vitl14`, `attention_type: linear`, `dropout_rate: 0.1`.
    * Execute the training routine via the CLI or Python API.
* **Task 3.3: Inference and Heatmap Extraction (`src/models/localization.py`)**
   * Process the test dataset.
   * Extract, save, and overlay the generated segmentation masks, bounding box coordinates, and thermal-scaled distance heatmaps onto the original imagery.

## Phase 4: Semantic Augmentation via VLM Distillation

**Objective:** Fine-tune Qwen2.5-VL for semantic reasoning and set up the student-teacher knowledge transfer loop.

* **Task 4.1: Micro-Experiment - VLM Prompting & JSON Output**
     * _Action:_ Load `Qwen/Qwen2.5-VL-3B-Instruct` in BF16 precision.
     * _Action:_ Pass a single anomalous image through the base model to verify prompt formatting and ensure stable, structured JSON output (Coordinates, Defect Type, Reasoning) before initiating LoRA.
* **Task 4.2: QLoRA Parameter Configuration**
    * Configure LoRA adapter parameters: `r=8`, `lora_alpha=32` (increased for semantic depth).
    * Target self-attention matrices: `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
    * Fine-tune the model to standardize the semantic bounding box extraction.
* **Task 4.3: Contrastive Student-Teacher Orchestration (`src/models/distillation.py`)**
    * Engineer the orchestration script.
    * Teacher (VLM) generates continuous soft labels/embeddings across the dataset.
    * Initialize the Student (micro-scale CNN or ViT-Tiny).
    * Train the Student to regress toward the Teacher's representations using a contrastive distillation loss function.

## Phase 5: AMD MIGraphX Compilation & Unified CI/CD Verification

**Objective:** Translate the high-level Python graphs into ultra-fast FP16 executable engines and run end-to-end mission-critical assertions.

* **Task 5.1: ONNX Graph Serialization**
    * Trace and export the frozen DINOv2 backbone, the fitted SVM logic, and the Student semantic model to ONNX. Explicitly define dynamic batching axes.
    * Export the fully trained Dinomaly model using `anomalib export --export_type ONNX`.
* **Task 5.2: AMD MIGraphX Optimization and Calibration**
    * Parse the serialized ONNX graphs utilizing the `migraphx` compiler.
    * Apply FP16 quantization to calibrate the network, directly targeting AMD Matrix Core (CDNA/RDNA3) acceleration.
    * _Micro-Experiment:_ Profile a single compiled graph to verify inference latency falls within the microsecond threshold.
* **Task 5.3: Unified Inference API (`src/pipeline.py`)**
    * Create a single entry point class that ingests an image, routes it through the compiled MIGraphX engines (SVM → Dinomaly → Distilled VLM), and outputs a unified JSON response payload (Classification + Bounding Box + Semantic Text).
* **Task 5.4: Final Requirement Verification**
    * Execute the automated test suite against `TEST_REQUIREMENTS.md`.
    * Programmatically assert LOOCV executed exactly 34 times.
    * Verify edge-deployment latency and E2E fault tolerance.
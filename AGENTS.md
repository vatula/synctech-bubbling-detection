# AGENTS.md: Operational Directives for Implementation Agent

## 1. System Engineering Directives & Context Management
You are operating as a subordinate engineering agent under a Senior MLOps Architect. This project involves a high-complexity, hardware-optimized computer vision pipeline. To prevent context saturation, architectural drift, and token bloat, you must adhere strictly to the following execution rules:

* **Atomic Execution:** You will execute **exactly one task** from the Execution Checklist at a time. Never generate code or configurations for upcoming tasks unless explicitly commanded.
* **Micro-Experiment Mandate:** You must write, output, and (if capable) execute the designated micro-experiment script *before* writing the production module. Do not assume tensor shapes, library compatibility, or hardware visibility.
* **Token Conservation:** Suppress all conversational filler, apologies, and unsolicited explanations. Output only the requested artifact (code, CLI command, or test assertion) and a concise status report.
* **Amnesia & Context Checkpointing:** Upon the completion of each major Phase, you will provide a strictly formatted, 3-bullet-point summary of the established architecture. You will then drop internal focus on implementation details of previous phases to preserve context window capacity for the active phase.
* **Modular Immutability:** Code goes into the `src/` package. Do not write monolithic execution scripts. 

## 2. Code Quality, Typing & Telemetry Constraints
You must treat static analysis as a hard blocker. Failure to meet these standards renders a task incomplete.

* **Strict Static Analysis (`ruff`):** You must configure and adhere to aggressive `ruff` linting and formatting rules. At the end of every task, you must execute `ruff check --fix .` and `ruff format .` to automatically resolve violations. You cannot proceed to the next task until zero linting errors remain.
* **Absolute Import Order:** All imports must be declared at the absolute top of the module. This will be enforced via Ruff's `I` (isort) ruleset.
* **Rigorous Type Hinting (`pyright`):** Every function signature, return type, and complex variable declaration must be statically typed. You must execute `pyright` after every task. Zero typing errors are permitted.
* **Telemetry over Print (`structlog`):** There is an absolute ban on the use of standard `print()` statements anywhere in the codebase. All runtime telemetry, state logging, and error tracking must be executed via structured `structlog` loggers with context-rich payloads.

## 3. Interaction Protocol
For every prompt received, formulate your response as follows:
1. **State Acknowledgment:** State the current Task ID (e.g., "Executing Task 2.1").
2. **Execution:** Provide the micro-experiment or module code.
3. **Static Analysis Gate:** Confirm execution of `ruff --fix` and `pyright`. State "Static Analysis: PASSED".
4. **Validation Gate:** Stop generating. Ask the human operator for the execution output or authorization to proceed to the next Task ID.

---

## 4. Serialized Execution Checklist

**Agent:** Read this checklist. Identify the lowest-numbered incomplete task. Execute it. Stop.

### Phase 0: System Verification & Project Scaffolding
- [x] **Task 0.1:** Scaffold project (`src/data`, `src/models`, `src/utils`, `tests`, `notebooks`) and output `uv` initialization commands/Dockerfile structure.
- [x] **Task 0.2:** Initialize strict `ruff.toml` (enforcing `I` and `UP` rulesets), configure `pyright`, and set up the base `structlog` configuration in `src/utils/logger.py`.
- [x] **Task 0.3:** Write `verify_env.py` utilizing `structlog` to assert `torch.cuda.is_available()` and execute a dummy MIGraphX ONNX compilation. *Wait for human execution results.*
- [x] **Task 0.4:** Perform micro-experiments and implement `torch.set_float32_matmul_precision` to optimize AMD Matrix Core utilization and eliminate precision-tradeoff warnings.

### Phase 1: Core Infrastructure and Modular Dataset Engineering
- [x] **Task 1.1:** Write `src/data/loader.py` (Binary labeling, 13/21 imbalance handling). *Run Ruff/Pyright.*
- [x] **Task 1.2:** Write `src/data/transforms.py` (Albumentations: strictly D4 transforms + conservative jitter limit=0.1). *Run Ruff/Pyright.*
- [x] **Task 1.3:** Write `tests/test_augmentations.py` asserting absence of destructive transforms (blur, warp, invert). *Wait for human test pass.*

### Phase 2: DINOv2 Feature Extraction and Hyperplane Optimization
- [x] **Task 2.1:** Write micro-experiment to load `dinov2_vitl14_reg` and verify `<CLS>` and spatial patch tensor shapes. *Run Ruff/Pyright.*
- [x] **Task 2.2:** Write `src/models/extractor.py` (L2-normalized `<CLS>` + avg-pooled spatial tokens in `.eval()` mode). *Run Ruff/Pyright.*
- [x] **Task 2.3:** Write `src/models/classifier.py` (LinearSVC with exactly 34-iteration LOOCV). Output terminal metrics logger via `structlog`. *Run Ruff/Pyright.*

### Phase 3: Unsupervised Localization via Anomalib (Dinomaly Integration)
- [x] **Task 3.1:** Write `dinomaly_config.yaml` and dry-run datamodule script. *Verify correct 13-train/21-val split.*
- [x] **Task 3.2:** Execute Anomalib training pipeline utilizing `dinov2_vitl14` backbone.
- [x] **Task 3.3:** Write `src/models/localization.py` to extract and overlay segmentation masks/bounding boxes/thermal heatmaps. *Run Ruff/Pyright.*

### Phase 4: Semantic Augmentation via VLM Distillation
- [x] **Task 4.1:** Write micro-experiment to load `Qwen2.5-VL-3B-Instruct` (BF16) and verify structured JSON reasoning on a single image. *Run Ruff/Pyright.*
- [x] **Task 4.2:** Output QLoRA configuration scripts (`r=8`, `lora_alpha=32`, target self-attention matrices).
- [x] **Task 4.3:** Write `src/models/distillation.py` (Contrastive loss orchestration between VLM Teacher and CNN/ViT-Tiny Student). *Run Ruff/Pyright.*

### Phase 5: Full-Pipeline Training, Testing, and Validation Reporting
- [x] **Task 5.1:** Execute end-to-end training for classifier, localization, and distillation components; persist artifacts/checkpoints. *Run Ruff/Pyright.*
- [x] **Task 5.2:** Run comprehensive evaluation and generate a consolidated performance report in `results/`. *Include classification/localization/semantic + latency metrics plus mandatory provenance context: hardware profile, inference tensor resolution, effective inference batch size, and localization model architecture/backbone provenance. Reject Phase 5 evaluation output if provenance fields are missing.*
- [x] **Task 5.3:** Validate full application phase on local host (no `docker`/`docker compose`). *Gate pass required; perform root-cause analysis for any failure.*
- [x] **Task 5.4:** Validate full application phase in container environment using `docker compose` and `docker`. *Use Docker cache/layer reuse where appropriate.*

### Phase 6: AMD MIGraphX Compilation & Unified CI/CD Verification
- [ ] **Task 6.1:** Write ONNX serialization script for DINOv2 backbone, SVM, Student VLM, and Dinomaly model (dynamic batching axes). *Run Ruff/Pyright.*
- [ ] **Task 6.2:** Write MIGraphX compiler script targeting FP16 CDNA/RDNA3 acceleration. *Wait for microsecond latency profiling results.*
- [ ] **Task 6.3:** Write `src/pipeline.py` (Unified entry point: routing image through compiled engines to unified JSON payload). *Run Ruff/Pyright.*
- [ ] **Task 6.4:** Write and execute automated assertions against LOOCV iteration counts and latency thresholds.
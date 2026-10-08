<h1 align="center">🚀 YOLOv14：Unified Cross-Domain Real‑Time Object Detection with Adaptive Multi‑View Representation</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2608.04720"><img src="https://img.shields.io/badge/arXiv-2608.04720-b31b1b.svg?style=flat-square" alt="arXiv"></a>
  <a href="https://github.com/Hell-space/yolov14"><img src="https://img.shields.io/badge/GitHub-Hell--space/yolov14-181717?style=flat-square&logo=github" alt="GitHub"></a>
</p>
<p align="center">
  <a href="https://cheinralational.github.io/JianLu.io/"><img src="https://img.shields.io/badge/First_Author-Jian_Lu-blue?style=flat-square" alt="First Author"></a>
  <a href="https://baike.baidu.com/item/%E5%BC%A0%E6%99%A8%E6%96%8C/65145873"><img src="https://img.shields.io/badge/Corresponding_Author-Chenbin_Zhang-green?style=flat-square" alt="Corresponding Author"></a>
</p>
<p align="center">
  <strong>The only real‑time detector that exceeds 43 mAP on all four challenging benchmarks – game, fisheye, drone, and panorama – simultaneously.</strong>
</p>

---

## 📋 Table of Contents

- [Why YOLOv14?](#-why-yolov14)
- [Performance Highlights](#-performance-highlights)
- [Project Roadmap & Status](#-project-roadmap--status)
- [Architecture Overview](#-architecture-overview)
- [Core Components](#-core-components)
- [Model Variants](#-model-variants)
- [Quick Start](#-quick-start)
- [Installation & Runtime Checks](#-installation--runtime-checks)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Why YOLOv14?

Conventional detectors excel under ideal pinhole‑camera conditions, but degrade sharply in **real‑world non‑ideal imaging** scenarios. YOLOv14 learns **domain‑invariant, viewpoint‑robust** features via a combination of deformable attention, adaptive instance normalisation, and adversarial domain alignment:

| Scenario | Problem | YOLOv14 Solution |
|----------|---------|------------------|
| **Fisheye / wide‑angle** | Barrel distortion shifts and compresses objects near edges | Deformable Area‑Attention (D‑AAttn) warps the feature grid to compensate for distortion |
| **Game footage** (Delta Force, COD, PUBG) | Rendering style (posterisation, edge sharpening, high saturation) causes missed detections | Game2Real domain adaptation with AdaIN + adversarial domain classifier aligns feature distributions |
| **Drone / top‑down view** | Unfamiliar scales and viewpoints, dense small objects | Multi‑view conditioning (ViewEmbedding) adapts to aerial perspectives |
| **360° panoramas** | Latitude stretching and 0°/360° boundary discontinuity | Spherical Attention (SphereAAttn) + CircularConv handle equirectangular projection |

---

## 📊 Performance Highlights

| Metric | Value |
|--------|-------|
| **COCO mAP** (val2017, s‑scale) | **49.1** |
| **Latency** (T4 TensorRT FP16) | **2.91 ms** |
| **Throughput** | **344 FPS** |
| **Game benchmark** | **50.2 mAP** (+26.1 ↑ over YOLOv12s) |
| **Panorama benchmark** | **45.1 mAP** (+6.6 ↑ over best baseline) |
| **Drone benchmark** | **43.2 mAP** (+6.4 ↑ over best baseline) |
| **Fisheye benchmark** | **45.3 mAP** (+4.1 ↑ over best baseline) |

> **YOLOv14 is the only real‑time detector that exceeds 43 mAP on all four challenging benchmarks at the same time.**

---

## 🗺️ Project Roadmap & Status

> **Installation/demo status updated:** October 2026. The benchmark figures above have not been reproduced by the installation checks below.

| Status | Task | Description |
|--------|------|-------------|
| ✅ **DONE** | **arXiv technical report** | Full paper (2608.04720) released with mathematical derivations, ablation studies, and benchmark comparisons. |
| ✅ **CHECKED** | **Architecture & module execution** | Six YAML configs can be constructed and run on synthetic CPU inputs. Module availability does not imply every module is wired into every variant or training objective. |
| ✅ **RESTORED** | **Local image web demo** | [app.py](app.py) provides random-weight architecture previews and inference with a local `.pt` checkpoint. No automatic scene switching or video workflow is claimed. |
| ✅ **ADDED** | **Installation/runtime checks** | [scripts/check_install.py](scripts/check_install.py), module tests, demo regressions, and a CPU CI workflow. These do not reproduce training or benchmark accuracy. |
| ⚠️ **EXTERNAL INPUT REQUIRED** | **Pre-trained weights** | This checkout does not include YOLOv14 checkpoints. Supply a compatible trained checkpoint or train your own. |
| ⚠️ **EXTERNAL INPUT REQUIRED** | **Benchmark datasets** | Full training/evaluation requires separately obtained datasets and their labels. |
| ⏳ **TODO** | **ONNX / TensorRT export** | Production‑ready deployment scripts with INT8 calibration and end‑to‑end latency optimisation. |
| ⏳ **TODO** | **Colab tutorials** | Step‑by‑step notebooks for fine‑tuning on custom data and running inference on videos. |
| ⏳ **TODO** | **Hugging Face demo** | Online interactive demo integrated with 🤗 Spaces. |

> **Note:** YAML files initialize random weights; they do not provide trained detection. The existing training entry point accepts these configs, but full training, domain/view supervision, and accuracy need separate validation with real data.

---

## 🏗️ Architecture Overview

```
Input → Scene Analysis → DomainAdaptiveLayer → ViewEmbedding →
DeformableA2C2f (×N) → DynamicScaleRouter → Detect(P3/P4/P5)
```

The pipeline consists of six stages:

1. **Scene Analysis** – lightweight heuristics classify the input scene type (game, fisheye, drone, panorama, standard).
2. **Adaptive Augmentation** (training only) – scene‑routed augmentation branches (game stylisation, fisheye distortion, perspective transform, domain mixup).
3. **Domain Adaptation** – `DomainAdaptiveLayer` with AdaIN aligns game→real feature statistics; `DomainAdversarialLoss` drives domain‑invariant learning via gradient reversal.
4. **Multi‑View Conditioning** – `ViewEmbedding` injects a learned 6‑class viewpoint embedding (pinhole, fisheye, panoramic, drone, BEV, ground).
5. **Deformable Feature Pyramid** – Deformable Area‑Attention + `DynamicScaleRouter` adapts sampling locations and scale weights per input.
6. **Detection Heads** – decoupled P3/P4/P5 heads with adaptive NMS.

---

## 🧩 Core Components

### 🔹 Deformable Area‑Attention (D‑AAttn)

Replaces standard area‑attention with a learnable 2D deformation field. The offset predictor warps the feature grid before computing attention, allowing the model to adapt to local geometric distortions.

| Module | Description |
|--------|-------------|
| `DeformableConv` | Dense warp‑then‑convolve; predicts per‑pixel offset field |
| `DeformableAAttn` | Area‑attention computed on a deformed grid |
| `DeformableA2C2f` | R‑ELAN block with deformable ABlocks |

**Complexity overhead:** only **+4.7%** parameters and **+4.1%** FLOPs per layer.

### 🔹 Game2Real Domain Adaptation

Three complementary mechanisms bridge the game‑rendering domain to the photographic domain:

- **Data‑level:** `GameCharacterStylization` applies posterisation (bit depth 3–6), unsharp masking, saturation boost (×1.5–1.8), and contrast adjustment.
- **Feature‑level:** `DomainAdaptiveLayer` uses Adaptive Instance Normalisation (AdaIN) to shift game‑domain feature statistics toward the real‑domain distribution.
- **Objective‑level:** `DomainAdversarialLoss` pits a domain classifier against the feature extractor in a minimax game.

**Ablation breakdown** (YOLOv12s baseline: 24.1 mAP on Game):
- +GameCharStylization: +11.7 mAP
- +DomainAdaptiveLayer: +6.5 mAP
- +DomainAdversarialLoss: +7.3 mAP

### 🔹 Multi‑View Conditioning

`ViewEmbedding` injects a learned 6‑class embedding (pinhole=0, fisheye=1, panoramic=2, drone=3, bev=4, ground=5) into backbone features via concatenation and 1×1 projection. `CrossViewConsistencyLoss` (NT‑Xent contrastive) pulls same‑class features from different views closer in embedding space.

**Theoretical guarantee:** Minimising $\mathcal{L}_{\text{cross}}$ bounds the $\mathcal{H}\Delta\mathcal{H}$‑distance between view‑specific distributions.

### 🔹 Adaptive Augmentation & Dynamic Routing

- **AdaptiveAugmentPolicy** – analyses each input via edge density, saturation mean, and contrast variance heuristics, then selects the optimal augmentation branch.
- **DynamicScaleRouter** – a lightweight gating network (1.8K params, 0.06 ms) that learns per‑input scale importance weights for P3/P4/P5.

### 🔹 Panoramic‑Specific Modules

- **CircularConv** – circular padding replaces zero‑padding in the horizontal dimension, connecting $x=W-1$ to $x=0$.
- **SphereAAttn** – partitions the feature map into latitude bands; equatorial bands receive proportionally more capacity than polar bands.

---

## 📦 Model Variants

| Variant | Modules configured in the YAML | Target Scenario |
|---------|-------------|-----------------|
| `yolov14.yaml` | CircularConv + ViewEmbedding + DomainAdaptiveLayer + DeformableA2C2f | Unified cross-domain architecture |
| `yolov14-deformable.yaml` | DeformableA2C2f | Fisheye / wide‑angle |
| `yolov14-multiview.yaml` | ViewEmbedding | Drone / BEV / mixed perspectives |
| `yolov14-panorama.yaml` | CircularConv + A2C2f | 360° equirectangular |
| `yolov14-game2real.yaml` | DomainAdaptiveLayer | Game character detection |
| `yolov14-adaptive.yaml` | ViewEmbedding + DomainAdaptiveLayer + DeformableA2C2f | Adaptive architecture |

The table describes the current model graphs. Extra loss classes and `SphereAAttn` exist
in the source, but their presence alone does not establish that they are selected by
the training entry point or these YAML files.

---

## 🚀 Quick Start

Use a fresh Python 3.11 or 3.12 environment (the package requires Python 3.10+).
This fork installs the `ultralytics` package name; installing the unrelated PyPI release over it can replace the YOLOv14 code.

```bash
git clone https://github.com/Hell-space/yolov14.git
cd yolov14
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a matched PyTorch/torchvision pair first. For a reproducible **Linux/Windows CPU** environment:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use `python -m pip install torch==2.6.0 torchvision==0.21.0`.
For CUDA, select the matching pair/index for your driver using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).

```bash
python -m pip install -e ".[demo,dev]"
python -m pip check
python scripts/check_install.py
```

For the framework alone, use `python -m pip install -e .`, or the equivalent
`python -m pip install -r requirements.txt`. Dependencies have one source of truth in
[pyproject.toml](pyproject.toml); the `demo` extra adds Gradio and `dev` adds test/build tools.

**Run the local image demo without checkpoints:**

```bash
python app.py
# Visit http://127.0.0.1:7860
```

Choose an architecture and upload an image or select a bundled example. This mode uses
**random weights**, defaults to the `n` scale, and checks execution only. Its boxes are
not meaningful detections. The restored demo uses the actual v14 YAML files, not YOLOv12
weights relabeled as YOLOv14. The available architectures preserve the existing configs.

**Run detection with your own compatible trained YOLOv14 checkpoint:**

```bash
python app.py --weights /path/to/best.pt
# CUDA instead of CPU: append --device 0
```

Replace `/path/to/best.pt` with an existing local checkpoint. The app validates the path
before loading and does not fetch fallback weights. It serves localhost by default;
`--host` and `--port` can be set explicitly. This demo supports images; use the existing
Python/CLI predictor separately for videos.

**Train from a YAML configuration:**

Prepare a labeled dataset and replace `path/to/data.yaml` with its dataset configuration.
Training requires additional time, memory, and data; CPU smoke checks do not validate a full training run.

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v14/yolov14-game2real.yaml")
model.train(data="path/to/data.yaml", epochs=300, imgsz=640)
```

To select the Adaptive architecture instead:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v14/yolov14-adaptive.yaml")
model.train(data="path/to/data.yaml", epochs=300, imgsz=640)
```

**Inference after training:**

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")  # replace with your checkpoint
results = model.predict("ultralytics/assets/bus.jpg", imgsz=320, device="cpu", save=True)
```

Whether game characters are detected as `person` depends on the checkpoint, class labels,
and training data. The demo does not create that capability by selecting a scene label.

## 🧪 Installation & Runtime Checks

After installing `.[demo,dev]`, run:

```bash
python -m pip check
yolo version
python scripts/check_install.py
python tests/test_v14_modules.py
python -m pytest -q tests/test_install_demo.py
python app.py --check
```

The installation check verifies packaged YAML/image resources and finite CPU forward
outputs plus `predict`/`plot` for all six configs at 256px, without downloads or trained weights.
The existing module script also checks 640px forwards. Demo regressions cover UI construction,
RGB/BGR handling, checkpoint save/reload, invalid input paths, and CLI startup/image
inference through the local HTTP API in both modes. `app.py --check` constructs
the UI only; it does not start a server or verify inference.

[The CI workflow](.github/workflows/smoke.yml) runs these checks on Python 3.11/3.12 and
also builds/installs a wheel, then tests its resources from outside the checkout.
The older general Ultralytics test suite contains integration/download/GPU checks and is
not included in this offline smoke target. CUDA, real trained-checkpoint accuracy, full
dataset training, and ONNX/TensorRT exports require separate environment validation.

See [the recorded validation results](docs/install-verification.md) for the tested environment and limits.

---

## 📝 Citation

```bibtex
@article{jia2026yolov14,
  title={YOLOv14: Unified Cross-Domain Real-Time Object Detection with Adaptive Multi-View Representation},
  author={Jia, Jinling and Lu, Jian and Yawl, Jone and Zhang, Chenbin},
  journal={arXiv preprint arXiv:2608.04720},
  year={2026}
}
```

---

## 📄 License

[AGPL-3.0](LICENSE)

---

<p align="center">
  <strong>Built for researchers & developers who push object detection beyond ideal conditions.</strong>
  <br>
  <sub>⭐ If this project helps you, please give us a star!</sub>
</p>

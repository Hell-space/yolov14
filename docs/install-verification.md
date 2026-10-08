# Installation and demo validation — 2026-10-08

Base: `Hell-space/yolov14` main at `3716b953ab8cc7403b2e6e2e37ed37972b3fff5b`.
No files under `ultralytics/`, including the model graphs and algorithm implementations,
were modified. `LICENSE` was restored verbatim from the first parent of the base commit
(Git blob `0ad25db4bd1d86c452db3f9602ccdbe172438f52`).

## Environment

- Linux x86_64, Python 3.12.14, CPU only.
- PyTorch 2.6.0+cpu / torchvision 0.21.0+cpu.
- Gradio 5.50.0, NumPy 2.5.2, OpenCV 5.0.0.93, pytest 9.1.1.
- Tests used `YOLO_OFFLINE=True`, `YOLO_AUTOINSTALL=False`, and one OpenMP thread.
- The runner injects SOCKS proxy environment variables without HTTPX's optional SOCKS
  support. Those variables were unset for the local demo tests, and Gradio analytics
  were disabled. No application proxy settings were changed.

## Results

| Check | Result |
| --- | --- |
| Editable install with `.[demo,dev]` | Passed in a new virtual environment after installing the matched CPU PyTorch pair |
| `python -m pip check` | No broken requirements |
| `yolo version` | Reports 8.3.63, matching the existing framework version |
| `python scripts/check_install.py` | All six configs passed finite 256px CPU forwards, `predict`, and `plot` |
| `python tests/test_v14_modules.py` | 32 checks passed, zero failed, including all six configs at 640px |
| `python -m pytest -q tests/test_install_demo.py` | 10 tests passed, including both local HTTP demo modes |
| `python app.py --check` | UI built without model inference or server startup |
| `python -m build` | Source distribution and wheel built successfully |
| Installed wheel from outside the repository | All six config forward/predict/plot checks passed; import came from `site-packages` |
| Bundled resources | Wheel contains six v14 YAML files, default config, bus/zidane images, and original license |

The HTTP tests start `app.py` as a subprocess from a different working directory,
check the localhost page, upload the bundled bus image via the Gradio client, and verify
the returned image dimensions and model-status message. They cover random-weight
preview and loading a locally saved **synthetic, untrained** checkpoint. The checkpoint
round trip checks serialization and loading only. It is not a trained accuracy test.

The restored app converts RGB uploads to Ultralytics' BGR array convention and converts
annotations back to RGB. It validates checkpoint paths before loading and never
falls back to downloading YOLOv12 weights. Its architecture options map to the existing
YOLOv14 configs. Without a checkpoint, both the UI and result status disclose random weights.

## Validation still needed

- Python 3.11 and platform coverage through the added CI matrix; local execution used 3.12.
- Windows/macOS, CUDA/FlashAttention, GPU memory and performance.
- A compatible, genuinely trained YOLOv14 checkpoint and real labeled datasets.
- Full training, auxiliary domain/view loss integration, and the README benchmark figures.
- ONNX/TensorRT or other export backends and end-to-end video workflows.

The original model implementation and YAML graphs are preserved. This change repairs
packaging, the image demo, and the documented installation/runtime checks; it does not
establish that every research component is connected to the training objective.

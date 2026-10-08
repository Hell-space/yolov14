"""Check installed YOLOv14 resources and CPU inference without external weights/data."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=256, help="Synthetic image size (multiple of 32, at least 256)")
    args = parser.parse_args(argv)
    if args.imgsz < 256 or args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32 and at least 256")

    # A smoke check must not silently install dependencies or download weights.
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    import numpy as np
    import torch

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS, ROOT

    print(f"Python {platform.python_version()}, torch {torch.__version__}")
    print(f"ultralytics {importlib.metadata.version('ultralytics')}: {ultralytics.__file__}")
    assert (ROOT / "cfg/default.yaml").is_file(), "Installed default configuration is missing"
    assert (ASSETS / "bus.jpg").is_file(), "Installed demo image is missing"
    configs = sorted((ROOT / "cfg/models/v14").glob("*.yaml"))
    assert len(configs) == 6, f"Expected six YOLOv14 configurations, found {len(configs)}"
    failures = []
    for config in configs:
        try:
            model = YOLO(str(config), task="detect").to("cpu")
            model.model.eval()
            with torch.inference_mode():
                prediction = model.model(torch.zeros(1, 3, args.imgsz, args.imgsz))
            decoded = prediction[0] if isinstance(prediction, tuple) else prediction
            assert torch.isfinite(decoded).all(), "Non-finite forward output"
            image = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            result = model.predict(image, imgsz=args.imgsz, device="cpu", conf=1.0, verbose=False, save=False)[0]
            assert result.orig_shape == image.shape[:2]
            assert result.plot().shape == image.shape
            print(f"PASS {config.name}: finite forward output and predict/plot")
        except Exception as exc:  # noqa: BLE001 - report all variants before returning a failure
            failures.append(config.name)
            print(f"FAIL {config.name}: {type(exc).__name__}: {exc}")
    print(f"YOLOv14 installation smoke check: {len(configs) - len(failures)} passed, {len(failures)} failed.")
    print("Random weights check execution only; no accuracy or training claim is made.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

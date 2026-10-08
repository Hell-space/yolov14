"""Local YOLOv14 image demo. YAML mode checks execution, not detection accuracy."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils import ASSETS, ROOT

CONFIG_DIR = ROOT / "cfg" / "models" / "v14"
MODEL_CONFIGS = {
    "Unified": CONFIG_DIR / "yolov14.yaml",
    "Adaptive": CONFIG_DIR / "yolov14-adaptive.yaml",
    "Game2Real": CONFIG_DIR / "yolov14-game2real.yaml",
    "Deformable": CONFIG_DIR / "yolov14-deformable.yaml",
    "Multi-view": CONFIG_DIR / "yolov14-multiview.yaml",
    "Panorama": CONFIG_DIR / "yolov14-panorama.yaml",
}
PREVIEW_NOTICE = (
    "Architecture preview: randomly initialized YOLOv14 weights. "
    "This verifies execution only; detections are not meaningful. "
    "Restart with --weights /path/to/best.pt for trained inference."
)


def validate_checkpoint(weights: str | Path) -> Path:
    """Require a local checkpoint so missing paths never trigger a download."""
    path = Path(weights).expanduser().resolve()
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"Checkpoint must be an existing local .pt file: {path}")
    return path


@lru_cache(maxsize=1)
def load_model(path: str) -> YOLO:
    """Keep only the most recently used model in memory."""
    # Model construction runs a stride-discovery forward. The demo never trains,
    # so avoid retaining its autograd graph in the domain-adaptation caches.
    with torch.no_grad():
        return YOLO(path, task="detect")


def predict_image(image, variant="Unified", imgsz=320, conf=0.25, *, weights=None, device="cpu"):
    """Accept RGB input and return RGB annotations plus an explicit model status."""
    if image is None:
        raise ValueError("Choose an image first.")
    if weights is None:
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown YOLOv14 variant: {variant}")
        model_path = MODEL_CONFIGS[variant]
        status = f"{PREVIEW_NOTICE}\nConfiguration: {model_path.name}"
    else:
        model_path = validate_checkpoint(weights)
        status = f"Local checkpoint: {model_path.name}"

    # Gradio supplies RGB. Ultralytics interprets NumPy arrays as BGR.
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    result = load_model(str(model_path)).predict(
        source=bgr, imgsz=int(imgsz), conf=float(conf), device=device,
        save=False, verbose=False,
    )[0]
    annotated = np.ascontiguousarray(result.plot()[:, :, ::-1])
    return annotated, f"{status}\nReturned boxes: {len(result.boxes)}"


def build_app(*, weights=None, device="cpu"):
    """Construct the UI without loading weights or running inference."""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        if exc.name != "gradio":
            raise
        raise RuntimeError('Install demo dependencies: python -m pip install -e ".[demo]"') from exc

    checkpoint = validate_checkpoint(weights) if weights is not None else None
    notice = PREVIEW_NOTICE if checkpoint is None else f"Using local checkpoint: {checkpoint.name}"
    with gr.Blocks(title="YOLOv14 local image demo", analytics_enabled=False) as demo:
        gr.Markdown(f"# YOLOv14 local image demo\n\n{notice}")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", image_mode="RGB", label="Input image")
                variant = gr.Dropdown(
                    choices=list(MODEL_CONFIGS), value="Unified",
                    label="Architecture (random weights)", interactive=checkpoint is None,
                    visible=checkpoint is None,
                )
                imgsz = gr.Slider(256, 640, value=320, step=32, label="Image size")
                conf = gr.Slider(0.01, 1.0, value=0.25, step=0.01, label="Confidence threshold")
                run = gr.Button("Run image", variant="primary")
            with gr.Column():
                output = gr.Image(type="numpy", label="Output image")
                status = gr.Textbox(value=notice, label="Model status", interactive=False)

        def infer(image, variant, imgsz, conf):
            try:
                return predict_image(image, variant, imgsz, conf, weights=checkpoint, device=device)
            except (ValueError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc

        run.click(infer, inputs=[image, variant, imgsz, conf], outputs=[output, status], api_name="predict_image")
        examples = [[str(ASSETS / name)] for name in ("bus.jpg", "zidane.jpg") if (ASSETS / name).is_file()]
        if examples:
            gr.Examples(examples=examples, inputs=[image], cache_examples=False)
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Existing local YOLOv14 .pt checkpoint; otherwise preview random weights")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or 0 for CUDA")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--check", action="store_true", help="Build the demo without starting a server")
    args = parser.parse_args(argv)
    try:
        demo = build_app(weights=args.weights, device=args.device)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.check:
        print("YOLOv14 demo UI check passed (no model inference performed).")
        return 0
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

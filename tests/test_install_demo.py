"""Offline regressions for the installation and local demo paths."""

import importlib.metadata
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from PIL import Image

# app.py is a checkout entry point; the framework itself must be installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MODEL_CONFIGS, build_app, load_model, predict_image, validate_checkpoint
from ultralytics.utils import ASSETS, ROOT


def test_installed_resources():
    assert importlib.metadata.version("ultralytics")
    assert (ROOT / "cfg/default.yaml").is_file()
    assert (ASSETS / "bus.jpg").is_file()
    assert len(MODEL_CONFIGS) == 6
    assert all(path.is_file() for path in MODEL_CONFIGS.values())


def test_demo_builds_without_loading_models(monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Building the UI must not load a model or download weights")

    monkeypatch.setattr("app.load_model", unexpected_load)
    demo = build_app()
    assert demo.config["dependencies"]
    demo.close()


def test_rgb_round_trip_and_preview_notice():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = 220
    image[:, :, 1] = 30
    image[:, :, 2] = 10
    annotated, status = predict_image(image, imgsz=256, conf=1.0)
    np.testing.assert_array_equal(annotated, image)
    assert "randomly initialized" in status
    assert "yolov14.yaml" in status


def test_checkpoint_save_reload_predict(tmp_path):
    # Synthetic checkpoint exercises serialization/loading, not trained accuracy.
    load_model.cache_clear()
    checkpoint = tmp_path / "synthetic.pt"
    load_model(str(MODEL_CONFIGS["Unified"])).save(checkpoint)
    load_model.cache_clear()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    annotated, status = predict_image(image, imgsz=256, conf=1.0, weights=checkpoint)
    assert annotated.shape == image.shape
    assert "synthetic.pt" in status
    assert "randomly initialized" not in status


@pytest.mark.parametrize("filename", ["missing.pt", "missing.yaml", "https://example.com/weights.pt"])
def test_invalid_checkpoint_does_not_load_or_download(filename, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("An invalid checkpoint must be rejected before model loading")

    monkeypatch.setattr("app.load_model", unexpected_load)
    with pytest.raises(ValueError, match="existing local .pt"):
        predict_image(np.zeros((256, 256, 3), dtype=np.uint8), weights=filename)
    with pytest.raises(ValueError):
        validate_checkpoint(filename)


def test_missing_image():
    with pytest.raises(ValueError, match="Choose an image"):
        predict_image(None)


@pytest.mark.parametrize("with_checkpoint", [False, True])
def test_cli_http_inference(tmp_path, with_checkpoint):
    """Exercise the documented CLI through an actual local Gradio server/API."""
    from gradio_client import Client, handle_file

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    args = [sys.executable, str(app_path), "--port", str(port)]
    if with_checkpoint:
        checkpoint = tmp_path / "synthetic.pt"
        load_model.cache_clear()
        load_model(str(MODEL_CONFIGS["Unified"])).save(checkpoint)
        args.extend(["--weights", str(checkpoint)])

    server_log = tmp_path / "server.log"
    with server_log.open("w") as output:
        process = subprocess.Popen(args, cwd=tmp_path, stdout=output, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(server_log.read_text())
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        else:
            pytest.fail("Server did not start: " + server_log.read_text())
        assert "YOLOv14" in response.text
        client = Client(url, verbose=False)
        output_image, status = client.predict(
            handle_file(str(ASSETS / "bus.jpg")), "Unified", 256, 1.0, api_name="/predict_image",
        )
        assert Image.open(output_image).size == Image.open(ASSETS / "bus.jpg").size
        assert ("randomly initialized" in status) == (not with_checkpoint)
        if with_checkpoint:
            assert "synthetic.pt" in status
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

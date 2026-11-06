"""Small CPU-only regressions for the maintained project layout."""
from __future__ import annotations

import os
from pathlib import Path
import pickle
import subprocess
import sys

from PIL import Image
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_independent_of_working_directory(tmp_path, monkeypatch):
    from fire_audit.config import get_configured_data_yaml_path, get_user_config_path
    monkeypatch.delenv("CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert get_user_config_path() == ROOT / "configs" / "runtime.yaml"
    assert get_configured_data_yaml_path() == ROOT / "configs" / "data.yaml"


@pytest.mark.parametrize("invalid", [False, True])
def test_dataset_verification_finds_bad_labels_without_modifying_them(tmp_path, invalid):
    from fire_detection.dataset import DatasetVerifier
    for split in ("train", "val", "test"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir()
        Image.new("RGB", (32, 24)).save(tmp_path / split / "images" / "sample.png")
        label = "0 0.5 0.5 0.25 0.25\n"
        if split == "val" and invalid:
            label = "7 0.5 0.5 0.25 0.25\n"
        (tmp_path / split / "labels" / "sample.txt").write_text(label, encoding="utf-8")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(dict(path=str(tmp_path), train="train/images", val="val/images", test="test/images", nc=2, names={0: "fire", 1: "smoke"})), encoding="utf-8")
    label = tmp_path / "val" / "labels" / "sample.txt"
    original = label.read_bytes()
    result = DatasetVerifier().verify(config)
    assert result.status == ("FAIL" if invalid else "PASS")
    assert result.total_images == 3
    assert label.read_bytes() == original


def test_wavelet_round_trip_and_legacy_class_reference():
    from fire_detection.wavelet.haar import haar_dwt2, haar_idwt2
    from fire_detection.wavelet.backbone import HaarResidual
    from src.modules.dwt.backbone import HaarResidual as OldHaarResidual
    value = torch.randn(1, 3, 17, 19)
    torch.testing.assert_close(haar_idwt2(haar_dwt2(value), value.shape[-2:]), value)
    assert OldHaarResidual is HaarResidual
    old_name = HaarResidual.__module__
    try:
        HaarResidual.__module__ = "src.modules.dwt.backbone"
        encoded = pickle.dumps(HaarResidual)
    finally:
        HaarResidual.__module__ = old_name
    assert pickle.loads(encoded) is HaarResidual


def test_detector_forward_shape_without_pretrained_downloads():
    from fire_detection.models.detector import MobileNetV4_PAN
    model = MobileNetV4_PAN(nc=2, pretrained=False, neck_channels=32).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    predictions = output[0] if isinstance(output, tuple) else output
    assert predictions.shape == (1, 6, 84)
    assert torch.isfinite(predictions).all()


@pytest.mark.parametrize("name", ["train", "evaluate", "infer", "run_pipeline", "verify_dataset", "warning"])
def test_entry_point_help_from_outside_repository(tmp_path, name):
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled", YOLO_CONFIG_DIR=str(tmp_path / "settings"))
    process = subprocess.run([sys.executable, str(ROOT / "scripts" / (name + ".py")), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "usage:" in process.stdout.lower()

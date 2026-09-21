"""
Transfer Learning Training Pipeline for Fire and Smoke Detection.
Loads the pretrained checkpoint and dataset configured in the config.yaml file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

try:
    from src.fire_audit.config import get_configured_data_yaml_path, get_training_config
except ImportError:
    from fire_audit.config import get_configured_data_yaml_path, get_training_config

logger = logging.getLogger(__name__)

def setup_wandb(env_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Load Weights & Biases API key from .env file and authenticate.

    Looks for 'wandb_api' in root .env file, sets WANDB_API_KEY, and calls wandb.login().
    """
    if env_path is None:
        workspace_env = Path.cwd() / ".env"
        pkg_root_env = Path(__file__).resolve().parent.parent.parent / ".env"
        env_path = workspace_env if workspace_env.is_file() else pkg_root_env

    env_path = Path(env_path).resolve()
    api_key = None

    # 1. Try python-dotenv
    try:
        import dotenv

        if env_path.is_file():
            dotenv.load_dotenv(dotenv_path=env_path)
            api_key = os.getenv("wandb_api")
    except ImportError:
        pass

    # 2. Fallback to direct parsing
    if not api_key and env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("wandb_api="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception as e:
            logger.warning(f"Could not parse {env_path}: {e}")

    # 3. Authenticate with W&B
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        try:
            import wandb

            wandb.login(key=api_key)
            logger.info("Weights & Biases authenticated successfully using 'wandb_api' from .env")
            return True
        except Exception as e:
            logger.warning(f"Failed to log in to Weights & Biases: {e}")
            return False
    else:
        logger.warning(
            f"'wandb_api' key not found in {env_path} or environment. Training will proceed without authenticated W&B sync."
        )
        return False

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, defaulting on None, NaN, Inf, or type conversion errors."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def validate_checkpoint_architecture(weights_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect and validate pretrained YOLO checkpoint."""
    p = Path(weights_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Pretrained weights checkpoint not found: {p}")

    # Load with weights_only=False for PyTorch 2.6+ compatibility
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint {p} does not contain valid Ultralytics model dict")

    model_obj = ckpt["model"]
    yaml_info = getattr(model_obj, "yaml", {}) if hasattr(model_obj, "yaml") else {}
    nc = getattr(model_obj, "nc", ckpt.get("nc", yaml_info.get("nc", 2) if isinstance(yaml_info, dict) else 2))
    names = getattr(model_obj, "names", ckpt.get("names", {}))

    info = {
        "checkpoint_path": str(p),
        "nc": nc,
        "names": names,
        "yaml_file": yaml_info.get("yaml_file", "") if isinstance(yaml_info, dict) else "",
        "scale": yaml_info.get("scale", "") if isinstance(yaml_info, dict) else "",
    }
    logger.info(f"Loaded source checkpoint architecture: {info}")
    return info


def train_yolo(
    weights: Optional[Union[str, Path]] = None,
    data: Optional[Union[str, Path]] = None,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    lr0: Optional[float] = None,
    lrf: Optional[float] = None,
    patience: Optional[int] = None,
    device: Optional[Union[str, int]] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    workers: Optional[int] = None,
    optimizer: Optional[str] = None,
    seed: Optional[int] = None,
    freeze: Optional[int] = None,
    save_period: Optional[int] = None,
    exist_ok: Optional[bool] = None,
    val: Optional[bool] = None,
    amp: Optional[bool] = None,
    verbose: Optional[bool] = None,
    extra_train_args: Optional[Dict[str, Any]] = None,
    wandb_project: Optional[str] = "home-fire-detection",
    wandb_name: Optional[str] = None,
    use_wandb: bool = True,
) -> Dict[str, Any]:
    """
    Run transfer learning fine-tuning using pretrained weights.

    Returns:
        Dict containing training summary, paths to best.pt and last.pt checkpoints, and metrics.
    """
    settings = get_training_config()
    weights = weights if weights is not None else settings.get("weights")
    data = data if data is not None else settings.get("data") or get_configured_data_yaml_path()
    if weights is None:
        raise ValueError("Training weights are not configured; set training.weights in the config.yaml")
    if data is None:
        raise ValueError("Dataset path is not configured; set data_path in the config.yaml")

    def setting(name: str, value: Any, fallback: Any) -> Any:
        return value if value is not None else settings.get(name, fallback)

    epochs = setting("epochs", epochs, 50)
    batch = setting("batch", batch, 16)
    imgsz = setting("imgsz", imgsz, 640)
    lr0 = setting("lr0", lr0, 0.001)
    lrf = setting("lrf", lrf, 0.01)
    patience = setting("patience", patience, 10)
    device = setting("device", device, "cuda:0" if torch.cuda.is_available() else "cpu")
    project = setting("project", project, "runs/train")
    name = setting("name", name, "yolov8n")
    workers = setting("workers", workers, 4)
    optimizer = setting("optimizer", optimizer, "AdamW")
    seed = setting("seed", seed, 42)
    freeze = setting("freeze", freeze, None)
    save_period = setting("save_period", save_period, -1)
    exist_ok = setting("exist_ok", exist_ok, True)
    val = setting("val", val, True)
    amp = setting("amp", amp, True)
    verbose = setting("verbose", verbose, True)

    weights_path = Path(weights).resolve()
    data_path = Path(data).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found at: {weights_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config YAML not found at: {data_path}")

    # Validate hyperparameters
    if epochs < 1:
        raise ValueError(f"epochs must be a positive integer, got {epochs}")
    if batch < 1 and batch != -1:
        raise ValueError(f"batch must be a positive integer or -1 (auto-batch), got {batch}")
    if imgsz < 32:
        raise ValueError(f"imgsz must be at least 32, got {imgsz}")
    if lr0 <= 0.0:
        raise ValueError(f"Initial learning rate lr0 must be positive, got {lr0}")
    if lrf < 0.0:
        raise ValueError(f"Final learning rate factor lrf must be non-negative, got {lrf}")
    if patience < 0:
        raise ValueError(f"Early stopping patience must be non-negative, got {patience}")
    if workers < 0:
        raise ValueError(f"Number of dataloader workers must be non-negative, got {workers}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if save_period < -1:
        raise ValueError(f"save_period must be >= -1, got {save_period}")
    if freeze is not None and freeze < 0:
        raise ValueError(f"freeze layer count must be non-negative, got {freeze}")

    # Inspect source checkpoint
    try:
        validate_checkpoint_architecture(weights_path)
    except Exception as e:
        logger.warning(f"Source checkpoint architecture validation note: {e}")

    # Determine execution device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    device_str = str(device)

    logger.info(f"Initializing YOLO model from checkpoint: {weights_path}")
    logger.info(f"Target dataset configuration: {data_path}")
    logger.info(f"Target device: {device_str} (CUDA available: {torch.cuda.is_available()})")

    # Load model from checkpoint
    model = YOLO(str(weights_path))

    # Keep relative or posix format to prevent Windows backslash/colon collisions in WandB logger
    project_path = Path(project)
    project_str = project_path.as_posix()

    train_kwargs: Dict[str, Any] = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "lr0": lr0,
        "lrf": lrf,
        "patience": patience,
        "device": device_str,
        "project": project_str,
        "name": name,
        "workers": workers,
        "optimizer": optimizer,
        "seed": seed,
        "save_period": save_period,
        "exist_ok": exist_ok,
        "val": val,
        "amp": amp,
        "verbose": verbose,
    }

    if freeze is not None and freeze > 0:
        train_kwargs["freeze"] = freeze

    if extra_train_args:
        train_kwargs.update(extra_train_args)

    # Pre-initialize W&B run if requested and authenticated
    wandb_run = None
    if use_wandb:
        if not os.getenv("WANDB_API_KEY"):
            setup_wandb()
        try:
            import wandb

            if os.getenv("WANDB_MODE") != "disabled" and wandb.run is None and os.getenv("WANDB_API_KEY"):
                wandb_run = wandb.init(
                    project=wandb_project or "home-fire-detection",
                    name=wandb_name or name,
                    config={
                        "weights": str(weights_path),
                        "data": str(data_path),
                        "epochs": epochs,
                        "batch": batch,
                        "imgsz": imgsz,
                        "lr0": lr0,
                        "lrf": lrf,
                        "patience": patience,
                        "device": device_str,
                        "workers": workers,
                        "optimizer": optimizer,
                        "seed": seed,
                        "freeze": freeze,
                    },
                )
            elif wandb.run is not None:
                wandb_run = wandb.run
        except Exception as e:
            logger.warning(f"Could not pre-initialize Weights & Biases run: {e}")
    else:
        os.environ["WANDB_MODE"] = "disabled"

    logger.info(f"Starting fine-tuning with arguments: {train_kwargs}")
    try:
        results = model.train(**train_kwargs)
    except Exception:
        if wandb_run is not None:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish(exit_code=1)
            except Exception:
                pass
        raise

    # Resolve output checkpoint paths
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / name))
    best_ckpt = save_dir / "weights" / "best.pt"
    last_ckpt = save_dir / "weights" / "last.pt"

    summary: Dict[str, Any] = {
        "status": "COMPLETED",
        "save_dir": str(save_dir),
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else None,
        "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else None,
        "epochs_trained": epochs,
        "device": device_str,
        "data_config": str(data_path),
        "source_weights": str(weights_path),
        "metrics_summary": {},
    }

    if wandb_run is not None and hasattr(wandb_run, "url"):
        summary["wandb_url"] = wandb_run.url

    if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
        summary["metrics_summary"] = {k: _safe_float(v) if isinstance(v, (int, float)) else str(v) for k, v in results.results_dict.items()}

    logger.info(f"Training completed successfully. Checkpoints saved to: {save_dir / 'weights'}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for model training."""
    parser = argparse.ArgumentParser(
        prog="train",
        description="Transfer learning training pipeline for fire and smoke detection",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights (default: config.yaml)")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML config (default: data_path from config.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: config.yaml)")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=None, help="Batch size (default: config.yaml)")
    parser.add_argument("--imgsz", "--img-size", dest="imgsz", type=int, default=None, help="Image size (default: config.yaml)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate (default: config.yaml)")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate factor (default: config.yaml)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: config.yaml)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run training on, e.g. 0, cuda:0, cpu (default: auto cuda/cpu)",
    )
    parser.add_argument("--project", type=str, default=None, help="Project save directory (default: config.yaml)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (default: config.yaml)")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader worker threads (default: config.yaml)")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer choice (default: config.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: config.yaml)")
    parser.add_argument("--freeze", type=int, default=None, help="Number of layers to freeze (optional)")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="home-fire-detection",
        help="Weights & Biases project name (default: home-fire-detection)",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name (default: same as --name)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases experiment tracking",
    )
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to output summary JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint for training."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        summary = train_yolo(
            weights=args.weights,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            lrf=args.lrf,
            patience=args.patience,
            device=args.device,
            project=args.project,
            name=args.name,
            workers=args.workers,
            optimizer=args.optimizer,
            seed=args.seed,
            freeze=args.freeze,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            use_wandb=not args.no_wandb,
        )

        if args.json_out:
            out_p = Path(args.json_out).resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to: {out_p}")

        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

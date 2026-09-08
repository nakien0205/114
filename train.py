#!/usr/bin/env python3
"""
Transfer Learning Training Pipeline CLI for Fire and Smoke Detection.
Loads pretrained YOLOv8-P2 checkpoint and fine-tunes on Home Fire Dataset.

Usage:
    python train.py [--weights D:/Python/Projects/Maritime-SAR/best.pt] [--data data.yaml] [--epochs 50] [--batch 16] [--imgsz 640] [--device 0]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add workspace root and src to python path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logger = logging.getLogger("fire_detection.train")


def setup_wandb(env_path: Path | None = None) -> bool:
    """
    Load Weights & Biases API key from .env file at root folder and authenticate.

    Looks for 'wandb_api' in .env, sets WANDB_API_KEY, and calls wandb.login().
    """
    if env_path is None:
        env_path = root_dir / ".env"

    api_key = None

    # 1. Try python-dotenv first
    try:
        import dotenv

        if env_path.is_file():
            dotenv.load_dotenv(dotenv_path=env_path)
            api_key = os.getenv("wandb_api")
    except ImportError:
        pass

    # 2. Fallback to direct file parsing if dotenv didn't find it or is not installed
    if not api_key and env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("wandb_api="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception as e:
            logger.warning("Could not read %s: %s", env_path, e)

    # 3. Authenticate with W&B
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        try:
            import wandb

            wandb.login(key=api_key)
            print("[W&B] Logged in successfully using 'wandb_api' from .env")
            return True
        except Exception as e:
            logger.warning("Failed to log in to Weights & Biases: %s", e)
            return False
    else:
        logger.warning(
            "'wandb_api' key not found in %s or environment. Training will continue without W&B sync.",
            env_path,
        )
        return False


try:
    from src.fire_detection.train import main
except ImportError:
    from fire_detection.train import main

if __name__ == "__main__":
    setup_wandb()
    sys.exit(main())

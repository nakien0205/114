# Fire and Smoke Detection & Dataset Pipeline

A modular Python framework for auditing fire & smoke datasets, preparing leakage-free train/validation/test splits, and training YOLO-based object detection models for fire and smoke detection.

---

## Overview

The project is organized into two primary packages:

1. **`fire_audit` (`src/fire_audit`)**:
   - **Audit**: Scans images and bounding box annotations for corruption, coordinate errors, and class balance.
   - **Segment**: Identifies continuous video sequences and near-duplicate frames using perceptual hashing (`dHash`).
   - **Prepare**: Partitions dataset sequences into train, validation, and test splits to prevent data leakage.
   - **Verify**: Acceptance testing on split integrity and YOLO annotation formats.

2. **`fire_detection` (`src/fire_detection`)**:
   - **Train**: Fine-tunes pretrained YOLO models on the prepared fire & smoke dataset.
   - **Evaluate**: Evaluates model performance (Precision, Recall, mAP50, mAP50-95).
   - **Infer**: Runs visual object detection inference on test images or directories.

---

## Directory Structure

```text
├── data.yaml            # YOLO dataset configuration (classes, split paths)
├── evaluate.py          # CLI to evaluate model checkpoints
├── infer.py             # CLI to run detection inference on images
├── run_pipeline.py      # CLI for dataset audit, segmentation, and preparation
├── train.py             # CLI to train/fine-tune YOLO models
└── src/
    ├── fire_audit/      # Dataset audit, sequence segmentation & partitioning
    └── fire_detection/  # Model training, evaluation & inference routines
```

---

## Getting Started

### 1. Dataset Audit and Preparation

Run the full dataset pipeline (audit, segmentation, partitioning, and verification):

```bash
python run_pipeline.py all --data-dir "path/to/dataset" --output-dir "path/to/output" --train-ratio 0.8 --seed 42
```

Or run individual stages:

```bash
# Audit images and annotations
python run_pipeline.py audit --data-dir "path/to/dataset"

# Segment video frames to prevent data leakage
python run_pipeline.py segment --data-dir "path/to/dataset"

# Partition into train/val/test splits
python run_pipeline.py prepare --data-dir "path/to/dataset" --train-ratio 0.8
```

---

### 2. Model Training

Fine-tune a YOLO model on the dataset specified in `data.yaml`:

```bash
python train.py --weights "path/to/pretrained.pt" --data data.yaml --epochs 50 --batch 16 --imgsz 640
```

---

### 3. Model Evaluation

Evaluate trained weights on validation or test sets:

```bash
python evaluate.py --weights "runs/train/weights/best.pt" --data data.yaml --split test
```

---

### 4. Inference and Visualization

Run detection on sample images:

```bash
python infer.py --weights "runs/train/weights/best.pt" --source "path/to/images" --output-dir "runs/infer"
```

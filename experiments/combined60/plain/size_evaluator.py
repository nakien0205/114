from __future__ import annotations

import contextlib

import copy

import csv

import hashlib

import importlib.metadata

import io

import json

from pathlib import Path

import sys

import zipfile

import numpy as np

from pycocotools.coco import COCO

from pycocotools.cocoeval import COCOeval

CATEGORIES = [{"id": 1, "name": "fire"}, {"id": 2, "name": "smoke"}]

CUSTOM_RANGES = [
    [0.0, 1e10],
    [0.0, float(np.nextafter(16.0**2, -np.inf))],
    [0.0, float(np.nextafter(32.0**2, -np.inf))],
    [16.0**2, float(np.nextafter(32.0**2, -np.inf))],
    [32.0**2, float(np.nextafter(96.0**2, -np.inf))],
    [96.0**2, 1e10],
]

CUSTOM_NAMES = ["all", "tiny_lt16", "small_lt32_including_tiny", "small16_to_lt32", "medium32_to_lt96", "large_ge96"]

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def xyxy_to_xywh(box, factor=1.0, allow_degenerate=False):
    x1, y1, x2, y2 = map(float, box)
    assert np.isfinite(box).all(), box
    assert np.isfinite(factor) and factor > 0, factor
    if allow_degenerate:
        # Direct-regression predictions can have inverted edges. Retain their
        # scores as zero-area false positives, rather than dropping detections
        # or swapping edges to invent a positive-area box. GT stays strict.
        width, height = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    else:
        assert x2 > x1 and y2 > y1, box
        width, height = x2 - x1, y2 - y1
    return [x1 * factor, y1 * factor, width * factor, height * factor]

def prediction_audit(rows):
    details = []
    count = degenerate = 0
    for row in rows:
        assert len(row['pred_cls']) == len(row['conf']) == len(row['pred_boxes'])
        for index, box in enumerate(row['pred_boxes']):
            xyxy_to_xywh(box, allow_degenerate=True)
            count += 1
            x1, y1, x2, y2 = box
            degenerate += int(x2 == x1 or y2 == y1)
            if x2 < x1 or y2 < y1:
                details.append(dict(image=row['image'], prediction_index=index,
                    cls=row['pred_cls'][index], confidence=row['conf'][index], raw_xyxy=box))
    return dict(total_predictions=count, inverted_predictions=len(details),
        already_degenerate_predictions=degenerate, inverted_details=details,
        policy='Clamp negative predicted width/height to zero for COCO; retain every score and detection. Raw predictions unchanged; strict GT; nonfinite values fail.')

def convert(rows, scaled=False):
    """Scale GT and detections together; retain all GT for COCO ignore matching."""
    images, annotations, detections = [], [], []
    for image_id, row in enumerate(sorted(rows, key=lambda r: r["image"]), 1):
        h, w = row["ori_shape"]
        factor = 640.0 / max(h, w) if scaled else 1.0
        images.append({"id": image_id, "file_name": row["image"], "width": w * factor, "height": h * factor})
        for i, (cls, box) in enumerate(zip(row["target_cls"], row["gt_boxes"])):
            bbox = xyxy_to_xywh(box, factor)
            # Use archived size to reproduce the project's exact <16/<32 membership.
            area = row["gt_size640"][i] ** 2 if scaled else bbox[2] * bbox[3]
            assert np.isclose(area, bbox[2] * bbox[3], rtol=1e-10, atol=1e-8)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": cls + 1, "bbox": bbox, "area": area, "iscrowd": 0})
        for cls, score, box in zip(row["pred_cls"], row["conf"], row["pred_boxes"]):
            assert np.isfinite(score) and 0.001 <= score <= 1.0
            detections.append({"image_id": image_id, "category_id": cls + 1,
                               "bbox": xyxy_to_xywh(box, factor, allow_degenerate=True), "score": score})
    return {"info": {"description": "Complete combined Home Fire + D-Fire validation; proportional-fit640 box areas"},
            "images": images, "annotations": annotations, "categories": copy.deepcopy(CATEGORIES)}, detections

def run_coco(dataset, predictions, ranges=None, names=None):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = copy.deepcopy(dataset)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(copy.deepcopy(predictions))
        else:
            dt = COCO()
            dt.dataset = {**copy.deepcopy(dataset), "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, 100, 300]
        if ranges is not None:
            evaluator.params.areaRng = copy.deepcopy(ranges)
            evaluator.params.areaRngLbl = list(names)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator

def mean_valid(array):
    valid = array[array >= 0]
    return float(valid.mean()) if valid.size else None

def extract(evaluator, dataset, coordinate_system):
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    results = []
    for ai, (area_name, bounds) in enumerate(zip(evaluator.params.areaRngLbl, evaluator.params.areaRng)):
        for max_dets in (100, 300):
            mi = evaluator.params.maxDets.index(max_dets)
            for category_id, category in [(None, "all"), (1, "fire"), (2, "smoke")]:
                ids = list(range(len(evaluator.params.catIds))) if category_id is None else [list(evaluator.params.catIds).index(category_id)]
                pr = np.take(precision[:, :, :, ai, mi], ids, axis=2)
                rc = np.take(recall[:, :, ai, mi], ids, axis=1)
                count = sum(bounds[0] <= a["area"] <= bounds[1] and
                            (category_id is None or a["category_id"] == category_id) for a in dataset["annotations"])
                active = [cid for cid in evaluator.params.catIds if (category_id is None or cid == category_id) and
                          any(bounds[0] <= a["area"] <= bounds[1] and a["category_id"] == cid for a in dataset["annotations"])]
                results.append({"coordinates": coordinate_system, "area_group": area_name,
                                "area_min_inclusive": bounds[0], "area_max_inclusive": bounds[1],
                                "category": category, "gt_count": count, "categories_with_gt": len(active),
                                "max_dets_per_image_category": max_dets,
                                "AP50_95": mean_valid(pr), "AP50": mean_valid(pr[0]),
                                "AP75": mean_valid(pr[5]), "AR50_95": mean_valid(rc)})
    return results

def self_checks():
    row = {"image": "fixture.jpg", "ori_shape": [640, 640], "target_cls": [0, 0, 0, 0, 0],
           "gt_boxes": [[0, 0, 8, 8], [50, 50, 66, 66], [100, 100, 132, 132],
                        [150, 150, 170, 170], [250, 250, 350, 350]],
           "gt_size640": [8, 16, 32, 20, 100], "pred_cls": [0] * 5,
           "conf": [.95, .94, .93, .92, .99]}
    row["pred_boxes"] = copy.deepcopy(row["gt_boxes"])
    ds, ps = convert([row], True)
    ev = run_coco(ds, ps, CUSTOM_RANGES, CUSTOM_NAMES)
    records = extract(ev, ds, "fixture")
    get = lambda group, cls="fire": next(r for r in records if r["area_group"] == group and r["category"] == cls and r["max_dets_per_image_category"] == 100)
    assert get("tiny_lt16")["gt_count"] == 1
    assert get("small_lt32_including_tiny")["gt_count"] == 3
    assert get("small16_to_lt32")["gt_count"] == 2
    assert get("medium32_to_lt96")["gt_count"] == 1
    assert get("tiny_lt16", "smoke")["AP50_95"] is None
    assert np.isclose(get("tiny_lt16")["AP50_95"], 1.0)
    # The high-scoring correct large detection must be ignored for small AP.
    assert np.isclose(get("small_lt32_including_tiny")["AP50_95"], 1.0)
    empty = run_coco(ds, [], CUSTOM_RANGES, CUSTOM_NAMES)
    assert mean_valid(empty.eval["precision"][:, :, 0, 1, 2]) == 0.0
    noisy = copy.deepcopy(ps)
    noisy.insert(0, {"image_id": 1, "category_id": 1, "bbox": [450, 450, 8, 8], "score": .999})
    noise_ev = run_coco(ds, noisy, CUSTOM_RANGES, CUSTOM_NAMES)
    assert mean_valid(noise_ev.eval["precision"][:, :, 0, 1, 2]) < .51
    # Common uniform scaling must preserve IoU and all-size AP.
    large_row = copy.deepcopy(row)
    large_row["ori_shape"] = [1280, 1280]
    large_row["gt_boxes"] = [[2 * x for x in b] for b in row["gt_boxes"]]
    large_row["pred_boxes"] = copy.deepcopy(large_row["gt_boxes"])
    scaled_ds, scaled_ps = convert([large_row], True)
    assert np.allclose([a["bbox"] for a in scaled_ds["annotations"]], [a["bbox"] for a in ds["annotations"]])
    assert np.allclose([a["bbox"] for a in scaled_ps], [a["bbox"] for a in ps])
    return {"passed": True, "checks": ["strict_16_and_32_boundaries", "no_gt_is_unavailable",
            "perfect_predictions_AP1", "large_correct_detection_ignored_for_small_AP",
            "empty_predictions_AP0", "small_false_positive_reduces_AP", "coordinate_scaling"]}

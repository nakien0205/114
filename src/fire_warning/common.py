"""Shared video I/O, frozen inference, tracking, worksheet and telemetry helpers."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from .fsm_engine import FireSmokeFSM, FSMConfig, scaled_size

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / 'weights' / 'baseline-best.pt'
BASELINE_SHA = 'd7e6c1e13a2ad938332707d4ca97ffbd46475777310ec67c9ee8171efbe51f74'
DATA = ROOT / 'data'
RESULTS = ROOT / 'results' / 'fire_warning'
CLASSES = {0: 'fire', 1: 'smoke'}
SCHEMA_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + '.partial')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def accepted_number(row, key):
    value = row.get(key, '')
    if value is None or str(value).strip() == '':
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'{key} must be finite or blank')
    if key == 'should_alert_human' and number not in (0, 1):
        raise ValueError('should_alert_human must be 0, 1 or blank')
    if key.startswith('t_') and number < 0 and number != -1:
        raise ValueError(f'{key} must be nonnegative, -1 or blank')
    return number


def resolve_video(value, base=ROOT):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def discover_videos(data_dir=DATA, manifest=None):
    rows = []
    for folder, subset in [('mmodelfire', 'mmodelfire'), ('negatives', 'public_negative'),
                           ('custom_4stage', 'custom_4stage')]:
        for p in sorted((Path(data_dir) / 'videos' / folder).glob('**/*')):
            if p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov'):
                rows.append(dict(video_path=str(p.resolve()), subset=subset))
    if manifest:
        p = Path(manifest)
        values = read_csv(p) if p.suffix.lower() == '.csv' else json.loads(p.read_text(encoding='utf-8'))
        for value in values:
            row = dict(value) if isinstance(value, dict) else dict(video_path=value)
            video = row.get('video_path') or row.get('output') or row.get('path') or row.get('source_path')
            if not video:
                raise ValueError('Each manifest item must specify video_path/path/output/source_path')
            row['video_path'] = str(resolve_video(video))
            row.setdefault('subset', 'custom_4stage')
            rows.append(row)
    return list({str(resolve_video(r['video_path'])): r for r in rows}.values())


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: object
    timestamp_quality: str
    read_ms: float


class VideoSource:
    """Recorded frames use PTS. Camera timestamps start at the first completed read."""
    def __init__(self, source, width=None, height=None, camera_fps=30):
        self.source = str(source)
        self.camera = self.source.isdecimal()
        self.closed = False
        self.origin = None
        self.last_time = None
        self.index = 0
        self.container = self.cap = None
        self.target_size = (width, height) if width and height else None
        if self.camera:
            import cv2
            self.cap = cv2.VideoCapture(int(self.source))
            if not self.cap.isOpened():
                self.cap.release()
                raise RuntimeError(f'Camera {source} is unavailable')
            if width: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or float(camera_fps)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration_sec = None
        else:
            try:
                import av
            except ImportError as e:
                raise RuntimeError('Install the package requirements: PyAV is required for recorded timestamps') from e
            self.container = av.open(str(resolve_video(source)))
            if not self.container.streams.video:
                self.container.close()
                raise RuntimeError(f'No video stream: {source}')
            stream = self.container.streams.video[0]
            stream.codec_context.thread_count = 2
            stream.codec_context.thread_type = 'AUTO'
            self.fps = float(stream.average_rate or stream.guessed_rate or 0)
            if self.fps <= 0 or not math.isfinite(self.fps):
                self.container.close()
                raise RuntimeError('Video FPS unavailable; specify a valid timestamped video')
            self.width, self.height = stream.width, stream.height
            self.duration_sec = float(stream.duration * stream.time_base) if stream.duration else (
                self.container.duration / 1e6 if self.container.duration else None)
            self.iterator = iter(self.container.decode(video=0))

    def read(self):
        start = time.perf_counter()
        if self.closed: return None
        if self.camera:
            ok, image = self.cap.read()
            if not ok: return None
            now = time.perf_counter()
            if self.origin is None: self.origin = now
            timestamp, quality = now - self.origin, 'host_read_monotonic'
        else:
            try: decoded = next(self.iterator)
            except StopIteration: return None
            if self.target_size and (decoded.width, decoded.height) != self.target_size:
                decoded = decoded.reformat(width=self.target_size[0], height=self.target_size[1], format='bgr24')
            image = decoded.to_ndarray(format='bgr24')
            pts = decoded.time
            if pts is None:
                timestamp, quality = self.index / self.fps, 'nominal_fps_fallback'
            else:
                if self.origin is None: self.origin = float(pts)
                timestamp, quality = float(pts) - self.origin, 'presentation_timestamp'
        if self.last_time is not None and timestamp <= self.last_time:
            raise RuntimeError('Video timestamps do not increase; exact event timing is invalid')
        self.last_time = timestamp
        frame = Frame(self.index, timestamp, image, quality, (time.perf_counter() - start) * 1000)
        self.index += 1
        return frame

    def close(self):
        self.closed = True
        if self.cap is not None: self.cap.release()
        if self.container is not None: self.container.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class VideoWriter:
    """PTS-preserving H.264 output; fixed dimensions within a recording."""
    def __init__(self, path, width, height, fps):
        import av
        from fractions import Fraction
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if width % 2 or height % 2:
            raise ValueError('Video output dimensions must be even')
        self.container = av.open(str(path), 'w')
        try:
            self.stream = self.container.add_stream('libx264', rate=Fraction(str(fps)).limit_denominator(100000))
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = 'yuv420p'
            self.stream.codec_context.time_base = Fraction(1, 1000000)
            self.stream.options = {'preset': 'ultrafast', 'crf': '23'}
        except Exception:
            self.container.close()
            raise
        self.closed = False

    def write(self, image, timestamp):
        import av
        frame = av.VideoFrame.from_ndarray(image, format='bgr24')
        frame.pts = int(round(timestamp * 1000000))
        frame.time_base = self.stream.codec_context.time_base
        for packet in self.stream.encode(frame): self.container.mux(packet)

    def close(self):
        if self.closed: return
        self.closed = True
        try:
            for packet in self.stream.encode(): self.container.mux(packet)
        finally: self.container.close()


class Detector:
    def __init__(self, weights=DEFAULT_WEIGHTS, device='auto', precision='fp32', verify=True):
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        import torch
        import ultralytics
        import cv2
        from ultralytics import YOLO
        if ultralytics.__version__ != '8.4.24':
            raise RuntimeError(f'Ultralytics 8.4.24 required; found {ultralytics.__version__}. Use the isolated environment.')
        self.weights = Path(weights).resolve()
        if not self.weights.is_file(): raise FileNotFoundError(self.weights)
        self.sha = sha256(self.weights)
        if verify and self.weights.suffix == '.pt' and self.sha != BASELINE_SHA:
            raise RuntimeError('Frozen baseline checkpoint hash changed')
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else str(device)
        self.precision = precision
        self.fallback_reason = None
        if self.device == 'cpu' and precision == 'fp16':
            self.precision, self.fallback_reason = 'fp32', 'CPU selected; CUDA FP16 unavailable'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # A previous worker in this process has already established it.
        if self.weights.suffix == '.onnx':
            import onnxruntime as ort
            if hasattr(ort, 'preload_dlls'): ort.preload_dlls()
            if self.device != 'cpu' and 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX GPU requested, but CUDAExecutionProvider is unavailable')
        self.model = YOLO(str(self.weights), task='detect')
        self.parameters = None
        if self.weights.suffix == '.pt':
            self.parameters = sum(p.numel() for p in self.model.model.parameters())
            if self.model.names != CLASSES or self.model.model.stride.tolist() != [4, 8, 16, 32]:
                raise RuntimeError('Expected the frozen four-head fire/smoke detector')
        self.options = dict(device=self.device, half=self.precision == 'fp16', imgsz=640,
            rect=False, batch=1, conf=0.14, iou=0.5, max_det=300, nms=False,
            verbose=False, save=False, save_txt=False, save_crop=False)

    def predict(self, image):
        import numpy as np
        import torch
        if self.device != 'cpu': torch.cuda.synchronize()
        start = time.perf_counter()
        result = self.model.predict(image, **self.options)[0]
        if self.model.names != CLASSES:
            raise RuntimeError('Backend class metadata must be {0: fire, 1: smoke}')
        boxes = result.boxes.data.detach().cpu().numpy()
        if not np.isfinite(boxes).all(): raise RuntimeError('Detector returned nonfinite predictions')
        if self.device != 'cpu': torch.cuda.synchronize()
        total = (time.perf_counter() - start) * 1000
        return boxes.tolist(), dict(detector_ms=total, **{k + '_ms': float(v) for k, v in result.speed.items()})

    def identity(self):
        import torch, ultralytics
        return dict(weights=str(self.weights), checkpoint_sha256=self.sha, parameters_before_fusion=self.parameters,
            ultralytics=ultralytics.__version__, torch=torch.__version__, device=self.device,
            gpu=torch.cuda.get_device_name(0) if self.device != 'cpu' else None,
            backend=self.weights.suffix.lstrip('.'), precision=self.precision, fallback_reason=self.fallback_reason,
            options=self.options, tf32=False, cpu_threads=torch.get_num_threads(),
            opencv_threads=__import__('cv2').getNumThreads())

    def predict_batch(self, images):
        """A bounded batch, never an unbounded list of dataset paths."""
        import numpy as np
        import torch
        if not images: return []
        if len(images) > 16: raise ValueError('Annotation batches are bounded to 16 frames')
        options = dict(self.options, batch=len(images))
        with torch.inference_mode():
            results = self.model.predict(images, **options)
            if len(results) != len(images): raise RuntimeError('Batched prediction lost source/frame correspondence')
            boxes = [r.boxes.data.detach().cpu().numpy() for r in results]
        if self.model.names != CLASSES or any(not np.isfinite(b).all() for b in boxes):
            raise RuntimeError('Invalid batched fire/smoke output')
        return [b.tolist() for b in boxes]


class ClassTrackers:
    def __init__(self, fps):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.args = SimpleNamespace(track_high_thresh=0.14, track_low_thresh=0.10,
            new_track_thresh=0.14, track_buffer=30, match_thresh=0.8, fuse_score=False)
        self.fps = fps
        self.trackers = [BYTETracker(self.args, frame_rate=fps) for _ in range(2)]
        self.ids = {}

    def update(self, boxes, image):
        import numpy as np
        from ultralytics.engine.results import Boxes
        width, height = image.shape[1], image.shape[0]
        rows = [r for r in boxes if r[4] >= (0.14 if scaled_size(r, (width, height))['scaled_side'] <= 32 else 0.22)]
        out = []
        for cls, tracker in enumerate(self.trackers):
            data = np.asarray([r for r in rows if int(r[5]) == cls], dtype=np.float32).reshape(-1, 6)
            tracked = tracker.update(Boxes(data, image.shape[:2]), image)
            for r in tracked:
                key = (cls, int(r[4]))
                if key not in self.ids: self.ids[key] = len(self.ids) + 1
                out.append([*map(float, r[:4]), self.ids[key], float(r[5]), cls])
        return out


class Pipeline:
    def __init__(self, detector, fps=30, glare_filter=True, persistence_only=False):
        self.detector, self.fps = detector, fps
        self.config = FSMConfig(fps=fps, glare_filter=glare_filter, persistence_only=persistence_only)
        self.reset()

    def reset(self):
        self.trackers = ClassTrackers(self.fps)
        self.fsm = FireSmokeFSM(self.config)

    def process(self, frame):
        boxes, timings = self.detector.predict(frame.image)
        start = time.perf_counter()
        tracks = self.trackers.update(boxes, frame.image)
        timings['tracking_ms'] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        state = self.fsm.update(tracks, frame.timestamp_sec, (frame.image.shape[1], frame.image.shape[0]))
        timings['fsm_ms'] = (time.perf_counter() - start) * 1000
        return dict(frame_index=frame.index, timestamp_sec=frame.timestamp_sec,
            timestamp_quality=frame.timestamp_quality, image_size=[frame.image.shape[1], frame.image.shape[0]],
            boxes=boxes, tracks=tracks, state=state, timings=dict(decode_ms=frame.read_ms, **timings))


def stats(values):
    import numpy as np
    if not values: return dict(count=0, mean=None, median=None, p95=None)
    a = np.asarray(values, dtype=float)
    return dict(count=len(values), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)))


def detector_arguments(parser):
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')


def run_main(main):
    try: main()
    except KeyboardInterrupt: raise SystemExit('Stopped; resources have been released.')
    except Exception as e: raise SystemExit(f'{type(e).__name__}: {e}') from e


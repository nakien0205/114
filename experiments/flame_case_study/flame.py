"""Independent FLAME-inspired postprocessing; see README for deviations.

Reference: Gragnaniello et al. (2025), doi:10.1007/s00521-024-10963-z,
Sections 2.3/2.4, Equation 2, Algorithm 1. No author source code was used.
Input boxes are original-image [x1,y1,x2,y2,confidence,class].
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    confidence: float = 0.4
    foreground_fraction: float = 0.2
    confirmation_sec: float = 1.0
    occlusion_sec: float = 0.1
    history: int = 500
    variance_threshold: float = 16.0
    initial_variance: float = 15.0
    mixtures: int = 3
    background_ratio: float = 0.9
    detect_shadows: bool = True
    fire_class: int = 0
    smoke_class: int = 1

    def __post_init__(self):
        for name in ('confidence', 'foreground_fraction', 'background_ratio'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and within [0,1]')
        for name in ('confirmation_sec', 'occlusion_sec', 'variance_threshold', 'initial_variance'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.history < 1 or self.mixtures < 1:
            raise ValueError('history and mixtures must be positive')
        if self.fire_class == self.smoke_class:
            raise ValueError('Fire and smoke class IDs must differ')


@dataclass
class Track:
    id: int
    box: list
    state: str = 'New'
    state_age_sec: float = 0.0
    observed: bool = True
    confirmed_at_sec: float | None = None


class CentroidConfirmation:
    """Greedy highest-similarity association and Algorithm 1 state transitions.

    Confirmed is absorbing in Algorithm 1, including on missing observations.
    It is therefore an alarm state, never proof of current-frame localization.
    """

    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_id = 1
        self.previous_time = None
        self.image_size = None

    def update(self, boxes, timestamp_sec, image_size):
        t = float(timestamp_sec)
        if not math.isfinite(t) or t < 0 or (self.previous_time is not None and t <= self.previous_time):
            raise ValueError('Timestamps must be finite, nonnegative and strictly increasing; reset at a cut')
        if self.image_size is not None and image_size != self.image_size:
            raise ValueError('Image dimensions changed; reset at a source/view boundary')
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError('Invalid image size')
        dt = 0.0 if self.previous_time is None else t - self.previous_time
        self.previous_time, self.image_size = t, image_size
        # Sorting makes equal-distance ties independent of detector output order.
        candidates = sorted([list(map(float, b)) for b in boxes])
        diagonal = math.hypot(width, height)
        pairs = []
        for key, track in self.tracks.items():
            track.observed = False
            a = track.box
            for index, b in enumerate(candidates):
                distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
                if distance < diagonal:
                    pairs.append((distance, key, index))
        used_tracks, used_boxes = set(), set()
        for _, key, index in sorted(pairs):
            if key not in used_tracks and index not in used_boxes:
                self.tracks[key].box = candidates[index]
                self.tracks[key].observed = True
                used_tracks.add(key)
                used_boxes.add(index)
        events, rejected = [], []
        for key, track in list(self.tracks.items()):
            if track.observed:
                if track.state == 'Occluded':
                    track.state, track.state_age_sec = 'New', 0.0
                else:
                    track.state_age_sec += dt
                if track.state == 'New' and track.state_age_sec > self.config.confirmation_sec:
                    track.state, track.confirmed_at_sec = 'Confirmed', t
                    events.append(dict(track_id=key, timestamp_sec=t, box=track.box.copy()))
            else:
                if track.state == 'New':
                    track.state, track.state_age_sec = 'Occluded', 0.0
                else:
                    track.state_age_sec += dt
                if track.state == 'Occluded' and track.state_age_sec > self.config.occlusion_sec:
                    rejected.append(key)
                    del self.tracks[key]
        for index, box in enumerate(candidates):
            if index not in used_boxes:
                self.tracks[self.next_id] = Track(self.next_id, box)
                self.next_id += 1
        confirmed = [s for s in self.tracks.values() if s.state == 'Confirmed']
        return dict(
            alarm=bool(confirmed), new_confirmations=events, rejected_track_ids=rejected,
            observed_confirmed_boxes=[s.box.copy() for s in confirmed if s.observed],
            tracks=[asdict(s) for s in self.tracks.values()],
        )


class FlameProcessor:
    """Stateful, causal postprocessor independent of detector architecture."""

    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        cfg = self.config
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=cfg.history, varThreshold=cfg.variance_threshold, detectShadows=cfg.detect_shadows)
        self.background.setNMixtures(cfg.mixtures)
        self.background.setVarInit(cfg.initial_variance)
        self.background.setBackgroundRatio(cfg.background_ratio)
        self.tracker = CentroidConfirmation(cfg)
        self.previous_time = None
        self.shape = None

    def update(self, image, boxes, timestamp_sec, foreground_mask=None):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Expected a uint8 BGR frame')
        h, w = image.shape[:2]
        t = float(timestamp_sec)
        if not math.isfinite(t) or t < 0 or (self.previous_time is not None and t <= self.previous_time):
            raise ValueError('Timestamps must increase strictly; reset between streams')
        if self.shape is not None and image.shape != self.shape:
            raise ValueError('Frame shape changed; reset the processor')
        raw = []
        for row in boxes:
            if len(row) != 6 or not all(math.isfinite(float(v)) for v in row):
                raise ValueError('Expected six finite box values')
            x1, y1, x2, y2, confidence, cls = map(float, row)
            if not 0 <= confidence <= 1 or not cls.is_integer():
                raise ValueError('Invalid detection box')
            if int(cls) not in (self.config.fire_class, self.config.smoke_class) or confidence < self.config.confidence:
                continue
            if x2 <= x1 or y2 <= y1:
                raise ValueError('Invalid detection box')
            x1, x2 = max(0., x1), min(float(w), x2)
            y1, y2 = max(0., y1), min(float(h), y2)
            if x2 > x1 and y2 > y1:
                raw.append([x1, y1, x2, y2, confidence, cls])
        if foreground_mask is None:
            mask = self.background.apply(image)
        else:
            mask = np.asarray(foreground_mask)
            if mask.shape != (h, w) or mask.dtype != np.uint8:
                raise ValueError('Foreground mask must be uint8 and match frame dimensions')
        self.previous_time, self.shape = t, image.shape
        moving, ratios = [], []
        for box in raw:
            x1, y1 = math.floor(box[0]), math.floor(box[1])
            x2, y2 = math.ceil(box[2]), math.ceil(box[3])
            # Shadows (127) are not counted as foreground (255).
            fraction = float(np.count_nonzero(mask[y1:y2, x1:x2] == 255)) / ((x2-x1)*(y2-y1))
            ratios.append(fraction)
            if fraction > self.config.foreground_fraction:
                moving.append(box)
        result = self.tracker.update(moving, t, (w, h))
        return dict(timestamp_sec=t, width=w, height=h, raw_boxes=raw, moving_boxes=moving,
                    foreground_fractions=ratios, **result)

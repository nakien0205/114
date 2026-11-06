"""Deterministic temporal confirmation and incident decision logic.

Rigidity is diagnostic during onset. The optional guard affects only the
five-second Stage 4 fallback; it never deletes or delays a confirmed onset.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
import statistics
from typing import Iterable


@dataclass(frozen=True)
class FSMConfig:
    fps: float = 30.0
    glare_filter: bool = True
    persistence_only: bool = False
    tiny_candidate_conf: float = 0.14
    standard_candidate_conf: float = 0.22
    tiny_confirm_conf: float = 0.17
    standard_confirm_conf: float = 0.25
    confirmation_window_sec: float = 0.4
    minimum_age_sec: float = 8 / 30
    smoothing_tau_sec: float = 0.2
    history_sec: float = 3.0
    lost_grace_sec: float = 0.5
    close_after_sec: float = 1.0
    persistence_alert_sec: float = 5.0
    alert_score: float = 65.0

    def __post_init__(self):
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("FSM FPS must be positive and finite")
        if self.smoothing_tau_sec <= 0:
            raise ValueError("Smoothing time constant must be positive")


def scaled_size(box, image_size):
    """image_size is (width, height); boxes are in original-image pixels."""
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    w = max(0.0, float(box[2]) - float(box[0])) * 640 / max(width, height)
    h = max(0.0, float(box[3]) - float(box[1])) * 640 / max(width, height)
    side = math.sqrt(w * h)
    tag = "TINY" if side <= 16 else "SMALL" if side <= 32 else "MEDIUM_LARGE"
    return dict(scaled_width=w, scaled_height=h, scaled_side=side, size_tag=tag)


@dataclass
class TrackState:
    track_id: int
    cls_id: int
    first_seen: float
    last_seen: float
    ema_conf: float
    smoothed_area: float
    box: list[float]
    size: dict
    confirmed_at: float | None = None
    progression_at: float | None = None
    onset_area: float | None = None
    onset_size: dict | None = None
    presence: deque = field(default_factory=deque)
    history: deque = field(default_factory=deque)

    def rigidity(self):
        duration = self.history[-1][0] - self.history[0][0] if self.history else 0.0
        areas = [r[1] for r in self.history]
        aspects = [r[2] for r in self.history]
        mean_area = statistics.fmean(areas) if areas else 0.0
        area_cv = statistics.pstdev(areas) / mean_area if mean_area > 0 else 0.0
        aspect_std = statistics.pstdev(aspects) if aspects else 0.0
        growth = self.smoothed_area / self.onset_area if self.onset_area else 1.0
        flag = (duration + 1e-9 >= 0.5 and self.ema_conf < 0.30 and growth < 1.10
                and aspect_std < 0.015 and area_cv < 0.02)
        return dict(history_span_sec=duration, area_cv=area_cv,
                    aspect_ratio_std=aspect_std, area_growth=growth,
                    static_low_confidence=flag)


class FireSmokeFSM:
    def __init__(self, config: FSMConfig | None = None, **kwargs):
        self.config = config or FSMConfig(**kwargs)
        self.reset()

    def reset(self):
        self.tracks: dict[int, TrackState] = {}
        self.last_timestamp = None
        self.last_confirmed_seen = None
        self.episode_onset = None
        self.critical = False
        self.alert_reason = None
        self.alert_at = None
        self.latched_score = 0.0
        self.audio_acknowledged = False

    def acknowledge(self):
        self.audio_acknowledged = True

    def update(self, detections: Iterable, timestamp_sec: float,
               image_size: tuple[int, int]):
        """Rows: [x1,y1,x2,y2,track_id,confidence,class_id]."""
        t = float(timestamp_sec)
        if not math.isfinite(t) or t < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if self.last_timestamp is not None and t <= self.last_timestamp:
            raise ValueError("Timestamps must increase strictly; reset at a source/loop boundary")
        self.last_timestamp = t
        cfg = self.config
        transitions, current = [], []
        seen = set()
        for row in detections:
            if len(row) != 7 or not all(math.isfinite(float(v)) for v in row):
                raise ValueError("Track rows require seven finite values")
            box = list(map(float, row[:4]))
            track_id, conf, cls_id = int(row[4]), float(row[5]), int(row[6])
            if cls_id not in (0, 1) or not 0 <= conf <= 1 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Invalid fire/smoke track")
            if track_id in seen:
                raise ValueError("Duplicate track ID in a frame")
            seen.add(track_id)
            size = scaled_size(box, image_size)
            small = size['scaled_side'] <= 32
            if conf < (cfg.tiny_candidate_conf if small else cfg.standard_candidate_conf):
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            old = self.tracks.get(track_id)
            if old and old.cls_id != cls_id:
                raise ValueError("Track class changed; use class-separated tracking")
            if old is None or t - old.last_seen > cfg.lost_grace_sec:
                old = TrackState(track_id, cls_id, t, t, conf, area, box, size)
                self.tracks[track_id] = old
            else:
                alpha = -math.expm1(-(t - old.last_seen) / cfg.smoothing_tau_sec)
                old.ema_conf += alpha * (conf - old.ema_conf)
                old.smoothed_area += alpha * (area - old.smoothed_area)
                old.box, old.size, old.last_seen = box, size, t
            old.presence.append(t)
            while old.presence and old.presence[0] < t - cfg.confirmation_window_sec + 1e-9:
                old.presence.popleft()
            old.history.append((t, area, (box[2] - box[0]) / (box[3] - box[1])))
            while old.history and old.history[0][0] < t - cfg.history_sec:
                old.history.popleft()
            n = math.ceil(cfg.confirmation_window_sec * cfg.fps - 1e-9)
            required = math.ceil(2 * n / 3)
            threshold = cfg.tiny_confirm_conf if small else cfg.standard_confirm_conf
            if (old.confirmed_at is None and len(old.presence) >= required
                    and t - old.first_seen + 1e-9 >= cfg.minimum_age_sec
                    and old.ema_conf + 1e-12 >= threshold):
                old.confirmed_at, old.onset_area = t, old.smoothed_area
                old.onset_size = size.copy()
                transitions.append(dict(stage=2, timestamp_sec=t, track_id=track_id))
                if self.episode_onset is None:
                    self.episode_onset = t
            current.append(old)
        confirmed = [s for s in current if s.confirmed_at is not None]
        if confirmed:
            self.last_confirmed_seen = t
        outputs, scores, progression = [], [], False
        for s in confirmed:
            cooccurrence = any(self._near(s, other) for other in confirmed if other.cls_id != s.cls_id)
            duration = t - s.confirmed_at
            rigidity = s.rigidity()
            growth = rigidity['area_growth']
            progressed = not cfg.persistence_only and (cooccurrence or (duration + 1e-9 >= 3 and growth >= 1.35))
            if progressed and s.progression_at is None:
                s.progression_at = t
                transitions.append(dict(stage=3, timestamp_sec=t, track_id=s.track_id))
            progression |= s.progression_at is not None
            score = 0.0 if cfg.persistence_only else min(100.0, max(0.0,
                30 * min(1, duration / 5) + 25 * min(1, max(0, growth - 1))
                + 25 * int(cooccurrence) + 20 * s.ema_conf))
            guarded = (cfg.glare_filter and not cfg.persistence_only and score < cfg.alert_score
                       and not cooccurrence and rigidity['static_low_confidence'])
            reason = 'hazard_score' if not cfg.persistence_only and score >= cfg.alert_score else (
                'five_second_persistence' if duration + 1e-9 >= cfg.persistence_alert_sec and not guarded else None)
            if reason and not self.critical:
                self.critical, self.alert_reason, self.alert_at = True, reason, t
                self.latched_score = score
                transitions.append(dict(stage=4, timestamp_sec=t, track_id=s.track_id, reason=reason))
            scores.append(score)
            outputs.append(dict(track_id=s.track_id, cls_id=s.cls_id, box=s.box,
                ema_conf=s.ema_conf, confirmed_at=s.confirmed_at, progression_at=s.progression_at,
                duration_sec=duration, cooccurrence=cooccurrence, hazard_score=score,
                persistence_blocked=guarded, onset_size=s.onset_size, **s.size, **rigidity))
        lost_for = t - self.last_confirmed_seen if self.last_confirmed_seen is not None else None
        if self.critical:
            stage, disposition = 4, 'CRITICAL_HUMAN_ALERT'
        elif confirmed:
            age = t - self.episode_onset
            stage = 3 if progression else 4 if age >= 3 else 2
            disposition = 'MONITORING'
        elif lost_for is not None and lost_for < cfg.close_after_sec:
            stage = 4 if lost_for >= cfg.lost_grace_sec else 2
            disposition = 'EVIDENCE_LOST_MONITORING'
        else:
            stage, disposition = 1, 'MONITORING'
            self.episode_onset = self.last_confirmed_seen = None
        for key in list(self.tracks):
            if t - self.tracks[key].last_seen > cfg.history_sec:
                del self.tracks[key]
        return dict(timestamp_sec=t, stage=stage, disposition=disposition,
            alarm=self.critical, alert_reason=self.alert_reason, alert_at=self.alert_at,
            hazard_score=max(scores + [self.latched_score]), confirmed_tracks=outputs,
            transitions=transitions, audio_acknowledged=self.audio_acknowledged,
            evidence_onset_sec=self.episode_onset, config=asdict(cfg))

    @staticmethod
    def _near(a, b):
        ax, ay = (a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2
        bx, by = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
        side = max(a.box[2] - a.box[0], a.box[3] - a.box[1],
                   b.box[2] - b.box[0], b.box[3] - b.box[1])
        return math.hypot(ax - bx, ay - by) <= 2.5 * side


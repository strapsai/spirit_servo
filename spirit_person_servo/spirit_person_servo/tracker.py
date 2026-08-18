"""Track identities across frames, so the servo loop follows one person rather
than whatever the detector happened to rank first this frame.

Tracking is delegated to ultralytics' BoT-SORT rather than hand-rolled. The
decisive reason is **camera motion compensation**: this loop slews the gimbal on
purpose, so the whole frame shifts between detections exactly when the loop is
driving hardest. BoT-SORT estimates that frame-to-frame transform (``gmc_method``)
and compensates association for it; plain IoU/centroid matching degrades badly in
precisely that regime. It also brings a Kalman filter and occlusion handling.

We drive the *standalone* ``BOTSORT`` class from our own ``Detection`` list rather
than calling ``model.track()``, so the same tracker works for any detector backend
(YOLO today, GroundingDINO next) -- ``model.track()`` is coupled to the YOLO model
object and would not survive the swap.

``Detection`` and ``Track`` are the seam: the target selector and controller are
written against these and know nothing about the tracker behind them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected person, in decoded-frame pixel coordinates (top-left origin)."""

    x: float
    y: float
    w: float
    h: float
    confidence: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w * 0.5, self.y + self.h * 0.5


@dataclass(frozen=True)
class Track:
    """A detection with a persistent identity across frames."""

    track_id: int
    detection: Detection
    last_seen: float

    @property
    def area(self) -> float:
        return self.detection.area

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center


class PersonTracker(Protocol):
    def update(
        self, detections: list[Detection], frame_bgr: np.ndarray, now: float
    ) -> list[Track]: ...

    def reset(self) -> None: ...


class _DetectionShim:
    """Adapts our ``Detection`` list to what ``BOTSORT.update()`` reads off a
    ``Results.boxes`` object: ``.conf``, ``.xywh``, ``.cls`` as numpy arrays.

    NOTE: this mirrors a semi-private ultralytics interface. It is safe here only
    because the image pins ultralytics tightly (>=8.3.78,<=8.3.80); re-verify the
    field set if that pin ever moves.
    """

    def __init__(self, detections: list[Detection]) -> None:
        if detections:
            self.xywh = np.array(
                [[d.center[0], d.center[1], d.w, d.h] for d in detections], dtype=np.float32
            )
            self.conf = np.array([d.confidence for d in detections], dtype=np.float32)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)
            self.conf = np.zeros((0,), dtype=np.float32)
        # Single-class: everything we track is a person.
        self.cls = np.zeros((len(detections),), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)


class BotSortTracker:
    """Thin adapter over ultralytics BoT-SORT."""

    def __init__(
        self,
        frame_rate: int = 10,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        gmc_method: str = "sparseOptFlow",
    ) -> None:
        self._cfg = dict(
            tracker_type="botsort",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
            # Camera motion compensation -- the reason we use BoT-SORT at all.
            gmc_method=gmc_method,
            # Appearance ReID is off in this phase; the re-ID phase owns identity.
            with_reid=False,
            proximity_thresh=0.5,
            appearance_thresh=0.25,
        )
        self._frame_rate = frame_rate
        self._tracker = None
        self.reset()

    def reset(self) -> None:
        from types import SimpleNamespace

        from ultralytics.trackers import BOTSORT

        self._tracker = BOTSORT(SimpleNamespace(**self._cfg), frame_rate=self._frame_rate)

    def update(
        self, detections: list[Detection], frame_bgr: np.ndarray, now: float
    ) -> list[Track]:
        # The frame is required for GMC; BoT-SORT reads it to estimate ego motion.
        raw = self._tracker.update(_DetectionShim(detections), frame_bgr)
        if raw is None or len(raw) == 0:
            return []

        tracks: list[Track] = []
        for row in np.asarray(raw):
            # BoT-SORT returns [x1, y1, x2, y2, track_id, conf, cls, det_idx].
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            tracks.append(
                Track(
                    track_id=int(row[4]),
                    detection=Detection(
                        x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=float(row[5])
                    ),
                    last_seen=now,
                )
            )
        return tracks

"""Person detectors behind one interface.

The interface exists so the GroundingDINO backend can be dropped in later without
touching the node, the tracker, or the servo loop. GDino needs a text prompt and a
local BERT directory, but those are constructor details invisible here -- which is
the test of whether the abstraction is real.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

import numpy as np

from .tracker import Detection

# COCO class 0 is "person" for all stock YOLO weights.
COCO_PERSON_CLASS = 0


class PersonDetector(Protocol):
    def warmup(self) -> None: ...

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...

    @property
    def name(self) -> str: ...


class NullDetector:
    """Detects nothing. Useful for exercising the state machine's empty paths."""

    @property
    def name(self) -> str:
        return "null"

    def warmup(self) -> None:
        return None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return []


class ReplayDetector:
    """Replays canned detections, so the whole downstream pipeline is testable
    with no detector, no GPU, and no camera."""

    def __init__(self, frames: list[list[Detection]], loop: bool = True) -> None:
        self._frames = frames
        self._loop = loop
        self._index = 0

    @property
    def name(self) -> str:
        return "replay"

    def warmup(self) -> None:
        return None

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self._frames:
            return []
        if self._index >= len(self._frames):
            if not self._loop:
                return []
            self._index = 0
        detections = self._frames[self._index]
        self._index += 1
        return detections


class YoloTorchDetector:
    """Ultralytics YOLO, torch backend.

    No TensorRT export in this phase: that is a latency optimisation, and it should
    follow a measurement rather than precede one. If the bench says we need it, the
    export is additive and this class stays as the fallback.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        confidence: float = 0.35,
        imgsz: int = 640,
        device: str = "cuda:0",
        half: bool = True,
        max_detections: int = 32,
    ) -> None:
        self._weights = weights
        self._confidence = confidence
        self._imgsz = imgsz
        self._device = device
        self._half = half
        self._max_detections = max_detections
        self._model = None

    @property
    def name(self) -> str:
        return f"yolo_torch({self._weights}@{self._imgsz})"

    def _ensure_loaded(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._weights)
            self._model.to(self._device)
        return self._model

    def warmup(self) -> None:
        model = self._ensure_loaded()
        blank = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        model.predict(
            blank,
            imgsz=self._imgsz,
            conf=self._confidence,
            device=self._device,
            half=self._half,
            verbose=False,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        model = self._ensure_loaded()
        # Ultralytics takes BGR arrays directly and handles the conversion.
        results = model.predict(
            frame_bgr,
            imgsz=self._imgsz,
            conf=self._confidence,
            device=self._device,
            half=self._half,
            classes=[COCO_PERSON_CLASS],
            max_det=self._max_detections,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [
            Detection(
                x=float(x1),
                y=float(y1),
                w=float(x2 - x1),
                h=float(y2 - y1),
                confidence=float(c),
            )
            for (x1, y1, x2, y2), c in zip(xyxy, conf)
        ]


# Backends are registered rather than if/elif'd so adding GroundingDINO is a
# single entry plus its module, with no edits to the node.
DetectorFactory = Callable[..., PersonDetector]

_REGISTRY: dict[str, DetectorFactory] = {
    "yolo": YoloTorchDetector,
    "null": NullDetector,
}


def register_detector(name: str, factory: DetectorFactory) -> None:
    _REGISTRY[name] = factory


def available_detectors() -> list[str]:
    return sorted(_REGISTRY)


def make_detector(backend: str, **kwargs) -> PersonDetector:
    try:
        factory = _REGISTRY[backend]
    except KeyError:
        raise ValueError(
            f"unknown detector_backend '{backend}'; available: {available_detectors()}"
        ) from None
    return factory(**kwargs)


class TimedDetector:
    """Wraps a detector to record per-call latency for the bench and telemetry."""

    def __init__(self, inner: PersonDetector) -> None:
        self._inner = inner
        self.last_latency_s = 0.0

    @property
    def name(self) -> str:
        return self._inner.name

    def warmup(self) -> None:
        self._inner.warmup()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        try:
            return self._inner.detect(frame_bgr)
        finally:
            self.last_latency_s = time.perf_counter() - started

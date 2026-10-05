"""Frame sources behind one interface.

The drone has no ROS image topic: the Gremsy payload serves RTSP and frames only
become ROS messages on the basestation. So this package opens its own RTSP client
rather than subscribing to anything.

Deliberately a *second, independent* RTSP session rather than a new tee branch in
gst_timestamp's sender.cpp: that file feeds the basestation video link and has a
watchdog that calls exit(1) if its SEI probe stalls, so adding a branch there
risks the operator's video feed. Opening our own session cannot affect it.

UNVERIFIED (Gate 0): whether the payload's RTSP server accepts concurrent
sessions. If it caps at one, this silently kills the basestation feed -- which is
worse than the tee risk. Test both streams together before trusting this path.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np


class ImageSource(Protocol):
    def start(self) -> None: ...

    def latest(self) -> tuple[np.ndarray, float] | None:
        """Most recent frame and its monotonic capture stamp, or None if none yet."""
        ...

    def stop(self) -> None: ...


def build_rtsp_pipeline(
    rtsp_url: str, latency_ms: int = 100, use_hardware_decode: bool = False,
    protocols: str = "tcp", raw_i420: bool = False
) -> str:
    """GStreamer pipeline string for cv2.VideoCapture(..., cv2.CAP_GSTREAMER).

    ``drop=true max-buffers=1`` matters: the servo loop wants the newest frame,
    never a queued backlog. A stale frame is worse than a dropped one -- it moves
    the gimbal based on where the person used to be.

    Two things here are load-bearing, both found the hard way on nx-03:

    * The sink must be a plain ``appsink`` with **no name=**. OpenCV's GStreamer
      backend fails to bind to a renamed sink and ``isOpened()`` silently returns
      False, with no error explaining why.
    * Hardware decode defaults **off**. ``nvvidconv`` needs an EGL display to
      convert NVMM buffers into system memory, and the drone is headless:
      "nvbufsurftransform: Could not get EGL display connection", then the
      pipeline fails to open. ``avdec_h264`` decodes 1080p fine for a 10 Hz loop.
      Only enable hardware decode once the EGL path is actually verified.

      ``protocols`` defaults to TCP because the payload serves no RTP over
      UDP: measured 2026-09-17, a UDP client completes DESCRIBE/SETUP/PLAY
      and then receives nothing, so rtspsrc sits in rtpjitterbuffer at 100%%
      of a core forever instead of falling back. The same URL over TCP
      negotiates H264 and plays.
    """
    decoder = (
        "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert"
        if use_hardware_decode
        else "avdec_h264 ! videoconvert"
    )
    if raw_i420:
        # No conversion in the pipeline: the consumer converts only the frames it uses.
        decoder = decoder.replace(" ! videoconvert", "")
    return (
        f"rtspsrc location={rtsp_url} latency={latency_ms} protocols={protocols} ! "
        f"rtph264depay ! h264parse ! {decoder} ! "
        f"video/x-raw,format={'I420' if raw_i420 else 'BGR'} ! "
        "appsink sync=false max-buffers=1 drop=true"
    )


def i420_to_half_bgr(frame: np.ndarray) -> np.ndarray:
    """I420 (OpenCV's h*3/2 x w buffer) to BGR at half size, without a full-size convert.

    The chroma planes are already half size, so every other luma pixel completes a
    half-size YUV image: ~1.5 ms where a full 1080p convert is ~31 ms on the Orin NX.
    """
    import cv2

    h, w = frame.shape[0] * 2 // 3, frame.shape[1]
    q = h * w // 4
    flat = frame.reshape(-1)
    y = frame[:h:2, ::2]
    u = flat[h * w:h * w + q].reshape(h // 2, w // 2)
    v = flat[h * w + q:h * w + 2 * q].reshape(h // 2, w // 2)
    return cv2.cvtColor(cv2.merge([y, u, v]), cv2.COLOR_YUV2BGR)


class RtspImageSource:
    """Reads frames from the payload's RTSP stream on a background thread.

    The reader thread always keeps only the newest frame, so a slow detector can
    never build a backlog of stale imagery.
    """

    def __init__(
        self,
        rtsp_url: str,
        latency_ms: int = 100,
        use_hardware_decode: bool = True,
        reconnect_delay_s: float = 2.0,
        half_resolution: bool = False,
    ) -> None:
        # Half resolution reads raw I420 and converts on latest(), so the reader thread
        # never converts the 2 in 3 frames the 10 Hz detector skips (spiritnx3,
        # 2026-10-04: ~105% -> ~57% of a core for 1080p30, and the streaming thread no
        # longer saturates and falls behind the live stream).
        self._half = half_resolution
        self._pipeline = build_rtsp_pipeline(
            rtsp_url, latency_ms, use_hardware_decode, raw_i420=half_resolution
        )
        self._rtsp_url = rtsp_url
        self._reconnect_delay_s = reconnect_delay_s
        self._capture = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp: float = 0.0
        self.frames_received = 0
        self.reconnects = 0

    @property
    def pipeline(self) -> str:
        return self._pipeline

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="rtsp-image-source", daemon=True
        )
        self._thread.start()

    def _open(self):
        import cv2

        capture = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _run(self) -> None:
        while self._running.is_set():
            if self._capture is None:
                self._capture = self._open()
                if self._capture is None:
                    time.sleep(self._reconnect_delay_s)
                    continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self._capture.release()
                self._capture = None
                self.reconnects += 1
                time.sleep(self._reconnect_delay_s)
                continue

            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
                self.frames_received += 1

    def latest(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._frame is None:
                return None
            frame, stamp = self._frame, self._stamp
        return (i420_to_half_bgr(frame) if self._half else frame), stamp

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class ReplayImageSource:
    """Plays a local video file at a fixed rate.

    Keeps the entire detector/tracker/controller/state-machine stack developable
    on a laptop with no drone, no gimbal, and no Orin.
    """

    def __init__(self, path: str, fps: float = 10.0, loop: bool = True) -> None:
        self._path = path
        self._period = 1.0 / max(fps, 1e-3)
        self._loop = loop
        self._capture = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp: float = 0.0
        self.exhausted = False

    def start(self) -> None:
        import cv2

        if self._thread is not None:
            return
        self._capture = cv2.VideoCapture(self._path)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open replay video: {self._path}")
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name="replay-image-source", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        import cv2

        while self._running.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self._loop:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.exhausted = True
                return
            with self._lock:
                self._frame = frame
                self._stamp = time.monotonic()
            time.sleep(self._period)

    def latest(self) -> tuple[np.ndarray, float] | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._stamp

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

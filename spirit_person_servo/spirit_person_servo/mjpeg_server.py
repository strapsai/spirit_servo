"""Minimal MJPEG-over-HTTP server for watching the servo loop live.

Deliberately not a ROS image topic: the drone runs ROS_DOMAIN_ID 3 and the
basestation runs 100, with no bridge entry for this feed, so a topic would need
domain-bridge config to be visible. An HTTP port on the drone is reachable from
the basestation today and opens in any browser.

Only ever serves the most recent frame -- a slow viewer drops frames rather than
building a backlog and delaying the picture.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_BOUNDARY = "servoframe"

_INDEX = b"""<!doctype html>
<title>person servo</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}
 header{padding:8px 12px;background:#1b1b1b;border-bottom:1px solid #333}
 img{display:block;width:100vw;height:auto}
</style>
<header>person servo &mdash; live detections</header>
<img src="/stream.mjpg">
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the ROS console clean
        return

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(_INDEX)))
            self.end_headers()
            self.wfile.write(_INDEX)
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}"
        )
        self.end_headers()

        store = self.server.frame_store
        last = None
        try:
            while True:
                frame = store.wait_for_frame(previous=last, timeout=5.0)
                if frame is None:
                    continue
                last = frame
                self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer closed the tab


class _FrameStore:
    """Holds one frame; readers wait for it to change."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: bytes | None = None

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._frame = jpeg
            self._cond.notify_all()

    def wait_for_frame(self, previous: bytes | None, timeout: float) -> bytes | None:
        with self._cond:
            if self._frame is None or self._frame is previous:
                self._cond.wait(timeout)
            return self._frame


class MjpegServer:
    def __init__(self, port: int, bind: str = "0.0.0.0") -> None:
        self._port = port
        self._bind = bind
        self._store = _FrameStore()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._port

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._httpd = ThreadingHTTPServer((self._bind, self._port), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.frame_store = self._store
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="mjpeg-server", daemon=True
        )
        self._thread.start()

    def publish(self, jpeg: bytes) -> None:
        self._store.publish(jpeg)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None

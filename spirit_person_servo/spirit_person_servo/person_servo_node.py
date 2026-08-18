"""Onboard person servoing node.

Runs entirely inside the drone's own ROS_DOMAIN_ID -- no basestation, no domain
bridge, no database. Detection and actuation live in one process because a
1920x1080 BGR frame is 6.2 MB and splitting the stages would mean DDS-serialising
that per stage.

Phase 1 target selection is "largest person, held with continuity". The re-ID
phase replaces only target_selector.py.

Timing note: internal staleness uses time.monotonic() throughout, matching the
image source's stamps. The ROS clock is used only for message headers, so a
simulated or stepped clock cannot stall the safety timer.
"""

from __future__ import annotations

import math
import os
import threading
import time
from enum import IntEnum

import rclpy
import yaml
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_srvs.srv import Trigger

from spirit_person_servo_msgs.msg import ServoState

from .backends import (
    GIMBAL_MODE_CMD_FOLLOW,
    ServoCommand,
    make_backend,
)
from .control import (
    AngleStepController,
    AxisPID,
    DeadbandHold,
    DivergenceGuard,
    Intrinsics,
    clamp,
    pixel_error_to_angles,
)
from .detector import TimedDetector, make_detector
from .image_source import ReplayImageSource, RtspImageSource
from .target_selector import TargetSelector
from .tracker import BotSortTracker

_BACKEND_ENUM = {
    "dry_run": ServoState.BACKEND_DRY_RUN,
    "track_touch": ServoState.BACKEND_TRACK_TOUCH,
    "angle": ServoState.BACKEND_ANGLE,
    "rate": ServoState.BACKEND_RATE,
    "single_axis_rate": ServoState.BACKEND_RATE,
}


class State(IntEnum):
    IDLE = ServoState.STATE_IDLE
    SEARCHING = ServoState.STATE_SEARCHING
    LOCKED = ServoState.STATE_LOCKED
    SERVOING = ServoState.STATE_SERVOING
    HOLD = ServoState.STATE_HOLD
    LOST = ServoState.STATE_LOST


def load_intrinsics(path: str, zoom: float = 1.0) -> Intrinsics:
    """Read fx/fy/cx/cy from config/reid/ufm/vehicles/<robot>_eo.yaml.

    Read from the shared calibration rather than copied into a servo config: a
    duplicated intrinsic that silently diverges from the real one is a classic.

    Two schemas are in use across the fleet and both are accepted:
      intrinsics: {fx: ...}                  # flat, fixed 1x
      intrinsics_by_zoom: {1.0: {fx: ...}}   # keyed by gimbal zoom_level
    """
    with open(path) as handle:
        data = yaml.safe_load(handle)

    table = data.get("intrinsics_by_zoom") or {}
    if table:
        # YAML keys may load as float or str depending on the writer.
        entry = table.get(zoom, table.get(str(zoom)))
        if entry is None:
            nearest = min(table, key=lambda k: abs(float(k) - zoom))
            entry = table[nearest]
    else:
        entry = data.get("intrinsics")

    if not entry:
        raise ValueError(f"no 'intrinsics' or 'intrinsics_by_zoom' in {path}")

    return Intrinsics(
        fx=float(entry["fx"]),
        fy=float(entry["fy"]),
        cx=float(entry["cx"]),
        cy=float(entry["cy"]),
        image_w=int(entry["image_w"]),
        image_h=int(entry["image_h"]),
    )


class PersonServoNode(Node):
    def __init__(self) -> None:
        super().__init__("person_servo_node")

        robot = os.environ.get("ROBOT_NAME", "spiritnx3")
        airlab = os.environ.get("AIRLAB_PATH", "/home/dtc/airlab_ws")

        # --- Parameters ---
        self.declare_parameter("robot_name", robot)
        self.declare_parameter("gimbal_namespace", f"/{robot}/gremsy")
        self.declare_parameter(
            "intrinsics_file", f"{airlab}/config/reid/ufm/vehicles/{robot}_eo.yaml"
        )

        self.declare_parameter("servo_backend", "dry_run")
        self.declare_parameter("detector_backend", "yolo")
        self.declare_parameter("image_source", "rtsp")

        self.declare_parameter("rtsp_url", "rtsp://192.168.70.23:8554/payload")
        self.declare_parameter("rtsp_latency_ms", 100)
        self.declare_parameter("rtsp_hardware_decode", False)
        self.declare_parameter("replay_video", "")
        self.declare_parameter("replay_fps", 10.0)

        self.declare_parameter("yolo_weights", "yolo11n.pt")
        self.declare_parameter("yolo_imgsz", 640)
        self.declare_parameter("yolo_confidence", 0.35)
        self.declare_parameter("yolo_device", "cuda:0")
        self.declare_parameter("yolo_half", True)

        self.declare_parameter("detect_rate_hz", 10.0)
        self.declare_parameter("control_rate_hz", 20.0)

        # Sign conventions are two independent unknowns (command->motion, and
        # motion->pixels) and must be measured, never guessed. The node refuses to
        # actuate until a calibration has been recorded.
        self.declare_parameter("pan_sign", 1.0)
        self.declare_parameter("tilt_sign", 1.0)
        self.declare_parameter("sign_calibration_verified", False)

        self.declare_parameter("angle_kp", 0.4)
        self.declare_parameter("max_step_deg", 3.0)
        self.declare_parameter("rate_kp", 1.2)
        self.declare_parameter("rate_ki", 0.15)
        self.declare_parameter("rate_kd", 0.0)
        self.declare_parameter("max_rate_dps", 3.0)
        self.declare_parameter("max_accel_dps2", 15.0)

        self.declare_parameter("tilt_min_deg", -90.0)
        self.declare_parameter("tilt_max_deg", 20.0)

        self.declare_parameter("enter_deadband_px", 25.0)
        self.declare_parameter("exit_deadband_px", 60.0)
        self.declare_parameter("hold_confirm_s", 0.4)
        self.declare_parameter("exit_confirm_s", 0.15)

        self.declare_parameter("image_timeout_s", 1.0)
        self.declare_parameter("detection_timeout_s", 0.5)
        self.declare_parameter("gimbal_timeout_s", 1.0)
        self.declare_parameter("search_timeout_s", 15.0)
        self.declare_parameter("lost_timeout_s", 8.0)
        self.declare_parameter("max_session_s", 120.0)

        # NOTE: asserting 1x zoom on arm (so the static intrinsics are true rather
        # than assumed) is deferred. Until then live zoom_level scales fx/fy, and
        # the loop derates when that reading is stale.
        self.declare_parameter("set_lock_mode", True)
        self.declare_parameter("restore_gimbal_mode", GIMBAL_MODE_CMD_FOLLOW)
        self.declare_parameter("publish_debug_image", False)
        # 0 disables. Serves annotated frames over HTTP for a browser.
        self.declare_parameter("mjpeg_port", 0)
        self.declare_parameter("debug_every_n", 2)

        # Which axis single_axis_rate drives. Only one Float64 rate topic may be
        # used at a time: cmd/gimbal_tilt and cmd/gimbal_pan each zero the other
        # axis, so publishing both alternates and cancels.
        self.declare_parameter("single_axis", "pan")

        # cmd/gimbal_angle yaw is NOT in the same frame as gimbal_state.pan_deg
        # (a +3 deg command produced a 202 deg slew on nx-03). The angle backend
        # refuses to start until this is explicitly set.
        self.declare_parameter("angle_frame_verified", False)
        self.declare_parameter("max_angle_jump_deg", 5.0)

        p = self.get_parameter
        self._robot = p("robot_name").value
        self._gimbal_ns = p("gimbal_namespace").value
        self._backend_name = p("servo_backend").value
        self._pan_sign = float(p("pan_sign").value)
        self._tilt_sign = float(p("tilt_sign").value)
        self._image_timeout_s = float(p("image_timeout_s").value)
        self._detection_timeout_s = float(p("detection_timeout_s").value)
        self._gimbal_timeout_s = float(p("gimbal_timeout_s").value)
        self._search_timeout_s = float(p("search_timeout_s").value)
        self._lost_timeout_s = float(p("lost_timeout_s").value)
        self._max_session_s = float(p("max_session_s").value)
        self._publish_debug_image = bool(p("publish_debug_image").value)

        # --- Safety gate -------------------------------------------------
        # Anything that actually moves the gimbal requires a recorded sign
        # calibration. Without it an inverted sign means a runaway slew.
        if self._backend_name != "dry_run" and not p("sign_calibration_verified").value:
            self.get_logger().error(
                f"servo_backend='{self._backend_name}' requested but "
                "sign_calibration_verified is false -- forcing dry_run. "
                "Run the sign calibration on a bench with props removed first."
            )
            self._backend_name = "dry_run"

        # --- Intrinsics ---
        intrinsics_file = p("intrinsics_file").value
        try:
            self._base_intrinsics = load_intrinsics(intrinsics_file)
            self.get_logger().info(f"intrinsics loaded from {intrinsics_file}")
        except Exception as exc:  # noqa: BLE001 - must not crash the node
            self.get_logger().error(
                f"could not load intrinsics from {intrinsics_file}: {exc}. "
                "Falling back to a nominal 1080p model; DO NOT fly on this."
            )
            self._base_intrinsics = Intrinsics(1378.08, 1375.56, 960.0, 540.0, 1920, 1080)

        # --- Control ---
        self._angle_ctrl = AngleStepController(
            kp=float(p("angle_kp").value),
            max_step_deg=float(p("max_step_deg").value),
            tilt_min_deg=float(p("tilt_min_deg").value),
            tilt_max_deg=float(p("tilt_max_deg").value),
        )
        self._pan_pid = AxisPID(
            kp=float(p("rate_kp").value),
            ki=float(p("rate_ki").value),
            kd=float(p("rate_kd").value),
            max_rate_dps=float(p("max_rate_dps").value),
            max_accel_dps2=float(p("max_accel_dps2").value),
        )
        self._tilt_pid = AxisPID(
            kp=float(p("rate_kp").value),
            ki=float(p("rate_ki").value),
            kd=float(p("rate_kd").value),
            max_rate_dps=float(p("max_rate_dps").value),
            max_accel_dps2=float(p("max_accel_dps2").value),
        )
        self._hold = DeadbandHold(
            enter_deadband_px=float(p("enter_deadband_px").value),
            exit_deadband_px=float(p("exit_deadband_px").value),
            hold_confirm_s=float(p("hold_confirm_s").value),
            exit_confirm_s=float(p("exit_confirm_s").value),
        )
        self._divergence = DivergenceGuard()

        # --- Perception ---
        self._tracker = BotSortTracker(frame_rate=int(float(p("detect_rate_hz").value)))
        self._selector = TargetSelector()
        self._detector = TimedDetector(self._build_detector())
        self._image_source = self._build_image_source()

        # --- State ---
        self._lock = threading.Lock()
        self._state = State.IDLE
        self._state_entered_s = time.monotonic()
        self._session_started_s = 0.0
        self._last_image_s = 0.0
        self._last_detection_s = 0.0
        self._last_gimbal_s = 0.0
        self._gimbal_pan_deg = 0.0
        self._gimbal_tilt_deg = 0.0
        self._gimbal_mode = 0
        self._zoom_level = 1.0
        self._n_detections = 0
        self._target = None
        self._err_px = (0.0, 0.0)
        self._frame_size = (0, 0)
        self._last_command = ServoCommand()
        self._actuation_idle = True
        self._status = "idle"
        self._last_control_s = time.monotonic()
        self._stop_repeats_left = 0
        self._debug_counter = 0

        # --- ROS interfaces ---
        self._backend = make_backend(
            self._backend_name, self, self._gimbal_ns, **self._backend_kwargs()
        )

        state_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._state_pub = self.create_publisher(ServoState, "~/state", state_qos)

        self._debug_pub = None
        if self._publish_debug_image:
            from sensor_msgs.msg import CompressedImage

            self._debug_pub = self.create_publisher(
                CompressedImage, "~/debug_image/compressed", 1
            )

        # Browser-viewable feed. Independent of ROS domains, so the basestation
        # can watch the drone's loop with no domain-bridge entry.
        self._mjpeg = None
        mjpeg_port = int(p("mjpeg_port").value)
        if mjpeg_port > 0:
            from .mjpeg_server import MjpegServer

            self._mjpeg = MjpegServer(mjpeg_port)
            self._mjpeg.start()
            self.get_logger().info(f"MJPEG debug view on http://<drone>:{mjpeg_port}/")

        self._subscribe_gimbal_state()

        # Perception is heavy and bursty; the control/safety timer must never wait
        # behind a YOLO inference, so they run in separate callback groups under a
        # MultiThreadedExecutor.
        self._perception_group = MutuallyExclusiveCallbackGroup()
        self._control_group = MutuallyExclusiveCallbackGroup()

        detect_period = 1.0 / max(float(p("detect_rate_hz").value), 1e-3)
        control_period = 1.0 / max(float(p("control_rate_hz").value), 1e-3)
        self._detect_timer = self.create_timer(
            detect_period, self._on_detect, callback_group=self._perception_group
        )
        self._control_timer = self.create_timer(
            control_period, self._on_control, callback_group=self._control_group
        )

        # Live-tunable gains. Reloading the node costs ~40 s (YOLO + RTSP warmup),
        # which makes gain sweeps impractical; these take effect on the next cycle.
        self.add_on_set_parameters_callback(self._on_param_update)

        self._start_srv = self.create_service(
            Trigger, "~/start", self._on_start, callback_group=self._control_group
        )
        self._stop_srv = self.create_service(
            Trigger, "~/stop", self._on_stop, callback_group=self._control_group
        )

        self._image_source.start()
        self.get_logger().info(
            f"person_servo_node up: robot={self._robot} backend={self._backend.name} "
            f"detector={self._detector.name} gimbal_ns={self._gimbal_ns}"
        )
        if self._backend_name == "dry_run":
            self.get_logger().warn("DRY RUN -- computing everything, commanding nothing.")

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_detector(self):
        p = self.get_parameter
        backend = p("detector_backend").value
        if backend == "yolo":
            return make_detector(
                "yolo",
                weights=p("yolo_weights").value,
                confidence=float(p("yolo_confidence").value),
                imgsz=int(p("yolo_imgsz").value),
                device=p("yolo_device").value,
                half=bool(p("yolo_half").value),
            )
        return make_detector(backend)

    def _backend_kwargs(self) -> dict:
        """Only pass options the chosen backend actually accepts.

        Passing them unconditionally would make a typo'd backend name fail with a
        confusing TypeError instead of the clear error from make_backend.
        """
        p = self.get_parameter
        if self._backend_name == "angle":
            return {
                "set_lock_mode": bool(p("set_lock_mode").value),
                "restore_gimbal_mode": int(p("restore_gimbal_mode").value),
                "angle_frame_verified": bool(p("angle_frame_verified").value),
                "max_jump_deg": float(p("max_angle_jump_deg").value),
            }
        if self._backend_name == "rate":
            return {
                "set_lock_mode": bool(p("set_lock_mode").value),
                "restore_gimbal_mode": int(p("restore_gimbal_mode").value),
            }
        if self._backend_name == "single_axis_rate":
            return {"axis": p("single_axis").value}
        return {}

    def _build_image_source(self):
        p = self.get_parameter
        kind = p("image_source").value
        if kind == "replay":
            path = p("replay_video").value
            if not path:
                raise ValueError("image_source='replay' requires replay_video")
            return ReplayImageSource(path, fps=float(p("replay_fps").value))
        return RtspImageSource(
            p("rtsp_url").value,
            latency_ms=int(p("rtsp_latency_ms").value),
            use_hardware_decode=bool(p("rtsp_hardware_decode").value),
        )

    def _subscribe_gimbal_state(self) -> None:
        """Subscribe to gimbal telemetry if lion_ros2_bridge is available.

        Kept optional so the package still builds and runs where that message
        package is missing -- it is declared in reid/db/basestation vcs files but
        was historically absent from spirit_drivers.yaml. Without it the angle
        backend cannot run, since it needs the current angle to step from.
        """
        self._gimbal_available = False
        try:
            from lion_ros2_bridge.msg import GimbalState
        except ImportError:
            self.get_logger().warn(
                "lion_ros2_bridge not available: no gimbal telemetry. "
                "The 'angle' backend and the tilt-limit guard are disabled."
            )
            return

        self._gimbal_available = True
        self.create_subscription(
            GimbalState, f"{self._gimbal_ns}/gimbal_state", self._on_gimbal_state, 10
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_param_update(self, params):
        """Apply gain/limit changes live, so tuning does not need a restart."""
        from rcl_interfaces.msg import SetParametersResult

        with self._lock:
            for prm in params:
                value = prm.value
                if prm.name == "rate_kp":
                    self._pan_pid.kp = self._tilt_pid.kp = float(value)
                elif prm.name == "rate_ki":
                    self._pan_pid.ki = self._tilt_pid.ki = float(value)
                    # Stale integral from the old gain would kick on the next cycle.
                    self._pan_pid.reset()
                    self._tilt_pid.reset()
                elif prm.name == "rate_kd":
                    self._pan_pid.kd = self._tilt_pid.kd = float(value)
                elif prm.name == "max_rate_dps":
                    self._pan_pid.max_rate_dps = self._tilt_pid.max_rate_dps = float(value)
                elif prm.name == "max_accel_dps2":
                    self._pan_pid.max_accel_dps2 = self._tilt_pid.max_accel_dps2 = float(value)
                elif prm.name == "enter_deadband_px":
                    self._hold.enter_deadband_px = float(value)
                elif prm.name == "exit_deadband_px":
                    self._hold.exit_deadband_px = float(value)
                else:
                    continue
                self.get_logger().info(f"param {prm.name} -> {value}")
        return SetParametersResult(successful=True)

    def _on_gimbal_state(self, msg) -> None:
        with self._lock:
            self._gimbal_pan_deg = float(msg.pan_deg)
            self._gimbal_tilt_deg = float(msg.tilt_deg)
            self._gimbal_mode = int(msg.mode)
            if msg.zoom_level > 0.0:
                self._zoom_level = float(msg.zoom_level)
            self._last_gimbal_s = time.monotonic()

    def _on_start(self, request, response):
        with self._lock:
            if self._state != State.IDLE:
                response.success = False
                response.message = f"already running (state={self._state.name})"
                return response
            self._selector.reset()
            self._tracker.reset()
            self._hold.reset()
            self._divergence.reset()
            self._pan_pid.reset()
            self._tilt_pid.reset()
            self._session_started_s = time.monotonic()
            self._transition(State.SEARCHING, "armed")
        self._backend.engage()
        response.success = True
        response.message = f"searching (backend={self._backend.name})"
        return response

    def _on_stop(self, request, response):
        self._disarm("stopped by service call")
        response.success = True
        response.message = "idle"
        return response

    def _on_detect(self) -> None:
        """Perception: grab a frame, detect, track, select. Commands nothing."""
        with self._lock:
            running = self._state != State.IDLE
        if not running:
            return

        latest = self._image_source.latest()
        if latest is None:
            return
        frame, stamp = latest

        detections = self._detector.detect(frame)
        now = time.monotonic()
        tracks = self._tracker.update(detections, frame, now)
        target = self._selector.select(tracks, now)

        with self._lock:
            self._last_image_s = stamp
            self._frame_size = (frame.shape[1], frame.shape[0])
            self._n_detections = len(detections)
            # Perception freshness is "did this cycle see anybody", NOT "did we
            # pick a target". Conflating them made every BoT-SORT track-id change
            # look like stale data: the selector coasts (max_coast_s) then needs
            # lock_votes_required frames to re-confirm, which together exceed
            # detection_timeout_s and forced a spurious SERVOING -> LOST.
            if detections:
                self._last_detection_s = now
            if target is not None:
                self._target = target
                self._last_target_s = now
                self._err_px = self._pixel_error(target, frame.shape[1], frame.shape[0])
            elif self._selector.held_id is None:
                # Not coasting -- the selector has genuinely let go, so stop
                # reporting a target that no longer exists.
                self._target = None

        self._debug_counter += 1
        if (self._debug_pub is not None or self._mjpeg is not None) and (
            self._debug_counter % max(int(self.get_parameter("debug_every_n").value), 1) == 0
        ):
            self._publish_debug(frame, tracks, target)

    def _pixel_error(self, target, width: int, height: int) -> tuple[float, float]:
        intr = self._effective_intrinsics(width, height)
        cx, cy = target.center
        return cx - intr.cx, cy - intr.cy

    def _effective_intrinsics(self, width: int, height: int) -> Intrinsics:
        intr = self._base_intrinsics.scaled_to(width, height)
        if self._zoom_valid():
            intr = intr.with_zoom(self._zoom_level)
        return intr

    def _zoom_valid(self) -> bool:
        if not self._gimbal_available:
            return False
        fresh = (time.monotonic() - self._last_gimbal_s) < 2.0
        return fresh and self._zoom_level > 0.0

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def _on_control(self) -> None:
        """Runs at a fixed rate regardless of the image callback.

        Independence is the point: if perception hangs or the camera dies, this
        timer still fires and still stops the gimbal.
        """
        now = time.monotonic()
        dt = max(now - self._last_control_s, 1e-3)
        self._last_control_s = now

        with self._lock:
            state = self._state
            self._flush_pending_stops()
            if state == State.IDLE:
                self._publish_state(idle=True)
                return

            image_age = now - self._last_image_s if self._last_image_s else math.inf
            detection_age = (
                now - self._last_detection_s if self._last_detection_s else math.inf
            )
            session_age = now - self._session_started_s

            if session_age > self._max_session_s:
                self._stop_motion("session timeout")
                self._transition(State.IDLE, "session timeout")
                self._publish_state(idle=True)
                return

            # Staleness: either feed going quiet must stop motion, regardless of
            # what the state machine thinks it is doing.
            if image_age > self._image_timeout_s or detection_age > self._detection_timeout_s:
                if state in (State.SERVOING, State.HOLD, State.LOCKED):
                    self._stop_motion(
                        f"stale data (image {image_age:.2f}s, detection {detection_age:.2f}s)"
                    )
                    self._transition(State.LOST, "stale data")
                elif state == State.LOST and now - self._state_entered_s > self._lost_timeout_s:
                    self._selector.release()
                    self._transition(State.SEARCHING, "reacquiring")
                elif state == State.SEARCHING and detection_age > self._search_timeout_s:
                    self._status = "searching: no people seen"
                self._publish_state(idle=True)
                return

            target = self._target
            if target is None:
                self._publish_state(idle=True)
                return

            if state in (State.SEARCHING, State.LOST):
                self._enter_servoing("target acquired")
                state = self._state

            err_x, err_y = self._err_px
            err_mag = math.hypot(err_x, err_y)
            should_drive = self._hold.update(err_mag, now)

            if not should_drive:
                if state != State.HOLD:
                    self._stop_motion("settled")
                    self._transition(State.HOLD, "holding")
                self._publish_state(idle=True)
                return

            if state == State.HOLD:
                self._transition(State.SERVOING, "target drifted")
                self._pan_pid.reset()
                self._tilt_pid.reset()
                self._divergence.reset()

            command = self._compute_command(err_x, err_y, dt)
            moved = self._backend.send(command)
            self._last_command = command
            self._actuation_idle = not moved

            if self._divergence.update(err_mag, commanding=moved):
                self.get_logger().error(
                    "SIGN CONVENTION LIKELY INVERTED: error grew for "
                    f"{self._divergence.max_consecutive} consecutive cycles while commanding "
                    "motion. Zeroing and aborting to LOST."
                )
                self._stop_motion("divergence guard tripped")
                self._transition(State.LOST, "divergence guard")

            self._publish_state(idle=self._actuation_idle)

    def _compute_command(self, err_x: float, err_y: float, dt: float) -> ServoCommand:
        width, height = self._frame_size
        intr = self._effective_intrinsics(width, height)
        err_yaw, err_pitch = pixel_error_to_angles(err_x, err_y, intr)

        # Wrong intrinsics make an aggressive loop oscillate; derate instead.
        derate = 1.0 if self._zoom_valid() else 0.5

        pan_rate = self._pan_pid.update(err_yaw * derate, dt) * self._pan_sign
        tilt_rate = self._tilt_pid.update(err_pitch * derate, dt) * self._tilt_sign

        angles_valid = self._gimbal_available and (
            time.monotonic() - self._last_gimbal_s
        ) < self._gimbal_timeout_s
        angle_pan, angle_tilt = self._gimbal_pan_deg, self._gimbal_tilt_deg
        if angles_valid:
            angle_pan, angle_tilt = self._angle_ctrl.step(
                self._gimbal_pan_deg,
                self._gimbal_tilt_deg,
                err_yaw * derate,
                err_pitch * derate,
                self._pan_sign,
                self._tilt_sign,
            )

        target = self._target
        touch_x, touch_y = target.center if target is not None else (0.0, 0.0)

        return ServoCommand(
            rate_pan_dps=pan_rate,
            rate_tilt_dps=tilt_rate,
            angle_pan_deg=angle_pan,
            angle_tilt_deg=angle_tilt,
            angles_valid=angles_valid,
            measured_pan_deg=self._gimbal_pan_deg,
            measured_tilt_deg=self._gimbal_tilt_deg,
            touch_px_x=clamp(touch_x, 0.0, float(max(width - 1, 0))),
            touch_px_y=clamp(touch_y, 0.0, float(max(height - 1, 0))),
            touch_valid=target is not None,
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _transition(self, state: State, reason: str) -> None:
        if state != self._state:
            self.get_logger().info(f"{self._state.name} -> {state.name} ({reason})")
            self._state = state
            self._state_entered_s = time.monotonic()
        self._status = reason

    def _enter_servoing(self, reason: str) -> None:
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._divergence.reset()
        self._hold.reset()
        self._transition(State.SERVOING, reason)

    def _stop_motion(self, reason: str) -> None:
        self._backend.stop()
        # Repeat on the next few control ticks. A single stop can be lost to a
        # dropped datagram, and for the rate backend the driver has no command
        # timeout -- an unheard stop means the gimbal keeps slewing.
        self._stop_repeats_left = 3
        self._pan_pid.reset()
        self._tilt_pid.reset()
        self._last_command = ServoCommand()
        self._actuation_idle = True
        self._status = reason

    def _flush_pending_stops(self) -> None:
        if self._stop_repeats_left > 0:
            self._stop_repeats_left -= 1
            self._backend.stop()

    def _disarm(self, reason: str) -> None:
        with self._lock:
            self._stop_motion(reason)
            self._selector.release()
            self._tracker.reset()
            self._target = None
            self._transition(State.IDLE, reason)
        # Outside the lock: restoring gimbal mode is I/O, not state.
        self._backend.disengage()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def _publish_state(self, idle: bool) -> None:
        now = time.monotonic()
        msg = ServoState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._robot
        msg.state = int(self._state)
        msg.backend = _BACKEND_ENUM.get(self._backend_name, ServoState.BACKEND_DRY_RUN)

        target = self._target
        msg.has_target = target is not None and self._state not in (State.IDLE,)
        msg.target_track_id = int(target.track_id) if target is not None else -1
        if target is not None:
            det = target.detection
            msg.bbox_x, msg.bbox_y = float(det.x), float(det.y)
            msg.bbox_w, msg.bbox_h = float(det.w), float(det.h)
            msg.detector_conf = float(det.confidence)
        msg.n_detections = int(self._n_detections)

        msg.err_px_x, msg.err_px_y = (float(v) for v in self._err_px)
        if self._frame_size[0] and self._frame_size[1]:
            intr = self._effective_intrinsics(*self._frame_size)
            yaw, pitch = pixel_error_to_angles(msg.err_px_x, msg.err_px_y, intr)
            msg.err_deg_yaw, msg.err_deg_pitch = float(yaw), float(pitch)

        # Report zeros when nothing was commanded, so the telemetry never shows a
        # rate next to actuation_idle=true and invite a misreading of the bag.
        cmd = ServoCommand() if idle else self._last_command
        msg.cmd_rate_pan_dps = float(cmd.rate_pan_dps)
        msg.cmd_rate_tilt_dps = float(cmd.rate_tilt_dps)
        msg.cmd_angle_pan_deg = float(cmd.angle_pan_deg)
        msg.cmd_angle_tilt_deg = float(cmd.angle_tilt_deg)
        msg.cmd_touch_px_x = float(cmd.touch_px_x)
        msg.cmd_touch_px_y = float(cmd.touch_px_y)

        msg.actuation_idle = bool(idle)
        msg.commands_enabled = bool(self._backend.commands_enabled)

        msg.gimbal_valid = self._gimbal_available and (
            now - self._last_gimbal_s
        ) < self._gimbal_timeout_s
        msg.gimbal_pan_deg = float(self._gimbal_pan_deg)
        msg.gimbal_tilt_deg = float(self._gimbal_tilt_deg)
        msg.gimbal_mode = int(self._gimbal_mode)

        msg.image_age_s = float(now - self._last_image_s) if self._last_image_s else -1.0
        msg.detection_age_s = (
            float(now - self._last_detection_s) if self._last_detection_s else -1.0
        )
        msg.gimbal_age_s = float(now - self._last_gimbal_s) if self._last_gimbal_s else -1.0
        msg.zoom_level = float(self._zoom_level)
        msg.zoom_valid = bool(self._zoom_valid())
        msg.status_text = self._status
        self._state_pub.publish(msg)

    def _render_debug(self, frame, tracks, target):
        """Annotate a frame with everything needed to debug the loop by eye."""
        import cv2

        canvas = frame.copy()
        height, width = canvas.shape[:2]
        intr = self._effective_intrinsics(width, height)
        cx, cy = int(intr.cx), int(intr.cy)

        for track in tracks:
            det = track.detection
            chosen = target is not None and track.track_id == target.track_id
            color = (0, 255, 0) if chosen else (140, 140, 140)
            p1 = (int(det.x), int(det.y))
            p2 = (int(det.x + det.w), int(det.y + det.h))
            cv2.rectangle(canvas, p1, p2, color, 3 if chosen else 1)
            cv2.putText(
                canvas, f"#{track.track_id} {det.confidence:.2f}",
                (p1[0], max(p1[1] - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            if chosen:
                tx, ty = (int(v) for v in track.center)
                cv2.line(canvas, (cx, cy), (tx, ty), (0, 255, 255), 2)
                cv2.circle(canvas, (tx, ty), 7, (0, 255, 255), -1)

        # image centre + the deadband the loop is trying to settle inside
        band = int(self._hold.enter_deadband_px)
        cv2.rectangle(canvas, (cx - band, cy - band), (cx + band, cy + band),
                      (255, 200, 0), 2)
        cv2.drawMarker(canvas, (cx, cy), (255, 200, 0), cv2.MARKER_CROSS, 26, 2)

        cmd = self._last_command
        err_x, err_y = self._err_px
        lines = [
            f"{self._state.name}  backend={self._backend.name}"
            f"{'' if self._backend.commands_enabled else ' (DRY RUN)'}",
            f"err {err_x:+7.1f},{err_y:+7.1f} px   cmd_pan {cmd.rate_pan_dps:+5.2f} dps"
            f"   idle={self._actuation_idle}",
            f"gimbal pan {self._gimbal_pan_deg:+7.2f}  tilt {self._gimbal_tilt_deg:+6.2f}"
            f"   dets={self._n_detections}",
            self._status,
        ]
        for i, text in enumerate(lines):
            y = 34 + i * 30
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (0, 0, 0), 4)
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (255, 255, 255), 2)

        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buffer.tobytes() if ok else None

    def _publish_debug(self, frame, tracks, target) -> None:
        jpeg = self._render_debug(frame, tracks, target)
        if jpeg is None:
            return

        if self._mjpeg is not None:
            self._mjpeg.publish(jpeg)

        if self._debug_pub is not None:
            from sensor_msgs.msg import CompressedImage

            out = CompressedImage()
            out.header.stamp = self.get_clock().now().to_msg()
            out.header.frame_id = "camera"
            out.format = "jpeg"
            out.data = jpeg
            self._debug_pub.publish(out)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop motion emphatically.

        Publisher destruction in ROS 2 is asynchronous, so a single publish right
        before shutdown() can be dropped on the floor. Repeat it.
        """
        self.get_logger().info("shutting down: stopping gimbal")
        for _ in range(5):
            try:
                self._backend.stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                break
            time.sleep(0.02)
        try:
            self._backend.disengage()
        except Exception:  # noqa: BLE001
            pass
        self._image_source.stop()
        if self._mjpeg is not None:
            self._mjpeg.stop()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PersonServoNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

"""Actuation backends -- the ways this node can move the gimbal.

Why more than one: `spirit_driver.cpp` maps `cmd/gimbal_tilt` and `cmd/gimbal_pan`
onto a single `setGimbalSpeed(pitch, roll, yaw, INPUT_SPEED)` call that always
carries a complete 3-axis setpoint:

    CMD_GIMBAL_TILT -> setGimbalSpeed(m.data, 0, 0, INPUT_SPEED)   # yaw forced to 0
    CMD_GIMBAL_PAN  -> setGimbalSpeed(0, 0, m.data, INPUT_SPEED)   # pitch forced to 0

So publishing both each cycle does NOT produce diagonal motion -- the two messages
alternate and cancel, each zeroing the axis the other just set. Two-axis rate
servoing is impossible through the topics that exist today. Hence:

* ANGLE  -- absolute setpoints via `cmd/gimbal_angle`. Works against the unmodified
  driver, drives both axes, and is the safest option: a dropped or stale command
  leaves the gimbal holding still rather than slewing.
* TRACK_TOUCH -- hands a pixel to the payload's own hardware tracker. No sign
  conventions, no tuning, no rate commands at all.
* RATE -- needs a new `cmd/gimbal_rate` (Vector3, INPUT_SPEED) topic added to
  gremsy_ros2. Inert until that lands; refuses to arm if nothing is subscribed.
* DRY_RUN -- computes and reports everything, sends nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64, Int32

# Relative topic names, matching gremsy_ros2/libs/gremsy_ros_topics.h. The node
# resolves them under the driver's namespace (default /<drone>/gremsy).
CMD_GIMBAL_TILT = "cmd/gimbal_tilt"
CMD_GIMBAL_PAN = "cmd/gimbal_pan"
CMD_GIMBAL_ANGLE = "cmd/gimbal_angle"
CMD_GIMBAL_MODE = "cmd/gimbal_mode"
CMD_TRACK_TOUCH = "cmd/track_touch"
CMD_TRACK = "cmd/track"
# PROPOSED -- not in gremsy_ros2 yet. See the plan's R1(a).
CMD_GIMBAL_RATE = "cmd/gimbal_rate"

# `cmd/gimbal_mode` takes the PAYLOAD_CAMERA_GIMBAL_MODE_* enum from
# libs/payload-define/mb1_sdk.h. This is NOT the same enum as GimbalState.mode
# (0=UNSPECIFIED, 1=FOLLOW, 2=LOCK, ...) -- do not cross-assign them.
GIMBAL_MODE_CMD_OFF = 0
GIMBAL_MODE_CMD_LOCK = 1
GIMBAL_MODE_CMD_FOLLOW = 2
GIMBAL_MODE_CMD_MAPPING = 3
GIMBAL_MODE_CMD_RESET = 4

# GimbalState.mode enum (lion_ros2_bridge/GimbalState) -- telemetry side.
GIMBAL_STATE_MODE_LOCK = 2


def clamp_to_window(value: float, centre: float, half_width: float) -> float:
    """Clamp ``value`` to +/- half_width around ``centre``."""
    if value < centre - half_width:
        return centre - half_width
    if value > centre + half_width:
        return centre + half_width
    return value


@dataclass(frozen=True)
class ServoCommand:
    """Everything the node computed this cycle. Each backend uses what it needs."""

    rate_pan_dps: float = 0.0
    rate_tilt_dps: float = 0.0
    angle_pan_deg: float = 0.0
    angle_tilt_deg: float = 0.0
    angles_valid: bool = False
    # Where the gimbal actually is, so a backend can bound its own setpoint.
    measured_pan_deg: float = 0.0
    measured_tilt_deg: float = 0.0
    touch_px_x: float = 0.0
    touch_px_y: float = 0.0
    touch_valid: bool = False


class ServoBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def commands_enabled(self) -> bool: ...

    def engage(self) -> None:
        """Called when the loop starts driving (assert gimbal mode, etc.)."""
        ...

    def send(self, command: ServoCommand) -> bool:
        """Issue one command. Returns True if actual motion was commanded."""
        ...

    def stop(self) -> None:
        """Bring the gimbal to rest. Must be safe to call repeatedly."""
        ...

    def disengage(self) -> None:
        """Release the gimbal: stop, then undo whatever engage() changed."""
        ...


class _BackendBase:
    def __init__(self, node, gimbal_ns: str) -> None:
        self._node = node
        self._ns = gimbal_ns.rstrip("/")

    def _topic(self, relative: str) -> str:
        return f"{self._ns}/{relative}"

    def engage(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def disengage(self) -> None:
        self.stop()


class DryRunBackend(_BackendBase):
    """Computes everything, sends nothing. The bring-up default."""

    @property
    def name(self) -> str:
        return "dry_run"

    @property
    def commands_enabled(self) -> bool:
        return False

    def send(self, command: ServoCommand) -> bool:
        return False


class AngleBackend(_BackendBase):
    """Absolute-angle position control via `cmd/gimbal_angle`.

    A dropped command leaves the gimbal holding still rather than slewing, which
    is why this looked like the safe option. It is NOT safe until the yaw frame
    is pinned down:

    !! MEASURED ON nx-03: commanding ``current_pan + 3 deg`` produced a 202 deg
       slew. In LOCK mode ``gimbal_state.pan_deg`` reports ``packet.yaw_absolute``
       (earth frame, payloadSdkInterface.cpp:1699) while ``cmd/gimbal_angle``
       INPUT_ANGLE expects body-frame yaw. Feeding one back as the other is an
       arbitrary slew, not a small step.

    Two consequences, both enforced below:
      * this backend refuses to run unless ``angle_frame_verified`` is set, and
      * every setpoint is clamped to ``max_jump_deg`` of the *measured* angle, so
        a frame error can never again produce an unbounded slew.

    Note ``cmd/gimbal_angle`` carries all three axes in one INPUT_ANGLE message,
    so there is no such thing as a tilt-only command here -- a yaw setpoint always
    goes with it. That is why the guard covers both axes.
    """

    def __init__(
        self,
        node,
        gimbal_ns: str,
        set_lock_mode: bool = True,
        restore_gimbal_mode: int = GIMBAL_MODE_CMD_FOLLOW,
        angle_frame_verified: bool = False,
        max_jump_deg: float = 5.0,
    ) -> None:
        super().__init__(node, gimbal_ns)
        if not angle_frame_verified:
            raise RuntimeError(
                "servo_backend='angle' refused: angle_frame_verified is false. "
                "cmd/gimbal_angle yaw is not in the same frame as "
                "gimbal_state.pan_deg (a +3 deg command produced a 202 deg slew "
                "on nx-03). Verify the yaw frame with a bounded rate probe and "
                "set angle_frame_verified:=true before using this backend."
            )
        self._set_lock_mode = set_lock_mode
        self._restore_gimbal_mode = restore_gimbal_mode
        self._max_jump_deg = max_jump_deg
        self._angle_pub = node.create_publisher(Vector3, self._topic(CMD_GIMBAL_ANGLE), 10)
        self._mode_pub = node.create_publisher(Int32, self._topic(CMD_GIMBAL_MODE), 10)
        self._track_pub = node.create_publisher(Int32, self._topic(CMD_TRACK), 10)

    @property
    def name(self) -> str:
        return "angle"

    @property
    def commands_enabled(self) -> bool:
        return True

    def engage(self) -> None:
        # LOCK holds an earth-frame attitude, which is what makes "stop commanding
        # and stay put" correct while the airframe moves. In FOLLOW the gimbal
        # tracks the airframe and the hold would drift.
        if self._set_lock_mode:
            self._mode_pub.publish(Int32(data=GIMBAL_MODE_CMD_LOCK))
        # Two controllers driving one gimbal is guaranteed oscillation.
        self._track_pub.publish(Int32(data=0))

    def send(self, command: ServoCommand) -> bool:
        if not command.angles_valid:
            return False

        # Last-ditch bound: never command further than max_jump_deg from where the
        # gimbal actually is. If the frame is wrong this turns an unbounded slew
        # into a small wrong step that the divergence guard will then catch.
        pan = clamp_to_window(
            command.angle_pan_deg, command.measured_pan_deg, self._max_jump_deg
        )
        tilt = clamp_to_window(
            command.angle_tilt_deg, command.measured_tilt_deg, self._max_jump_deg
        )

        # Vector3 is (pitch, roll, yaw) for this topic -- see gremsy_ros_topics.h.
        self._angle_pub.publish(Vector3(x=float(tilt), y=0.0, z=float(pan)))
        return True

    def stop(self) -> None:
        # Nothing to do: absolute-angle control has no latched motion to cancel.
        # Ceasing to publish *is* the stop, which is the point of this backend.
        return None

    def disengage(self) -> None:
        if self._set_lock_mode:
            self._mode_pub.publish(Int32(data=self._restore_gimbal_mode))


class TrackTouchBackend(_BackendBase):
    """Hands the target's pixel centre to the payload's own hardware tracker.

    Zero sign conventions, zero gain tuning, and no upstream driver change. The
    trade is observability: we cannot see what the payload actually locked onto,
    and its reacquisition behaviour is undocumented.
    """

    def __init__(self, node, gimbal_ns: str, retouch_period_s: float = 1.0) -> None:
        super().__init__(node, gimbal_ns)
        self._touch_pub = node.create_publisher(Vector3, self._topic(CMD_TRACK_TOUCH), 10)
        self._track_pub = node.create_publisher(Int32, self._topic(CMD_TRACK), 10)
        self._retouch_period_s = retouch_period_s
        self._last_touch_s = 0.0

    @property
    def name(self) -> str:
        return "track_touch"

    @property
    def commands_enabled(self) -> bool:
        return True

    def engage(self) -> None:
        self._track_pub.publish(Int32(data=1))
        self._last_touch_s = 0.0

    def send(self, command: ServoCommand) -> bool:
        if not command.touch_valid:
            return False
        # Re-touching every frame would fight the payload's own tracker; nudge it
        # only periodically, when our idea of the target has settled.
        now = self._node.get_clock().now().nanoseconds * 1e-9
        if now - self._last_touch_s < self._retouch_period_s:
            return False
        self._last_touch_s = now
        self._touch_pub.publish(
            Vector3(x=float(command.touch_px_x), y=float(command.touch_px_y), z=0.0)
        )
        return True

    def stop(self) -> None:
        self._track_pub.publish(Int32(data=0))
        self._last_touch_s = 0.0


class RateBackend(_BackendBase):
    """Angular-rate control.

    Requires a `cmd/gimbal_rate` (Vector3 pitch/roll/yaw deg/s -> INPUT_SPEED)
    topic that does not exist in gremsy_ros2 yet. It deliberately does NOT fall
    back to cmd/gimbal_tilt + cmd/gimbal_pan, because those cancel each other.

    This is the only backend where a crash leaves motion latched, so it is the
    reason gimbal_deadman_node exists.
    """

    def __init__(
        self,
        node,
        gimbal_ns: str,
        set_lock_mode: bool = True,
        restore_gimbal_mode: int = GIMBAL_MODE_CMD_FOLLOW,
    ) -> None:
        super().__init__(node, gimbal_ns)
        self._set_lock_mode = set_lock_mode
        self._restore_gimbal_mode = restore_gimbal_mode
        self._rate_pub = node.create_publisher(Vector3, self._topic(CMD_GIMBAL_RATE), 10)
        self._mode_pub = node.create_publisher(Int32, self._topic(CMD_GIMBAL_MODE), 10)
        self._track_pub = node.create_publisher(Int32, self._topic(CMD_TRACK), 10)

    @property
    def name(self) -> str:
        return "rate"

    @property
    def commands_enabled(self) -> bool:
        return True

    def driver_supports_rate(self) -> bool:
        """False means no driver is listening -- cmd/gimbal_rate does not exist yet."""
        return self._rate_pub.get_subscription_count() > 0

    def engage(self) -> None:
        if self._set_lock_mode:
            self._mode_pub.publish(Int32(data=GIMBAL_MODE_CMD_LOCK))
        self._track_pub.publish(Int32(data=0))

    def disengage(self) -> None:
        self.stop()
        if self._set_lock_mode:
            self._mode_pub.publish(Int32(data=self._restore_gimbal_mode))

    def send(self, command: ServoCommand) -> bool:
        self._rate_pub.publish(
            Vector3(x=float(command.rate_tilt_dps), y=0.0, z=float(command.rate_pan_dps))
        )
        return command.rate_tilt_dps != 0.0 or command.rate_pan_dps != 0.0

    def stop(self) -> None:
        self._rate_pub.publish(Vector3(x=0.0, y=0.0, z=0.0))


class SingleAxisRateBackend(_BackendBase):
    """One-axis rate control against the *unmodified* driver.

    Legitimate only because a single axis never needs the other one held: with
    just `cmd/gimbal_tilt` in play there is no second message zeroing it. Useful
    for bench sign-calibration before the dual-axis path exists.
    """

    def __init__(self, node, gimbal_ns: str, axis: str = "tilt") -> None:
        super().__init__(node, gimbal_ns)
        if axis not in ("tilt", "pan"):
            raise ValueError(f"axis must be 'tilt' or 'pan', got {axis!r}")
        self._axis = axis
        topic = CMD_GIMBAL_TILT if axis == "tilt" else CMD_GIMBAL_PAN
        self._pub = node.create_publisher(Float64, self._topic(topic), 10)

    @property
    def name(self) -> str:
        return f"single_axis_rate({self._axis})"

    @property
    def commands_enabled(self) -> bool:
        return True

    def send(self, command: ServoCommand) -> bool:
        value = command.rate_tilt_dps if self._axis == "tilt" else command.rate_pan_dps
        self._pub.publish(Float64(data=float(value)))
        return value != 0.0

    def stop(self) -> None:
        self._pub.publish(Float64(data=0.0))


BACKEND_NAMES = ("dry_run", "angle", "track_touch", "rate", "single_axis_rate")


def make_backend(backend: str, node, gimbal_ns: str, **kwargs) -> ServoBackend:
    if backend == "dry_run":
        return DryRunBackend(node, gimbal_ns)
    if backend == "angle":
        return AngleBackend(node, gimbal_ns, **kwargs)
    if backend == "track_touch":
        return TrackTouchBackend(node, gimbal_ns, **kwargs)
    if backend == "rate":
        return RateBackend(node, gimbal_ns, **kwargs)
    if backend == "single_axis_rate":
        return SingleAxisRateBackend(node, gimbal_ns, **kwargs)
    raise ValueError(f"unknown servo_backend '{backend}'; available: {list(BACKEND_NAMES)}")

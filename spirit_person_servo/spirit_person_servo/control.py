"""Control laws for person servoing. ROS-free and GPU-free so it is unit-testable.

Two control strategies live here because the Gremsy driver exposes two usable
actuation paths (see docs/person-servoing-spirit-indago.md and the plan):

* ``AngleStepController`` -- absolute angle setpoints via ``cmd/gimbal_angle``.
  Preferred: a dropped or stale command leaves the gimbal holding still.
* ``AxisPID`` -- angular rate via ``cmd/gimbal_rate``. A latched rate command
  slews forever if the commander dies, so it is guarded by the deadman node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for one zoom level, in the resolution they were calibrated at."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_w: int
    image_h: int

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Rescale to the actual decoded frame size.

        The payload's stream resolution is reconfigurable, so the calibrated
        resolution is not guaranteed to match what we decode.
        """
        sx = width / float(self.image_w)
        sy = height / float(self.image_h)
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            image_w=width,
            image_h=height,
        )

    def with_zoom(self, zoom: float) -> "Intrinsics":
        """Optical zoom multiplies focal length; the principal point is unchanged."""
        z = max(zoom, 1e-6)
        return Intrinsics(
            fx=self.fx * z,
            fy=self.fy * z,
            cx=self.cx,
            cy=self.cy,
            image_w=self.image_w,
            image_h=self.image_h,
        )


def pixel_error_to_angles(
    err_px_x: float, err_px_y: float, intr: Intrinsics
) -> tuple[float, float]:
    """Convert a pixel offset from the principal point into yaw/pitch error in degrees.

    Uses atan rather than the small-angle approximation: at the edge of a 1920-wide
    frame with fx=1378 the true angle is 34.9 deg where small-angle gives 39.9 -- a
    14% overshoot exactly when the loop is most aggressive.
    """
    yaw_deg = math.degrees(math.atan2(err_px_x, intr.fx))
    pitch_deg = math.degrees(math.atan2(err_px_y, intr.fy))
    return yaw_deg, pitch_deg


@dataclass
class AxisPID:
    """Parallel PI with derivative-on-measurement.

    Input is angular error in degrees and output is deg/s, so ``kp`` is
    dimensionless: kp=1.2 means "close the error with a ~1.2 s time constant".
    """

    kp: float = 1.2
    ki: float = 0.15
    kd: float = 0.0
    max_rate_dps: float = 3.0
    max_accel_dps2: float = 30.0
    integral_limit: float = 5.0

    _integral: float = field(default=0.0, init=False)
    _prev_measurement: float | None = field(default=None, init=False)
    _prev_output: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_output = 0.0

    def update(self, error_deg: float, dt: float, *, freeze_integral: bool = False) -> float:
        if dt <= 0.0:
            return self._prev_output

        proportional = self.kp * error_deg

        derivative = 0.0
        if self.kd != 0.0 and self._prev_measurement is not None:
            # Derivative on measurement: differentiating the error would kick on
            # every setpoint change (i.e. every time the target track switches).
            derivative = -self.kd * (error_deg - self._prev_measurement) / dt
        self._prev_measurement = error_deg

        unsaturated = proportional + self.ki * self._integral + derivative
        saturated = clamp(unsaturated, -self.max_rate_dps, self.max_rate_dps)

        # Anti-windup: only integrate when not fighting a saturated output.
        if not freeze_integral and unsaturated == saturated:
            self._integral = clamp(
                self._integral + error_deg * dt, -self.integral_limit, self.integral_limit
            )

        # Slew limit: stops a detector blink from becoming a step command.
        max_delta = self.max_accel_dps2 * dt
        output = clamp(saturated, self._prev_output - max_delta, self._prev_output + max_delta)
        output = clamp(output, -self.max_rate_dps, self.max_rate_dps)
        self._prev_output = output
        return output

    def decay_toward_zero(self, dt: float) -> float:
        """Ramp the last command down to zero, respecting the slew limit.

        Used when data goes stale: snapping to zero is a step input, and
        continuing to integrate a stale error is worse.
        """
        max_delta = self.max_accel_dps2 * dt
        if abs(self._prev_output) <= max_delta:
            self._prev_output = 0.0
        else:
            self._prev_output -= math.copysign(max_delta, self._prev_output)
        return self._prev_output


@dataclass
class AngleStepController:
    """Absolute-angle position control.

    Each cycle commands ``current + kp * error``, re-reading the measured angle
    every time, so it is proportional feedback rather than an accumulator -- if
    the gimbal has not moved yet, the same absolute target is re-sent rather than
    the setpoint creeping away.
    """

    kp: float = 0.4
    max_step_deg: float = 3.0
    tilt_min_deg: float = -90.0
    tilt_max_deg: float = 20.0

    def reset(self) -> None:  # kept for interface parity with AxisPID
        return None

    def step(
        self,
        current_pan_deg: float,
        current_tilt_deg: float,
        err_deg_yaw: float,
        err_deg_pitch: float,
        pan_sign: float,
        tilt_sign: float,
    ) -> tuple[float, float]:
        pan_step = clamp(self.kp * err_deg_yaw, -self.max_step_deg, self.max_step_deg)
        tilt_step = clamp(self.kp * err_deg_pitch, -self.max_step_deg, self.max_step_deg)

        target_pan = current_pan_deg + pan_sign * pan_step
        target_tilt = current_tilt_deg + tilt_sign * tilt_step

        target_tilt = clamp(target_tilt, self.tilt_min_deg, self.tilt_max_deg)
        target_pan = wrap_deg_180(target_pan)
        return target_pan, target_tilt


def wrap_deg_180(deg: float) -> float:
    """Wrap to [-180, 180]; the SDK rejects yaw setpoints outside that range."""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class DeadbandHold:
    """Hysteretic settle-and-hold.

    Separate enter/exit thresholds so detector jitter around the deadband edge
    cannot chatter the loop between driving and holding.
    """

    enter_deadband_px: float = 25.0
    exit_deadband_px: float = 60.0
    hold_confirm_s: float = 0.4
    exit_confirm_s: float = 0.15

    _holding: bool = field(default=False, init=False)
    _inside_since: float | None = field(default=None, init=False)
    _outside_since: float | None = field(default=None, init=False)

    @property
    def holding(self) -> bool:
        return self._holding

    def reset(self, *, holding: bool = False) -> None:
        self._holding = holding
        self._inside_since = None
        self._outside_since = None

    def update(self, err_px: float, now: float) -> bool:
        """Feed the current pixel error magnitude; returns True when the loop should drive."""
        if self._holding:
            if err_px > self.exit_deadband_px:
                if self._outside_since is None:
                    self._outside_since = now
                elif now - self._outside_since >= self.exit_confirm_s:
                    self._holding = False
                    self._inside_since = None
                    self._outside_since = None
            else:
                self._outside_since = None
        else:
            if err_px < self.enter_deadband_px:
                if self._inside_since is None:
                    self._inside_since = now
                elif now - self._inside_since >= self.hold_confirm_s:
                    self._holding = True
                    self._inside_since = None
                    self._outside_since = None
            else:
                self._inside_since = None

        return not self._holding


@dataclass
class DivergenceGuard:
    """Trips when the error grows while we are actively commanding motion.

    This is the cheap insurance against an inverted sign convention: a runaway
    slew shows up as monotonically increasing error under non-zero command.
    """

    max_consecutive: int = 8
    min_growth_px: float = 2.0

    _count: int = field(default=0, init=False)
    _prev_err: float | None = field(default=None, init=False)

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
        self._prev_err = None

    def update(self, err_px: float, commanding: bool) -> bool:
        """Returns True if the guard has tripped."""
        if not commanding:
            self.reset()
            return False

        if self._prev_err is not None and err_px > self._prev_err + self.min_growth_px:
            self._count += 1
        else:
            self._count = 0
        self._prev_err = err_px
        return self._count >= self.max_consecutive

"""Measure the gimbal sign conventions, safely.

Two independent unknowns have to be measured, never guessed:
  1. command -> motion   (does +rate move pan positive?)
  2. motion  -> pixels   (does +pan move the target left or right in frame?)

USE RATE PROBES, NOT ANGLE PROBES. A rate command is bounded by construction:
2 deg/s for 1 s is 2 deg of travel however wrong the convention turns out to be.
An absolute-angle probe has no such bound -- if the commanded frame differs from
the telemetry frame, "current + 3 deg" is an arbitrary slew. That is not
hypothetical: on nx-03 it produced a 202 deg slew and left the gimbal hunting.

Everything here is gated. Probe 1 must pass before probe 2 runs, and any probe
whose measured travel disagrees with the prediction aborts the whole run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ProbeResult:
    axis: str
    commanded_dps: float
    duration_s: float
    expected_deg: float
    measured_deg: float
    pixel_delta: float

    @property
    def follows_command(self) -> bool:
        """True when the gimbal moved the way the command implies."""
        return self.measured_deg * self.commanded_dps > 0.0

    @property
    def travel_ratio(self) -> float:
        if self.expected_deg == 0.0:
            return 0.0
        return abs(self.measured_deg) / abs(self.expected_deg)

    @property
    def deg_per_px(self) -> float:
        return self.pixel_delta / self.measured_deg if self.measured_deg else 0.0


class CalibrationAbort(RuntimeError):
    """Raised the moment a probe looks wrong. Never continue past this."""


def check_travel(result: ProbeResult, tolerance_deg: float = 1.5,
                 max_ratio: float = 3.0) -> None:
    """Interlock: the gimbal must have moved roughly as far as we asked.

    This is the check whose absence caused the 202 deg slew. It existed as a
    printed diagnostic; it needed to be an abort.
    """
    if abs(result.measured_deg) < 0.2:
        raise CalibrationAbort(
            f"{result.axis}: commanded {result.expected_deg:+.2f} deg but the gimbal "
            f"barely moved ({result.measured_deg:+.2f} deg). Command not taking "
            f"effect -- check the driver and gimbal mode."
        )
    if result.travel_ratio > max_ratio or (
        abs(result.measured_deg - result.expected_deg) > tolerance_deg
        and result.travel_ratio > max_ratio
    ):
        raise CalibrationAbort(
            f"{result.axis}: commanded {result.expected_deg:+.2f} deg but measured "
            f"{result.measured_deg:+.2f} deg (ratio {result.travel_ratio:.1f}x). "
            f"The command and telemetry frames disagree -- ABORTING before this "
            f"becomes a slew."
        )


def sign_from(result: ProbeResult, min_pixel_delta: float = 8.0) -> float:
    """Derive the servo sign for one axis.

    We want a target at positive pixel error to be driven toward zero, so the
    commanded direction must be opposite to the direction that increases the
    pixel coordinate.
    """
    if abs(result.pixel_delta) < min_pixel_delta:
        raise CalibrationAbort(
            f"{result.axis}: target moved only {result.pixel_delta:+.1f} px for "
            f"{result.measured_deg:+.2f} deg of gimbal travel. Too small to infer a "
            f"sign -- move the target closer to the camera, or increase the probe."
        )
    px_per_deg = result.pixel_delta / result.measured_deg
    return -1.0 if px_per_deg > 0 else 1.0


def format_yaml(pan_sign: float | None, tilt_sign: float | None) -> str:
    """The exact block to paste into config/<drone>.yaml."""
    lines = ["    # measured by sign_calibration -- do not hand-edit"]
    if pan_sign is not None:
        lines.append(f"    pan_sign: {pan_sign:+.1f}")
    if tilt_sign is not None:
        lines.append(f"    tilt_sign: {tilt_sign:+.1f}")
    if pan_sign is not None and tilt_sign is not None:
        lines.append("    sign_calibration_verified: true")
    else:
        lines.append("    # sign_calibration_verified: NOT set -- an axis is unmeasured")
    return "\n".join(lines)

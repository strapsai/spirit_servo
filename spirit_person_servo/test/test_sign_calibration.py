"""Tests for the calibration interlocks.

The 202 deg slew on nx-03 happened because the travel check was a printed
diagnostic instead of an abort. These tests pin it down as an abort.
"""

import pytest

from spirit_person_servo.sign_calibration import (
    CalibrationAbort,
    ProbeResult,
    check_travel,
    format_yaml,
    sign_from,
)


def probe(measured, expected=2.0, pixel_delta=50.0, dps=2.0, axis="pan"):
    return ProbeResult(
        axis=axis,
        commanded_dps=dps,
        duration_s=1.0,
        expected_deg=expected,
        measured_deg=measured,
        pixel_delta=pixel_delta,
    )


class TestTravelInterlock:
    def test_accepts_matching_travel(self):
        check_travel(probe(measured=2.0))
        check_travel(probe(measured=1.7))
        check_travel(probe(measured=2.4))

    def test_aborts_on_the_202_degree_case(self):
        """The real failure: commanded ~3 deg, gimbal moved 202 deg."""
        with pytest.raises(CalibrationAbort, match="frames disagree"):
            check_travel(probe(measured=202.84, expected=3.0))

    def test_aborts_when_gimbal_does_not_move(self):
        with pytest.raises(CalibrationAbort, match="barely moved"):
            check_travel(probe(measured=0.05))

    def test_aborts_on_large_overtravel_even_if_signed_correctly(self):
        with pytest.raises(CalibrationAbort):
            check_travel(probe(measured=20.0, expected=2.0))

    def test_inverted_but_bounded_travel_is_not_an_abort(self):
        """An inverted sign is a finding, not a fault -- that's what we're measuring."""
        check_travel(probe(measured=-1.9, expected=2.0))

    def test_follows_command_reports_inversion(self):
        assert probe(measured=2.0).follows_command is True
        assert probe(measured=-2.0).follows_command is False


class TestSignDerivation:
    def test_positive_px_per_deg_gives_negative_sign(self):
        # gimbal moved +2 deg and the target moved +50 px -> to reduce a positive
        # error we must command the opposite direction.
        assert sign_from(probe(measured=2.0, pixel_delta=50.0)) == -1.0

    def test_negative_px_per_deg_gives_positive_sign(self):
        assert sign_from(probe(measured=2.0, pixel_delta=-50.0)) == +1.0

    def test_inverted_motion_flips_the_sign(self):
        assert sign_from(probe(measured=-2.0, pixel_delta=50.0)) == +1.0

    def test_aborts_when_pixels_barely_move(self):
        with pytest.raises(CalibrationAbort, match="Too small"):
            sign_from(probe(measured=2.0, pixel_delta=3.0))

    def test_real_tilt_measurement_from_nx03(self):
        """The one axis that did calibrate cleanly: +2.91 deg -> +8.6 px."""
        r = probe(measured=2.91, expected=3.0, pixel_delta=8.625, axis="tilt")
        check_travel(r)
        assert sign_from(r) == -1.0


class TestFormatYaml:
    def test_both_axes_sets_verified_flag(self):
        out = format_yaml(-1.0, -1.0)
        assert "pan_sign: -1.0" in out
        assert "tilt_sign: -1.0" in out
        assert "sign_calibration_verified: true" in out

    def test_missing_axis_withholds_verified_flag(self):
        out = format_yaml(None, -1.0)
        assert "sign_calibration_verified: true" not in out
        assert "NOT set" in out


class TestPixelConventionRegression:
    """The convention that caused the gimbal to servo AWAY from the target.

    pixel_delta must be the scene shift as-measured, NOT negated: the target is
    part of the scene, so it moves with it. Negating produced a sign that drove
    the error larger instead of smaller.
    """

    def test_scene_moving_right_under_positive_command_gives_negative_sign(self):
        # camera commanded +, gimbal moved +2 deg, scene (and target) moved +45 px
        r = probe(measured=2.0, expected=2.0, pixel_delta=45.0)
        assert sign_from(r) == -1.0

    def test_the_corrected_nx03_pan_measurement(self):
        """Raw phase-correlation dx was -45 px for +1.91 deg on nx-03.

        With the negation bug that was recorded as +45 -> pan_sign -1.0, which
        drove the gimbal away. Un-negated it is -45 -> pan_sign +1.0.
        """
        buggy = probe(measured=1.91, expected=2.0, pixel_delta=+45.0)
        fixed = probe(measured=1.91, expected=2.0, pixel_delta=-45.0)
        assert sign_from(buggy) == -1.0     # what we shipped, and it was wrong
        assert sign_from(fixed) == +1.0     # what the loop actually needs
        assert sign_from(buggy) != sign_from(fixed)

    def test_sign_is_what_shrinks_a_positive_error(self):
        """Directly assert the control intent rather than the formula."""
        r = probe(measured=2.0, expected=2.0, pixel_delta=-45.0)
        sign = sign_from(r)
        px_per_deg_of_command = r.pixel_delta / r.measured_deg
        err = +100.0                       # target right of centre
        commanded = sign * err             # what the loop would command
        predicted_pixel_change = commanded * px_per_deg_of_command
        assert predicted_pixel_change < 0, "command must reduce a positive error"

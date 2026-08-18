"""Control-law tests. No ROS, no GPU, no ultralytics.

The property tests here encode the safety invariants that keep a gimbal from
running away. They are the reason control.py is ROS-free.
"""

import math
import random

import pytest

from spirit_person_servo.control import (
    AngleStepController,
    AxisPID,
    DeadbandHold,
    DivergenceGuard,
    Intrinsics,
    pixel_error_to_angles,
    wrap_deg_180,
)

SPIRIT_EO = Intrinsics(fx=1378.08, fy=1375.56, cx=933.483, cy=512.023, image_w=1920, image_h=1080)


class TestIntrinsics:
    def test_scaled_to_half_resolution_halves_focal_and_center(self):
        scaled = SPIRIT_EO.scaled_to(960, 540)
        assert scaled.fx == pytest.approx(689.04)
        assert scaled.cx == pytest.approx(466.7415)
        assert (scaled.image_w, scaled.image_h) == (960, 540)

    def test_zoom_scales_focal_but_not_principal_point(self):
        zoomed = SPIRIT_EO.with_zoom(2.0)
        assert zoomed.fx == pytest.approx(2756.16)
        assert zoomed.cx == pytest.approx(SPIRIT_EO.cx)

    def test_zoom_of_zero_does_not_divide_by_zero(self):
        assert SPIRIT_EO.with_zoom(0.0).fx > 0.0


class TestPixelErrorToAngles:
    def test_zero_error_is_zero_angle(self):
        assert pixel_error_to_angles(0.0, 0.0, SPIRIT_EO) == (0.0, 0.0)

    def test_uses_atan_not_small_angle(self):
        # 960 px at fx=1378.08 -> atan gives 34.87 deg; small-angle would give 39.9.
        yaw, _ = pixel_error_to_angles(960.0, 0.0, SPIRIT_EO)
        assert yaw == pytest.approx(34.87, abs=0.1)
        small_angle = math.degrees(960.0 / SPIRIT_EO.fx)
        assert small_angle - yaw > 4.0

    def test_sign_is_preserved(self):
        yaw, pitch = pixel_error_to_angles(-100.0, -50.0, SPIRIT_EO)
        assert yaw < 0.0 and pitch < 0.0


class TestAxisPID:
    def test_output_never_exceeds_max_rate(self):
        pid = AxisPID(kp=5.0, ki=1.0, max_rate_dps=3.0, max_accel_dps2=1000.0)
        for _ in range(200):
            out = pid.update(45.0, dt=0.05)
            assert abs(out) <= 3.0 + 1e-9

    def test_slew_limit_bounds_first_step(self):
        pid = AxisPID(kp=5.0, ki=0.0, max_rate_dps=20.0, max_accel_dps2=10.0)
        out = pid.update(45.0, dt=0.05)
        assert out == pytest.approx(0.5)  # 10 dps^2 * 0.05 s

    def test_zero_error_settles_to_zero(self):
        pid = AxisPID(kp=1.2, ki=0.15, max_rate_dps=20.0, max_accel_dps2=1000.0)
        for _ in range(50):
            pid.update(0.0, dt=0.05)
        assert pid.update(0.0, dt=0.05) == pytest.approx(0.0, abs=1e-6)

    def test_integral_is_bounded(self):
        pid = AxisPID(kp=0.0, ki=1.0, max_rate_dps=100.0, integral_limit=2.0,
                      max_accel_dps2=1000.0)
        for _ in range(500):
            pid.update(10.0, dt=0.05)
        assert abs(pid.update(10.0, dt=0.05)) <= 2.0 + 1e-9

    def test_freeze_integral_stops_windup(self):
        pid = AxisPID(kp=0.0, ki=1.0, max_rate_dps=100.0, max_accel_dps2=1000.0)
        for _ in range(20):
            pid.update(10.0, dt=0.05, freeze_integral=True)
        assert pid.update(0.0, dt=0.05) == pytest.approx(0.0, abs=1e-9)

    def test_reset_clears_state(self):
        pid = AxisPID(kp=1.0, ki=1.0, max_rate_dps=20.0, max_accel_dps2=1000.0)
        for _ in range(10):
            pid.update(10.0, dt=0.05)
        pid.reset()
        assert pid.update(0.0, dt=0.05) == pytest.approx(0.0, abs=1e-9)

    def test_nonpositive_dt_repeats_last_output(self):
        pid = AxisPID(max_accel_dps2=1000.0)
        first = pid.update(5.0, dt=0.05)
        assert pid.update(5.0, dt=0.0) == first
        assert pid.update(5.0, dt=-1.0) == first

    def test_decay_reaches_exactly_zero(self):
        pid = AxisPID(kp=2.0, ki=0.0, max_rate_dps=20.0, max_accel_dps2=10.0)
        pid.update(30.0, dt=0.5)
        for _ in range(100):
            out = pid.decay_toward_zero(0.05)
        assert out == 0.0

    def test_decay_is_monotonic_and_never_flips_sign(self):
        pid = AxisPID(kp=2.0, ki=0.0, max_rate_dps=20.0, max_accel_dps2=10.0)
        prev = pid.update(30.0, dt=0.5)
        assert prev > 0.0
        for _ in range(100):
            out = pid.decay_toward_zero(0.05)
            assert -1e-12 <= out <= prev + 1e-12
            prev = out

    def test_property_random_errors_stay_within_limits(self):
        rng = random.Random(1234)
        pid = AxisPID(kp=1.2, ki=0.15, max_rate_dps=3.0, max_accel_dps2=15.0)
        prev = 0.0
        for _ in range(2000):
            dt = rng.uniform(0.01, 0.2)
            out = pid.update(rng.uniform(-60.0, 60.0), dt=dt)
            assert abs(out) <= 3.0 + 1e-9
            assert abs(out - prev) <= 15.0 * dt + 1e-6
            prev = out


class TestAngleStepController:
    def test_step_is_bounded(self):
        ctrl = AngleStepController(kp=1.0, max_step_deg=2.0)
        pan, _ = ctrl.step(0.0, 0.0, 45.0, 0.0, pan_sign=1.0, tilt_sign=1.0)
        assert pan == pytest.approx(2.0)

    def test_tilt_is_clamped_to_limits(self):
        ctrl = AngleStepController(kp=1.0, max_step_deg=30.0, tilt_min_deg=-90.0,
                                   tilt_max_deg=20.0)
        _, up = ctrl.step(0.0, 15.0, 0.0, 30.0, pan_sign=1.0, tilt_sign=1.0)
        assert up == 20.0
        _, down = ctrl.step(0.0, -80.0, 0.0, -30.0, pan_sign=1.0, tilt_sign=1.0)
        assert down == -90.0

    def test_signs_invert_direction(self):
        ctrl = AngleStepController(kp=1.0, max_step_deg=5.0)
        pos, _ = ctrl.step(0.0, 0.0, 3.0, 0.0, pan_sign=1.0, tilt_sign=1.0)
        neg, _ = ctrl.step(0.0, 0.0, 3.0, 0.0, pan_sign=-1.0, tilt_sign=1.0)
        assert pos == pytest.approx(-neg)

    def test_is_proportional_not_accumulating(self):
        """Re-sending with an unmoved gimbal must re-send the same target."""
        ctrl = AngleStepController(kp=0.5, max_step_deg=10.0)
        first = ctrl.step(10.0, 0.0, 4.0, 0.0, pan_sign=1.0, tilt_sign=1.0)
        second = ctrl.step(10.0, 0.0, 4.0, 0.0, pan_sign=1.0, tilt_sign=1.0)
        assert first == second

    def test_converges_against_a_simulated_gimbal(self):
        ctrl = AngleStepController(kp=0.5, max_step_deg=5.0)
        pan, target = 0.0, 30.0
        for _ in range(200):
            cmd, _ = ctrl.step(pan, 0.0, target - pan, 0.0, pan_sign=1.0, tilt_sign=1.0)
            pan += (cmd - pan) * 0.5  # first-order lag toward the commanded angle
        assert pan == pytest.approx(target, abs=0.5)

    def test_yaw_wraps_rather_than_exceeding_sdk_range(self):
        ctrl = AngleStepController(kp=1.0, max_step_deg=10.0)
        pan, _ = ctrl.step(178.0, 0.0, 8.0, 0.0, pan_sign=1.0, tilt_sign=1.0)
        assert -180.0 <= pan <= 180.0
        assert pan == pytest.approx(-174.0)


def test_wrap_deg_180():
    assert wrap_deg_180(0.0) == 0.0
    assert wrap_deg_180(190.0) == pytest.approx(-170.0)
    assert wrap_deg_180(-190.0) == pytest.approx(170.0)
    assert wrap_deg_180(180.0) == pytest.approx(-180.0)


class TestDeadbandHold:
    def test_drives_when_error_is_large(self):
        hold = DeadbandHold()
        assert hold.update(500.0, now=0.0) is True
        assert hold.holding is False

    def test_holds_only_after_confirmation_window(self):
        hold = DeadbandHold(enter_deadband_px=25.0, hold_confirm_s=0.4)
        assert hold.update(10.0, now=0.0) is True   # inside, but not yet confirmed
        assert hold.update(10.0, now=0.3) is True
        assert hold.update(10.0, now=0.45) is False  # confirmed -> holding
        assert hold.holding is True

    def test_hysteresis_ignores_jitter_between_thresholds(self):
        hold = DeadbandHold(enter_deadband_px=25.0, exit_deadband_px=60.0, hold_confirm_s=0.0)
        hold.update(10.0, now=0.0)
        assert hold.update(10.0, now=0.1) is False
        # 40 px is above enter but below exit: must not wake the loop.
        for i in range(20):
            assert hold.update(40.0, now=0.2 + i * 0.1) is False

    def test_exits_hold_after_confirmed_large_error(self):
        hold = DeadbandHold(enter_deadband_px=25.0, exit_deadband_px=60.0,
                            hold_confirm_s=0.0, exit_confirm_s=0.15)
        hold.update(10.0, now=0.0)
        hold.update(10.0, now=0.1)
        assert hold.holding is True
        assert hold.update(100.0, now=0.2) is False   # over threshold, unconfirmed
        assert hold.update(100.0, now=0.4) is True    # confirmed -> driving
        assert hold.holding is False

    def test_transient_spike_does_not_break_hold(self):
        hold = DeadbandHold(enter_deadband_px=25.0, exit_deadband_px=60.0,
                            hold_confirm_s=0.0, exit_confirm_s=0.15)
        hold.update(10.0, now=0.0)
        hold.update(10.0, now=0.1)
        assert hold.update(100.0, now=0.15) is False  # single spike
        assert hold.update(10.0, now=0.2) is False    # back inside, still holding
        assert hold.holding is True

    def test_reset_restores_driving(self):
        hold = DeadbandHold(hold_confirm_s=0.0)
        hold.update(1.0, now=0.0)
        hold.update(1.0, now=0.1)
        assert hold.holding is True
        hold.reset()
        assert hold.holding is False


class TestDivergenceGuard:
    def test_trips_when_error_grows_under_command(self):
        # The first sample only establishes the baseline, so tripping on 8
        # consecutive growths takes 9 updates.
        guard = DivergenceGuard(max_consecutive=8, min_growth_px=2.0)
        err = 100.0
        guard.update(err, commanding=True)
        for _ in range(7):
            err += 10.0
            assert guard.update(err, commanding=True) is False
        err += 10.0
        assert guard.update(err, commanding=True) is True

    def test_does_not_trip_when_converging(self):
        guard = DivergenceGuard(max_consecutive=8)
        err = 200.0
        for _ in range(50):
            err *= 0.9
            assert guard.update(err, commanding=True) is False

    def test_does_not_trip_when_not_commanding(self):
        guard = DivergenceGuard(max_consecutive=3)
        err = 100.0
        for _ in range(50):
            err += 50.0
            assert guard.update(err, commanding=False) is False

    def test_noise_resets_the_streak(self):
        guard = DivergenceGuard(max_consecutive=4, min_growth_px=2.0)
        err = 100.0
        for _ in range(3):
            err += 10.0
            guard.update(err, commanding=True)
        guard.update(err - 5.0, commanding=True)  # one improvement clears it
        assert guard.count == 0

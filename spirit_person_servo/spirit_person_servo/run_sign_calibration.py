#!/usr/bin/env python3
"""Bounded, gated gimbal sign calibration.

Uses RATE probes only (cmd/gimbal_tilt, cmd/gimbal_pan). Travel is bounded by
rate x duration, so a wrong convention costs a couple of degrees rather than an
arbitrary slew. cmd/gimbal_angle is deliberately never used here.

Order matters: each probe is checked before the next one runs, and any
disagreement between commanded and measured travel aborts the whole run.

    ros2 run spirit_person_servo run_sign_calibration --ros-args \
        -p gimbal_namespace:=/spiritnx3/gremsy

No detector needed: travel is measured from global scene shift, so the scene just
needs some texture. People may move freely during the run.
"""

from __future__ import annotations

import sys
import time

import rclpy
from geometry_msgs.msg import Vector3  # noqa: F401  (kept: same msg set as backends)
from rclpy.node import Node
from std_msgs.msg import Float64, Int32

from .sign_calibration import (
    CalibrationAbort,
    ProbeResult,
    check_travel,
    format_yaml,
    sign_from,
)

GIMBAL_MODE_CMD_FOLLOW = 2


class SignCalibrator(Node):
    def __init__(self) -> None:
        super().__init__("sign_calibration")
        import os

        robot = os.environ.get("ROBOT_NAME", "spiritnx3")
        self.declare_parameter("gimbal_namespace", f"/{robot}/gremsy")
        self.declare_parameter("rtsp_url", "rtsp://192.168.70.23:8554/payload")
        self.declare_parameter("weights", "/opt/person_servo/weights/yolo11n.pt")
        self.declare_parameter("probe_dps", 3.0)
        self.declare_parameter("probe_s", 1.0)
        self.declare_parameter("settle_s", 2.5)
        self.declare_parameter("scale", 0.35)

        ns = self.get_parameter("gimbal_namespace").value.rstrip("/")
        self._dps = float(self.get_parameter("probe_dps").value)
        self._probe_s = float(self.get_parameter("probe_s").value)
        self._settle_s = float(self.get_parameter("settle_s").value)
        self._scale = float(self.get_parameter("scale").value)
        self._last_response = 0.0

        self._tilt_pub = self.create_publisher(Float64, f"{ns}/cmd/gimbal_tilt", 10)
        self._pan_pub = self.create_publisher(Float64, f"{ns}/cmd/gimbal_pan", 10)
        self._mode_pub = self.create_publisher(Int32, f"{ns}/cmd/gimbal_mode", 10)

        self._gimbal = {}
        from lion_ros2_bridge.msg import GimbalState

        self.create_subscription(GimbalState, f"{ns}/gimbal_state", self._on_gimbal, 10)

        self._src = None
        self._det = None

    # ---------------- plumbing ----------------
    def _on_gimbal(self, msg) -> None:
        self._gimbal = {"pan": float(msg.pan_deg), "tilt": float(msg.tilt_deg)}

    def spin(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def stop_rates(self) -> None:
        for _ in range(3):
            self._tilt_pub.publish(Float64(data=0.0))
            self._pan_pub.publish(Float64(data=0.0))
            self.spin(0.05)

    def start_perception(self) -> None:
        from .image_source import RtspImageSource

        self._src = RtspImageSource(self.get_parameter("rtsp_url").value)
        self._src.start()
        deadline = time.time() + 25
        while self._src.latest() is None and time.time() < deadline:
            self.spin(0.2)
        if self._src.latest() is None:
            raise CalibrationAbort("no video frames -- cannot calibrate")

    def _grab_gray(self):
        """One downscaled greyscale frame, float32, for phase correlation."""
        import cv2
        import numpy as np

        got = self._src.latest()
        if got is None:
            raise CalibrationAbort("video dropped out mid-probe")
        frame, _ = got
        small = cv2.resize(frame, (0, 0), fx=self._scale, fy=self._scale)
        return np.float32(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

    def scene_shift(self, before, after) -> tuple[float, float]:
        """Global image shift in FULL-RESOLUTION pixels.

        Measured with phase correlation over the whole frame rather than a
        detected bbox. The bbox approach fails here for two reasons found on
        nx-03: a person close to the camera has a bbox clipped by the frame edge,
        so its centroid is pinned and barely moves when the camera does; and human
        sway is far larger than the few pixels a small probe produces. Global
        scene shift measures exactly the thing we want -- how the image moved when
        the camera moved -- and does not care whether anyone is in frame.
        """
        import cv2

        (dx, dy), response = cv2.phaseCorrelate(before, after)
        self._last_response = response
        return dx / self._scale, dy / self._scale

    def measure_angles(self) -> tuple[float, float]:
        self.spin(0.3)
        return self._gimbal["pan"], self._gimbal["tilt"]

    # ---------------- the probe ----------------
    def probe(self, axis: str, direction: float) -> ProbeResult:
        pub = self._pan_pub if axis == "pan" else self._tilt_pub
        before = self._grab_gray()
        pan0, tilt0 = self.measure_angles()

        dps = self._dps * direction
        # Hold the rate for probe_s, republishing so a single dropped message
        # cannot leave it latched or starve it.
        end = time.time() + self._probe_s
        while time.time() < end:
            pub.publish(Float64(data=float(dps)))
            self.spin(0.05)
        self.stop_rates()
        self.spin(self._settle_s)

        after = self._grab_gray()
        pan1, tilt1 = self.measure_angles()
        dx, dy = self.scene_shift(before, after)

        measured = (pan1 - pan0) if axis == "pan" else (tilt1 - tilt0)
        # NO negation. The target is part of the scene, so its pixel coordinate
        # moves exactly as the scene does -- phase correlation already reports the
        # quantity we want. An earlier version negated this on the mistaken
        # reasoning that "the scene moves opposite to the target"; it does not,
        # and the inverted sign drove the gimbal away from the person.
        pixel_delta = dx if axis == "pan" else dy

        return ProbeResult(
            axis=axis,
            commanded_dps=dps,
            duration_s=self._probe_s,
            expected_deg=dps * self._probe_s,
            measured_deg=measured,
            pixel_delta=pixel_delta,
        )

    def calibrate_axis(self, axis: str) -> float:
        print(f"\n--- {axis.upper()} ---")
        forward = self.probe(axis, +1.0)
        print(
            f"  +{self._dps:.1f} dps x {self._probe_s:.1f}s -> "
            f"expected {forward.expected_deg:+.2f} deg, "
            f"measured {forward.measured_deg:+.2f} deg, "
            f"scene moved {forward.pixel_delta:+.1f} px "
            f"(corr {self._last_response:.2f})"
        )
        # GATE. Nothing else runs until this passes.
        check_travel(forward)
        print(f"  travel ok ({forward.travel_ratio:.2f}x), "
              f"command->motion {'FOLLOWS' if forward.follows_command else 'INVERTED'}")

        reverse = self.probe(axis, -1.0)
        print(
            f"  -{self._dps:.1f} dps x {self._probe_s:.1f}s -> "
            f"measured {reverse.measured_deg:+.2f} deg, "
            f"scene moved {reverse.pixel_delta:+.1f} px "
            f"(corr {self._last_response:.2f})"
        )
        check_travel(reverse)
        if forward.measured_deg * reverse.measured_deg > 0:
            raise CalibrationAbort(
                f"{axis}: forward and reverse probes moved the SAME way "
                f"({forward.measured_deg:+.2f}, {reverse.measured_deg:+.2f}) -- "
                f"the gimbal is not tracking the command sign."
            )

        sign = sign_from(forward)
        reverse_sign = sign_from(reverse)
        if sign != reverse_sign:
            raise CalibrationAbort(
                f"{axis}: forward and reverse disagree on sign "
                f"({sign:+.1f} vs {reverse_sign:+.1f}) -- measurement is not trustworthy."
            )
        print(f"  => {axis}_sign = {sign:+.1f}  (confirmed both directions)")
        return sign

    def run(self) -> int:
        print("waiting for gimbal telemetry...")
        deadline = time.time() + 10
        while not self._gimbal and time.time() < deadline:
            self.spin(0.2)
        if not self._gimbal:
            print("ABORT: no gimbal telemetry"); return 1
        print(f"gimbal: pan={self._gimbal['pan']:.2f} tilt={self._gimbal['tilt']:.2f}")

        # FOLLOW, not LOCK: in LOCK the driver reports yaw_absolute
        # (payloadSdkInterface.cpp:1699), a different frame from the command side.
        # Keeping body-frame telemetry makes "how far did it actually move" honest.
        self._mode_pub.publish(Int32(data=GIMBAL_MODE_CMD_FOLLOW))
        self.spin(1.5)
        self.stop_rates()

        print("starting camera + detector...")
        self.start_perception()

        pan_sign = tilt_sign = None
        try:
            # Tilt first: it is the axis with no frame ambiguity, so if something
            # is structurally wrong we find out on the cheaper axis.
            tilt_sign = self.calibrate_axis("tilt")
            pan_sign = self.calibrate_axis("pan")
        except CalibrationAbort as exc:
            print(f"\n*** CALIBRATION ABORTED ***\n{exc}")
            return 1
        finally:
            self.stop_rates()
            if self._src is not None:
                self._src.stop()

        print("\nPaste into config/<drone>.yaml:\n")
        print(format_yaml(pan_sign, tilt_sign))
        return 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SignCalibrator()
    code = 1
    try:
        code = node.run()
    except CalibrationAbort as exc:
        print(f"\n*** CALIBRATION ABORTED ***\n{exc}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_rates()
        except Exception:  # noqa: BLE001 - must still shut down
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()

"""Independent watchdog that stops the gimbal if the servo node dies.

Separate process on purpose. The servo node's own shutdown handler covers a clean
exit, but not SIGKILL, an OOM kill, or a segfault in a CUDA library -- and those
are exactly the cases where a rate command stays latched and the gimbal keeps
slewing. No GPU, no torch, no camera here: this must be the most boring process
on the drone.

It keys off `person_servo/state`, which the servo node publishes at a fixed rate
in *every* state including HOLD. That is what makes silence unambiguous: it means
the node is gone, not that it is holding still.
"""

from __future__ import annotations

import os

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float64, Int32

from spirit_person_servo_msgs.msg import ServoState

from .backends import (
    CMD_GIMBAL_PAN,
    CMD_GIMBAL_RATE,
    CMD_GIMBAL_TILT,
    CMD_TRACK,
)


class GimbalDeadmanNode(Node):
    def __init__(self) -> None:
        super().__init__("gimbal_deadman_node")

        robot = os.environ.get("ROBOT_NAME", "spiritnx3")
        self.declare_parameter("gimbal_namespace", f"/{robot}/gremsy")
        self.declare_parameter("state_topic", f"/{robot}/person_servo_node/state")
        self.declare_parameter("timeout_s", 0.5)
        self.declare_parameter("fast_stop_hz", 5.0)
        self.declare_parameter("fast_stop_duration_s", 3.0)
        self.declare_parameter("slow_stop_hz", 1.0)

        ns = self.get_parameter("gimbal_namespace").value.rstrip("/")
        self._timeout_s = float(self.get_parameter("timeout_s").value)
        self._fast_stop_duration_s = float(self.get_parameter("fast_stop_duration_s").value)
        self._fast_period = 1.0 / max(float(self.get_parameter("fast_stop_hz").value), 1e-3)
        self._slow_period = 1.0 / max(float(self.get_parameter("slow_stop_hz").value), 1e-3)

        self._rate_pub = self.create_publisher(Vector3, f"{ns}/{CMD_GIMBAL_RATE}", 10)
        self._tilt_pub = self.create_publisher(Float64, f"{ns}/{CMD_GIMBAL_TILT}", 10)
        self._pan_pub = self.create_publisher(Float64, f"{ns}/{CMD_GIMBAL_PAN}", 10)
        self._track_pub = self.create_publisher(Int32, f"{ns}/{CMD_TRACK}", 10)

        # Best-effort matches the servo node's state publisher.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            ServoState, self.get_parameter("state_topic").value, self._on_state, qos
        )

        self._last_seen_s: float | None = None
        self._was_moving = False
        self._last_backend = ServoState.BACKEND_DRY_RUN
        self._triggered_at_s: float | None = None
        self._last_stop_s = 0.0

        self.create_timer(0.05, self._on_tick)
        self.get_logger().info(
            f"gimbal deadman armed: watching {self.get_parameter('state_topic').value} "
            f"(timeout {self._timeout_s}s)"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_state(self, msg: ServoState) -> None:
        self._last_seen_s = self._now_s()
        self._was_moving = not msg.actuation_idle
        self._last_backend = msg.backend
        if self._triggered_at_s is not None:
            self.get_logger().info("servo node is alive again; deadman disarmed")
            self._triggered_at_s = None

    def _on_tick(self) -> None:
        if self._last_seen_s is None:
            return  # never seen the node; nothing to guard yet

        now = self._now_s()
        if now - self._last_seen_s <= self._timeout_s:
            return

        # Only intervene if the node was actually commanding motion when it went
        # silent. A node that died while idle needs no rescue, and publishing
        # commands nobody asked for is its own hazard.
        if not self._was_moving:
            return

        if self._triggered_at_s is None:
            self._triggered_at_s = now
            self.get_logger().error(
                f"servo node silent for >{self._timeout_s}s while commanding motion "
                "-- stopping gimbal"
            )

        elapsed = now - self._triggered_at_s
        period = self._fast_period if elapsed < self._fast_stop_duration_s else self._slow_period
        if now - self._last_stop_s < period:
            return
        self._last_stop_s = now
        self._publish_stop()

    def _publish_stop(self) -> None:
        if self._last_backend == ServoState.BACKEND_TRACK_TOUCH:
            # Nothing is latched, but the payload's own tracker is still driving
            # the gimbal with no supervisor -- turn it off.
            self._track_pub.publish(Int32(data=0))
            return

        # Zero every rate path. cmd/gimbal_tilt and cmd/gimbal_pan each zero the
        # other axis, which is a problem when servoing but is exactly what we want
        # here: either message alone commands a full stop.
        self._rate_pub.publish(Vector3(x=0.0, y=0.0, z=0.0))
        self._tilt_pub.publish(Float64(data=0.0))
        self._pan_pub.publish(Float64(data=0.0))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GimbalDeadmanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

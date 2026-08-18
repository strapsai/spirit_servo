# spirit_servo

Onboard person detection and gimbal servoing for **Spirit** drones. Detects people
in the payload's EO stream, picks one, and holds the gimbal on them.

Runs entirely inside the drone's own `ROS_DOMAIN_ID` — no basestation, no domain
bridge, no database.

Named for the airframe, not the function: other platforms in the org have their
own servoing, and this one is specific to the Spirit's Gremsy payload.

## Packages

Two packages, one repo, versioned together so the message contract cannot drift:

| Package | Build type | Contents |
|---|---|---|
| [`spirit_person_servo`](spirit_person_servo/) | `ament_python` | Node, control laws, tracker, detector, actuation backends, sign calibration, tests |
| [`spirit_person_servo_msgs`](spirit_person_servo_msgs/) | `ament_cmake` | `ServoState.msg` |

They are separate packages only because `ament_python` cannot run `rosidl` —
message generation needs `ament_cmake`. They are one repo because they are one
unit of change.

**Full documentation, including the safety model and bring-up order, is in
[`spirit_person_servo/README.md`](spirit_person_servo/README.md). Read the safety
section before running anything that moves the gimbal.**

## Why `ServoState` is not in `drone_msgs`

`drone_msgs` is the CMU↔LM ICD — every message in it is a cross-organisation
contract. `ServoState` is internal telemetry: a state-machine enum, pixel errors,
and debug counters that nothing outside this node consumes. Keeping it here also
avoids `drone_msgs`' multi-workspace version lockstep.

## Quick start

```bash
# safe: computes and publishes everything, commands nothing
ros2 launch spirit_person_servo person_servo.launch.py
ros2 service call /spiritnx3/person_servo_node/start std_srvs/srv/Trigger
```

Live annotated view — detections, chosen target, deadband, state — at
`http://<drone-ip>:8099/` when `mjpeg_port` is set.

## Status

**Phase 1 works on spiritnx3**: YOLO11n at p50 32 ms on GPU, 1080p RTSP at 30 fps
alongside the basestation video feed, closed-loop pan servoing that converges and
settles into the deadband.

Known gaps are listed at the end of the package README — most importantly the
`angle` backend is gated off until the gimbal yaw frame is resolved.

Phase 2 (re-ID against a casualty gallery) replaces only `target_selector.py`;
image source, detector, tracker, controller, state machine and safety layers all
carry forward unchanged.

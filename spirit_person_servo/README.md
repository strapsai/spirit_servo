# spirit_person_servo

Onboard person detection and gimbal servoing for Spirit drones. Detects people in
the payload's video stream, picks one, and holds the gimbal on them.

Runs entirely inside the drone's own `ROS_DOMAIN_ID`. No basestation, no domain
bridge, no database.

This is **phase 1** of the person servoing work: it proves the perception →
actuation loop with a simple target rule. Phase 2 (see
`spirit_drivers_ws/docs/person-servoing-spirit-indago.md`) replaces only
`target_selector.py`, swapping "largest person" for re-ID matching against a
gallery of a specific casualty. Everything else here carries forward.

---

## Safety first

**Scope of what this node can move: the gimbal, and nothing else.** Every topic it
publishes (`cmd/gimbal_angle`, `cmd/gimbal_tilt`, `cmd/gimbal_pan`,
`cmd/gimbal_rate`, `cmd/gimbal_mode`, `cmd/track`, `cmd/track_touch`,
`cmd/toggle_eo_ir`) goes to the Gremsy payload over its own MAVLink link. There is
no MAVROS dependency and no path to the flight controller — it cannot arm the
aircraft or command a motor. Props on or off is irrelevant to it; what matters is
clear space around the **gimbal**.

**Two gates must both be cleared before this node can move anything:**

1. `servo_backend` must be something other than `dry_run`, and
2. `sign_calibration_verified: true` must be set in the per-drone config.

If a backend is requested without the calibration flag, the node logs an error and
**forces `dry_run`**. This is deliberate: the sign conventions are two independent
unknowns (command → gimbal motion, and gimbal motion → pixel motion), and an
inverted sign produces a runaway slew.

Other layers:

| Layer | What it covers |
|---|---|
| Staleness timer (20 Hz, independent of the image callback) | Camera or detector goes quiet → stop, enter `LOST` |
| Divergence guard | Error grows for 8 consecutive cycles under command → stop, log `SIGN CONVENTION LIKELY INVERTED` |
| Clean shutdown | Stop published 5× at 20 ms spacing (ROS 2 publisher teardown is async and can drop a single publish) |
| `gimbal_deadman_node` | Separate process. Servo node SIGKILLed / OOM-killed while commanding motion → stops the gimbal |
| Tilt clamp | Angle setpoints clamped to `[tilt_min_deg, tilt_max_deg]` |

---

## Actuation backends

`spirit_driver.cpp` maps `cmd/gimbal_tilt` and `cmd/gimbal_pan` onto a single
`setGimbalSpeed(pitch, roll, yaw, INPUT_SPEED)` call that always carries a
complete 3-axis setpoint:

```cpp
CMD_GIMBAL_TILT -> setGimbalSpeed(m.data, 0, 0, INPUT_SPEED);  // yaw forced to 0
CMD_GIMBAL_PAN  -> setGimbalSpeed(0, 0, m.data, INPUT_SPEED);  // pitch forced to 0
```

So publishing both each cycle does **not** produce diagonal motion — they
alternate and cancel. Two-axis rate servoing is impossible through the topics that
exist today. Hence:

| `servo_backend` | Works today? | Notes |
|---|---|---|
| `dry_run` | yes | Computes and publishes telemetry, commands nothing. **Default.** |
| `angle` | yes | Absolute setpoints via `cmd/gimbal_angle`. Safest: a dropped command holds still rather than slewing. Needs `GimbalState`. |
| `track_touch` | yes | Hands the pixel to the payload's own hardware tracker. No signs, no tuning. Can't see what it locked onto. |
| `rate` | **yes, untested on hardware** | `cmd/gimbal_rate` landed in `gremsy_ros2` (`lorenzo/person-servo-inspection`). Rebuild the driver on the aircraft; `RateBackend` refuses to arm if nothing is subscribed. |
| `single_axis_rate` | yes | One axis only, so nothing zeroes it. For bench sign calibration. |

---

## Who arms it, on an aircraft

Nothing here decides when to servo. This node sits in `IDLE` until something
calls `~/start`, and on a Spirit that caller is **`spirit_task_executor`'s
`gimbal_pointing` node**, which arbitrates the gimbal between its own geometric
look-at aim and this servo — see `servo_arbiter.py` there. It has to be one
thing deciding, because this node and that one drive the same gimbal through
the same `setGimbalSpeed` call, and `backends.py` is blunt about what two
controllers on one gimbal produce.

The contract, from this side:

| | |
|---|---|
| It calls `~/start` | ~2 s after the executor reports `INSPECTING` — enough for the payload to reach the aim **and** the inspection zoom, because `control.py` scales the intrinsics by the live `zoom_level` and a frame grabbed mid-zoom is measured against the wrong focal length |
| It stops commanding the gimbal | only once `ServoState` says `state ∈ {SEARCHING, LOCKED, SERVOING, HOLD}` **and** `commands_enabled` — the telemetry, never the service reply |
| It takes the gimbal back | the moment `ServoState` says `IDLE`/`LOST`, or goes stale (1 s), or says `commands_enabled: false` |
| It calls `~/stop` | on leaving `INSPECTING`, then waits (bounded, 1 s) for `actuation_idle` before commanding again |

Two consequences worth knowing:

- **`commands_enabled` is the fail-safe, and it is load-bearing.** A `dry_run`
  servo — which is what an un-calibrated one forces itself into — publishes
  everything and moves nothing, so the pointer keeps the camera and the
  inspection is exactly as good as it was before this node existed. That is why
  the pointer reads the telemetry and not the `~/start` reply: the reply says
  "armed", only `ServoState` says "actually driving".
- **`HOLD` must not be read as a release.** ServoState's invariant is
  `state ∈ {IDLE, HOLD, LOST} ⇒ actuation_idle`, so a settled servo publishes
  idle. An earlier arbiter read that as "it let go", handed the gimbal back the
  instant the servo succeeded, and the two traded it across the deadband edge.
  Ownership follows the **state**; `actuation_idle` answers only "has it
  stopped slewing yet", which is what the release wait needs.

This container comes up with the rest of the Spirit payload stack
(`launch/spirit/spirit-containers.sh`), brought up separately and allowed to
fail so a missing servo image cannot take MAVROS down with it. Whether the
pointer actually hands over is `servo_enabled` in ITS per-drone config, default
true — not an `airlab.env` variable, because this is part of the payload stack
rather than a per-deployment choice.

## Quick start

Dry run against the live camera (safe — nothing moves):

```bash
ros2 launch spirit_person_servo person_servo.launch.py
ros2 service call /spiritnx3/person_servo_node/start std_srvs/srv/Trigger
ros2 topic echo /spiritnx3/person_servo_node/state
ros2 service call /spiritnx3/person_servo_node/stop std_srvs/srv/Trigger
```

Develop with no drone at all — a video file instead of the camera:

```bash
ros2 run spirit_person_servo person_servo_node --ros-args \
  -p image_source:=replay -p replay_video:=/path/to/clip.mp4 \
  -p detector_backend:=yolo -p yolo_device:=cpu -p yolo_half:=false
```

Watch what it sees (`publish_debug_image: true`, then any image viewer on
`~/debug_image/compressed`) — draws every track, the chosen one in green, and the
deadband rectangle.

### RF-DETR engine

The default detector is RF-DETR Medium as a TensorRT FP16 engine (`detector_backend:
rfdetr_trt`). Engines only load on the GPU and TensorRT version they were built with,
so build them on each aircraft, inside the servo image. One engine per camera, each
at that camera's frame shape, so neither is stretched: EO `448x800` (16:9, fed the
960x540 half-res frame) and IR `512x640` (the native 640x512 frame, no resize). The
ONNX comes from `rfdetr==1.11.1`: `RFDETRMedium().export(shape=(H, W))` (needs
`onnx<1.19`).

```bash
E=$AIRLAB_PATH/weights/person_servo/engines/sm87-trt10.4
for S in 448x800 512x640; do
docker run --rm --runtime nvidia -v $AIRLAB_PATH:$AIRLAB_PATH \
  --entrypoint /usr/src/tensorrt/bin/trtexec dtc/dtc:jp6.1-06e-person-servo \
  --onnx=$AIRLAB_PATH/weights/person_servo/onnx/rfdetr_m_$S.onnx \
  --saveEngine=$E/rfdetr_m_$S.fp16.engine --fp16 --memPoolSize=workspace:4096
done
```

Without the engine the node logs an error and runs YOLO11n instead.

### EO or IR

`modality: eo | ir` picks the camera. On an aircraft it is the Spirit stream the
session profile's `reid/streams.yaml` sends (EO when both are), and the servo reads
that stream's dtc-streams re-host (`launch/spirit/launch_person_servo.sh`). With no
EO or IR stream sent over RTSP, the servo does not start. Both cameras use the
same RF-DETR model, signs and gains; each runs its own engine, shaped to its frame.
What changes per modality:

| | eo | ir |
|---|---|---|
| intrinsics | `<drone>_eo.yaml`, scaled by live `zoom_level` | `<drone>_ir.yaml`, fixed (the reported zoom is the EO's) |
| default `rtsp_url` | `127.0.0.1:8554/eo` | `127.0.0.1:8555/ir` |
| default `rtsp_half_resolution` | true (1080p) | false (640x512) |
| default `rfdetr_engine` | `rfdetr_m_448x800.fp16.engine` | `rfdetr_m_512x640.fp16.engine` (native, no resize) |

`rate_zoom_factor_table` applies to both cameras: the gimbal slows its rate response
on EO zoom whichever camera is being watched.

```bash
ros2 launch spirit_person_servo person_servo.launch.py modality:=ir
```

---

## Bring-up order

Do not skip stages 0–3.

0. **Gate 0 — imagery.** Confirm on the NX that a second RTSP session works *while*
   the basestation feed is running, and that both keep receiving frames. The
   payload's tolerance for concurrent sessions is unverified; if it caps at one,
   this silently kills the operator's video. Fall back to `image_source: replay`
   until resolved.
1. **Dry run on the bench.** Real camera, real YOLO, `dry_run`. Confirm detections,
   target selection, and computed commands look sane. Nothing moves.
2. **Sign calibration.** Props removed, E-stop in reach, `max_rate_dps: 3.0`.
   Measure both sign conventions, record them in the config, then set
   `sign_calibration_verified: true`.
3. **Fault injection.** `kill -9` the node mid-servo; unplug the RTSP source.
   The gimbal must stop in both cases. This is the only way to find "the gimbal
   kept slewing after the node died".
   With the handover wired, the same run has a second half: the gimbal pointer
   must then TAKE THE CAMERA BACK and return to its geometric aim, within a tick
   of the telemetry going stale. Watch `ros2 topic hz /<drone>/gimbal/command` —
   it should go from silent to ~5 Hz as the servo dies.
4. Tethered flight, gimbal only, operator on E-stop.
5. Free flight. Single person first, then several, to check the largest-bbox rule
   doesn't visibly thrash between people.

---

## Layout

| File | Responsibility |
|---|---|
| `control.py` | Control laws, deadband/hold, divergence guard. ROS-free, fully unit-tested. |
| `tracker.py` | `Detection`/`Track` seam + BoT-SORT adapter. |
| `target_selector.py` | Which track to follow. **The phase-2 seam.** |
| `detector.py` | `PersonDetector` interface, RF-DETR TensorRT and YOLO backends, replay/null backends. |
| `image_source.py` | RTSP and replay frame sources. |
| `backends.py` | Actuation paths and the gimbal topic contract. |
| `person_servo_node.py` | State machine, timers, telemetry. |
| `gimbal_deadman_node.py` | Independent watchdog. |

`control.py` and `target_selector.py` are deliberately ROS-free and GPU-free so
the safety-critical logic is testable:

```bash
cd spirit_drivers_ws/src/spirit_servo/spirit_person_servo
PYTHONPATH=. python3 -m pytest test/ -q
```

## Known gaps

- **Only ONE axis is servoed in the shipped config** (`single_axis_rate`,
  `single_axis: pan`). `tilt_sign: -1.0` is measured and waiting. Two paths are
  open and both need a bench run nobody has done: `servo_backend: "rate"` now
  that `cmd/gimbal_rate` exists, or `servo_backend: "angle"` once the yaw frame
  is resolved. `config/spiritnx3.yaml` spells out both, including the untested
  hypothesis that the frame problem is LOCK-vs-FOLLOW and not the command.
- `cmd/gimbal_rate` now exists in `gremsy_ros2`, so the `rate` backend is no
  longer inert — but it has never been run against hardware, and the driver has
  to be rebuilt on the aircraft before anything is subscribed.
- Asserting 1× zoom on arm is not implemented. Live `zoom_level` scales the EO
  intrinsics, and the loop derates its gain by half when that reading is stale.
- IR pixel errors use the pinhole model; the IR lens distortion (`distortion` in
  `<drone>_ir.yaml`, ~16 px at the corners) is not applied. It is ~0 near the
  centre, where the servo settles.
- The BoT-SORT adapter reads a semi-private ultralytics interface
  (`.conf`/`.xywh`/`.cls`). Safe under the current tight pin
  (`>=8.3.78,<=8.3.80`); re-verify if that pin moves.
- `STATE_LOCKED` is defined in the message but currently unused — acquisition goes
  straight to `SERVOING`. Phase 2's verification step will occupy it.

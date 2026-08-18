# Person Servoing Module — Implementation Plan

## Context

When a Spirit drone is tasked to inspect a specific casualty, it currently receives only an
`InspectCasualty` command carrying an LLA position and a radius. It flies there and points the
gimbal geometrically. Nothing onboard knows *which person in frame* is the casualty of interest,
so with two or more people in view the drone cannot reliably keep the right one centered.

Meanwhile the ground reid stack has already built exactly the knowledge that's missing: for every
`global_id` in `global_identities` it holds a trace back to every 2D detection and every source
frame, i.e. a per-identity gallery of person crops
(`db_api/db.py:get_images_for_global_id`, line ~1336).

This module closes that gap. It runs person detection onboard the Orin NX, encodes each person
bbox crop with a RADIO encoder, compares against a RADIO-encoded gallery of the target
`global_id`, picks the matching bbox, and then runs a gimbal servo loop that centers and holds
that person. When no candidate is the target, it says so upward with numbers rather than silently
staring at the wrong person.

Two hard constraints shape everything:
- **Gimbal rate commands (`gremsy/cmd/*`) are not in the domain bridge.** The servo loop must run
  onboard, in the drone's own `ROS_DOMAIN_ID`.
- **There is no ROS image topic on the drone.** The Gremsy payload serves RTSP; images only become
  ROS messages on the basestation. Onboard imagery has to be created.

---

## Findings that change the design (all verified in-tree)

**1. `InspectCasualty` cannot carry `global_id`.** Two live producers with incompatible semantics:

| Producer | `command_id` | `casualty_id` |
|---|---|---|
| `reid_ws/src/filter_3d_bbox/scripts/filter_py_db_node.py:785` | `str(global_id)` | `global_id % 256` (lossy) |
| `task_allocation_ws/.../spirit/spirit_command_builder.py:83` | `task_id` (a UUID) | hardcoded `0` |

`spirit_command_builder.py`'s own docstring (lines 16–21) states the `casualty_id` field is unused
and always 0, and that the value to plumb "is the task's `metadata.global_id` — never a task-id
derivative." **The servo module must never parse `global_id` out of `InspectCasualty`.** Identity
arrives on the new gallery topic as a full-precision `int32`; `InspectCasualty` is an *arm* signal
and a correlation key only.

**2. RADIO's normalization is the identity transform.**
`UniCeption/uniception/models/encoders/image_normalizations.py:29` →
`"radio": mean=[0,0,0], std=[1,1,1]`. RADIO takes **raw `[0,1]` RGB** and applies its own
`input_conditioner` internally. Applying ImageNet mean/std produces a plausible-but-wrong embedding
space — the highest-probability silent-correctness bug in this project.

**3. A non-leaky tee branch in `sender.cpp` will kill the basestation video feed.**
`last_frame_time` is set inside `h264_sei_inject_probe` ([sender.cpp:158]), which is attached to
`parse:src` — **upstream of the tee**. A watchdog thread calls `exit(1)` if that probe hasn't fired
for 1.0 s ([sender.cpp:296-302]). Any new branch that backpressures the tee stalls `parse`, stops
the probe, and kills the process. `leaky=downstream` on the new queue is non-negotiable.

---

## A. Packages

| Package | Lives in | Build | Runs on |
|---|---|---|---|
| `person_descriptor` | `reid_ws/src/` **and** `spirit_drivers_ws/src/` | `ament_python` + `pyproject.toml` | both |
| `person_servo_msgs` | `spirit_drivers_ws/src/messages/`, `basestation_drivers_ws/src/messages/`, `reid_ws/src/` | `ament_cmake` | all three |
| `spirit_person_servo` | `spirit_drivers_ws/src/servoing/` | `ament_python` | drone (arm64) |
| `person_gallery` | `reid_ws/src/` | `ament_python` | basestation (x86) |

**`person_descriptor` is a correctness requirement, not convenience.** Preprocessing (resize policy,
interpolation, color order, normalization) and descriptor construction must be identical logic on
both sides or cosine similarity degrades *silently* — it doesn't error, the argmax just picks the
wrong person. ROS-free, torch+numpy only, arch-agnostic. Pin to a **git tag** in both
`version_control/reid.yaml` and `version_control/spirit_drivers.yaml` (most repos there are pinned
to `main`, which would let the two sides drift).

```
person_descriptor/
  __init__.py     # __version__, MODEL_KEY
  preprocess.py   # PreprocessSpec, crop_and_resize()
  encoder.py      # RadioDescriptorEncoder (torch), TrtDescriptorEncoder
  descriptor.py   # summary/GAP pooling, L2, concat
  matching.py     # GalleryScorer, hysteresis, N-of-M voting
  tracking.py     # SimpleTracker (SORT-like, numpy only)
  control.py      # AxisPID, DeadbandHold  <- ROS-free, unit-testable
  parity.py       # golden-vector helpers
```

Putting `control.py` and `tracking.py` here is deliberate: the servo law and tracker become
testable with zero ROS and zero GPU.

**Not `drone_msgs`.** `spirit_drivers_ws/src/messages/drone_msgs/README.md:3` calls it "the
high-level ICD defined between CMU and LM". Adding a 1536-float gallery message and an internal
state-machine enum to a vendor ICD forces a renegotiation for a purely-internal feature. New
package. It depends only on `std_msgs` + `builtin_interfaces` — bboxes are plain floats — so the
domain-bridge host doesn't need `vision_msgs` at a matching version.

**Two basestation nodes, not one:** `gallery_worker_node` is GPU-bound and bursty (pulls JPEGs from
MinIO); `gallery_publisher_node` must respond instantly to an inspect command. Coupled, an inspect
command queues behind a 60-crop encode batch.

**One drone process** (`person_servo_node`): a 1920×1080 BGR frame is 6.2 MB; splitting
detect/encode/servo across nodes means DDS-serializing that per stage. Servo timer on a separate
`MutuallyExclusiveCallbackGroup` under a `MultiThreadedExecutor`. The crash risk this creates is
handled by a separate `gimbal_deadman_node` — which is precisely why *that* is its own process.

---

## B. Messages (`person_servo_msgs`)

**`PersonGallery.msg`** — basestation → drone, the core new contract:
```
std_msgs/Header header
int32  global_id          # AUTHORITATIVE. full precision. never from InspectCasualty
string session_id
string command_id         # correlation copy only; "" when pushed proactively
string class_label
string model_key          # "radio_v2.5-b/sum+gap/384x192/rgb/squash/v1"
string lib_version
uint32 dim
PersonDescriptor   mean       # quality-weighted mean, re-L2-normalised
PersonDescriptor[] exemplars  # top-K by quality+diversity (K=8)
PersonDescriptor[] negatives  # OTHER nearby global_ids -- see E
uint32 n_source_crops
float32 intra_class_mean_sim
float32 intra_class_min_sim
float32 nearest_other_sim
float32 suggested_tau
float32 suggested_margin
builtin_interfaces/Time built_at
```

`negatives` is the highest-value field and is free — the DB already holds every other identity's
embedding. Absolute cosine over RADIO descriptors is badly calibrated (everything human looks
similar); a score *relative to real negatives from the same scene* is far more discriminative.

**`PersonDescriptor.msg`**: `int64 detection_id`, `uint32 dim`, `float32[] descriptor` (L2-normalised), `float32 quality`.

**`PersonGalleryRequest.msg`** (drone → base): `header`, `robot_name`, `command_id`,
`int32 global_id` (`-1` = "armed but I don't know the id"), `string reason`.

**`CandidateScore.msg`**: `track_id`, bbox floats, `detector_conf`, `sim_mean`, `sim_topk`,
`sim_negative`, `score`, `bool encoded_this_frame`.

**`MatchDebug.msg`**: `header`, `global_id`, `model_key`, `n_detections`, `n_encoded`,
`CandidateScore[] candidates` (sorted desc), `int32 winner_track_id`, `float32 margin`,
`uint8 decision`, timing floats.

**`ServoState.msg`** @10 Hz — the observability spine. `state` enum
(`IDLE/GALLERY_WAIT/SEARCHING/VERIFYING/LOCKED/SERVOING/HOLD/LOST/ABORT`), `lock_status` enum
(`NONE/CONFIDENT/LOW_CONFIDENCE/AMBIGUOUS/NO_CONFIDENT_MATCH`), winner bbox + score + margin,
`err_px_x/y`, `err_deg_yaw/pitch`, `cmd_rate_pan_dps`, `cmd_rate_tilt_dps`, **`bool rates_zeroed`**,
`zoom_level`, `bool zoom_valid`, `image_age_s`, rate counters, `status_text`.

`rates_zeroed` is explicit rather than inferred so a bag reader can trivially assert the safety
invariant "in HOLD/LOST/IDLE, `rates_zeroed == true`".

**`PersonServoResult.msg`** (drone → base, `transient_local`) — the machine-readable answer to
"what if no candidate is the target": `outcome` ∈ `LOCKED / NO_TARGET / AMBIGUOUS / NO_PEOPLE /
LOST / TIMEOUT / ABORTED / NO_GALLERY / MODEL_MISMATCH`, plus `best_score`, `second_score`,
`margin`, `time_to_lock_s`, `hold_duration_s`, `n_candidates_seen`, `detail`.

---

## C. pgvector schema

Store **both** per-crop embeddings (source of truth: needed for top-k, exemplar reselection,
re-aggregation when the model changes, and the offline eval) and a per-identity materialized
aggregate (so the publisher answers an inspect command with one PK lookup).

Dimension **1536** = `concat(summary_768, gap_768)` for `radio_v2.5-b`. Under pgvector's 2000-dim
indexed limit — one more reason to prefer `-b` over `-l` (which would be 2048 and over the limit).

Add to `reid_ws/src/triage_database/init.sql` after TIER 5, **and** as an idempotent
`ensure_embedding_tables()` in `db.py` mirroring `ensure_stream_ingest_tables()` (db.py:1637), so
already-deployed databases pick it up without a re-init.

```sql
-- TIER 5b — APPEARANCE EMBEDDINGS. Uses the pgvector extension created at
-- init.sql:8 (installed since day one, unused until now).

ALTER TABLE global_identities
    ADD COLUMN IF NOT EXISTS embed_status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (embed_status IN ('pending','processing','done','failed')),
    ADD COLUMN IF NOT EXISTS embed_updated_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS embed_n_crops INT NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_global_identities_embed_pending
    ON global_identities(session_id, global_id) WHERE embed_status = 'pending';

CREATE TABLE IF NOT EXISTS identity_crop_embeddings (
    crop_embed_id BIGSERIAL PRIMARY KEY,
    session_id   UUID   NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    global_id    INT    NOT NULL REFERENCES global_identities(global_id) ON DELETE CASCADE,
    detection_id BIGINT NOT NULL REFERENCES detections_2d(detection_id) ON DELETE CASCADE,
    frame_id     BIGINT NOT NULL REFERENCES frame_assets_database_gate_2(frame_id) ON DELETE CASCADE,
    captured_at  TIMESTAMPTZ NOT NULL,
    model_key    TEXT   NOT NULL,          -- MUST match person_descriptor.MODEL_KEY
    lib_version  TEXT   NOT NULL,
    embed_dim    INT    NOT NULL CHECK (embed_dim = 1536),
    embedding    vector(1536) NOT NULL,    -- L2-normalised concat(summary, GAP)
    crop_w INT NOT NULL, crop_h INT NOT NULL, crop_area_px INT NOT NULL,
    blur_var REAL,                          -- variance of Laplacian
    truncated BOOLEAN NOT NULL DEFAULT FALSE,
    quality  REAL NOT NULL DEFAULT 1.0 CHECK (quality BETWEEN 0.0 AND 1.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (detection_id, model_key)
);
CREATE INDEX IF NOT EXISTS idx_ice_global_quality
    ON identity_crop_embeddings(global_id, model_key, quality DESC);

CREATE TABLE IF NOT EXISTS identity_gallery (
    global_id  INT  NOT NULL REFERENCES global_identities(global_id) ON DELETE CASCADE,
    model_key  TEXT NOT NULL,
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    embed_dim  INT  NOT NULL CHECK (embed_dim = 1536),
    mean_embedding vector(1536) NOT NULL,
    n_crops    INT  NOT NULL,
    exemplar_ids BIGINT[] NOT NULL DEFAULT '{}',
    intra_mean_sim REAL, intra_min_sim REAL,
    nearest_other_id INT, nearest_other_sim REAL,
    suggested_tau REAL, suggested_margin REAL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (global_id, model_key)
);
```

**No ANN index.** The servo lookup is `WHERE global_id = %s AND model_key = %s` — an exact PK hit.
An `ivfflat`/`hnsw` index would be pure cost for zero benefit. Add one only when a real
nearest-neighbour query exists (reid dedup, "find identities like this"); the DDL is one line then.

**Staleness:** no DB trigger (this schema has none, and triggers are invisible to anyone reading
`db.py`). `gallery_worker_node` runs an `UPDATE ... SET embed_status='pending' WHERE embed_status='done'
AND embed_n_crops < (SELECT count(*) FROM reid_results WHERE global_id = gi.global_id)` each cycle.

**New `db.py` functions**, following existing style (module-level, `get_cur`, `execute_values`, no ORM):
`ensure_embedding_tables`, `claim_identities_for_embedding` (structural clone of
`claim_identities_for_vlm` at db.py:994 — same `FOR UPDATE SKIP LOCKED`),
`complete/fail/reset_identity_embedding`, `mark_stale_identity_embeddings`,
`insert_crop_embedding_batch`, `get_crop_embeddings_for_global_id`, `upsert_identity_gallery`,
`get_identity_gallery`, `get_gallery_exemplars`,
`get_negative_galleries_near` (reuse `find_identities_near` semantics, db.py:1598),
`get_embedding_coverage`.

**pgvector + psycopg2 gotcha:** `register_vector()` is **per connection**, and `db.py` uses a
`ThreadedConnectionPool` — it must go in `get_conn()` behind a per-connection flag, not once at init.

---

## D. RADIO encoder

**Model: `radio_v2.5-b`** (ViT-B/16, 768-d). At 384×192 that's 288 patch tokens. `-l` is ~3× the
FLOPs and pushes past pgvector's indexed limit; `-h`/`-g` are not viable on an NX shared with a
detector. **Fallback #1 is `e-radio_v2`** — its fixed-input-shape requirement
(`set_optimal_window_size`, radio.py:93) is actually a *benefit* for TensorRT, which wants static
shapes anyway.

**Write a thin standalone wrapper** in `person_descriptor/encoder.py` rather than depending on
`uniception.models.encoders.radio.RADIOEncoder`, because that class (a) subclasses
`UniCeptionViTEncoderBase` and would pull all of UniCeption onto the drone, (b) is protected by an
explicit "UniCeption must not be importable from the image" assert in `08b-reid.dockerfile`, and
(c) **discards the summary token** — its `forward()` returns only `features`. We need `summary`.

Copy verbatim (this *is* the parity contract): the `torch.hub.load("NVlabs/RADIO", "radio_model",
version=...)` entrypoint, the `(H//p, W//p)` reshape, the `H%16==0 and W%16==0` assert, and
`data_norm_type="radio"` ⇒ raw `[0,1]` input.

**`PreprocessSpec`** — frozen, and hashed into `MODEL_KEY`:
```python
out_h=384; out_w=192          # 24x12 patches; both multiples of 16
pad_ratio=0.08                # expand bbox 8% each side before crop
interpolation="bilinear"      # cv2.INTER_LINEAR, antialias=False
color_order="RGB"             # inputs are BGR -> convert
value_range="unit"            # /255.0, NO ImageNet mean/std  (finding #2)
resize_policy="squash"        # NO letterbox
```

*Squash, not letterbox:* person bboxes run 1:4 (standing) to 3:1 (prone casualty — very much our
case). Letterbox padding enters the GAP pool, so the descriptor drifts as a function of aspect
ratio rather than appearance.

*Color order is the second parity landmine:* `cv_bridge`/`cv2.imdecode` give BGR, `PIL` gives RGB,
RADIO wants RGB. A one-sided miss drops cosine by ~0.1–0.2 and the system still "works", just
badly. `preprocess.py` takes `input_color_order` **with no default** so every call site declares it.

**Descriptor:**
```python
s   = F.normalize(summary, dim=-1)                    # (B, D)
gap = F.normalize(feat_map.mean(dim=(2,3)), dim=-1)   # (B, D)
d   = torch.cat([w_summary * s, (1-w_summary) * gap], dim=-1)
return F.normalize(d, dim=-1)                          # (B, 2D) unit norm
```
Summary token = CLIP/DFN-distilled, semantic, pose-invariant. GAP over spatial tokens = dominated
by DINOv2-distilled dense features, i.e. appearance/texture — what distinguishes *this* casualty
from *that* one when both are "person lying on grass". **L2 ordering is part of the contract:**
normalize each half → weight → concat → normalize. Start `w_summary=0.5`; the offline sweep (§H)
settles it and the winner is baked into `MODEL_KEY`.

**Export, staged, with parity gates at each step:** torch fp32 on the Orin first (eliminates every
export variable) → torch fp16 (expect cosine > 0.9999 vs fp32) → ONNX opset 17 static shapes →
TRT FP16 engine **built on the target device**. TRT engines are tied to TensorRT version, CUDA
version, and SM arch — they cannot be built on x86 and shipped. Cache under
`engines/{sm_arch}-trt{ver}/`, `.gitignore` them, never put them in the weights bucket, and never
fail closed: fall back to torch fp16 and say so in `ServoState.status_text`.

**Do not use `torch.compile`** here — multi-minute warm compile on every process start, fragile on
torch 2.4/aarch64, and it gives a *third* set of numerics to reconcile.

**Offline `torch.hub`:** `torch.hub.load` needs internet; the drone has none. Pre-populate
`TORCH_HOME=$AIRLAB_PATH/weights/person_servo/torchhub` (the cloned repo + `hub/checkpoints/`) from
the weights bucket. In the TRT steady state, RADIO's python isn't needed at all — only the engine.

---

## E. Matching and lock-on

```
sim_mean = dot(d, G.mean)
sim_topk = mean(top3(sorted([dot(d,e) for e in G.exemplars], desc)))
sim_neg  = max([dot(d,n) for n in G.negatives]) if G.negatives else 0.0
s_pos    = alpha*sim_mean + (1-alpha)*sim_topk      # alpha 0.5
score    = s_pos - beta*sim_neg                     # beta 0.5
```
top-k handles pose/viewpoint variation (a prone view matches the prone exemplars, not the mean);
the mean handles noise; the negative subtraction converts a badly-calibrated absolute similarity
into a discriminative one.

**Do not hardcode a global tau.** Compute per-identity on the basestation from the gallery's own
statistics and ship it: `suggested_tau = clip(intra_min_sim - 0.05, 0.35, 0.85)`,
`suggested_margin = clip(0.5*(intra_mean_sim - nearest_other_sim), 0.02, 0.15)`. The drone uses
`max(param_tau_floor, gallery.suggested_tau)` so a pathological 2-blurry-crop gallery can't lower
the bar. If `n_source_crops < 3`, the drone cannot reach `LOCK_CONFIDENT`.

**Exemplar selection** (worker): greedy diversity — take highest quality, then repeatedly take the
highest-quality member whose max cosine to the selected set is `< 0.97`. Gives pose/lighting spread
instead of 8 near-duplicates from one burst.

**Temporal voting before LOCK:** the same `track_id` must win `3` of the last `5` *encoded* verdicts
with mean winning score ≥ tau. At a 3 Hz re-ID cadence that's ~1.0–1.7 s to lock, and it kills the
single-frame flicker that would otherwise slew the gimbal to the wrong person for 300 ms.
**Hysteresis after LOCK:** hold while `score >= tau - 0.05` and margin ≥ `margin_req * 0.5`; drop to
`LOST` after 0.6 s of failing.

### The "argmax isn't the target" case — designed behavior, not a comment

| Condition | `lock_status` | Behavior | `outcome` |
|---|---|---|---|
| No detections for `search_timeout_s` (15 s) | `NONE` | stay in `SEARCHING`, run bounded search scan, no servo | `NO_PEOPLE` |
| `best.score < tau` (people present, none match) | `NO_CONFIDENT_MATCH` | **do not servo**; publish every candidate score; search scan | `NO_TARGET` |
| `score >= tau` but `margin < margin_req` | `AMBIGUOUS` | run 4-stripe spatial refinement on top-2; if unresolved **do not servo**, hold attitude, keep re-evaluating for 8 s | `AMBIGUOUS` |
| `score >= tau`, margin OK, but `n_source_crops < 3` or `sim_neg > s_pos - 0.02` | `LOW_CONFIDENCE` | servo **only if** `allow_low_confidence_lock` (default **false**) | as configured |

**The search scan** (what "not servoing" actually does, so the drone isn't inert): a slow bounded
raster around the attitude commanded at inspection start — `±20°` yaw at `5 °/s`, pitch stepped 5°
per sweep end, hard-bounded, disabled the instant a candidate is accepted.

**Every one of these publishes `PersonServoResult`** on the bridged, `transient_local`
`/<robot>/person_servo/result`, so the basestation can re-task, request a richer gallery, or fall
back to LLA-only pointing. And the full sorted `CandidateScore[]` is always published even when
nothing is accepted — when the operator says "it locked onto the wrong guy", the bag has the scores.

---

## F. Tracking between re-ID

Detector at 10 Hz; RADIO encoding on a strict subset — every **new** track immediately, the
**locked** track every 1.0 s, **other** tracks round-robin every 2.0 s capped at 4 encodes/cycle so
a crowd can't blow the budget. Forced re-encode on re-association after a >0.4 s gap, bbox area
change >2×/<0.5×, or weak IoU association.

`SimpleTracker` (~200 lines, numpy only) — deliberately **not** ByteTrack/BoT-SORT from
`ultralytics`, which are coupled to `ultralytics.Results` objects and would break the
pluggable-detector requirement (GDino doesn't produce them). Constant-velocity Kalman on
`(cx, cy, area, aspect)`, cost = weighted IoU + centroid distance + log-area-ratio + optional
appearance cosine, greedy assignment with a gate.

**Drift protection, three mechanisms:**
1. **Guarded appearance EMA** — a track's descriptor memory updates *only when `score(d) >= tau`*.
   This is the difference between "recovers from a 2 s occlusion" and "slowly latches onto the
   person who walked in front".
2. **The gallery is the anchor, never the track.** The lock decision always scores against the
   shipped gallery; the EMA is used only for association cost. There is no path by which the
   identity criterion itself can drift.
3. **Hard re-verification floor** every 3.0 s regardless. On failure drop to `VERIFYING` (not
   straight to `LOST`) and re-run the vote.

**ID-switch detection:** if the locked track's score falls below `tau - hysteresis` in the same
frame another track's rises above `tau`, encode both immediately; transfer the lock only on a full
`margin_req` win, and `WARN` with both track ids into `MatchDebug`.

---

## G. Servo controller

**Pixel → angle.** Use `atan`, not the small-angle approximation: at the image edge
`err_x ≈ 960, fx = 1378` → 34.9°, where small-angle gives 39.9° — a 14% overshoot at exactly the
moment the loop is most aggressive.
```
yaw_err_deg   = degrees(atan2(bbox_cx - cx_eff, fx_eff))
pitch_err_deg = degrees(atan2(bbox_cy - cy_eff, fy_eff))
```

**Intrinsics** are read from `config/reid/ufm/vehicles/${ROBOT_NAME}_eo.yaml` directly
(`$AIRLAB_PATH` is bind-mounted whole into the drone container). Do **not** copy fx/fy into a second
config — a duplicated intrinsic that silently diverges is a classic. Rescale per frame by
`decoded_w / intrinsics.image_w` since the payload's stream resolution is reconfigurable.

**Zoom.** The `spiritnx3_eo.yaml` "no zoom to key on" comment is about the *reid DB path*, not live
telemetry — `spirit_driver` does publish `gremsy/params/eo_zoom` and `gimbal_state.zoom_level`.
Handling: (1) assert 1× on entering `SEARCHING` via `cmd/zoom_range`, wait 2 s (`force_zoom_1x`
default true) so the static intrinsics become *true* rather than assumed; (2) otherwise scale
`fx_eff` by live `zoom_level`; (3) if stale >2 s or ≤0, fall back to 1.0, set `zoom_valid=false`,
and **derate the PID output by 0.5** — a conservative loop with wrong intrinsics converges slowly;
an aggressive one oscillates.

**Signs are unknown a priori and must not be guessed.** Two independent unknowns: command→gimbal
motion, and gimbal motion→pixel motion. (`camera_extrinsic_rpy: [90,0,90]` strongly suggests image
axes are *not* trivially aligned with gimbal axes.) `pan_sign`/`tilt_sign` live in
`config/servo/${ROBOT_NAME}.yaml` and the node **refuses to leave `dry_run` until
`sign_calibration_verified: true`**. Procedure in §I.

> **Telemetry footgun:** `gremsy_ros_topics.h` documents `gimbal_orientation` as `(x=roll, y=pitch,
> z=yaw)` while `cmd/gimbal_angle` is `(x=pitch, y=roll, z=yaw)`. **The servo module never
> subscribes to `gimbal_orientation`** — it uses `gremsy/gimbal_state` exclusively, which has named
> fields and cannot be transposed by accident. Comment this at the subscription.

**Controller** (`control.py:AxisPID`): parallel PI with derivative-on-measurement. Input is angular
error in **degrees**, output **deg/s**, so `Kp` is dimensionless — `Kp = 1.2` means "close the error
with a ~1.2 s time constant", a physically interpretable number. Start `Kp=1.2, Ki=0.15, Kd=0.0`.
Anti-windup: integrator clamped, and integration frozen when saturated, in `HOLD`/`LOST`/`IDLE`, or
when `zoom_valid == false`; reset on every entry to `SERVOING` and every lock transfer. Clamps:
`max_rate_dps` 20 pan / 15 tilt (**3 during bring-up**) plus a `max_accel_dps2` slew limiter, which
is what stops a detector blink from producing a step command. Stale errors (>0.35 s) decay the
commanded rate toward zero rather than being integrated.

### Settle-and-hold — the "lock on and just stay there" requirement

```
SERVOING:
  |err_px| < enter_deadband_px  for hold_confirm_s
     -> publish (0.0, 0.0) once, reset integrators
     -> HOLD, and STOP PUBLISHING RATE COMMANDS ENTIRELY

HOLD:  (no rate commands at all; gimbal LOCK mode holds earth-frame attitude)
  |err_px| > exit_deadband_px for exit_confirm_s   -> SERVOING
  enter < |err_px| <= exit AND err monotonically increasing over creep_window_s
                                                   -> CREEP: one bounded low-gain nudge, back to HOLD
  time_in_HOLD > hold_reassert_s                   -> publish (0,0) keep-alive, stay in HOLD
```
Defaults: `enter=25 px`, `exit=60 px`, `hold_confirm=0.4 s`, `exit_confirm=0.15 s`,
`creep_gain=0.25*Kp`, `hold_reassert=2.0 s`. All params. 25/60 px on a 1920-wide frame is 1.0°/2.5°
— tight enough to keep the person well inside frame, loose enough that detector jitter (±5 px)
never wakes the loop. The CREEP branch tracks a slowly-drifting casualty without chattering.

**Gimbal mode:** command `cmd/gimbal_mode = 1 (LOCK)` on entering `SERVOING`. In FOLLOW the gimbal
yaw tracks the airframe, so "stop commanding and stay put" only works if the aircraft is perfectly
stationary — which it isn't. LOCK is what makes settle-and-hold *correct*. Also publish
`cmd/track = 0` — two controllers driving one gimbal is guaranteed oscillation.
> Note the command enum (`0=OFF,1=LOCK,2=FOLLOW,3=MAPPING,4=RESET`) is **not** `GimbalState.mode`'s
> enum (`0=UNSPECIFIED,1=FOLLOW,2=LOCK,...`). Loud comment at both call sites.
> `spirit_driver.cpp:312` hardcodes `msg.mode = 0`, so the prior mode is unreadable — restore to a
> configured `default_gimbal_mode` and file a request for real mode telemetry.

### Safety — four layers, because they fail differently

1. **In-loop staleness.** 20 Hz timer independent of the image callback. No valid detection for
   0.5 s, or no image for 1.0 s → publish `(0,0)` three times, `rates_zeroed=true`, enter `LOST`.
2. **Clean shutdown.** `on_shutdown` + `atexit` publish `(0,0)` **five times at 20 ms spacing** —
   ROS 2 publisher destruction is asynchronous and a single publish right before `shutdown()` can be
   dropped.
3. **Hard death (SIGKILL/OOM/TRT segfault).** `gimbal_deadman_node` — ~100 lines, separate process,
   no GPU, no torch. Subscribes a 10 Hz heartbeat published in **every** state including `HOLD` (so
   silence unambiguously means "the node is gone", not "we're holding"). On a >0.5 s gap *and* last
   observed `rates_zeroed == false`, it publishes `(0,0)` at 5 Hz for 3 s then 1 Hz indefinitely.
   Register the servo node with the existing `zombie_killer` package too.
4. **Driver-side timeout — the correct fix, needs the gremsy owner.** `spirit_driver` forwards
   `cmd/gimbal_tilt`/`pan` straight to `setGimbalSpeed(..., INPUT_SPEED)` with **no timeout** — a
   rate command persists forever. Propose `rate_cmd_timeout_ms` (default 500, 0 = disabled for
   backward compat). **Raise this in the same conversation as the tee-branch change.** Until it
   exists, layer 3 is the mitigation.

Also: `spirit_driver` does no clamping at all. Our node additionally refuses a rate that would drive
`gimbal_state.tilt_deg` past `[-90, +20]` within a 0.5 s lookahead.

**`servo_backend` param:** `rate_pid` (default) | `payload_tracker` (hand the winning bbox centre to
`cmd/track_touch` and let the payload's hardware tracker hold it — zero tuning, no sign conventions,
but we can't see what it's actually tracking and reacquisition is undocumented; **worth benching,
could end up the shipped default for stationary casualties**) | `dry_run` (computes and publishes
everything, sends nothing — the bring-up default).

---

## H. Node and state machine

Namespace `/<ROBOT_NAME>/person_servo`. Internal topics relative; cross-subsystem contract topics
**absolute and injected as ROS params**, exactly the pattern documented in
`gremsy_ros2/config/spiritnx3.yaml`. Config at `config/servo/${ROBOT_NAME}.yaml`.

**Triggering.** Per finding #1, the **gallery is the trigger of record** — inspection context is
`(global_id, session_id, command_id)` and it comes from `PersonGallery`, full stop.
`InspectCasualty` (both flavors) is subscribed for exactly three things: **arm**
(`IDLE → GALLERY_WAIT`, start the 10 s clock), **correlate** (echo `command_id` in the result,
whatever it means), and **request** (publish `PersonGalleryRequest{global_id: -1, reason:
"no_gallery"}`). Gallery↔command matching: exact `command_id` match, else accept if armed with no
command_id, else accept within a 20 s window when no other gallery is pending, else keep waiting.
Add a comment block citing `spirit_command_builder.py` so nobody "fixes" this into `int(command_id)`.

**States:** `IDLE` → `GALLERY_WAIT` (10 s timeout → `ABORT/NO_GALLERY`) → `SEARCHING` (asserts zoom
1×, LOCK mode, payload tracker off; 15 s timeout) → `VERIFYING` (N-of-M vote, **no servo commands
yet**) → `LOCKED` (transient: reset PID, record `time_to_lock_s`, publish result) → `SERVOING` ⇄
`HOLD` → `LOST` (zeroes rates immediately, attempts reacquisition, 8 s → widen → `ABORT`) →
`ABORT` (zero rates, restore gimbal mode, publish result) → `IDLE`. Cross-cutting: a
`person_servo/abort` Trigger service, and `inspect_max_duration_s` (120 s).

**Gallery validation on receipt — fail loud**, `ABORT(MODEL_MISMATCH)` on: `model_key` string
mismatch, any `dim`/length disagreement, `|norm(mean) - 1.0| > 1e-3` (catches an un-normalized
producer), `n_source_crops < 1`. The `model_key` check is the entire reason that field exists — a
silently mismatched preprocessing spec is undetectable at runtime *except* here.

**Published:** `state` (`ServoState`, 10 Hz, best_effort, bridged), `result` (reliable,
transient_local, bridged), `gallery_request` (bridged), `match_debug` (2 Hz, bag only),
`heartbeat` (10 Hz), `debug_image/compressed` (2 Hz, **opt-in, default off** — JPEG-encoding 1080p
is real CPU; draws all boxes with scores, the winner, the deadband rectangle, and the state, which
is what makes field debugging tractable). Add all to
`config/logging/bag_record_pid_spiritnx3.yaml` and `config/diagnostics/health_agg_spirit.yaml`.

---

## I. Onboard imagery — decision pending the gremsy owner conversation

All three options implement the same `ImageSource` protocol (`start()` / `latest() -> (bgr, stamp)`
/ `stop()`), so the choice is a config change, not a rewrite. Implementations: `RosImageSource`,
`RtspImageSource`, `ReplayImageSource`.

### Option A — opt-in third tee branch in `sender.cpp` *(preferred)*
```
t. ! queue name=aiq leaky=downstream max-size-buffers=2 !
     h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx !
     videoconvert ! video/x-raw,format=BGR !
     appsink name=aisink sync=false max-buffers=1 drop=true
```
Guarded by `enable_ai_branch`, **default false** — the pipeline string is built without the branch
when false, so existing behavior is byte-identical. Publishes `sensor_msgs/Image` (bgr8),
`frame_id="camera"`, stamped from the **SEI timestamp** (reuse `receiver.cpp`'s parse) so drone-side
and basestation-side stamps match. Rate-limited by `ai_publish_every_n` (default 3 → 10 Hz from 30).
- **Pros:** one RTSP connection, one decode, hardware decode, identical timestamps, smallest new code.
- **Cons:** touches another team's safety-critical file; `leaky=downstream` is doing all the work
  protecting the 1 s `exit(1)` watchdog (finding #3); 1080p BGR over DDS is ~62 MB/s at 10 Hz even
  on loopback.

### Option B — standalone second RTSP client inside the servo node *(fallback 1)*
`RtspImageSource` opens its own `rtspsrc` against the payload, entirely inside `person_servo_node`.
- **Pros:** no dependency on another team, no DDS image traffic at all (fastest), complete isolation
  — a crash here cannot affect the basestation feed, and it can be built today.
- **Cons:** a second concurrent RTSP client on the Gremsy payload — **unverified whether the payload
  supports concurrent sessions.** If it caps at one, this silently kills the basestation stream,
  which is *worse* than Option A's risk. Also duplicates decode.
- **5-minute test that decides viability:** open two `rtspsrc` sessions against
  `rtsp://192.168.70.23:8554/payload` simultaneously and confirm both get frames.

### Option C — composable node + intra-process comms *(fallback 2, not recommended)*
Zero-copy `unique_ptr<Image>` passing. Technically cleanest, but `sender.cpp` is a `main()` with
globals, a `GMainLoop`, and an `exit(1)` thread — none of it component-safe — and **intra-process
comms only works if the consumer is C++ in the same container.** The servo node is Python, so this
would require rewriting it in C++ or accepting serialization anyway, which eliminates the entire
benefit. Only worth it if the servo node becomes C++, which the python-first detector/encoder
ecosystem makes unlikely.

**Regardless of the outcome, `ReplayImageSource` keeps phases 5–7 unblocked.**

---

## J. Detector — YOLO default, GDino benchmarked

```python
class PersonDetector(Protocol):
    def warmup(self) -> None: ...
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...
    @property
    def name(self) -> str: ...
```
Implementations: `YoloTrtDetector` (default), `YoloTorchDetector`, `GDinoDetector`,
`ReplayDetector` (reads boxes from a bag/JSON — makes the whole downstream pipeline testable with
**no detector at all**), `NullDetector`. Selected by a `detector_backend` param via a registry.
GDino needs a text prompt (`"person . casualty ."`) and the local BERT dir
(`humanflow_core.weights.ensure_bert`) — a constructor detail, invisible to the interface. That's
the test of whether the abstraction is real, and it passes.

`tools/bench_detector.py`, run **on the Orin NX** over replayed frames, measuring p50/p95/p99
latency, peak GPU memory and sustained power (`tegrastats`, must coexist with RADIO), recall and
precision @ IoU 0.5 against `detections_2d` rows as pseudo-GT, **end-to-end time-to-lock** driven
through the real node in `dry_run`, and missed-frame rate on the *target's* box specifically.

Contenders: YOLO11n/s at 640 and 960 (small aerial people need the resolution), and GroundingDINO
(`weights/humanflow/gdino_swint_darpa-ir-v1-1k_20_1.pth` — already provisioned and already
fine-tuned on DARPA-relevant data, a real accuracy advantage).

**Decision criterion, stated up front so the bench answers a question rather than producing
numbers:** GDino ships if `recall_gdino − recall_yolo > 0.10` **and** `p95_gdino < 220 ms` **and**
end-to-end time-to-lock is within 1.5× of YOLO. Otherwise YOLO-TRT is default and GDino stays
selectable via `detector_backend: gdino`.
> Unproven: whether `humanflow_core`'s GDino wrapper is importable on arm64 — it needs the CUDA
> extension from `strapsai/GroundingDINO`, built in `08b-reid.dockerfile` for x86 arch lists only.
> An arm64 build with `TORCH_CUDA_ARCH_LIST="8.7"` is a new dockerfile line and untested.

---

## K. Deployment

**Onboard:** new `dtc-dockerfiles/jetpack6.1/nx/06d-jp6.1-person-servo.dockerfile` from
`jp6.1-05d-triage-sensor`, producing `jp6.1-06d-person-servo`; add the service to
`dtc-dockerfiles/jetpack6.1/docker-compose.yaml` and point `launch/spirit/jetpack.env`'s
`SPIRIT_BASE_IMG` at it. torch 2.4.0/CUDA 12.6/torchvision 0.19 come free from `01-jp6.1-torch`;
`einops` from `04-stable-third-party`. New: `timm`, `huggingface-hub`, `safetensors`, `ultralytics`
— all with `--no-deps`, matching the existing convention (ultralytics otherwise upgrades numpy and
opencv out from under the image; 05c/11a have scar tissue about exactly this).

**Basestation:** no new image. `08b-reid.dockerfile` already has torch 2.8, TRT, timm, einops,
transformers, psycopg2, boto3. Add one line: `pip3 install pgvector`. The gallery nodes run
**inside the existing `reid_ros2` container** as a new window in `launch/reid-mapping/reid.yaml` —
they need DB creds + MinIO + GPU + `db_api` + torch, all of which it already has, and a new
container would add a fifth thing to `just dbr-up`.

**Weights:** `$AIRLAB_PATH/weights/person_servo/` + a Swift-bucket download script structurally
cloned from `humanflow_bringup/download_models.py`. Holds `torchhub/` (the offline hub cache),
`onnx/`, `detector/`, and `parity/` (64 golden crops + reference descriptors).
`engines/` is built on-device and `.gitignore`d.

**tmuxp:** new `Person Servo` window in `launch/spirit/dtc-drivers.yaml` after `Gremsy`, with two
panes — `person_servo_node` and `gimbal_deadman_node`.

**Domain bridge:** four new entries per drone in `config/ros_domain/basestation_drivers_spirit.yaml`
— `person_servo/gallery` (to drone, reliable, **transient_local** so a servo-node restart
mid-inspection isn't stranded in `GALLERY_WAIT`), and `result` / `gallery_request` / `state` back.
> **`person_servo_msgs` must also be built in `basestation_drivers_ws`** — `domain_bridge` runs
> there and needs the typesupport library. Forgetting this yields a bridge that starts cleanly and
> silently drops the topic. Add the repo to all three `version_control/*.yaml`.
>
> **Bandwidth:** 1536 × 4 B = 6.1 KB/descriptor → mean + 8 exemplars + 8 negatives ≈ **104 KB** per
> gallery. Size the link budget for that; if it hurts, drop to 4+4 (~55 KB) or add fp16 encoding.

---

## L. Verification

**Unit tests** (`person_descriptor/tests/`, pytest, no ROS/GPU): preprocessing determinism and that
BGR-vs-RGB inputs produce *different* outputs (guards a dropped `cvtColor`); descriptor unit-norm and
that the wrong normalize/concat order gives a different vector (a regression tripwire); every branch
of the §E table; tracker ID-switch and `max_age`; and a **property test on the controller** — for
any random error/dt sequence, `state == HOLD ⟹ no commands after the settling one`, and
`state ∈ {LOST, IDLE, ABORT} ⟹ last command was (0,0)`. That invariant is what keeps a gimbal from
running away.

**Cross-architecture parity — the load-bearing test.** 64 fixed crops + reference descriptors
generated once on x86. Run as `ros2 run spirit_person_servo parity_check` on the target *and as a
startup self-check* (`parity_on_startup`, default true) that refuses to leave `dry_run` on failure.

| Comparison | Gate |
|---|---|
| our wrapper vs `RADIOEncoder`, both x86 fp32 | max abs diff on `features` < 1e-5 |
| x86 fp32 vs Orin fp32 / fp16 / TRT-fp16 | min cosine > 0.9999 / 0.999 / 0.995 |
| **rank stability**: 64×64 sim matrix, Spearman ρ | **> 0.99** |
| **decision stability**: §E scorer on 200 labeled pairs | **100% identical accept/reject** |

The last two matter more than raw cosine. A uniform 0.996 shift is harmless; a *reordering* of two
close candidates is exactly the failure that puts the gimbal on the wrong casualty.

**Offline eval — the DB is already a labeled ReID dataset, and it's free.** `tools/replay_eval.py`:
leave-one-out over each `global_id` (each crop is a query, the rest of its identity is the gallery,
all other identities are negatives) → rank-1, rank-5, mAP, and the positive/negative score
distributions. **Sweep `w_summary` ∈ {0, .25, .5, .75, 1}, input size ∈ {256×128, 384×192, 448×224},
`alpha`, `beta`, `k`, model ∈ {`-b`, `-l`, `e-radio_v2`}. The winner defines `MODEL_KEY`, and the
score histograms replace the guessed `suggested_tau` constants in §E.** Everything in §D/§E is a
starting hypothesis until this runs.

**Gimbal sign calibration — safe procedure.** Preconditions: bench stand, **props removed**, not
armed, E-stop in reach, `max_rate_dps: 3.0`, `servo_backend: dry_run`.
1. *Command → motion (no camera):* `sign_calibration_node` records `pan_deg`, commands +2 °/s for
   1 s, zeroes, settles, records again; repeats at −2 °/s. Asserts opposite signs and |Δ| ≈ 2°;
   aborts on any |Δ| > 10°.
2. *Motion → pixels:* high-contrast target in view, record `bbox_cx`, command +2 °/s pan for 1 s,
   record again. Same for tilt/`bbox_cy`.
3. `pan_sign = -sign(A_pan * B_pan)`; the node prints the exact YAML block to paste, including
   `sign_calibration_verified: true`.
4. *Closed-loop, still on the bench:* `rate_pid` at 3 °/s, target 100 px off-center — error must
   decrease monotonically. **Ship the divergence guard: if |err| grows for 8 consecutive cycles
   while commanding non-zero rate, abort to `LOST`, zero rates, log `SIGN CONVENTION LIKELY
   INVERTED`.** Cheapest possible insurance against a runaway.
5. Raise to 10 then 20 °/s, tuning `Kp`/`Ki` **on the bench, not in flight**. Repeat per airframe —
   do not assume the three gimbals are mounted identically.

**Staged bring-up:** (0) x86 replay + `ReplayDetector` + `dry_run` → (1) Orin on bench, RTSP from a
payload on a stand, parity passes, thermals stable 10 min → (2) bench `rate_pid` at 3 °/s, sign
calibration on all three airframes → (3) **fault injection: SIGKILL the node, unplug RTSP, feed an
empty gallery — rates zeroed in every case, deadman fires, no gimbal motion after node death** →
(4) tethered flight, gimbal only, operator on E-stop → (5) free flight. Stage 3 is not optional:
"gimbal keeps slewing after the node died" is only findable by actually killing the node.

---

## M. Phases

Phase 1 gates everything. `1→2→3` (basestation) and `1→5→6→7` (drone) then run in parallel;
Phase 4 is independent.

1. **`person_descriptor`** — no ROS, no drone, no DB. The lib, the full unit suite, `replay_eval.py`.
   *Done when:* pytest green and the eval sweep has picked `MODEL_KEY`, `tau`, `margin_req`. **Do not
   shortcut the eval** — every later phase assumes its outputs.
2. **DB schema + `gallery_worker_node`** — DDL, `ensure_embedding_tables()`, the ~12 `db.py`
   functions, `pgvector` in `08b-reid.dockerfile`, `backfill_embeddings`.
   *Done when:* backfill runs on an existing session, `intra_mean_sim > nearest_other_sim` for
   well-separated identities, and the claim loop is verified under two concurrent workers.
3. **`person_servo_msgs` + `gallery_publisher_node` + bridge** — package in three workspaces and
   three vcs yamls. *Done when:* a fake `InspectCasualty` on domain 100 produces a `PersonGallery`
   that crosses to domain 3 with the right QoS; serialized size measured.
4. **Onboard imagery** — whichever option is approved, plus `ReplayImageSource`. *Done when:*
   `ros2 topic hz` on the Orin **and** the basestation feed is provably unaffected over a 10-minute
   soak with zero `exit(1)`s. **Start Phase 5 in parallel on a bag; do not let this block.**
5. **Detector + encoder onboard** — `06d` layer, weights bucket, `PersonDetector` impls, ONNX export
   + on-device engine build, `parity_check`. *Done when:* parity passes on the Orin and the
   GDino-vs-YOLO decision is made with data.
6. **State machine, `dry_run` only** — the full node, all messages, `debug_image`, diagnostics,
   `gimbal_deadman_node`. *Done when:* a replayed bag drives `IDLE → … → LOCKED`, and hand-built
   galleries force each of the four §E outcomes. Rates are computed and published; nothing moves.
7. **Closed loop** — `sign_calibration_node`, `config/servo/*.yaml`, divergence guard, tuned gains.
8. **Integration + hardening** — tmuxp windows, bag config, health aggregator, `payload_tracker`
   benched against `rate_pid`, the `spirit_driver` rate-timeout request, runbook.

---

## N. Risk register (critical only)

| Risk | Mitigation |
|---|---|
| Preprocessing/normalization/color mismatch between sides — **silent** | Shared lib; `model_key` check on gallery receipt; parity gate at startup; `input_color_order` has **no default** |
| ImageNet normalization applied by mistake (RADIO wants raw [0,1]) | `PreprocessSpec.value_range` explicit + comment citing `image_normalizations.py:29`; parity test catches it |
| `global_id` derived from `command_id`/`casualty_id` | Never done; comes only from `PersonGallery`; comment citing both producers |
| Tee branch backpressures `parse` → `exit(1)` → basestation loses video | `leaky=downstream max-size-buffers=2` + `appsink drop=true`; default-off param; 10-min soak |
| Gimbal sign inverted → runaway slew | `dry_run` default; `sign_calibration_verified` gate; divergence guard; props-off bench calibration |
| Servo node SIGKILLed with a rate latched | `gimbal_deadman_node`; request driver-side `rate_cmd_timeout_ms` |
| TRT FP16 reorders two close candidates — **silent** | Rank- and decision-stability gates, not just cosine; layer-precision override if needed |
| Stale/non-portable TRT engine | Built on-device, cached per `{sm_arch}-trt{ver}`, never shipped, torch fallback with a loud log |
| `torch.hub` needs internet on an offline drone | Pre-provisioned `TORCH_HOME`; the TRT path needs no hub at all |
| `person_servo_msgs` missing in `basestation_drivers_ws` → bridge silently drops | Explicit Phase 3 item + a cross-domain smoke test |
| `person_descriptor` drifts between the two workspaces | Pin to a **tag**, not `main`; runtime `model_key` + `lib_version` check |

---

## O. Open items to verify (do not treat as designed)

1. Whether TensorRT / `nvv4l2decoder` / `nvvidconv` exist inside `jp6.1-05d-triage-sensor` at
   **build** time or only at **runtime** via the nvidia container runtime's CSV mounts. Gates
   whether engines can be built in the dockerfile (§D assumes first-run build either way).
2. Whether the Gremsy payload's RTSP server accepts concurrent sessions — gates imagery Option B.
   5-minute test.
3. The pgvector version in `chiron_db` (gates `halfvec`, and the 2000-dim indexed ceiling).
4. Whether the pinned `NVlabs/RADIO` hub commit exposes C-RADIO v3, and at what dims/patch size.
5. Whether `gimbal_state.zoom_level` is a linear optical magnification factor or a step index —
   ruler-at-known-distance test during bring-up. If non-linear, replace the scalar with a LUT.
6. Whether `/spiritnx3/basestation_logic/current_run_mode` is bridged to domain 3 — it is **not** in
   `basestation_drivers_spirit.yaml` today. The ESTOP/run-mode abort hook depends on it; either add
   the entry or drop the hook.
7. Real `radio_v2.5-b` latency at 384×192 on an Orin NX under TRT FP16. Every cadence number in §F
   is budgeted from an assumption; Phase 5's bench replaces them.

### To raise with the gremsy driver owner (tomorrow)
- The opt-in `enable_ai_branch` tee addition to `sender.cpp` (Option A) — and specifically that the
  new queue must be `leaky=downstream`, because the SEI probe that feeds the 1 s `exit(1)` watchdog
  sits upstream of the tee.
- A `rate_cmd_timeout_ms` parameter on `spirit_driver`'s rate-command path (safety layer 4).
- Publishing the real `PAYLOAD_CAMERA_GIMBAL_MODE` into `GimbalState.mode` instead of the
  hardcoded `0` at `spirit_driver.cpp:312`.
- Whether the payload's RTSP server tolerates a second concurrent client (decides Option B).

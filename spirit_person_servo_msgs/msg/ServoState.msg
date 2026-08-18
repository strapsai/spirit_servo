# Person servo telemetry. Published at a fixed rate in every state, including HOLD,
# so silence unambiguously means "the node is gone" rather than "we are holding".
#
# Phase 1 (standalone, no re-ID): the target is chosen by largest-bbox + continuity.
# Identity/gallery fields (match score, margin, lock_status) are deliberately absent
# until the re-ID phase adds them.
std_msgs/Header header

# --- State machine ---
# IDLE      : not armed. Nothing commanded.
# SEARCHING : armed, looking for a person to lock onto. Nothing commanded.
# LOCKED    : transient; target just confirmed, controller reset.
# SERVOING  : actively driving the gimbal to center the target.
# HOLD      : settled inside the deadband. Nothing commanded.
# LOST      : target gone or data stale. Actuation zeroed/stopped.
uint8 state
uint8 STATE_IDLE=0
uint8 STATE_SEARCHING=1
uint8 STATE_LOCKED=2
uint8 STATE_SERVOING=3
uint8 STATE_HOLD=4
uint8 STATE_LOST=5

# Which actuation path is active. DRY_RUN computes and reports everything but
# sends nothing to the gimbal.
uint8 backend
uint8 BACKEND_DRY_RUN=0
uint8 BACKEND_TRACK_TOUCH=1
uint8 BACKEND_ANGLE=2
uint8 BACKEND_RATE=3

# --- Target ---
bool  has_target
int32 target_track_id      # -1 when no target
float32 bbox_x             # top-left, pixels, in decoded-frame coordinates
float32 bbox_y
float32 bbox_w
float32 bbox_h
float32 detector_conf
uint32  n_detections       # people seen this frame

# --- Error ---
float32 err_px_x           # target center - effective image center
float32 err_px_y
float32 err_deg_yaw        # atan-based, NOT the small-angle approximation
float32 err_deg_pitch

# --- Command actually issued this cycle (zeros when nothing was sent) ---
# RATE backend
float32 cmd_rate_pan_dps
float32 cmd_rate_tilt_dps
# ANGLE backend (absolute setpoint)
float32 cmd_angle_pan_deg
float32 cmd_angle_tilt_deg
# TRACK_TOUCH backend (pixel handed to the payload's own tracker)
float32 cmd_touch_px_x
float32 cmd_touch_px_y

# True whenever this cycle issued no motion at all (nothing published, or an
# explicit zero-rate stop). Explicit rather than inferred so a bag reader can
# assert the safety invariant:  state in {IDLE, HOLD, LOST} => actuation_idle
bool actuation_idle

# False in dry_run: everything is computed and reported, nothing is sent.
bool commands_enabled

# --- Gimbal telemetry (from lion_ros2_bridge/GimbalState) ---
bool    gimbal_valid       # false when gimbal_state is missing or stale
float32 gimbal_pan_deg
float32 gimbal_tilt_deg
uint8   gimbal_mode

# --- Health ---
float32 image_age_s
float32 detection_age_s
float32 gimbal_age_s
float32 zoom_level
bool    zoom_valid
string  status_text

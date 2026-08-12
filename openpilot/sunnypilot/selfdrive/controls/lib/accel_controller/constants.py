import math

import numpy as np

from openpilot.cereal import custom


AccelProfile = custom.LongitudinalPlanSP.AccelController.Profile
ACCEL_PROFILES = tuple(AccelProfile.schema.enumerants.values())

COMFORT_DECEL = {
  AccelProfile.eco: 0.25,
  AccelProfile.normal: 0.30,
  AccelProfile.sport: 0.35,
}

ACCEL_PROFILE_MAX_BP = [0.0, 3.0, 10.0, 25.0, 40.0]
ACCEL_PROFILE_MAX_V = {
  AccelProfile.eco: [1.67, 1.30, 0.72, 0.32, 0.24],
  AccelProfile.normal: [1.80, 1.51, 0.98, 0.53, 0.35],
  AccelProfile.sport: [2.00, 1.91, 1.16, 0.73, 0.47],
}

BRAKING_ACCEL_THRESHOLD = -0.11

CAP_FILTER_FRAMES = 5
PERSIST_TIME = 0.50            # trust-register relief corroboration window
LEAD_DROPOUT_COAST_TIME = 1.50  # pre-roll before a vanished lead counts as a relief candidate
LEAD_SWITCH_MAX_HOLD_TIME = 6.0  # fail-safe: a sustained relief can't be blocked forever

SPEED_DEADBAND = 0.15

TARGET_RELEASE_SLEW = 8.75      # also used for the launch ramp
LAUNCH_TARGET_HEADROOM = 3.0
LAUNCH_END_SPEED = 3.0

LEAD_MATCH_HEADROOM = 1.25
LEAD_MATCH_ACCEL_SLEW = 0.25    # slew rate for the matched-lead accel ceiling
MATCHED_SPEED_DECEL_RATE = 0.50  # separate, faster rate for target_speed's own pull-down while matched

STOP_HOLD_EGO_SPEED = 0.30
STOP_HOLD_SPEED_FLOOR = 0.15    # "is the lead basically stopped / not really moving" floor
STOP_HOLD_EXIT_SPEED = 0.80     # fast-lane: unambiguous departure, skip the growth check
STOP_HOLD_EXIT_FRAMES = 4
STOP_HOLD_CREEP_DISTANCE = 0.30
STOP_HOLD_MAX_LEAD_DISTANCE = 30.0  # a nearly-stopped lead farther than this can't arm stop-hold on cap alone

STOP_GAP_RESERVE = 0.75

RADAR_STALE_TIMEOUT = 0.50
MAX_LEAD_ACCEL_TAU = 10.0
MIN_LEAD_SPEED = -1.0
VEGO_NOISE_TOLERANCE = 0.10
PARAM_READ_INTERVAL = 0.25
ACCEL_LIMIT_HORIZON_JERK = 1.0

MPC_DECEL_JERK_COST_MULTIPLIER = 1.05
MPC_DECEL_JERK_MAX_REQUIRED_DECEL = 0.80
MPC_DECEL_JERK_MAX_REQUIRED_DECEL_RATE = 0.35
MPC_DECEL_JERK_LONG_TREND_FRAMES = 6   # 0.3s - long enough to separate "rose once, plateaued" from "still climbing"
MPC_DECEL_JERK_LONG_TREND_RATE = 0.02   # much lower than the short-window rate: catches a slow, sustained climb
MPC_DECEL_JERK_MAX_TARGET_REDUCTION = 9.0
MPC_DECEL_TREND_FRAMES = 4


def sanitize_profile(profile: int) -> int:
  return profile if profile in ACCEL_PROFILES else AccelProfile.normal


def profile_accel_max(profile: int, v_ego: float) -> float:
  if not math.isfinite(v_ego):
    return math.nan
  return float(np.interp(max(v_ego, 0.0), ACCEL_PROFILE_MAX_BP, ACCEL_PROFILE_MAX_V[sanitize_profile(profile)]))

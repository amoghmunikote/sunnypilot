import math

from opendbc.car.interfaces import ACCEL_MAX
from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.constants import (
  CAP_FILTER_FRAMES, COMFORT_DECEL, LEAD_DROPOUT_COAST_TIME, LEAD_SWITCH_MAX_HOLD_TIME, MPC_DECEL_JERK_COST_MULTIPLIER,
  MPC_DECEL_JERK_LONG_TREND_FRAMES, MPC_DECEL_JERK_LONG_TREND_RATE, MPC_DECEL_JERK_MAX_REQUIRED_DECEL,
  MPC_DECEL_JERK_MAX_REQUIRED_DECEL_RATE, MPC_DECEL_JERK_MAX_TARGET_REDUCTION, MPC_DECEL_TREND_FRAMES,
  PARAM_READ_INTERVAL, PERSIST_TIME, RADAR_STALE_TIMEOUT, SPEED_DEADBAND, VEGO_NOISE_TOLERANCE,
  AccelProfile, profile_accel_max, sanitize_profile,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.helpers import build_accel_ceiling, is_valid_context
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.lead import calculate_lead_plan
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.pace import Pace
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource

AccelControllerState = custom.LongitudinalPlanSP.AccelController.State


class AccelController:
  def __init__(self, CP, dt: float = DT_MDL):
    if not math.isfinite(dt) or dt <= 0.0:
      raise ValueError("dt must be finite and positive")

    self.dt = dt
    self.delay = float(CP.longitudinalActuatorDelay) + DT_MDL
    self.persist_frames = max(CAP_FILTER_FRAMES, math.ceil(PERSIST_TIME / dt))
    self.dropout_frames = max(self.persist_frames, math.ceil(LEAD_DROPOUT_COAST_TIME / dt))
    self.switch_max_frames = max(self.dropout_frames, math.ceil(LEAD_SWITCH_MAX_HOLD_TIME / dt))
    self.radar_stale_frames = max(1, math.ceil(RADAR_STALE_TIMEOUT / dt))
    self.params = Params()
    self.available = bool(CP.openpilotLongitudinalControl)
    self.enabled = False
    self.profile = AccelProfile.normal
    self._param_read_frames = max(1, int(round(PARAM_READ_INTERVAL / dt)))
    self._param_frame = 0
    self._jerk_smoothing_blocked = False
    self._required_decel_samples: list[float] = []
    self._required_decel_long_samples: list[float] = []
    self._required_decel_lead = -1
    self._required_decel_lead_track_id = -1
    self._lead_trend_warmup = False

    self.pace = Pace()
    self._stale_frames = 0
    self._cruise_accel_limited = False

    self.is_active = False
    self.output_v_target = 0.0
    self.mpc_accel_max: tuple[float, ...] | None = None
    self.cruise_accel_max: float | None = None
    self.state = AccelControllerState.inactive
    self.selected_lead = -1
    self.selected_lead_track_id = -1
    self.required_decel = 0.0

  @property
  def is_enabled(self) -> bool:
    return self.available and self.enabled

  @property
  def launching(self) -> bool:
    return self.pace.launching

  @property
  def departure_launching(self) -> bool:
    return self.pace.departure_launching

  def update_params(self) -> None:
    if self._param_frame % self._param_read_frames == 0:
      self.enabled = self.params.get_bool("AccelPersonalityEnabled")
      self.profile = get_sanitize_int_param("AccelPersonality", AccelProfile.eco, AccelProfile.sport, self.params)
    self._param_frame += 1

  def reset(self) -> None:
    self.pace = Pace()
    self._stale_frames = 0
    self._jerk_smoothing_blocked = False
    self._required_decel_samples.clear()
    self._required_decel_long_samples.clear()
    self._required_decel_lead = self._required_decel_lead_track_id = -1
    self._lead_trend_warmup = False
    self._cruise_accel_limited = False
    self.is_active = False
    self.output_v_target = 0.0
    self.mpc_accel_max = None
    self.cruise_accel_max = None
    self.state = AccelControllerState.inactive
    self.selected_lead = -1
    self.selected_lead_track_id = -1
    self.required_decel = 0.0

  def update(self, radar_state, *, base_speed: float, v_ego: float, a_ego: float, follow_personality, acc_selected: bool,
             engaged: bool, cruise_initialized: bool, stock_accel_max: float, radar_fresh: bool = True,
             previous_mpc_source=None, planner_speed: float | None = None, planner_accel: float = 0.0,
             previous_plan_accel: float = 0.0) -> None:
    self.profile = sanitize_profile(self.profile)
    sanitized_v_ego = max(v_ego, 0.0) if math.isfinite(v_ego) and v_ego >= -VEGO_NOISE_TOLERANCE else v_ego
    profile_max_accel = profile_accel_max(self.profile, sanitized_v_ego)
    stock_accel_max = float(stock_accel_max)
    positive_accel_max = (max(0.0, min(profile_max_accel, stock_accel_max, ACCEL_MAX))
                          if math.isfinite(profile_max_accel) and math.isfinite(stock_accel_max) else math.nan)
    planner_speed = sanitized_v_ego if planner_speed is None else planner_speed
    valid_context = is_valid_context(base_speed, sanitized_v_ego, a_ego, planner_speed, planner_accel, stock_accel_max, self.delay,
                                     engaged, cruise_initialized)
    enabled_context = valid_context and self.is_enabled and bool(acc_selected)

    if enabled_context and radar_fresh:
      self._stale_frames = 0
      lead_plan = calculate_lead_plan(radar_state, sanitized_v_ego, a_ego, self.delay, self.profile, follow_personality)
      comfort_decel = COMFORT_DECEL[self.profile]
      self.pace.update(lead_plan, base_speed, sanitized_v_ego, comfort_decel, profile_max_accel, self.dt,
                       self.persist_frames, self.dropout_frames, self.switch_max_frames, planner_speed, planner_accel,
                       previous_mpc_source, previous_plan_accel)
    elif enabled_context:
      self._stale_frames += 1
      if self._stale_frames >= self.radar_stale_frames and not (self.pace.stop_hold or self.pace.launching):
        self.pace.reset()

    active = enabled_context and (radar_fresh or self.pace.target_speed is not None)
    self.is_active = active
    if not active:
      self.pace.reset()
      self._cruise_accel_limited = False
      self.output_v_target = base_speed
      self.mpc_accel_max = None
      self.cruise_accel_max = None
      self.state = AccelControllerState.inactive
      self.selected_lead = -1
      self.selected_lead_track_id = -1
      self.required_decel = 0.0
      return

    pace = self.pace
    self.output_v_target = pace.target_speed if pace.target_speed is not None else base_speed
    self.selected_lead = pace.selected_lead
    self.selected_lead_track_id = pace.selected_lead_track_id
    self.required_decel = pace.required_decel

    matched_limit_active = pace.matched_lead and pace.matched_accel_limit is not None and not pace.e2e_braking_handoff
    lead_accel_request = pace.matched_lead and planner_accel >= 0.0
    profile_limit_active = (not pace.stop_hold and not pace.launching and (not pace.has_lead or lead_accel_request)
                            and not pace.e2e_braking_handoff)
    if matched_limit_active:
      effective_accel_max = min(positive_accel_max, pace.matched_accel_limit)
    elif profile_limit_active:
      effective_accel_max = positive_accel_max
    else:
      effective_accel_max = math.inf
    self.mpc_accel_max = (build_accel_ceiling(effective_accel_max, planner_accel)
                          if matched_limit_active or profile_limit_active else None)

    lead_context = pace.has_lead or math.isfinite(pace.cap_trusted)
    start_cruise_accel_limit = (pace.target_speed is not None and not pace.restricting and not pace.releasing
                                and pace.has_lead and previous_mpc_source == LongitudinalPlanSource.cruise)
    keep_cruise_accel_limit = (self._cruise_accel_limited and lead_context and not pace.restricting and not pace.releasing
                               and not pace.e2e_braking_handoff)
    self._cruise_accel_limited = start_cruise_accel_limit or keep_cruise_accel_limit
    self.cruise_accel_max = positive_accel_max if self._cruise_accel_limited else None

    if pace.stop_hold:
      self.state = AccelControllerState.stopHold
    elif pace.restricting:
      self.state = AccelControllerState.restrict
    elif pace.releasing:
      self.state = AccelControllerState.release
    elif pace.target_speed >= base_speed - SPEED_DEADBAND:
      self.state = AccelControllerState.free
    else:
      self.state = AccelControllerState.hold

  def get_jerk_cost_multiplier(self, actuating: bool, prev_accel_constraint: bool, target_reduction: float,
                               previous_mpc_failed: bool) -> float:
    lead_restriction = (actuating and prev_accel_constraint and self.state == AccelControllerState.restrict and self.selected_lead >= 0
      and not self.launching and target_reduction > 1e-6)
    same_lead = self.selected_lead == self._required_decel_lead and self.selected_lead_track_id == self._required_decel_lead_track_id
    lead_changed = lead_restriction and self._required_decel_lead >= 0 and not same_lead
    if lead_changed:
      self._lead_trend_warmup = True
    elif not lead_restriction:
      self._lead_trend_warmup = False
    if not lead_restriction or not same_lead or not math.isfinite(self.required_decel):
      self._required_decel_samples.clear()
      self._required_decel_long_samples.clear()
    if lead_restriction and math.isfinite(self.required_decel):
      self._required_decel_samples.append(self.required_decel)
      if len(self._required_decel_samples) > MPC_DECEL_TREND_FRAMES:
        self._required_decel_samples.pop(0)
      self._required_decel_long_samples.append(self.required_decel)
      if len(self._required_decel_long_samples) > MPC_DECEL_JERK_LONG_TREND_FRAMES:
        self._required_decel_long_samples.pop(0)
    self._required_decel_lead = self.selected_lead if lead_restriction else -1
    self._required_decel_lead_track_id = self.selected_lead_track_id if lead_restriction else -1

    history = self._required_decel_samples
    history_ready = len(history) == MPC_DECEL_TREND_FRAMES
    tightening_lead = (history_ready
      and (history[-1] - history[0]) / (self.dt * (len(history) - 1)) > MPC_DECEL_JERK_MAX_REQUIRED_DECEL_RATE
      and sum(after > before for before, after in zip(history[:-1], history[1:], strict=True)) >= 2)
    long_history = self._required_decel_long_samples
    long_history_ready = len(long_history) == MPC_DECEL_JERK_LONG_TREND_FRAMES
    sustained_tightening = (long_history_ready
      and (long_history[-1] - long_history[0]) / (self.dt * (len(long_history) - 1)) > MPC_DECEL_JERK_LONG_TREND_RATE)
    modest_decel = (lead_restriction and target_reduction < MPC_DECEL_JERK_MAX_TARGET_REDUCTION
      and 0.0 < self.required_decel < MPC_DECEL_JERK_MAX_REQUIRED_DECEL)
    smoothing_eligible = (modest_decel and (not self._lead_trend_warmup or history_ready)
      and not tightening_lead and not sustained_tightening)
    if history_ready:
      self._lead_trend_warmup = False
    if previous_mpc_failed or (lead_restriction and not self._jerk_smoothing_blocked
                               and (not modest_decel or tightening_lead or sustained_tightening)):
      self._jerk_smoothing_blocked = True
    elif not lead_restriction:
      self._jerk_smoothing_blocked = False
    return MPC_DECEL_JERK_COST_MULTIPLIER if smoothing_eligible and not self._jerk_smoothing_blocked else 1.0

  def update_should_stop(self, should_stop: bool) -> bool:
    if not self.is_active:
      return should_stop
    if self.pace.departure_launching:
      return False
    return should_stop or self.pace.stop_hold

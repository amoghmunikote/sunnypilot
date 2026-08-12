"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.constants import (
  BRAKING_ACCEL_THRESHOLD, CAP_FILTER_FRAMES, LAUNCH_END_SPEED, LAUNCH_TARGET_HEADROOM, LEAD_MATCH_ACCEL_SLEW,
  LEAD_MATCH_HEADROOM, MATCHED_SPEED_DECEL_RATE, SPEED_DEADBAND, STOP_HOLD_CREEP_DISTANCE, STOP_HOLD_EGO_SPEED,
  STOP_HOLD_EXIT_FRAMES, STOP_HOLD_EXIT_SPEED, STOP_HOLD_MAX_LEAD_DISTANCE, STOP_HOLD_SPEED_FLOOR, TARGET_RELEASE_SLEW,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.lead import LeadPlan


def _median(samples: list[float]) -> float:
  return sorted(samples)[len(samples) // 2]


def _slew(current: float, target: float, rate: float, dt: float) -> float:
  return float(np.clip(target, current - rate * dt, current + rate * dt))


class Pace:
  def __init__(self) -> None:
    self.lead_speed_samples = [math.inf] * CAP_FILTER_FRAMES
    self.lead_accel_samples = [0.0] * CAP_FILTER_FRAMES

    self.cap_trusted = math.inf
    self.relief_streak = 0
    self.relief_hold_elapsed = 0
    self.lead_loss_frames = 0
    self._loss_was_restricting = False

    self.target_speed: float | None = None
    self.e2e_braking_handoff = False
    self.matched_lead = False
    self.matched_accel_limit: float | None = None

    self.stop_hold = False
    self.launching = False
    self.departure_launching = False
    self.confirm_frames = 0
    self.no_departure_lead_frames = 0
    self.motion_ref: float | None = None
    self.last_raw_distance: float | None = None
    self.lead_braking = False
    self._active_frames = 0

    self.restricting = False
    self.releasing = False
    self.has_lead = False
    self.required_decel = 0.0
    self.selected_lead = -1
    self.selected_lead_track_id = -1
    self.raw_cap = math.inf

  @property
  def filtered_lead_speed(self) -> float:
    return _median(self.lead_speed_samples)

  @property
  def filtered_lead_accel(self) -> float:
    return _median(self.lead_accel_samples)

  def reset(self) -> None:
    self.__init__()  # noqa: PLC2801

  def _update_lead_bookkeeping(self, lead_plan: LeadPlan, was_restricting: bool) -> None:
    self.has_lead = lead_plan.selected_lead >= 0
    self.raw_cap = lead_plan.cap if self.has_lead else math.inf
    if self.has_lead:
      self.lead_loss_frames = 0
    else:
      if self.lead_loss_frames == 0:
        self._loss_was_restricting = was_restricting
      self.lead_loss_frames += 1
    self.selected_lead = lead_plan.selected_lead
    self.selected_lead_track_id = lead_plan.selected_lead_track_id if self.has_lead else -1

  def _update_speed_sample(self, lead_plan: LeadPlan) -> None:
    if not self.has_lead:
      self.lead_speed_samples.append(math.inf)
      self.lead_speed_samples.pop(0)
      self.lead_accel_samples.append(0.0)
      self.lead_accel_samples.pop(0)
      return
    if self.raw_cap <= self.cap_trusted + 1e-9:
      self.lead_speed_samples.append(lead_plan.selected_lead_speed)
      self.lead_speed_samples.pop(0)
      self.lead_accel_samples.append(lead_plan.selected_lead_accel)
      self.lead_accel_samples.pop(0)

  def _update_trust_register(self, persist_frames: int, dropout_frames: int, switch_max_frames: int) -> None:
    candidate = self.raw_cap
    if candidate <= self.cap_trusted:
      self.cap_trusted = candidate
      self.relief_streak = 0
      self.relief_hold_elapsed = 0
      return

    if not self.has_lead:
      pre_roll = dropout_frames if self._loss_was_restricting else persist_frames
      if self.lead_loss_frames <= pre_roll:
        return
      self.cap_trusted = candidate
      self.relief_streak = self.relief_hold_elapsed = 0
      return

    if candidate >= self.cap_trusted + SPEED_DEADBAND:
      self.relief_streak += 1
    else:
      self.relief_streak = 0
    self.relief_hold_elapsed += 1
    if self.relief_streak > persist_frames or self.relief_hold_elapsed >= switch_max_frames:
      self.cap_trusted = candidate
      self.relief_streak = self.relief_hold_elapsed = 0

  def _guarded_distance(self, raw: float, lead_speed: float, dt: float) -> float:
    if self.last_raw_distance is not None:
      max_step = max(STOP_HOLD_CREEP_DISTANCE / 2.0, 3.0 * max(lead_speed, 0.0) * dt)
      if abs(raw - self.last_raw_distance) > max_step:
        raw = self.last_raw_distance
    self.last_raw_distance = raw
    return raw

  def _update_stop_hold(self, lead_plan: LeadPlan, v_ego: float, base_speed: float, dt: float) -> bool:
    if self.stop_hold:
      has_departure_lead = lead_plan.departure_lead_index >= 0
      self.no_departure_lead_frames = 0 if has_departure_lead else self.no_departure_lead_frames + 1
      lead_speed = lead_plan.departure_lead_speed if has_departure_lead else 0.0
      evidence = has_departure_lead and lead_speed > STOP_HOLD_SPEED_FLOOR
      distance = None
      if has_departure_lead:
        raw_distance = lead_plan.departure_lead_distances[lead_plan.departure_lead_index]
        distance = self._guarded_distance(raw_distance, lead_speed, dt)

      if evidence:
        if self.confirm_frames == 0:
          self.motion_ref = distance
        self.confirm_frames += 1
      else:
        self.confirm_frames = 0
        self.motion_ref = None

      growth = 0.0
      if self.confirm_frames > 0 and self.motion_ref is not None and distance is not None:
        growth = distance - self.motion_ref

      dwell_ready = self.confirm_frames >= STOP_HOLD_EXIT_FRAMES
      departing_with_lead = has_departure_lead and dwell_ready and growth > 0.0 and (
        growth >= STOP_HOLD_CREEP_DISTANCE or lead_speed > STOP_HOLD_EXIT_SPEED
      )
      departing_no_lead = (not has_departure_lead and self.no_departure_lead_frames >= STOP_HOLD_EXIT_FRAMES
                           and base_speed > STOP_HOLD_SPEED_FLOOR)

      if departing_with_lead or departing_no_lead:
        self.stop_hold = False
        self.launching = True
        self.departure_launching = has_departure_lead
        self.confirm_frames = 0
        self.motion_ref = None
        self.no_departure_lead_frames = 0
        self.target_speed = v_ego
      else:
        self.target_speed = 0.0
      return self.stop_hold

    departure_separation = (lead_plan.departure_lead_separations[lead_plan.departure_lead_index]
                            if lead_plan.departure_lead_index >= 0 else math.inf)
    stopped_lead_hold = lead_plan.has_nearly_stopped_lead and (
      lead_plan.departure_cap < STOP_HOLD_SPEED_FLOOR
      or (self.lead_braking and departure_separation <= STOP_HOLD_MAX_LEAD_DISTANCE)
    )
    if not self.launching and v_ego < STOP_HOLD_EGO_SPEED and (
      stopped_lead_hold or (math.isfinite(self.cap_trusted) and self.cap_trusted < STOP_HOLD_SPEED_FLOOR)
    ):
      self.stop_hold = True
      self.launching = self.departure_launching = False
      self.matched_lead = False
      self.confirm_frames = 0
      self.motion_ref = None
      self.no_departure_lead_frames = 0
      self.target_speed = 0.0
      return True

    return False

  def _update_launch(self, lead_plan: LeadPlan, base_speed: float, v_ego: float, dt: float, persist_frames: int) -> None:
    if not self.launching:
      return
    if v_ego >= LAUNCH_END_SPEED:
      self.launching = self.departure_launching = False
      return
    invalid_lead = lead_plan.lead_status and not self.has_lead
    renewed_stop = self.has_lead and lead_plan.has_nearly_stopped_lead
    if invalid_lead or renewed_stop:
      self.launching = self.departure_launching = False
      if v_ego < STOP_HOLD_EGO_SPEED:
        self.stop_hold = True
        self.confirm_frames = 0
        self.motion_ref = None
        self.no_departure_lead_frames = 0
        self.target_speed = 0.0
      return
    self.releasing = True
    if self.departure_launching:
      self.target_speed = base_speed
    elif not self.has_lead and self.lead_loss_frames >= persist_frames:
      self.target_speed = min(base_speed, self.cap_trusted)
    else:
      launch_target = min(base_speed, v_ego + LAUNCH_TARGET_HEADROOM)
      self.target_speed = min(base_speed, max(self.target_speed or 0.0, launch_target) + TARGET_RELEASE_SLEW * dt)

  def _update_matched(self, ceiling: float, base_speed: float, v_ego: float, profile_max_accel: float, dt: float) -> None:
    if math.isfinite(self.filtered_lead_speed):
      # clamp against base_speed, not ceiling, or this collapses toward zero as v_ego nears cap_trusted
      recovery_speed = min(base_speed, self.filtered_lead_speed + LEAD_MATCH_HEADROOM)
      desired_accel_limit = float(np.clip(recovery_speed - v_ego, 0.0, profile_max_accel))
    else:
      desired_accel_limit = 0.0
    if self.filtered_lead_accel < BRAKING_ACCEL_THRESHOLD:
      # the matched lead is itself braking hard right now - don't let a stale recovery_speed
      # calc (which lags the lead's speed, not its accel) hold the ceiling down; pre-position it
      # at profile_max so a later recovery isn't slew-limited on top of the lead's own braking
      desired_accel_limit = profile_max_accel
    if self.matched_accel_limit is None:
      self.matched_accel_limit = profile_max_accel
    self.matched_accel_limit = _slew(self.matched_accel_limit, desired_accel_limit, LEAD_MATCH_ACCEL_SLEW, dt)

    if ceiling <= self.target_speed - SPEED_DEADBAND:
      self.target_speed = max(ceiling, self.target_speed - MATCHED_SPEED_DECEL_RATE * dt)
      self.restricting = True
    elif ceiling >= self.target_speed + SPEED_DEADBAND:
      self.target_speed = min(ceiling, self.target_speed + profile_max_accel * dt)
      self.releasing = True

  def _update_target_law(self, lead_plan: LeadPlan, base_speed: float, v_ego: float, comfort_decel: float,
                         profile_max_accel: float, dt: float, planner_speed: float,
                         dropout_frames: int, was_restricting: bool) -> None:
    ceiling = min(base_speed, self.cap_trusted)
    newly_matched = self.has_lead and lead_plan.closing_speed <= 0.0
    still_within_dropout = not self.has_lead and self.lead_loss_frames <= dropout_frames
    self.matched_lead = newly_matched or (self.matched_lead and (self.has_lead or still_within_dropout))
    if self.matched_lead:
      self._update_matched(ceiling, base_speed, v_ego, profile_max_accel, dt)
      return

    self.matched_accel_limit = None
    synced_to_planner = ceiling < self.target_speed and planner_speed < self.target_speed
    if synced_to_planner:
      self.target_speed = max(planner_speed, self.target_speed - comfort_decel * dt)

    if ceiling <= self.target_speed - SPEED_DEADBAND or (was_restricting and ceiling < self.target_speed):
      if not synced_to_planner:
        self.target_speed = max(ceiling, self.target_speed - comfort_decel * dt)
      self.restricting = True
    elif ceiling >= self.target_speed + SPEED_DEADBAND:
      self.target_speed = min(ceiling, self.target_speed + TARGET_RELEASE_SLEW * dt)
      self.releasing = True

  def update(self, lead_plan: LeadPlan, base_speed: float, v_ego: float, comfort_decel: float, profile_max_accel: float,
             dt: float, persist_frames: int, dropout_frames: int, switch_max_frames: int, planner_speed: float,
             planner_accel: float, previous_mpc_source, previous_plan_accel: float) -> float:
    was_restricting = self.restricting
    holding_below_cruise = (not self.matched_lead and self.target_speed is not None and math.isfinite(self.cap_trusted)
                            and self.cap_trusted < base_speed - SPEED_DEADBAND
                            and self.cap_trusted - v_ego < LEAD_MATCH_HEADROOM)
    self.restricting = self.releasing = False
    self._update_lead_bookkeeping(lead_plan, was_restricting or holding_below_cruise)
    self._update_trust_register(persist_frames, dropout_frames, switch_max_frames)
    self._update_speed_sample(lead_plan)
    self.required_decel = lead_plan.required_decel

    self._active_frames += 1
    if self._active_frames >= persist_frames and math.isfinite(self.cap_trusted) and self.has_lead and planner_accel <= BRAKING_ACCEL_THRESHOLD:
      self.lead_braking = True
    elif not self.has_lead and self.lead_loss_frames >= persist_frames:
      self.lead_braking = False

    if self.target_speed is None:
      self.target_speed = min(base_speed, v_ego)
      e2e_handoff = previous_mpc_source == LongitudinalPlanSource.e2e
      self.e2e_braking_handoff = e2e_handoff and math.isfinite(previous_plan_accel) and previous_plan_accel <= BRAKING_ACCEL_THRESHOLD
      stop_hold_reason = lead_plan.has_nearly_stopped_lead or (math.isfinite(self.cap_trusted) and self.cap_trusted < STOP_HOLD_SPEED_FLOOR)
      if v_ego < STOP_HOLD_EGO_SPEED and not stop_hold_reason:
        self.target_speed = min(base_speed, v_ego + LAUNCH_TARGET_HEADROOM)
        self.launching = True
        self.departure_launching = False
    elif self.e2e_braking_handoff and planner_accel >= 0.0:
      self.e2e_braking_handoff = False

    self.target_speed = min(self.target_speed, base_speed)

    if self._update_stop_hold(lead_plan, v_ego, base_speed, dt):
      return self.target_speed

    self._update_launch(lead_plan, base_speed, v_ego, dt, persist_frames)
    if self.launching or self.stop_hold:
      return self.target_speed

    self._update_target_law(lead_plan, base_speed, v_ego, comfort_decel, profile_max_accel, dt, planner_speed,
                            dropout_frames, was_restricting)
    return self.target_speed

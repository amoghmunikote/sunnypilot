import math

import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.constants import (
  CAP_FILTER_FRAMES, COMFORT_DECEL, LEAD_DROPOUT_COAST_TIME, LEAD_SWITCH_MAX_HOLD_TIME, PERSIST_TIME,
  STOP_HOLD_EXIT_FRAMES, AccelProfile,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.lead import LeadPlan
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.pace import Pace

DT = DT_MDL
COMFORT_DECEL_NORMAL = COMFORT_DECEL[AccelProfile.normal]
PROFILE_MAX_ACCEL = 1.5


def _frames(seconds: float) -> int:
  import math as _math
  return _math.ceil(seconds / DT)


PERSIST_FRAMES = max(CAP_FILTER_FRAMES, _frames(PERSIST_TIME))
DROPOUT_FRAMES = max(PERSIST_FRAMES, _frames(LEAD_DROPOUT_COAST_TIME))
SWITCH_MAX_FRAMES = max(DROPOUT_FRAMES, _frames(LEAD_SWITCH_MAX_HOLD_TIME))


def _lead_plan(speed: float, distance: float, cap: float = 0.0, closing_speed: float = 0.0,
              required_decel: float = 0.0) -> LeadPlan:
  return LeadPlan(
    cap=cap, selected_lead=0, selected_lead_track_id=1, selected_lead_speed=speed, selected_lead_accel=0.0,
    departure_lead_index=0, departure_lead_speed=speed, departure_cap=cap,
    departure_lead_speeds=(speed, math.inf), departure_lead_distances=(distance, -math.inf),
    departure_lead_track_ids=(1, -1), departure_lead_separations=(distance, -math.inf),
    usable_gap=distance, closing_speed=closing_speed, required_decel=required_decel,
    has_nearly_stopped_lead=speed < 0.15, lead_status=True,
  )


def _no_lead() -> LeadPlan:
  return LeadPlan(lead_status=False)


def _run(pace: Pace, lead_plan: LeadPlan, base_speed: float, v_ego: float, planner_speed: float | None = None,
        planner_accel: float = 0.0, previous_mpc_source=None, previous_plan_accel: float = 0.0) -> float:
  return pace.update(lead_plan, base_speed, v_ego, COMFORT_DECEL_NORMAL, PROFILE_MAX_ACCEL, DT,
                      PERSIST_FRAMES, DROPOUT_FRAMES, SWITCH_MAX_FRAMES, v_ego if planner_speed is None else planner_speed,
                      planner_accel, previous_mpc_source, previous_plan_accel)


def test_route_51d_duplicate_lead_speed_pulse_cannot_release_stop_hold():
  pace = Pace()
  # settle into stop-hold: ego and lead both stationary
  for _ in range(20):
    _run(pace, _lead_plan(0.0, 6.0), base_speed=8.0, v_ego=0.0)
  assert pace.stop_hold

  pulse_speeds = (0.1361, 0.1731, 0.2146, 0.2253, 0.2137, 0.1877)
  pulse_distances = (6.0, 6.0, 6.0, 5.96, 6.04, 6.04)
  for speed, distance in zip(pulse_speeds, pulse_distances, strict=True):
    target = _run(pace, _lead_plan(speed, distance), base_speed=8.0, v_ego=0.0)
    assert pace.stop_hold, "noise pulse must not release stop-hold"
    assert target == 0.0

  # real departure: sustained motion + growing gap
  released_frame = None
  for frame in range(10):
    distance = 6.0 + 2.0 * (frame + 1) * DT
    target = _run(pace, _lead_plan(2.0, distance), base_speed=8.0, v_ego=min(2.0, frame * 0.3))
    if not pace.stop_hold:
      released_frame = frame
      break
  assert released_frame is not None, "real departure must eventually release stop-hold"
  assert released_frame <= STOP_HOLD_EXIT_FRAMES, "must release within the required frame budget"
  assert target > 0.0


def test_route_520_slow_lead_pulse_cannot_release_stop_hold_or_dampen_real_departure():
  pace = Pace()
  for _ in range(20):
    _run(pace, _lead_plan(0.0, 6.0), base_speed=8.0, v_ego=0.0)
  assert pace.stop_hold

  pulse_speeds = (0.01, 0.03, 0.07, 0.10, 0.14, 0.20, 0.26, 0.32, 0.34, 0.33, 0.31, 0.28, 0.24, 0.20, 0.15, 0.09, 0.05, 0.01)
  pulse_offsets = (0.00, 0.00, 0.00, 0.01, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.09, 0.11, 0.12, 0.13, 0.14, 0.15, 0.15, 0.16)
  for speed, offset in zip(pulse_speeds, pulse_offsets, strict=True):
    target = _run(pace, _lead_plan(speed, 6.0 + offset), base_speed=8.0, v_ego=0.0)
    assert pace.stop_hold, "slow ramp pulse must not release stop-hold"
    assert target == 0.0

  released_frame = None
  for frame in range(10):
    distance = 6.16 + 2.0 * (frame + 1) * DT
    target = _run(pace, _lead_plan(2.0, distance), base_speed=8.0, v_ego=min(2.0, frame * 0.3))
    if not pace.stop_hold:
      released_frame = frame
      break
  assert released_frame is not None
  assert released_frame <= STOP_HOLD_EXIT_FRAMES
  assert target > 0.0


def test_trust_register_accepts_tightening_immediately():
  pace = Pace()
  _run(pace, _lead_plan(20.0, 100.0, cap=25.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(25.0)
  # a single frame with a much lower (tightening) cap must be trusted at once, no persistence
  # and no median-filter warm-up delay - this is the fast-reaction side of the trust register
  _run(pace, _lead_plan(15.0, 40.0, cap=10.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(10.0)


def test_trust_register_requires_persistence_for_relief():
  pace = Pace()
  for _ in range(CAP_FILTER_FRAMES + 2):
    _run(pace, _lead_plan(20.0, 100.0, cap=20.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(20.0)

  # a brief (shorter than PERSIST_FRAMES) relief spike must not be accepted
  glitch_len = PERSIST_FRAMES - 2
  for _ in range(glitch_len):
    _run(pace, _lead_plan(20.0, 100.0, cap=28.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(20.0), "brief relief spike must not be trusted"

  # drop back to the original (tightening relative to the glitch, but not relative to trusted) value
  for _ in range(CAP_FILTER_FRAMES + 2):
    _run(pace, _lead_plan(20.0, 100.0, cap=20.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(20.0)

  # a SUSTAINED relief (longer than PERSIST_FRAMES) must eventually be trusted
  for _ in range(PERSIST_FRAMES + CAP_FILTER_FRAMES + 2):
    _run(pace, _lead_plan(20.0, 100.0, cap=28.0), base_speed=30.0, v_ego=20.0)
  assert pace.cap_trusted == pytest.approx(28.0), "sustained relief must eventually be trusted"


def test_trust_register_dropout_coasts_before_release():
  pace = Pace()
  for _ in range(CAP_FILTER_FRAMES + 2):
    _run(pace, _lead_plan(6.0, 50.0, cap=6.0), base_speed=18.0, v_ego=10.0)
  assert pace.cap_trusted == pytest.approx(6.0)

  # within the dropout coast window, cap_trusted must stay frozen (no snap to clear-road)
  for _ in range(DROPOUT_FRAMES - 1):
    _run(pace, _no_lead(), base_speed=18.0, v_ego=10.0)
  assert pace.cap_trusted == pytest.approx(6.0), "must coast, not snap, before the dropout window elapses"

  # once the coast window elapses, relief is accepted without further persistence delay
  for _ in range(CAP_FILTER_FRAMES + 2):
    _run(pace, _no_lead(), base_speed=18.0, v_ego=10.0)
  assert math.isinf(pace.cap_trusted), "must release promptly once the dropout window has elapsed"


def test_target_law_relief_bounded_by_release_slew():
  from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.constants import TARGET_RELEASE_SLEW

  pace = Pace()
  for _ in range(CAP_FILTER_FRAMES + 2):
    _run(pace, _lead_plan(20.0, 60.0, cap=22.0), base_speed=30.0, v_ego=22.0)
  before = pace.target_speed

  # sustained relief, once trusted, must still only move target_speed at TARGET_RELEASE_SLEW*dt per frame
  target = _run(pace, _lead_plan(28.0, 200.0, cap=30.0), base_speed=30.0, v_ego=22.0)
  assert target <= before + TARGET_RELEASE_SLEW * DT + 1e-9

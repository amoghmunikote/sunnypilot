import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  STOP_DISTANCE, T_IDXS, LongitudinalMpc, LongitudinalPlanSource, get_T_FOLLOW,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.accel_controller import AccelController, AccelControllerState
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.constants import (
  ACCEL_LIMIT_HORIZON_JERK, ACCEL_PROFILE_MAX_BP, ACCEL_PROFILE_MAX_V, ACCEL_PROFILES, CAP_FILTER_FRAMES, COMFORT_DECEL,
  LAUNCH_END_SPEED, LAUNCH_TARGET_HEADROOM, LEAD_MATCH_ACCEL_SLEW, LEAD_MATCH_HEADROOM, MPC_DECEL_JERK_COST_MULTIPLIER,
  STOP_GAP_RESERVE, STOP_HOLD_EXIT_FRAMES, TARGET_RELEASE_SLEW, AccelProfile, profile_accel_max, sanitize_profile,
)
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.helpers import build_accel_ceiling
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller.lead import _project_ego, calculate_lead_plan

# Rewrite of the pre-rewrite test_accel_controller.py (106 tests against the deleted TargetState/
# DepartureTracker FSM, git rev 7d08d3bcd1). Behavioral intent is preserved against the new Pace/
# AccelController surface; tests whose entire premise was a mechanism deliberately cut in the rewrite
# are skipped rather than force-fit. Cut mechanisms, gathered in one place instead of repeated per test:
#  - CAP_FILTER_FRAMES median pre-filter on cap (old TargetState.filtered_cap) - the trust register
#    (cap_trusted) has no median prefilter by design, see Pace's own docstring.
#  - lead_switch_guard_frames / release_slew_armed / release_settle_frames - replaced by the trust
#    register's persist_frames/dropout_frames/switch_max_frames, already unit-tested directly against
#    Pace in tests/test_pace.py (trust register + dropout-coast + release-slew-bound tests).
#  - TARGET_SPEED_RESERVE / speed_reserve_armed - the "extra step held back before touching cap" is gone.
#  - lead_braking / previous_should_stop - update() no longer takes previous_should_stop at all; stop-hold
#    entry is now purely reactive to the current frame's lead geometry, not a carried-over planner verdict.
#  - filtered_lead_accel / LEAD_BRAKING_ACCEL_THRESHOLD - the matched-lead accel limit no longer looks at
#    the lead's own acceleration signal, only its (median-filtered) speed.
#  - per-track-id DepartureTracker bookkeeping (two-slot samples/references/track_ids, recent_motion() with
#    DEPARTURE_MOTION_NOISE_FLOOR/STEP_MIN) - collapsed into one scalar last_raw_distance/motion_ref that
#    only guards against implausible single-frame jumps, independent of track identity.
# test_far_stopped_lead_should_not_create_stop_hold and test_fast_speed_glitch_without_distance_progress_
# should_stay_in_stop_hold below were originally xfail (real behavioral regressions from the rewrite, not
# test bugs) - both are now fixed (lead_braking + STOP_HOLD_MAX_LEAD_DISTANCE port; growth > 0.0 floor on
# the fast-lane) and pass unconditionally.


def make_lead(*, status=False, d_rel=0.0, v_lead_k=0.0, a_lead_k=0.0, a_lead_tau=1.5, radar_track_id=-1):
  return SimpleNamespace(present=status, dRel=d_rel, vLeadK=v_lead_k, aLeadK=a_lead_k, aLeadTau=a_lead_tau,
                         radarTrackId=radar_track_id)


def make_radar(lead_one=None, lead_two=None):
  return SimpleNamespace(leadOne=lead_one or make_lead(), leadTwo=lead_two or make_lead())


def make_controller(delay=0.10):
  return AccelController(SimpleNamespace(longitudinalActuatorDelay=delay, openpilotLongitudinalControl=True))


def get_lead_plan(controller, radar_state, v_ego: float, a_ego: float, profile: int):
  return calculate_lead_plan(radar_state, v_ego, a_ego, controller.delay, profile)


def update(controller, radar_state=None, **overrides):
  args = {
    "base_speed": 25.0,
    "v_ego": 10.0,
    "a_ego": 0.0,
    "profile": AccelProfile.normal,
    "follow_personality": log.LongitudinalPersonality.standard,
    "enabled": True,
    "acc_selected": True,
    "engaged": True,
    "cruise_initialized": True,
    "stock_accel_max": ACCEL_MAX,
  }
  args.update(overrides)
  controller.profile = args.pop("profile")
  controller.enabled = args.pop("enabled")
  controller.update(radar_state or make_radar(), **args)
  pace = controller.pace
  return SimpleNamespace(
    target_speed=controller.output_v_target, active=controller.is_active, launching=pace.launching,
    departure_launching=pace.departure_launching, mpc_accel_max=controller.mpc_accel_max,
    cruise_accel_max=controller.cruise_accel_max, state=controller.state, selected_lead=controller.selected_lead,
    required_decel=controller.required_decel, stop_hold=pace.stop_hold, matched_lead=pace.matched_lead,
    matched_accel_limit=pace.matched_accel_limit, cap_trusted=pace.cap_trusted, restricting=pace.restricting,
    releasing=pace.releasing,
  )


def effective_accel_max(result):
  return math.inf if result.mpc_accel_max is None else min(result.mpc_accel_max)


def restrictive_radar():
  return make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0, a_lead_k=-0.5))


def enter_stop_hold(controller, *, base_speed=8.0, v_ego=0.0, frames=6):
  stopped = make_radar(make_lead(status=True, d_rel=6.0, v_lead_k=0.0))
  result = None
  for _ in range(frames):
    result = update(controller, stopped, base_speed=base_speed, v_ego=v_ego)
  assert result.stop_hold
  return result


class TestProfiles:
  def test_lookup_table_is_explicit_and_tunable(self):
    assert ACCEL_PROFILE_MAX_BP == [0.0, 3.0, 10.0, 25.0, 40.0]
    assert ACCEL_PROFILE_MAX_V == {
      AccelProfile.eco: [1.67, 1.30, 0.72, 0.32, 0.24],
      AccelProfile.normal: [1.80, 1.51, 0.98, 0.53, 0.35],
      AccelProfile.sport: [2.00, 1.91, 1.16, 0.73, 0.47],
    }

  @pytest.mark.parametrize("profile", ACCEL_PROFILES)
  def test_lookup_interpolates_and_stays_inside_global_limit(self, profile):
    for speed, expected in zip(ACCEL_PROFILE_MAX_BP, ACCEL_PROFILE_MAX_V[profile], strict=True):
      assert profile_accel_max(profile, speed) == expected

    limits = [profile_accel_max(profile, speed) for speed in np.linspace(-1.0, 50.0, 201)]
    assert all(0.0 <= limit <= ACCEL_MAX for limit in limits)
    assert np.all(np.diff(limits) <= 0.0)

  @pytest.mark.parametrize("speed", ACCEL_PROFILE_MAX_BP)
  def test_profile_order_is_distinct(self, speed):
    eco, normal, sport = (profile_accel_max(profile, speed) for profile in ACCEL_PROFILES)
    assert eco < normal < sport

  def test_invalid_profile_defaults_to_normal(self):
    assert sanitize_profile(999) == AccelProfile.normal

  def test_stock_limit_intersects_profile_before_mpc(self):
    controller = make_controller()
    results = [update(controller, v_ego=10.0, profile=AccelProfile.sport, stock_accel_max=0.30)
               for _ in range(controller.persist_frames)]
    result = results[-1]
    assert profile_accel_max(AccelProfile.sport, 10.0) > 0.30
    assert effective_accel_max(result) == pytest.approx(0.30)
    assert all(sample.mpc_accel_max is not None for sample in results)
    assert all(max(sample.mpc_accel_max) <= 0.30 + 1e-9 for sample in results)

  def test_runtime_profile_switch_applies_the_lookup_value_directly(self):
    controller = make_controller()
    sport = [update(controller, v_ego=10.0, profile=AccelProfile.sport, stock_accel_max=1.20)
             for _ in range(controller.persist_frames)][-1]
    eco = update(controller, v_ego=10.0, profile=AccelProfile.eco, stock_accel_max=1.20)

    assert effective_accel_max(sport) == pytest.approx(profile_accel_max(AccelProfile.sport, 10.0))
    assert effective_accel_max(eco) == pytest.approx(profile_accel_max(AccelProfile.eco, 10.0))

  def test_stock_limit_reduction_applies_immediately(self):
    controller = make_controller()
    for _ in range(controller.persist_frames):
      update(controller, v_ego=10.0, profile=AccelProfile.sport, stock_accel_max=1.20)

    reduced = update(controller, v_ego=10.0, profile=AccelProfile.sport, stock_accel_max=0.30)
    assert effective_accel_max(reduced) == pytest.approx(0.30)
    assert reduced.mpc_accel_max is not None
    assert max(reduced.mpc_accel_max) <= 0.30 + 1e-9

  def test_one_frame_stock_zero_does_not_poison_profile_recovery(self):
    clean_controller, glitch_controller = make_controller(), make_controller()
    for _ in range(clean_controller.persist_frames + 10):
      clean = update(clean_controller, v_ego=10.0, stock_accel_max=1.5)
      update(glitch_controller, v_ego=10.0, stock_accel_max=1.5)

    limited = update(glitch_controller, v_ego=10.0, stock_accel_max=0.0)
    clean = update(clean_controller, v_ego=10.0, stock_accel_max=1.5)
    recovered = update(glitch_controller, v_ego=10.0, stock_accel_max=1.5)

    assert effective_accel_max(limited) == 0.0
    assert effective_accel_max(recovered) == pytest.approx(effective_accel_max(clean))

  @pytest.mark.parametrize("radar_fresh", (True, False), ids=("dropout", "stale"))
  def test_matched_lead_ceiling_obeys_current_stock_limit(self, radar_fresh):
    controller = make_controller()
    radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0))
    for _ in range(controller.persist_frames + 10):
      update(controller, radar, v_ego=10.0, planner_accel=-0.2)
    for _ in range(20):
      update(controller, radar, v_ego=8.0, planner_accel=-0.2)
    assert controller.pace.matched_lead

    limited = update(controller, stock_accel_max=0.0, radar_fresh=radar_fresh)
    assert effective_accel_max(limited) == 0.0
    assert limited.mpc_accel_max is not None
    assert max(limited.mpc_accel_max) == 0.0

  def test_exact_global_max_uses_stock_ceiling(self):
    result = update(make_controller(), base_speed=8.0, v_ego=0.0, profile=AccelProfile.sport)
    assert profile_accel_max(AccelProfile.sport, 0.0) == ACCEL_MAX
    assert result.mpc_accel_max is None


class TestBuildAccelCeiling:
  @pytest.mark.parametrize("planner_accel", (-1.0, 0.0, 1.2, ACCEL_MAX))
  def test_ceiling_is_finite_feasible_and_jerk_bounded(self, planner_accel):
    limit = 0.50
    ceiling = np.asarray(build_accel_ceiling(limit, planner_accel))
    a0 = float(np.clip(planner_accel, ACCEL_MIN, ACCEL_MAX))

    assert ceiling.shape == T_IDXS.shape
    assert np.all(np.isfinite(ceiling))
    assert np.all((0.0 <= ceiling) & (ceiling <= ACCEL_MAX))
    assert ceiling[0] + 1e-9 >= a0
    assert np.all(ceiling + 1e-9 >= limit)
    assert np.all(np.diff(ceiling) <= 1e-9)
    assert np.all(-np.diff(ceiling) <= ACCEL_LIMIT_HORIZON_JERK * np.diff(T_IDXS) + 1e-9)

  def test_zero_limit_remains_feasible_for_positive_x0(self):
    ceiling = np.asarray(build_accel_ceiling(0.0, 0.8))
    assert ceiling[0] == pytest.approx(0.8)
    assert ceiling[-1] == pytest.approx(0.0)
    assert np.all(ceiling >= 0.0)


class TestMpcCeilingIntegration:
  def test_inactive_controller_has_no_custom_ceiling(self):
    controller = make_controller()
    result = update(controller, enabled=False)
    assert not result.active
    assert result.mpc_accel_max is None
    assert math.isinf(effective_accel_max(result))
    assert controller.pace.target_speed is None

  def test_closing_on_a_lead_has_no_ceiling_regardless_of_planner_accel_sign(self):
    # while ego is still approaching (not yet matched/departing) a lead, the profile ceiling
    # deliberately does not apply - the MPC's own obstacle cost must be free to do the early,
    # anticipatory slowdown unfought, regardless of which way planner_accel currently points
    controller = make_controller()
    radar = restrictive_radar()
    for planner_accel in (-0.2, 0.2, -0.2):
      result = update(controller, radar, v_ego=10.0, planner_accel=planner_accel)
      assert result.mpc_accel_max is None
    assert not controller.pace.matched_lead

    bypassed = update(controller, radar, planner_accel=-0.2, acc_selected=False)
    assert not bypassed.active and bypassed.mpc_accel_max is None

  def test_eco_cruise_limit_remains_active_while_closing_on_a_lead(self):
    controller = make_controller()
    radar = make_radar(make_lead(status=True, d_rel=80.0, v_lead_k=19.25))
    result = update(controller, radar, base_speed=20.0, v_ego=20.0, profile=AccelProfile.eco, planner_accel=0.16,
                    previous_mpc_source=LongitudinalPlanSource.cruise)

    assert result.state == AccelControllerState.free
    assert result.mpc_accel_max is None
    assert result.cruise_accel_max == pytest.approx(profile_accel_max(AccelProfile.eco, 20.0))

  def test_lead_cruise_limit_does_not_follow_mpc_source_or_accel_sign(self):
    controller = make_controller()
    expected = profile_accel_max(AccelProfile.eco, 20.0)
    inputs = (
      (LongitudinalPlanSource.cruise, 0.02, 19.9, 100),
      (LongitudinalPlanSource.lead0, -0.02, 20.1, -1),
      (LongitudinalPlanSource.cruise, -0.10, 19.9, 101),
      (LongitudinalPlanSource.lead0, -0.12, 20.1, -1),
      (LongitudinalPlanSource.lead1, 0.02, 20.1, -1),
    )

    for source, planner_accel, lead_speed, track_id in inputs:
      radar = make_radar(make_lead(status=True, d_rel=150.0, v_lead_k=lead_speed, radar_track_id=track_id))
      result = update(controller, radar, base_speed=20.0, v_ego=20.0, profile=AccelProfile.eco,
                      planner_accel=planner_accel, previous_mpc_source=source)
      assert result.cruise_accel_max == pytest.approx(expected)

  def test_profile_ceiling_stays_continuous_while_a_lead_begins_pulling_away(self):
    controller = make_controller()
    for _ in range(controller.persist_frames):
      update(controller, restrictive_radar(), v_ego=10.0, planner_accel=-0.2)

    pulling_away = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=12.0))
    result = update(controller, pulling_away, v_ego=10.0, planner_accel=0.2)

    assert result.state == AccelControllerState.restrict
    assert result.matched_lead
    assert result.mpc_accel_max is not None
    # the transition into "matched" seeds matched_accel_limit at the plain profile ceiling, then
    # takes exactly one LEAD_MATCH_ACCEL_SLEW*dt step toward the (still median-filtered) recovery
    # limit - a big relief in the raw lead speed must not appear as a discontinuous ceiling jump
    profile_ceiling = profile_accel_max(AccelProfile.normal, 10.0)
    assert profile_ceiling - LEAD_MATCH_ACCEL_SLEW * DT_MDL - 1e-9 <= effective_accel_max(result) <= profile_ceiling + 1e-9


class TestLead:
  def test_cap_matches_stopping_energy_formula_with_flat_reserve(self):
    controller = make_controller()
    lead = make_lead(status=True, d_rel=50.0, v_lead_k=8.0)
    result = get_lead_plan(controller, make_radar(lead), 10.0, 0.0, AccelProfile.normal)
    delay = controller.delay
    lead_xv = LongitudinalMpc.extrapolate_lead(lead.dRel, lead.vLeadK, lead.aLeadK, lead.aLeadTau)
    x_lead = float(np.interp(delay, T_IDXS, lead_xv[:, 0]))
    v_lead = float(np.interp(delay, T_IDXS, lead_xv[:, 1]))
    x_ego, _ = _project_ego(10.0, 0.0, delay)
    safety_gap = max(x_lead - x_ego - STOP_DISTANCE - get_T_FOLLOW(log.LongitudinalPersonality.standard) * v_lead, 0.0)
    usable_gap = max(safety_gap - STOP_GAP_RESERVE, 0.0)
    expected = v_lead + math.sqrt(2.0 * COMFORT_DECEL[AccelProfile.normal] * usable_gap)

    assert result.cap == pytest.approx(expected)
    assert safety_gap - result.usable_gap == pytest.approx(STOP_GAP_RESERVE)

  def test_profile_order_controls_approach_timing(self):
    radar = make_radar(make_lead(status=True, d_rel=50.0, v_lead_k=8.0))
    caps = [get_lead_plan(make_controller(), radar, 10.0, 0.0, profile).cap for profile in ACCEL_PROFILES]
    assert caps[0] < caps[1] < caps[2]

  @pytest.mark.parametrize("v_lead_k", (0.0, 8.0), ids=("stopped", "moving"))
  def test_reserve_is_flat_not_speed_or_decel_scaled(self, v_lead_k):
    # the old reserve tapered with lead speed/required_decel via STOP_GAP_RESERVE_LEAD_SPEED and
    # STOP_GAP_RESERVE_DECEL_BP (both gone) so it "only reduced the gap for stopped leads"; the
    # rewrite always subtracts the same flat STOP_GAP_RESERVE, regardless of how fast the lead goes
    lead = get_lead_plan(make_controller(), make_radar(make_lead(status=True, d_rel=60.0, v_lead_k=v_lead_k)),
                         5.0, 0.0, AccelProfile.normal)
    comfort_decel = COMFORT_DECEL[AccelProfile.normal]
    safety_gap = (lead.departure_cap - lead.departure_lead_speed) ** 2 / (2.0 * comfort_decel)
    assert safety_gap - lead.usable_gap == pytest.approx(STOP_GAP_RESERVE)
    assert lead.departure_cap > lead.cap

  def test_more_restrictive_lead_is_selected(self):
    radar = make_radar(make_lead(status=True, d_rel=70.0, v_lead_k=12.0), make_lead(status=True, d_rel=25.0, v_lead_k=8.0))
    assert get_lead_plan(make_controller(), radar, 10.0, 0.0, AccelProfile.normal).selected_lead == 1

  def test_departure_lead_index_prefers_nearer_lead_over_the_cap_governing_lead(self):
    radar = make_radar(make_lead(status=True, d_rel=3.0, v_lead_k=0.2, radar_track_id=100),
                       make_lead(status=True, d_rel=6.0, v_lead_k=0.1, radar_track_id=200))
    result = get_lead_plan(make_controller(), radar, 0.0, 0.0, AccelProfile.normal)
    assert result.selected_lead == 1
    assert result.departure_lead_index == 0

  @pytest.mark.parametrize("field,value", [
    ("aLeadK", math.nan), ("aLeadK", math.inf), ("aLeadTau", math.nan), ("aLeadTau", -1.0), ("radarTrackId", math.nan),
  ])
  def test_nonessential_invalid_lead_fields_are_sanitized(self, field, value):
    lead = make_lead(status=True, d_rel=30.0, v_lead_k=8.0)
    setattr(lead, field, value)
    result = get_lead_plan(make_controller(), make_radar(lead), 10.0, 0.0, AccelProfile.normal)
    assert result.selected_lead == 0
    assert math.isfinite(result.cap)

  @pytest.mark.parametrize("field,value", [("dRel", math.nan), ("dRel", -1.0), ("vLeadK", math.nan), ("vLeadK", -2.0)])
  def test_invalid_geometry_is_not_used(self, field, value):
    lead = make_lead(status=True, d_rel=30.0, v_lead_k=8.0)
    setattr(lead, field, value)
    result = get_lead_plan(make_controller(), make_radar(lead), 10.0, 0.0, AccelProfile.normal)
    assert result.selected_lead == -1
    assert result.lead_status
    assert math.isinf(result.cap)

  def test_raw_radar_is_never_mutated(self):
    lead = make_lead(status=True, d_rel=30.0, v_lead_k=8.0, a_lead_k=-15.0, a_lead_tau=math.nan)
    before = vars(lead).copy()
    get_lead_plan(make_controller(), make_radar(lead), 10.0, 0.0, AccelProfile.normal)
    assert vars(lead) == before


class TestTargetLawAndTrustRegister:
  def test_matched_lead_accel_limit_ignores_a_two_frame_speed_jump(self):
    clean_controller, noisy_controller = make_controller(), make_controller()
    radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0))
    for controller in (clean_controller, noisy_controller):
      for _ in range(controller.persist_frames + 10):
        update(controller, radar, v_ego=10.0, planner_accel=-0.2)
      for _ in range(20):
        update(controller, radar, v_ego=8.0, planner_accel=-0.2)

    speed_jump = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=16.0))
    for _ in range(2):
      clean = update(clean_controller, radar, v_ego=8.0)
      noisy = update(noisy_controller, speed_jump, v_ego=8.0)
      assert effective_accel_max(noisy) == pytest.approx(effective_accel_max(clean))
      assert noisy.target_speed == pytest.approx(clean.target_speed)
    # NOTE: the old acceleration-jump counterpart (filtered_lead_accel / LEAD_BRAKING_ACCEL_THRESHOLD)
    # has no equivalent anymore - the matched-lead accel limit no longer reads the lead's own aLeadK.

  def test_restriction_target_speed_decays_at_bounded_comfort_rate_without_median_warmup(self):
    # cap_trusted has no median pre-filter (by design, see Pace's docstring), so the very first
    # restrictive frame already reflects the tightened cap; only target_speed's own comfort_decel
    # rate limit bounds how fast it can follow that cap down from there
    controller = make_controller()
    targets = [update(controller, restrictive_radar()).target_speed for _ in range(15)]
    max_step = COMFORT_DECEL[AccelProfile.normal] * DT_MDL

    steps = -np.diff(targets[1:])
    assert np.all(steps <= max_step + 1e-9)
    assert controller.state == AccelControllerState.restrict
    assert targets[-1] < targets[1]

  @pytest.mark.parametrize("previous_mpc_source", (None, LongitudinalPlanSource.cruise, LongitudinalPlanSource.lead0))
  def test_target_speed_syncs_down_to_planner_speed_regardless_of_previous_mpc_source(self, previous_mpc_source):
    # the old code only synced down when is_lead_source(previous_mpc_source); the rewrite dropped
    # that gate entirely - sync-down now runs whenever the ceiling has already caught the target and
    # planner_speed is lower still, independent of where the plan came from
    controller = make_controller()
    for _ in range(15):
      restricted = update(controller, restrictive_radar())
    planner_speed = restricted.target_speed - 2.0

    synced = update(controller, restrictive_radar(), previous_mpc_source=previous_mpc_source, planner_speed=planner_speed,
                    planner_accel=-0.2)
    assert synced.target_speed == pytest.approx(restricted.target_speed - COMFORT_DECEL[AccelProfile.normal] * DT_MDL)

  def test_short_dropout_holds_then_releases_at_a_bounded_rate(self):
    controller = make_controller()
    for _ in range(15):
      restricted = update(controller, restrictive_radar())

    held = [update(controller) for _ in range(controller.dropout_frames - 1)]
    assert all(result.target_speed <= restricted.target_speed + 1e-9 for result in held)

    released = [update(controller) for _ in range(60)]
    targets = [restricted.target_speed, *(result.target_speed for result in released)]
    assert np.max(np.diff(targets)) <= TARGET_RELEASE_SLEW * DT_MDL + 1e-9
    # once within SPEED_DEADBAND of base_speed the relief branch's own deadband gate stops firing,
    # so the release parks just short of an exact 25.0 by design - "free" uses a wider deadband window
    assert released[-1].target_speed >= 25.0 - 0.15 - 1e-9
    assert released[-1].state == AccelControllerState.free


class TestMatchedLead:
  def test_matched_accel_limit_unthrottled_when_ego_well_below_lead_speed(self):
    # matched_lead now just means "not currently closing" (has_lead and closing_speed<=0), not
    # "literally caught the lead" - the throttling itself lives entirely in matched_accel_limit,
    # which must stay at the plain profile ceiling while the recovery-speed headroom is still huge
    controller = make_controller()
    radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0))
    for _ in range(20):
      update(controller, radar, v_ego=10.0, planner_accel=-0.2)
    result = update(controller, radar, v_ego=3.0, planner_accel=-0.2)

    assert result.matched_lead
    assert result.matched_accel_limit == pytest.approx(profile_accel_max(AccelProfile.normal, 3.0))
    assert effective_accel_max(result) == pytest.approx(profile_accel_max(AccelProfile.normal, 3.0))

  def test_matched_accel_limit_throttles_toward_recovery_headroom_when_near_lead_speed(self):
    controller = make_controller()
    slow_radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=3.0))
    for _ in range(20):
      update(controller, slow_radar, v_ego=5.0, planner_accel=-0.2)
    for _ in range(40):
      result = update(controller, slow_radar, v_ego=3.0, planner_accel=-0.2)

    assert result.matched_lead
    assert result.matched_accel_limit == pytest.approx(LEAD_MATCH_HEADROOM)
    assert result.matched_accel_limit < profile_accel_max(AccelProfile.normal, 3.0)

  def test_matched_accel_limit_slew_bounded_and_independent_of_planner_accel_sign(self):
    braking_controller, accelerating_controller = make_controller(), make_controller()
    radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0))
    for controller in (braking_controller, accelerating_controller):
      for _ in range(20):
        update(controller, radar, v_ego=10.0, planner_accel=-0.2)
      for _ in range(20):
        update(controller, radar, v_ego=8.0, planner_accel=-0.2)
    before = braking_controller.pace.matched_accel_limit
    assert accelerating_controller.pace.matched_accel_limit == pytest.approx(before)

    braking = update(braking_controller, radar, v_ego=8.0, planner_accel=-0.2)
    accelerating = update(accelerating_controller, radar, v_ego=8.0, planner_accel=0.2)

    assert abs(braking.matched_accel_limit - before) <= LEAD_MATCH_ACCEL_SLEW * DT_MDL + 1e-9
    assert accelerating.matched_accel_limit == pytest.approx(braking.matched_accel_limit)


class TestLaunchAndDeparture:
  def test_clear_road_launch_has_immediate_headroom_and_bounded_target_slew(self):
    controller = make_controller()
    initial = update(controller, base_speed=12.0, v_ego=0.0, profile=AccelProfile.normal)
    rolling = update(controller, base_speed=12.0, v_ego=0.31, profile=AccelProfile.normal)

    assert initial.active and initial.launching
    assert LAUNCH_TARGET_HEADROOM <= initial.target_speed <= LAUNCH_TARGET_HEADROOM + TARGET_RELEASE_SLEW * DT_MDL
    assert rolling.launching
    assert rolling.target_speed >= 0.31 + LAUNCH_TARGET_HEADROOM
    assert rolling.target_speed - max(initial.target_speed, 0.31 + LAUNCH_TARGET_HEADROOM) <= TARGET_RELEASE_SLEW * DT_MDL + 1e-9

    finished = update(controller, base_speed=12.0, v_ego=LAUNCH_END_SPEED, profile=AccelProfile.normal)
    assert not finished.launching

  def test_e2e_braking_handoff_arms_only_on_seed_frame_from_previous_plan_accel(self):
    # arming is keyed off previous_plan_accel (the MPC's last solved accel), not planner_accel
    # (this frame's plan), and only on the very frame that seeds target_speed for the first time
    armed = make_controller()
    update(armed, base_speed=20.0, v_ego=15.0, previous_mpc_source=LongitudinalPlanSource.e2e,
          previous_plan_accel=-1.0, planner_accel=0.5)
    assert armed.pace.e2e_braking_handoff

    not_armed = make_controller()
    update(not_armed, base_speed=20.0, v_ego=15.0, previous_mpc_source=LongitudinalPlanSource.e2e,
          previous_plan_accel=0.5, planner_accel=-1.0)
    assert not not_armed.pace.e2e_braking_handoff

  def test_stop_hold_needs_four_confirmed_departure_frames_with_real_radar(self):
    controller = make_controller()
    held = enter_stop_hold(controller)
    assert held.stop_hold and held.target_speed == 0.0 and held.mpc_accel_max is None

    results = [update(controller, make_radar(make_lead(status=True, d_rel=6.0 + (frame + 1) * 0.1, v_lead_k=2.0)),
                      base_speed=8.0, v_ego=0.1) for frame in range(STOP_HOLD_EXIT_FRAMES + 4)]
    launch_index = next(index for index, result in enumerate(results) if result.launching)

    assert launch_index == STOP_HOLD_EXIT_FRAMES - 1
    assert all(result.stop_hold and not result.launching for result in results[:launch_index])
    assert results[launch_index].departure_launching
    assert results[launch_index].target_speed == 8.0

  @pytest.mark.parametrize("replacement_track_id", (100, 200), ids=("same-track", "replaced-track"))
  def test_stop_hold_rejects_implausible_same_frame_distance_step(self, replacement_track_id):
    # the per-track DepartureTracker (reseed-on-identity-change) is gone; the single scalar guard in
    # Pace only looks at the size of the jump, not whether the track identity changed underneath it -
    # a static, non-growing step reads as "no real motion" either way and never departs
    controller = make_controller()
    original = make_radar(make_lead(status=True, d_rel=6.0, v_lead_k=0.0, radar_track_id=100))
    update(controller, original, base_speed=8.0, v_ego=0.0)
    stepped = make_radar(make_lead(status=True, d_rel=6.4, v_lead_k=0.2, radar_track_id=replacement_track_id))
    results = [update(controller, stepped, base_speed=8.0, v_ego=0.0) for _ in range(STOP_HOLD_EXIT_FRAMES + 6)]

    assert all(result.stop_hold and result.target_speed == 0.0 and not result.launching for result in results)

  def test_renewed_stop_mid_launch_aborts_back_to_stop_hold(self):
    controller = make_controller()
    enter_stop_hold(controller)
    for frame in range(STOP_HOLD_EXIT_FRAMES + 2):
      moving = make_radar(make_lead(status=True, d_rel=6.0 + (frame + 1) * 0.1, v_lead_k=2.0))
      update(controller, moving, base_speed=8.0, v_ego=0.1)
    assert controller.pace.launching and controller.pace.departure_launching

    renewed_stop_lead = make_radar(make_lead(status=True, d_rel=6.5, v_lead_k=0.05))
    result = update(controller, renewed_stop_lead, base_speed=8.0, v_ego=0.1)
    assert result.stop_hold
    assert result.target_speed == 0.0
    assert not result.launching

  def test_invalid_lead_mid_launch_aborts_launch_without_reentering_stop_hold(self):
    controller = make_controller()
    enter_stop_hold(controller)
    for frame in range(STOP_HOLD_EXIT_FRAMES + 2):
      moving = make_radar(make_lead(status=True, d_rel=6.0 + (frame + 1) * 0.1, v_lead_k=2.0))
      update(controller, moving, base_speed=8.0, v_ego=0.5)
    assert controller.pace.launching

    invalid = make_radar(make_lead(status=True, d_rel=math.nan, v_lead_k=2.0))
    result = update(controller, invalid, base_speed=8.0, v_ego=0.5)
    assert not result.stop_hold
    assert not result.launching
    assert result.active

  def test_genuine_departure_survives_lead_slot_and_track_flicker(self):
    controller = make_controller()
    enter_stop_hold(controller)
    results = []
    for frame in range(STOP_HOLD_EXIT_FRAMES + 4):
      moving = make_lead(status=True, d_rel=6.0 + (frame + 1) * 0.1, v_lead_k=2.0, radar_track_id=100)
      secondary = make_lead(status=True, d_rel=7.0, v_lead_k=2.0, radar_track_id=200)
      radar = make_radar(moving, secondary) if frame % 2 == 0 else make_radar(secondary, moving)
      results.append(update(controller, radar, base_speed=8.0, v_ego=0.0))

    launch_index = next(index for index, result in enumerate(results) if result.launching)
    assert launch_index == STOP_HOLD_EXIT_FRAMES - 1
    assert all(result.stop_hold for result in results[:launch_index])
    assert results[-1].launching and results[-1].departure_launching

  def test_confirmed_creep_departure_departs_within_budget(self):
    controller = make_controller()
    enter_stop_hold(controller)
    results = []
    for frame in range(60):
      creeping = make_radar(make_lead(status=True, d_rel=6.0 + frame * 0.01, v_lead_k=0.2))
      results.append(update(controller, creeping, base_speed=8.0, v_ego=0.0))

    launch_index = next(index for index, result in enumerate(results) if result.launching)
    assert launch_index * DT_MDL <= 2.0
    assert all(not result.stop_hold for result in results[launch_index:])

  def test_departure_dropout_holds_without_resurrecting_stop_hold(self):
    controller = make_controller()
    enter_stop_hold(controller)
    for frame in range(STOP_HOLD_EXIT_FRAMES + 2):
      moving = make_radar(make_lead(status=True, d_rel=6.0 + (frame + 1) * 0.1, v_lead_k=2.0))
      update(controller, moving, base_speed=8.0, v_ego=0.1)
    assert controller.pace.launching

    dropout = [update(controller, base_speed=8.0, v_ego=0.1) for _ in range(controller.persist_frames + 5)]
    assert all(not result.stop_hold for result in dropout)
    assert all(result.launching for result in dropout)

  def test_stop_hold_without_usable_lead_stays_pinned_to_zero(self):
    controller = make_controller()
    enter_stop_hold(controller)
    missing = update(controller, base_speed=8.0, v_ego=0.1)

    assert missing.stop_hold
    assert missing.target_speed == 0.0
    assert missing.mpc_accel_max is None

  def test_moving_departure_does_not_reenter_stop_hold_once_launching(self):
    controller = make_controller()
    enter_stop_hold(controller)
    distance = 6.0
    results = []
    for speed in (0.81, 0.82, 0.83, 0.84, 0.79, 0.76, 0.74, 0.72):
      distance += speed * DT_MDL
      radar = make_radar(make_lead(status=True, d_rel=distance, v_lead_k=speed, radar_track_id=100))
      results.append(update(controller, radar, base_speed=8.0, v_ego=0.0))

    launch_index = next(index for index, result in enumerate(results) if result.launching)
    assert all(not result.stop_hold for result in results[launch_index:])
    assert all(result.target_speed > 0.0 and result.departure_launching for result in results[launch_index:])

  def test_far_stopped_lead_should_not_create_stop_hold(self):
    controller = make_controller()
    far_stopped = make_radar(make_lead(status=True, d_rel=60.0, v_lead_k=0.0))
    results = [update(controller, far_stopped, base_speed=12.0, v_ego=0.0) for _ in range(4)]
    assert all(not result.stop_hold for result in results)

  def test_fast_speed_glitch_without_distance_progress_should_stay_in_stop_hold(self):
    controller = make_controller()
    stopped = make_radar(make_lead(status=True, d_rel=6.0, v_lead_k=0.0, radar_track_id=100))
    update(controller, stopped, base_speed=8.0, v_ego=0.0)
    glitch = make_radar(make_lead(status=True, d_rel=6.0, v_lead_k=0.9, radar_track_id=100))
    results = [update(controller, glitch, base_speed=8.0, v_ego=0.0) for _ in range(STOP_HOLD_EXIT_FRAMES + 2)]

    assert all(result.stop_hold and result.target_speed == 0.0 and not result.launching for result in results)


class TestFreshnessAndReset:
  def test_frozen_output_during_a_single_non_fresh_radar_frame(self):
    controller = make_controller()
    radar = make_radar(make_lead(status=True, d_rel=20.0, v_lead_k=8.0, a_lead_k=-0.5))
    for _ in range(15):
      fresh = update(controller, radar, v_ego=10.0, planner_accel=-0.2)

    held = update(controller, radar, v_ego=10.0, planner_accel=-0.2, radar_fresh=False)
    assert held.target_speed == pytest.approx(fresh.target_speed)
    assert held.state == fresh.state
    assert held.selected_lead == fresh.selected_lead
    assert effective_accel_max(held) == pytest.approx(effective_accel_max(fresh))

  def test_stale_timeout_fully_resets_live_state(self):
    controller = make_controller()
    radar = restrictive_radar()
    for _ in range(15):
      update(controller, radar)
    held = [update(controller, radar_fresh=False) for _ in range(controller.radar_stale_frames - 1)]
    timed_out = update(controller, radar_fresh=False)

    assert all(result.active for result in held)
    assert not timed_out.active
    assert timed_out.target_speed == 25.0
    assert timed_out.mpc_accel_max is None
    assert timed_out.selected_lead == -1
    assert controller.pace.target_speed is None

  @pytest.mark.parametrize("override", [{"enabled": False}, {"acc_selected": False}, {"engaged": False},
                                        {"cruise_initialized": False}, {"a_ego": math.inf}])
  def test_bypass_or_invalid_context_resets_live_state(self, override):
    controller = make_controller()
    for _ in range(15):
      update(controller, restrictive_radar())
    result = update(controller, restrictive_radar(), **override)

    assert not result.active
    assert result.target_speed == 25.0
    assert result.mpc_accel_max is None
    assert controller.pace.target_speed is None

  def test_acc_bypass_does_not_retain_state_for_live_actuation(self):
    controller = make_controller()
    for _ in range(20):
      bypassed = update(controller, restrictive_radar(), acc_selected=False)
      assert not bypassed.active
      assert controller.pace.target_speed is None
    live = update(controller)

    # re-activation always re-seeds at min(base_speed, v_ego) and ramps up from there (see Pace.update's
    # seed block) rather than snapping straight to base_speed - a deliberate smoothing change from the
    # old code's seed_from_ego branch, which could seed directly at base_speed when there was no lead
    assert live.active
    assert 10.0 < live.target_speed <= 10.0 + TARGET_RELEASE_SLEW * DT_MDL + 1e-9

  def test_explicit_reset_clears_pace(self):
    controller = make_controller()
    for _ in range(15):
      update(controller, restrictive_radar())
    controller._jerk_smoothing_blocked = True
    controller._required_decel_samples = [0.2]
    controller._required_decel_lead = controller._required_decel_lead_track_id = 1
    controller._lead_trend_warmup = True
    controller.reset()

    assert not controller._jerk_smoothing_blocked
    assert controller._required_decel_samples == []
    assert controller._required_decel_lead == controller._required_decel_lead_track_id == -1
    assert not controller._lead_trend_warmup
    pace = controller.pace
    assert pace.target_speed is None and pace.matched_accel_limit is None
    assert not pace.stop_hold and not pace.launching and not pace.matched_lead
    assert math.isinf(pace.cap_trusted) and math.isinf(pace.filtered_lead_speed)
    assert pace.lead_speed_samples == [math.inf] * CAP_FILTER_FRAMES
    assert controller.state == AccelControllerState.inactive
    assert controller.selected_lead == controller.selected_lead_track_id == -1


class TestJerkCostMultiplier:
  @pytest.mark.parametrize("replacement_track_id", (200, -1), ids=("radar-track", "vision-track"))
  def test_track_id_change_requires_new_history_before_jerk_smoothing(self, replacement_track_id):
    controller = make_controller()
    controller.state = AccelControllerState.restrict
    controller.selected_lead = 0
    controller.selected_lead_track_id = 100
    controller.required_decel = 0.2
    original = [controller.get_jerk_cost_multiplier(True, True, 1.0, False) for _ in range(4)]

    controller.selected_lead_track_id = replacement_track_id
    replacement = [controller.get_jerk_cost_multiplier(True, True, 1.0, False) for _ in range(4)]

    assert original == [MPC_DECEL_JERK_COST_MULTIPLIER] * 4
    assert replacement == [1.0, 1.0, 1.0, MPC_DECEL_JERK_COST_MULTIPLIER]
    assert controller._required_decel_samples == [0.2] * 4

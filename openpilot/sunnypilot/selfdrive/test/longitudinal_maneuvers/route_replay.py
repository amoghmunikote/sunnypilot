"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Route-replay adapter for the AccelController offline validation/tuning harness.

Turns a real recorded route (fetched via `LogReader`) into the `LeadObservationFn` /
`ModelActionFn` / `EgoObservationFn` callback shapes `PlantSP` already accepts, so any
historical route becomes a replayable closed-loop regression case.

Named limitation: closed-loop for ego (the real AccelController/MPC drives the simulated
car), open-loop for the lead/model - the recorded radarState/model signal replays verbatim
as fixed ground truth and won't react to a different ego trajectory. Useful for input-noise
regressions; not a substitute for on-road confirmation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from openpilot.tools.lib.logreader import LogReader, ReadMode
from openpilot.tools.lib.log_time_series import msgs_to_time_series
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.test.longitudinal_maneuvers.plant import (
  EgoObservationFn, LeadObservation, LeadObservationFn, ModelActionFn,
)

# `msgs_to_time_series` calls capnp's to_dict() on every message with no per-message error
# handling; on some real routes this throws on message types unrelated to what we need
# (observed on selfdriveState/onroadEvents/pandaStates/can/sendcan). Filter to only what we
# read. selfdriveState is excluded here (it's one of the crashing types) but its
# experimentalMode field is still read directly off the raw capnp message below.
_WANTED_MSG_TYPES = frozenset({"radarState", "carState", "modelV2", "drivingModelData"})

_LEAD_CONTINUOUS_FIELDS: tuple[str, ...] = (
  "dRel", "yRel", "vRel", "vLead", "vLeadK", "aLeadK", "aLeadTau", "modelProb",
)
_LEAD_STEP_FIELDS: tuple[tuple[str, type], ...] = (
  ("present", bool),
  ("radar", bool),
  ("radarTrackId", int),
)


def _time_series_for_route(identifier: str, default_mode: ReadMode = ReadMode.AUTO, sources=None) -> dict[str, dict[str, np.ndarray]]:
  """Same underlying machinery as `LogReader(...).time_series`, scoped to the message types this adapter needs."""
  kwargs: dict[str, Any] = {"default_mode": default_mode}
  if sources is not None:
    kwargs["sources"] = sources
  lr = LogReader(identifier, **kwargs)
  experimental_mode_samples: list[tuple[float, bool]] = []

  def wanted_msgs():
    for msg in lr:
      which = msg.which()
      if which == "selfdriveState":
        try:
          experimental_mode_samples.append((msg.logMonoTime / 1.0e9, bool(msg.selfdriveState.experimentalMode)))
        except Exception:
          pass  # additive only - a miss here just means one fewer experimentalMode sample
      elif which in _WANTED_MSG_TYPES:
        yield msg

  time_series = msgs_to_time_series(wanted_msgs())
  if experimental_mode_samples:
    t, experimental_mode = zip(*experimental_mode_samples, strict=True)
    time_series["selfdriveState"] = {"t": np.asarray(t, dtype=float), "experimentalMode": np.asarray(experimental_mode, dtype=bool)}
  return time_series


@dataclass(frozen=True)
class _Series:
  """A recorded field as a function of route-relative time. Continuous fields interpolate
  linearly; categorical/boolean fields (present, radar, radarTrackId) zero-order-hold."""
  t: np.ndarray
  y: np.ndarray
  step: bool = False

  @property
  def empty(self) -> bool:
    return self.t.size == 0

  def at(self, current_time: float) -> float:
    if self.empty:
      return 0.0
    if self.step:
      idx = int(np.clip(np.searchsorted(self.t, current_time, side="right") - 1, 0, self.t.size - 1))
      return self.y[idx]
    return float(np.interp(current_time, self.t, self.y))

  @staticmethod
  def empty_series(step: bool = False) -> _Series:
    return _Series(np.asarray([], dtype=float), np.asarray([], dtype=float), step=step)


def _series_from_group(group: dict[str, np.ndarray] | None, key: str, t0: float, *, step: bool = False, dtype=float,
                        presence_mask: np.ndarray | None = None) -> _Series:
  if group is None or key not in group or len(group[key]) == 0:
    return _Series.empty_series(step=step)
  t = np.asarray(group["t"], dtype=float) - t0
  y = np.asarray(group[key], dtype=dtype)
  if presence_mask is not None:
    t = t[presence_mask]
    y = y[presence_mask]
  return _Series(t, y, step=step)


@dataclass(frozen=True)
class _LeadSeries:
  present: _Series
  radar: _Series
  radar_track_id: _Series
  d_rel: _Series
  y_rel: _Series
  v_rel: _Series
  v_lead: _Series
  v_lead_k: _Series
  a_lead_k: _Series
  a_lead_tau: _Series
  model_prob: _Series

  @property
  def empty(self) -> bool:
    return self.present.empty

  def at(self, current_time: float) -> LeadObservation | None:
    if self.empty or self.present.at(current_time) < 0.5:
      return None
    return {
      "dRel": self.d_rel.at(current_time),
      "yRel": self.y_rel.at(current_time),
      "vRel": self.v_rel.at(current_time),
      "vLead": self.v_lead.at(current_time),
      "vLeadK": self.v_lead_k.at(current_time),
      "aLeadK": self.a_lead_k.at(current_time),
      "aLeadTau": self.a_lead_tau.at(current_time),
      "modelProb": self.model_prob.at(current_time),
      "radar": bool(self.radar.at(current_time) >= 0.5),
      "radarTrackId": int(round(self.radar_track_id.at(current_time))),
    }


def _build_lead_series(radar_group: dict[str, np.ndarray] | None, prefix: str, t0: float) -> _LeadSeries:
  # cereal zero-fills a lead's continuous fields the instant `present` goes False. Building the
  # interpolatable series from unmasked samples would linearly ramp the last real detection into
  # that phantom zero - a fabricated "lead rushes to point-blank and vanishes" for any query time
  # before the transition sample is reached. Restrict to present-only samples instead.
  present_key = f"{prefix}/present"
  presence_mask = None
  if radar_group is not None and present_key in radar_group and len(radar_group[present_key]):
    presence_mask = np.asarray(radar_group[present_key], dtype=bool)
  cont = {f: _series_from_group(radar_group, f"{prefix}/{f}", t0, presence_mask=presence_mask) for f in _LEAD_CONTINUOUS_FIELDS}
  step = {f: _series_from_group(radar_group, f"{prefix}/{f}", t0, step=True, dtype=dtype) for f, dtype in _LEAD_STEP_FIELDS}
  return _LeadSeries(
    present=step["present"], radar=step["radar"], radar_track_id=step["radarTrackId"],
    d_rel=cont["dRel"], y_rel=cont["yRel"], v_rel=cont["vRel"], v_lead=cont["vLead"], v_lead_k=cont["vLeadK"],
    a_lead_k=cont["aLeadK"], a_lead_tau=cont["aLeadTau"], model_prob=cont["modelProb"],
  )


@dataclass(frozen=True)
class _ModelSeries:
  desired_acceleration: _Series
  should_stop: _Series

  @property
  def empty(self) -> bool:
    return self.desired_acceleration.empty

  def at(self, current_time: float, fallback_acceleration: float) -> tuple[float, bool]:
    if self.empty:
      return fallback_acceleration + 0.1, False  # matches PlantSP's own no-model_action_fn default
    return self.desired_acceleration.at(current_time), bool(self.should_stop.at(current_time) >= 0.5)


@dataclass(frozen=True)
class RouteReplay:
  """Time-indexed view of one recorded route, ready to drive `PlantSP`."""

  identifier: str
  duration: float
  initial_speed: float
  initial_distance_lead: float
  lead_one: _LeadSeries
  lead_two: _LeadSeries
  model: _ModelSeries
  v_cruise: _Series
  v_ego: _Series
  a_ego: _Series
  e2e: bool

  def lead_observation_fn(self) -> LeadObservationFn:
    leads = {"leadOne": self.lead_one, "leadTwo": self.lead_two}

    def observe(current_time: float, lead_name: str, truth: LeadObservation) -> LeadObservation | None:
      del truth  # replaying real recorded signal instead of Plant's own synthetic truth
      lead = leads.get(lead_name)
      return None if lead is None else lead.at(current_time)

    return observe

  def model_action_fn(self) -> ModelActionFn:
    model = self.model

    def model_action(current_time: float, speed: float, acceleration: float) -> tuple[float, bool]:
      del speed
      return model.at(current_time, fallback_acceleration=acceleration)

    return model_action

  def ego_observation_fn(self) -> EgoObservationFn:
    # ego dynamics are genuinely closed-loop here (no recorded "truth" to splice in) - identity
    # passthrough kept so callers wanting to layer sensor noise on top have a matching hook
    def ego_observation(current_time: float, true_v_ego: float, true_a_ego: float) -> tuple[float, float]:
      del current_time
      return true_v_ego, true_a_ego

    return ego_observation

  def v_lead_fn(self) -> Callable[[float], float]:
    """Feeds `PlantSP.step(v_lead=...)` so its distance_lead bookkeeping tracks the real lead
    independent of whatever `lead_observation_fn` shows the controller that frame.

    Gates on `present` and returns 0.0 during a real presence gap (freezing distance_lead)
    rather than interpolating v_lead_k across it, which would fabricate a trajectory between two
    potentially unrelated detections. Freezing is a one-directional, permanent low bias (ego's
    distance keeps advancing, distance_lead doesn't) - a full fix would re-anchor distance_lead
    to distance + dRel on reacquisition, which needs plant state this function doesn't have.
    Treat min_gap/gap! in the report as a possibly-falsely-tight bound on low-presence routes.
    """
    lead = self.lead_one

    def v_lead(current_time: float) -> float:
      if lead.empty or lead.present.at(current_time) < 0.5:
        return 0.0
      return lead.v_lead_k.at(current_time)

    return v_lead

  def v_cruise_fn(self) -> Callable[[float], float]:
    def v_cruise(current_time: float) -> float:
      return self.v_cruise.at(current_time)

    return v_cruise

  def plant_kwargs(self, *, e2e: bool | None = None) -> dict[str, Any]:
    """Convenience bundle of the PlantSP constructor kwargs this route determines.

    `e2e=None` uses the route's own recorded Experimental Mode setting (see `from_time_series`).
    Pass an explicit True/False to override - e.g. `e2e=False` to validate the acc-mode path on
    a route that was actually recorded in Experimental Mode.
    """
    return {
      # a route with no recorded lead should replay lead-free, not force PlantSP's synthetic
      # 100m fallback obstacle on - combined with a stationary start, that produced a real,
      # reproducible solver-failure storm that froze the replay at v_ego=0
      "lead_relevancy": not self.lead_one.empty,
      "speed": self.initial_speed,
      "distance_lead": self.initial_distance_lead,
      "e2e": self.e2e if e2e is None else e2e,
      "lead_observation_fn": self.lead_observation_fn(),
      "model_action_fn": self.model_action_fn(),
      "ego_observation_fn": self.ego_observation_fn(),
    }

  @staticmethod
  def from_time_series(identifier: str, time_series: dict[str, dict[str, np.ndarray]]) -> RouteReplay:
    radar_group = time_series.get("radarState")
    car_group = time_series.get("carState")
    # modelV2 is full-rate but only present in rlogs; drivingModelData is qlog-decimated 10x
    model_group = time_series.get("modelV2")
    model_prefix = "action"
    if model_group is None or f"{model_prefix}/desiredAcceleration" not in model_group:
      model_group = time_series.get("drivingModelData")

    groups_t0 = [g["t"][0] for g in (radar_group, car_group, model_group) if g is not None and len(g["t"])]
    if not groups_t0:
      raise ValueError(f"route {identifier!r} has none of radarState/carState/modelV2/drivingModelData")
    t0 = float(min(groups_t0))
    groups_t1 = [g["t"][-1] for g in (radar_group, car_group, model_group) if g is not None and len(g["t"])]
    t1 = float(max(groups_t1))

    # carState.vCruise reads the V_CRUISE_UNSET sentinel (255) verbatim whenever cruise isn't
    # engaged. Feeding that into the planner reliably drives the MPC solver to fail every frame
    # while stationary. AccelController can't activate without cruise_initialized anyway, so trim
    # the replay to the span bracketed by real (non-sentinel) vCruise samples rather than
    # fabricating an "engaged" scenario that never happened.
    if car_group is not None and "vCruise" in car_group and len(car_group["vCruise"]):
      car_t = np.asarray(car_group["t"], dtype=float)
      car_v_cruise_kph = np.asarray(car_group["vCruise"], dtype=float)
      initialized = car_v_cruise_kph != V_CRUISE_UNSET
      if not initialized.any():
        raise ValueError(f"route {identifier!r} carState.vCruise is V_CRUISE_UNSET for the entire route - nothing to replay")
      t0 = max(t0, float(car_t[initialized].min()))
      t1 = min(t1, float(car_t[initialized].max()))
      if t1 <= t0:
        raise ValueError(f"route {identifier!r} has no time span left after trimming to the cruise-initialized window (t0={t0}, t1={t1})")
    duration = t1 - t0

    # experimentalMode is a per-drive toggle, not per-frame - take the majority value over the
    # replayed window. Without this, PlantSP's e2e kwarg was never set, so is_e2e() could never
    # return True and AccelController's e2e_braking_handoff path had zero replay coverage. Note:
    # this alone doesn't guarantee the handoff *transition* is exercised - that also needs DEC's
    # own mode() to toggle blended/acc mid-run, which is route-dependent.
    e2e = False
    selfdrive_group = time_series.get("selfdriveState")
    if selfdrive_group is not None and "experimentalMode" in selfdrive_group and len(selfdrive_group["experimentalMode"]):
      selfdrive_t = np.asarray(selfdrive_group["t"], dtype=float)
      experimental_mode = np.asarray(selfdrive_group["experimentalMode"], dtype=bool)
      in_window = (selfdrive_t >= t0) & (selfdrive_t <= t1)
      if in_window.any():
        e2e = bool(experimental_mode[in_window].mean() >= 0.5)

    lead_one = _build_lead_series(radar_group, "leadOne", t0)
    lead_two = _build_lead_series(radar_group, "leadTwo", t0)
    model = _ModelSeries(
      desired_acceleration=_series_from_group(model_group, f"{model_prefix}/desiredAcceleration", t0),
      should_stop=_series_from_group(model_group, f"{model_prefix}/shouldStop", t0, step=True, dtype=bool),
    )
    v_ego = _series_from_group(car_group, "vEgo", t0)
    a_ego = _series_from_group(car_group, "aEgo", t0)
    v_cruise_kph = _series_from_group(car_group, "vCruise", t0)
    # drop any brief mid-window re-blip back to V_CRUISE_UNSET rather than trimming around it, so
    # _Series.at() interpolates across the gap instead of ever handing the sentinel to the plant
    if not v_cruise_kph.empty:
      keep = v_cruise_kph.y != V_CRUISE_UNSET
      v_cruise_kph = _Series(v_cruise_kph.t[keep], v_cruise_kph.y[keep]) if keep.any() else _Series.empty_series()
    v_cruise = _Series(v_cruise_kph.t, v_cruise_kph.y / 3.6) if not v_cruise_kph.empty else v_cruise_kph

    initial_speed = v_ego.at(0.0) if not v_ego.empty else 0.0
    initial_distance_lead = lead_one.d_rel.at(0.0) if lead_one.at(0.0) is not None else 100.0

    return RouteReplay(
      identifier=identifier, duration=duration, initial_speed=initial_speed, initial_distance_lead=initial_distance_lead,
      lead_one=lead_one, lead_two=lead_two, model=model, v_cruise=v_cruise, v_ego=v_ego, a_ego=a_ego, e2e=e2e,
    )


def load_route(identifier: str, *, default_mode: ReadMode = ReadMode.AUTO, sources=None) -> RouteReplay:
  """Fetch `identifier` (a route name, optionally `dongle|route[/seg][/mode-selector]`) and build
  a `RouteReplay`. `default_mode=ReadMode.AUTO` falls back to qlog per-segment when rlogs aren't
  available (non-prime devices only upload qlogs)."""
  time_series = _time_series_for_route(identifier, default_mode=default_mode, sources=sources)
  return RouteReplay.from_time_series(identifier, time_series)


def load_routes(identifiers: Iterable[str], **kwargs) -> list[RouteReplay]:
  return [load_route(identifier, **kwargs) for identifier in identifiers]

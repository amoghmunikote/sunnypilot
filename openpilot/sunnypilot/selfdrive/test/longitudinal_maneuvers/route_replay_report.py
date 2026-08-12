#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Batch offline-validation runner for `AccelController`, built on `route_replay.py`.

For each given route: replay it through `PlantSP` driven by the real, current
`AccelController`, then run the same pump/jerk/gap-safety metric helpers the
closed-loop unit-test suite already uses (`_has_propulsion_brake_cycle`,
`_has_brake_coast_brake`, `_command_jerk`, `_filtered_realized_jerk`, all
imported unmodified from `test_accel_controller_closed_loop.py`) and report
which routes trip which metric.

Plain script, invoked the same way as the existing
`tools/longitudinal_maneuvers/mpc_longitudinal_tuning_report.py` - no
pytest/CI integration:

  python3 route_replay_report.py <route_id> [route_id...] [options]

Options:
  --duration SECS        cap replay duration per route (default: full route)
  --profile {0,1,2}      AccelController profile: eco/normal/sport (default: 1, normal)
  --dec                  leave DEC (Dynamic Experimental Control) enabled (default: off,
                          matching the closed-loop suite's default harness configuration)
  --qlog | --rlog | --auto  log source preference (default: auto = rlog, falling back to
                          qlog per-segment; most non-prime devices only upload qlogs)
  --sweep NAME=v1,v2,...  sweep one `pace.py` constant across the given values (repeatable);
                          each route is re-run once per sweep value, with the module attribute
                          monkeypatched and restored around the run
  --out PATH              write an HTML report (per-route acceleration/gap plots, mirroring
                          `mpc_longitudinal_tuning_report.py`'s plot style) to PATH

Named limit (see route_replay.py for the full explanation): this is closed-loop for ego,
open-loop for the lead/model - a real regression/tuning signal, not a substitute for a final
on-road confirmation pass.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller import pace as pace_module
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_accel_controller_closed_loop import (
  _command_jerk, _filtered_realized_jerk, _has_brake_coast_brake, _has_propulsion_brake_cycle,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.sunnypilot.selfdrive.test.longitudinal_maneuvers.plant import PRIUS_TSS2_ROUTE_MODEL, PlantSP
from openpilot.sunnypilot.selfdrive.test.longitudinal_maneuvers.route_replay import RouteReplay, load_route


@dataclass
class SimpleTrace:
  """Minimal duck-typed stand-in for `ClosedLoopTrace` - `_command_jerk`/`_filtered_realized_jerk`
  only ever read `.time`/`.a_target`/`.acceleration`, so building the real 26-field dataclass
  (whose other fields don't apply outside the synthetic test harness) would be pure boilerplate."""
  time: np.ndarray
  a_target: np.ndarray
  acceleration: np.ndarray


@dataclass
class RouteRunResult:
  route_id: str
  sweep: tuple[tuple[str, float], ...]
  duration: float
  frames: int
  solver_failures: int
  pump_cycle: bool
  brake_coast_brake: bool
  command_jerk_p95: float
  realized_jerk_p95: float
  realized_jerk_max: float
  min_gap_with_lead: float
  gap_violation: bool
  solver_failure_rate: float
  speed_frozen: bool
  lead_presence_fraction: float
  stuck_despite_go_signal: bool
  trace: SimpleTrace
  distance: np.ndarray = field(repr=False)
  distance_lead: np.ndarray = field(repr=False)
  speed: np.ndarray = field(repr=False)
  lead_present: np.ndarray = field(repr=False)


def _configure_controller(plant: PlantSP, *, enabled: bool, profile: int, dec_enabled: bool) -> None:
  # Mirrors `_configure_plant()` in test_accel_controller_closed_loop.py.
  plant.planner.accel_controller.enabled = enabled
  plant.planner.accel_controller.profile = profile
  plant.planner.accel_controller.update_params = lambda: None
  plant.planner.dec._enabled = dec_enabled
  plant.planner.dec._read_params = lambda: None


def replay_route(
  route: RouteReplay,
  *,
  duration: float | None = None,
  profile: int = 1,
  controller_enabled: bool = True,
  dec_enabled: bool = False,
  sweep: tuple[tuple[str, float], ...] = (),
) -> RouteRunResult:
  """Drive `PlantSP` with the real `AccelController` over one replayed route and score it
  with the existing closed-loop-suite metric helpers."""
  gc.collect()
  end_time = route.duration if duration is None else min(duration, route.duration)

  plant = PlantSP(actuator_model=PRIUS_TSS2_ROUTE_MODEL, **route.plant_kwargs())
  _configure_controller(plant, enabled=controller_enabled, profile=profile, dec_enabled=dec_enabled)

  solver_failures = 0
  original_mpc_reset = plant.planner.mpc.reset

  def count_failed_solve(*args, **kwargs) -> None:
    nonlocal solver_failures
    if plant.planner.mpc.solution_status != 0:
      solver_failures += 1
    original_mpc_reset(*args, **kwargs)

  plant.planner.mpc.reset = count_failed_solve

  v_lead_fn = route.v_lead_fn()
  v_cruise_fn = route.v_cruise_fn()
  lead_present_fn = route.lead_one.present

  times, a_targets, accelerations, distances, distances_lead, speeds, lead_present, controller_targets = [], [], [], [], [], [], [], []
  try:
    while plant.current_time < end_time:
      current_time = plant.current_time
      result = plant.step(v_lead=v_lead_fn(current_time), v_cruise=v_cruise_fn(current_time))
      times.append(current_time)
      a_targets.append(result["a_target"])
      accelerations.append(result["realized_acceleration"])
      distances.append(result["distance"])
      distances_lead.append(result["distance_lead"])
      speeds.append(result["speed"])
      lead_present.append(lead_present_fn.at(current_time) >= 0.5)
      controller_targets.append(result["controller_target"])
  finally:
    plant.planner.mpc.reset = original_mpc_reset
  gc.collect()

  time = np.asarray(times)
  a_target = np.asarray(a_targets)
  acceleration = np.asarray(accelerations)
  distance = np.asarray(distances)
  distance_lead = np.asarray(distances_lead)
  speed = np.asarray(speeds)
  lead_present_arr = np.asarray(lead_present, dtype=bool)
  controller_target = np.asarray(controller_targets)

  trace = SimpleTrace(time=time, a_target=a_target, acceleration=acceleration)
  command_jerk = _command_jerk(trace) if time.size >= 2 else np.asarray([0.0])
  realized_jerk = _filtered_realized_jerk(trace) if time.size >= 4 else np.asarray([0.0])

  gap = distance_lead - distance
  gap_with_lead = gap[lead_present_arr] if lead_present_arr.any() else np.asarray([np.inf])
  min_gap_with_lead = float(np.min(gap_with_lead))

  # a replay that mostly fails to solve, or whose speed never plausibly moves, isn't a clean pass
  # just because no assertion tripped - every other metric on the row is meaningless, not "no findings"
  solver_failure_rate = solver_failures / time.size if time.size else 0.0
  speed_frozen = bool(end_time > 5.0 and np.ptp(speed) < 0.5)
  # distance_lead freezes (doesn't re-anchor) during a presence gap - see route_replay.py's
  # v_lead_fn - so a low-presence route makes min_gap/gap_violation an ever-growing harness
  # artifact rather than a real finding. Flag it instead of printing a silent false violation.
  lead_presence_fraction = float(lead_present_arr.mean()) if lead_present_arr.size else 0.0
  # ego is closed-loop, the lead is open-loop (replayed verbatim) - if ego's own simulated speed
  # ever falls to a full stop while the real recorded lead keeps moving on its own real trajectory,
  # the two diverge: every subsequent "gap"/closing-speed reading the MPC's obstacle cost sees is
  # computed against a lead position that assumed a moving ego, not a stalled one, and can look like
  # an imminent collision regardless of what AccelController's own target_speed says. Confirmed on a
  # real route: AccelController's controller_target stayed at ~full v_cruise (a clean "go" signal)
  # for over 300s while sim speed stayed pinned at 0 - not a controller defect, a replay-fidelity
  # limit. Flag it structurally: one sustained (contiguous) near-zero-speed run despite a
  # controller_target that says go - not the aggregate of many short launch-transient blips (a car
  # legitimately reads v_ego<0.1 for a few tenths of a second right as it starts moving, and a route
  # with enough launches sums past any reasonable total-time threshold without ever really stalling).
  dt = float(time[1] - time[0]) if time.size > 1 else 0.0
  stuck_mask = (np.abs(speed) < 0.1) & (controller_target > 5.0)
  longest_stuck_run = 0
  if stuck_mask.any():
    run = 0
    for is_stuck in stuck_mask:
      run = run + 1 if is_stuck else 0
      longest_stuck_run = max(longest_stuck_run, run)
  stuck_despite_go_signal = bool(longest_stuck_run * dt > 30.0)

  return RouteRunResult(
    route_id=route.identifier, sweep=sweep, duration=float(end_time), frames=time.size, solver_failures=solver_failures,
    pump_cycle=bool(_has_propulsion_brake_cycle(a_target)), brake_coast_brake=bool(_has_brake_coast_brake(a_target)),
    command_jerk_p95=float(np.percentile(np.abs(command_jerk), 95)), realized_jerk_p95=float(np.percentile(np.abs(realized_jerk), 95)),
    realized_jerk_max=float(np.max(np.abs(realized_jerk))), min_gap_with_lead=min_gap_with_lead,
    gap_violation=bool(min_gap_with_lead < STOP_DISTANCE), solver_failure_rate=solver_failure_rate, speed_frozen=speed_frozen,
    lead_presence_fraction=lead_presence_fraction, stuck_despite_go_signal=stuck_despite_go_signal,
    trace=trace, distance=distance, distance_lead=distance_lead, speed=speed, lead_present=lead_present_arr,
  )


def _print_report(results: list[RouteRunResult]) -> None:
  header = (
    f"{'route':<48} {'sweep':<24} {'dur':>6} {'frames':>7} {'solver_fail':>11} {'pump':>5} {'brk_coast_brk':>14} " +
    f"{'cmd_jerk_p95':>13} {'real_jerk_p95':>14} {'real_jerk_max':>14} {'min_gap':>8} {'gap!':>5}"
  )
  print(header)
  print("-" * len(header))
  for r in results:
    sweep_label = ",".join(f"{k}={v:g}" for k, v in r.sweep) if r.sweep else "(baseline)"
    flags = []
    if r.pump_cycle:
      flags.append("PUMP")
    if r.brake_coast_brake:
      flags.append("BRK-COAST-BRK")
    if r.gap_violation:
      flags.append("GAP<STOP_DISTANCE" + (" (UNRELIABLE, see below)" if r.lead_presence_fraction < 0.7 else ""))
    if r.solver_failures:
      flags.append(f"{r.solver_failures} solver failures")
    if r.solver_failure_rate > 0.10 or r.speed_frozen:
      flags.append(
        f"UNRELIABLE REPLAY (solver failed {r.solver_failure_rate:.0%} of frames"
        + (", speed never moved" if r.speed_frozen else "") + ") - other metrics on this row are not meaningful"
      )
    if r.lead_presence_fraction < 0.7:
      flags.append(
        f"UNRELIABLE min_gap/GAP<STOP_DISTANCE (lead present only {r.lead_presence_fraction:.0%} of frames - "
        + "distance_lead freezes rather than re-anchors during absence, see route_replay.py's v_lead_fn docstring)"
      )
    if r.stuck_despite_go_signal:
      flags.append(
        "STUCK DESPITE GO SIGNAL (sim speed pinned near 0 for 30s+ while controller_target says go) - "
        + "likely closed-loop-ego/open-loop-lead divergence, not necessarily an AccelController defect; "
        + "pump_cycle/brake_coast_brake/min_gap on this row are not meaningful"
      )
    print(
      f"{r.route_id:<48} {sweep_label:<24} {r.duration:6.1f} {r.frames:7d} {r.solver_failures:11d} " +
      f"{str(r.pump_cycle):>5} {str(r.brake_coast_brake):>14} {r.command_jerk_p95:13.3f} {r.realized_jerk_p95:14.3f} " +
      f"{r.realized_jerk_max:14.3f} {r.min_gap_with_lead:8.2f} {str(r.gap_violation):>5}"
    )
    if flags:
      print(f"    -> {', '.join(flags)}")


def _write_html_report(results: list[RouteRunResult], out_path: str) -> None:
  import io
  import markdown
  import matplotlib.pyplot as plt

  def plot(results_for_route: list[RouteRunResult], title: str) -> str:
    fig, (ax_a, ax_gap) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for r in results_for_route:
      label = ",".join(f"{k}={v:g}" for k, v in r.sweep) if r.sweep else "baseline"
      ax_a.plot(r.trace.time, r.trace.a_target, label=label)
      ax_gap.plot(r.trace.time, r.distance_lead - r.distance, label=label)
    ax_a.set_ylabel("a_target (m/s^2)")
    ax_a.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax_a.grid(True, linestyle="--", alpha=0.7)
    ax_gap.set_ylabel("gap (m)")
    ax_gap.set_xlabel("time (s)")
    ax_gap.grid(True, linestyle="--", alpha=0.7)
    fig.suptitle(title)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue() + "<br/>"

  by_route: dict[str, list[RouteRunResult]] = {}
  for r in results:
    by_route.setdefault(r.route_id, []).append(r)

  with open(out_path, "w") as f:
    f.write(markdown.markdown("# AccelController route-replay report"))
    for route_id, route_results in by_route.items():
      f.write(markdown.markdown(f"## {route_id}"))
      f.write(plot(route_results, route_id))


def _parse_sweep(spec: str) -> tuple[str, list[float]]:
  name, _, values = spec.partition("=")
  if not name or not values:
    raise ValueError(f"--sweep expects NAME=v1,v2,... got {spec!r}")
  return name, [float(v) for v in values.split(",")]


def main(argv: list[str]) -> int:
  from openpilot.tools.lib.logreader import ReadMode

  route_ids: list[str] = []
  duration: float | None = None
  profile = 1
  dec_enabled = False
  default_mode = ReadMode.AUTO
  sweeps: list[tuple[str, list[float]]] = []
  out_path: str | None = None

  args = iter(argv)
  for arg in args:
    if arg == "--duration":
      duration = float(next(args))
    elif arg == "--profile":
      profile = int(next(args))
    elif arg == "--dec":
      dec_enabled = True
    elif arg == "--qlog":
      default_mode = ReadMode.QLOG
    elif arg == "--rlog":
      default_mode = ReadMode.RLOG
    elif arg == "--auto":
      default_mode = ReadMode.AUTO
    elif arg == "--sweep":
      sweeps.append(_parse_sweep(next(args)))
    elif arg == "--out":
      out_path = next(args)
    elif arg.startswith("-"):
      raise SystemExit(f"unknown option {arg!r}")
    else:
      route_ids.append(arg)

  if not route_ids:
    print(__doc__)
    return 1

  sweep_combinations: list[tuple[tuple[str, float], ...]] = [()]
  for name, values in sweeps:
    sweep_combinations = [combo + ((name, value),) for combo in sweep_combinations for value in values]

  results: list[RouteRunResult] = []
  for route_id in route_ids:
    route = load_route(route_id, default_mode=default_mode)
    for combo in sweep_combinations:
      originals = {name: getattr(pace_module, name) for name, _ in combo}
      try:
        for name, value in combo:
          setattr(pace_module, name, value)
        results.append(replay_route(route, duration=duration, profile=profile, dec_enabled=dec_enabled, sweep=combo))
      finally:
        for name, value in originals.items():
          setattr(pace_module, name, value)

  _print_report(results)
  if out_path is not None:
    _write_html_report(results, out_path)
    print(f"\nwrote HTML report to {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))

import pytest

from opendbc.car import DT_CTRL, gen_empty_fingerprint, structs
from opendbc.car.car_helpers import interfaces
from opendbc.car.gm.values import CAR as GM
from opendbc.car.honda.values import CAR as HONDA
from opendbc.car.hyundai.values import CAR as HYUNDAI
from opendbc.car.rivian.values import CAR as RIVIAN
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.volkswagen.values import CAR as VOLKSWAGEN
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.sunnypilot.selfdrive.controls.lib.longcontrol import STOPPED_SPEED, STOPPING_SETTLE_FRAMES
from openpilot.sunnypilot.selfdrive.test.longitudinal_maneuvers.plant import PRIUS_TSS2_ROUTE_MODEL, PlantSP


STOP_ACCEL_VEHICLES = (TOYOTA.TOYOTA_RAV4_TSS2, HONDA.HONDA_CIVIC_2022, VOLKSWAGEN.VOLKSWAGEN_ARTEON_MK1, RIVIAN.RIVIAN_R1)
SETTLE_VEHICLES = (TOYOTA.TOYOTA_RAV4_TSS2, HONDA.HONDA_CIVIC_2022, VOLKSWAGEN.VOLKSWAGEN_ARTEON_MK1)
ROUTE_STOP_ONSETS = (
  (0.280, -0.290, -0.220, -0.220), (0.290, -0.497, -0.270, -0.302), (0.464, -0.223, -0.264, -0.292),
  (0.467, -0.582, -0.316, -0.359),
  (0.530, -0.311, -0.309, -0.333), (0.581, -0.467, -0.312, -0.352), (0.398, -0.557, -0.311, -0.348),
  (0.517, -0.290, -0.301, -0.327), (0.312, -0.420, -0.271, -0.304), (0.474, -0.509, -0.303, -0.347),
  (0.241, -0.554, -0.573, -0.617), (0.292, -0.154, -0.302, -0.326),
)


def get_car_params(candidate):
  fingerprint = gen_empty_fingerprint()
  interface = interfaces[candidate]
  CP = interface.get_params(candidate, fingerprint, [], True, False, False)
  return CP, interface.get_params_sp(CP, candidate, fingerprint, [], True, False, False)


def make_car_state(v_ego=0.2, a_ego=0.0, standstill=False) -> structs.CarState:
  state = structs.CarState(vEgo=float(v_ego), aEgo=float(a_ego), standstill=standstill)
  state.cruiseState.standstill = standstill
  return state


def make_control(candidate, initial_accel=-0.33):
  CP, CP_SP = get_car_params(candidate)
  control = LongControl(CP, CP_SP)
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = initial_accel
  return CP, control


def stock_stopping_output(output_accel, stop_accel):
  return min(output_accel, 0.0) - DT_CTRL if output_accel > stop_accel else output_accel


def test_stop_threshold_remains_unchanged():
  assert should_stop(0.24, 0.0)
  assert not should_stop(0.26, 0.0)
  assert not should_stop(0.24, 0.1)


@pytest.mark.parametrize(("v_ego", "a_ego", "a_target", "initial_accel"), ROUTE_STOP_ONSETS)
def test_logged_stop_onsets_hold_the_existing_brake(v_ego, a_ego, a_target, initial_accel):
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, initial_accel)
  output = control.update(True, make_car_state(v_ego, a_ego), a_target, True, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.stopping
  assert output == pytest.approx(initial_accel)


def test_glide_hold_survives_a_soft_deceleration_sample():
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, -0.166)
  samples = ((0.388, -0.201, -0.164), (0.330, -0.120, -0.140), (0.283, -0.0675, -0.120))
  outputs = [control.update(True, make_car_state(v_ego, a_ego), a_target, True, (-3.5, 2.0))
             for v_ego, a_ego, a_target in samples]

  assert outputs == pytest.approx([-0.166] * len(samples))


def test_glide_hold_releases_when_the_vehicle_stops_decelerating():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, -0.166)
  control.update(True, make_car_state(0.388, -0.201), -0.164, True, (-3.5, 2.0))
  output = control.update(True, make_car_state(0.330, -0.01), -0.140, True, (-3.5, 2.0))

  assert output == pytest.approx(stock_stopping_output(-0.166, CP.stopAccel))


def test_glide_hold_keeps_the_stopping_distance_bounded():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, -0.166)
  control.update(True, make_car_state(0.388, -0.201), -0.164, True, (-3.5, 2.0))
  output = control.update(True, make_car_state(0.400, -0.050), -0.140, True, (-3.5, 2.0))

  assert output == pytest.approx(stock_stopping_output(-0.166, CP.stopAccel))


def test_glide_hold_yields_to_stronger_planner_braking():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, -0.166)
  control.update(True, make_car_state(0.388, -0.201), -0.164, True, (-3.5, 2.0))
  output = control.update(True, make_car_state(0.330, -0.120), -1.0, True, (-3.5, 2.0))

  assert output == pytest.approx(stock_stopping_output(-0.166, CP.stopAccel))


@pytest.mark.parametrize("candidate", STOP_ACCEL_VEHICLES)
def test_urgent_braking_matches_the_stock_ramp(candidate):
  CP, control = make_control(candidate)
  CS = make_car_state(0.8, -0.1)
  output = control.last_output_accel

  for _ in range(round(1.0 / DT_CTRL)):
    output = control.update(True, CS, -3.0, True, (-3.5, 2.0))

  expected = -0.33
  for _ in range(round(1.0 / DT_CTRL)):
    expected = stock_stopping_output(expected, CP.stopAccel)
  assert output == pytest.approx(expected)


@pytest.mark.parametrize("candidate", STOP_ACCEL_VEHICLES)
def test_stronger_planner_brake_matches_the_stock_ramp(candidate):
  CP, control = make_control(candidate)
  outputs = [control.update(True, make_car_state(0.3, -0.3), -1.0, True, (-3.5, 2.0)) for _ in range(10)]
  expected = []
  output = -0.33
  for _ in range(10):
    output = stock_stopping_output(output, CP.stopAccel)
    expected.append(output)
  assert outputs == pytest.approx(expected)


@pytest.mark.parametrize("candidate", STOP_ACCEL_VEHICLES)
def test_insufficient_deceleration_uses_the_stock_ramp_immediately(candidate):
  CP, control = make_control(candidate)
  output = control.update(True, make_car_state(0.6, -0.1), -0.1, True, (-3.5, 2.0))
  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


def test_deceleration_noise_cannot_release_the_brake():
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  outputs = [control.update(True, make_car_state(0.3, -0.3 if frame % 2 else 0.0), -0.1, True, (-3.5, 2.0)) for frame in range(40)]
  assert all(current <= previous for previous, current in zip(outputs[:-1], outputs[1:], strict=True))


def test_planner_noise_cannot_release_the_brake():
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  outputs = [control.update(True, make_car_state(0.3, -0.3), -1.0 if frame % 2 else -0.1, True, (-3.5, 2.0)) for frame in range(40)]
  assert all(current <= previous for previous, current in zip(outputs[:-1], outputs[1:], strict=True))


@pytest.mark.parametrize(("v_ego", "a_ego", "a_target"), (
  (float("nan"), -0.3, -0.1), (0.3, float("nan"), -0.1), (0.3, -0.3, float("nan")),
  (float("inf"), -0.3, -0.1), (0.3, -float("inf"), -0.1), (0.3, -0.3, float("inf")),
))
def test_invalid_state_uses_the_stock_ramp(v_ego, a_ego, a_target):
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  output = control.update(True, make_car_state(v_ego, a_ego), a_target, True, (-3.5, 2.0))
  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


@pytest.mark.parametrize(("speed", "initial_accel", "grade_accel", "actuator_lag"), (
  (0.24, 0.0, 0.0, 0.15), (0.464, -0.223, 0.0, 0.25), (0.53, -0.31, 0.0, 0.35),
  (0.24, 0.0, 0.49, 0.15), (0.53, -0.31, 0.49, 0.25), (0.6, -0.3, 0.49, 0.35), (0.6, -0.3, 0.49, 0.5),
))
def test_smooth_stop_distance_is_bounded(speed, initial_accel, grade_accel, actuator_lag):
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, initial_accel)
  applied_accel = initial_accel
  distance = 0.0

  for _ in range(round(4.0 / DT_CTRL)):
    command = control.update(True, make_car_state(speed, applied_accel), -0.1, True, (-3.5, 2.0))
    applied_accel += DT_CTRL / actuator_lag * (command + grade_accel - applied_accel)
    speed = max(0.0, speed + applied_accel * DT_CTRL)
    distance += speed * DT_CTRL
    if speed == 0.0:
      break

  assert speed == 0.0
  assert distance < 1.0


@pytest.mark.parametrize("candidate", STOP_ACCEL_VEHICLES)
def test_standstill_uses_the_stock_ramp(candidate):
  CP, control = make_control(candidate)
  control.long_control_state = LongCtrlState.off
  CS = make_car_state(0.0, 0.0, standstill=True)
  outputs = [control.update(True, CS, 0.0, False, (-3.5, 2.0)) for _ in range(round(2.0 / DT_CTRL))]
  expected = -0.33
  for _ in range(round(2.0 / DT_CTRL)):
    expected = stock_stopping_output(expected, CP.stopAccel)
  assert outputs[0] == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))
  assert outputs[-1] == pytest.approx(expected)


@pytest.mark.parametrize("candidate", SETTLE_VEHICLES)
def test_final_stop_holds_brake_until_vehicle_settles(candidate):
  CP, control = make_control(candidate)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  outputs = [control.update(True, make_car_state(0.0006, a_ego, standstill=True), -0.032, True, (-3.5, 2.0))
             for a_ego in (-1.098, -0.950, -0.609, -0.286)]
  assert outputs == pytest.approx([-0.33] * len(outputs))

  settled = make_car_state(0.0, -0.065, standstill=True)
  output = control.update(True, settled, -0.032, True, (-3.5, 2.0))
  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


@pytest.mark.parametrize("a_ego", (-0.09, 0.0, 0.1))
def test_settled_vehicle_uses_the_stock_hold_ramp(a_ego):
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  output = control.update(True, make_car_state(0.0, a_ego, standstill=True), -0.1, True, (-3.5, 2.0))
  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


@pytest.mark.parametrize("candidate", SETTLE_VEHICLES)
def test_final_settling_hold_is_bounded(candidate):
  CP, control = make_control(candidate)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  CS = make_car_state(0.0, -0.3, standstill=True)
  outputs = [control.update(True, CS, -0.1, True, (-3.5, 2.0)) for _ in range(STOPPING_SETTLE_FRAMES + 1)]

  assert outputs[:STOPPING_SETTLE_FRAMES] == pytest.approx([-0.33] * STOPPING_SETTLE_FRAMES)
  assert outputs[-1] == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


@pytest.mark.parametrize(("v_ego", "a_ego", "standstill"), ((0.6, -0.1, False), (0.0, 0.0, True)))
def test_stopping_never_releases_a_stronger_command(v_ego, a_ego, standstill):
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2, -3.0)
  output = control.update(True, make_car_state(v_ego, a_ego, standstill), 0.0, True, (-3.5, 2.0))
  assert output == pytest.approx(-3.0)


def test_reported_standstill_while_moving_can_hold_the_brake():
  _, control = make_control(GM.CHEVROLET_BOLT_EUV)
  control.long_control_state = LongCtrlState.off
  output = control.update(True, make_car_state(0.3, -0.3, standstill=True), -0.1, False, (-3.5, 2.0))
  assert output == pytest.approx(-0.33)


def test_stopping_removes_positive_acceleration_immediately():
  _, control = make_control(HYUNDAI.HYUNDAI_SONATA, 0.2)
  output = control.update(True, make_car_state(0.2, -0.2), -0.1, True, (-3.5, 2.0))
  assert output == pytest.approx(-DT_CTRL)


def test_rollback_uses_the_stock_ramp():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  output = control.update(True, make_car_state(-0.1, 0.1), -0.1, True, (-3.5, 2.0))
  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


def test_rollback_after_settling_arms_uses_the_stock_ramp():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  control.update(True, make_car_state(0.01, -0.3), -0.1, True, (-3.5, 2.0))
  output = control.update(True, make_car_state(-0.04, -0.3), -0.1, True, (-3.5, 2.0))

  assert output == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


def test_small_velocity_noise_does_not_interrupt_final_settling():
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  output = control.update(True, make_car_state(-0.04, -0.3, standstill=True), -0.1, True, (-3.5, 2.0))
  assert output == pytest.approx(-0.33)


def test_terminal_speed_chatter_cannot_extend_settling_hold():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  outputs = [control.update(True, make_car_state(0.019 if frame % 2 == 0 else 0.021, -0.3), -0.1, True, (-3.5, 2.0))
             for frame in range(STOPPING_SETTLE_FRAMES + 2)]

  assert outputs[STOPPING_SETTLE_FRAMES] == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))
  assert outputs[-1] < outputs[STOPPING_SETTLE_FRAMES]


def test_terminal_speed_plateau_cannot_extend_settling_hold():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  CS = make_car_state(0.03, -0.3)
  outputs = [control.update(True, CS, -0.1, True, (-3.5, 2.0)) for _ in range(STOPPING_SETTLE_FRAMES + 1)]

  assert outputs[-1] == pytest.approx(stock_stopping_output(-0.33, CP.stopAccel))


def test_interrupted_stop_cannot_reuse_settling_hold():
  CP, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.update(True, make_car_state(0.28, -0.29), -0.22, True, (-3.5, 2.0))
  control.update(False, make_car_state(0.0, 0.0, standstill=True), 0.0, False, (-3.5, 2.0))
  output = control.update(True, make_car_state(0.0, -0.3, standstill=True), -0.1, True, (-3.5, 2.0))

  assert output == pytest.approx(stock_stopping_output(0.0, CP.stopAccel))


def test_departure_uses_the_stock_pid_path():
  _, control = make_control(TOYOTA.TOYOTA_RAV4_TSS2)
  control.long_control_state = LongCtrlState.stopping
  output = control.update(True, make_car_state(0.0), 0.6, False, (-3.5, 2.0))
  assert control.long_control_state == LongCtrlState.pid
  assert output > 0.0


def test_planner_mpc_and_longcontrol_complete_a_smooth_stop():
  plant = PlantSP(
    lead_relevancy=True, speed=0.6, distance_lead=3.6, run_long_control=True,
    actuator_model=PRIUS_TSS2_ROUTE_MODEL,
  )
  plant.planner.accel_controller.enabled = True
  plant.planner.accel_controller.profile = 1
  plant.planner.accel_controller.update_params = lambda: None
  plant.planner.dec._enabled = False
  plant.planner.dec._read_params = lambda: None
  commands = []
  speeds = []
  states = []
  solver_statuses = []

  while plant.current_time < 5.0:
    result = plant.step(v_lead=0.0, v_cruise=8.0)
    commands.append(result["actuator_command"])
    speeds.append(result["speed"])
    states.append(result["long_control_state"])
    solver_statuses.append(plant.planner.mpc.last_solution_status)

  stopping = states.index(LongCtrlState.stopping)
  moving_stop_commands = [command for command, state, speed in zip(commands, states, speeds, strict=True)
                          if state == LongCtrlState.stopping and speed > STOPPED_SPEED]
  assert all(current <= previous + 1e-9 for previous, current in zip(commands[stopping:-1], commands[stopping + 1:], strict=True))
  assert len(moving_stop_commands) > 1 and max(moving_stop_commands) - min(moving_stop_commands) < 1e-9
  assert plant.speed == 0.0 and plant.distance < 1.0
  assert plant.distance_lead - plant.distance > 3.0
  assert all(status == 0 for status in solver_statuses)

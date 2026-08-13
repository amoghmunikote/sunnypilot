"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

STOPPING_DISTANCE = 0.75
STOPPED_SPEED = 0.02
STOPPING_TIME = 2.5
STOPPING_ACCEL_TOLERANCE = 0.1
STOPPING_SPEED_TOLERANCE = 0.05
STOPPING_SETTLE_FRAMES = 30


class LongControlSP:
  def __init__(self):
    self._stopping_settle_frames: int | None = None
    self._stopping_active = False

  def update_state(self, stopping: bool) -> None:
    if stopping != self._stopping_active:
      self._stopping_settle_frames = None
    self._stopping_active = stopping

  def stopping_decel_rate(self, CS, a_target: float) -> float:
    if not all(math.isfinite(value) for value in (self.last_output_accel, a_target, CS.vEgo, CS.aEgo)):
      return 1.0
    can_hold = self.last_output_accel <= 0.0 and a_target >= self.last_output_accel
    terminal_speed = (0.0 <= CS.vEgo <= STOPPING_SPEED_TOLERANCE
                      or CS.standstill and abs(CS.vEgo) <= STOPPING_SPEED_TOLERANCE)
    if self.last_output_accel > 0.0 or CS.vEgo < 0.0 and not terminal_speed:
      return 1.0
    if terminal_speed and self._stopping_settle_frames is None:
      return 1.0

    time_decel = 0.0 if self._stopping_settle_frames is not None else CS.vEgo / STOPPING_TIME
    required_decel = max(time_decel, CS.vEgo ** 2 / (2.0 * STOPPING_DISTANCE), 1e-3)
    adequacy = min(max(-CS.aEgo / required_decel, 0.0), 1.0)
    if not terminal_speed and self._stopping_settle_frames is None and can_hold and adequacy >= 1.0:
      self._stopping_settle_frames = 0

    motion_need = 1.0 - adequacy ** 2
    planner_need = min(max((self.last_output_accel - a_target) / max(required_decel, STOPPING_ACCEL_TOLERANCE), 0.0), 1.0)
    terminal_need = 0.0
    if terminal_speed or self._stopping_settle_frames not in (None, 0):
      self._stopping_settle_frames = min(self._stopping_settle_frames + 1, STOPPING_SETTLE_FRAMES)
      terminal_need = (self._stopping_settle_frames / STOPPING_SETTLE_FRAMES) ** 2

    return max(motion_need, planner_need, terminal_need)

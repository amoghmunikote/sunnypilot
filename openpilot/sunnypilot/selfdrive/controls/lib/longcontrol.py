"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import math

STOPPING_DISTANCE = 0.75
STOPPED_SPEED = 0.02
STOPPING_TIME = 2.5
STOPPING_SETTLE_ACCEL = 0.1
STOPPING_SPEED_TOLERANCE = 0.05
STOPPING_SETTLE_FRAMES = 30


class LongControlSP:
  def __init__(self):
    self._stopping_settle_frames: int | None = None
    self._stopping_settle_complete = False
    self._stopping_active = False

  def update_state(self, stopping: bool) -> None:
    if stopping != self._stopping_active:
      self._stopping_settle_frames = None
      self._stopping_settle_complete = False
    self._stopping_active = stopping

  def should_hold_stopping(self, CS, a_target: float) -> bool:
    if not all(math.isfinite(value) for value in (self.last_output_accel, a_target, CS.vEgo, CS.aEgo)):
      self._stopping_settle_complete = True
      return False
    if self._stopping_settle_complete:
      return False

    can_hold = self.last_output_accel <= 0.0 and a_target >= self.last_output_accel
    terminal_speed = (0.0 <= CS.vEgo <= STOPPING_SPEED_TOLERANCE + 1e-6
                      or CS.standstill and abs(CS.vEgo) <= STOPPING_SPEED_TOLERANCE + 1e-6)
    moving = (not terminal_speed and CS.vEgo > STOPPED_SPEED and CS.aEgo < 0.0 and CS.vEgo <= -CS.aEgo * STOPPING_TIME
              and CS.vEgo ** 2 <= -2.0 * CS.aEgo * STOPPING_DISTANCE)
    if moving and can_hold and self._stopping_settle_frames in (None, 0):
      self._stopping_settle_frames = 0
      return True

    settling = can_hold and CS.aEgo < -STOPPING_SETTLE_ACCEL and terminal_speed
    if settling and self._stopping_settle_frames is not None and 0 <= self._stopping_settle_frames < STOPPING_SETTLE_FRAMES:
      self._stopping_settle_frames += 1
      return True

    if self._stopping_settle_frames is not None:
      self._stopping_settle_complete = True
    return False

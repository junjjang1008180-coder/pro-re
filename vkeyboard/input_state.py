"""손가락 누름 상태 머신 (HOVER -> PRESSING -> PRESSED -> RELEASED).

dy 는 손가락 끝이 기준선(휴지 위치)보다 아래로 내려간 거리(px, 추론 좌표계)이며
양수가 "아래"다. 임계값은 손 크기에 맞춰 호출 측에서 보정해 전달한다.

규칙
- dy >= release 임계값: 아래로 움직이기 시작 -> PRESSING
- PRESSING 중 dy >= press 임계값이 min_down_frames 프레임 연속 -> PRESSED (1회만 입력)
- PRESSED 중에는 아래에 계속 머물러도 재입력 없음
- dy <= release 임계값까지 다시 올라와야 RELEASED -> 새 입력 허용
- 쿨다운 중 확정되면 입력 없이 PRESSED 로만 전환(해제 후 다시 눌러야 함)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FingerState(Enum):
    HOVER = "HOVER"
    PRESSING = "PRESSING"
    PRESSED = "PRESSED"
    RELEASED = "RELEASED"


@dataclass
class PressParams:
    press_distance: float
    release_distance: float
    min_down_frames: int
    cooldown: float
    press_max_duration: float = 0.6
    released_hold: float = 0.15


class FingerStateMachine:
    def __init__(self, params: PressParams) -> None:
        if params.release_distance >= params.press_distance:
            raise ValueError("RELEASE_DISTANCE 는 PRESS_DISTANCE 보다 작아야 합니다 (히스테리시스)")
        self.params = params
        self.state = FingerState.HOVER
        self.down_frames = 0
        self.last_fire_time = float("-inf")
        self.press_start = 0.0
        self.released_at = 0.0
        self.suppressed = 0            # 쿨다운 때문에 무시된 누름 수
        self.needs_rebaseline = False  # 느린 드리프트로 취소됨 -> 기준선 재설정 필요

    def reset(self) -> None:
        self.state = FingerState.HOVER
        self.down_frames = 0

    def update(self, dy: float, t: float, valid: bool = True,
               press_threshold: Optional[float] = None,
               release_threshold: Optional[float] = None) -> bool:
        """한 프레임 갱신. 이번 프레임에 키 입력이 확정되면 True."""
        self.needs_rebaseline = False
        press_th = self.params.press_distance if press_threshold is None else press_threshold
        release_th = self.params.release_distance if release_threshold is None else release_threshold

        if not valid:
            # 누르는 도중 추적이 끊기면 확정하지 않고 취소. PRESSED 는 유지(복귀 시 재입력 방지).
            if self.state is FingerState.PRESSING:
                self.state = FingerState.HOVER
                self.down_frames = 0
            return False

        if self.state in (FingerState.HOVER, FingerState.RELEASED):
            if self.state is FingerState.RELEASED and t - self.released_at >= self.params.released_hold:
                self.state = FingerState.HOVER
            if dy < release_th:
                return False
            # 아래로 움직이기 시작
            self.state = FingerState.PRESSING
            self.press_start = t
            self.down_frames = 1 if dy >= press_th else 0
            return self._maybe_confirm(t)

        if self.state is FingerState.PRESSING:
            if dy < release_th:
                self.state = FingerState.HOVER
                self.down_frames = 0
                return False
            self.down_frames = self.down_frames + 1 if dy >= press_th else 0
            if self.down_frames == 0 and t - self.press_start > self.params.press_max_duration:
                # 짧은 누름이 아니라 천천히 내려온 것 -> 취소하고 기준선 재설정
                self.state = FingerState.HOVER
                self.needs_rebaseline = True
                return False
            return self._maybe_confirm(t)

        # PRESSED
        if dy <= release_th:
            self.state = FingerState.RELEASED
            self.released_at = t
            self.down_frames = 0
        return False

    def _maybe_confirm(self, t: float) -> bool:
        if self.down_frames < self.params.min_down_frames:
            return False
        self.state = FingerState.PRESSED
        if t - self.last_fire_time >= self.params.cooldown:
            self.last_fire_time = t
            return True
        self.suppressed += 1
        return False

    @property
    def is_down(self) -> bool:
        return self.state in (FingerState.PRESSING, FingerState.PRESSED)

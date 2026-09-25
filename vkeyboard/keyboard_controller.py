"""손가락별 독립 상태 머신으로 키 입력을 판정한다 (동시 입력 지원)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math

from .config import (BASELINE_ALPHA, KEY_SNAP_RADIUS, MIN_DOWN_TIME, RELEASED_HOLD_TIME, AppConfig,
                     hand_scale)
from .events import KeyEvent
from .finger_tracker import FingerSample
from .geometry import Vec2
from .input_state import FingerState, FingerStateMachine, PressParams
from .keyboard_layout import FINGER_ORDER, Finger, KeyboardLayout, KeyCode


@dataclass
class FingerView:
    """UI 표시용 손가락 상태 스냅샷."""

    finger: Finger
    state: FingerState
    valid: bool
    reason: str
    tip: Optional[Vec2]
    selected_key: Optional[KeyCode]
    dy: float
    press_threshold: float


class KeyboardController:
    def __init__(self, cfg: AppConfig, layout: KeyboardLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        params = PressParams(cfg.press_distance, cfg.release_distance, cfg.min_down_frames,
                             cfg.key_cooldown, cfg.press_max_duration, RELEASED_HOLD_TIME)
        self.machines: Dict[Finger, FingerStateMachine] = {f: FingerStateMachine(params) for f in FINGER_ORDER}
        self._baseline: Dict[Finger, float] = {}
        self._hover_key: Dict[Finger, Optional[KeyCode]] = {f: None for f in FINGER_ORDER}
        self._locked_key: Dict[Finger, Optional[KeyCode]] = {f: None for f in FINGER_ORDER}
        self._last_key_time: Dict[KeyCode, float] = {}
        self.views: Dict[Finger, FingerView] = {}
        self._last_t: Optional[float] = None
        self._dt = 1 / 30.0            # 프레임 간격 추정 (EMA)

    def set_layout(self, layout: KeyboardLayout) -> None:
        self.layout = layout

    def update(self, samples: Dict[Finger, FingerSample], t: float, active: bool) -> List[KeyEvent]:
        """한 프레임 처리. active 가 False 면 상태만 갱신하고 이벤트는 만들지 않는다."""
        events: List[KeyEvent] = []
        if self._last_t is not None and 0 < t - self._last_t < 0.5:
            self._dt = 0.8 * self._dt + 0.2 * (t - self._last_t)
        self._last_t = t
        # FPS 보정: 30fps 면 3프레임(0.1초), 15fps 면 2프레임 — 같은 '시간' 기준으로 확정
        min_frames = min(self.cfg.min_down_frames, max(1, math.ceil(MIN_DOWN_TIME / self._dt - 1e-9)))
        for f in FINGER_ORDER:
            s = samples.get(f)
            m = self.machines[f]
            if s is None or not s.valid or s.tip is None:
                m.update(0.0, t, valid=False)
                if s is None or s.reason in ("missing", "mouse"):  # 자세가 바뀌므로 기준선 폐기
                    self._baseline.pop(f, None)
                self.views[f] = FingerView(f, m.state, False, s.reason if s else "missing",
                                           s.tip if s else None, self._current_key(f), 0.0, 0.0)
                continue

            scale = hand_scale(s.hand_size, self.cfg.reference_hand_size)
            press_th = self.cfg.press_distance * scale
            release_th = self.cfg.release_distance * scale

            if s.resynced:
                # 추적이 새로 잡히거나 크게 이동한 직후: 이전 기준선은 무효 -> 현재 자세를 휴지 위치로
                self._baseline[f] = s.rel_y
                m.reset()
            baseline = self._baseline.setdefault(f, s.rel_y)
            dy = s.rel_y - baseline

            prev_state = m.state
            if prev_state in (FingerState.HOVER, FingerState.RELEASED):
                key = self.layout.key_for_finger(f, s.tip, KEY_SNAP_RADIUS)
                self._hover_key[f] = key.code if key else None

            fired = m.update(dy, t, True, press_th, release_th, min_frames)

            if prev_state in (FingerState.HOVER, FingerState.RELEASED) and m.is_down:
                # 누르기 시작한 순간의 키로 고정 (누르는 동안 손끝이 아래 줄로 미끄러져도 유지)
                self._locked_key[f] = self._hover_key[f]

            if m.needs_rebaseline:
                self._baseline[f] = s.rel_y
            elif m.state in (FingerState.HOVER, FingerState.RELEASED):
                # 기준선: 위로는 즉시, 아래로는 천천히 따라감 (느린 드리프트 흡수)
                if s.rel_y < baseline:
                    self._baseline[f] = s.rel_y
                else:
                    self._baseline[f] = baseline + BASELINE_ALPHA * (s.rel_y - baseline)

            if fired and active:
                code = self._locked_key[f]
                if code is not None and t - self._last_key_time.get(code, float("-inf")) >= self.cfg.key_cooldown:
                    self._last_key_time[code] = t
                    events.append(KeyEvent(f, code, t, dy, s.hand_size, s.confidence))

            self.views[f] = FingerView(f, m.state, True, s.reason, s.tip, self._current_key(f), dy, press_th)
        return events

    def _current_key(self, f: Finger) -> Optional[KeyCode]:
        return self._locked_key[f] if self.machines[f].is_down else self._hover_key[f]

    def any_pressing(self) -> bool:
        return any(m.is_down for m in self.machines.values())

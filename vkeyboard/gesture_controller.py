"""양손 손바닥 펼침 1초 유지 -> ACTIVE/INACTIVE 전환."""
from __future__ import annotations

from typing import Optional, Sequence

from .geometry import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
                       PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_MCP, RING_PIP, RING_TIP, THUMB_TIP,
                       WRIST, HandObservation, Vec2, dist, hand_size)

_FINGERS = ((INDEX_TIP, INDEX_PIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP), (PINKY_TIP, PINKY_PIP, PINKY_MCP))


# 판정 기준 (손 크기 = 손목~중지 끝 대비 비율). 실제 손에서 너무 엄격하지 않게 맞춘 값.
EXTEND_RATIO = 1.1        # 손목~손끝 / 손목~PIP  (굽히면 1.0 근처)
UP_RATIO = 0.2            # 손끝이 MCP 보다 이만큼 위
SPREAD_RATIO = 0.35       # 검지 끝 ~ 새끼 끝 거리
THUMB_RATIO = 0.25        # 엄지 끝 ~ 검지 MCP 거리
HOLD_GRACE = 0.3          # 1초 유지 중 인식이 이 시간까지 끊겨도 진행 유지


def is_palm_open(landmarks: Sequence[Vec2]) -> bool:
    """손바닥을 활짝 편 상태인지 판정.

    타이핑 자세(손가락이 굽혀져 아래를 향함)와 구분하기 위해
    ① 네 손가락이 펴짐 ② 손끝이 기준 관절보다 위 ③ 손가락 벌어짐 ④ 엄지 벌어짐 을 확인한다.
    """
    if len(landmarks) < 21:
        return False
    size = hand_size(landmarks)
    if size <= 1e-6:
        return False
    wrist = landmarks[WRIST]
    for tip_i, pip_i, mcp_i in _FINGERS:
        tip, pip, mcp = landmarks[tip_i], landmarks[pip_i], landmarks[mcp_i]
        if dist(wrist, tip) < dist(wrist, pip) * EXTEND_RATIO:
            return False                       # 굽혀짐
        if tip.y > mcp.y - UP_RATIO * size:
            return False                       # 위로 펴지지 않음
    if dist(landmarks[INDEX_TIP], landmarks[PINKY_TIP]) < SPREAD_RATIO * size:
        return False                           # 손가락이 모여 있음
    if dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) < THUMB_RATIO * size:
        return False                           # 엄지가 접힘
    return True


POINT_UP_RATIO = 0.5      # 검지 끝이 검지 MCP 보다 손바닥 길이의 이 비율 이상 위
FOLD_RATIO = 1.05         # 손목~손끝 < 손목~PIP * 이 값 이면 접힌 손가락
MOUSE_ENTER_TIME = 0.25   # 가리키기 자세를 이만큼 유지하면 마우스 모드
MOUSE_EXIT_TIME = 0.4     # 자세를 풀고 이만큼 지나면 타이핑 모드


def is_pointing(landmarks: Sequence[Vec2]) -> bool:
    """마우스 모드 자세: 검지를 위로 펴고 약지·새끼를 접은 '가리키기' 손 모양.

    타이핑 자세(손가락들이 아래/앞을 향해 펴져 있음)와 겹치지 않도록
    검지가 '위'를 향해야 하고 약지·새끼가 손바닥 쪽으로 접혀야 한다. 중지는 자유(우클릭 핀치용).
    """
    if len(landmarks) < 21:
        return False
    wrist = landmarks[WRIST]
    palm = dist(wrist, landmarks[MIDDLE_MCP])
    if palm <= 1e-6:
        return False
    tip, pip, mcp = landmarks[INDEX_TIP], landmarks[INDEX_PIP], landmarks[INDEX_MCP]
    if dist(wrist, tip) < dist(wrist, pip) * EXTEND_RATIO:
        return False                           # 검지가 굽혀짐
    if tip.y > mcp.y - POINT_UP_RATIO * palm:
        return False                           # 검지가 위를 향하지 않음
    for tip_i, pip_i in ((RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)):
        if dist(wrist, landmarks[tip_i]) > dist(wrist, landmarks[pip_i]) * FOLD_RATIO:
            return False                       # 약지/새끼가 펴져 있음
    return True


class MouseModeDetector:
    """오른손 가리키기 자세로 마우스 모드 on/off (히스테리시스로 깜빡임 방지)."""

    def __init__(self, enter_time: float = MOUSE_ENTER_TIME, exit_time: float = MOUSE_EXIT_TIME) -> None:
        self.enter_time = enter_time
        self.exit_time = exit_time
        self.on = False
        self._since: Optional[float] = None

    def update(self, hand: Optional[HandObservation], t: float) -> bool:
        pointing = hand is not None and is_pointing(hand.landmarks)
        if pointing == self.on:
            self._since = None
            return self.on
        if self._since is None:
            self._since = t
        if t - self._since >= (self.exit_time if self.on else self.enter_time):
            self.on = pointing
            self._since = None
        return self.on

    def reset(self) -> None:
        self.on = False
        self._since = None


class GestureController:
    def __init__(self, hold_seconds: float = 1.0) -> None:
        self.hold_seconds = hold_seconds
        self.active = False            # 시작 상태는 반드시 INACTIVE
        self._hold_start: Optional[float] = None
        self._armed = True             # 전환 후엔 한 번 손을 내려야 다시 전환 가능
        self.progress = 0.0            # UI 표시용 0~1
        self.open_state = {"Left": False, "Right": False}   # UI 표시용: 손별 펼침 인식 여부
        self._last_open: Optional[float] = None

    def update(self, hands: Sequence[HandObservation], t: float) -> Optional[bool]:
        """전환이 일어나면 새 active 값을, 아니면 None 을 반환."""
        for label in ("Left", "Right"):
            hand = next((h for h in hands if h.handedness == label), None)
            self.open_state[label] = hand is not None and is_palm_open(hand.landmarks)
        both_open = self.open_state["Left"] and self.open_state["Right"]
        if both_open:
            self._last_open = t
        elif self._last_open is not None and t - self._last_open <= HOLD_GRACE:
            return None                # 잠깐 인식이 끊긴 것: 진행 상태 유지
        if not both_open:
            self._hold_start = None
            self._armed = True
            self.progress = 0.0
            return None
        if not self._armed:
            self.progress = 1.0
            return None
        if self._hold_start is None:
            self._hold_start = t
        held = t - self._hold_start
        self.progress = min(1.0, held / self.hold_seconds)
        if held >= self.hold_seconds:
            self.active = not self.active
            self._armed = False
            self._hold_start = None
            return self.active
        return None

    def set_active(self, active: bool) -> bool:
        """수동 전환(창에서 'a' 키). 상태가 바뀌었으면 True."""
        changed = self.active != active
        self.active = active
        self._hold_start = None
        self.progress = 0.0
        return changed

    def force_inactive(self) -> bool:
        """추적 실패 등으로 강제 비활성화. 상태가 바뀌었으면 True."""
        changed = self.active
        self.active = False
        self._hold_start = None
        self.progress = 0.0
        return changed

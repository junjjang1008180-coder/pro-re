"""양손 손바닥 펼침 1초 유지 -> ACTIVE/INACTIVE 전환."""
from __future__ import annotations

from typing import Optional, Sequence

from .geometry import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
                       PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_MCP, RING_PIP, RING_TIP, THUMB_TIP,
                       WRIST, HandObservation, Vec2, dist, hand_size)

_FINGERS = ((INDEX_TIP, INDEX_PIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP), (PINKY_TIP, PINKY_PIP, PINKY_MCP))


def is_palm_open(landmarks: Sequence[Vec2]) -> bool:
    """손바닥을 활짝 편 상태인지 판정.

    타이핑 자세(손가락이 모여 있고 굽혀져 있음)와 구분하기 위해
    ① 네 손가락이 모두 펴짐 ② 손끝이 기준 관절보다 충분히 위 ③ 손가락 벌어짐 ④ 엄지 벌어짐
    을 모두 만족해야 한다.
    """
    if len(landmarks) < 21:
        return False
    size = hand_size(landmarks)
    if size <= 1e-6:
        return False
    wrist = landmarks[WRIST]
    for tip_i, pip_i, mcp_i in _FINGERS:
        tip, pip, mcp = landmarks[tip_i], landmarks[pip_i], landmarks[mcp_i]
        if dist(wrist, tip) < dist(wrist, pip) * 1.2:
            return False                       # 굽혀짐
        if tip.y > mcp.y - 0.35 * size:
            return False                       # 위로 곧게 펴지지 않음
    if dist(landmarks[INDEX_TIP], landmarks[PINKY_TIP]) < 0.5 * size:
        return False                           # 손가락이 모여 있음
    if dist(landmarks[THUMB_TIP], landmarks[INDEX_MCP]) < 0.35 * size:
        return False                           # 엄지가 접힘
    return True


class GestureController:
    def __init__(self, hold_seconds: float = 1.0) -> None:
        self.hold_seconds = hold_seconds
        self.active = False            # 시작 상태는 반드시 INACTIVE
        self._hold_start: Optional[float] = None
        self._armed = True             # 전환 후엔 한 번 손을 내려야 다시 전환 가능
        self.progress = 0.0            # UI 표시용 0~1

    def update(self, hands: Sequence[HandObservation], t: float) -> Optional[bool]:
        """전환이 일어나면 새 active 값을, 아니면 None 을 반환."""
        lefts = [h for h in hands if h.handedness == "Left"]
        rights = [h for h in hands if h.handedness == "Right"]
        both_open = (len(lefts) >= 1 and len(rights) >= 1
                     and is_palm_open(lefts[0].landmarks) and is_palm_open(rights[0].landmarks))
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

    def force_inactive(self) -> bool:
        """추적 실패 등으로 강제 비활성화. 상태가 바뀌었으면 True."""
        changed = self.active
        self.active = False
        self._hold_start = None
        self.progress = 0.0
        return changed

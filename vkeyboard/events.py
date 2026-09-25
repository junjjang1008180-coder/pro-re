"""추론 스레드 -> 메인(렌더/입력) 스레드로 전달되는 이벤트 타입."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .keyboard_layout import Finger, KeyCode


@dataclass(frozen=True)
class KeyEvent:
    finger: Finger
    key: KeyCode
    t: float
    dy: float               # 확정 순간의 누름 깊이(px, 추론 좌표계)
    hand_size: float
    confidence: float


class MouseAction(Enum):
    MOVE = "move"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"   # 더블클릭의 두 번째 클릭 (첫 클릭은 LEFT_CLICK 으로 이미 전송됨)
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"


@dataclass(frozen=True)
class MouseEvent:
    action: MouseAction
    x: int                  # 화면 좌표
    y: int
    t: float
    pinch_distance: float = 0.0
    hand_size: float = 0.0
    confidence: float = 0.0


@dataclass(frozen=True)
class ModeEvent:
    active: bool
    reason: str             # "gesture" | "tracking_lost"
    t: float

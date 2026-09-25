"""QWERTY 가상 키보드 배치와 손가락별 담당 키."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .geometry import Vec2


class Finger(Enum):
    LEFT_THUMB = "left_thumb"
    LEFT_INDEX = "left_index"
    LEFT_MIDDLE = "left_middle"
    LEFT_RING = "left_ring"
    LEFT_PINKY = "left_pinky"
    RIGHT_THUMB = "right_thumb"
    RIGHT_INDEX = "right_index"
    RIGHT_MIDDLE = "right_middle"
    RIGHT_RING = "right_ring"
    RIGHT_PINKY = "right_pinky"

    @property
    def hand(self) -> str:
        return "Left" if self.value.startswith("left") else "Right"

    @property
    def digit(self) -> str:
        return self.value.split("_", 1)[1]


FINGER_ORDER: Tuple[Finger, ...] = tuple(Finger)


class KeyCode(Enum):
    A = "A"; B = "B"; C = "C"; D = "D"; E = "E"; F = "F"; G = "G"; H = "H"; I = "I"  # noqa: E702,E741
    J = "J"; K = "K"; L = "L"; M = "M"; N = "N"; O = "O"; P = "P"; Q = "Q"; R = "R"  # noqa: E702
    S = "S"; T = "T"; U = "U"; V = "V"; W = "W"; X = "X"; Y = "Y"; Z = "Z"  # noqa: E702
    SEMICOLON = ";"
    COMMA = ","
    PERIOD = "."
    SLASH = "/"
    SPACE = "Space"
    TAB = "Tab"
    CAPS_LOCK = "Caps"
    LEFT_SHIFT = "LShift"
    RIGHT_SHIFT = "RShift"
    BACKSPACE = "Bksp"
    ENTER = "Enter"
    HAN_ENG = "Han/En"         # 한/영 전환 (OS 입력기 토글)

    @property
    def label(self) -> str:
        return self.value

    @property
    def char(self) -> Optional[str]:
        """문자 키면 입력 문자(소문자), 아니면 None."""
        if len(self.value) == 1:
            return self.value.lower()
        if self is KeyCode.SPACE:
            return " "
        return None

    @property
    def is_modifier(self) -> bool:
        return self in (KeyCode.LEFT_SHIFT, KeyCode.RIGHT_SHIFT)


def key_for_char(ch: str) -> Optional[KeyCode]:
    for k in KeyCode:
        if k.char == ch.lower():
            return k
    return None


# 손가락별 담당 키 (요구사항 표 그대로)
_K = KeyCode
FINGER_KEYS: Dict[Finger, Tuple[KeyCode, ...]] = {
    Finger.LEFT_PINKY: (_K.Q, _K.A, _K.Z, _K.TAB, _K.CAPS_LOCK, _K.LEFT_SHIFT),
    Finger.LEFT_RING: (_K.W, _K.S, _K.X),
    Finger.LEFT_MIDDLE: (_K.E, _K.D, _K.C),
    Finger.LEFT_INDEX: (_K.R, _K.F, _K.V, _K.T, _K.G, _K.B),
    Finger.LEFT_THUMB: (_K.SPACE,),
    Finger.RIGHT_INDEX: (_K.Y, _K.H, _K.N, _K.U, _K.J, _K.M),
    Finger.RIGHT_MIDDLE: (_K.I, _K.K, _K.COMMA),
    Finger.RIGHT_RING: (_K.O, _K.L, _K.PERIOD),
    Finger.RIGHT_PINKY: (_K.P, _K.SEMICOLON, _K.SLASH, _K.BACKSPACE, _K.ENTER, _K.RIGHT_SHIFT),
    Finger.RIGHT_THUMB: (_K.SPACE, _K.HAN_ENG),
}

# 두벌식 자판: 한국어 모드일 때 키 위에 표시할 자모
HANGUL_JAMO: Dict[KeyCode, str] = {
    _K.Q: "ㅂ", _K.W: "ㅈ", _K.E: "ㄷ", _K.R: "ㄱ", _K.T: "ㅅ", _K.Y: "ㅛ", _K.U: "ㅕ", _K.I: "ㅑ",
    _K.O: "ㅐ", _K.P: "ㅔ", _K.A: "ㅁ", _K.S: "ㄴ", _K.D: "ㅇ", _K.F: "ㄹ", _K.G: "ㅎ", _K.H: "ㅗ",
    _K.J: "ㅓ", _K.K: "ㅏ", _K.L: "ㅣ", _K.Z: "ㅋ", _K.X: "ㅌ", _K.C: "ㅊ", _K.V: "ㅍ", _K.B: "ㅠ",
    _K.N: "ㅜ", _K.M: "ㅡ",
}


def fingers_for_key(code: KeyCode) -> Tuple[Finger, ...]:
    return tuple(f for f, keys in FINGER_KEYS.items() if code in keys)


def primary_finger_for_char(ch: str) -> Optional[Finger]:
    """연습 모드용: 문자를 쳐야 하는 기본 손가락 (Space 는 오른손 엄지)."""
    code = key_for_char(ch)
    if code is None:
        return None
    if code is KeyCode.SPACE:
        return Finger.RIGHT_THUMB
    fingers = fingers_for_key(code)
    return fingers[0] if fingers else None


@dataclass(frozen=True)
class Key:
    code: KeyCode
    ux: float   # 키보드 단위 좌표 (1 unit = 일반 키 1칸)
    uy: float
    uw: float
    uh: float = 1.0

    @property
    def fingers(self) -> Tuple[Finger, ...]:
        return fingers_for_key(self.code)


UNITS_W = 13.5
UNITS_H = 4.0

# (코드, 폭) 행 정의
_ROWS: Tuple[Tuple[Tuple[KeyCode, float], ...], ...] = (
    ((_K.TAB, 1.5), (_K.Q, 1), (_K.W, 1), (_K.E, 1), (_K.R, 1), (_K.T, 1), (_K.Y, 1), (_K.U, 1),
     (_K.I, 1), (_K.O, 1), (_K.P, 1), (_K.BACKSPACE, 2.0)),
    ((_K.CAPS_LOCK, 1.75), (_K.A, 1), (_K.S, 1), (_K.D, 1), (_K.F, 1), (_K.G, 1), (_K.H, 1),
     (_K.J, 1), (_K.K, 1), (_K.L, 1), (_K.SEMICOLON, 1), (_K.ENTER, 1.75)),
    ((_K.LEFT_SHIFT, 2.25), (_K.Z, 1), (_K.X, 1), (_K.C, 1), (_K.V, 1), (_K.B, 1), (_K.N, 1),
     (_K.M, 1), (_K.COMMA, 1), (_K.PERIOD, 1), (_K.SLASH, 1), (_K.RIGHT_SHIFT, 1.25)),
)
_SPACE_X = 3.5
_SPACE_W = 6.5
_HAN_ENG_X = 10.25          # 실제 한국어 키보드처럼 Space 오른쪽
_HAN_ENG_W = 1.75


def _build_keys() -> List[Key]:
    keys: List[Key] = []
    for row, defs in enumerate(_ROWS):
        x = 0.0
        for code, w in defs:
            keys.append(Key(code, x, float(row), float(w)))
            x += w
    keys.append(Key(_K.SPACE, _SPACE_X, 3.0, _SPACE_W))
    keys.append(Key(_K.HAN_ENG, _HAN_ENG_X, 3.0, _HAN_ENG_W))
    return keys


BASE_KEYS: Tuple[Key, ...] = tuple(_build_keys())


class KeyboardLayout:
    """키보드 위치/크기(rect, 추론 좌표계)에 맞춰 키 좌표를 계산."""

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        self.x, self.y, self.width, self.height = x, y, width, height
        self.unit_w = width / UNITS_W
        self.unit_h = height / UNITS_H
        self.keys: Tuple[Key, ...] = BASE_KEYS
        self._by_code: Dict[KeyCode, Key] = {k.code: k for k in self.keys}

    @classmethod
    def from_calibration(cls, cal) -> "KeyboardLayout":
        return cls(cal.x, cal.y, cal.width, cal.height)

    def key(self, code: KeyCode) -> Key:
        return self._by_code[code]

    def key_rect(self, key: Key) -> Tuple[float, float, float, float]:
        return (self.x + key.ux * self.unit_w, self.y + key.uy * self.unit_h,
                key.uw * self.unit_w, key.uh * self.unit_h)

    def key_center(self, code: KeyCode) -> Vec2:
        x, y, w, h = self.key_rect(self.key(code))
        return Vec2(x + w / 2, y + h / 2)

    def hit_test(self, p: Vec2) -> Optional[Key]:
        for k in self.keys:
            x, y, w, h = self.key_rect(k)
            if x <= p.x < x + w and y <= p.y < y + h:
                return k
        return None

    def key_for_finger(self, finger: Finger, p: Vec2, snap_radius: float = 0.0) -> Optional[Key]:
        """손가락 끝 아래의 담당 키. 없으면 snap_radius(키 1칸 대비) 안에서 가장 가까운 담당 키.

        손끝이 키 경계에 걸치거나 캘리브레이션이 조금 어긋나도 인식되도록 한다.
        다른 손가락 담당 키 위에 확실히 올라가 있으면(그 키 중심에 더 가까우면) 스냅하지 않는다.
        """
        k = self.hit_test(p)
        if k is not None and finger in k.fingers:
            return k
        if snap_radius <= 0:
            return None
        best, best_d = None, float("inf")
        for key in self.keys:
            if finger not in key.fingers:
                continue
            x, y, w, h = self.key_rect(key)
            # 키 사각형까지의 거리 (키 1칸 크기로 정규화)
            dx = max(x - p.x, 0.0, p.x - (x + w)) / self.unit_w
            dy = max(y - p.y, 0.0, p.y - (y + h)) / self.unit_h
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best, best_d = key, d
        return best if best_d <= snap_radius else None

    def keys_for_finger(self, finger: Finger) -> List[Key]:
        return [k for k in self.keys if finger in k.fingers]

    def contains(self, p: Vec2, margin: float = 0.0) -> bool:
        return (self.x - margin <= p.x <= self.x + self.width + margin
                and self.y - margin <= p.y <= self.y + self.height + margin)

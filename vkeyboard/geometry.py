"""기본 2D 좌표 타입과 손 랜드마크 정의 (OpenCV 비의존)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def lerp(self, o: "Vec2", t: float) -> "Vec2":
        return Vec2(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)

    def as_int(self) -> tuple:
        return (int(round(self.x)), int(round(self.y)))


def dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------------------------
# MediaPipe 21점 손 랜드마크 인덱스
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass
class HandObservation:
    """추론 결과 한 손. landmarks 는 추론 좌표계(1280x720) 픽셀 좌표."""

    handedness: str                      # "Left" | "Right" (거울 모드 영상 기준 사용자 시점)
    landmarks: List[Vec2] = field(default_factory=list)
    confidence: float = 1.0              # 추적 신뢰도

    @property
    def is_left(self) -> bool:
        return self.handedness == "Left"


def hand_size(landmarks: Sequence[Vec2]) -> float:
    """손 크기 = 손목 ~ 중지 끝 거리."""
    if len(landmarks) < 21:
        return 0.0
    return dist(landmarks[WRIST], landmarks[MIDDLE_TIP])

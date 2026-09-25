"""좌표 안정화(이동 평균, One Euro Filter)와 이상치 제거."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Hashable, Iterable, Optional, Set, Tuple

from .geometry import Vec2, dist


class MovingAverage:
    """최근 N개 좌표의 이동 평균."""

    def __init__(self, size: int) -> None:
        self.size = max(1, int(size))
        self._buf: deque = deque(maxlen=self.size)

    def add(self, p: Vec2) -> Vec2:
        self._buf.append(p)
        return self.value()

    def value(self) -> Optional[Vec2]:
        if not self._buf:
            return None
        n = len(self._buf)
        return Vec2(sum(p.x for p in self._buf) / n, sum(p.y for p in self._buf) / n)

    def reset(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


class ScalarMovingAverage:
    def __init__(self, size: int) -> None:
        self._buf: deque = deque(maxlen=max(1, int(size)))

    def add(self, v: float) -> float:
        self._buf.append(v)
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()


def _smoothing_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro Filter (Casiez et al. 2012). 느릴 땐 강하게, 빠를 땐 약하게 필터링."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    def __call__(self, x: float, t: float) -> float:
        if self._t is None or self._x is None:
            self._x, self._t, self._dx = x, t, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            return self._x
        a_d = _smoothing_alpha(self.d_cutoff, dt)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _smoothing_alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0) -> None:
        self.fx = OneEuroFilter(min_cutoff, beta)
        self.fy = OneEuroFilter(min_cutoff, beta)

    def __call__(self, p: Vec2, t: float) -> Vec2:
        return Vec2(self.fx(p.x, t), self.fy(p.y, t))

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()


class PointGate:
    """갑자기 튀는 좌표 / 비정상적으로 빠른 움직임을 걸러낸다.

    연속으로 reset_frames 번 거부되면 손이 실제로 이동한 것으로 보고 새 위치로 재동기화한다.
    관측이 gap_reset 초 이상 끊겼다가 돌아오면 이전 위치와 비교하지 않는다.
    """

    def __init__(self, max_jump: float, max_speed: float, reset_frames: int = 4,
                 gap_reset: float = 0.25) -> None:
        self.max_jump = max_jump
        self.max_speed = max_speed
        self.reset_frames = reset_frames
        self.gap_reset = gap_reset
        self.reset()

    def reset(self) -> None:
        self._last: Optional[Vec2] = None
        self._last_t: Optional[float] = None
        self._rejects = 0

    def check(self, p: Vec2, t: float) -> Tuple[bool, str]:
        """(허용 여부, 사유). 사유: ok | first | jump | speed | reset"""
        if self._last is None or self._last_t is None or t - self._last_t > self.gap_reset:
            self._accept(p, t)
            return True, "first"
        d = dist(p, self._last)
        dt = t - self._last_t
        reason = "ok"
        if d > self.max_jump:
            reason = "jump"
        elif dt > 0 and d / dt > self.max_speed:
            reason = "speed"
        if reason == "ok":
            self._accept(p, t)
            return True, "ok"
        self._rejects += 1
        if self._rejects >= self.reset_frames:
            self._accept(p, t)
            return True, "reset"
        return False, reason

    def _accept(self, p: Vec2, t: float) -> None:
        self._last, self._last_t, self._rejects = p, t, 0


def is_near_edge(p: Vec2, width: float, height: float, margin: float) -> bool:
    return p.x < margin or p.y < margin or p.x > width - margin or p.y > height - margin


def find_overlapping(points: Dict[Hashable, Vec2], min_dist: float,
                     thresholds: Optional[Dict[Hashable, float]] = None) -> Set[Hashable]:
    """서로 min_dist 보다 가까운(겹친) 점들의 키 집합.

    thresholds 가 주어지면 두 점의 임계값 평균을 사용한다(손 크기별 보정).
    """
    keys = list(points.keys())
    result: Set[Hashable] = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            th = min_dist
            if thresholds is not None:
                th = 0.5 * (thresholds.get(a, min_dist) + thresholds.get(b, min_dist))
            if dist(points[a], points[b]) < th:
                result.add(a)
                result.add(b)
    return result


def mean_point(points: Iterable[Vec2]) -> Optional[Vec2]:
    pts = list(points)
    if not pts:
        return None
    return Vec2(sum(p.x for p in pts) / len(pts), sum(p.y for p in pts) / len(pts))

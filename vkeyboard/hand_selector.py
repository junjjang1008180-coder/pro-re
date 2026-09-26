"""내 손만 고르기: 화면에 보이는 여러 손 중 사용자 손 최대 2개만 입력에 사용한다.

손 모양만으로는 '누구 손'인지 알 수 없으므로 다음 단서를 쓴다.
1. 크기: 사용자는 카메라에 가장 가까워 손이 크게 보인다. 뒤에 있는 사람 손은 작다.
2. 등록: 양손을 펴서 ACTIVE 로 켤 때(= 확실히 사용자 손) 손 크기를 기억하고,
   이후엔 그 크기와 비슷한 손만 쓴다 (천천히 적응).
3. 연속성: 직전 프레임에 쓰던 손과 가까운 위치의 손을 우선한다.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .config import (HAND_SELECT_KEEP_DIST, HAND_SIZE_ADAPT, HAND_SIZE_MAX_RATIO, HAND_SIZE_MIN_RATIO,
                     SMALL_HAND_RATIO)
from .geometry import HandObservation, hand_size


def _center(hand: HandObservation) -> Tuple[float, float]:
    n = len(hand.landmarks)
    return (sum(p.x for p in hand.landmarks) / n, sum(p.y for p in hand.landmarks) / n)


def _palm_size(hand: HandObservation) -> float:
    """손 크기 = 손목~중지 끝. 손가락을 접으면 줄어드므로 손바닥 기준(손목~중지 MCP x 2)과 큰 쪽을 쓴다."""
    lm = hand.landmarks
    palm = ((lm[0].x - lm[9].x) ** 2 + (lm[0].y - lm[9].y) ** 2) ** 0.5
    return max(hand_size(lm), palm * 2.0)


class HandSelector:
    def __init__(self, enabled: bool = True, frame_width: float = 1280.0) -> None:
        self.enabled = enabled
        self.frame_width = frame_width
        self.ref_size: Optional[float] = None      # 등록된 사용자 손 크기
        self._last_centers: List[Tuple[float, float]] = []
        self._register_pending = False

    @property
    def registered(self) -> bool:
        return self.ref_size is not None

    def register(self, hands: Sequence[HandObservation]) -> bool:
        """지금 보이는 손(사용자 손)의 크기를 기억한다. 손이 없으면 False."""
        sizes = [_palm_size(h) for h in hands if len(h.landmarks) >= 21]
        if not sizes:
            return False
        self.ref_size = sum(sizes) / len(sizes)
        self._last_centers = [_center(h) for h in hands if len(h.landmarks) >= 21]
        self._register_pending = False
        return True

    def request_register(self) -> None:
        """다음에 손이 2개 보이는 프레임에서 등록 ('a' 키로 켰을 때 등)."""
        self._register_pending = True

    def reset(self) -> None:
        self.ref_size = None
        self._last_centers = []

    def select(self, hands: Sequence[HandObservation]) -> Tuple[List[HandObservation], List[HandObservation]]:
        """(사용할 손 최대 2개, 무시한 손) 을 반환."""
        cands = [h for h in hands if len(h.landmarks) >= 21]
        if not cands:
            return [], []
        if not self.enabled:
            chosen = sorted(cands, key=_palm_size, reverse=True)[:2]
            return chosen, [h for h in cands if h not in chosen]

        sizes = {id(h): _palm_size(h) for h in cands}
        if self.ref_size is not None:
            lo, hi = HAND_SIZE_MIN_RATIO * self.ref_size, HAND_SIZE_MAX_RATIO * self.ref_size
            pool = [h for h in cands if lo <= sizes[id(h)] <= hi]
            if not pool:
                pool = cands            # 사용자가 카메라에 크게 다가가거나 멀어진 경우: 아래에서 크기 기준으로
        else:
            pool = cands
        # 가장 큰 손보다 훨씬 작은 손(멀리 있는 사람)은 제외
        largest = max(sizes[id(h)] for h in pool)
        pool = [h for h in pool if sizes[id(h)] >= SMALL_HAND_RATIO * largest]

        keep = HAND_SELECT_KEEP_DIST * self.frame_width

        def score(h: HandObservation) -> Tuple[int, float]:
            if self._last_centers:
                cx, cy = _center(h)
                d = min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in self._last_centers)
                if d <= keep:
                    return (0, d)                     # 직전에 쓰던 손과 이어지는 손 우선
            return (1, -sizes[id(h)])                 # 그다음은 큰 손

        chosen = sorted(pool, key=score)[:2]
        ignored = [h for h in cands if h not in chosen]

        if chosen:
            self._last_centers = [_center(h) for h in chosen]
            if self._register_pending and len(chosen) == 2:
                self.register(chosen)
            elif self.ref_size is not None:
                mean = sum(sizes[id(h)] for h in chosen) / len(chosen)
                self.ref_size += HAND_SIZE_ADAPT * (mean - self.ref_size)   # 거리 변화에 천천히 적응
        return chosen, ignored

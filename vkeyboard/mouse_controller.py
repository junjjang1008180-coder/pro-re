"""오른손으로 마우스 커서 이동 + 핀치 클릭/더블클릭/드래그.

키보드와 완전히 분리된 독립 컨트롤러다. 오른손 검지가 키보드 영역 위에 있으면(타이핑 중)
마우스는 일시 정지(suspended)되어 커서 이동/클릭을 만들지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import (CLICK_MAX_DURATION, CLICK_RELEASE_FACTOR, DOUBLE_CLICK_MIN_INTERVAL,
                     DRAG_START_DISTANCE, MOUSE_REGION, ONE_EURO_BETA, ONE_EURO_MIN_CUTOFF,
                     PINCH_MIN_FRAMES, AppConfig, hand_scale)
from .events import MouseAction, MouseEvent
from .geometry import (INDEX_TIP, MIDDLE_TIP, THUMB_TIP, HandObservation, Vec2, dist, hand_size)
from .smoothing import OneEuroFilter2D


class PinchDetector:
    """히스테리시스 + N프레임 디바운스 핀치 판정."""

    def __init__(self, min_frames: int = PINCH_MIN_FRAMES) -> None:
        self.min_frames = min_frames
        self.closed = False
        self._count = 0
        self.closed_at = 0.0

    def reset(self) -> None:
        self.closed = False
        self._count = 0

    def update(self, distance: float, threshold: float, t: float, allow_close: bool = True) -> Optional[str]:
        """'press' | 'release' | None"""
        if not self.closed:
            if allow_close and distance < threshold:
                self._count += 1
                if self._count >= self.min_frames:
                    self.closed = True
                    self.closed_at = t
                    self._count = 0
                    return "press"
            else:
                self._count = 0
            return None
        if distance > threshold * CLICK_RELEASE_FACTOR:
            self._count += 1
            if self._count >= self.min_frames:
                self.closed = False
                self._count = 0
                return "release"
        else:
            self._count = 0
        return None


@dataclass
class MouseView:
    cursor: Optional[Tuple[int, int]] = None
    pointer: Optional[Vec2] = None      # 추론 좌표계 손가락 위치
    left_closed: bool = False
    right_closed: bool = False
    dragging: bool = False
    suspended: bool = True


class MouseController:
    def __init__(self, cfg: AppConfig, screen_size: Tuple[int, int],
                 region: Tuple[float, float, float, float] = MOUSE_REGION) -> None:
        self.cfg = cfg
        self.screen_w, self.screen_h = screen_size
        self.region = region
        self.left = PinchDetector()
        self.right = PinchDetector()
        self._filter = OneEuroFilter2D(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA)
        self.cursor: Optional[Tuple[int, int]] = None
        self.dragging = False
        self._down_center: Optional[Vec2] = None
        self._last_left_click = float("-inf")
        self._last_click_was_double = False
        self._last_right_click = float("-inf")
        self.view = MouseView()

    # ------------------------------------------------------------------
    def map_to_screen(self, p: Vec2) -> Tuple[float, float]:
        x0, y0, x1, y1 = self.region
        nx = (p.x / self.cfg.infer_width - x0) / (x1 - x0)
        ny = (p.y / self.cfg.infer_height - y0) / (y1 - y0)
        nx, ny = min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)
        return nx * (self.screen_w - 1), ny * (self.screen_h - 1)

    def update(self, hand: Optional[HandObservation], t: float, active: bool,
               suspended: bool = False) -> List[MouseEvent]:
        if hand is None or len(hand.landmarks) < 21:
            return self.process(None, None, None, 0.0, 0.0, t, active, suspended)
        lm = hand.landmarks
        return self.process(lm[INDEX_TIP], lm[THUMB_TIP], lm[MIDDLE_TIP], hand_size(lm),
                            hand.confidence, t, active, suspended)

    def _stop(self, t: float) -> List[MouseEvent]:
        events: List[MouseEvent] = []
        if self.dragging and self.cursor is not None:
            events.append(MouseEvent(MouseAction.DRAG_END, *self.cursor, t))
        self.dragging = False
        self.left.reset()
        self.right.reset()
        self._filter.reset()
        return events

    def process(self, index_tip: Optional[Vec2], thumb_tip: Optional[Vec2], middle_tip: Optional[Vec2],
                size: float, confidence: float, t: float, active: bool,
                suspended: bool = False) -> List[MouseEvent]:
        """테스트 가능한 핵심 로직 (랜드마크 3점만 사용)."""
        low_conf = confidence < self.cfg.min_tracking_confidence
        if not active or index_tip is None or thumb_tip is None or middle_tip is None or suspended or low_conf:
            events = self._stop(t)
            self.view = MouseView(self.cursor, index_tip, False, False, False, True)
            return events

        events: List[MouseEvent] = []
        scale = hand_scale(size, self.cfg.reference_hand_size)
        threshold = self.cfg.click_pinch_distance * scale
        d_left = dist(thumb_tip, index_tip)
        d_right = dist(thumb_tip, middle_tip)

        # 둘 다 가까우면 더 가까운 쪽만 핀치로 인정 (좌/우클릭 혼동 방지)
        left_ev = self.left.update(d_left, threshold, t,
                                   allow_close=not self.right.closed and d_left <= d_right)
        right_ev = self.right.update(d_right, threshold, t,
                                     allow_close=not self.left.closed and d_right < d_left)

        # 커서: 핀치(클릭 준비) 중엔 고정해 클릭 순간 흔들림 방지, 드래그 중엔 따라감
        f = self._filter(Vec2(*self.map_to_screen(index_tip)), t)
        new_cursor = (int(round(f.x)), int(round(f.y)))
        frozen = (self.left.closed or self.right.closed) and not self.dragging
        if not frozen and new_cursor != self.cursor:
            self.cursor = new_cursor
            events.append(MouseEvent(MouseAction.MOVE, *new_cursor, t, 0.0, size, confidence))
        elif self.cursor is None:
            self.cursor = new_cursor
        cx, cy = self.cursor

        # --- 왼쪽 핀치: 클릭 / 더블클릭 / 드래그 ---
        # 드래그 판정은 검지 끝 이동량으로 (핀치를 떼는 순간 엄지가 움직여도 드래그로 오인하지 않도록
        # 아직 핀치 거리 안에 있을 때만 판정)
        if left_ev == "press":
            self._down_center = index_tip
        if (self.left.closed and not self.dragging and self._down_center is not None
                and d_left < threshold):
            if dist(index_tip, self._down_center) >= DRAG_START_DISTANCE * scale:
                self.dragging = True
                events.append(MouseEvent(MouseAction.DRAG_START, cx, cy, t, d_left, size, confidence))
        if left_ev == "release":
            if self.dragging:
                self.dragging = False
                events.append(MouseEvent(MouseAction.DRAG_END, cx, cy, t, d_left, size, confidence))
            elif t - self.left.closed_at <= CLICK_MAX_DURATION:
                since = t - self._last_left_click
                if (not self._last_click_was_double and DOUBLE_CLICK_MIN_INTERVAL <= since
                        <= self.cfg.double_click_window):
                    self._last_left_click = t
                    self._last_click_was_double = True
                    events.append(MouseEvent(MouseAction.DOUBLE_CLICK, cx, cy, t, d_left, size, confidence))
                elif since >= self.cfg.click_cooldown:
                    self._last_left_click = t
                    self._last_click_was_double = False
                    events.append(MouseEvent(MouseAction.LEFT_CLICK, cx, cy, t, d_left, size, confidence))
            self._down_center = None

        # --- 오른쪽 핀치: 우클릭 ---
        if right_ev == "release" and t - self.right.closed_at <= CLICK_MAX_DURATION:
            if t - self._last_right_click >= self.cfg.click_cooldown:
                self._last_right_click = t
                events.append(MouseEvent(MouseAction.RIGHT_CLICK, cx, cy, t, d_right, size, confidence))

        self.view = MouseView(self.cursor, index_tip, self.left.closed, self.right.closed,
                              self.dragging, False)
        return events

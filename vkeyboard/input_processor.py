"""추론 스레드에서 실행되는 판정 단계: 손가락 추적 + 키/마우스/제스처 판정.

OpenCV/MediaPipe 에 의존하지 않으므로 시뮬레이션과 테스트에서 그대로 재사용한다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .calibration import Calibration
from .config import AppConfig
from .events import ModeEvent
from .finger_tracker import FingerTracker, normalize_handedness
from .geometry import INDEX_TIP, HandObservation
from .gesture_controller import GestureController
from .keyboard_controller import FingerView, KeyboardController
from .keyboard_layout import Finger, KeyboardLayout
from .mouse_controller import MouseController, MouseView


@dataclass
class Snapshot:
    """렌더링 스레드가 그릴 최신 판정 결과 (불변으로 취급)."""

    t: float
    hands: List[HandObservation]
    finger_views: Dict[Finger, FingerView]
    active: bool
    toggle_progress: float
    tracking_confidence: float
    mouse: MouseView
    calibration: Calibration
    frame_id: int = -1
    infer_fps: float = 0.0
    extra: dict = field(default_factory=dict)


class InputProcessor:
    def __init__(self, cfg: AppConfig, calibration: Calibration, screen_size: Tuple[int, int],
                 mouse_enabled: bool = True) -> None:
        self.cfg = cfg
        self.calibration = calibration
        self.layout = KeyboardLayout.from_calibration(calibration)
        self.fingers = FingerTracker(cfg)
        self.keyboard = KeyboardController(cfg, self.layout)
        self.mouse = MouseController(cfg, screen_size)
        self.gesture = GestureController(cfg.palm_toggle_hold)
        self.mouse_enabled = mouse_enabled
        self._pending_cal: Optional[Calibration] = None
        self._lock = threading.Lock()
        self._last_seen: Optional[float] = None

    def set_calibration(self, cal: Calibration) -> None:
        """다른 스레드(메인)에서 호출 가능. 다음 프레임에 반영된다."""
        with self._lock:
            self._pending_cal = cal

    def _apply_pending(self) -> None:
        with self._lock:
            cal, self._pending_cal = self._pending_cal, None
        if cal is not None:
            self.calibration = cal
            self.layout = KeyboardLayout.from_calibration(cal)
            self.keyboard.set_layout(self.layout)

    def process(self, hands: Sequence[HandObservation], t: float) -> Tuple[Snapshot, list]:
        self._apply_pending()
        split_x = self.layout.x + self.layout.width / 2 if self.cfg.handedness == "position" else None
        hands = normalize_handedness(hands, split_x)
        events: list = []

        toggled = self.gesture.update(hands, t)
        if toggled is not None:
            events.append(ModeEvent(toggled, "gesture", t))

        # 손 추적 실패: ACTIVE 중 손이 일정 시간 사라지면 강제 INACTIVE
        if hands:
            self._last_seen = t
        elif self.gesture.active and self._last_seen is not None and \
                t - self._last_seen > self.cfg.tracking_lost_timeout:
            if self.gesture.force_inactive():
                events.append(ModeEvent(False, "tracking_lost", t))

        active = self.gesture.active
        # 전환 제스처(양손 펼침) 중에는 키 입력을 만들지 않는다
        keys_active = active and self.gesture.progress == 0.0

        samples = self.fingers.update(hands, t)
        events.extend(self.keyboard.update(samples, t, keys_active))

        right = next((h for h in hands if h.handedness == "Right"), None)
        suspended = True
        if right is not None:
            margin = self.layout.unit_w * 0.5
            suspended = self.layout.contains(right.landmarks[INDEX_TIP], margin)
        events.extend(self.mouse.update(right, t, active and self.mouse_enabled,
                                        suspended or not self.mouse_enabled))

        conf = min((h.confidence for h in hands), default=0.0)
        snap = Snapshot(t=t, hands=list(hands), finger_views=dict(self.keyboard.views), active=active,
                        toggle_progress=self.gesture.progress, tracking_confidence=conf,
                        mouse=self.mouse.view, calibration=self.calibration)
        return snap, events

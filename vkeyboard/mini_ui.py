"""화면 표시 (랜드마크, 가상 키보드, 상태 패널, 미니 키보드/미니 마우스, 연습 모드).

모든 판정 좌표는 1280x720 추론 좌표계이므로 FrameScaler 로 캡처 해상도(4K)로 스케일업해 그린다.
OpenCV Hershey 폰트는 한글을 그릴 수 없어 화면 문구는 영어로 표시한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2

from .calibration import Calibration
from .events import MouseAction
from .frame_scaler import FrameScaler
from .geometry import HAND_CONNECTIONS, WRIST, Vec2
from .input_processor import Snapshot
from .input_state import FingerState
from .keyboard_layout import FINGER_ORDER, Finger, KeyboardLayout, KeyCode
from .practice_mode import PracticeResult, PracticeSession

FONT = cv2.FONT_HERSHEY_SIMPLEX

FINGER_COLORS: Dict[Finger, Tuple[int, int, int]] = {   # BGR
    Finger.LEFT_THUMB: (180, 180, 180), Finger.LEFT_INDEX: (255, 140, 0),
    Finger.LEFT_MIDDLE: (0, 200, 255), Finger.LEFT_RING: (200, 0, 200), Finger.LEFT_PINKY: (0, 180, 0),
    Finger.RIGHT_THUMB: (220, 220, 220), Finger.RIGHT_INDEX: (0, 110, 255),
    Finger.RIGHT_MIDDLE: (255, 255, 0), Finger.RIGHT_RING: (147, 20, 255), Finger.RIGHT_PINKY: (80, 255, 80),
}
HAND_COLORS = {"Left": (255, 160, 60), "Right": (60, 160, 255)}
STATE_COLORS = {FingerState.HOVER: (200, 200, 200), FingerState.PRESSING: (0, 220, 255),
                FingerState.PRESSED: (0, 255, 0), FingerState.RELEASED: (255, 200, 0)}
WHITE, BLACK, RED, GREEN = (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 220, 0)


def short_name(f: Finger) -> str:
    return ("L." if f.hand == "Left" else "R.") + f.digit


@dataclass
class RenderInfo:
    now: float
    display_fps: float = 0.0
    capture_fps: float = 0.0
    capture_size: Tuple[int, int] = (0, 0)
    test_mode: bool = True
    real_input: bool = False
    backend_name: str = "null"
    last_key: str = "-"
    last_mouse: str = "-"
    key_highlights: Dict[KeyCode, float] = field(default_factory=dict)   # key -> 하이라이트 종료 시각
    mouse_flash: Optional[Tuple[MouseAction, float]] = None             # (동작, 종료 시각)
    calibration_mode: bool = False
    calibration: Optional[Calibration] = None
    practice: Optional[PracticeSession] = None
    practice_result: Optional[PracticeResult] = None
    message: str = ""


class Renderer:
    def __init__(self, scaler: FrameScaler) -> None:
        self.scaler = scaler

    # --- 유틸 ---------------------------------------------------------------
    def _s(self, canvas) -> float:
        return canvas.shape[0] / 720.0

    def _text(self, canvas, text: str, org, scale: float, color=WHITE, thick: int = 1, bg: bool = True):
        s = self._s(canvas)
        fs, th = scale * s, max(1, int(round(thick * s)))
        x, y = int(org[0]), int(org[1])
        if bg:
            (tw, tht), base = cv2.getTextSize(text, FONT, fs, th)
            pad = int(3 * s)
            cv2.rectangle(canvas, (x - pad, y - tht - pad), (x + tw + pad, y + base + pad), BLACK, -1)
        cv2.putText(canvas, text, (x, y), FONT, fs, color, th, cv2.LINE_AA)

    def _rect(self, x: float, y: float, w: float, h: float):
        p0 = self.scaler.infer_to_display_pt(Vec2(x, y))
        p1 = self.scaler.infer_to_display_pt(Vec2(x + w, y + h))
        return p0, p1

    # --- 메인 ---------------------------------------------------------------
    def draw(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        cal = info.calibration or (snap.calibration if snap else None)
        layout = KeyboardLayout.from_calibration(cal) if cal else None
        if layout is not None:
            self.draw_keyboard(canvas, layout, snap, info)
        if snap is not None:
            self.draw_hands(canvas, snap)
            self.draw_finger_labels(canvas, snap)
            self.draw_mouse_pointer(canvas, snap)
        self.draw_status(canvas, snap, info)
        self.draw_finger_panel(canvas, snap)
        self.draw_mini_keyboard(canvas, snap, info)
        self.draw_mini_mouse(canvas, snap, info)
        if info.practice is not None:
            self.draw_practice(canvas, info)

    # --- 가상 키보드 ----------------------------------------------------------
    def draw_keyboard(self, canvas, layout: KeyboardLayout, snap: Optional[Snapshot], info: RenderInfo) -> None:
        s = self._s(canvas)
        (x0, y0), (x1, y1) = self._rect(layout.x, layout.y, layout.width, layout.height)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return
        selected: Dict[KeyCode, Finger] = {}
        pressing: Dict[KeyCode, FingerState] = {}
        if snap is not None:
            for f, v in snap.finger_views.items():
                if v.selected_key is not None:
                    selected[v.selected_key] = f
                    if v.state in (FingerState.PRESSING, FingerState.PRESSED):
                        pressing[v.selected_key] = v.state

        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        for k in layout.keys:
            (kx0, ky0), (kx1, ky1) = self._rect(*layout.key_rect(k))
            gap = int(2 * s)
            p0 = (kx0 - x0 + gap, ky0 - y0 + gap)
            p1 = (kx1 - x0 - gap, ky1 - y0 - gap)
            fill = (60, 60, 60)
            if info.key_highlights.get(k.code, 0) > info.now:
                fill = (255, 255, 255)                     # 확정 순간 ~100ms 밝게
            elif pressing.get(k.code) is FingerState.PRESSED:
                fill = (0, 170, 0)
            elif pressing.get(k.code) is FingerState.PRESSING:
                fill = (0, 170, 200)
            cv2.rectangle(overlay, p0, p1, fill, -1)
        cv2.addWeighted(overlay, 0.45, roi, 0.55, 0, roi)

        for k in layout.keys:
            (kx0, ky0), (kx1, ky1) = self._rect(*layout.key_rect(k))
            color, thick = (230, 230, 230), max(1, int(s))
            if k.code in selected:
                color, thick = FINGER_COLORS[selected[k.code]], max(2, int(3 * s))
            if info.calibration_mode:
                color = (0, 255, 255)
            cv2.rectangle(canvas, (kx0, ky0), (kx1, ky1), color, thick)
            fs = 0.45 if len(k.code.label) == 1 else 0.35
            self._text(canvas, k.code.label, (kx0 + 6 * s, ky1 - 8 * s), fs, WHITE, 1, bg=False)

    # --- 손 ------------------------------------------------------------------
    def draw_hands(self, canvas, snap: Snapshot) -> None:
        s = self._s(canvas)
        for hand in snap.hands:
            color = HAND_COLORS.get(hand.handedness, WHITE)
            pts = [self.scaler.infer_to_display_pt(p) for p in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], color, max(1, int(2 * s)), cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, p, max(2, int(3 * s)), WHITE, -1, cv2.LINE_AA)
            wx, wy = pts[WRIST]
            self._text(canvas, f"{hand.handedness} hand ({hand.confidence:.2f})",
                       (wx - 60 * s, wy + 24 * s), 0.5, color, 1)

    def draw_finger_labels(self, canvas, snap: Snapshot) -> None:
        s = self._s(canvas)
        for f, v in snap.finger_views.items():
            if v.tip is None:
                continue
            p = self.scaler.infer_to_display_pt(v.tip)
            color = STATE_COLORS[v.state] if v.valid else (0, 0, 180)
            cv2.circle(canvas, p, max(3, int(6 * s)), FINGER_COLORS[f], -1, cv2.LINE_AA)
            cv2.circle(canvas, p, max(4, int(8 * s)), color, max(1, int(2 * s)), cv2.LINE_AA)
            key = v.selected_key.label if v.selected_key else ""
            label = f"{short_name(f)} {key}" if v.valid else f"{short_name(f)} x{v.reason}"
            self._text(canvas, label, (p[0] - 20 * s, p[1] - 14 * s), 0.35, color, 1)

    def draw_mouse_pointer(self, canvas, snap: Snapshot) -> None:
        mv = snap.mouse
        if mv.suspended or mv.pointer is None:
            return
        s = self._s(canvas)
        p = self.scaler.infer_to_display_pt(mv.pointer)
        color = (0, 140, 255) if mv.dragging else ((0, 255, 255) if mv.left_closed or mv.right_closed else WHITE)
        r = int(14 * s)
        cv2.line(canvas, (p[0] - r, p[1]), (p[0] + r, p[1]), color, max(1, int(2 * s)))
        cv2.line(canvas, (p[0], p[1] - r), (p[0], p[1] + r), color, max(1, int(2 * s)))

    # --- 상태 패널 --------------------------------------------------------------
    def draw_status(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        s = self._s(canvas)
        x, y, dy = 12 * s, 30 * s, 24 * s
        active = snap.active if snap else False
        self._text(canvas, "ACTIVE" if active else "INACTIVE", (x, y), 0.8, GREEN if active else (0, 0, 230), 2)
        y += dy * 1.3
        if info.practice is not None:
            mode, mcol = "PRACTICE MODE (local scoring, no real input)", (0, 255, 255)
        elif info.real_input:
            mode, mcol = f"REAL INPUT ENABLED [{info.backend_name}]", RED
        else:
            mode, mcol = "TEST MODE (no real input)", (0, 255, 255)
        lines = [
            (mode, mcol),
            (f"FPS display {info.display_fps:4.1f} | capture {info.capture_fps:4.1f} | "
             f"infer {snap.infer_fps if snap else 0:4.1f}", WHITE),
            (f"Capture {info.capture_size[0]}x{info.capture_size[1]} -> infer "
             f"{self.scaler.infer_width}x{self.scaler.infer_height}", (200, 200, 200)),
            (f"Tracking confidence: {snap.tracking_confidence if snap else 0:.2f}  hands: "
             f"{len(snap.hands) if snap else 0}", WHITE),
            (f"Last key: {info.last_key}", WHITE),
            (f"Last mouse: {info.last_mouse}", WHITE),
        ]
        for text, color in lines:
            self._text(canvas, text, (x, y), 0.5, color, 1)
            y += dy
        if snap is not None and snap.toggle_progress > 0:
            w = int(220 * s * snap.toggle_progress)
            cv2.rectangle(canvas, (int(x), int(y)), (int(x + 220 * s), int(y + 12 * s)), WHITE, max(1, int(s)))
            cv2.rectangle(canvas, (int(x), int(y)), (int(x) + w, int(y + 12 * s)), GREEN, -1)
            y += dy
            self._text(canvas, "Hold both palms open to toggle", (x, y), 0.45, GREEN, 1)
            y += dy
        if info.calibration_mode:
            for t in ("CALIBRATION: i/j/k/l move  +/- size  [ ] height",
                      "f = fit to index fingers (F/J)  s = save  r = reset  c = exit"):
                self._text(canvas, t, (x, y), 0.45, (0, 255, 255), 1)
                y += dy
        else:
            self._text(canvas, "ESC quit | c calibrate", (x, y), 0.4, (180, 180, 180), 1)
            y += dy
        if info.message:
            self._text(canvas, info.message, (x, y), 0.5, (0, 200, 255), 1)

    def draw_finger_panel(self, canvas, snap: Optional[Snapshot]) -> None:
        s = self._s(canvas)
        x = canvas.shape[1] - 250 * s
        y = 30 * s
        self._text(canvas, "Finger     State     Key", (x, y), 0.45, WHITE, 1)
        for f in FINGER_ORDER:
            y += 20 * s
            v = snap.finger_views.get(f) if snap else None
            if v is None:
                text, color = f"{f.value:<13} -", (120, 120, 120)
            else:
                key = v.selected_key.label if v.selected_key else "-"
                state = v.state.value if v.valid else f"({v.reason})"
                text, color = f"{f.value:<13}{state:<10}{key}", STATE_COLORS[v.state] if v.valid else (80, 80, 200)
            self._text(canvas, text, (x, y), 0.4, color, 1)

    # --- 미니 키보드 / 미니 마우스 ----------------------------------------------
    def draw_mini_keyboard(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        s = self._s(canvas)
        H, W = canvas.shape[:2]
        mw = int(300 * s)
        mh = int(mw / 13.5 * 4)
        x0, y0 = W - mw - int(12 * s), H - mh - int(12 * s)
        mini = KeyboardLayout(x0, y0, mw, mh)
        cv2.rectangle(canvas, (x0 - 4, y0 - 4), (x0 + mw + 4, y0 + mh + 4), (30, 30, 30), -1)
        selected: Dict[KeyCode, Finger] = {}
        if snap is not None:
            for f, v in snap.finger_views.items():
                if v.selected_key is not None and v.valid:
                    selected[v.selected_key] = f
        for k in mini.keys:
            kx, ky, kw, kh = mini.key_rect(k)
            p0, p1 = (int(kx) + 1, int(ky) + 1), (int(kx + kw) - 1, int(ky + kh) - 1)
            fill = (70, 70, 70)
            if info.key_highlights.get(k.code, 0) > info.now:
                fill = WHITE
            elif k.code in selected:
                fill = FINGER_COLORS[selected[k.code]]
            cv2.rectangle(canvas, p0, p1, fill, -1)
            if len(k.code.label) == 1:
                cv2.putText(canvas, k.code.label, (p0[0] + int(3 * s), p1[1] - int(4 * s)), FONT, 0.3 * s,
                            BLACK if fill != (70, 70, 70) else WHITE, max(1, int(s)), cv2.LINE_AA)

    def draw_mini_mouse(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        s = self._s(canvas)
        H, W = canvas.shape[:2]
        mw, mh = int(56 * s), int(84 * s)
        kb_w = int(300 * s)
        x0 = W - kb_w - int(12 * s) - mw - int(16 * s)
        y0 = H - mh - int(12 * s)
        mv = snap.mouse if snap else None
        body = (90, 90, 90) if (mv is None or mv.suspended) else (150, 150, 150)
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), body, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + mw, y0 + mh), WHITE, max(1, int(s)))
        half = mw // 2
        left_col = right_col = (60, 60, 60)
        flash = info.mouse_flash if info.mouse_flash and info.mouse_flash[1] > info.now else None
        if mv is not None and mv.left_closed:
            left_col = (0, 220, 255)
        if mv is not None and mv.right_closed:
            right_col = (0, 220, 255)
        if mv is not None and mv.dragging:
            left_col = (0, 140, 255)
        if flash:
            if flash[0] in (MouseAction.LEFT_CLICK, MouseAction.DOUBLE_CLICK):
                left_col = GREEN
            elif flash[0] is MouseAction.RIGHT_CLICK:
                right_col = GREEN
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + half - 1, y0 + int(mh * 0.4)), left_col, -1)
        cv2.rectangle(canvas, (x0 + half + 1, y0 + 2), (x0 + mw - 2, y0 + int(mh * 0.4)), right_col, -1)
        cv2.line(canvas, (x0 + half, y0), (x0 + half, y0 + int(mh * 0.4)), WHITE, max(1, int(s)))
        label = "off" if (mv is None or mv.suspended) else ("drag" if mv.dragging else "on")
        cv2.putText(canvas, label, (x0 + int(6 * s), y0 + mh - int(10 * s)), FONT, 0.4 * s, WHITE,
                    max(1, int(s)), cv2.LINE_AA)

    # --- 연습 모드 ------------------------------------------------------------
    def draw_practice(self, canvas, info: RenderInfo) -> None:
        s = self._s(canvas)
        p = info.practice
        W = canvas.shape[1]
        cx = int(W * 0.30)
        y = int(40 * s)
        if info.practice_result is not None and p.finished:
            self._text(canvas, "PRACTICE COMPLETE  (p = restart)", (cx, y), 0.7, GREEN, 2)
            for line in info.practice_result.summary_lines():
                y += int(28 * s)
                self._text(canvas, line, (cx, y), 0.6, WHITE, 1)
            return
        sentence = p.current_sentence
        done, rest = sentence[:p.position], sentence[p.position:]
        self._text(canvas, f"Type ({p.index + 1}/{len(p.sentences)}):", (cx, y), 0.55, (0, 255, 255), 1)
        y += int(34 * s)
        self._text(canvas, done, (cx, y), 0.8, GREEN, 2)
        (tw, _), _ = cv2.getTextSize(done, FONT, 0.8 * s, max(1, int(2 * s)))
        if rest:
            cur = rest[0] if rest[0] != " " else "_"
            self._text(canvas, cur, (cx + tw, y), 0.8, (0, 255, 255), 2)
            (cw, _), _ = cv2.getTextSize(cur, FONT, 0.8 * s, max(1, int(2 * s)))
            self._text(canvas, rest[1:], (cx + tw + cw, y), 0.8, (200, 200, 200), 2, bg=False)
        r = p.result(info.now)
        y += int(30 * s)
        last = f"  last: {p.last_result}" if p.last_result else ""
        self._text(canvas, f"acc {r.accuracy * 100:.1f}%  wpm {r.wpm:.1f}  keys {r.total_keystrokes}{last}",
                   (cx, y), 0.5, WHITE, 1)


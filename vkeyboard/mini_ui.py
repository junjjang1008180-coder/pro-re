"""화면 표시 (랜드마크, 가상 키보드, 상태 카드, 손가락 카드, 미니 키보드/마우스, 연습 모드).

모든 판정 좌표는 1280x720 추론 좌표계이므로 FrameScaler 로 캡처 해상도(4K)로 스케일업해 그린다.
UI 크기는 캔버스 높이 720 기준 단위(s)로 정의해 해상도와 무관하게 같은 비율로 보인다.
OpenCV Hershey 폰트는 한글을 그릴 수 없어 화면 문구는 영어로 표시한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import Calibration
from .events import MouseAction
from .frame_scaler import FrameScaler
from .geometry import HAND_CONNECTIONS, WRIST, Vec2
from .input_processor import Snapshot
from .input_state import FingerState
from .keyboard_layout import FINGER_ORDER, Finger, KeyboardLayout, KeyCode
from .practice_mode import PracticeResult, PracticeSession

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# ---------------------------------------------------------------------------
# 테마 (BGR)
# ---------------------------------------------------------------------------
PANEL = (38, 28, 22)          # 짙은 남색 카드
PANEL_ALPHA = 0.78
KEY_FILL = (46, 36, 30)
KEY_EDGE = (110, 96, 88)
TEXT = (245, 242, 240)
MUTED = (165, 155, 148)
FAINT = (105, 96, 90)
ACCENT = (220, 200, 40)       # 청록 (#28C8DC)
GOOD = (120, 215, 80)         # 초록 (#50D778)
WARN = (40, 175, 255)         # 주황 (#FFAF28)
BAD = (95, 90, 240)           # 빨강 (#F05A5F)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# 손가락 색은 "손가락 종류" 기준 (왼손/오른손 같은 손가락 = 같은 색)
_DIGIT_COLORS = {
    "thumb": (200, 190, 185),
    "index": (250, 165, 70),     # 파랑
    "middle": (200, 215, 60),    # 청록
    "ring": (230, 125, 175),     # 보라
    "pinky": (120, 140, 250),    # 코랄
}
FINGER_COLORS: Dict[Finger, Tuple[int, int, int]] = {f: _DIGIT_COLORS[f.digit] for f in Finger}
HAND_COLORS = {"Left": (250, 175, 90), "Right": (90, 170, 250)}
STATE_STYLE = {   # 상태 -> (표시 문자, 색)
    FingerState.HOVER: ("HOVER", MUTED),
    FingerState.PRESSING: ("PRESS", WARN),
    FingerState.PRESSED: ("DOWN", GOOD),
    FingerState.RELEASED: ("UP", ACCENT),
}
MOUSE_NAMES = {MouseAction.LEFT_CLICK: "Left click", MouseAction.RIGHT_CLICK: "Right click",
               MouseAction.DOUBLE_CLICK: "Double click", MouseAction.DRAG_START: "Drag",
               MouseAction.DRAG_END: "Drop", MouseAction.MOVE: "Move"}

MARGIN = 14      # 화면 가장자리 여백 (720 기준 px)
CARD_R = 10      # 카드 모서리 반경
RIGHT_W = 236    # 오른쪽 카드 폭


def short_name(f: Finger) -> str:
    return ("L " if f.hand == "Left" else "R ") + f.digit


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


# ---------------------------------------------------------------------------
# 그리기 기본 도구
# ---------------------------------------------------------------------------
def rounded_rect(img, x0: float, y0: float, x1: float, y1: float, r: float, color,
                 thickness: int = -1) -> None:
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        return
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if r == 0:
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness, AA)
        return
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
        return
    t = thickness
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, t, AA)
    cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, t, AA)
    cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, t, AA)
    cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, t, AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, t, AA)
    cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, t, AA)
    cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, t, AA)
    cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, t, AA)


def _clip(canvas, x0, y0, x1, y1):
    H, W = canvas.shape[:2]
    return max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))


def glass_panel(canvas, x0, y0, x1, y1, r, color=PANEL, alpha=PANEL_ALPHA) -> None:
    """반투명 둥근 카드 (카드 영역만 블렌딩해서 빠름)."""
    cx0, cy0, cx1, cy1 = _clip(canvas, x0, y0, x1, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    roi = canvas[cy0:cy1, cx0:cx1]
    overlay = roi.copy()
    rounded_rect(overlay, x0 - cx0, y0 - cy0, x1 - cx0 - 1, y1 - cy0 - 1, r, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


class Renderer:
    def __init__(self, scaler: FrameScaler) -> None:
        self.scaler = scaler
        self.s = 1.0

    # --- 단위 변환 / 텍스트 -------------------------------------------------
    def u(self, v: float) -> int:
        """720 기준 길이 -> 캔버스 픽셀."""
        return int(round(v * self.s))

    def th(self, v: float = 1.0) -> int:
        return max(1, int(round(v * self.s)))

    def text(self, canvas, txt: str, x: float, y: float, scale: float, color=TEXT, weight: float = 1.0,
             shadow: bool = False) -> None:
        fs = scale * self.s
        t = self.th(weight)
        org = (int(x), int(y))
        if shadow:
            cv2.putText(canvas, txt, (org[0] + self.th(), org[1] + self.th()), FONT, fs, BLACK, t + self.th(), AA)
        cv2.putText(canvas, txt, org, FONT, fs, color, t, AA)

    def text_w(self, txt: str, scale: float, weight: float = 1.0) -> int:
        (w, _), _ = cv2.getTextSize(txt, FONT, scale * self.s, self.th(weight))
        return w

    def text_h(self, scale: float) -> int:
        (_, h), _ = cv2.getTextSize("Ag", FONT, scale * self.s, self.th())
        return h

    def pill(self, canvas, txt: str, x: float, y: float, scale: float, fg, bg, pad: float = 6,
             weight: float = 1.0, align_right: bool = False) -> int:
        """둥근 태그. y 는 태그 윗변. 태그 폭을 반환."""
        tw, h = self.text_w(txt, scale, weight), self.text_h(scale)
        ph = h + self.u(pad)
        pw = tw + self.u(pad) * 2
        if align_right:
            x = x - pw
        rounded_rect(canvas, x, y, x + pw, y + ph, ph / 2, bg)
        self.text(canvas, txt, x + self.u(pad), y + ph / 2 + h / 2, scale, fg, weight)
        return pw

    def _rect(self, x: float, y: float, w: float, h: float):
        p0 = self.scaler.infer_to_display_pt(Vec2(x, y))
        p1 = self.scaler.infer_to_display_pt(Vec2(x + w, y + h))
        return p0, p1

    # --- 메인 ---------------------------------------------------------------
    def draw(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        self.s = canvas.shape[0] / 720.0
        cal = info.calibration or (snap.calibration if snap else None)
        if cal is not None:
            self.draw_keyboard(canvas, KeyboardLayout.from_calibration(cal), snap, info)
        if snap is not None:
            self.draw_hands(canvas, snap)
            self.draw_fingertips(canvas, snap)
            self.draw_mouse_pointer(canvas, snap)
        self.draw_status_card(canvas, snap, info)
        bottom = self.draw_finger_card(canvas, snap)
        self.draw_input_card(canvas, snap, info, bottom + self.u(10))
        if info.practice is not None:
            self.draw_practice(canvas, info)

    # --- 가상 키보드 ----------------------------------------------------------
    def draw_keyboard(self, canvas, layout: KeyboardLayout, snap: Optional[Snapshot], info: RenderInfo) -> None:
        (bx0, by0), (bx1, by1) = self._rect(layout.x, layout.y, layout.width, layout.height)
        pad = self.u(8)
        x0, y0, x1, y1 = _clip(canvas, bx0 - pad, by0 - pad, bx1 + pad, by1 + pad)
        if x1 <= x0 or y1 <= y0:
            return

        selected: Dict[KeyCode, Finger] = {}
        state_of: Dict[KeyCode, FingerState] = {}
        if snap is not None:
            for f, v in snap.finger_views.items():
                if v.selected_key is not None and v.valid:
                    selected[v.selected_key] = f
                    state_of[v.selected_key] = v.state

        gap = max(2, self.u(2.5))
        radius = self.u(6)
        rects = {k.code: self._rect(*layout.key_rect(k)) for k in layout.keys}

        # 1) 채우기: 키보드 영역 한 번만 블렌딩
        roi = canvas[y0:y1, x0:x1]
        overlay = roi.copy()
        rounded_rect(overlay, bx0 - pad - x0, by0 - pad - y0, bx1 + pad - x0, by1 + pad - y0, self.u(12), PANEL)
        for code, ((kx0, ky0), (kx1, ky1)) in rects.items():
            fill = KEY_FILL
            if info.key_highlights.get(code, 0) > info.now:
                fill = WHITE
            elif state_of.get(code) is FingerState.PRESSED:
                fill = GOOD
            elif state_of.get(code) is FingerState.PRESSING:
                fill = WARN
            elif code in selected:
                fill = tuple(int(c * 0.45 + k * 0.55) for c, k in zip(FINGER_COLORS[selected[code]], KEY_FILL))
            rounded_rect(overlay, kx0 + gap - x0, ky0 + gap - y0, kx1 - gap - x0, ky1 - gap - y0, radius, fill)
        cv2.addWeighted(overlay, 0.62, roi, 0.38, 0, roi)

        # 2) 테두리 + 라벨
        for code, ((kx0, ky0), (kx1, ky1)) in rects.items():
            p0, p1 = (kx0 + gap, ky0 + gap), (kx1 - gap, ky1 - gap)
            flashing = info.key_highlights.get(code, 0) > info.now
            if code in selected:
                rounded_rect(canvas, *p0, *p1, radius, FINGER_COLORS[selected[code]], self.th(2))
            elif info.calibration_mode:
                rounded_rect(canvas, *p0, *p1, radius, ACCENT, self.th(1))
            else:
                rounded_rect(canvas, *p0, *p1, radius, KEY_EDGE, self.th(1))
            label = code.label
            scale = 0.5 if len(label) == 1 else 0.36
            color = BLACK if flashing else (TEXT if code in selected else (225, 220, 215))
            tw, tht = self.text_w(label, scale), self.text_h(scale)
            if len(label) == 1:
                self.text(canvas, label, (p0[0] + p1[0]) / 2 - tw / 2, (p0[1] + p1[1]) / 2 + tht / 2, scale, color)
            else:
                self.text(canvas, label, p0[0] + self.u(7), p1[1] - self.u(7), scale, color)

        if info.calibration_mode:
            rounded_rect(canvas, bx0 - pad, by0 - pad, bx1 + pad, by1 + pad, self.u(12), ACCENT, self.th(2))
            hs = self.u(6)
            for cx, cy in ((bx0 - pad, by0 - pad), (bx1 + pad, by0 - pad), (bx0 - pad, by1 + pad),
                           (bx1 + pad, by1 + pad)):
                cv2.rectangle(canvas, (cx - hs, cy - hs), (cx + hs, cy + hs), ACCENT, -1)

    # --- 손 ------------------------------------------------------------------
    def draw_hands(self, canvas, snap: Snapshot) -> None:
        for hand in snap.hands:
            color = HAND_COLORS.get(hand.handedness, TEXT)
            pts = [self.scaler.infer_to_display_pt(p) for p in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                cv2.line(canvas, pts[a], pts[b], color, self.th(1.6), AA)
            for i, p in enumerate(pts):
                if i in (4, 8, 12, 16, 20):
                    continue
                cv2.circle(canvas, p, self.u(2.6), WHITE, -1, AA)
            wx, wy = pts[WRIST]
            tag = f"{'LEFT' if hand.handedness == 'Left' else 'RIGHT'}  {hand.confidence:.2f}"
            w = self.text_w(tag, 0.38) + self.u(12)
            self.pill(canvas, tag, wx - w / 2, wy + self.u(10), 0.38, BLACK, color)

    def draw_fingertips(self, canvas, snap: Snapshot) -> None:
        for f, v in snap.finger_views.items():
            if v.tip is None:
                continue
            p = self.scaler.infer_to_display_pt(v.tip)
            col = FINGER_COLORS[f]
            if not v.valid:
                cv2.circle(canvas, p, self.u(6), BAD, self.th(1.5), AA)
                continue
            ring = STATE_STYLE[v.state][1]
            cv2.circle(canvas, p, self.u(9), ring, self.th(2 if v.state is FingerState.HOVER else 3), AA)
            cv2.circle(canvas, p, self.u(5.5), col, -1, AA)
            if v.selected_key is not None:
                lbl = v.selected_key.label
                w = self.text_w(lbl, 0.4, 1.2) + self.u(12)
                self.pill(canvas, lbl, p[0] - w / 2, p[1] - self.u(32), 0.4, BLACK, col, weight=1.2)

    def draw_mouse_pointer(self, canvas, snap: Snapshot) -> None:
        mv = snap.mouse
        if mv.suspended or mv.pointer is None:
            return
        p = self.scaler.infer_to_display_pt(mv.pointer)
        color = WARN if mv.dragging else (GOOD if (mv.left_closed or mv.right_closed) else ACCENT)
        r, g = self.u(16), self.u(6)
        cv2.circle(canvas, p, r, color, self.th(2), AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(canvas, (p[0] + dx * g, p[1] + dy * g), (p[0] + dx * (r + g), p[1] + dy * (r + g)),
                     color, self.th(2), AA)

    # --- 상태 카드 (왼쪽 위) ---------------------------------------------------
    def draw_status_card(self, canvas, snap: Optional[Snapshot], info: RenderInfo) -> None:
        x0, y0 = self.u(MARGIN), self.u(MARGIN)
        w = self.u(262)
        px = self.u(14)
        row = self.u(21)
        active = snap.active if snap else False

        rows: List[Tuple[str, str, tuple]] = [
            ("FPS", f"{info.display_fps:.0f}  /  cam {info.capture_fps:.0f}  /  ai {snap.infer_fps if snap else 0:.0f}",
             TEXT),
            ("Camera", f"{info.capture_size[0]}x{info.capture_size[1]} -> "
                       f"{self.scaler.infer_width}x{self.scaler.infer_height}", TEXT),
            ("Tracking", f"{snap.tracking_confidence if snap else 0:.2f}   {len(snap.hands) if snap else 0} hands",
             GOOD if snap and snap.tracking_confidence >= 0.75 else (WARN if snap and snap.hands else MUTED)),
            ("Last key", info.last_key, TEXT),
            ("Mouse", info.last_mouse, TEXT),
        ]
        toggling = snap is not None and snap.toggle_progress > 0
        if info.calibration_mode:
            hints = ["i j k l  move    + -  size", "[ ]  height   F  fit to fingers", "S  save   R  reset   C  done"]
        else:
            hints = ["ESC  quit      C  calibrate"]
        h = self.u(16) + self.u(28) + self.u(14) + row * len(rows) + self.u(8)
        h += self.u(34) if toggling else 0
        h += self.u(17) * len(hints) + self.u(10)
        h += self.u(22) if info.message else 0
        glass_panel(canvas, x0, y0, x0 + w, y0 + h, self.u(CARD_R))

        # 헤더: 상태 필 + 모드 필
        y = y0 + self.u(14)
        status, scol = ("ACTIVE", GOOD) if active else ("INACTIVE", BAD)
        pw = self.pill(canvas, status, x0 + px, y, 0.5, BLACK, scol, pad=8, weight=1.6)
        if info.practice is not None:
            mode, mcol = "PRACTICE", ACCENT
        elif info.real_input:
            mode, mcol = "REAL INPUT", BAD
        else:
            mode, mcol = "TEST MODE", WARN
        mh = self.text_h(0.5) + self.u(8)
        mode_h = self.text_h(0.36) + self.u(8)
        self.pill(canvas, mode, x0 + px + pw + self.u(8), y + (mh - mode_h) / 2, 0.36, mcol, (60, 48, 40), pad=8)
        y += self.u(28) + self.u(14)
        cv2.line(canvas, (x0 + px, y - self.u(7)), (x0 + w - px, y - self.u(7)), (70, 60, 54), self.th(1), AA)

        for label, value, color in rows:
            y += row
            self.text(canvas, label, x0 + px, y - self.u(6), 0.38, MUTED)
            self.text(canvas, value, x0 + px + self.u(66), y - self.u(6), 0.4, color)
        y += self.u(8)

        if toggling:
            bar_w = w - 2 * px
            by = y + self.u(4)
            rounded_rect(canvas, x0 + px, by, x0 + px + bar_w, by + self.u(6), self.u(3), (70, 60, 54))
            rounded_rect(canvas, x0 + px, by, x0 + px + int(bar_w * snap.toggle_progress), by + self.u(6),
                         self.u(3), GOOD)
            self.text(canvas, "Hold both palms open", x0 + px, by + self.u(24), 0.38, GOOD)
            y += self.u(34)

        for hint in hints:
            y += self.u(17)
            self.text(canvas, hint, x0 + px, y - self.u(4), 0.34, ACCENT if info.calibration_mode else FAINT)
        if info.message:
            y += self.u(22)
            self.text(canvas, info.message, x0 + px, y - self.u(6), 0.38, WARN)

    # --- 손가락 카드 (오른쪽 위) -------------------------------------------------
    def draw_finger_card(self, canvas, snap: Optional[Snapshot]) -> int:
        W = canvas.shape[1]
        w = self.u(RIGHT_W)
        x0, y0 = W - self.u(MARGIN) - w, self.u(MARGIN)
        px, row = self.u(14), self.u(19)
        h = self.u(36) + 2 * (self.u(20) + 5 * row) + self.u(6)
        glass_panel(canvas, x0, y0, x0 + w, y0 + h, self.u(CARD_R))
        self.text(canvas, "FINGERS", x0 + px, y0 + self.u(24), 0.42, TEXT, 1.4)
        self.text(canvas, "state      key", x0 + w - px - self.text_w("state      key", 0.34), y0 + self.u(24),
                  0.34, FAINT)
        y = y0 + self.u(36)
        for hand in ("Left", "Right"):
            y += self.u(20)
            self.text(canvas, hand.upper(), x0 + px, y - self.u(5), 0.34, HAND_COLORS[hand], 1.3)
            cv2.line(canvas, (x0 + px + self.u(46), y - self.u(9)), (x0 + w - px, y - self.u(9)), (70, 60, 54),
                     self.th(1), AA)
            for f in (f for f in FINGER_ORDER if f.hand == hand):
                y += row
                cy = y - self.u(9)
                v = snap.finger_views.get(f) if snap else None
                cv2.circle(canvas, (x0 + px + self.u(5), cy), self.u(4.5), FINGER_COLORS[f], -1, AA)
                self.text(canvas, f.digit, x0 + px + self.u(16), y - self.u(5), 0.38,
                          TEXT if v is not None and v.valid else MUTED)
                sx = x0 + self.u(118)
                if v is None or v.reason == "missing":
                    self.text(canvas, "--", sx, y - self.u(5), 0.36, FAINT)
                elif not v.valid:
                    self.text(canvas, v.reason.replace("_", " ")[:8], sx, y - self.u(5), 0.34, BAD)
                else:
                    name, col = STATE_STYLE[v.state]
                    ph = self.text_h(0.32) + self.u(6)
                    self.pill(canvas, name, sx, cy - ph / 2, 0.32, BLACK if v.state is not FingerState.HOVER
                              else TEXT, col if v.state is not FingerState.HOVER else (72, 62, 56), pad=6)
                key = v.selected_key.label if (v is not None and v.selected_key) else ""
                if key:
                    self.text(canvas, key, x0 + w - px - self.text_w(key, 0.4, 1.3), y - self.u(5), 0.4,
                              FINGER_COLORS[f], 1.3)
        return y0 + h

    # --- 입력 카드: 미니 키보드 + 미니 마우스 -------------------------------------
    def draw_input_card(self, canvas, snap: Optional[Snapshot], info: RenderInfo, top: int) -> None:
        W = canvas.shape[1]
        w = self.u(RIGHT_W)
        x0 = W - self.u(MARGIN) - w
        px = self.u(12)
        mouse_w, mouse_h = self.u(40), self.u(60)
        kb_w = w - 2 * px - mouse_w - self.u(12)
        kb_h = int(kb_w / 13.5 * 4)
        h = self.u(34) + max(kb_h, mouse_h) + self.u(24)
        glass_panel(canvas, x0, top, x0 + w, top + h, self.u(CARD_R))
        self.text(canvas, "INPUT", x0 + px, top + self.u(22), 0.42, TEXT, 1.4)

        # 미니 키보드
        kx, ky = x0 + px, top + self.u(34)
        mini = KeyboardLayout(kx, ky, kb_w, kb_h)
        selected: Dict[KeyCode, Finger] = {}
        if snap is not None:
            for f, v in snap.finger_views.items():
                if v.selected_key is not None and v.valid:
                    selected[v.selected_key] = f
        g = max(1, self.u(1))
        for k in mini.keys:
            x, y, kw, kh = mini.key_rect(k)
            fill = (70, 60, 54)
            if info.key_highlights.get(k.code, 0) > info.now:
                fill = WHITE
            elif k.code in selected:
                fill = FINGER_COLORS[selected[k.code]]
            rounded_rect(canvas, x + g, y + g, x + kw - g, y + kh - g, self.u(2), fill)
        self.text(canvas, f"key  {info.last_key}", kx, ky + kb_h + self.u(16), 0.34, MUTED)

        # 미니 마우스
        mx0 = x0 + w - px - mouse_w
        my0 = top + self.u(30)
        mv = snap.mouse if snap else None
        off = mv is None or mv.suspended
        body = (62, 54, 48) if off else (96, 84, 76)
        r = mouse_w // 2
        rounded_rect(canvas, mx0, my0, mx0 + mouse_w, my0 + mouse_h, r, body)
        split_y = my0 + int(mouse_h * 0.42)
        flash = info.mouse_flash if info.mouse_flash and info.mouse_flash[1] > info.now else None
        left_col = right_col = None
        if mv is not None and not off:
            if mv.left_closed:
                left_col = ACCENT
            if mv.right_closed:
                right_col = ACCENT
            if mv.dragging:
                left_col = WARN
        if flash:
            if flash[0] in (MouseAction.LEFT_CLICK, MouseAction.DOUBLE_CLICK):
                left_col = GOOD
            elif flash[0] is MouseAction.RIGHT_CLICK:
                right_col = GOOD
            elif flash[0] in (MouseAction.DRAG_START, MouseAction.DRAG_END):
                left_col = WARN
        mid = mx0 + mouse_w // 2
        if left_col is not None or right_col is not None:
            # 몸통 모양 마스크로 버튼 영역(위쪽 절반의 왼/오른쪽)만 칠한다
            mask = np.zeros((mouse_h + 1, mouse_w + 1), np.uint8)
            rounded_rect(mask, 0, 0, mouse_w, mouse_h, r, 255)
            top_h = split_y - my0
            for col, (a0, a1) in ((left_col, (0, mid - mx0)), (right_col, (mid - mx0, mouse_w + 1))):
                if col is None:
                    continue
                region = canvas[my0:my0 + top_h, mx0 + a0:mx0 + a1]
                m = mask[:region.shape[0], a0:a0 + region.shape[1]] > 0
                region[m] = col
        cv2.line(canvas, (mx0 + self.u(3), split_y), (mx0 + mouse_w - self.u(3), split_y), (40, 32, 28),
                 self.th(1.5), AA)
        cv2.line(canvas, (mid, my0 + self.u(3)), (mid, split_y), (40, 32, 28), self.th(1.5), AA)
        rounded_rect(canvas, mid - self.u(2.5), my0 + self.u(8), mid + self.u(2.5), my0 + self.u(18), self.u(2.5),
                     (180, 170, 160))
        status = "off" if off else ("drag" if mv.dragging else "on")
        stw = self.text_w(status, 0.34)
        self.text(canvas, status, mx0 + mouse_w / 2 - stw / 2, my0 + mouse_h + self.u(15), 0.34,
                  FAINT if off else (WARN if status == "drag" else GOOD))

    # --- 연습 모드 (위쪽 가운데) ------------------------------------------------
    def draw_practice(self, canvas, info: RenderInfo) -> None:
        p = info.practice
        W = canvas.shape[1]
        w = self.u(480)
        x0 = W // 2 - w // 2
        y0 = self.u(MARGIN)
        px = self.u(18)

        if info.practice_result is not None and p.finished:
            lines = info.practice_result.summary_lines()
            h = self.u(46) + self.u(24) * len(lines) + self.u(18)
            glass_panel(canvas, x0, y0, x0 + w, y0 + h, self.u(CARD_R))
            self.text(canvas, "PRACTICE COMPLETE", x0 + px, y0 + self.u(30), 0.55, GOOD, 1.6)
            hint = "P  restart"
            self.text(canvas, hint, x0 + w - px - self.text_w(hint, 0.36), y0 + self.u(30), 0.36, FAINT)
            y = y0 + self.u(46)
            for line in lines:
                y += self.u(24)
                self.text(canvas, line.strip(), x0 + px, y - self.u(6), 0.46, TEXT)
            return

        h = self.u(128)
        glass_panel(canvas, x0, y0, x0 + w, y0 + h, self.u(CARD_R))
        self.text(canvas, f"PRACTICE  {p.index + 1}/{len(p.sentences)}", x0 + px, y0 + self.u(26), 0.4, ACCENT, 1.4)

        sentence = p.current_sentence
        done, rest = sentence[:p.position], sentence[p.position:]
        scale = 0.78
        full_w = self.text_w(sentence, scale, 1.6)
        tx = x0 + (w - full_w) / 2
        ty = y0 + self.u(70)
        self.text(canvas, done, tx, ty, scale, GOOD, 1.6)
        cx = tx + self.text_w(done, scale, 1.6)
        if rest:
            cur = rest[0]
            cw = max(self.text_w(cur, scale, 1.6), self.text_w("n", scale, 1.6))
            rounded_rect(canvas, cx - self.u(2), ty - self.text_h(scale) - self.u(6), cx + cw + self.u(2),
                         ty + self.u(8), self.u(4), (70, 60, 54))
            cv2.line(canvas, (int(cx), int(ty + self.u(6))), (int(cx + cw), int(ty + self.u(6))), WARN, self.th(2), AA)
            self.text(canvas, cur, cx, ty, scale, WARN, 1.6)
            self.text(canvas, rest[1:], cx + self.text_w(cur, scale, 1.6), ty, scale, MUTED, 1.6)

        r = p.result(info.now)
        chips = [(f"ACC {r.accuracy * 100:.0f}%", TEXT), (f"WPM {r.wpm:.1f}", TEXT),
                 (f"KEYS {r.total_keystrokes}", TEXT)]
        if p.last_result == "correct":
            chips.append(("OK", GOOD))
        elif p.last_result == "wrong":
            chips.append(("MISS", BAD))
        cx = x0 + px
        cy = y0 + self.u(94)
        for label, col in chips:
            fg, bg = (BLACK, col) if col in (GOOD, BAD) else (col, (70, 60, 54))
            cx += self.pill(canvas, label, cx, cy, 0.38, fg, bg, pad=8) + self.u(8)


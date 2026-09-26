"""데스크톱 오버레이: 미니 키보드 / 미니 마우스 / ACTIVE / 한영 상태를 바탕화면 위에 항상 표시.

카메라 창(Virtual Keyboard)과 별개인 작은 창을 띄운다.
- Windows: 테두리 없음 + 항상 위 + 반투명 + 둥근 모서리(컬러키 투명) + 클릭 통과 + 포커스 안 뺏음
  (user32 SetWindowLongPtr / SetLayeredWindowAttributes / SetWindowPos 를 ctypes 로 호출)
- 그 외 OS: OpenCV 의 '항상 위' 속성만 적용한 일반 창
"""
from __future__ import annotations

import ctypes
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

from .frame_scaler import FrameScaler
from .input_processor import Snapshot
from .input_state import FingerState
from .keyboard_layout import HANGUL_JAMO, KeyCode
from .mini_ui import (ACCENT, BAD, BLACK, FINGER_COLORS, GOOD, MUTED, PANEL, TEXT, WARN, Renderer,
                      RenderInfo, rounded_rect, unicode_font)

WINDOW = "VK Overlay"
KEY_COLOR = (1, 0, 1)            # 투명하게 뺄 배경색 (BGR). 실제 UI 에는 쓰지 않는 색
BASE_W, BASE_H = 340, 172        # 배율 1.0 기준 크기 (px)
CORNERS = ("br", "bl", "tr", "tl")


class DesktopOverlay:
    def __init__(self, screen_size: Tuple[int, int], corner: str = "br", scale: float = 1.0,
                 alpha: float = 0.92) -> None:
        if corner not in CORNERS:
            raise ValueError(f"corner 는 {CORNERS} 중 하나")
        self.screen_w, self.screen_h = screen_size
        self.corner = corner
        self.scale = min(max(scale, 0.5), 3.0)
        scale = self.scale
        self.alpha = alpha
        self.w, self.h = int(BASE_W * scale), int(BASE_H * scale)
        # Renderer 는 캔버스 높이 720 을 기준 단위로 쓰므로, s 를 직접 지정해 픽셀 크기를 맞춘다
        self.renderer = Renderer(FrameScaler(self.w, self.h, self.w, self.h))
        self._styled = False
        self.win32_ok = False
        self.closed = False

    # ------------------------------------------------------------------
    def position(self) -> Tuple[int, int]:
        """작업 표시줄을 뺀 작업 영역의 모서리에 배치 (Windows). 그 외엔 화면 기준 + 여백."""
        area = _work_area() if sys.platform.startswith("win") else None
        m = int(12 * self.scale)
        if area is not None:
            left, top, right, bottom = area
        else:
            left, top, right, bottom = 0, 0, self.screen_w, self.screen_h - int(56 * self.scale)
        x = right - self.w - m if self.corner.endswith("r") else left + m
        y = bottom - self.h - m if self.corner.startswith("b") else top + m
        return max(0, x), max(0, y)

    def render(self, snap: Optional[Snapshot], info: RenderInfo):
        r = self.renderer
        r.s = self.scale                 # 1 단위 = 1px * scale
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = KEY_COLOR
        rounded_rect(img, 0, 0, self.w - 1, self.h - 1, r.u(14), PANEL)
        rounded_rect(img, 0, 0, self.w - 1, self.h - 1, r.u(14), (70, 60, 54), r.th(1))

        px = r.u(12)
        # 1) 헤더: ACTIVE / 모드 / 언어
        active = snap.active if snap else False
        y = r.u(10)
        status, scol = ("ACTIVE", GOOD) if active else ("INACTIVE", BAD)
        x = px + r.pill(img, status, px, y, 0.42, BLACK, scol, pad=7, weight=1.5) + r.u(6)
        if info.practice is not None:
            mode, mcol = "PRACTICE", ACCENT
        elif info.real_input:
            mode, mcol = "REAL", BAD
        else:
            mode, mcol = "TEST", WARN
        x += r.pill(img, mode, x, y + r.u(2), 0.34, mcol, (60, 48, 40), pad=6) + r.u(6)
        lang = "KO" if info.korean else "EN"
        lang_txt = f"{lang} 한" if info.korean else lang
        r.pill(img, lang_txt, self.w - px, y + r.u(2), 0.34, BLACK if info.korean else TEXT,
               ACCENT if info.korean else (72, 62, 56), pad=6, align_right=True)

        # 전환 진행 막대 (양손 펼침 중)
        top = r.u(40)
        if snap is not None and snap.toggle_progress > 0:
            bw = self.w - 2 * px
            rounded_rect(img, px, top - r.u(4), px + bw, top - r.u(1), r.u(2), (70, 60, 54))
            rounded_rect(img, px, top - r.u(4), px + int(bw * snap.toggle_progress), top - r.u(1), r.u(2), GOOD)

        # 2) 미니 키보드 + 미니 마우스
        mouse_w, mouse_h = r.u(34), r.u(52)
        kb_w = self.w - 2 * px - mouse_w - r.u(12)
        kb_h = r.mini_keyboard(img, px, top + r.u(4), kb_w, snap, info, labels=True)
        r.mini_mouse(img, self.w - px - mouse_w, top + r.u(2), mouse_w, mouse_h, snap, info)

        # 3) 지금 누르고 있는 키 / 마지막 입력 키 배지
        fy = top + r.u(4) + max(kb_h, mouse_h) + r.u(10)
        badge_h = r.u(22)
        pressing = []
        if snap is not None:
            for f, v in snap.finger_views.items():
                if v.valid and v.selected_key is not None and v.state in (FingerState.PRESSING, FingerState.PRESSED):
                    pressing.append((v.selected_key, v.state))
        x = px
        if pressing:
            for code, state in pressing[:4]:
                col = GOOD if state is FingerState.PRESSED else WARN
                x += r.pill(img, _key_text(code, info.korean), x, fy, 0.42, BLACK, col, pad=7, weight=1.5) + r.u(5)
        elif info.last_key_code is not None:
            recent = info.now - info.last_key_time < 0.6
            col = FINGER_COLORS.get(info.last_key_finger, TEXT) if info.last_key_finger else TEXT
            bg = col if recent else (72, 62, 56)
            fg = BLACK if recent else col
            x += r.pill(img, _key_text(info.last_key_code, info.korean), x, fy, 0.42, fg, bg, pad=7,
                        weight=1.5) + r.u(5)
            finger = info.last_key_finger.value.replace("_", " ") if info.last_key_finger else ""
            r.text(img, finger, x + r.u(2), fy + badge_h / 2 + r.text_h(0.32) / 2, 0.32, MUTED)
        else:
            r.text(img, "key  -", px, fy + badge_h / 2 + r.text_h(0.32) / 2, 0.32, MUTED)
        mouse_txt = _clip_text(info.last_mouse, 12)
        mw = r.text_w(mouse_txt, 0.34)
        r.text(img, mouse_txt, self.w - px - mw, fy + badge_h / 2 + r.text_h(0.34) / 2, 0.34, TEXT)
        return img

    # ------------------------------------------------------------------
    def show(self, snap: Optional[Snapshot], info: RenderInfo) -> None:
        if self.closed:
            return
        img = self.render(snap, info)
        if not self._styled:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW, img)
            cv2.waitKey(1)
            self._apply_style()
            self._styled = True
        cv2.imshow(WINDOW, img)

    def _apply_style(self) -> None:
        x, y = self.position()
        if sys.platform.startswith("win"):
            try:
                self.win32_ok = _style_win32(WINDOW, x, y, self.w, self.h, KEY_COLOR, self.alpha)
            except Exception as e:  # noqa: BLE001
                print(f"[경고] 오버레이 창 스타일 적용 실패 (일반 창으로 표시): {e}")
        if not self.win32_ok:
            try:
                cv2.moveWindow(WINDOW, x, y)
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1)
            except cv2.error:
                pass

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass


def _key_text(code: KeyCode, korean: bool) -> str:
    if code is KeyCode.HAN_ENG:
        return "한/영" if unicode_font() is not None else "Han/En"
    if korean and code in HANGUL_JAMO and unicode_font() is not None:
        return f"{HANGUL_JAMO[code]} {code.label}"
    return code.label


def _clip_text(txt: str, n: int) -> str:
    return txt if len(txt) <= n else txt[: n - 2] + ".."


# ---------------------------------------------------------------------------
# Windows 창 스타일
# ---------------------------------------------------------------------------
GWL_STYLE, GWL_EXSTYLE = -16, -20
WS_POPUP, WS_VISIBLE = 0x80000000, 0x10000000
WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x8, 0x20, 0x80
WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x80000, 0x08000000
LWA_COLORKEY, LWA_ALPHA = 0x1, 0x2
HWND_TOPMOST = -1
SWP_NOACTIVATE, SWP_FRAMECHANGED, SWP_SHOWWINDOW = 0x10, 0x20, 0x40


def _work_area() -> Optional[Tuple[int, int, int, int]]:
    """주 모니터에서 작업 표시줄을 제외한 영역 (left, top, right, bottom)."""
    from ctypes import wintypes

    try:
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
            return rect.left, rect.top, rect.right, rect.bottom
    except (AttributeError, OSError):
        pass
    return None


def _style_win32(title: str, x: int, y: int, w: int, h: int, key_bgr, alpha: float) -> bool:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowW.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR)
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.SetLayeredWindowAttributes.argtypes = (wintypes.HWND, wintypes.DWORD, wintypes.BYTE, wintypes.DWORD)
    user32.SetWindowPos.argtypes = (wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, wintypes.UINT)

    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return False
    # 제목 표시줄/테두리 제거
    user32.SetWindowLongPtrW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
    # 항상 위 + 반투명 + 클릭 통과 + 포커스 안 뺏음 + 작업표시줄에 안 보임
    ex = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    ex |= WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex)
    b, g, r = key_bgr
    colorref = r | (g << 8) | (b << 16)
    user32.SetLayeredWindowAttributes(hwnd, colorref, int(alpha * 255), LWA_COLORKEY | LWA_ALPHA)
    user32.SetWindowPos(hwnd, wintypes.HWND(HWND_TOPMOST), x, y, w, h,
                        SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_SHOWWINDOW)
    return True

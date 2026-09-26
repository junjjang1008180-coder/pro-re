"""OS 표준 입력 API 추상화 레이어 + 안전장치(게이트).

- Windows: user32.SendInput (ctypes)
- Linux:   libXtst XTestFake*Event (ctypes, X11)
- macOS:   CoreGraphics CGEvent (ctypes)

InputBackend 는 반드시 메인(렌더/입력) 스레드에서만 호출한다.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from .events import KeyEvent, MouseAction, MouseEvent
from .keyboard_layout import HANGUL_JAMO, KeyCode


class MouseButton(Enum):
    LEFT = "left"
    RIGHT = "right"


class InputBackend(ABC):
    name = "abstract"

    @abstractmethod
    def key_down(self, code: KeyCode) -> None: ...

    @abstractmethod
    def key_up(self, code: KeyCode) -> None: ...

    def tap_key(self, code: KeyCode) -> None:
        self.key_down(code)
        self.key_up(code)

    @abstractmethod
    def mouse_move(self, x: int, y: int) -> None: ...

    @abstractmethod
    def mouse_button(self, button: MouseButton, down: bool) -> None: ...

    def click(self, button: MouseButton, count: int = 1) -> None:
        for _ in range(count):
            self.mouse_button(button, True)
            self.mouse_button(button, False)

    @abstractmethod
    def screen_size(self) -> Tuple[int, int]: ...

    def close(self) -> None:
        pass


class NullInputBackend(InputBackend):
    """아무 것도 하지 않는 백엔드 (테스트 모드)."""

    name = "null"

    def __init__(self, screen: Tuple[int, int] = (1920, 1080)) -> None:
        self._screen = screen

    def key_down(self, code: KeyCode) -> None:
        pass

    def key_up(self, code: KeyCode) -> None:
        pass

    def mouse_move(self, x: int, y: int) -> None:
        pass

    def mouse_button(self, button: MouseButton, down: bool) -> None:
        pass

    def screen_size(self) -> Tuple[int, int]:
        return self._screen


class RecordingInputBackend(NullInputBackend):
    """호출을 기록만 하는 백엔드 (단위 테스트용)."""

    name = "recording"

    def __init__(self, screen: Tuple[int, int] = (1920, 1080)) -> None:
        super().__init__(screen)
        self.calls: List[tuple] = []

    def key_down(self, code: KeyCode) -> None:
        self.calls.append(("key_down", code))

    def key_up(self, code: KeyCode) -> None:
        self.calls.append(("key_up", code))

    def mouse_move(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))

    def mouse_button(self, button: MouseButton, down: bool) -> None:
        self.calls.append(("button", button, down))


# ===========================================================================
# Windows — SendInput
# ===========================================================================
_WIN_VK: Dict[KeyCode, int] = {**{KeyCode(c): ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
                               KeyCode.SEMICOLON: 0xBA, KeyCode.COMMA: 0xBC, KeyCode.PERIOD: 0xBE,
                               KeyCode.SLASH: 0xBF, KeyCode.SPACE: 0x20, KeyCode.TAB: 0x09,
                               KeyCode.CAPS_LOCK: 0x14, KeyCode.LEFT_SHIFT: 0xA0,
                               KeyCode.RIGHT_SHIFT: 0xA1, KeyCode.BACKSPACE: 0x08, KeyCode.ENTER: 0x0D,
                               KeyCode.HAN_ENG: 0x15}   # VK_HANGUL: 한국어 IME 한/영 전환


class WindowsInputBackend(InputBackend):
    name = "windows-sendinput"

    def __init__(self) -> None:
        from ctypes import wintypes

        ulong_ptr = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT, self._MOUSEINPUT, self._KEYBDINPUT = INPUT, MOUSEINPUT, KEYBDINPUT
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self._user32.SendInput.restype = wintypes.UINT
        self._user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
        self._user32.MapVirtualKeyW.restype = wintypes.UINT
        try:
            self._user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

    def _send(self, *inputs) -> None:
        arr = (self._INPUT * len(inputs))(*inputs)
        sent = self._user32.SendInput(len(inputs), arr, ctypes.sizeof(self._INPUT))
        if sent != len(inputs):
            raise OSError(f"SendInput 실패 (error={ctypes.get_last_error()}) — 관리자 권한 창에는 입력을 보낼 수 없습니다")

    def _key(self, code: KeyCode, up: bool):
        vk = _WIN_VK[code]
        # 스캔코드도 채워 둔다 (가상키만 보면 일부 앱/게임이 입력을 무시). 한/영 키는 가상키만 보낸다.
        scan = 0 if code is KeyCode.HAN_ENG else self._user32.MapVirtualKeyW(vk, 0) & 0xFF
        ki = self._KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=0x0002 if up else 0, time=0, dwExtraInfo=0)
        inp = self._INPUT(type=1)
        inp.u.ki = ki
        return inp

    def _mouse(self, flags: int, dx: int = 0, dy: int = 0):
        inp = self._INPUT(type=0)
        inp.u.mi = self._MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
        return inp

    def key_down(self, code: KeyCode) -> None:
        self._send(self._key(code, False))

    def key_up(self, code: KeyCode) -> None:
        self._send(self._key(code, True))

    def mouse_move(self, x: int, y: int) -> None:
        w, h = self.screen_size()
        ax = int(x * 65535 / max(1, w - 1))
        ay = int(y * 65535 / max(1, h - 1))
        self._send(self._mouse(0x0001 | 0x8000, ax, ay))  # MOVE | ABSOLUTE

    def mouse_button(self, button: MouseButton, down: bool) -> None:
        if button is MouseButton.LEFT:
            flag = 0x0002 if down else 0x0004
        else:
            flag = 0x0008 if down else 0x0010
        self._send(self._mouse(flag))

    def screen_size(self) -> Tuple[int, int]:
        return self._user32.GetSystemMetrics(0), self._user32.GetSystemMetrics(1)


# ===========================================================================
# Linux — X11 XTest
# ===========================================================================
_X11_KEYSYM: Dict[KeyCode, str] = {**{KeyCode(c): c.lower() for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
                                   KeyCode.SEMICOLON: "semicolon", KeyCode.COMMA: "comma",
                                   KeyCode.PERIOD: "period", KeyCode.SLASH: "slash",
                                   KeyCode.SPACE: "space", KeyCode.TAB: "Tab",
                                   KeyCode.CAPS_LOCK: "Caps_Lock", KeyCode.LEFT_SHIFT: "Shift_L",
                                   KeyCode.RIGHT_SHIFT: "Shift_R", KeyCode.BACKSPACE: "BackSpace",
                                   KeyCode.ENTER: "Return", KeyCode.HAN_ENG: "Hangul"}


class X11InputBackend(InputBackend):
    name = "x11-xtest"

    def __init__(self) -> None:
        xlib_path = ctypes.util.find_library("X11")
        xtst_path = ctypes.util.find_library("Xtst")
        if not xlib_path or not xtst_path:
            raise RuntimeError("libX11/libXtst 를 찾을 수 없습니다 (sudo apt install libxtst6)")
        self._x = ctypes.cdll.LoadLibrary(xlib_path)
        self._xt = ctypes.cdll.LoadLibrary(xtst_path)
        self._x.XOpenDisplay.restype = ctypes.c_void_p
        self._x.XOpenDisplay.argtypes = (ctypes.c_char_p,)
        self._x.XStringToKeysym.restype = ctypes.c_ulong
        self._x.XStringToKeysym.argtypes = (ctypes.c_char_p,)
        self._x.XKeysymToKeycode.restype = ctypes.c_ubyte
        self._x.XKeysymToKeycode.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        self._x.XFlush.argtypes = (ctypes.c_void_p,)
        self._x.XDefaultScreen.argtypes = (ctypes.c_void_p,)
        self._x.XDisplayWidth.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._x.XDisplayHeight.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self._x.XCloseDisplay.argtypes = (ctypes.c_void_p,)
        self._xt.XTestFakeKeyEvent.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong)
        self._xt.XTestFakeButtonEvent.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong)
        self._xt.XTestFakeMotionEvent.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_ulong)
        self._dpy = self._x.XOpenDisplay(None)
        if not self._dpy:
            raise RuntimeError("X 디스플레이를 열 수 없습니다 (DISPLAY 환경변수 확인, Wayland 는 XWayland 필요)")
        self._screen = self._x.XDefaultScreen(self._dpy)

    def _keycode(self, code: KeyCode) -> int:
        sym = self._x.XStringToKeysym(_X11_KEYSYM[code].encode())
        keycode = self._x.XKeysymToKeycode(self._dpy, sym)
        if keycode == 0:
            # 키맵에 없는 키(예: 한/영 키가 없는 배열)를 그대로 보내면 X 오류로 프로세스가 종료될 수 있다
            raise OSError(f"현재 X11 키보드 배열에 '{_X11_KEYSYM[code]}' 키가 없습니다")
        return keycode

    def key_down(self, code: KeyCode) -> None:
        self._xt.XTestFakeKeyEvent(self._dpy, self._keycode(code), 1, 0)
        self._x.XFlush(self._dpy)

    def key_up(self, code: KeyCode) -> None:
        self._xt.XTestFakeKeyEvent(self._dpy, self._keycode(code), 0, 0)
        self._x.XFlush(self._dpy)

    def mouse_move(self, x: int, y: int) -> None:
        self._xt.XTestFakeMotionEvent(self._dpy, self._screen, int(x), int(y), 0)
        self._x.XFlush(self._dpy)

    def mouse_button(self, button: MouseButton, down: bool) -> None:
        self._xt.XTestFakeButtonEvent(self._dpy, 1 if button is MouseButton.LEFT else 3, 1 if down else 0, 0)
        self._x.XFlush(self._dpy)

    def screen_size(self) -> Tuple[int, int]:
        return (self._x.XDisplayWidth(self._dpy, self._screen), self._x.XDisplayHeight(self._dpy, self._screen))

    def close(self) -> None:
        if self._dpy:
            self._x.XCloseDisplay(self._dpy)
            self._dpy = None


# ===========================================================================
# macOS — CoreGraphics CGEvent
# ===========================================================================
_MAC_VK: Dict[KeyCode, int] = {
    KeyCode.A: 0x00, KeyCode.S: 0x01, KeyCode.D: 0x02, KeyCode.F: 0x03, KeyCode.H: 0x04, KeyCode.G: 0x05,
    KeyCode.Z: 0x06, KeyCode.X: 0x07, KeyCode.C: 0x08, KeyCode.V: 0x09, KeyCode.B: 0x0B, KeyCode.Q: 0x0C,
    KeyCode.W: 0x0D, KeyCode.E: 0x0E, KeyCode.R: 0x0F, KeyCode.Y: 0x10, KeyCode.T: 0x11, KeyCode.O: 0x1F,
    KeyCode.U: 0x20, KeyCode.I: 0x22, KeyCode.P: 0x23, KeyCode.L: 0x25, KeyCode.J: 0x26, KeyCode.K: 0x28,
    KeyCode.N: 0x2D, KeyCode.M: 0x2E, KeyCode.SEMICOLON: 0x29, KeyCode.COMMA: 0x2B, KeyCode.SLASH: 0x2C,
    KeyCode.PERIOD: 0x2F, KeyCode.ENTER: 0x24, KeyCode.TAB: 0x30, KeyCode.SPACE: 0x31,
    KeyCode.BACKSPACE: 0x33, KeyCode.LEFT_SHIFT: 0x38, KeyCode.CAPS_LOCK: 0x39, KeyCode.RIGHT_SHIFT: 0x3C,
}


class _CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class MacInputBackend(InputBackend):
    name = "macos-cgevent"
    _K_CG_HID_EVENT_TAP = 0
    _MOUSE_TYPES = {  # (button, down) -> CGEventType
        (MouseButton.LEFT, True): 1, (MouseButton.LEFT, False): 2,
        (MouseButton.RIGHT, True): 3, (MouseButton.RIGHT, False): 4,
    }
    _K_CG_MOUSE_EVENT_CLICK_STATE = 1

    def __init__(self) -> None:
        path = ctypes.util.find_library("ApplicationServices")
        if not path:
            raise RuntimeError("ApplicationServices 프레임워크를 찾을 수 없습니다")
        cg = ctypes.cdll.LoadLibrary(path)
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        cg.CGEventCreateKeyboardEvent.argtypes = (ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool)
        cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
        cg.CGEventCreateMouseEvent.argtypes = (ctypes.c_void_p, ctypes.c_uint32, _CGPoint, ctypes.c_uint32)
        cg.CGEventPost.argtypes = (ctypes.c_uint32, ctypes.c_void_p)
        cg.CGEventSetIntegerValueField.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int64)
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayPixelsWide.argtypes = (ctypes.c_uint32,)
        cg.CGDisplayPixelsWide.restype = ctypes.c_size_t
        cg.CGDisplayPixelsHigh.argtypes = (ctypes.c_uint32,)
        cg.CGDisplayPixelsHigh.restype = ctypes.c_size_t
        cf.CFRelease.argtypes = (ctypes.c_void_p,)
        self._cg, self._cf = cg, cf
        self._pos = _CGPoint(0, 0)
        self._buttons_down: Set[MouseButton] = set()
        self.click_state = 1

    def _post(self, ev) -> None:
        if not ev:
            raise OSError("CGEvent 생성 실패 (손쉬운 사용 권한 확인)")
        self._cg.CGEventPost(self._K_CG_HID_EVENT_TAP, ev)
        self._cf.CFRelease(ev)

    _MAC_CONTROL = 0x3B

    def key_down(self, code: KeyCode) -> None:
        self._post(self._cg.CGEventCreateKeyboardEvent(None, _MAC_VK[code], True))

    def key_up(self, code: KeyCode) -> None:
        self._post(self._cg.CGEventCreateKeyboardEvent(None, _MAC_VK[code], False))

    def tap_key(self, code: KeyCode) -> None:
        if code is KeyCode.HAN_ENG:
            # macOS 에는 한/영 키코드가 없음 -> 기본 입력 소스 전환 단축키 Ctrl+Space
            cg = self._cg
            self._post(cg.CGEventCreateKeyboardEvent(None, self._MAC_CONTROL, True))
            self._post(cg.CGEventCreateKeyboardEvent(None, _MAC_VK[KeyCode.SPACE], True))
            self._post(cg.CGEventCreateKeyboardEvent(None, _MAC_VK[KeyCode.SPACE], False))
            self._post(cg.CGEventCreateKeyboardEvent(None, self._MAC_CONTROL, False))
            return
        super().tap_key(code)

    def mouse_move(self, x: int, y: int) -> None:
        self._pos = _CGPoint(float(x), float(y))
        if MouseButton.LEFT in self._buttons_down:
            ev_type = 6   # kCGEventLeftMouseDragged
        elif MouseButton.RIGHT in self._buttons_down:
            ev_type = 7   # kCGEventRightMouseDragged
        else:
            ev_type = 5   # kCGEventMouseMoved
        self._post(self._cg.CGEventCreateMouseEvent(None, ev_type, self._pos, 0))

    def mouse_button(self, button: MouseButton, down: bool) -> None:
        ev = self._cg.CGEventCreateMouseEvent(None, self._MOUSE_TYPES[(button, down)], self._pos,
                                              0 if button is MouseButton.LEFT else 1)
        if ev:
            self._cg.CGEventSetIntegerValueField(ev, self._K_CG_MOUSE_EVENT_CLICK_STATE, self.click_state)
        if down:
            self._buttons_down.add(button)
        else:
            self._buttons_down.discard(button)
        self._post(ev)

    def click(self, button: MouseButton, count: int = 1) -> None:
        # macOS 는 click state 필드로 더블클릭을 인식한다
        for i in range(count):
            self.click_state = i + 1
            self.mouse_button(button, True)
            self.mouse_button(button, False)
        self.click_state = 1

    def screen_size(self) -> Tuple[int, int]:
        d = self._cg.CGMainDisplayID()
        return int(self._cg.CGDisplayPixelsWide(d)), int(self._cg.CGDisplayPixelsHigh(d))


def create_platform_backend() -> InputBackend:
    """OS 를 감지해 해당 API 백엔드를 생성."""
    if sys.platform.startswith("win"):
        return WindowsInputBackend()
    if sys.platform == "darwin":
        return MacInputBackend()
    if sys.platform.startswith("linux"):
        return X11InputBackend()
    raise RuntimeError(f"지원하지 않는 OS: {sys.platform}")


def query_screen_size(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """입력을 보내지 않고 화면 크기만 조회 (테스트 모드에서도 커서 좌표 표시용)."""
    try:
        if sys.platform.startswith("win"):
            u = ctypes.windll.user32
            return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
        backend = create_platform_backend()
        try:
            return backend.screen_size()
        finally:
            backend.close()
    except Exception:  # noqa: BLE001 — 조회 실패 시 기본값
        return default


def play_feedback_beep() -> None:
    """OS 기본 비프음 (비동기). 지원하지 않으면 무시."""
    try:
        if sys.platform.startswith("win"):
            import winsound

            winsound.MessageBeep(winsound.MB_OK)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass


# ===========================================================================
# 안전장치 게이트
# ===========================================================================
class InputDispatcher:
    """키/마우스 이벤트를 받아, 안전 조건이 모두 만족될 때만 InputBackend 로 전달.

    실제 입력 조건: TEST_MODE=false AND ENABLE_REAL_INPUT=true AND ACTIVE AND 비상정지 아님
    """

    def __init__(self, backend: InputBackend, real_input_allowed: bool) -> None:
        self.backend = backend
        self.real_input_allowed = real_input_allowed
        self.active = False
        self.stopped = False
        self.stop_reason = ""
        self.pending_shift: Optional[KeyCode] = None
        self._held_buttons: Set[MouseButton] = set()
        self._held_keys: Set[KeyCode] = set()
        self.korean = False            # 한/영 키로 전환한 현재 언어 (화면 표시용)
        self.last_key_text = "-"
        self.last_mouse_text = "-"
        self.sent_count = 0
        self.error_count = 0           # OS 입력 전송 실패 횟수 (예: 관리자 권한 창에 포커스)
        self.last_error = ""
        self._last_error_print = float("-inf")

    def _safe(self, fn, *args) -> bool:
        """OS 입력 전송. 실패해도 예외를 올리지 않는다 (입력 한 번 실패로 프로그램이 죽지 않게)."""
        try:
            fn(*args)
            return True
        except Exception as e:  # noqa: BLE001
            self.error_count += 1
            self.last_error = str(e)
            now = time.monotonic()
            if now - self._last_error_print > 3.0:
                self._last_error_print = now
                print(f"[경고] 실제 입력 전송 실패 (프로그램은 계속 동작): {e}")
            return False

    @property
    def can_send(self) -> bool:
        return self.real_input_allowed and self.active and not self.stopped

    def set_active(self, active: bool) -> None:
        if self.active and not active:
            self.release_all()
        self.active = active

    # --- 키보드 ---------------------------------------------------------
    def handle_key(self, ev: KeyEvent) -> bool:
        if ev.key.is_modifier:
            # Shift 는 원샷 수식키: 다음 문자 키 하나에만 적용
            self.pending_shift = None if self.pending_shift else ev.key
            self.last_key_text = "Shift (next key)" if self.pending_shift else "Shift off"
            return False
        if ev.key is KeyCode.HAN_ENG:
            self.korean = not self.korean
            self.last_key_text = "KO" if self.korean else "EN"
        shift = self.pending_shift if ev.key.char is not None else None
        self.pending_shift = None
        if ev.key is not KeyCode.HAN_ENG:
            label = HANGUL_JAMO.get(ev.key, ev.key.label) if self.korean else ev.key.label
            self.last_key_text = f"Shift+{label}" if shift else label
        if not self.can_send:
            return False
        ok = self._safe(self._send_key, ev.key, shift)
        if ok:
            self.sent_count += 1
        return ok

    def _send_key(self, key: KeyCode, shift: Optional[KeyCode]) -> None:
        if shift:
            self.backend.key_down(shift)
            self._held_keys.add(shift)
        try:
            self.backend.tap_key(key)
        finally:
            if shift:
                self.backend.key_up(shift)
                self._held_keys.discard(shift)

    # --- 마우스 ---------------------------------------------------------
    def handle_mouse(self, ev: MouseEvent) -> bool:
        names = {MouseAction.MOVE: "Move", MouseAction.LEFT_CLICK: "Left click",
                 MouseAction.RIGHT_CLICK: "Right click", MouseAction.DOUBLE_CLICK: "Double click",
                 MouseAction.DRAG_START: "Drag start", MouseAction.DRAG_END: "Drop"}
        if ev.action is not MouseAction.MOVE or self.last_mouse_text in ("-", "Move"):
            self.last_mouse_text = names[ev.action]
        if not self.can_send:
            return False
        ok = self._safe(self._send_mouse, ev)
        if ok:
            self.sent_count += 1
        return ok

    def _send_mouse(self, ev: MouseEvent) -> None:
        b = self.backend
        if ev.action is MouseAction.MOVE:
            b.mouse_move(ev.x, ev.y)
        elif ev.action is MouseAction.LEFT_CLICK:
            b.mouse_move(ev.x, ev.y)
            b.click(MouseButton.LEFT)
        elif ev.action is MouseAction.DOUBLE_CLICK:
            # 첫 클릭은 이미 전송됨 -> 한 번 더 클릭해 OS 가 더블클릭으로 인식
            b.mouse_move(ev.x, ev.y)
            if isinstance(b, MacInputBackend):
                b.click_state = 2
                b.mouse_button(MouseButton.LEFT, True)
                b.mouse_button(MouseButton.LEFT, False)
                b.click_state = 1
            else:
                b.click(MouseButton.LEFT)
        elif ev.action is MouseAction.RIGHT_CLICK:
            b.mouse_move(ev.x, ev.y)
            b.click(MouseButton.RIGHT)
        elif ev.action is MouseAction.DRAG_START:
            b.mouse_move(ev.x, ev.y)
            b.mouse_button(MouseButton.LEFT, True)
            self._held_buttons.add(MouseButton.LEFT)
        elif ev.action is MouseAction.DRAG_END:
            b.mouse_move(ev.x, ev.y)
            if MouseButton.LEFT in self._held_buttons:
                b.mouse_button(MouseButton.LEFT, False)
                self._held_buttons.discard(MouseButton.LEFT)

    # --- 안전 정지 ------------------------------------------------------
    def release_all(self) -> None:
        """눌린 채로 남은 버튼/키를 모두 뗀다."""
        for btn in list(self._held_buttons):
            try:
                self.backend.mouse_button(btn, False)
            except Exception:  # noqa: BLE001 — 정지 경로에서는 절대 예외를 올리지 않음
                pass
        for key in list(self._held_keys):
            try:
                self.backend.key_up(key)
            except Exception:  # noqa: BLE001
                pass
        self._held_buttons.clear()
        self._held_keys.clear()
        self.pending_shift = None

    def emergency_stop(self, reason: str) -> None:
        if not self.stopped:
            self.release_all()
            self.stopped = True
            self.stop_reason = reason
            if self.real_input_allowed:
                print(f"[안전] 실제 입력 즉시 중지: {reason}")

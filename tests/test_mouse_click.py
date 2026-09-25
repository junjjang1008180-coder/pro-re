"""마우스 좌클릭 / 우클릭 / 더블클릭 / 드래그 / 클릭 쿨다운."""
import pytest
from conftest import P

from vkeyboard.config import AppConfig
from vkeyboard.events import MouseAction
from vkeyboard.mouse_controller import MouseController, PinchDetector

DT = 1 / 30
INDEX = P(640, 200)
MIDDLE = P(600, 190)
OPEN_THUMB = P(700, 280)          # 검지/중지와 멀리 떨어진 엄지


class Driver:
    def __init__(self, cfg=None):
        self.mc = MouseController(cfg or AppConfig(), (1920, 1080))
        self.t = 0.0
        self.events = []

    def frames(self, n, thumb=OPEN_THUMB, index=INDEX, middle=MIDDLE, active=True, suspended=False):
        for _ in range(n):
            self.events += self.mc.process(index, thumb, middle, 180.0, 0.95, self.t, active, suspended)
            self.t += DT

    def pinch_left(self, frames=4, index=INDEX):
        self.frames(frames, thumb=P(index.x + 5, index.y + 3), index=index)
        self.frames(3)

    def pinch_right(self, frames=4):
        self.frames(frames, thumb=P(MIDDLE.x + 4, MIDDLE.y + 4))
        self.frames(3)

    def actions(self):
        return [e.action for e in self.events if e.action is not MouseAction.MOVE]


def test_pinch_detector_debounces_single_frame_noise():
    p = PinchDetector(min_frames=2)
    assert p.update(10, 22, 0.0) is None
    assert p.update(40, 22, 0.03) is None          # 1프레임 노이즈 -> 무시
    assert p.update(10, 22, 0.06) is None
    assert p.update(10, 22, 0.1) == "press"
    assert p.update(25, 22, 0.13) is None          # 히스테리시스 (22*1.4 = 30.8 이하)
    assert p.update(40, 22, 0.16) is None
    assert p.update(40, 22, 0.2) == "release"


def test_cursor_moves_with_right_index():
    d = Driver()
    d.frames(3)
    moves = [e for e in d.events if e.action is MouseAction.MOVE]
    assert moves
    x, y = moves[-1].x, moves[-1].y
    assert x == pytest.approx(959.5, abs=2)         # 640/1280=0.5 -> 화면 중앙
    assert 0 <= y < 1080


def test_left_click():
    d = Driver()
    d.frames(5)
    d.pinch_left()
    assert d.actions() == [MouseAction.LEFT_CLICK]


def test_right_click():
    d = Driver()
    d.frames(5)
    d.pinch_right()
    assert d.actions() == [MouseAction.RIGHT_CLICK]


def test_double_click():
    d = Driver()
    d.frames(5)
    d.pinch_left(frames=3)
    d.pinch_left(frames=3)
    assert d.actions() == [MouseAction.LEFT_CLICK, MouseAction.DOUBLE_CLICK]


def test_slow_second_click_is_not_double_click():
    d = Driver()
    d.frames(5)
    d.pinch_left()
    d.frames(20)                                     # 0.66초 대기 > DOUBLE_CLICK_WINDOW
    d.pinch_left()
    assert d.actions() == [MouseAction.LEFT_CLICK, MouseAction.LEFT_CLICK]


def test_click_cooldown_after_double_click():
    d = Driver()
    d.frames(5)
    d.pinch_left(frames=3)
    d.pinch_left(frames=3)
    d.pinch_left(frames=3)                           # 더블클릭 직후 세 번째: 쿨다운으로 무시
    assert d.actions() == [MouseAction.LEFT_CLICK, MouseAction.DOUBLE_CLICK]


def test_right_click_cooldown():
    d = Driver()
    d.frames(5)
    d.pinch_right(frames=3)
    d.pinch_right(frames=3)                          # 0.4초 안에 두 번째 우클릭 -> 무시
    assert d.actions() == [MouseAction.RIGHT_CLICK]
    d.frames(15)
    d.pinch_right()
    assert d.actions() == [MouseAction.RIGHT_CLICK, MouseAction.RIGHT_CLICK]


def test_drag_and_drop():
    d = Driver()
    d.frames(5)
    idx = INDEX
    d.frames(3, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    for i in range(1, 11):                            # 핀치 유지한 채 오른쪽으로 이동
        idx = P(INDEX.x + i * 8, INDEX.y)
        d.frames(1, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    assert d.mc.dragging
    d.frames(3, index=idx)                            # 떼기 -> 드롭
    assert d.actions() == [MouseAction.DRAG_START, MouseAction.DRAG_END]
    drop = [e for e in d.events if e.action is MouseAction.DRAG_END][0]
    start = [e for e in d.events if e.action is MouseAction.DRAG_START][0]
    assert drop.x > start.x                           # 커서가 이동한 위치에서 드롭


def test_long_hold_without_move_is_not_click():
    d = Driver()
    d.frames(5)
    d.pinch_left(frames=30)                           # 1초 유지 -> 클릭 아님
    assert d.actions() == []


def test_inactive_produces_no_mouse_events():
    d = Driver()
    d.frames(5, active=False)
    d.frames(4, thumb=P(INDEX.x + 5, INDEX.y), active=False)
    d.frames(3, active=False)
    assert d.events == []


def test_suspended_over_keyboard_produces_no_events_and_ends_drag():
    d = Driver()
    d.frames(5)
    idx = INDEX
    d.frames(3, thumb=P(idx.x + 5, idx.y + 3))
    for i in range(1, 6):
        idx = P(INDEX.x + i * 10, INDEX.y)
        d.frames(1, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    assert d.mc.dragging
    n = len(d.events)
    d.frames(5, suspended=True)
    assert [e.action for e in d.events[n:]] == [MouseAction.DRAG_END]   # 안전하게 드롭 후 정지


def test_left_and_right_pinch_not_confused():
    """엄지가 검지 쪽에 더 가까우면 좌클릭만."""
    d = Driver()
    d.frames(5)
    d.frames(4, thumb=P(INDEX.x - 2, INDEX.y - 2), middle=P(INDEX.x - 15, INDEX.y - 5))
    d.frames(3)
    assert d.actions() == [MouseAction.LEFT_CLICK]


# ---------------------------------------------------------------- 마우스 모드 (가리키기 자세)
from vkeyboard.calibration import Calibration  # noqa: E402
from vkeyboard.gesture_controller import MouseModeDetector, is_pointing  # noqa: E402
from vkeyboard.input_processor import InputProcessor  # noqa: E402
from vkeyboard.simulation import open_palm_hand, pointing_hand, typing_hand  # noqa: E402


def _typing_right():
    tips = {"thumb": P(820, 600), "index": P(760, 520), "middle": P(800, 520), "ring": P(840, 520),
            "pinky": P(880, 520)}
    return typing_hand("Right", tips, {})


def test_pointing_pose_detection():
    assert is_pointing(pointing_hand("Right", 880, 600).landmarks)
    assert not is_pointing(open_palm_hand("Right", 880, 600).landmarks)   # 손바닥 펼침
    assert not is_pointing(_typing_right().landmarks)                      # 타이핑 자세


def test_mouse_mode_enter_and_exit_with_hysteresis():
    m = MouseModeDetector(enter_time=0.25, exit_time=0.4)
    point, typing = pointing_hand("Right", 880, 600), _typing_right()
    assert m.update(point, 0.0) is False
    assert m.update(point, 0.1) is False
    assert m.update(point, 0.3) is True               # 0.25초 유지 -> 마우스 모드
    assert m.update(typing, 0.4) is True              # 잠깐 풀어도 유지
    assert m.update(point, 0.5) is True
    assert m.update(typing, 0.6) is True
    assert m.update(typing, 1.05) is False            # 0.4초 넘게 풀면 타이핑 모드


def test_processor_mouse_mode_independent_of_keyboard_position(cfg):
    """키보드를 화면 위쪽 전체에 둬도 가리키기 자세면 마우스가 동작하고, 오른손은 키를 치지 않는다."""
    p = InputProcessor(cfg, Calibration(100, 150, 1080, 400), (1920, 1080))
    p.gesture.set_active(True)
    left = open_palm_hand("Left", 300, 650)
    events, t = [], 0.0
    for _ in range(12):
        snap, ev = p.process([left, pointing_hand("Right", 880, 600)], t)
        events += ev
        t += 1 / 30
    assert snap.mouse_mode and not snap.mouse.suspended
    assert any(getattr(e, "action", None) is MouseAction.MOVE for e in events)
    assert all(snap.finger_views[f].reason == "mouse" for f in snap.finger_views if f.hand == "Right")
    assert not any(hasattr(e, "key") and e.finger.hand == "Right" for e in events)


def test_default_keyboard_is_raised():
    c = Calibration.default()
    assert c.y + c.height < 720 * 0.85            # 화면 맨 아래에 붙지 않음

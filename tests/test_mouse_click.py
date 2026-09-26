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


def test_long_hold_starts_drag_without_moving():
    """핀치를 0.4초 이상 꾹 유지하면 움직이지 않아도 드래그 시작 (클릭 아님)."""
    d = Driver()
    d.frames(5)
    d.pinch_left(frames=30)
    assert d.actions() == [MouseAction.DRAG_START, MouseAction.DRAG_END]


def test_small_move_starts_drag():
    """12px 정도만 움직여도 드래그 시작 (예전 25px)."""
    d = Driver()
    d.frames(5)
    d.frames(3, thumb=P(INDEX.x + 5, INDEX.y + 3))
    idx = P(INDEX.x + 14, INDEX.y)
    d.frames(1, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    assert d.mc.dragging


def test_drag_survives_momentary_pinch_widening():
    """드래그 중 손을 빨리 움직여 엄지-검지가 잠깐 벌어져도 드롭되지 않는다."""
    d = Driver()
    d.frames(5)
    idx = INDEX
    d.frames(3, thumb=P(idx.x + 5, idx.y + 3))
    for i in range(1, 6):
        idx = P(INDEX.x + i * 10, INDEX.y)
        d.frames(1, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    assert d.mc.dragging
    # 핀치 거리 22*1.5=33px 로 2프레임 벌어짐 (클릭 기준이면 해제, 드래그 기준 1.8배=39.6px 이면 유지)
    d.frames(2, thumb=P(idx.x + 33, idx.y), index=idx)
    d.frames(3, thumb=P(idx.x + 5, idx.y + 3), index=idx)
    assert d.mc.dragging
    assert MouseAction.DRAG_END not in d.actions()


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


# ---------------------------------------------------------------- 마우스/키보드 모드 분리
from dataclasses import replace as _replace  # noqa: E402

from vkeyboard.config import AppConfig as _AppConfig  # noqa: E402
from vkeyboard.gesture_controller import is_mouse_hold  # noqa: E402
from vkeyboard.geometry import HandObservation  # noqa: E402


def _pinch_bent_index():
    """클릭 핀치 순간: 검지가 굽혀져 엄지에 닿음 (가리키기 자세는 깨지지만 약지·새끼는 접힌 상태)."""
    h = pointing_hand("Right", 880, 600)
    lm = list(h.landmarks)
    lm[8] = lm[5] + P(-5, -20)          # 굽힌 검지 끝
    lm[4] = lm[8] + P(4, 2)             # 엄지가 검지 끝에 닿음
    return HandObservation("Right", lm, h.confidence)


def _left_typing(press=0.0):
    tips = {"thumb": P(420, 600), "index": P(460, 470), "middle": P(420, 470), "ring": P(380, 470),
            "pinky": P(340, 470)}
    return typing_hand("Left", tips, {"index": press})


def test_pinch_with_bent_index_keeps_mouse_mode():
    assert not is_pointing(_pinch_bent_index().landmarks)
    assert is_mouse_hold(_pinch_bent_index().landmarks)
    m = MouseModeDetector(enter_time=0.25, exit_time=0.4)
    m.update(pointing_hand("Right", 880, 600), 0.0)
    assert m.update(pointing_hand("Right", 880, 600), 0.3)
    for i in range(30):                                   # 1초 동안 핀치 유지해도
        assert m.update(_pinch_bent_index(), 0.3 + i / 30)  # 마우스 모드 유지


def test_mouse_mode_exits_when_hand_lost():
    m = MouseModeDetector(enter_time=0.25, exit_time=0.4, lost_time=0.6)
    m.update(pointing_hand("Right", 880, 600), 0.0)
    assert m.update(pointing_hand("Right", 880, 600), 0.3)
    assert m.update(None, 0.5) is True                    # 잠깐 안 보임 -> 유지
    assert m.update(None, 1.2) is False                   # 0.6초 이상 -> 해제


def _run(p, frames, hands_fn, t0=0.0):
    events, t = [], t0
    for i in range(frames):
        snap, ev = p.process(hands_fn(i), t)
        events += ev
        t += 1 / 30
    return snap, events, t


def test_right_hand_keys_blocked_from_first_pointing_frame(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    snap, _, _ = _run(p, 1, lambda i: [_left_typing(), pointing_hand("Right", 880, 600)])
    assert not snap.mouse_mode                            # 아직 확정 전이지만
    assert all(v.reason == "mouse" for f, v in snap.finger_views.items() if f.hand == "Right")


def test_mouse_mode_pauses_left_hand_typing(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    point = pointing_hand("Right", 880, 600)
    _, _, t = _run(p, 12, lambda i: [_left_typing(), point])
    # 마우스 모드 중 왼손 검지로 키를 눌러도 입력 안 됨
    snap, events, _ = _run(p, 10, lambda i: [_left_typing(25.0 if i >= 3 else 0.0), point], t)
    assert snap.mouse_mode
    assert not [e for e in events if hasattr(e, "key")]
    assert all(v.reason == "mouse" for v in snap.finger_views.values())


def test_mouse_exclusive_false_keeps_left_hand_typing():
    cfg = _replace(_AppConfig(), mouse_exclusive=False)
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    point = pointing_hand("Right", 880, 600)
    snap, _, _ = _run(p, 12, lambda i: [_left_typing(), point])
    assert snap.mouse_mode
    assert all(v.reason != "mouse" for f, v in snap.finger_views.items() if f.hand == "Left")


def test_keys_blocked_briefly_after_leaving_mouse_mode(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    point = pointing_hand("Right", 880, 600)
    _, _, t = _run(p, 12, lambda i: [_left_typing(), point])
    right_typing = _typing_right()
    # 약지·새끼를 펴서 타이핑 자세로 복귀: 0.4초 뒤 해제, 그 뒤 0.4초는 여전히 차단
    snap, _, t = _run(p, 14, lambda i: [_left_typing(), right_typing], t)
    assert not snap.mouse_mode
    assert all(v.reason == "mouse" for v in snap.finger_views.values())
    snap, _, _ = _run(p, 15, lambda i: [_left_typing(), right_typing], t)
    assert all(v.reason != "mouse" for v in snap.finger_views.values())


def test_click_with_bent_index_pinch_produces_click_not_keys(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    point = pointing_hand("Right", 880, 600)
    _, events, t = _run(p, 12, lambda i: [_left_typing(), point])
    _, ev2, t = _run(p, 4, lambda i: [_left_typing(), _pinch_bent_index()], t)
    _, ev3, _ = _run(p, 4, lambda i: [_left_typing(), point], t)
    all_ev = events + ev2 + ev3
    assert [e.action for e in all_ev if getattr(e, "action", None) not in (None, MouseAction.MOVE)] == \
        [MouseAction.LEFT_CLICK]
    assert not [e for e in all_ev if hasattr(e, "key")]



# ---------------------------------------------------------------- 내 손만 인식
from vkeyboard.hand_selector import HandSelector  # noqa: E402


def _scaled_hand(label, cx, wrist_y, scale):
    """open_palm_hand 를 scale 배로 줄이거나 키운 손 (멀리 있는 사람 손 = 작게)."""
    base = open_palm_hand(label, cx, wrist_y)
    w = base.landmarks[0]
    lm = [P(w.x + (p.x - w.x) * scale, w.y + (p.y - w.y) * scale) for p in base.landmarks]
    return HandObservation(label, lm, base.confidence)


def test_selector_ignores_small_far_away_hand():
    sel = HandSelector(True, 1280)
    me_l, me_r = _scaled_hand("Left", 400, 600, 1.0), _scaled_hand("Right", 880, 600, 1.0)
    other = _scaled_hand("Right", 640, 300, 0.45)             # 뒤에 있는 사람 (작게 보임)
    chosen, ignored = sel.select([other, me_l, me_r])
    assert me_l in chosen and me_r in chosen and ignored == [other]


def test_selector_after_register_ignores_different_sized_hands():
    sel = HandSelector(True, 1280)
    me_l, me_r = _scaled_hand("Left", 400, 600, 1.0), _scaled_hand("Right", 880, 600, 1.0)
    assert sel.register([me_l, me_r])
    big = _scaled_hand("Right", 1100, 650, 2.0)               # 카메라 바로 앞에 불쑥 들어온 다른 사람 손
    chosen, ignored = sel.select([big, me_l, me_r])
    assert set(map(id, chosen)) == {id(me_l), id(me_r)} and ignored == [big]


def test_selector_prefers_continuity_between_similar_hands():
    sel = HandSelector(True, 1280)
    me_l, me_r = _scaled_hand("Left", 400, 600, 1.0), _scaled_hand("Right", 880, 600, 1.0)
    sel.register([me_l, me_r])
    other = _scaled_hand("Right", 640, 620, 1.05)             # 비슷한 크기의 다른 사람 손
    for _ in range(5):
        chosen, ignored = sel.select([other, me_l, me_r])
    assert set(map(id, chosen)) == {id(me_l), id(me_r)} and ignored == [other]


def test_selector_disabled_uses_largest_two():
    sel = HandSelector(False, 1280)
    a, b, c = (_scaled_hand("Left", 300, 600, 1.0), _scaled_hand("Right", 900, 600, 1.2),
               _scaled_hand("Right", 640, 300, 0.5))
    chosen, ignored = sel.select([c, a, b])
    assert set(map(id, chosen)) == {id(a), id(b)} and ignored == [c]


def test_processor_registers_on_activation_and_ignores_extra_hand(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    me = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    t = 0.0
    for _ in range(40):
        snap, _ = p.process(me, t)
        t += 1 / 30
    assert snap.active and snap.hand_registered
    stranger = _scaled_hand("Right", 640, 250, 0.4)
    snap, _ = p.process([stranger] + me, t)
    assert len(snap.hands) == 2 and len(snap.ignored_hands) == 1

"""키보드 쿨다운 / 한 번 누를 때 한 번 입력 (KeyboardController 수준)."""
from conftest import P, run_press, sample

from vkeyboard.calibration import Calibration
from vkeyboard.input_state import FingerStateMachine, PressParams
from vkeyboard.keyboard_controller import KeyboardController
from vkeyboard.keyboard_layout import Finger, KeyboardLayout, KeyCode

PRESS = [0, 0, 20, 20, 20, 20, 0, 0]
DT = 1 / 30


def make(cfg):
    layout = KeyboardLayout.from_calibration(Calibration.default())
    return KeyboardController(cfg, layout), layout


def test_state_machine_cooldown_blocks_fast_repress():
    m = FingerStateMachine(PressParams(18, 10, 3, 0.35))
    fires = []
    t = 0.0
    for d in [20, 20, 20, 0, 20, 20, 20, 0]:   # 두 번째 누름이 0.35초 안에 확정
        fires.append(m.update(d, t))
        t += DT
    assert sum(fires) == 1
    assert m.suppressed == 1


def test_state_machine_allows_after_cooldown():
    m = FingerStateMachine(PressParams(18, 10, 3, 0.35))
    assert sum(m.update(d, i * DT) for i, d in enumerate([20, 20, 20, 0])) == 1
    assert sum(m.update(d, 0.5 + i * DT) for i, d in enumerate([20, 20, 20, 0])) == 1


def test_suppressed_press_needs_release_even_after_cooldown():
    m = FingerStateMachine(PressParams(18, 10, 3, 0.35))
    for i, d in enumerate([20, 20, 20, 0, 20, 20, 20]):
        m.update(d, i * DT)
    # 쿨다운 중 확정된 누름은 버려지고, 계속 누르고 있어도 쿨다운 끝난 뒤 자동 입력되지 않음
    assert not any(m.update(20, 1.0 + i * DT) for i in range(10))


def test_controller_single_press_single_event(cfg):
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), PRESS)
    assert [e.key for e in ev] == [KeyCode.F]
    assert ev[0].finger is Finger.LEFT_INDEX


def test_controller_hold_does_not_repeat(cfg):
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), [0] + [25] * 90)
    assert len(ev) == 1


def test_controller_cooldown_between_taps(cfg):
    kc, layout = make(cfg)
    tip = layout.key_center(KeyCode.F)
    ev1, t = run_press(kc, Finger.LEFT_INDEX, tip, [0, 20, 20, 20, 0])
    ev2, t = run_press(kc, Finger.LEFT_INDEX, tip, [20, 20, 20, 0], t)          # 너무 빠른 재입력
    ev3, _ = run_press(kc, Finger.LEFT_INDEX, tip, [0] * 10 + [20, 20, 20, 0], t)  # 쿨다운 이후
    assert len(ev1) == 1 and len(ev2) == 0 and len(ev3) == 1


def test_same_key_cooldown_across_fingers(cfg):
    """양손 엄지가 거의 동시에 Space 를 눌러도 한 번만 입력."""
    kc, layout = make(cfg)
    sp = layout.key_rect(layout.key(KeyCode.SPACE))
    lt, rt = P(sp[0] + 30, sp[1] + 10), P(sp[0] + sp[2] - 30, sp[1] + 10)
    events = []
    for i, d in enumerate([0, 20, 20, 20, 20, 0]):
        events += kc.update({Finger.LEFT_THUMB: sample(Finger.LEFT_THUMB, lt, d),
                             Finger.RIGHT_THUMB: sample(Finger.RIGHT_THUMB, rt, d)}, i * DT, True)
    assert [e.key for e in events] == [KeyCode.SPACE]


def test_inactive_generates_no_events(cfg):
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), PRESS, active=False)
    assert ev == []


def test_wrong_finger_for_key_does_not_type(cfg):
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_RING, layout.key_center(KeyCode.F), PRESS)
    assert ev == []


def test_whole_hand_baseline_drift_absorbed(cfg):
    """손가락이 천천히 내려오는 것(자세 변화)은 누름으로 보지 않는다."""
    kc, layout = make(cfg)
    depths = [i * 0.5 for i in range(80)]        # 2.7초간 40px 천천히 이동
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), depths)
    assert ev == []


def test_key_locked_at_press_start(cfg):
    """누르는 동안 손끝이 아래 줄로 미끄러져도 처음 선택한 키가 입력된다."""
    kc, layout = make(cfg)
    f_center, v_center = layout.key_center(KeyCode.F), layout.key_center(KeyCode.V)
    seq = [(f_center, 0), (f_center, 0), (f_center, 12), (v_center, 20), (v_center, 20), (v_center, 20)]
    events = []
    for i, (tip, d) in enumerate(seq):
        events += kc.update({Finger.LEFT_INDEX: sample(Finger.LEFT_INDEX, tip, d)}, i * DT, True)
    assert [e.key for e in events] == [KeyCode.F]


# ---------------------------------------------------------------- 인식률 개선
def test_quick_tap_detected_on_15fps_camera(cfg):
    """15fps 카메라에서 약 0.13초짜리 짧은 '톡' 누름도 인식 (2프레임이면 확정)."""
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), [0, 0, 0, 22, 22, 0, 0], dt=1 / 15)
    assert [e.key for e in ev] == [KeyCode.F]


def test_30fps_still_requires_three_frames(cfg):
    kc, layout = make(cfg)
    ev, _ = run_press(kc, Finger.LEFT_INDEX, layout.key_center(KeyCode.F), [0, 0, 0, 22, 22, 0, 0], dt=1 / 30)
    assert ev == []                                   # 30fps 에서 2프레임(0.067초)은 노이즈로 간주


def test_tip_slightly_outside_key_snaps_to_owned_key(cfg):
    kc, layout = make(cfg)
    x, y, w, h = layout.key_rect(layout.key(KeyCode.F))
    just_below = P(x + w / 2, y + h + h * 0.3)       # F 아래 V 키 경계 근처가 아니라 F-V 사이 틈 (V 도 검지 담당)
    left_of_a = P(layout.key_rect(layout.key(KeyCode.A))[0] - w * 0.3, y + h / 2)   # A 왼쪽 Caps 쪽으로 살짝 벗어남
    assert layout.key_for_finger(Finger.LEFT_PINKY, left_of_a, 0.6).code in (KeyCode.A, KeyCode.CAPS_LOCK)
    assert layout.key_for_finger(Finger.LEFT_INDEX, just_below, 0.6) is not None
    far = P(x + w / 2, y - h * 3)                    # 키보드 훨씬 위 -> 스냅 안 함
    assert layout.key_for_finger(Finger.LEFT_INDEX, far, 0.6) is None


def test_snap_does_not_steal_other_fingers_key(cfg):
    kc, layout = make(cfg)
    d_center = layout.key_center(KeyCode.D)          # 중지 담당 D 한가운데에 검지가 있으면
    assert layout.key_for_finger(Finger.LEFT_INDEX, d_center, 0.6) is None or \
        layout.key_for_finger(Finger.LEFT_INDEX, d_center, 0.6).code is not KeyCode.D

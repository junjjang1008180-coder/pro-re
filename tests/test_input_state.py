"""손가락 상태 전환 / 한 번 누르면 한 번만 입력."""
import pytest

from vkeyboard.input_state import FingerState, FingerStateMachine, PressParams

DT = 1 / 30


def machine(min_frames=3, cooldown=0.35):
    return FingerStateMachine(PressParams(18.0, 10.0, min_frames, cooldown, 0.6, 0.15))


def feed(m, depths, t0=0.0):
    fires, states, t = [], [], t0
    for d in depths:
        fires.append(m.update(d, t))
        states.append(m.state)
        t += DT
    return fires, states, t


def test_initial_state_is_hover():
    assert machine().state is FingerState.HOVER


def test_full_transition_cycle():
    m = machine()
    fires, states, _ = feed(m, [0, 12, 20, 20, 20, 5, 5])
    assert states[0] is FingerState.HOVER
    assert states[1] is FingerState.PRESSING           # release(10) 이상 -> 누르는 중
    assert states[2] is FingerState.PRESSING           # 1프레임째 press 임계값 통과
    assert states[4] is FingerState.PRESSED            # 3프레임 연속 -> 확정
    assert states[5] is FingerState.RELEASED           # 다시 올라옴
    assert fires == [False, False, False, False, True, False, False]


def test_released_returns_to_hover_after_hold_time():
    m = machine()
    _, _, t = feed(m, [20, 20, 20, 0])
    assert m.state is FingerState.RELEASED
    m.update(0, t + 0.2)
    assert m.state is FingerState.HOVER


def test_one_press_fires_exactly_once_even_if_held_down():
    m = machine()
    fires, states, _ = feed(m, [20] * 60)             # 2초간 계속 누르고 있음
    assert sum(fires) == 1
    assert states[-1] is FingerState.PRESSED


def test_requires_release_before_next_input():
    m = machine(cooldown=0.0)
    fires, _, t = feed(m, [20, 20, 20, 14, 20, 20, 20])  # 14 는 release(10) 보다 커서 해제 아님
    assert sum(fires) == 1
    fires2, _, _ = feed(m, [5, 20, 20, 20], t)            # 충분히 올라온 뒤 다시 누름
    assert sum(fires2) == 1


def test_short_noise_does_not_fire():
    m = machine()
    fires, _, _ = feed(m, [0, 20, 20, 0, 20, 0, 25, 25, 0])  # 연속 3프레임 미만
    assert sum(fires) == 0


def test_movement_below_press_threshold_does_not_fire():
    m = machine()
    fires, _, _ = feed(m, [0, 12, 15, 17, 17, 17, 12, 0])
    assert sum(fires) == 0


def test_min_down_frames_two_is_faster():
    m = machine(min_frames=2)
    fires, _, _ = feed(m, [20, 20])
    assert fires == [False, True]


def test_tracking_loss_while_pressing_cancels():
    m = machine()
    feed(m, [20, 20])
    m.update(0, 1.0, valid=False)
    assert m.state is FingerState.HOVER
    fires, _, _ = feed(m, [20], 1.1)
    assert not any(fires)


def test_tracking_loss_while_pressed_keeps_pressed_no_refire():
    m = machine(cooldown=0.0)
    _, _, t = feed(m, [20, 20, 20])
    assert m.state is FingerState.PRESSED
    m.update(0, t, valid=False)
    assert m.state is FingerState.PRESSED
    fires, _, _ = feed(m, [20, 20, 20], t + DT)
    assert sum(fires) == 0


def test_slow_drift_is_cancelled_and_requests_rebaseline():
    m = machine()
    t = 0.0
    rebaseline = False
    for _ in range(30):                   # 1초 동안 press 임계값 아래에서 머묾
        m.update(12, t)
        rebaseline |= m.needs_rebaseline
        t += DT
    assert rebaseline


def test_hysteresis_requires_release_smaller_than_press():
    with pytest.raises(ValueError):
        FingerStateMachine(PressParams(10.0, 10.0, 3, 0.35))


def test_custom_thresholds_per_update():
    m = machine()
    # 손이 커서(가까워서) 임계값이 36/20 으로 커진 경우 20px 누름은 입력이 아님
    fires = [m.update(25, i * DT, True, 36.0, 20.0) for i in range(5)]
    assert sum(fires) == 0

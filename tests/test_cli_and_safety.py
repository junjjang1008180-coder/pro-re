"""CLI 오버라이드 / 테스트 모드·실제 입력 안전장치 / ACTIVE 전환 제스처."""
import os

import pytest
from conftest import P

from vkeyboard.cli_options import parse_bool, parse_cli
from vkeyboard.config import AppConfig
from vkeyboard.events import KeyEvent, ModeEvent, MouseAction, MouseEvent
from vkeyboard.gesture_controller import GestureController, is_palm_open
from vkeyboard.input_backend import InputDispatcher, MouseButton, RecordingInputBackend
from vkeyboard.input_processor import InputProcessor
from vkeyboard.calibration import Calibration
from vkeyboard.keyboard_layout import Finger, KeyCode
from vkeyboard.simulation import open_palm_hand, typing_hand

# ---------------------------------------------------------------- CLI


def test_defaults_without_arguments():
    cfg = parse_cli([])
    assert cfg == AppConfig()
    assert (cfg.camera_index, cfg.capture_width, cfg.capture_height) == (0, 3840, 2160)
    assert cfg.test_mode is True and cfg.enable_real_input is False
    # 기본 파일들은 실행 위치와 무관하게 프로젝트 폴더에 둔다
    assert os.path.isabs(cfg.calibration_file) and os.path.basename(cfg.calibration_file) == "calibration.json"
    assert os.path.isabs(cfg.model_path) and cfg.model_path.endswith("hand_landmarker.task")
    assert cfg.log_input is None and cfg.practice is False


def test_overrides():
    cfg = parse_cli(["--camera-index", "2", "--capture-width", "1920", "--capture-height", "1080",
                     "--test-mode", "false", "--enable-real-input", "true",
                     "--calibration-file", "my.json", "--log-input", "log.csv", "--practice"])
    assert (cfg.camera_index, cfg.capture_width, cfg.capture_height) == (2, 1920, 1080)
    assert cfg.test_mode is False and cfg.enable_real_input is True
    assert cfg.calibration_file == "my.json" and cfg.log_input == "log.csv" and cfg.practice


@pytest.mark.parametrize("value, expected", [("true", True), ("False", False), ("1", True), ("no", False)])
def test_parse_bool(value, expected):
    assert parse_bool(value) is expected


def test_invalid_bool_is_rejected():
    with pytest.raises(SystemExit):
        parse_cli(["--test-mode", "maybe"])


@pytest.mark.parametrize("args, allowed", [
    ([], False),
    (["--test-mode", "false"], False),
    (["--enable-real-input", "true"], False),
    (["--test-mode", "false", "--enable-real-input", "true"], True),
    (["--test-mode", "false", "--enable-real-input", "true", "--practice"], False),
    (["--test-mode", "false", "--enable-real-input", "true", "--simulate"], False),
])
def test_real_input_requires_both_flags(args, allowed):
    assert parse_cli(args).real_input_allowed is allowed


# ---------------------------------------------------------------- Dispatcher 안전장치

KEY = KeyEvent(Finger.LEFT_INDEX, KeyCode.F, 0.0, 20.0, 180.0, 0.9)


def test_test_mode_never_calls_backend():
    b = RecordingInputBackend()
    d = InputDispatcher(b, real_input_allowed=False)
    d.set_active(True)
    d.handle_key(KEY)
    d.handle_mouse(MouseEvent(MouseAction.LEFT_CLICK, 10, 10, 0.0))
    assert b.calls == []
    assert d.last_key_text == "F"                     # 화면 표시는 됨


def test_inactive_never_calls_backend():
    b = RecordingInputBackend()
    d = InputDispatcher(b, real_input_allowed=True)
    d.handle_key(KEY)
    assert b.calls == []


def test_real_input_when_allowed_and_active():
    b = RecordingInputBackend()
    d = InputDispatcher(b, real_input_allowed=True)
    d.set_active(True)
    d.handle_key(KEY)
    assert b.calls == [("key_down", KeyCode.F), ("key_up", KeyCode.F)]


def test_shift_is_one_shot_modifier():
    b = RecordingInputBackend()
    d = InputDispatcher(b, True)
    d.set_active(True)
    d.handle_key(KeyEvent(Finger.LEFT_PINKY, KeyCode.LEFT_SHIFT, 0, 20, 180, 0.9))
    d.handle_key(KEY)
    d.handle_key(KEY)
    assert b.calls == [("key_down", KeyCode.LEFT_SHIFT), ("key_down", KeyCode.F), ("key_up", KeyCode.F),
                       ("key_up", KeyCode.LEFT_SHIFT), ("key_down", KeyCode.F), ("key_up", KeyCode.F)]


def test_emergency_stop_releases_drag_and_blocks_input():
    b = RecordingInputBackend()
    d = InputDispatcher(b, True)
    d.set_active(True)
    d.handle_mouse(MouseEvent(MouseAction.DRAG_START, 5, 5, 0.0))
    assert ("button", MouseButton.LEFT, True) in b.calls
    d.emergency_stop("웹캠 연결 실패")
    assert b.calls[-1] == ("button", MouseButton.LEFT, False)
    n = len(b.calls)
    d.handle_key(KEY)
    assert len(b.calls) == n and d.stopped


def test_going_inactive_releases_held_buttons():
    b = RecordingInputBackend()
    d = InputDispatcher(b, True)
    d.set_active(True)
    d.handle_mouse(MouseEvent(MouseAction.DRAG_START, 5, 5, 0.0))
    d.set_active(False)
    assert b.calls[-1] == ("button", MouseButton.LEFT, False)


# ---------------------------------------------------------------- ACTIVE / INACTIVE 전환

def _typing(label):
    x = 500 if label == "Left" else 800
    tips = {"thumb": P(x + 20, 600), "index": P(x + 40, 520), "middle": P(x, 520), "ring": P(x - 40, 520),
            "pinky": P(x - 80, 520)}
    return typing_hand(label, tips, {})


def test_palm_open_detection():
    assert is_palm_open(open_palm_hand("Left", 400, 600).landmarks)
    assert is_palm_open(open_palm_hand("Right", 880, 600).landmarks)
    assert not is_palm_open(_typing("Left").landmarks)


def test_starts_inactive_and_toggles_after_one_second():
    g = GestureController(1.0)
    assert g.active is False
    hands = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    results = [g.update(hands, i / 30) for i in range(29)]
    assert all(r is None for r in results) and not g.active          # 1초 미만
    assert g.update(hands, 1.0) is True and g.active


def test_toggle_requires_both_hands():
    g = GestureController(1.0)
    one = [open_palm_hand("Left", 400, 600), _typing("Right")]
    assert all(g.update(one, i / 30) is None for i in range(60))


def test_toggle_requires_lowering_hands_before_next_toggle():
    g = GestureController(1.0)
    hands = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    t = 0.0
    for _ in range(60):
        if g.update(hands, t) is not None:
            break
        t += 1 / 30
    assert g.active
    assert all(g.update(hands, t + i / 30) is None for i in range(1, 90))   # 계속 펴고 있어도 재전환 없음
    for i in range(15):                                                    # 0.5초간 손을 내림
        g.update([], t + 3.0 + i / 30)
    t2 = t + 3.6
    for _ in range(60):
        if g.update(hands, t2) is not None:
            break
        t2 += 1 / 30
    assert not g.active


def test_brief_detection_dropout_does_not_reset_hold():
    """1초 유지 중 인식이 0.3초 이내로 잠깐 끊겨도 진행이 이어진다."""
    g = GestureController(1.0)
    hands = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    t = 0.0
    for _ in range(15):
        g.update(hands, t)
        t += 1 / 30
    for _ in range(5):                        # 약 0.17초 인식 끊김
        g.update([], t)
        t += 1 / 30
    toggled = False
    for _ in range(20):
        toggled |= g.update(hands, t) is True
        t += 1 / 30
    assert toggled and g.active


def test_manual_toggle_via_processor(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.request_toggle()
    _, events = p.process([], 0.0)
    assert [(e.active, e.reason) for e in events if isinstance(e, ModeEvent)] == [(True, "manual")]
    p.request_toggle()
    _, events = p.process([], 0.1)
    assert [(e.active, e.reason) for e in events if isinstance(e, ModeEvent)] == [(False, "manual")]


def test_han_eng_key_toggles_language_and_sends_key():
    b = RecordingInputBackend()
    d = InputDispatcher(b, True)
    d.set_active(True)
    d.handle_key(KeyEvent(Finger.RIGHT_THUMB, KeyCode.HAN_ENG, 0, 20, 180, 0.9))
    assert d.korean and d.last_key_text == "KO"
    assert b.calls == [("key_down", KeyCode.HAN_ENG), ("key_up", KeyCode.HAN_ENG)]
    d.handle_key(KEY)
    assert d.last_key_text == "ㄹ"                   # 한국어 모드: F 키 = ㄹ
    d.handle_key(KeyEvent(Finger.RIGHT_THUMB, KeyCode.HAN_ENG, 1, 20, 180, 0.9))
    assert not d.korean and d.last_key_text == "EN"


def test_han_eng_platform_codes():
    from vkeyboard.input_backend import _WIN_VK, _X11_KEYSYM

    assert _WIN_VK[KeyCode.HAN_ENG] == 0x15          # VK_HANGUL
    assert _X11_KEYSYM[KeyCode.HAN_ENG] == "Hangul"


def test_processor_forces_inactive_on_tracking_loss(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    hands = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    events, t = [], 0.0
    for _ in range(40):
        events += p.process(hands, t)[1]
        t += 1 / 30
    assert any(isinstance(e, ModeEvent) and e.active for e in events)
    for _ in range(30):
        events += p.process([], t)[1]
        t += 1 / 30
    lost = [e for e in events if isinstance(e, ModeEvent) and not e.active]
    assert lost and lost[0].reason == "tracking_lost"


@pytest.mark.parametrize("level, press, frames", [("low", 22.0, 3), ("normal", 18.0, 3), ("high", 13.0, 2)])
def test_sensitivity_presets(level, press, frames):
    cfg = parse_cli(["--sensitivity", level])
    assert (cfg.press_distance, cfg.min_down_frames) == (press, frames)
    assert cfg.release_distance < cfg.press_distance


def test_explicit_values_override_sensitivity():
    cfg = parse_cli(["--sensitivity", "high", "--press-distance", "16"])
    assert cfg.press_distance == 16 and cfg.min_down_frames == 2


# ---------------------------------------------------------------- 리뷰 수정 사항
@pytest.mark.parametrize("args", [
    ["--overlay-scale", "0"], ["--overlay-scale", "10"], ["--min-down-frames", "0"],
    ["--press-distance", "-5"], ["--key-cooldown", "-1"],
])
def test_invalid_cli_values_are_rejected(args):
    with pytest.raises(SystemExit):
        parse_cli(args)


class _FailingBackend(RecordingInputBackend):
    """관리자 권한 창에 포커스가 있을 때처럼 SendInput 이 실패하는 백엔드."""

    def key_down(self, code):
        raise OSError("SendInput 실패")

    def mouse_button(self, button, down):
        raise OSError("SendInput 실패")


def test_backend_failure_does_not_crash_and_is_counted():
    d = InputDispatcher(_FailingBackend(), True)
    d.set_active(True)
    assert d.handle_key(KEY) is False                      # 예외 대신 False
    assert d.handle_mouse(MouseEvent(MouseAction.DRAG_START, 5, 5, 0.0)) is False
    assert d.error_count == 2 and "SendInput" in d.last_error
    assert not d._held_buttons                              # 실패한 드래그는 '눌림'으로 기록하지 않음
    d.emergency_stop("종료")                                 # 정지 경로도 예외 없음


def test_manual_activation_gives_time_to_bring_hands(cfg):
    """'a' 키로 켠 직후엔 손이 안 보여도 바로 추적 실패로 꺼지지 않는다."""
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.request_toggle()
    events, t = [], 0.0
    for _ in range(60):                                     # 2초 동안 손 없음
        events += p.process([], t)[1]
        t += 1 / 30
    assert p.gesture.active
    for _ in range(60):                                     # 유예(3초)가 지나면 추적 실패로 INACTIVE
        events += p.process([], t)[1]
        t += 1 / 30
    assert not p.gesture.active
    assert [e.reason for e in events if isinstance(e, ModeEvent)] == ["manual", "tracking_lost"]


def test_request_deactivate_turns_processor_inactive(cfg):
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    p.gesture.set_active(True)
    p.request_deactivate()
    _, events = p.process([], 0.0)
    assert not p.gesture.active
    assert [(e.active, e.reason) for e in events if isinstance(e, ModeEvent)] == [(False, "stall")]


def test_mouse_exclusive_option():
    assert parse_cli([]).mouse_exclusive is True
    assert parse_cli(["--mouse-exclusive", "false"]).mouse_exclusive is False



@pytest.mark.parametrize("mode", ["track", "body", "strict", "size", "off"])
def test_hand_lock_option(mode):
    assert parse_cli([]).hand_lock == "track"
    assert parse_cli(["--hand-lock", mode]).hand_lock == mode


def test_hand_lock_invalid_mode():
    with pytest.raises(SystemExit):
        parse_cli(["--hand-lock", "maybe"])

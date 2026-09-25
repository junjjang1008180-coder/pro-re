"""타자 연습 모드: 정확도 / WPM / 손가락별 오류율."""
import json

import pytest

from vkeyboard.config import AppConfig
from vkeyboard.keyboard_layout import Finger, KeyCode, key_for_char, primary_finger_for_char
from vkeyboard.practice_mode import PracticeSession, compute_accuracy, compute_wpm
from vkeyboard.simulation import run_simulation


def type_text(session, text, t0=0.0, dt=0.2, finger=None):
    t = t0
    for ch in text:
        session.on_key(key_for_char(ch), finger or primary_finger_for_char(ch), t)
        t += dt
    return t


def test_accuracy_formula():
    assert compute_accuracy(9, 10) == pytest.approx(0.9)
    assert compute_accuracy(0, 0) == 0.0


def test_wpm_formula():
    assert compute_wpm(50, 60) == pytest.approx(10.0)       # 50글자 = 10단어 / 1분
    assert compute_wpm(25, 30) == pytest.approx(10.0)
    assert compute_wpm(10, 0) == 0.0


def test_perfect_session():
    s = PracticeSession(["abc de"])
    type_text(s, "abc de", t0=0.0, dt=1.0)                    # 6글자, 첫 키~마지막 키 5초
    r = s.result(100.0)
    assert s.finished and r.finished
    assert r.accuracy == 1.0 and r.total_keystrokes == 6
    assert r.elapsed_seconds == pytest.approx(5.0)
    assert r.wpm == pytest.approx((6 / 5) / (5 / 60))


def test_wrong_key_counts_and_does_not_advance():
    s = PracticeSession(["ab"])
    assert s.on_key(KeyCode.A, Finger.LEFT_PINKY, 0.0) == "correct"
    assert s.on_key(KeyCode.V, Finger.LEFT_INDEX, 0.5) == "wrong"
    assert s.expected_char == "b"
    assert s.on_key(KeyCode.B, Finger.LEFT_INDEX, 1.0) == "correct"
    r = s.result(1.0)
    assert (r.correct_keystrokes, r.total_keystrokes, r.errors) == (2, 3, 1)
    assert r.accuracy == pytest.approx(2 / 3)


def test_per_finger_error_rate():
    s = PracticeSession(["fff"])
    s.on_key(KeyCode.F, Finger.LEFT_INDEX, 0.0)
    s.on_key(KeyCode.G, Finger.LEFT_INDEX, 0.1)     # 오타
    s.on_key(KeyCode.D, Finger.LEFT_MIDDLE, 0.2)    # 오타 (다른 손가락)
    s.on_key(KeyCode.F, Finger.LEFT_INDEX, 0.3)
    s.on_key(KeyCode.F, Finger.LEFT_INDEX, 0.4)
    r = s.result(0.4)
    assert r.per_finger["left_index"] == {"keystrokes": 4, "errors": 1, "error_rate": 0.25}
    assert r.per_finger["left_middle"]["error_rate"] == 1.0
    assert r.missed_by_expected_finger == {"left_index": 2}


def test_non_char_keys_are_ignored():
    s = PracticeSession(["a"])
    assert s.on_key(KeyCode.LEFT_SHIFT, Finger.LEFT_PINKY, 0.0) == "ignored"
    assert s.on_key(KeyCode.BACKSPACE, Finger.RIGHT_PINKY, 0.0) == "ignored"
    assert s.total == 0


def test_multiple_sentences_and_reset():
    s = PracticeSession(["ab", "c"])
    type_text(s, "ab")
    assert s.index == 1 and s.current_sentence == "c"
    type_text(s, "c", t0=1.0)
    assert s.finished
    assert s.on_key(KeyCode.A, Finger.LEFT_PINKY, 2.0) == "ignored"
    s.reset()
    assert not s.finished and s.total == 0


def test_result_json(tmp_path):
    s = PracticeSession(["ab"])
    type_text(s, "ab")
    path = tmp_path / "practice_result.json"
    s.result(1.0).save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["accuracy"] == 1.0 and data["total_keystrokes"] == 2 and data["finished"] is True
    assert "per_finger" in data and "wpm" in data


def test_summary_lines_show_worst_finger():
    s = PracticeSession(["aa"])
    s.on_key(KeyCode.Q, Finger.LEFT_PINKY, 0)
    type_text(s, "aa", t0=1)
    lines = s.result(2).summary_lines()
    assert lines[0].startswith("Accuracy: 66.7%")
    assert any("left_pinky" in line for line in lines)


def test_practice_uses_same_detection_path_without_webcam():
    """더미(합성) 손 입력 -> 실제 판정 경로 -> 채점. 웹캠 없이 정확도/WPM 검증."""
    cfg = AppConfig(practice=True, simulate=True)
    report = run_simulation(cfg, sentences=("the quick brown fox",), mistakes=(4,), verbose=False)
    r = report.practice_result
    assert report.typed_text == "the aquick brown fox"         # 4번째 글자 앞에 의도적 오타 'a'
    assert r.finished
    assert (r.correct_keystrokes, r.total_keystrokes) == (19, 20)
    assert r.accuracy == pytest.approx(0.95)
    assert r.per_finger["left_pinky"]["errors"] == 1
    assert r.wpm > 0
    assert report.backend_calls == 0                            # 실제 OS 입력 없음

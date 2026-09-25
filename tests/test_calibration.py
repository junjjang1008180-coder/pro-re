"""좌표 보정: 캘리브레이션 저장/복원 + 좌표 안정화/이상치 제거 + 손 크기 자동 보정."""
import json

import pytest
from conftest import P

from vkeyboard.calibration import Calibration
from vkeyboard.config import AppConfig, hand_scale
from vkeyboard.finger_tracker import FingerTracker
from vkeyboard.geometry import HandObservation
from vkeyboard.keyboard_layout import Finger, KeyboardLayout, KeyCode
from vkeyboard.simulation import typing_hand
from vkeyboard.smoothing import (MovingAverage, OneEuroFilter, PointGate, find_overlapping, is_near_edge)

# ---------------------------------------------------------------- 캘리브레이션


def test_default_is_inside_infer_frame():
    c = Calibration.default()
    assert 0 <= c.x and c.x + c.width <= 1280
    assert 0 <= c.y and c.y + c.height <= 720


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "calibration.json"
    c = Calibration(100, 300, 900, 260)
    c.save(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["width"] == 900
    assert Calibration.load(str(path)) == c


def test_missing_or_corrupt_file_falls_back_to_default(tmp_path):
    assert Calibration.load(str(tmp_path / "nope.json")) == Calibration.default()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert Calibration.load(str(bad)) == Calibration.default()


def test_validated_clamps_into_frame_and_min_size():
    c = Calibration(-50, 700, 10, 5).validated()
    assert c.x == 0 and c.width >= 200 and c.height >= 60
    assert c.y + c.height <= 720


def test_move_and_scale():
    c = Calibration(200, 300, 800, 240)
    m = c.moved(10, -20)
    assert (m.x, m.y) == (210, 280)
    s = c.scaled(1.1, 1.0)
    assert s.width == pytest.approx(880)
    assert s.x + s.width / 2 == pytest.approx(c.x + c.width / 2)   # 중심 유지


def test_fit_to_index_fingers_puts_f_and_j_under_fingers():
    left, right = P(500, 520), P(650, 520)
    c = Calibration.from_index_fingers(left, right)
    layout = KeyboardLayout.from_calibration(c)
    assert layout.key_center(KeyCode.F).x == pytest.approx(500)
    assert layout.key_center(KeyCode.J).x == pytest.approx(650)
    assert layout.key_center(KeyCode.F).y == pytest.approx(520)


def test_fit_rejects_crossed_hands():
    with pytest.raises(ValueError):
        Calibration.from_index_fingers(P(700, 500), P(500, 500))


def test_load_rescales_other_infer_resolution(tmp_path):
    path = tmp_path / "c.json"
    Calibration(20, 100, 600, 200, 640, 360).save(str(path))
    c = Calibration.load(str(path), 1280, 720)
    assert (c.x, c.y, c.width, c.height) == (40, 200, 1200, 400)


# ---------------------------------------------------------------- 안정화 / 이상치


def test_moving_average_reduces_jitter():
    ma = MovingAverage(5)
    for x in [100, 104, 96, 103, 97]:
        v = ma.add(P(x, 0))
    assert v.x == pytest.approx(100)


def test_one_euro_filter_smooths_noise_but_tracks_motion():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    out = [f(100 + (3 if i % 2 else -3), i / 30) for i in range(30)]
    assert max(out[10:]) - min(out[10:]) < 3        # ±3 노이즈 -> 감쇠
    g = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    vals = [g(i * 10.0, i / 30) for i in range(30)]
    assert vals[-1] > 200                            # 빠른 이동은 따라감


def test_point_gate_rejects_jump_then_resyncs():
    gate = PointGate(max_jump=60, max_speed=2400, reset_frames=4)
    assert gate.check(P(100, 100), 0.0) == (True, "first")
    assert gate.check(P(110, 100), 1 / 30) == (True, "ok")
    assert gate.check(P(400, 100), 2 / 30) == (False, "jump")
    assert gate.check(P(400, 100), 3 / 30)[0] is False
    assert gate.check(P(400, 100), 4 / 30)[0] is False
    assert gate.check(P(400, 100), 5 / 30) == (True, "reset")


def test_point_gate_rejects_abnormal_speed():
    gate = PointGate(max_jump=60, max_speed=2400)
    gate.check(P(100, 100), 0.0)
    ok, why = gate.check(P(150, 100), 0.01)          # 50px / 10ms = 5000 px/s
    assert (ok, why) == (False, "speed")


def test_edge_and_overlap_helpers():
    assert is_near_edge(P(5, 300), 1280, 720, 12)
    assert not is_near_edge(P(640, 360), 1280, 720, 12)
    assert find_overlapping({"a": P(0, 0), "b": P(5, 5), "c": P(100, 100)}, 12) == {"a", "b"}


def test_hand_scale_adapts_threshold_to_distance():
    assert hand_scale(180) == pytest.approx(1.0)
    assert hand_scale(90) == pytest.approx(0.6)       # 멀리: 하한
    assert hand_scale(270) == pytest.approx(1.5)
    assert hand_scale(1000) == pytest.approx(1.8)     # 가까이: 상한


def _hand(offset_x=0.0, conf=0.95, press=None):
    tips = {"thumb": P(560 + offset_x, 600), "index": P(600 + offset_x, 520), "middle": P(560 + offset_x, 520),
            "ring": P(520 + offset_x, 520), "pinky": P(480 + offset_x, 520)}
    return typing_hand("Left", tips, press or {}, conf)


def test_finger_tracker_rejects_low_confidence():
    ft = FingerTracker(AppConfig())
    s = ft.update([_hand(conf=0.5)], 0.0)
    assert not s[Finger.LEFT_INDEX].valid and s[Finger.LEFT_INDEX].reason == "low_confidence"


def test_finger_tracker_rejects_edge_points():
    ft = FingerTracker(AppConfig())
    s = ft.update([_hand(offset_x=-595)], 0.0)       # 검지 끝 x=5 (가장자리)
    assert s[Finger.LEFT_INDEX].reason == "edge"


def test_finger_tracker_rejects_overlapping_fingers():
    ft = FingerTracker(AppConfig())
    hand = _hand()
    hand.landmarks[12] = P(hand.landmarks[8].x + 3, hand.landmarks[8].y)   # 중지 끝을 검지 끝에 겹침
    s = ft.update([hand], 0.0)
    assert s[Finger.LEFT_INDEX].reason == "overlap" and s[Finger.LEFT_MIDDLE].reason == "overlap"
    assert s[Finger.LEFT_RING].valid


def test_finger_tracker_rejects_sudden_jump():
    ft = FingerTracker(AppConfig())
    ft.update([_hand()], 0.0)
    s = ft.update([_hand(offset_x=200)], 1 / 30)
    assert s[Finger.LEFT_INDEX].reason == "jump"


def test_finger_tracker_smooths_and_reports_rel_y():
    ft = FingerTracker(AppConfig())
    for i in range(5):
        s = ft.update([_hand(press={"index": 20})], i / 30)
    assert s[Finger.LEFT_INDEX].valid
    assert s[Finger.LEFT_INDEX].rel_y == pytest.approx(60)       # 끝 - MCP = 40 + 20
    assert s[Finger.RIGHT_INDEX].reason == "missing"


def test_duplicate_handedness_is_resolved_by_position():
    from vkeyboard.finger_tracker import normalize_handedness

    a = HandObservation("Right", _hand().landmarks, 0.9)
    b = HandObservation("Right", _hand(offset_x=300).landmarks, 0.95)
    hands = normalize_handedness([b, a])
    assert [h.handedness for h in hands] == ["Left", "Right"]
    assert hands[0].landmarks[0].x < hands[1].landmarks[0].x

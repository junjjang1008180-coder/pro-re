"""손가락별 키 배치 / 키보드 좌표."""
import pytest
from conftest import P

from vkeyboard.keyboard_layout import (BASE_KEYS, FINGER_KEYS, UNITS_W, Finger, KeyboardLayout, KeyCode,
                                       fingers_for_key, key_for_char, primary_finger_for_char)

K = KeyCode

EXPECTED = {
    Finger.LEFT_PINKY: {K.Q, K.A, K.Z, K.TAB, K.CAPS_LOCK, K.LEFT_SHIFT},
    Finger.LEFT_RING: {K.W, K.S, K.X},
    Finger.LEFT_MIDDLE: {K.E, K.D, K.C},
    Finger.LEFT_INDEX: {K.R, K.F, K.V, K.T, K.G, K.B},
    Finger.LEFT_THUMB: {K.SPACE},
    Finger.RIGHT_INDEX: {K.Y, K.H, K.N, K.U, K.J, K.M},
    Finger.RIGHT_MIDDLE: {K.I, K.K, K.COMMA},
    Finger.RIGHT_RING: {K.O, K.L, K.PERIOD},
    Finger.RIGHT_PINKY: {K.P, K.SEMICOLON, K.SLASH, K.BACKSPACE, K.ENTER, K.RIGHT_SHIFT},
    Finger.RIGHT_THUMB: {K.SPACE, K.HAN_ENG},
}


@pytest.mark.parametrize("finger", list(Finger))
def test_finger_assignment_matches_spec(finger):
    assert set(FINGER_KEYS[finger]) == EXPECTED[finger]


def test_ten_fingers_with_expected_names():
    assert [f.value for f in Finger] == [
        "left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
        "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky"]


def test_every_layout_key_has_owner_and_only_space_is_shared():
    for key in BASE_KEYS:
        owners = fingers_for_key(key.code)
        assert owners, key.code
        if key.code is K.SPACE:
            assert set(owners) == {Finger.LEFT_THUMB, Finger.RIGHT_THUMB}
        else:
            assert len(owners) == 1, key.code


def test_every_keycode_is_on_layout_once():
    codes = [k.code for k in BASE_KEYS]
    assert sorted(codes, key=lambda c: c.value) == sorted(KeyCode, key=lambda c: c.value)


def test_rows_fill_full_width():
    for row in range(3):
        width = sum(k.uw for k in BASE_KEYS if k.uy == row)
        assert width == pytest.approx(UNITS_W)


def test_hit_test_returns_key_at_center():
    layout = KeyboardLayout(100, 400, 1080, 320)
    for key in layout.keys:
        assert layout.hit_test(layout.key_center(key.code)).code is key.code


def test_hit_test_outside_keyboard():
    layout = KeyboardLayout(100, 400, 1080, 320)
    assert layout.hit_test(P(50, 50)) is None
    assert not layout.contains(P(50, 50))
    assert layout.contains(P(50, 450), margin=60)


def test_key_for_finger_only_returns_owned_keys():
    layout = KeyboardLayout(100, 400, 1080, 320)
    j = layout.key_center(K.J)
    assert layout.key_for_finger(Finger.RIGHT_INDEX, j).code is K.J
    assert layout.key_for_finger(Finger.RIGHT_MIDDLE, j) is None


def test_qwerty_order_and_row_positions():
    layout = KeyboardLayout(0, 0, 1350, 400)   # 1 unit = 100px
    row0 = [k.code for k in sorted((k for k in layout.keys if k.uy == 0), key=lambda k: k.ux)]
    assert row0 == [K.TAB, K.Q, K.W, K.E, K.R, K.T, K.Y, K.U, K.I, K.O, K.P, K.BACKSPACE]
    assert layout.key_center(K.F).x == pytest.approx(525)
    assert layout.key_center(K.J).x == pytest.approx(825)


def test_char_helpers():
    assert key_for_char("q") is K.Q
    assert key_for_char(" ") is K.SPACE
    assert key_for_char(";") is K.SEMICOLON
    assert key_for_char("1") is None
    assert primary_finger_for_char("a") is Finger.LEFT_PINKY
    assert primary_finger_for_char(" ") is Finger.RIGHT_THUMB
    assert K.LEFT_SHIFT.char is None and K.LEFT_SHIFT.is_modifier

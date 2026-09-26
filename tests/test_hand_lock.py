"""내 손만 인식 (추적 잠금): 등록한 손에서 이어지는 손만 쓰고, 새로 들어온 손은 무시."""
from conftest import P

from vkeyboard.calibration import Calibration
from vkeyboard.config import REACQUIRE_TIME
from vkeyboard.geometry import HandObservation
from vkeyboard.hand_lock import TrackLockSelector
from vkeyboard.input_processor import InputProcessor
from vkeyboard.simulation import open_palm_hand, typing_hand

DT = 1 / 30


def typing(label, cx, cy=470):
    tips = {"thumb": P(cx + 20, cy + 130), "index": P(cx + 40, cy), "middle": P(cx, cy), "ring": P(cx - 40, cy),
            "pinky": P(cx - 80, cy)}
    return typing_hand(label, tips, {})


def lm_id(h):
    return id(h.landmarks)


def locked():
    sel = TrackLockSelector(1280)
    sel.register([open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)], t=0.0)
    return sel


def test_new_hand_entering_is_ignored_even_if_same_size():
    sel = locked()
    me_l, me_r = typing("Left", 420), typing("Right", 860)
    stranger = typing("Right", 640, 300)                        # 같은 크기, 화면 가운데 위로 들어옴
    t = DT
    for _ in range(10):
        chosen, ignored, labeled = sel.select([stranger, me_l, me_r], None, t)
        t += DT
    assert labeled and sel.status == "locked"
    assert {lm_id(h) for h in chosen} == {lm_id(me_l), lm_id(me_r)}
    assert [lm_id(h) for h in ignored] == [lm_id(stranger)]


def test_hands_moving_fast_are_followed():
    sel = locked()
    t = DT
    for i in range(30):                                          # 오른손이 30px/frame(900px/s)로 왼쪽 끝까지 이동
        right = typing("Right", 860 - i * 30)
        chosen, _, _ = sel.select([typing("Left", 420), right], None, t)
        t += DT
    assert {h.handedness for h in chosen} == {"Left", "Right"}
    moved = next(h for h in chosen if h.handedness == "Right")
    assert moved.landmarks is right.landmarks                   # 교차해도 오른손 라벨 유지


def test_briefly_hidden_hand_is_reacquired_nearby():
    sel = locked()
    me_l, me_r = typing("Left", 420), typing("Right", 860)
    t = DT
    sel.select([me_l, me_r], None, t)
    for _ in range(20):                                          # 오른손 0.66초 사라짐
        t += DT
        sel.select([me_l], None, t)
    assert sel.status == "lost_Right"
    back = typing("Right", 900)                                  # 근처에서 다시 나타남
    chosen, _, _ = sel.select([me_l, back], None, t + DT)
    assert any(h.landmarks is back.landmarks and h.handedness == "Right" for h in chosen)
    assert sel.status == "locked"


def test_stranger_far_away_does_not_take_lost_slot():
    sel = locked()
    me_l, me_r = typing("Left", 420), typing("Right", 860)
    t = DT
    sel.select([me_l, me_r], None, t)
    for _ in range(10):
        t += DT
        sel.select([me_l], None, t)
    stranger = typing("Right", 200, 250)                         # 내 오른손이 사라진 위치에서 멀리
    chosen, ignored, _ = sel.select([me_l, stranger], None, t + DT)
    assert [lm_id(h) for h in ignored] == [lm_id(stranger)]


def test_after_timeout_hand_must_show_palm_to_reclaim():
    sel = locked()
    me_l = typing("Left", 420)
    t = DT
    for _ in range(int((REACQUIRE_TIME + 0.5) / DT)):            # 오른손이 오래 사라짐
        sel.select([me_l], None, t)
        t += DT
    back = typing("Right", 880)                                  # 근처로 돌아와도 타이핑 자세면 안 받음
    chosen, ignored, _ = sel.select([me_l, back], None, t)
    assert [lm_id(h) for h in ignored] == [lm_id(back)]
    palm = open_palm_hand("Right", 880, 600)
    for _ in range(25):                                          # 손바닥 0.8초 펴 보이기 -> 되찾기
        t += DT
        chosen, _, _ = sel.select([me_l, palm], None, t)
    assert any(h.landmarks is palm.landmarks and h.handedness == "Right" for h in chosen)


def test_register_with_one_hand_leaves_other_slot_claimable():
    sel = TrackLockSelector(1280)
    sel.register([open_palm_hand("Left", 400, 600)], t=0.0, by_position=True)
    assert set(sel.slots) == {"Left", "Right"} and sel.slots["Right"].lost_at is not None
    t = 0.0
    palm = open_palm_hand("Right", 900, 600)
    for _ in range(25):
        t += DT
        chosen, _, _ = sel.select([open_palm_hand("Left", 400, 600), palm], None, t)
    assert len(chosen) == 2


def test_not_locked_uses_largest_hands():
    sel = TrackLockSelector(1280)
    chosen, _, labeled = sel.select([typing("Left", 420), typing("Right", 860)], None, 0.0)
    assert len(chosen) == 2 and not labeled and sel.status == "not_locked"


def test_processor_locks_on_activation_and_ignores_new_hand(cfg):
    assert cfg.hand_lock == "track"
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    palms = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
    t = 0.0
    for _ in range(40):
        snap, _ = p.process(palms, t)
        t += DT
    assert snap.active and snap.owner_status == "locked"
    stranger = HandObservation("Right", typing("Right", 640, 250).landmarks, 0.95)
    snap, _ = p.process([stranger] + palms, t)
    assert len(snap.hands) == 2 and len(snap.ignored_hands) == 1

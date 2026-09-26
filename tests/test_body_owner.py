"""내 손만 인식 (몸 기준): 사용자 팔에 붙은 손만 쓰는지."""
from conftest import P

from vkeyboard.body_owner import (L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER, R_WRIST,
                                  BodyOwnerSelector, PoseObservation)
from vkeyboard.calibration import Calibration
from vkeyboard.input_processor import InputProcessor
from vkeyboard.simulation import open_palm_hand


def pose(shoulder_l, shoulder_r, wrist_l, wrist_r, elbow_l=None, elbow_r=None, wrist_vis=1.0):
    """사람 한 명의 상체. elbow 를 안 주면 어깨-손목 중간."""
    elbow_l = elbow_l or shoulder_l.lerp(wrist_l, 0.5)
    elbow_r = elbow_r or shoulder_r.lerp(wrist_r, 0.5)
    pts = {L_SHOULDER: shoulder_l, R_SHOULDER: shoulder_r, L_ELBOW: elbow_l, R_ELBOW: elbow_r,
           L_WRIST: wrist_l, R_WRIST: wrist_r}
    vis = {k: 1.0 for k in pts}
    vis[L_WRIST] = vis[R_WRIST] = wrist_vis
    return PoseObservation(pts, vis)


# 사용자: 화면 가운데, 손목이 (400,600) (880,600)
ME = pose(P(500, 300), P(780, 300), P(400, 600), P(880, 600))
MY_HANDS = [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
# 옆 사람: 화면 오른쪽 위, 한 손을 내 쪽으로 뻗음 (손목 (700,560))
OTHER = pose(P(1000, 200), P(1200, 200), P(700, 560), P(1250, 500))
OTHER_HAND = open_palm_hand("Right", 700, 560)


def ids(hands):
    return {id(h.landmarks) for h in hands}


def test_only_hands_attached_to_my_arms_are_used():
    sel = BodyOwnerSelector("body")
    sel.register(MY_HANDS, [ME, OTHER])
    chosen, ignored, labeled = sel.select(MY_HANDS + [OTHER_HAND], [ME, OTHER])
    assert labeled and sel.status == "body"
    assert ids(chosen) == ids(MY_HANDS) and ids(ignored) == {id(OTHER_HAND.landmarks)}


def test_other_hand_near_my_wrist_is_rejected_when_my_hand_is_hidden():
    """내 오른손이 가려진 순간 옆 사람 손이 내 손목 근처에 와도, 그 손은 옆 사람 팔에 더 잘 맞으므로 제외."""
    sel = BodyOwnerSelector("body")
    sel.register(MY_HANDS, [ME, OTHER])
    near = open_palm_hand("Right", 830, 590)                  # 내 오른손목(880,600)에서 50px
    other = pose(P(1000, 200), P(1200, 200), P(830, 590), P(1250, 500))
    chosen, ignored, _ = sel.select([MY_HANDS[0], near], [ME, other])
    assert ids(chosen) == {id(MY_HANDS[0].landmarks)}
    assert ids(ignored) == {id(near.landmarks)}


def test_registration_picks_the_body_that_owns_the_palms_not_the_biggest():
    big_stranger = pose(P(200, 150), P(700, 150), P(150, 650), P(750, 650))   # 더 가까이(어깨 넓음)
    sel = BodyOwnerSelector("body")
    assert sel.register(MY_HANDS, [big_stranger, ME])
    assert sel.user_center.x == 640 and sel.registered


def test_registered_user_not_visible_means_no_hands():
    sel = BodyOwnerSelector("body")
    sel.register(MY_HANDS, [ME])
    chosen, ignored, _ = sel.select([OTHER_HAND], [OTHER])   # 나는 화면 밖, 다른 사람만 보임
    assert chosen == [] and ids(ignored) == {id(OTHER_HAND.landmarks)}
    assert sel.status == "no_user"


def test_no_body_visible_body_mode_falls_back_to_size():
    sel = BodyOwnerSelector("body")
    chosen, _, labeled = sel.select(MY_HANDS, [])            # 카메라가 손만 비춤
    assert ids(chosen) == ids(MY_HANDS) and not labeled and sel.status == "fallback"


def test_no_body_visible_strict_mode_uses_no_hands():
    sel = BodyOwnerSelector("strict")
    chosen, ignored, _ = sel.select(MY_HANDS, [])
    assert chosen == [] and len(ignored) == 2 and sel.status == "no_user"


def test_labels_follow_arms_even_when_hands_cross():
    """손을 교차해 오른손이 화면 왼쪽에 있어도, 오른쪽 어깨 팔에 붙어 있으면 오른손."""
    crossed = pose(P(500, 300), P(780, 300), P(900, 600), P(380, 600))   # 왼팔 손목이 오른쪽, 오른팔 손목이 왼쪽
    hands = [open_palm_hand("Left", 380, 600), open_palm_hand("Right", 900, 600)]
    sel = BodyOwnerSelector("body")
    chosen, _, labeled = sel.select(hands, [crossed])
    by_x = sorted(chosen, key=lambda h: h.landmarks[0].x)
    assert labeled and [h.handedness for h in by_x] == ["Right", "Left"]


def test_hidden_wrist_uses_elbow_and_forearm_length():
    hidden = pose(P(500, 300), P(780, 300), P(400, 600), P(880, 600), wrist_vis=0.0)
    sel = BodyOwnerSelector("body")
    chosen, _, _ = sel.select(MY_HANDS, [hidden])
    assert ids(chosen) == ids(MY_HANDS)


def test_size_and_off_modes_ignore_poses():
    for mode in ("size", "off"):
        sel = BodyOwnerSelector(mode)
        chosen, _, labeled = sel.select(MY_HANDS + [OTHER_HAND], [ME, OTHER])
        assert not labeled and len(chosen) == 2


def test_processor_with_poses_ignores_stranger_and_registers_on_activation(cfg):
    cfg.hand_lock = "body"
    p = InputProcessor(cfg, Calibration.default(), (1920, 1080))
    t = 0.0
    for _ in range(40):                                        # 양손 펴서 ACTIVE -> 사용자 등록
        snap, _ = p.process(MY_HANDS, t, [ME, OTHER])
        t += 1 / 30
    assert snap.active and snap.hand_registered and snap.owner_status == "body"
    snap, _ = p.process(MY_HANDS + [OTHER_HAND], t, [ME, OTHER])
    assert len(snap.hands) == 2 and len(snap.ignored_hands) == 1
    assert {h.handedness for h in snap.hands} == {"Left", "Right"}

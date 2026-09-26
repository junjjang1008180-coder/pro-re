"""내 손만 인식: '사용자 몸(팔)에 붙어 있는 손'만 입력에 쓴다.

손 모양이나 크기만으로는 누구 손인지 알 수 없다. 대신 MediaPipe 포즈(몸 자세) 인식으로
사용자의 어깨 → 팔꿈치 → 손목을 찾고, 손 랜드마크의 손목(0번)이 사용자 팔의 손목과 만나는 손만 인정한다.
옆/뒤 사람이 손을 내밀어도 그 손목은 그 사람의 팔에 붙어 있으므로 걸러진다.

- 사용자 등록: 양손을 펴서 ACTIVE 로 켤 때, 그 두 손이 붙어 있는 몸을 사용자로 등록하고 이후 그 몸만 따라간다.
- 좌/우 판별: 손이 붙은 팔의 어깨가 화면 왼쪽이면 왼손 (손을 교차해도 맞음).

모드 (--hand-lock)
- body   : 몸 기준. 화면에 사람 몸이 아무도 안 보이면(카메라가 손만 비출 때) 크기 기준으로 대체
- strict : 몸 기준. 사용자 몸이 안 보이면 어떤 손도 쓰지 않음
- size   : 크기/위치 기준 (HandSelector)
- off    : 끔 (가장 큰 손 2개)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (OWNER_ELBOW_TOL, OWNER_KEEP_DIST, OWNER_MIN_TOL_PX, OWNER_WIDTH_RANGE, OWNER_WRIST_TOL,
                     POSE_MIN_VISIBILITY)
from .geometry import HandObservation, Vec2, dist
from .hand_selector import HandSelector

# MediaPipe Pose 랜드마크 인덱스
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
POSE_POINTS = (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)
MODES = ("body", "strict", "size", "off")


@dataclass
class PoseObservation:
    """한 사람의 상체 랜드마크 (추론 좌표계). visibility < 기준이면 안 보이는 점."""

    points: Dict[int, Vec2] = field(default_factory=dict)
    visibility: Dict[int, float] = field(default_factory=dict)

    def get(self, idx: int) -> Optional[Vec2]:
        if self.visibility.get(idx, 0.0) < POSE_MIN_VISIBILITY:
            return None
        return self.points.get(idx)

    def shoulders(self) -> Optional[Tuple[Vec2, Vec2]]:
        a, b = self.get(L_SHOULDER), self.get(R_SHOULDER)
        return (a, b) if a is not None and b is not None else None

    @property
    def center(self) -> Optional[Vec2]:
        s = self.shoulders()
        return s[0].lerp(s[1], 0.5) if s else None

    @property
    def width(self) -> float:
        s = self.shoulders()
        return dist(*s) if s else 0.0

    def arms(self) -> List[Tuple[str, Optional[Vec2], Optional[Vec2]]]:
        """[(화면 기준 'Left'|'Right', 팔꿈치, 손목)]. 어깨가 화면 왼쪽인 팔 = Left (거울 영상 = 사용자 시점)."""
        s = self.shoulders()
        if s is None:
            return []
        arm_a = (s[0], self.get(L_ELBOW), self.get(L_WRIST))
        arm_b = (s[1], self.get(R_ELBOW), self.get(R_WRIST))
        left, right = sorted((arm_a, arm_b), key=lambda a: a[0].x)
        return [("Left", left[1], left[2]), ("Right", right[1], right[2])]


def _arm_distance(hand_wrist: Vec2, elbow: Optional[Vec2], wrist: Optional[Vec2], width: float) -> Optional[float]:
    """손이 이 팔에 붙어 있으면 정규화 거리(작을수록 확실), 아니면 None."""
    tol = max(OWNER_WRIST_TOL * width, OWNER_MIN_TOL_PX)
    if wrist is not None:
        d = dist(hand_wrist, wrist)
        return d / tol if d <= tol else None
    if elbow is not None:                       # 손목이 가려졌으면 팔꿈치에서 팔뚝 길이 이내인지
        d = dist(hand_wrist, elbow)
        limit = max(OWNER_ELBOW_TOL * width, OWNER_MIN_TOL_PX * 2)
        return 1.0 + d / limit if d <= limit else None
    return None


class BodyOwnerSelector:
    def __init__(self, mode: str = "body", frame_width: float = 1280.0) -> None:
        if mode not in MODES:
            raise ValueError(f"hand-lock 모드는 {MODES} 중 하나")
        self.mode = mode
        self.frame_width = frame_width
        self.size_selector = HandSelector(mode != "off", frame_width)
        self.user_center: Optional[Vec2] = None     # 등록된 사용자 몸(어깨 중심)
        self.user_width = 0.0
        self.registered = False
        self._register_pending = False
        self.status = "size" if mode in ("size", "off") else "waiting"
        self.user_pose: Optional[PoseObservation] = None

    # ------------------------------------------------------------------
    def request_register(self) -> None:
        self._register_pending = True
        self.size_selector.request_register()

    def register(self, hands: Sequence[HandObservation], poses: Optional[Sequence[PoseObservation]],
                 t: float = 0.0) -> bool:
        """ACTIVE 전환 순간: 지금 보이는 손(=사용자 손)이 붙은 몸을 사용자로 등록."""
        self.size_selector.register(hands)
        if not poses:
            return False
        best, best_n = None, 0
        for pose in poses:
            n = sum(1 for h in hands if self._owner_arm(h, pose) is not None)
            if n > best_n or (n == best_n and best is not None and pose.width > best.width):
                best, best_n = pose, n
        if best is None or best_n == 0 or best.center is None:
            return False
        self._set_user(best)
        self.registered = True
        self._register_pending = False
        return True

    def _set_user(self, pose: PoseObservation) -> None:
        self.user_pose = pose
        self.user_center = pose.center
        self.user_width = pose.width

    # ------------------------------------------------------------------
    def _pick_user(self, poses: Sequence[PoseObservation]) -> Optional[PoseObservation]:
        valid = [p for p in poses if p.center is not None and p.width > 0]
        if not valid:
            return None
        if self.user_center is not None:
            keep = OWNER_KEEP_DIST * self.frame_width
            lo, hi = OWNER_WIDTH_RANGE
            near = [p for p in valid if dist(p.center, self.user_center) <= keep
                    and (self.user_width <= 0 or lo <= p.width / self.user_width <= hi)]
            if near:
                return min(near, key=lambda p: dist(p.center, self.user_center))
            if self.registered:
                return None                      # 등록한 사용자가 안 보임 (다른 사람만 보임)
        return max(valid, key=lambda p: p.width)  # 등록 전: 카메라에 가장 가까운(어깨가 넓게 보이는) 사람

    def _owner_arm(self, hand: HandObservation, pose: PoseObservation) -> Optional[Tuple[str, float]]:
        best = None
        for label, elbow, wrist in pose.arms():
            d = _arm_distance(hand.landmarks[0], elbow, wrist, pose.width)
            if d is not None and (best is None or d < best[1]):
                best = (label, d)
        return best

    def select(self, hands: Sequence[HandObservation], poses: Optional[Sequence[PoseObservation]],
               t: float = 0.0) -> Tuple[List[HandObservation], List[HandObservation], bool]:
        """(사용할 손(라벨 확정), 무시한 손, 라벨이 몸 기준으로 확정됐는지)."""
        cands = [h for h in hands if len(h.landmarks) >= 21]
        if self.mode in ("size", "off") or poses is None:
            chosen, ignored = self.size_selector.select(cands)
            self.status = "size"
            return chosen, ignored, False

        user = self._pick_user(poses)
        if user is None:
            self.user_pose = None
            if not poses and self.mode == "body":
                # 화면에 사람 몸이 아무도 없음(카메라가 손만 비춤) -> 크기 기준으로 대체
                chosen, ignored = self.size_selector.select(cands)
                self.status = "fallback"
                return chosen, ignored, False
            self.status = "no_user"
            return [], cands, False

        # 각 손을 '가장 잘 맞는 사람의 팔'에 배정하고, 그 사람이 사용자일 때만 인정.
        # (사용자 손이 잠깐 가려졌을 때 옆 사람 손이 사용자 손목 근처에 와도 그 사람 팔에 더 가까우면 제외)
        matches = []
        for h in cands:
            best_pose, best = None, None
            for pose in poses:
                arm = self._owner_arm(h, pose)
                if arm is not None and (best is None or arm[1] < best[1]):
                    best_pose, best = pose, arm
            if best is not None and best_pose is user:
                matches.append((best[1], best[0], h))
        chosen: List[HandObservation] = []
        used_arms = set()
        for _, label, h in sorted(matches, key=lambda m: m[0]):
            if label in used_arms:
                continue
            used_arms.add(label)
            chosen.append(HandObservation(label, h.landmarks, h.confidence))
        chosen_ids = {id(h.landmarks) for h in chosen}
        ignored = [h for h in cands if id(h.landmarks) not in chosen_ids]

        self._set_user(user)
        self.status = "body"
        if self._register_pending and len(chosen) == 2:
            self.registered = True
            self._register_pending = False
        return chosen, ignored, True

"""내 손만 인식 (추적 잠금, 기본): 등록한 두 손을 프레임마다 이어서 따라가고, 새로 들어온 손은 무시한다.

몸(포즈) 기준은 손을 카메라 앞으로 들면 팔이 몸을 가려 인식이 깨지고, 크기 기준은 비슷한 체격의
다른 사람 손을 막지 못한다. 대신 '연속성'을 쓴다: 손은 한 프레임 사이에 순간이동하지 않으므로,
등록한 손에서 이어지는 손만 내 손이다. 다른 사람 손은 어디서 들어오든 새 손이라 무시된다.

- 등록: 양손을 펴서 ACTIVE 로 켜는 순간의 두 손 (a 키로 켰다면 다음에 두 손이 보일 때)
- 추적: 직전 위치에서 (기본 거리 + 속도 x 경과 시간) 안의 손만 같은 손으로 이어 받음
- 잠깐 가려짐: REACQUIRE_TIME 안에 사라진 위치 근처(REACQUIRE_RADIUS)에서 다시 나타나면 이어 받음
- 되찾기: 그 밖의 경우엔 손바닥을 CLAIM_TIME 동안 펴 보이면 비어 있는 자리(왼손/오른손)로 다시 등록
- 좌/우: 등록 때 정한 라벨을 계속 유지 (손을 교차해도 안 바뀜)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

from .config import (CLAIM_TIME, HAND_SIZE_MAX_RATIO, HAND_SIZE_MIN_RATIO, REACQUIRE_RADIUS, REACQUIRE_TIME,
                     TRACK_BASE_DIST, TRACK_SPEED)
from .geometry import HandObservation, Vec2, dist, hand_size
from .gesture_controller import is_palm_open
from .hand_selector import HandSelector


def _center(hand: HandObservation) -> Vec2:
    n = len(hand.landmarks)
    return Vec2(sum(p.x for p in hand.landmarks) / n, sum(p.y for p in hand.landmarks) / n)


def _size(hand: HandObservation) -> float:
    lm = hand.landmarks
    return max(hand_size(lm), 2.0 * dist(lm[0], lm[9]))   # 손가락을 접어도 줄지 않도록 손바닥 기준과 큰 쪽


@dataclass
class HandSlot:
    label: str               # "Left" | "Right"
    center: Vec2
    size: float
    last_seen: float
    lost_at: Optional[float] = None      # None 이면 추적 중


class TrackLockSelector:
    def __init__(self, frame_width: float = 1280.0) -> None:
        self.frame_width = frame_width
        self.slots: Dict[str, HandSlot] = {}
        self.size_selector = HandSelector(True, frame_width)   # 등록 전에만 사용
        self._register_pending = False
        self._claim: Optional[Tuple[Vec2, float]] = None       # (손바닥 편 손 위치, 시작 시각)
        self.status = "not_locked"
        self.user_pose = None                                   # BodyOwnerSelector 와 같은 인터페이스용

    @property
    def registered(self) -> bool:
        return bool(self.slots)

    def request_register(self) -> None:
        self._register_pending = True

    def register(self, hands: Sequence[HandObservation], poses=None, t: float = 0.0,
                 by_position: bool = False) -> bool:
        """지금 보이는 손을 내 손으로 잠근다. 한 손만 보이면 다른 쪽은 '되찾기 대기' 자리로 둔다."""
        valid = [h for h in hands if len(h.landmarks) >= 21][:2]
        if not valid:
            return False
        if by_position or (len(valid) == 2 and valid[0].handedness == valid[1].handedness):
            valid = sorted(valid, key=lambda h: _center(h).x)
            labels = ["Left", "Right"] if len(valid) == 2 else \
                ["Left" if _center(valid[0]).x < self.frame_width / 2 else "Right"]
        else:
            labels = [h.handedness for h in valid]
        self.slots = {lab: HandSlot(lab, _center(h), _size(h), t) for lab, h in zip(labels, valid)}
        for lab in ("Left", "Right"):
            if lab not in self.slots:          # 안 보인 손: 손바닥을 펴 보이면 등록되는 빈 자리
                self.slots[lab] = HandSlot(lab, Vec2(-1e6, -1e6), 0.0, t, lost_at=float("-inf"))
        self._register_pending = False
        self._claim = None
        return True

    # ------------------------------------------------------------------
    def _allowed(self, slot: HandSlot, t: float) -> Optional[float]:
        """이 자리(slot)가 지금 이어 받을 수 있는 최대 거리. None 이면 위치로는 이어 받지 않음(되찾기 필요)."""
        if slot.lost_at is None:
            return TRACK_BASE_DIST + TRACK_SPEED * max(0.0, t - slot.last_seen)
        if t - slot.lost_at <= REACQUIRE_TIME:
            return REACQUIRE_RADIUS * self.frame_width
        return None

    def _size_ok(self, slot: HandSlot, size: float) -> bool:
        return slot.size <= 0 or HAND_SIZE_MIN_RATIO <= size / slot.size <= HAND_SIZE_MAX_RATIO

    def select(self, hands: Sequence[HandObservation], poses=None, t: float = 0.0
               ) -> Tuple[List[HandObservation], List[HandObservation], bool]:
        """(사용할 손(라벨 확정), 무시한 손, 라벨 확정 여부)."""
        cands = [h for h in hands if len(h.landmarks) >= 21]
        if not self.slots:
            chosen, ignored = self.size_selector.select(cands)
            self.status = "not_locked"
            if self._register_pending and len(chosen) == 2:
                # 'a' 키로 켠 뒤 처음 두 손이 보이면 등록 (라벨은 이후 단계에서 정해지므로 위치로 정함)
                self.register(chosen, t=t, by_position=True)
                return self.select(hands, poses, t)
            return chosen, ignored, False

        centers = [_center(h) for h in cands]
        sizes = [_size(h) for h in cands]
        slot_list = list(self.slots.values())

        # 자리 <-> 손 최적 배정 (자리 최대 2개라 모든 경우를 비교)
        best: List[Tuple[HandSlot, int]] = []
        best_key = (0, 0.0)
        idxs = list(range(len(cands)))
        for k in range(min(len(slot_list), len(cands)), 0, -1):
            for slots in permutations(slot_list, k):
                for hand_idx in permutations(idxs, k):
                    pairs, total, ok = [], 0.0, True
                    for slot, i in zip(slots, hand_idx):
                        limit = self._allowed(slot, t)
                        d = dist(centers[i], slot.center)
                        if limit is None or d > limit or not self._size_ok(slot, sizes[i]):
                            ok = False
                            break
                        pairs.append((slot, i))
                        total += d
                    key = (len(pairs), -total)
                    if ok and key > best_key:
                        best, best_key = pairs, key
            if best:
                break

        chosen: List[HandObservation] = []
        used = set()
        for slot, i in best:
            h = cands[i]
            chosen.append(HandObservation(slot.label, h.landmarks, h.confidence))
            slot.center, slot.last_seen, slot.lost_at = centers[i], t, None
            slot.size += 0.05 * (sizes[i] - slot.size)
            used.add(i)
        for slot in slot_list:
            if slot.lost_at is None and slot.last_seen < t and all(s is not slot for s, _ in best):
                if t - slot.last_seen > 0.15:
                    slot.lost_at = slot.last_seen          # 잠깐(0.15초)은 추적 중으로 봐준다

        # 되찾기: 비어 있는 자리가 있고, 남은 손 중 하나가 손바닥을 CLAIM_TIME 동안 펴고 있으면 그 자리로 등록
        free = [s for s in slot_list if s.lost_at is not None]
        rest = [i for i in idxs if i not in used]
        claimed = self._update_claim(cands, centers, rest, free, t)
        if claimed is not None:
            i, slot = claimed
            h = cands[i]
            slot.center, slot.size, slot.last_seen, slot.lost_at = centers[i], sizes[i], t, None
            chosen.append(HandObservation(slot.label, h.landmarks, h.confidence))
            used.add(i)

        ignored = [cands[i] for i in idxs if i not in used]
        lost = [s.label for s in slot_list if s.lost_at is not None]
        self.status = "locked" if not lost else ("lost_" + "_".join(sorted(lost)))
        return chosen, ignored, True

    def _update_claim(self, cands, centers, rest, free, t) -> Optional[Tuple[int, HandSlot]]:
        if not free:
            self._claim = None
            return None
        palms = [i for i in rest if is_palm_open(cands[i].landmarks)]
        if not palms:
            self._claim = None
            return None
        # 직전에 펴고 있던 손과 이어지는 손바닥을 우선
        if self._claim is not None:
            prev_c, start = self._claim
            i = min(palms, key=lambda j: dist(centers[j], prev_c))
            if dist(centers[i], prev_c) > TRACK_BASE_DIST + TRACK_SPEED * 0.1:
                self._claim = (centers[i], t)
                return None
        else:
            i = palms[0]
            self._claim = (centers[i], t)
            return None
        self._claim = (centers[i], start)
        if t - start < CLAIM_TIME:
            return None
        self._claim = None
        if len(free) == 1:
            return i, free[0]
        # 두 자리가 다 비었으면 화면 위치로 (왼쪽 절반 = 왼손)
        want = "Left" if centers[i].x < self.frame_width / 2 else "Right"
        return i, next(s for s in free if s.label == want)

"""10개 손가락 끝 좌표를 안정화하고, 입력에 쓰면 안 되는 좌표를 걸러낸다."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import (HAND_SIZE_SMOOTHING, OUTLIER_RESET_FRAMES, PRESS_SMOOTHING_FRAMES, TRACKING_GAP_RESET,
                     AppConfig, hand_scale)
from .geometry import (HandObservation, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP,
                       PINKY_TIP, RING_MCP, RING_TIP, THUMB_MCP, THUMB_TIP, Vec2, hand_size)
from .keyboard_layout import FINGER_ORDER, Finger
from .smoothing import MovingAverage, PointGate, ScalarMovingAverage, find_overlapping, is_near_edge

# 손가락 -> (끝, 기준 관절) 랜드마크 인덱스. 누름 깊이는 "끝 - 기준 관절" 의 y 차이로 측정해
# 손 전체가 위아래로 움직이는 것은 누름으로 보지 않는다.
FINGER_LANDMARKS: Dict[str, tuple] = {
    "thumb": (THUMB_TIP, THUMB_MCP),
    "index": (INDEX_TIP, INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_MCP),
    "ring": (RING_TIP, RING_MCP),
    "pinky": (PINKY_TIP, PINKY_MCP),
}


@dataclass
class FingerSample:
    finger: Finger
    valid: bool
    reason: str = "ok"                 # ok | missing | low_confidence | edge | overlap | jump | speed
    tip: Optional[Vec2] = None         # 안정화된 끝 좌표 (추론 좌표계)
    raw_tip: Optional[Vec2] = None
    rel_y: float = 0.0                 # 안정화된 (끝.y - 기준관절.y)
    hand_size: float = 0.0
    confidence: float = 0.0
    resynced: bool = False             # 필터가 새 위치로 재동기화됨 -> 누름 기준선도 다시 잡아야 함


def _center_x(hand: HandObservation) -> float:
    return sum(p.x for p in hand.landmarks) / len(hand.landmarks)


def normalize_handedness(hands: Sequence[HandObservation],
                         split_x: Optional[float] = None,
                         prev_centers: Optional[Dict[str, float]] = None,
                         keep_dist: float = 0.0) -> List[HandObservation]:
    """최대 2손의 좌/우 라벨을 정리한다.

    split_x 가 없으면 MediaPipe 라벨을 쓰되, 두 손이 같은 쪽으로 나오면 화면 x 위치로 나눈다.
    split_x 가 있으면(위치 기준 모드) 라벨을 무시하고 화면 위치로 정한다:
    두 손이면 왼쪽 손 = Left, 한 손이면 split_x(키보드 중앙) 왼쪽 = Left.
    단, 한 손만 보일 때 prev_centers(직전 프레임 손 중심 x)에서 keep_dist 안이면 같은 손으로 유지한다
    (오른손으로 마우스를 움직이다 화면 왼쪽으로 가도 왼손으로 바뀌지 않게).
    MediaPipe 는 손바닥 기준으로 좌우를 판별하므로, 타이핑 자세처럼 손등이 카메라를 향하면
    좌우가 뒤집혀 나온다 -> 가상 키보드에서는 위치 기준이 더 안정적이다.
    """
    hands = [h for h in hands if len(h.landmarks) >= 21]
    hands = sorted(hands, key=lambda h: h.confidence, reverse=True)[:2]
    if len(hands) == 2 and (split_x is not None or hands[0].handedness == hands[1].handedness):
        a, b = sorted(hands, key=_center_x)
        return [HandObservation("Left", a.landmarks, a.confidence),
                HandObservation("Right", b.landmarks, b.confidence)]
    if len(hands) == 1 and split_x is not None:
        h = hands[0]
        cx = _center_x(h)
        label = "Left" if cx < split_x else "Right"
        if prev_centers:
            near = min(prev_centers.items(), key=lambda kv: abs(kv[1] - cx))
            if abs(near[1] - cx) <= keep_dist:
                label = near[0]
        return [HandObservation(label, h.landmarks, h.confidence)]
    return hands


class FingerTracker:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._tip_avg = {f: MovingAverage(cfg.smoothing_frames) for f in FINGER_ORDER}
        # 위치(어느 키 위인지)는 강하게, 누름 깊이는 약하게 평균: 짧은 누름도 살아남도록
        self._rel_avg = {f: ScalarMovingAverage(min(cfg.smoothing_frames, PRESS_SMOOTHING_FRAMES))
                         for f in FINGER_ORDER}
        self._gates = {f: PointGate(cfg.max_allowed_jump, cfg.max_speed, OUTLIER_RESET_FRAMES,
                                    TRACKING_GAP_RESET) for f in FINGER_ORDER}
        self._hand_size: Dict[str, float] = {}
        self.last_samples: Dict[Finger, FingerSample] = {}

    def _smoothed_hand_size(self, label: str, raw: float) -> float:
        prev = self._hand_size.get(label)
        value = raw if prev is None else prev + HAND_SIZE_SMOOTHING * (raw - prev)
        self._hand_size[label] = value
        return value

    def _reset_finger(self, f: Finger) -> None:
        self._tip_avg[f].reset()
        self._rel_avg[f].reset()
        self._gates[f].reset()

    def update(self, hands: Sequence[HandObservation], t: float) -> Dict[Finger, FingerSample]:
        cfg = self.cfg
        hands = normalize_handedness(hands)
        by_label = {h.handedness: h for h in hands}
        for label in ("Left", "Right"):
            if label not in by_label:
                self._hand_size.pop(label, None)
        hand_sizes = {label: self._smoothed_hand_size(label, hand_size(h.landmarks))
                      for label, h in by_label.items()}

        # 1) 원시 좌표 수집
        raw: Dict[Finger, tuple] = {}
        sizes: Dict[Finger, float] = {}
        for f in FINGER_ORDER:
            hand = by_label.get(f.hand)
            if hand is None:
                continue
            tip_i, base_i = FINGER_LANDMARKS[f.digit]
            sizes[f] = hand_sizes[f.hand]
            raw[f] = (hand.landmarks[tip_i], hand.landmarks[base_i], hand.confidence)

        # 2) 겹친 손가락 (양손 전체) — 손 크기에 비례한 거리로 판정
        overlap_th = {f: cfg.finger_overlap_distance * hand_scale(sizes[f], cfg.reference_hand_size)
                      for f in raw}
        overlapping = find_overlapping({f: v[0] for f, v in raw.items()},
                                       cfg.finger_overlap_distance, overlap_th)

        samples: Dict[Finger, FingerSample] = {}
        for f in FINGER_ORDER:
            if f not in raw:
                self._reset_finger(f)
                samples[f] = FingerSample(f, False, "missing")
                continue
            tip, base, conf = raw[f]
            s = FingerSample(f, False, "ok", raw_tip=tip, hand_size=sizes[f], confidence=conf)
            prev_tip = self._tip_avg[f].value()
            s.tip = prev_tip if prev_tip is not None else tip

            if conf < cfg.min_tracking_confidence:
                s.reason = "low_confidence"
            elif is_near_edge(tip, cfg.infer_width, cfg.infer_height, cfg.edge_margin):
                s.reason = "edge"
            elif f in overlapping:
                s.reason = "overlap"
            else:
                ok, why = self._gates[f].check(tip, t)
                if not ok:
                    s.reason = why
                else:
                    if why in ("reset", "first"):
                        self._tip_avg[f].reset()
                        self._rel_avg[f].reset()
                        s.resynced = True
                    s.tip = self._tip_avg[f].add(tip)
                    s.rel_y = self._rel_avg[f].add(tip.y - base.y)
                    s.valid = True
            samples[f] = s
        self.last_samples = samples
        return samples

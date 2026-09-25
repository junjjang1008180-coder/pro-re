import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vkeyboard.config import AppConfig  # noqa: E402
from vkeyboard.finger_tracker import FingerSample  # noqa: E402
from vkeyboard.geometry import Vec2  # noqa: E402


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


def sample(finger, tip, rel_y=0.0, hand_size=180.0, conf=0.95, valid=True, resynced=False):
    """KeyboardController 에 직접 넣을 FingerSample 생성 (hand_size 180 -> 임계값 배율 1.0)."""
    return FingerSample(finger, valid, "ok" if valid else "missing", tip=tip, raw_tip=tip, rel_y=rel_y,
                        hand_size=hand_size, confidence=conf, resynced=resynced)


def run_press(controller, finger, tip, depths, t0=0.0, dt=1 / 30, active=True):
    """depths(누름 깊이) 시퀀스를 한 손가락에 적용하고 (이벤트 목록, 끝 시각) 반환."""
    events = []
    t = t0
    for d in depths:
        events += controller.update({finger: sample(finger, tip, rel_y=d)}, t, active)
        t += dt
    return events, t


P = Vec2

"""--simulate: 웹캠 없이 합성 손 랜드마크로 실제 판정 경로 전체를 검증 (헤드리스).

합성 손 -> InputProcessor(FingerTracker/KeyboardController/MouseController/GestureController)
       -> KeyEvent/MouseEvent -> InputDispatcher(테스트 모드) + PracticeSession 채점

실제 입력 경로와 완전히 같은 코드를 쓰며, 카메라/MediaPipe/OpenCV 가 필요 없다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .calibration import Calibration
from .config import AppConfig
from .events import KeyEvent, ModeEvent, MouseAction, MouseEvent
from .geometry import HandObservation, Vec2
from .input_backend import InputDispatcher, RecordingInputBackend
from .input_processor import InputProcessor
from .keyboard_layout import (FINGER_KEYS, Finger, KeyboardLayout, KeyCode, key_for_char,
                              primary_finger_for_char)
from .practice_mode import DEFAULT_SENTENCES, PracticeResult, PracticeSession

FPS = 30.0
UNIT = 42.0                       # 합성 키보드의 키 1칸 크기(px, 추론 좌표계)
PRESS_PROFILE = (8, 16, 26, 36, 36, 36, 36, 24, 12, 4, 0, 0, 0, 0, 0, 0)
_BASES = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_HOME = {
    "Left": {"pinky": KeyCode.A, "ring": KeyCode.S, "middle": KeyCode.D, "index": KeyCode.F},
    "Right": {"index": KeyCode.J, "middle": KeyCode.K, "ring": KeyCode.L, "pinky": KeyCode.SEMICOLON},
}


def simulation_calibration(cfg: AppConfig) -> Calibration:
    """왼손 검지 F, 오른손 검지 J 위치로 맞춘 키보드 (캘리브레이션 'f' 기능과 같은 계산)."""
    left = Vec2(520.0, 520.0)
    return Calibration.from_index_fingers(left, Vec2(left.x + 3 * UNIT, left.y),
                                          cfg.infer_width, cfg.infer_height)


def home_tips(layout: KeyboardLayout, label: str) -> Dict[str, Vec2]:
    tips = {d: layout.key_center(k) for d, k in _HOME[label].items()}
    sp = layout.key_rect(layout.key(KeyCode.SPACE))
    thumb_x = layout.x + (5.0 if label == "Left" else 8.5) * layout.unit_w
    tips["thumb"] = Vec2(thumb_x, sp[1] + sp[3] / 2)
    return tips


def typing_hand(label: str, tips: Dict[str, Vec2], press: Dict[str, float],
                confidence: float = 0.95) -> HandObservation:
    """타이핑 자세(손가락이 모이고 굽혀짐)의 21점 손. press[digit] 만큼 손가락 끝만 아래로 이동."""
    lm: List[Optional[Vec2]] = [None] * 21
    for digit, base in _BASES.items():
        t = tips[digit]
        tip = t + Vec2(0, press.get(digit, 0.0))
        lm[base] = t + Vec2(0, -40)           # MCP (손끝보다 위)
        lm[base + 1] = t + Vec2(0, -27)
        lm[base + 2] = tip + Vec2(0, -13)
        lm[base + 3] = tip
    lm[0] = lm[9] + Vec2(0, -140)             # 손목 -> 손 크기(손목~중지 끝) = 180
    th = tips["thumb"]
    s = 1.0 if label == "Left" else -1.0
    lm[4] = th + Vec2(0, press.get("thumb", 0.0))
    lm[3] = th + Vec2(-s * 10, -25)
    lm[2] = th + Vec2(-s * 20, -45)
    lm[1] = th + Vec2(-s * 30, -80)
    return HandObservation(label, lm, confidence)  # type: ignore[arg-type]


def open_palm_hand(label: str, cx: float, wrist_y: float, confidence: float = 0.95) -> HandObservation:
    """손바닥을 활짝 편 21점 손 (ACTIVE/INACTIVE 전환 제스처)."""
    s = 1.0 if label == "Left" else -1.0      # 거울 영상에서 왼손 엄지는 화면 오른쪽(안쪽)
    w = Vec2(cx, wrist_y)
    lm: List[Optional[Vec2]] = [None] * 21
    lm[0] = w
    for digit, ox in (("index", 60), ("middle", 20), ("ring", -20), ("pinky", -60)):
        base = _BASES[digit]
        mcp = w + Vec2(s * ox * 0.6, -90)
        tip = w + Vec2(s * ox * 1.3, -190)
        lm[base], lm[base + 1], lm[base + 2], lm[base + 3] = mcp, mcp.lerp(tip, 0.4), mcp.lerp(tip, 0.75), tip
    lm[1], lm[2], lm[3], lm[4] = (w + Vec2(s * 30, -30), w + Vec2(s * 60, -60), w + Vec2(s * 95, -85),
                                  w + Vec2(s * 130, -100))
    return HandObservation(label, lm, confidence)  # type: ignore[arg-type]


@dataclass
class SimulationReport:
    typed_text: str
    key_events: List[KeyEvent] = field(default_factory=list)
    mouse_events: List[MouseEvent] = field(default_factory=list)
    mode_events: List[ModeEvent] = field(default_factory=list)
    practice_result: Optional[PracticeResult] = None
    backend_calls: int = 0
    frames: int = 0
    duration: float = 0.0


class HandSimulator:
    def __init__(self, cfg: AppConfig, processor: InputProcessor) -> None:
        self.cfg = cfg
        self.p = processor
        self.layout = processor.layout
        self.t = 0.0
        self.frames = 0
        self.offset = {"Left": Vec2(0, 0), "Right": Vec2(0, 0)}
        self.press: Dict[str, Dict[str, float]] = {"Left": {}, "Right": {}}
        self.thumb_override: Dict[str, Optional[Vec2]] = {"Left": None, "Right": None}
        self.pose = "typing"          # typing | open | none
        self.events: list = []
        self._home = {lab: home_tips(self.layout, lab) for lab in ("Left", "Right")}

    def _hands(self) -> List[HandObservation]:
        if self.pose == "none":
            return []
        if self.pose == "open":
            return [open_palm_hand("Left", 400, 600), open_palm_hand("Right", 880, 600)]
        hands = []
        for lab in ("Left", "Right"):
            tips = {d: p + self.offset[lab] for d, p in self._home[lab].items()}
            if self.thumb_override[lab] is not None:
                tips["thumb"] = self.thumb_override[lab]
            hands.append(typing_hand(lab, tips, self.press[lab]))
        return hands

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            _, ev = self.p.process(self._hands(), self.t)
            self.events.extend(ev)
            self.t += 1.0 / FPS
            self.frames += 1

    def move_hand(self, label: str, target: Vec2, max_step: float = 16.0) -> None:
        while True:
            cur = self.offset[label]
            d = target - cur
            n = (d.x ** 2 + d.y ** 2) ** 0.5
            if n <= max_step:
                self.offset[label] = target
                self.step()
                return
            self.offset[label] = cur + d * (max_step / n)
            self.step()

    def tap(self, finger: Finger, key: KeyCode) -> None:
        """손가락을 key 위로 옮기고, 짧게 아래로 눌렀다 뗀다."""
        lab, digit = finger.hand, finger.digit
        target_center = self.layout.key_center(key)
        if key is KeyCode.SPACE:
            target_center = Vec2(self._home[lab]["thumb"].x, target_center.y)
        self.move_hand(lab, target_center - self._home[lab][digit])
        self.step(6)                                   # 좌표 안정화 대기 (HOVER)
        for dy in PRESS_PROFILE:
            self.press[lab] = {digit: float(dy)}
            self.step()
        self.press[lab] = {}

    def type_text(self, text: str, mistakes: Set[int]) -> None:
        for i, ch in enumerate(text):
            finger = primary_finger_for_char(ch)
            key = key_for_char(ch)
            if finger is None or key is None:
                continue
            if i in mistakes:
                wrong = next((k for k in FINGER_KEYS[finger] if k.char is not None and k is not key), None)
                if wrong is not None:
                    self.tap(finger, wrong)
            self.tap(finger, key)


def run_simulation(cfg: AppConfig, sentences: Sequence[str] = DEFAULT_SENTENCES,
                   mistakes: Sequence[int] = (4, 11), verbose: bool = True) -> SimulationReport:
    """합성 손으로 활성화 -> 문장 타이핑 -> 마우스 클릭 -> 추적 실패까지 전체 흐름을 재생."""
    cal = simulation_calibration(cfg)
    processor = InputProcessor(cfg, cal, (1920, 1080), mouse_enabled=True)
    backend = RecordingInputBackend()
    dispatcher = InputDispatcher(backend, cfg.real_input_allowed)
    practice = PracticeSession(sentences)
    sim = HandSimulator(cfg, processor)
    report = SimulationReport(typed_text="")

    # 1) INACTIVE 상태 확인 후, 양손 펼침 1초 이상 -> ACTIVE
    sim.step(10)
    sim.pose = "open"
    sim.step(int(FPS * (cfg.palm_toggle_hold + 0.3)))
    sim.pose = "typing"
    sim.step(15)

    # 2) 연습 문장 타이핑 (문장마다 오타 몇 개 포함 가능)
    for sentence in sentences:
        sim.type_text(sentence, set(mistakes))
        sim.step(5)

    # 3) 마우스: 오른손을 키보드 위쪽으로 올리고 엄지-검지 핀치로 좌클릭
    sim.move_hand("Right", Vec2(40, -330))
    sim.step(8)
    index_tip = sim._home["Right"]["index"] + sim.offset["Right"]
    sim.thumb_override["Right"] = index_tip + Vec2(6, 2)
    sim.step(4)
    sim.thumb_override["Right"] = None
    sim.step(6)

    # 4) 손 추적 실패 -> 강제 INACTIVE
    sim.pose = "none"
    sim.step(int(FPS * (cfg.tracking_lost_timeout + 0.3)))

    typed: List[str] = []
    for ev in sim.events:
        if isinstance(ev, KeyEvent):
            report.key_events.append(ev)
            dispatcher.handle_key(ev)
            practice.on_key(ev.key, ev.finger, ev.t)
            if ev.key.char is not None:
                typed.append(ev.key.char)
        elif isinstance(ev, MouseEvent):
            report.mouse_events.append(ev)
            dispatcher.handle_mouse(ev)
        elif isinstance(ev, ModeEvent):
            report.mode_events.append(ev)
            dispatcher.set_active(ev.active)

    report.typed_text = "".join(typed)
    report.practice_result = practice.result(sim.t)
    report.backend_calls = len(backend.calls)
    report.frames = sim.frames
    report.duration = sim.t

    if verbose:
        clicks = [e.action.value for e in report.mouse_events if e.action is not MouseAction.MOVE]
        print("=== 시뮬레이션 결과 (웹캠 없이 합성 손 데이터) ===")
        print(f"프레임 {report.frames}개, 시뮬레이션 시간 {report.duration:.1f}s")
        print("모드 전환:", ", ".join(f"{'ACTIVE' if m.active else 'INACTIVE'}({m.reason}@{m.t:.2f}s)"
                                    for m in report.mode_events))
        print(f"입력된 문자열: {report.typed_text!r}")
        print(f"마우스 이벤트: {clicks} (+ 이동 {len(report.mouse_events) - len(clicks)}회)")
        print(f"실제 OS 입력 호출 수: {report.backend_calls} "
              f"({'테스트 모드 - 전송 안 함' if not cfg.real_input_allowed else '실제 입력'})")
        print("--- 연습 모드 채점 ---")
        for line in report.practice_result.summary_lines():
            print(line)
    return report

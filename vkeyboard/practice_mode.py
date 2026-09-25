"""--practice: 타자 연습 / 정확도 측정 모드.

실제 OS 입력은 보내지 않고, 실제 입력 경로와 동일한 KeyboardController 가 만든 KeyEvent 로 채점한다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .keyboard_layout import FINGER_ORDER, Finger, KeyCode, primary_finger_for_char

DEFAULT_SENTENCES = (
    "the quick brown fox",
    "jumps over the lazy dog",
    "pack my box with five dozen jugs",
)


def compute_accuracy(correct: int, total: int) -> float:
    """정확도 = 정답 키 입력 수 / 전체 키 입력 수 (입력이 없으면 0)."""
    return correct / total if total > 0 else 0.0


def compute_wpm(correct_chars: int, seconds: float) -> float:
    """WPM = (정답 문자 수 / 5) / 분. 표준 타자 측정 방식(5글자 = 1단어)."""
    if seconds <= 0:
        return 0.0
    return (correct_chars / 5.0) / (seconds / 60.0)


@dataclass
class FingerStats:
    keystrokes: int = 0
    errors: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors / self.keystrokes if self.keystrokes else 0.0


@dataclass
class PracticeResult:
    sentences: List[str]
    total_keystrokes: int
    correct_keystrokes: int
    errors: int
    accuracy: float
    wpm: float
    elapsed_seconds: float
    finished: bool
    per_finger: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missed_by_expected_finger: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sentences": self.sentences,
            "total_keystrokes": self.total_keystrokes,
            "correct_keystrokes": self.correct_keystrokes,
            "errors": self.errors,
            "accuracy": round(self.accuracy, 4),
            "wpm": round(self.wpm, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "finished": self.finished,
            "per_finger": self.per_finger,
            "missed_by_expected_finger": self.missed_by_expected_finger,
        }

    def save(self, path: str) -> None:
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def summary_lines(self) -> List[str]:
        lines = [
            f"Accuracy: {self.accuracy * 100:.1f}%  ({self.correct_keystrokes}/{self.total_keystrokes})",
            f"Speed: {self.wpm:.1f} WPM   Time: {self.elapsed_seconds:.1f}s",
        ]
        worst = sorted(((k, v) for k, v in self.per_finger.items() if v["keystrokes"] > 0),
                       key=lambda kv: kv[1]["error_rate"], reverse=True)
        for name, v in worst[:3]:
            if v["errors"] > 0:
                lines.append(f"  {name}: {v['error_rate'] * 100:.0f}% errors ({int(v['errors'])}/{int(v['keystrokes'])})")
        return lines


class PracticeSession:
    def __init__(self, sentences: Sequence[str] = DEFAULT_SENTENCES) -> None:
        if not sentences:
            raise ValueError("연습 문장이 필요합니다")
        self.sentences = [s.lower() for s in sentences]
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.position = 0
        self.total = 0
        self.correct = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.finger_stats: Dict[Finger, FingerStats] = {f: FingerStats() for f in FINGER_ORDER}
        self.missed_expected: Dict[Finger, int] = {f: 0 for f in FINGER_ORDER}
        self.last_result: Optional[str] = None

    @property
    def finished(self) -> bool:
        return self.index >= len(self.sentences)

    @property
    def current_sentence(self) -> str:
        return self.sentences[self.index] if not self.finished else ""

    @property
    def expected_char(self) -> Optional[str]:
        s = self.current_sentence
        return s[self.position] if s else None

    def on_key(self, key: KeyCode, finger: Finger, t: float) -> str:
        """키 입력 채점. 'correct' | 'wrong' | 'ignored'"""
        if self.finished:
            return "ignored"
        ch = key.char
        if ch is None:
            return "ignored"   # Shift/Backspace/Enter 등 비문자 키는 채점에서 제외
        if self.start_time is None:
            self.start_time = t
        expected = self.expected_char
        self.total += 1
        self.finger_stats[finger].keystrokes += 1
        if ch == expected:
            self.correct += 1
            self.position += 1
            if self.position >= len(self.current_sentence):
                self.index += 1
                self.position = 0
                if self.finished:
                    self.end_time = t
            self.last_result = "correct"
        else:
            self.finger_stats[finger].errors += 1
            exp_finger = primary_finger_for_char(expected) if expected else None
            if exp_finger is not None:
                self.missed_expected[exp_finger] += 1
            self.last_result = "wrong"
        return self.last_result

    def elapsed(self, now: float) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time is not None else now
        return max(0.0, end - self.start_time)

    def result(self, now: float) -> PracticeResult:
        elapsed = self.elapsed(now)
        per_finger = {f.value: {"keystrokes": s.keystrokes, "errors": s.errors,
                                "error_rate": round(s.error_rate, 4)}
                      for f, s in self.finger_stats.items()}
        return PracticeResult(
            sentences=list(self.sentences),
            total_keystrokes=self.total,
            correct_keystrokes=self.correct,
            errors=self.total - self.correct,
            accuracy=compute_accuracy(self.correct, self.total),
            wpm=compute_wpm(self.correct, elapsed),
            elapsed_seconds=elapsed,
            finished=self.finished,
            per_finger=per_finger,
            missed_by_expected_finger={f.value: n for f, n in self.missed_expected.items() if n},
        )

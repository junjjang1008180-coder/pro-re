"""--log-input: 키/클릭 이벤트를 CSV로 비동기 기록 (설정값 튜닝용)."""
from __future__ import annotations

import csv
import datetime as _dt
import os
import threading
from typing import List

from .events import KeyEvent, MouseAction, MouseEvent

CSV_COLUMNS = ["timestamp", "event_type", "finger_or_hand", "value", "dy_at_fire", "hand_size",
               "tracking_confidence"]


class InputLogger:
    """이벤트를 메모리에 버퍼링했다가 백그라운드 스레드가 주기적으로 파일에 쓴다.

    렌더/입력 스레드는 리스트에 append 만 하므로 FPS 에 영향이 없다.
    """

    def __init__(self, path: str, flush_interval: float = 0.25) -> None:
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        self.path = path
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()
        self._buf: List[list] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = flush_interval
        self.rows_written = 0
        self._thread = threading.Thread(target=self._run, name="InputLogger", daemon=True)
        self._thread.start()

    @staticmethod
    def _ts() -> str:
        return _dt.datetime.now().isoformat(timespec="milliseconds")

    def log_key(self, ev: KeyEvent) -> None:
        self._append([self._ts(), "key", ev.finger.value, ev.key.label, f"{ev.dy:.2f}",
                      f"{ev.hand_size:.1f}", f"{ev.confidence:.3f}"])

    def log_mouse(self, ev: MouseEvent) -> None:
        if ev.action is MouseAction.MOVE:
            return  # 이동은 너무 많아 기록하지 않음
        self._append([self._ts(), "click", "right_hand", ev.action.value, f"{ev.pinch_distance:.2f}",
                      f"{ev.hand_size:.1f}", f"{ev.confidence:.3f}"])

    def _append(self, row: list) -> None:
        with self._lock:
            self._buf.append(row)

    def _flush(self) -> None:
        with self._lock:
            rows, self._buf = self._buf, []
        if rows:
            self._writer.writerows(rows)
            self._file.flush()
            self.rows_written += len(rows)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._flush()
        self._flush()

    def close(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._flush()
            self._file.close()

    def __enter__(self) -> "InputLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

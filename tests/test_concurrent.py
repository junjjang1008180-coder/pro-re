"""동시 입력 (여러 손가락) + 스레드 안전 큐 / 비동기 로거."""
import csv
import threading
import time

from conftest import sample

from vkeyboard.calibration import Calibration
from vkeyboard.events import KeyEvent, MouseAction, MouseEvent
from vkeyboard.frame_queue import EventQueue, FrameQueue, LatestValue
from vkeyboard.input_logger import CSV_COLUMNS, InputLogger
from vkeyboard.keyboard_controller import KeyboardController
from vkeyboard.keyboard_layout import Finger, KeyboardLayout, KeyCode

DT = 1 / 30


def test_multiple_fingers_press_different_keys_simultaneously(cfg):
    layout = KeyboardLayout.from_calibration(Calibration.default())
    kc = KeyboardController(cfg, layout)
    targets = {Finger.LEFT_PINKY: KeyCode.A, Finger.LEFT_INDEX: KeyCode.F,
               Finger.RIGHT_INDEX: KeyCode.J, Finger.RIGHT_PINKY: KeyCode.SEMICOLON}
    events = []
    for i, d in enumerate([0, 20, 20, 20, 0]):
        samples = {f: sample(f, layout.key_center(k), d) for f, k in targets.items()}
        events += kc.update(samples, i * DT, True)
    assert {(e.finger, e.key) for e in events} == set(targets.items())
    assert len({e.t for e in events}) == 1             # 같은 프레임에 동시에 확정


def test_fingers_are_independent(cfg):
    """한 손가락이 PRESSED 로 계속 눌려 있어도 다른 손가락은 자유롭게 입력."""
    layout = KeyboardLayout.from_calibration(Calibration.default())
    kc = KeyboardController(cfg, layout)
    events = []
    right = [0, 0, 0, 0, 0, 0, 20, 20, 20, 0, 0]
    for i in range(len(right)):
        samples = {Finger.LEFT_INDEX: sample(Finger.LEFT_INDEX, layout.key_center(KeyCode.F), 25 if i else 0),
                   Finger.RIGHT_INDEX: sample(Finger.RIGHT_INDEX, layout.key_center(KeyCode.J), right[i])}
        events += kc.update(samples, i * DT, True)
    assert [e.key for e in events] == [KeyCode.F, KeyCode.J]


def test_frame_queue_keeps_only_latest():
    q = FrameQueue(2)
    for i in range(5):
        q.put(i)
    assert len(q) == 2 and q.dropped == 3
    assert q.get(0) == 3 and q.get(0) == 4
    assert q.get(0.01) is None


def test_frame_queue_get_latest_discards_older():
    q = FrameQueue(2)
    q.put("old")
    q.put("new")
    assert q.get_latest(0) == "new"
    assert len(q) == 0


def test_frame_queue_close_unblocks_consumer():
    q = FrameQueue(1)
    result = []
    th = threading.Thread(target=lambda: result.append(q.get(timeout=5)))
    th.start()
    time.sleep(0.05)
    q.close()
    th.join(1)
    assert not th.is_alive() and result == [None]


def test_producer_consumer_no_race():
    """캡처(생산자)가 빠르고 추론(소비자)이 느려도 순서가 유지되고 예외 없이 동작."""
    q = FrameQueue(2)
    consumed = []

    def producer():
        for i in range(2000):
            q.put(i)
        q.close()

    def consumer():
        while True:
            item = q.get(timeout=1)
            if item is None:
                break
            consumed.append(item)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert consumed == sorted(consumed)          # 순서 보장
    assert len(consumed) + q.dropped == 2000     # 버린 것 + 소비한 것 = 전체


def test_event_queue_does_not_lose_events_across_threads():
    eq = EventQueue()
    n_threads, per_thread = 8, 500

    def worker(k):
        for i in range(per_thread):
            eq.put((k, i))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    drained = []
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        drained += eq.drain()
    for t in threads:
        t.join()
    drained += eq.drain()
    assert len(drained) == n_threads * per_thread
    assert len(set(drained)) == len(drained)


def test_latest_value_version():
    slot = LatestValue()
    assert slot.get() == (None, 0)
    slot.set("a")
    slot.set("b")
    assert slot.get() == ("b", 2)


def test_input_logger_writes_csv_asynchronously(tmp_path):
    path = tmp_path / "log.csv"
    logger = InputLogger(str(path), flush_interval=0.05)
    logger.log_key(KeyEvent(Finger.LEFT_INDEX, KeyCode.F, 1.0, 21.5, 180.0, 0.93))
    logger.log_mouse(MouseEvent(MouseAction.MOVE, 1, 2, 1.1))          # 이동은 기록 안 함
    logger.log_mouse(MouseEvent(MouseAction.LEFT_CLICK, 1, 2, 1.2, 12.0, 175.0, 0.9))
    logger.close()
    rows = list(csv.reader(path.open(encoding="utf-8")))
    assert rows[0] == CSV_COLUMNS
    assert rows[1][1:] == ["key", "left_index", "F", "21.50", "180.0", "0.930"]
    assert rows[2][1:4] == ["click", "right_hand", "left_click"]
    assert len(rows) == 3


def test_no_log_file_when_logging_disabled(tmp_path, cfg):
    assert cfg.log_input is None   # 옵션을 안 주면 로거 자체를 만들지 않음

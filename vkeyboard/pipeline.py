"""캡처 / 추론 스레드 (렌더링과 입력 전송은 메인 스레드).

   [CaptureThread] --FrameQueue(최신 2개)--> [InferenceThread] --EventQueue/LatestValue--> [Main]
        |                                                                                   ^
        +---------------------- LatestValue(표시용 최신 프레임) ----------------------------+

OpenCV 의 read/resize 와 MediaPipe 추론은 GIL 을 풀기 때문에 파이썬 스레드로도 병렬로 동작한다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import CAMERA_FAIL_LIMIT, FALLBACK_RESOLUTIONS
from .frame_queue import EventQueue, FrameQueue, LatestValue


@dataclass
class CapturedFrame:
    frame_id: int
    t: float          # time.monotonic()
    image: object     # numpy BGR (캡처 해상도, 거울 모드 반전 완료)


def open_camera(index: int, width: int, height: int):
    """카메라를 열고 요청 해상도를 시도. 미지원이면 가장 가까운 지원 해상도로 자동 폴백.

    반환: (cap, actual_width, actual_height)
    """
    import sys

    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    cap = None
    for api in backends:
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            break
        cap.release()
        cap = None
    if cap is None:
        raise RuntimeError(f"웹캠(index {index})을 열 수 없습니다")

    # 4K 등 고해상도는 MJPG 가 아니면 30fps 가 안 나오는 카메라가 많다
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    candidates = [(width, height)] + [r for r in FALLBACK_RESOLUTIONS
                                      if r[0] * r[1] < width * height and r != (width, height)]
    actual = (0, 0)
    for w, h in candidates:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()
        if not ok or frame is None:
            # 해상도 미지원이면 카메라가 다른 해상도로라도 프레임을 준다.
            # 읽기 자체가 실패 = 장치 사용 중/권한 차단 (MSMF 는 1회 타임아웃이 ~10초) -> 즉시 포기
            break
        actual = (frame.shape[1], frame.shape[0])
        if actual == (w, h):
            break
        if actual[0] * actual[1] <= w * h:
            # 카메라가 요청보다 작은 해상도로 스스로 맞춘 경우 = 지원하는 가장 가까운 해상도
            break
    if actual == (0, 0):
        cap.release()
        raise RuntimeError(f"웹캠(index {index})에서 프레임을 읽을 수 없습니다")
    if actual != (width, height):
        print(f"[안내] {width}x{height} 미지원 → {actual[0]}x{actual[1]}으로 자동 전환됨")
    return cap, actual[0], actual[1]


class CaptureThread(threading.Thread):
    """웹캠에서 프레임만 읽어 큐에 넣는다."""

    def __init__(self, cap, frame_queue: FrameQueue, display_slot: LatestValue, mirror: bool = True) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self.cap = cap
        self.frame_queue = frame_queue
        self.display_slot = display_slot
        self.mirror = mirror
        self._stop_evt = threading.Event()
        self.error: Optional[str] = None
        self.fps = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        import cv2

        frame_id = 0
        fails = 0
        last = time.monotonic()
        try:
            while not self._stop_evt.is_set():
                ok, img = self.cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= CAMERA_FAIL_LIMIT:
                        self.error = "웹캠 프레임 읽기 연속 실패 (연결 끊김)"
                        break
                    time.sleep(0.01)
                    continue
                fails = 0
                if self.mirror:
                    img = cv2.flip(img, 1)
                now = time.monotonic()
                item = CapturedFrame(frame_id, now, img)
                self.frame_queue.put(item)
                self.display_slot.set(item)
                frame_id += 1
                dt = now - last
                last = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        except Exception as e:  # noqa: BLE001
            self.error = f"캡처 스레드 예외: {e}"
        finally:
            self.frame_queue.close()
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass


class InferenceThread(threading.Thread):
    """큐에서 프레임을 꺼내 1280x720 리사이즈 -> 손 랜드마크 추론 -> 손가락 상태 판정."""

    def __init__(self, frame_queue: FrameQueue, tracker, scaler, processor,
                 event_queue: EventQueue, snapshot_slot: LatestValue) -> None:
        super().__init__(name="InferenceThread", daemon=True)
        self.frame_queue = frame_queue
        self.tracker = tracker
        self.scaler = scaler
        self.processor = processor
        self.event_queue = event_queue
        self.snapshot_slot = snapshot_slot
        self._stop_evt = threading.Event()
        self.error: Optional[str] = None
        self.fps = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        last = None
        try:
            while not self._stop_evt.is_set():
                item: Optional[CapturedFrame] = self.frame_queue.get_latest(timeout=0.1)
                if item is None:
                    if self.frame_queue.closed:
                        break
                    continue
                small = self.scaler.resize_for_inference(item.image)
                hands = self.tracker.detect(small, int(item.t * 1000))
                snap, events = self.processor.process(hands, item.t)
                now = time.monotonic()
                if last is not None and now > last:
                    inst = 1.0 / (now - last)
                    self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
                last = now
                snap.frame_id = item.frame_id
                snap.infer_fps = self.fps
                self.event_queue.put_many(events)
                self.snapshot_slot.set(snap)
        except Exception as e:  # noqa: BLE001
            self.error = f"추론 스레드 예외: {e}"


class FpsCounter:
    def __init__(self) -> None:
        self.fps = 0.0
        self._last: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = 0.9 * self.fps + 0.1 * inst if self.fps else inst
        self._last = now
        return self.fps


def display_size(width: int, height: int, max_width: int) -> Tuple[int, int]:
    if width <= max_width:
        return width, height
    return max_width, int(round(height * max_width / width))

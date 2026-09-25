"""스레드 간 데이터 전달용 자료구조 (뮤텍스 + 조건변수)."""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class FrameQueue(Generic[T]):
    """최신 항목 maxsize 개만 유지하는 스레드 안전 큐. 가득 차면 가장 오래된 항목을 버린다."""

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError("maxsize >= 1")
        self._items: Deque[T] = deque()
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item: T) -> int:
        """항목 추가. 버려진 오래된 항목 수를 반환."""
        with self._cond:
            if self._closed:
                return 0
            dropped = 0
            while len(self._items) >= self._maxsize:
                self._items.popleft()
                dropped += 1
            self._items.append(item)
            self.dropped += dropped
            self._cond.notify()
            return dropped

    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """가장 오래된 항목을 꺼낸다. 타임아웃/종료 시 None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            if self._items:
                return self._items.popleft()
            return None

    def get_latest(self, timeout: Optional[float] = None) -> Optional[T]:
        """가장 최신 항목만 꺼내고 나머지는 버린다."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            if not self._items:
                return None
            item = self._items.pop()
            self.dropped += len(self._items)
            self._items.clear()
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)


class LatestValue(Generic[T]):
    """가장 최근 값 하나만 보관하는 슬롯 (버전 번호로 새 값 여부 확인)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._version = 0

    def set(self, value: T) -> None:
        with self._lock:
            self._value = value
            self._version += 1

    def get(self) -> Tuple[Optional[T], int]:
        with self._lock:
            return self._value, self._version


class EventQueue(Generic[T]):
    """무제한 스레드 안전 이벤트 큐 (입력 이벤트는 절대 버리지 않는다)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[T] = []

    def put(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def put_many(self, items: List[T]) -> None:
        if items:
            with self._lock:
                self._items.extend(items)

    def drain(self) -> List[T]:
        with self._lock:
            items, self._items = self._items, []
            return items

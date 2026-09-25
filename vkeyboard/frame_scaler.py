"""캡처 해상도(4K) <-> 추론 해상도(1280x720) 좌표 변환과 리사이즈를 한 곳에 모은 모듈."""
from __future__ import annotations

from .config import INFER_HEIGHT, INFER_WIDTH
from .geometry import Vec2


class FrameScaler:
    def __init__(self, capture_width: int, capture_height: int,
                 infer_width: int = INFER_WIDTH, infer_height: int = INFER_HEIGHT) -> None:
        if min(capture_width, capture_height, infer_width, infer_height) <= 0:
            raise ValueError("해상도는 양수여야 합니다")
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.infer_width = infer_width
        self.infer_height = infer_height
        # 예: 3840/1280 = 3.0, 2160/720 = 3.0
        self.scale_x = capture_width / infer_width
        self.scale_y = capture_height / infer_height

    # --- 좌표 변환 ---------------------------------------------------------
    def infer_to_display(self, p: Vec2) -> Vec2:
        return Vec2(p.x * self.scale_x, p.y * self.scale_y)

    def display_to_infer(self, p: Vec2) -> Vec2:
        return Vec2(p.x / self.scale_x, p.y / self.scale_y)

    def normalized_to_infer(self, nx: float, ny: float) -> Vec2:
        """MediaPipe 정규화 좌표(0~1) -> 추론 픽셀 좌표."""
        return Vec2(nx * self.infer_width, ny * self.infer_height)

    def infer_to_display_pt(self, p: Vec2) -> tuple:
        """OpenCV 그리기용 정수 좌표."""
        return self.infer_to_display(p).as_int()

    def scale_length(self, length: float) -> float:
        """추론 좌표계 길이 -> 표시 좌표계 길이 (두 축 평균)."""
        return length * (self.scale_x + self.scale_y) * 0.5

    # --- 리사이즈 ----------------------------------------------------------
    def resize_for_inference(self, frame):
        """캡처 프레임을 추론 해상도로 다운스케일 (INTER_AREA)."""
        import cv2

        h, w = frame.shape[:2]
        if w == self.infer_width and h == self.infer_height:
            return frame
        return cv2.resize(frame, (self.infer_width, self.infer_height), interpolation=cv2.INTER_AREA)

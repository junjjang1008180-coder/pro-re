"""MediaPipe Tasks HandLandmarker 로 양손 21점 랜드마크를 추론한다.

입력: 추론 해상도(1280x720)로 줄인 BGR 프레임 (거울 모드로 이미 좌우 반전됨)
출력: HandObservation 목록 (좌표는 1280x720 추론 좌표계 픽셀)
"""
from __future__ import annotations

import os
import urllib.request
from typing import List

from .config import MODEL_URL, AppConfig
from .frame_scaler import FrameScaler
from .geometry import HandObservation


def ensure_model(path: str, url: str = MODEL_URL) -> str:
    """모델 파일이 없으면 공식 배포 주소에서 내려받는다."""
    if os.path.exists(path):
        return path
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    print(f"[정보] 손 랜드마크 모델이 없어 다운로드합니다: {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"[정보] 모델 저장 완료: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


class HandTracker:
    def __init__(self, cfg: AppConfig, scaler: FrameScaler) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.scaler = scaler
        model_path = ensure_model(cfg.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=cfg.min_tracking_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr_infer, timestamp_ms: int) -> List[HandObservation]:
        import cv2

        ts = max(int(timestamp_ms), self._last_ts + 1)   # VIDEO 모드는 단조 증가 타임스탬프 필요
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr_infer, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        hands: List[HandObservation] = []
        for i, lms in enumerate(result.hand_landmarks or []):
            label, score = "Right", 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            pts = [self.scaler.normalized_to_infer(lm.x, lm.y) for lm in lms]
            hands.append(HandObservation(label, pts, score))
        return hands

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass

"""MediaPipe Tasks HandLandmarker 로 양손 21점 랜드마크를 추론한다.

입력: 추론 해상도(1280x720)로 줄인 BGR 프레임 (거울 모드로 이미 좌우 반전됨)
출력: HandObservation 목록 (좌표는 1280x720 추론 좌표계 픽셀)
"""
from __future__ import annotations

import os
import shutil
import urllib.request
from typing import List, Optional, Tuple

from .config import (MAX_HANDS_DETECT, MAX_POSES, MIN_MODEL_BYTES, MODEL_DOWNLOAD_TIMEOUT, MODEL_URL,
                     POSE_EVERY_N_FRAMES, POSE_MODEL_URL, AppConfig)
from .frame_scaler import FrameScaler
from .body_owner import POSE_POINTS, PoseObservation
from .geometry import HandObservation


def ensure_model(path: str, url: str = MODEL_URL, timeout: float = MODEL_DOWNLOAD_TIMEOUT) -> str:
    """모델 파일이 없거나 손상(너무 작음)됐으면 공식 배포 주소에서 내려받는다."""
    if os.path.exists(path) and os.path.getsize(path) >= MIN_MODEL_BYTES:
        return path
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    print(f"[정보] 손 랜드마크 모델이 없어 다운로드합니다: {url}")
    tmp = path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        if os.path.getsize(tmp) < MIN_MODEL_BYTES:
            raise RuntimeError("받은 파일이 너무 작습니다 (네트워크/프록시 차단 가능성)")
        os.replace(tmp, path)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"모델 다운로드 실패: {e}\n"
                           f"  → 인터넷 연결을 확인하거나, 위 주소에서 직접 받아 {path} 에 두세요") from e
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
        # 경로 대신 바이트로 넘긴다: MediaPipe(C++)는 Windows 에서 한글 등 비ASCII 경로를 열지 못한다
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS_DETECT,     # 여러 손을 찾은 뒤 HandSelector 가 내 손 2개만 고른다
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=cfg.min_tracking_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self.handedness_mode = cfg.handedness
        self.presence_threshold = cfg.min_tracking_confidence
        self._last_ts = -1

        # 내 손만 인식(몸 기준)용 포즈 인식. 실패해도 프로그램은 크기 기준으로 계속 동작한다.
        self._pose = None
        self._pose_frame = 0
        self._last_poses: Optional[List[PoseObservation]] = None
        if cfg.hand_lock in ("body", "strict"):
            try:
                pose_path = ensure_model(cfg.pose_model_path, POSE_MODEL_URL)
                with open(pose_path, "rb") as f:
                    pose_bytes = f.read()
                self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_buffer=pose_bytes),
                    running_mode=vision.RunningMode.VIDEO, num_poses=MAX_POSES))
            except Exception as e:  # noqa: BLE001
                print(f"[경고] 몸(포즈) 인식을 쓸 수 없어 '내 손만 인식'을 손 크기 기준으로 합니다: {e}")

    def detect(self, frame_bgr_infer, timestamp_ms: int) -> List[HandObservation]:
        return self.detect_all(frame_bgr_infer, timestamp_ms)[0]

    def detect_all(self, frame_bgr_infer, timestamp_ms: int
                   ) -> Tuple[List[HandObservation], Optional[List[PoseObservation]]]:
        """(손 목록, 사람 포즈 목록 또는 None=포즈 인식 안 씀)."""
        import cv2

        ts = max(int(timestamp_ms), self._last_ts + 1)   # VIDEO 모드는 단조 증가 타임스탬프 필요
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr_infer, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        poses = self._detect_poses(image, ts)
        hands: List[HandObservation] = []
        for i, lms in enumerate(result.hand_landmarks or []):
            label, score = "Right", 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                cat = result.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            pts = [self.scaler.normalized_to_infer(lm.x, lm.y) for lm in lms]
            # MediaPipe 는 손 존재 신뢰도(min_hand_presence_confidence)를 통과한 손만 돌려준다.
            # handedness 점수는 '왼손/오른손 판별' 확신도라, 손등이 보이는 타이핑 자세에서는 낮아진다.
            # 위치 기준 좌우 판별 모드에서는 이 점수로 손가락을 버리지 않도록 통과 기준 이상으로 올려 준다.
            if self.handedness_mode == "position":
                score = max(score, self.presence_threshold)
            hands.append(HandObservation(label, pts, score))
        return hands, poses

    def _detect_poses(self, image, ts: int) -> Optional[List[PoseObservation]]:
        if self._pose is None:
            return None
        self._pose_frame += 1
        if self._last_poses is not None and self._pose_frame % POSE_EVERY_N_FRAMES != 0:
            return self._last_poses
        result = self._pose.detect_for_video(image, ts)
        poses: List[PoseObservation] = []
        for lms in result.pose_landmarks or []:
            pose = PoseObservation()
            for idx in POSE_POINTS:
                lm = lms[idx]
                pose.points[idx] = self.scaler.normalized_to_infer(lm.x, lm.y)
                vis = getattr(lm, "visibility", None)
                pose.visibility[idx] = 1.0 if vis is None else float(vis)   # 값이 없으면 보이는 것으로
            poses.append(pose)
        self._last_poses = poses
        return poses

    def close(self) -> None:
        for task in (self._landmarker, self._pose):
            try:
                if task is not None:
                    task.close()
            except Exception:  # noqa: BLE001
                pass

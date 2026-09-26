"""전역 설정값.

- 모듈 상수: 기본값 (코드 수정으로 바꾸는 값)
- AppConfig: 실행 시점 설정 (CLI 인자로 덮어쓸 수 있음)

좌표/거리 관련 값은 모두 INFER_WIDTH x INFER_HEIGHT(1280x720) 추론 좌표계 기준이다.
캡처 해상도(4K)가 바뀌어도 입력 감도가 흔들리지 않도록, 입력 판정은 항상 추론
좌표계에서만 수행한다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# 해상도 (캡처는 4K, 추론/판정은 1280x720 고정)
# ---------------------------------------------------------------------------
CAPTURE_WIDTH = 3840
CAPTURE_HEIGHT = 2160
INFER_WIDTH = 1280
INFER_HEIGHT = 720

# 4K 미지원 시 순서대로 시도하는 폴백 해상도 목록
FALLBACK_RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (3840, 2160),
    (2560, 1440),
    (1920, 1080),
    (1280, 720),
    (640, 480),
)

# ---------------------------------------------------------------------------
# 키 입력 판정 (1280x720 좌표계 기준) — "오입력 최소화" 우선 기본값
# ---------------------------------------------------------------------------
PRESS_DISTANCE = 18.0            # 15 -> 18: 우발적 입력 방지
RELEASE_DISTANCE = 10.0          # 8 -> 10: 손 떨림 재입력 방지
MIN_DOWN_FRAMES = 3              # 2 -> 3: 순간 노이즈로 인한 오입력 방지
KEY_COOLDOWN = 0.35              # 0.30 -> 0.35: 연타 오류 방지 (초)
SMOOTHING_FRAMES = 5             # 4 -> 5: 좌표 흔들림 억제 강화
MIN_TRACKING_CONFIDENCE = 0.75   # 0.70 -> 0.75: 불안정한 추적 결과 필터링
MAX_ALLOWED_JUMP = 60.0          # 80 -> 60: 튀는 좌표 더 엄격히 차단

# ---------------------------------------------------------------------------
# 마우스 클릭 판정
# ---------------------------------------------------------------------------
CLICK_PINCH_DISTANCE = 22.0      # 핀치 판정 거리
CLICK_COOLDOWN = 0.40            # 초
DOUBLE_CLICK_WINDOW = 0.45       # 초

# ---------------------------------------------------------------------------
# 보조 튜닝 상수
# ---------------------------------------------------------------------------
REFERENCE_HAND_SIZE = 180.0      # 손목~중지 끝 기준 거리(px). 이 크기일 때 임계값 배율 1.0
HAND_SCALE_MIN = 0.6             # 손 크기 배율 하한 (멀리 있는 손)
HAND_SCALE_MAX = 1.8             # 손 크기 배율 상한 (가까이 있는 손)
HAND_SIZE_SMOOTHING = 0.2        # 손 크기 EMA 계수 (손가락 굽힘으로 인한 흔들림 억제)
MAX_SPEED = 2400.0               # px/s, 이보다 빠른 움직임은 이상치로 간주
EDGE_MARGIN = 12.0               # 화면 가장자리 불안정 영역(px)
FINGER_OVERLAP_DISTANCE = 12.0   # 손가락 끝끼리 이보다 가까우면 겹침으로 간주(손 크기 배율 적용)
OUTLIER_RESET_FRAMES = 4         # 연속 이상치 N회면 새 위치로 재동기화
TRACKING_GAP_RESET = 0.25        # 이 시간(초) 이상 관측이 끊기면 필터 초기화

PRESS_SMOOTHING_FRAMES = 2       # 누름 깊이(dy) 신호는 가볍게만 평균 -> 짧은 '톡' 누름이 뭉개지지 않게
MIN_DOWN_TIME = 0.08             # 임계값 통과 유지 최소 시간(초). 저 FPS 카메라에서는 필요 프레임 수를 줄여 줌
KEY_SNAP_RADIUS = 0.6            # 손끝이 담당 키 밖이어도 키 중심에서 이 거리(키 1칸 대비) 안이면 그 키로 인정
PRESS_MAX_DURATION = 0.6         # PRESSING 상태로 이 시간 안에 확정 못 하면 취소(느린 드리프트)
BASELINE_ALPHA = 0.15            # HOVER 중 기준선(손가락 휴지 위치) 적응 속도
RELEASED_HOLD_TIME = 0.15        # RELEASED 상태 표시 유지 시간
KEY_HIGHLIGHT_TIME = 0.10        # 키 확정 하이라이트 시간(약 100ms)
MOUSE_FLASH_TIME = 0.15          # 미니 마우스 클릭 표시 시간

PALM_TOGGLE_HOLD = 1.0           # 양손 펼침 유지 시간(초) -> ACTIVE/INACTIVE 전환
TRACKING_LOST_TIMEOUT = 0.7      # ACTIVE 중 손이 이 시간 이상 사라지면 INACTIVE로 강제 전환

CLICK_RELEASE_FACTOR = 1.4       # 핀치 해제 거리 = 핀치 거리 * 배율 (히스테리시스)
PINCH_MIN_FRAMES = 2             # 핀치가 N프레임 연속 유지돼야 인정(디바운스)
CLICK_MAX_DURATION = 0.6         # 이보다 오래 붙이고 있으면 클릭이 아님
DRAG_START_DISTANCE = 12.0       # 핀치 상태에서 이만큼 이동하면 드래그 시작 (25 -> 12: 드래그 감도 향상)
DRAG_HOLD_TIME = 0.4             # 핀치를 이만큼 유지하면 움직이지 않아도 드래그 시작 (꾹 눌러 끌기)
DRAG_RELEASE_FACTOR = 1.8        # 드래그 중 해제 거리 배율 (클릭보다 크게: 빨리 움직일 때 엄지가 살짝 벌어져도 유지)
DRAG_RELEASE_FRAMES = 3          # 드래그 중 해제에 필요한 연속 프레임 수
DOUBLE_CLICK_MIN_INTERVAL = 0.06 # 더블클릭 두 번째 클릭 최소 간격(디바운스)
MOUSE_REGION = (0.15, 0.10, 0.85, 0.70)  # 추론 프레임 정규화 좌표 (x0, y0, x1, y1) -> 화면 전체로 매핑
ONE_EURO_MIN_CUTOFF = 1.2
ONE_EURO_BETA = 0.02

# ---------------------------------------------------------------------------
# 파이프라인 / 표시
# ---------------------------------------------------------------------------
FRAME_QUEUE_SIZE = 2             # 최신 프레임 1~2개만 유지
DISPLAY_MAX_WIDTH = 1920         # 창에 표시할 최대 폭 (그리기는 캡처 해상도에서 수행)
CAMERA_FAIL_LIMIT = 30           # 연속 읽기 실패 허용 횟수

# 기본 파일 경로는 실행 위치(현재 폴더)가 아니라 프로젝트 폴더 기준 -> 어디서 실행해도 같은 파일을 쓴다
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
DEFAULT_POSE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "pose_landmarker_lite.task")
DEFAULT_CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "calibration.json")
DEFAULT_PRACTICE_OUTPUT = os.path.join(PROJECT_ROOT, "practice_result.json")
MIN_MODEL_BYTES = 1_000_000      # 이보다 작은 모델 파일은 다운로드 실패/손상으로 간주
MODEL_DOWNLOAD_TIMEOUT = 30.0    # 초

# 안전/안정성
FRAME_STALL_TIMEOUT = 1.0        # ACTIVE 중 판정 결과가 이 시간 이상 안 오면(영상 멈춤) 입력 중지
MANUAL_ACTIVATION_GRACE = 3.0    # 'a' 키로 켠 직후, 손을 카메라 앞으로 가져올 시간 (추적 실패 판정 유예)
HOTKEY_SUPPRESS_AFTER_SEND = 0.5 # 가상 키를 실제로 보낸 직후 창 단축키(a, c, ...) 무시: 카메라 창에 자기 입력이 들어가는 것 방지
MOUSE_EXIT_SETTLE = 0.4         # 마우스 모드에서 나온 직후 키 입력 막는 시간 (손을 내리는 동작이 키로 잡히지 않게)
MOUSE_LOST_TIME = 0.6           # 마우스 모드 중 손이 이 시간 이상 안 보이면 마우스 모드 해제
MAX_HANDS_DETECT = 4             # MediaPipe 가 찾는 손 수 (다른 사람 손이 내 손 자리를 빼앗지 않도록 2보다 크게)
HAND_SIZE_MIN_RATIO = 0.6        # 등록한 내 손 크기 대비 허용 범위
HAND_SIZE_MAX_RATIO = 1.7
SMALL_HAND_RATIO = 0.6           # 가장 큰 손의 이 비율보다 작은 손(멀리 있는 사람)은 무시
HAND_SELECT_KEEP_DIST = 0.2      # 직전 손 위치에서 이 거리(화면 폭 비율) 안이면 같은 손으로 우선
HAND_SIZE_ADAPT = 0.01           # 등록 크기가 현재 손 크기를 따라가는 속도 (카메라 거리 변화 적응)
# 내 손만 인식 (추적 잠금, 기본): 등록한 손에서 이어지는 손만 사용 (hand_lock.py)
TRACK_BASE_DIST = 50.0           # 프레임 사이 허용 이동 거리 기본값(px)
TRACK_SPEED = 3000.0             # + 경과 시간 x 이 속도(px/s) 까지 이동 허용 (빠른 손동작)
REACQUIRE_TIME = 2.0             # 손이 사라진 뒤 이 시간 안에
REACQUIRE_RADIUS = 0.25          # 사라진 위치 근처(화면 폭 비율)에서 다시 나타나면 같은 손으로 이어 받음
CLAIM_TIME = 0.6                 # 그 밖엔 손바닥을 이 시간 동안 펴 보이면 비어 있는 자리로 다시 등록

# 내 손만 인식 (몸 기준, --hand-lock body): 손목이 사용자 팔 끝에 붙어 있는 손만 사용
POSE_EVERY_N_FRAMES = 2          # 포즈 인식은 N프레임마다 (CPU 절약, 사이 프레임은 직전 결과 사용)
MAX_POSES = 3                    # 동시에 찾을 사람 수
POSE_MIN_VISIBILITY = 0.3        # 포즈 점이 이보다 안 보이면 없는 것으로 취급
OWNER_WRIST_TOL = 0.45           # 손 손목 ~ 팔 손목 허용 거리 (어깨너비 배율)
OWNER_ELBOW_TOL = 1.1            # 팔 손목이 가려졌을 때: 손 손목 ~ 팔꿈치 허용 거리 (어깨너비 배율)
OWNER_MIN_TOL_PX = 40.0          # 허용 거리 최소값(px)
OWNER_KEEP_DIST = 0.3            # 등록한 사용자 몸 위치에서 이 거리(화면 폭 비율) 안의 사람만 사용자로 인정
OWNER_WIDTH_RANGE = (0.6, 1.6)   # 사용자 어깨너비 변화 허용 범위
HAND_LABEL_MEMORY = 0.25         # 한 손만 보일 때, 직전 손 위치에서 이 거리(화면 폭 비율) 안이면 같은 손으로 유지
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class AppConfig:
    """실행 시점 설정. 기본값은 위 상수, CLI 인자로 덮어쓴다."""

    camera_index: int = 0
    capture_width: int = CAPTURE_WIDTH
    capture_height: int = CAPTURE_HEIGHT
    infer_width: int = INFER_WIDTH
    infer_height: int = INFER_HEIGHT

    test_mode: bool = True
    enable_real_input: bool = False
    calibration_file: str = DEFAULT_CALIBRATION_FILE
    log_input: Optional[str] = None
    practice: bool = False
    practice_output: Optional[str] = DEFAULT_PRACTICE_OUTPUT
    model_path: str = DEFAULT_MODEL_PATH
    simulate: bool = False
    beep: bool = False
    overlay: bool = True              # 바탕화면에 항상 위 HUD(미니 키보드/마우스/상태) 표시
    overlay_corner: str = "br"        # br | bl | tr | tl
    overlay_scale: float = 1.3
    sensitivity: str = "normal"       # low | normal | high (키 누름 민감도 프리셋)
    hand_lock: str = "track"          # 내 손만 인식: track | body | strict | size | off
    pose_model_path: str = DEFAULT_POSE_MODEL_PATH
    mouse_exclusive: bool = True      # 마우스 모드 중 키보드 입력(양손) 전부 멈춤. False 면 왼손은 계속 타이핑
    handedness: str = "position"      # position: 화면 위치로 좌/우 손 판별 | model: MediaPipe 라벨 사용
    run_seconds: float = 0.0          # >0 이면 N초 후 자동 정상 종료 (자동 점검/데모용)

    press_distance: float = PRESS_DISTANCE
    release_distance: float = RELEASE_DISTANCE
    min_down_frames: int = MIN_DOWN_FRAMES
    key_cooldown: float = KEY_COOLDOWN
    smoothing_frames: int = SMOOTHING_FRAMES
    min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE
    max_allowed_jump: float = MAX_ALLOWED_JUMP

    click_pinch_distance: float = CLICK_PINCH_DISTANCE
    click_cooldown: float = CLICK_COOLDOWN
    double_click_window: float = DOUBLE_CLICK_WINDOW

    reference_hand_size: float = REFERENCE_HAND_SIZE
    max_speed: float = MAX_SPEED
    edge_margin: float = EDGE_MARGIN
    finger_overlap_distance: float = FINGER_OVERLAP_DISTANCE
    press_max_duration: float = PRESS_MAX_DURATION
    palm_toggle_hold: float = PALM_TOGGLE_HOLD
    tracking_lost_timeout: float = TRACKING_LOST_TIMEOUT

    @property
    def real_input_allowed(self) -> bool:
        """TEST_MODE=false 이고 ENABLE_REAL_INPUT=true 일 때만 실제 입력 허용.

        연습 모드와 시뮬레이션은 항상 로컬 채점만 하므로 실제 입력을 보내지 않는다.
        """
        return (not self.test_mode) and self.enable_real_input and not self.practice and not self.simulate


# 민감도 프리셋: (PRESS_DISTANCE, RELEASE_DISTANCE, MIN_DOWN_FRAMES)
SENSITIVITY_PRESETS = {
    "low": (22.0, 12.0, 3),       # 오입력 최소화 (살짝 깊게 눌러야 함)
    "normal": (PRESS_DISTANCE, RELEASE_DISTANCE, MIN_DOWN_FRAMES),
    "high": (13.0, 7.0, 2),       # 얕고 빠른 누름도 인식 (오입력 조금 증가)
}


def hand_scale(hand_size: float, reference: float = REFERENCE_HAND_SIZE) -> float:
    """손 크기에 따른 임계값 배율 (카메라 거리 보정)."""
    if hand_size <= 0 or reference <= 0:
        return 1.0
    return max(HAND_SCALE_MIN, min(HAND_SCALE_MAX, hand_size / reference))

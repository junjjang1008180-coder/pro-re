# CLAUDE.md

웹캠으로 양손 10개 손가락을 추적해 허공의 QWERTY 키보드를 "아래로 짧게 누르기"로 타이핑하고,
오른손 제스처로 마우스를 조작하는 **Python** 프로그램. 사용자 문서는 `README.md`(한국어).

## 명령어

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt   # 최초 설치
python -m pytest                                  # 전체 테스트 (웹캠/모델 불필요, 1초 내외)
python main.py --simulate --practice              # 웹캠 없이 합성 손으로 전체 판정 경로 검증
python main.py --capture-width 1920 --capture-height 1080   # 실제 실행 (기본 = 테스트 모드)
```

- 변경 후에는 `pytest` 와 `--simulate` 둘 다 돌린다. 시뮬레이션 기준값: 입력 문자열에 의도적 오타 4개,
  정확도 94.9% (74/78), 마우스 `left_click` 1회, 실제 OS 입력 0건. 이 값이 바뀌면 원인을 확인할 것.
- 테스트에서 `while ... is None:` 같은 무한 대기 루프를 쓰지 말 것 (반복 횟수 상한을 둔다).
- `python main.py --run-seconds N` 으로 N초 후 자동 종료 (자동 점검용).

## 구조 (`vkeyboard/`)

스레드 파이프라인: `CaptureThread` → `FrameQueue`(최신 2개) → `InferenceThread`(리사이즈 + MediaPipe +
`InputProcessor`) → `EventQueue`/스냅샷 → 메인 스레드(`app.py`: 렌더링 + `InputDispatcher` 로 OS 입력).

| 파일 | 역할 |
|---|---|
| `config.py` | 모든 상수 + `AppConfig`(CLI 로 덮어씀), 민감도 프리셋 |
| `cli_options.py` | argparse → `AppConfig` |
| `hand_tracker.py` | MediaPipe Tasks `HandLandmarker` (모델 `models/hand_landmarker.task` 자동 다운로드) |
| `frame_scaler.py` | 캡처 해상도 ↔ 1280×720 추론 좌표 변환, 리사이즈 |
| `finger_tracker.py` / `smoothing.py` | 손끝 안정화, 이상치(점프/속도/가장자리/겹침/저신뢰도) 제거, 좌우 손 정리 |
| `input_state.py` | 손가락 상태 머신 HOVER→PRESSING→PRESSED→RELEASED |
| `keyboard_layout.py` | 키 배치, 손가락별 담당 키, 두벌식 자모 |
| `keyboard_controller.py` | 손가락별 누름 판정, 기준선, 쿨다운, 키 스냅, FPS 보정 |
| `mouse_controller.py` | 커서/클릭/더블클릭/드래그 (핀치) |
| `gesture_controller.py` | 양손 펼침 ACTIVE 전환, 가리키기 자세 → 마우스 모드 |
| `input_processor.py` | 추론 스레드의 판정 단계 묶음 (OpenCV 비의존) |
| `input_backend.py` | OS 입력 (Windows SendInput / X11 XTest / macOS CGEvent, ctypes) + 안전 게이트 `InputDispatcher` |
| `mini_ui.py` | 카메라 창 HUD (카드, 가상 키보드, 손가락 카드, 연습 카드) |
| `overlay.py` | 바탕화면 상시 HUD (항상 위, 클릭 통과, 포커스 안 뺏음 — Windows user32) |
| `practice_mode.py` / `simulation.py` / `input_logger.py` | 타자 연습 채점 / 합성 손 시뮬레이션 / CSV 로깅 |

## 반드시 지킬 설계 규칙

- **판정은 항상 1280×720 추론 좌표계에서.** 거리 상수(`PRESS_DISTANCE` 등)는 이 좌표계 기준이며 손 크기로 자동 배율.
  4K/1080p 캡처 좌표는 그리기에만 쓴다 (`FrameScaler`).
- **안전장치:** 실제 OS 입력은 `--test-mode false` + `--enable-real-input true` + ACTIVE + 비상정지 아님 일 때만.
  연습/시뮬레이션 모드는 절대 실제 입력 금지. 시작 상태는 항상 INACTIVE. `InputBackend` 호출은 메인 스레드에서만.
- **좌우 손은 화면 위치 기준**(`--handedness position`, 기본). MediaPipe 좌우 라벨은 손등이 보이는 타이핑 자세에서
  뒤집히므로 쓰지 않는다. 같은 이유로 handedness 점수로 손가락을 버리지 않는다.
- **마우스 모드는 오른손 가리키기 자세**(검지 위, 약지·새끼 접기)로 켠다. 키보드 위치와 무관하게 동작해야 하며,
  마우스 모드 중 오른손은 키 입력에서 제외하고 누름 기준선을 버린다 (복귀 시 오입력 방지).
- 누름 깊이 = 손끝 y − 해당 손가락 MCP y (손 전체 이동은 누름 아님). 자세가 바뀌거나 추적이 재동기화되면 기준선 재설정.
- 화면 문구는 영어(Hershey 폰트). 한글은 `mini_ui.unicode_font()`(OpenCV 5 `FontFace` + 시스템 한글 폰트)로만,
  폰트가 없으면 영문 대체 문구로 동작해야 한다.
- 새 기능에는 pytest 테스트를 추가하고, 가능하면 `simulation.py` 의 합성 손으로 end-to-end 확인.

## 환경 메모 (개발 PC: Windows)

- `origin` 은 SSH 주소지만 SSH 키가 없다. 푸시는 HTTPS 주소로:
  `git push -u https://github.com/junjjang1008180-coder/pro-re.git <branch>`
- GitHub CLI 는 PATH 에 없을 수 있다: `"C:\Program Files\GitHub CLI\gh.exe"` (계정 `junjjang1008180-coder` 로 로그인됨).
- PR 흐름: `main` 에서 새 브랜치 → 커밋 → 푸시 → `gh pr create --base main`. 머지된 브랜치는 재사용하지 않는다.
- 웹캠 "프레임을 읽을 수 없습니다" 는 대개 Chrome 등 다른 앱이 카메라를 점유한 것. 이 PC 웹캠은 저조도에서 15fps.
- 셸 heredoc 에 한글+따옴표가 섞이면 파싱이 깨질 수 있다. 긴 패치는 스크립트 파일로 작성해 실행.

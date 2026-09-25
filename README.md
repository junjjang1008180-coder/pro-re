# 웹캠 가상 키보드 / 마우스 (Python)

웹캠으로 양손 10개 손가락을 추적해서, 허공에 떠 있는 QWERTY 키보드를 "아래로 짧게 누르는" 동작으로
타이핑하고 오른손 핀치로 마우스를 조작하는 프로그램입니다.

- 4K 캡처 + 1280×720 추론 분리, 캡처/추론/렌더 **3-스레드 파이프라인**
- 손가락별 독립 상태 머신 (HOVER → PRESSING → PRESSED → RELEASED), 쿨다운, 동시 입력
- One Euro/이동 평균 좌표 안정화, 이상치 제거, 손 크기 기반 임계값 자동 보정
- 마우스 이동·좌클릭·우클릭·더블클릭·드래그
- 테스트 모드 기본값 + 이중 안전 스위치 + 비상 정지
- `--log-input` CSV 로깅, `--practice` 타자 정확도/WPM 측정, `--simulate` 웹캠 없는 검증

---

## 0. 기술 스택과 선택 이유

| 항목 | 사용 기술 | 비고 |
|---|---|---|
| 언어 | **Python 3.10+** (3.13에서 검증) | 원래 요청은 C++17 이었으나, 개발 PC에 C++ 툴체인이 없어 요청자 결정으로 Python 전환 |
| 영상 처리 | OpenCV (`opencv-python`) | 캡처, 리사이즈, 그리기 |
| 손 랜드마크 | **MediaPipe Tasks `HandLandmarker`** | 아래 참고 |
| 실제 입력 | Windows `SendInput` / Linux `XTest` / macOS `CGEvent` | 모두 `ctypes`로 OS API 직접 호출, `InputBackend` 추상화 |
| 테스트 | pytest | GoogleTest 대응 |
| CLI | argparse (표준 라이브러리) | cxxopts/CLI11 대응 |
| JSON | `json` (표준 라이브러리) | nlohmann/json 대응 |
| 패키지 관리 | pip + `requirements.txt` + venv | vcpkg/Conan 대응 |

**MediaPipe 선택 이유 (vs ONNX Runtime):** C++ 에서는 MediaPipe 가 Bazel 빌드를 요구해 ONNX Runtime 대안을
고려해야 했지만, Python 에서는 `pip install mediapipe` 한 줄로 공식 HandLandmarker(손바닥 검출 + 21점 랜드마크 +
좌/우 판별 + 프레임 간 추적)를 그대로 쓸 수 있습니다. ONNX 로 직접 올리면 손바닥 검출 앵커 디코딩, ROI 회전 크롭,
NMS, 추적 로직을 모두 재구현해야 해서 버그 위험이 큽니다. 따라서 더 안정적인 MediaPipe 를 선택했습니다.

**pip 선택 이유:** Python 표준 도구라 추가 설치가 없고, 모든 의존성이 Windows/macOS/Linux 용 바이너리 휠로
배포되어 컴파일 없이 설치됩니다.

| C++ 요청 구조 | Python 구현 |
|---|---|
| `Config.h` | `vkeyboard/config.py` |
| `CliOptions.h/.cpp` | `vkeyboard/cli_options.py` |
| `HandTracker` | `vkeyboard/hand_tracker.py` (+ 좌표 변환은 `frame_scaler.py`) |
| `FingerTracker` / `Smoothing` | `finger_tracker.py` / `smoothing.py` |
| `GestureController` | `gesture_controller.py` |
| `KeyboardLayout` / `KeyboardController` | `keyboard_layout.py` / `keyboard_controller.py` |
| `MouseController` | `mouse_controller.py` |
| `InputBackend` | `input_backend.py` (Windows/X11/macOS + 안전 게이트 `InputDispatcher`) |
| `InputState` | `input_state.py` (손가락 상태 머신) |
| `Calibration` | `calibration.py` |
| `MiniUI` | `mini_ui.py` |
| `InputLogger` | `input_logger.py` |
| `PracticeMode` | `practice_mode.py` |
| `CaptureThread` / `FrameQueue.h` | `pipeline.py` / `frame_queue.py` |
| `main.cpp` | `app.py` (+ 루트 `main.py`) |
| (추가) | `input_processor.py`: 추론 스레드의 판정 단계 묶음, `simulation.py`: 웹캠 없는 검증 |

---

## 1. 설치

### 공통
```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate
```
(macOS/Linux 는 `source .venv/bin/activate`)
```bash
pip install -r requirements.txt
```

손 랜드마크 모델 `models/hand_landmarker.task` (약 7.8MB)는 **첫 실행 때 Google 공식 배포 주소에서 자동 다운로드**됩니다.
오프라인 환경이면 아래 주소에서 받아 `models/` 에 넣거나 `--model-path` 로 경로를 지정하세요.
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`

### 플랫폼별 실제 입력 준비물
| OS | API | 준비 |
|---|---|---|
| Windows | `user32.SendInput` | 없음. 단, 관리자 권한으로 실행된 창에는 일반 권한 프로그램이 입력을 보낼 수 없음(UIPI) |
| Linux | `libXtst` (XTest) | `sudo apt install libxtst6` , X11 세션 필요 (Wayland 는 XWayland 앱에만 동작) |
| macOS | CoreGraphics `CGEvent` | 시스템 설정 → 개인정보 보호 → **손쉬운 사용** 및 **카메라**에 터미널/파이썬 허용 |

> 빌드 단계는 없습니다 (C++ 의 "CMake 빌드"에 해당하는 단계 = 위 `pip install`).

---

## 2. 실행

```bash
python main.py
```
(`python -m vkeyboard` 와 동일)

기본값은 **테스트 모드**: 화면에 감지 결과만 표시하고 실제 키보드/마우스는 절대 조작하지 않습니다.

### 조작 순서
1. 카메라 앞에 양손을 보이게 합니다. 시작 상태는 항상 **INACTIVE** 입니다.
2. **양손 손바닥을 활짝 펴고 1초 유지** → ACTIVE (다시 1초 유지 → INACTIVE). 전환 후엔 손을 한 번 내려야 다시 전환됩니다.
   - 상태 카드의 `Palms` 줄에 손별 인식 여부(`L open  R --`)가 나오니, 안 되는 쪽 손을 더 활짝 펴 보세요.
   - 인식이 잘 안 되면 창을 클릭한 뒤 **`a` 키**로도 ACTIVE/INACTIVE 를 전환할 수 있습니다.
3. 손가락 끝을 원하는 키 위에 올리고, 그 손가락만 **아래로 짧게 톡** 누릅니다.
4. 오른손으로 **가리키기 자세**(검지만 위로 펴고 약지·새끼는 접기)를 0.25초 유지하면 마우스 모드 (검지로 커서 이동).
   손을 다시 펴면 0.4초 뒤 타이핑 모드로 돌아옵니다.
5. **한/영 전환**: Space 오른쪽 `한/영` 키를 오른손 엄지로 누릅니다. 한국어 모드면 키 위에 두벌식 자모가 표시됩니다.
6. `ESC` 로 즉시 종료.

### CLI 옵션 전체 목록
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--camera-index <int>` | 0 | 웹캠 번호 |
| `--capture-width <int>` | 3840 | 캡처 폭 |
| `--capture-height <int>` | 2160 | 캡처 높이 |
| `--test-mode <true\|false>` | true | true 면 실제 입력 절대 안 보냄 |
| `--enable-real-input <true\|false>` | false | `--test-mode false` 와 **동시에** 줘야 실제 입력 |
| `--calibration-file <path>` | calibration.json | 키보드 위치/크기 저장 파일 |
| `--log-input <path>` | (없음) | 지정 시 키/클릭 이벤트 CSV 로깅 |
| `--practice` | off | 타자 연습/정확도 측정 모드 |
| `--practice-output <path>` | practice_result.json | 연습 결과 저장 경로 (`""` 이면 저장 안 함) |
| `--model-path <path>` | models/hand_landmarker.task | 손 랜드마크 모델 |
| `--simulate` | off | 웹캠 없이 합성 손 데이터로 전체 판정 경로 검증 (헤드리스) |
| `--beep` | off | 키 확정 시 OS 기본 비프음 |
| `--sensitivity <low\|normal\|high>` | normal | 키 누름 민감도 프리셋 (high = 얕고 빠른 누름도 인식) |
| `--overlay <true\|false>` | true | 바탕화면에 항상 위 HUD(미니 키보드/마우스/ACTIVE/한영) 표시 |
| `--overlay-corner <br\|bl\|tr\|tl>` | br | HUD 위치 (오른쪽 아래/왼쪽 아래/오른쪽 위/왼쪽 위) |
| `--overlay-scale <float>` | 1.3 | HUD 크기 배율 |
| `--handedness <position\|model>` | position | 좌/우 손 판별. position=화면 위치 기준(손등이 보여도 안정적), model=MediaPipe 라벨 |
| `--run-seconds <float>` | 0 | 0보다 크면 N초 후 자동 종료 (자동 점검/데모용) |
| `--press-distance`, `--release-distance`, `--min-down-frames`, `--key-cooldown` | Config 값 | 튜닝 실험용 오버라이드 |

### 사용 예시
```bash
python main.py --camera-index 1 --capture-width 1920 --capture-height 1080
```
```bash
python main.py --log-input logs/session1.csv
```
```bash
python main.py --practice --practice-output results/me.json
```
```bash
python main.py --test-mode false --enable-real-input true
```
```bash
python main.py --simulate --practice
```

---

## 3. 테스트 모드 / 실제 입력 활성화 (안전장치)

실제 OS 입력은 아래가 **모두** 참일 때만 나갑니다.

1. `--test-mode false`
2. `--enable-real-input true`
3. 연습(`--practice`)/시뮬레이션(`--simulate`) 모드가 아님
4. 현재 상태가 **ACTIVE**
5. 비상 정지 상태가 아님

즉시 중지 조건:
- **웹캠 연결 실패** (시작 시 / 실행 중 연속 읽기 실패) → 입력 중지 후 종료 (종료 코드 2)
- **손 추적 실패** → 손이 사라지는 즉시 해당 손가락 판정 중단, 0.7초 이상 사라지면 강제 INACTIVE, 드래그 중이던 버튼 해제
- **창 닫기** → 입력 중지 후 종료
- **예외 발생** → `try/except/finally` 로 눌린 버튼·키를 모두 떼고 종료 (종료 코드 1)
- `ESC` → 즉시 종료

화면 좌상단에 `TEST MODE (no real input)` 또는 빨간색 `REAL INPUT ENABLED` 가 항상 표시됩니다.

---

## 4. 캘리브레이션

실행 중 `c` 를 누르면 캘리브레이션 모드(키보드 테두리가 노란색)가 됩니다.

| 키 | 동작 |
|---|---|
| `i` `j` `k` `l` | 위/왼쪽/아래/오른쪽 이동 |
| `+` / `-` | 전체 크기 조절 |
| `]` / `[` | 높이만 조절 |
| `f` | **손가락 맞춤**: 왼손 검지를 원하는 F 위치, 오른손 검지를 J 위치에 두고 누르면 키보드가 손 크기에 맞춰 자동 배치 |
| `r` | 기본값으로 초기화 |
| `s` | `calibration.json` 에 저장 |
| `c` | 캘리브레이션 종료 |

저장 형식 (추론 좌표계 1280×720 기준 픽셀):
```json
{ "x": 256.0, "y": 463.6, "width": 768.0, "height": 227.6, "infer_width": 1280, "infer_height": 720, "version": 1 }
```
추론 해상도를 바꾸면 불러올 때 자동으로 비율 변환됩니다.

---

## 5. 손가락별 키 배치

| 손가락 | 담당 키 |
|---|---|
| 왼손 새끼손가락 | Q, A, Z, Tab, Caps Lock, Left Shift |
| 왼손 약지 | W, S, X |
| 왼손 중지 | E, D, C |
| 왼손 검지 | R, F, V, T, G, B |
| 왼손 엄지 | Space |
| 오른손 검지 | Y, H, N, U, J, M |
| 오른손 중지 | I, K, , |
| 오른손 약지 | O, L, . |
| 오른손 새끼손가락 | P, ;, /, Backspace, Enter, Right Shift |
| 오른손 엄지 | Space, 한/영 |

- 담당이 아닌 손가락으로 누르면 입력되지 않습니다 (오입력 방지).
- 누르기 시작한 순간의 키로 고정되므로, 누르는 동안 손끝이 아래 줄로 미끄러져도 원래 키가 입력됩니다.
- Shift 는 "원샷" 수식키입니다: Shift 를 한 번 누르면 다음 문자 키 하나에 적용됩니다.
- **한/영 키**는 OS 입력기의 한/영 전환을 보냅니다: Windows `VK_HANGUL`, Linux X11 `Hangul` 키,
  macOS `Ctrl+Space`(입력 소스 전환 단축키). 실제 한글 조합은 OS 의 한국어 입력기(두벌식)가 합니다.
  프로그램은 한/영 키를 누른 횟수로 현재 언어를 추적해 표시하므로, 다른 방법으로 입력기를 바꾸면 표시가 어긋날 수 있습니다.
- 한글 표시는 OpenCV 5 이상 + 시스템 한글 폰트(맑은 고딕 / Apple SD 고딕 Neo / 나눔고딕·Noto CJK)가 있을 때 나옵니다.
- 영상은 거울 모드(좌우 반전)로 표시되며, 좌/우 손 판별도 사용자 시점 기준입니다.

---

## 6. 마우스 제스처 (오른손)

마우스는 키보드와 완전히 분리된 독립 컨트롤러이며 **ACTIVE 상태에서만** 동작합니다.

**마우스 모드 켜기:** 오른손 **가리키기 자세**(검지를 위로 펴고 약지·새끼를 주먹 쥐듯 접기, 중지는 자유)를 0.25초 유지.
손 모양으로 모드를 정하므로 **키보드를 화면 어디에 두든 마우스와 겹치지 않습니다.**
- 마우스 모드 중에는 오른손 손가락이 키 입력에 쓰이지 않습니다 (손가락 카드에 `mouse` 로 표시).
- 손을 펴서 타이핑 자세로 돌아가면 0.4초 뒤 타이핑 모드로 복귀합니다.
- 커서 범위: 카메라 화면 가운데 영역(가로 15~85%, 세로 10~70%)이 모니터 전체에 대응합니다.

| 동작 | 제스처 |
|---|---|
| 마우스 모드 | 오른손 가리키기 자세 유지 (검지 위로, 약지·새끼 접기) |
| 커서 이동 | 오른손 검지 끝 위치 (One Euro Filter 로 흔들림 제거) |
| 좌클릭 | 엄지-검지를 짧게 붙였다 뗌 (0.6초 이내) |
| 우클릭 | 엄지-중지를 짧게 붙였다 뗌 |
| 더블클릭 | 좌클릭 제스처를 0.45초 안에 2번 |
| 드래그 | 엄지-검지를 붙인 채 손을 이동 → 떼면 드롭 |

- 핀치는 2프레임 연속 유지돼야 인정(디바운스), 떼는 거리는 붙는 거리의 1.4배(히스테리시스)
- 같은 버튼 클릭 사이 쿨다운 0.4초 (더블클릭의 두 번째 클릭만 예외)
- 핀치 중엔 커서를 고정해 클릭 순간 커서가 튀지 않게 함
- 핀치는 키 입력 판정에 전혀 쓰지 않습니다.

---

## 7. 화면 표시

- 손 랜드마크 + 좌/우 손 구분 색상 + 손별 추적 신뢰도
- 10개 손가락 이름/상태/선택 키 (손끝 라벨 + 우측 상단 표)
- ACTIVE/INACTIVE, 테스트 모드 여부, FPS(표시/캡처/추론), 캡처→추론 해상도
- 마지막 입력 키, 마지막 마우스 동작(이동/좌클릭/우클릭/더블클릭/드래그)
- 키 확정 순간 해당 키 약 100ms 하얗게 하이라이트 (가상 키보드 + 미니 키보드)
- 우측 **미니 키보드**(손가락 색으로 선택 키 표시)와 **미니 마우스**(핀치 중 청록, 클릭 순간 초록, 드래그 주황)
- **데스크톱 HUD (오버레이)**: 카메라 창과 별도로 바탕화면 오른쪽 아래에 작은 창이 **항상 맨 위**에 떠서
  ACTIVE/INACTIVE, TEST/REAL, 한/영 상태, 미니 키보드·마우스, 마지막 입력을 보여 줍니다.
  - HUD 키보드: 키 글자(한국어 모드면 자모) 표시, 손가락이 올라간 키는 손가락 색 테두리,
    **누르는 중 주황 → 눌림 확정 초록 → 입력 순간 흰색으로 커짐**. 아래에 지금 누르는 키 / 마지막 입력 키 배지.
  - Windows: 테두리 없는 반투명 창이며 **클릭이 뒤 창으로 통과**하고 **포커스를 뺏지 않아** 메모장 등에 타이핑할 때 방해되지 않습니다.
  - 카메라 창은 최소화해도 되고, HUD 는 계속 갱신됩니다.
  - 끄려면 `--overlay false`, 위치/크기는 `--overlay-corner`, `--overlay-scale`
  - macOS/Linux 에서는 '항상 위' 속성만 적용된 일반 창으로 표시됩니다.
- `--beep` 로 키 확정 시 OS 기본 비프음

---

## 8. 해상도 처리 전략 (4K 캡처 / 1280×720 추론)

**왜 분리하나?** 손 랜드마크 모델 입력은 어차피 192~224px 로 줄어들기 때문에 4K 로 추론해도 정확도 이득이 거의
없고 리사이즈/색변환 비용만 9배가 됩니다. 그래서
- **캡처(3840×2160)**: 화면 표시/영상 품질용
- **추론(1280×720)**: 추론 스레드에서 `FrameScaler.resize_for_inference()` 로 다운스케일 후 MediaPipe 실행
- 랜드마크는 1280×720 좌표로 나오고, 화면에 그릴 때만 `scale_x = 3840/1280`, `scale_y = 2160/720` 로 스케일업
- `PRESS_DISTANCE`, `RELEASE_DISTANCE`, `MAX_ALLOWED_JUMP` 등 모든 판정 값은 **1280×720 기준 고정** → 캡처 해상도가
  바뀌어도 감도가 변하지 않음

**카메라가 4K 를 지원하지 않으면** 2560×1440 → 1920×1080 → 1280×720 → 640×480 순으로 자동 폴백하고
`[안내] 3840x2160 미지원 → 1920x1080으로 자동 전환됨` 을 출력합니다.

**해상도 바꾸기**
- 실행 시: `--capture-width 1920 --capture-height 1080`
- 기본값: `vkeyboard/config.py` 의 `CAPTURE_WIDTH/HEIGHT`
- 추론 해상도: `config.py` 의 `INFER_WIDTH/HEIGHT`. **주의:** 거리 관련 설정값이 모두 이 좌표계 기준이므로 추론
  해상도를 바꾸면 `PRESS_DISTANCE` 등도 같은 비율로 바꿔야 합니다 (예: 640×360 이면 절반).

---

## 9. 멀티스레드 파이프라인

```
 [CaptureThread]  웹캠 read + 거울 반전 (4K)
       │  FrameQueue(최신 2개만 유지, 오래된 프레임 버림)      LatestValue(표시용 최신 프레임)
       ▼                                                           │
 [InferenceThread]  1280×720 리사이즈 → MediaPipe 추론                │
                    → FingerTracker(안정화/이상치) → KeyboardController│
                    → MouseController → GestureController            │
       │  EventQueue(키/마우스/모드 이벤트, 절대 버리지 않음)            │
       │  LatestValue(판정 스냅샷)                                     │
       ▼                                                           ▼
 [Main 스레드]  이벤트 → InputDispatcher → InputBackend (OS 입력은 이 스레드에서만)
               스냅샷 + 최신 프레임 → 4K 캔버스에 그리기 → 표시 크기로 축소 → imshow
```

- 스레드 간 공유 자료구조는 모두 `threading.Lock`/`Condition` 으로 보호 (`frame_queue.py`)
- 판정 상태(상태 머신, 필터)는 추론 스레드만 소유하고, 메인 스레드는 불변 스냅샷만 읽으므로 데이터 경쟁이 없음
- 캘리브레이션 변경은 `InputProcessor.set_calibration()`(락)으로 다음 프레임에 반영
- 렌더러는 프레임 **복사본**에 그려서 추론 스레드가 읽는 원본과 충돌하지 않음
- OpenCV `read/resize` 와 MediaPipe 추론은 GIL 을 해제하므로 Python 스레드로도 실제 병렬 처리됨

**목표 성능:** 4K 캡처 기준 표시 FPS ≥ 24, 추론 FPS ≥ 15.
참고 측정 (개발 PC, CPU 전용): 4K→720p 리사이즈+추론 평균 18.5ms(≈54 FPS), 4K 캔버스 복사+그리기+축소 ≈22ms(≈45 FPS).
실제 FPS 는 카메라(4K 30fps 는 대부분 MJPG 필요), CPU, USB 대역폭에 따라 달라집니다. 부족하면
`--capture-width 1920 --capture-height 1080` 으로 낮추세요 (판정 감도는 동일).

---

## 10. 좌표 안정화 / 이상치 제거

| 대상 | 처리 |
|---|---|
| 손끝 좌표 흔들림 | 최근 `SMOOTHING_FRAMES`(5) 프레임 이동 평균 |
| 마우스 커서 | One Euro Filter |
| 갑자기 튀는 좌표 | 이전 좌표와 `MAX_ALLOWED_JUMP`(60px) 초과 시 제외, 4프레임 연속이면 새 위치로 재동기화 |
| 비정상적으로 빠른 움직임 | `MAX_SPEED`(2400px/s) 초과 시 제외 |
| 낮은 추적 신뢰도 | `MIN_TRACKING_CONFIDENCE`(0.75) 미만 제외 (MediaPipe 손 존재/추적 임계값에도 적용) |
| 화면 가장자리 | 가장자리 12px 이내 제외 |
| 겹친 손가락 | 손끝끼리 12px(손 크기 비례) 이내면 둘 다 제외 |
| 카메라 거리 차이 | 손목~중지 끝 거리(손 크기)/180px 비율로 누름·해제·핀치 임계값 자동 배율(0.6~1.8) |

누름 깊이는 **손가락 끝 - 그 손가락 기준 관절(MCP)** 의 세로 차이로 재기 때문에, 손 전체가 위아래로 움직이는 것은
누름으로 보지 않습니다. 기준선(휴지 위치)은 HOVER 중 천천히 적응하므로 천천히 내려오는 자세 변화도 무시됩니다.

---

## 11. 설정값 튜닝 가이드

기본값은 **"오입력 최소화" 우선** 설정입니다 (반응 속도보다 정확도). `vkeyboard/config.py` 에서 바꾸거나, 일부는
CLI 로 바로 실험할 수 있습니다.

| 설정 | 기본값 | 반응 속도↑ (빠르게) | 정확도↑ (오입력↓) | 트레이드오프 |
|---|---|---|---|---|
| `PRESS_DISTANCE` | 18 | 15 로 ↓ | 20+ 로 ↑ | 낮추면 얕게 눌러도 입력되지만 손 떨림 오입력 증가 |
| `RELEASE_DISTANCE` | 10 | 8 로 ↓ | 12 로 ↑ | 낮추면 손을 조금만 들어도 재입력 가능 / 떨림 재입력 위험 |
| `MIN_DOWN_FRAMES` | 3 | 2 로 ↓ | 4 로 ↑ | 1프레임(30fps 기준 33ms)만큼 지연 증가/감소 |
| `KEY_COOLDOWN` | 0.35s | 0.25 로 ↓ | 0.45 로 ↑ | 낮추면 같은 손가락 연타 가능 / 이중 입력 위험 |
| `SMOOTHING_FRAMES` | 5 | 3 으로 ↓ | 6~7 로 ↑ | 높을수록 흔들림 적지만 반응 지연 |
| `MIN_TRACKING_CONFIDENCE` | 0.75 | 0.6 으로 ↓ | 0.85 로 ↑ | 낮추면 어두운 곳에서도 동작하지만 오검출 증가 |
| `MAX_ALLOWED_JUMP` | 60 | 80 으로 ↑ | 50 으로 ↓ | 높이면 빠른 손 이동 허용 / 튀는 좌표 통과 |
| `CLICK_PINCH_DISTANCE` | 22 | 26 으로 ↑ | 18 로 ↓ | 높이면 살짝만 붙여도 클릭 / 의도치 않은 클릭 |

**민감도 프리셋** (`--sensitivity`):

| 프리셋 | PRESS / RELEASE / MIN_DOWN_FRAMES | 용도 |
|---|---|---|
| `low` | 22 / 12 / 3 | 오입력 최소화 (깊게 눌러야 함) |
| `normal` (기본) | 18 / 10 / 3 | 균형 |
| `high` | 13 / 7 / 2 | 얕고 빠른 누름도 인식 (키가 잘 안 눌릴 때) |

```bash
python main.py --sensitivity high
```
`--press-distance` 등을 직접 주면 프리셋보다 우선합니다.

**인식률 관련 자동 보정**
- **FPS 보정:** `MIN_DOWN_FRAMES` 는 30fps 기준이며, 카메라가 느리면 같은 시간(`MIN_DOWN_TIME` 0.08초)이
  되도록 필요 프레임 수를 줄입니다 (15fps → 2프레임). 저조도에서 15fps 로 떨어지는 웹캠에서도 짧은 톡 누름이 인식됩니다.
- **누름 신호는 약하게만 평균:** 손끝 위치(어느 키인지)는 5프레임 평균, 누름 깊이는 2프레임 평균(`PRESS_SMOOTHING_FRAMES`)이라
  빠른 누름이 평균에 묻히지 않습니다.
- **키 스냅:** 손끝이 담당 키 경계를 살짝 벗어나도 키 1칸의 0.6배(`KEY_SNAP_RADIUS`) 안이면 가장 가까운 담당 키로 인정합니다.
- **신뢰도:** 위치 기준 좌우 판별(기본)에서는 MediaPipe 좌우 판별 점수(손등이 보이면 낮아짐)로 손가락을 버리지 않습니다.
  손 존재 신뢰도 `MIN_TRACKING_CONFIDENCE` 는 MediaPipe 가 그대로 적용합니다.

**누름 게이지:** 손가락 카드의 각 손가락 줄에 작은 막대가 있습니다. 누를수록 차오르고, **흰 눈금을 넘어야 입력**됩니다.
막대가 눈금 근처까지만 오고 입력이 안 되면 `--sensitivity high` 를, 가만히 있어도 눈금을 넘나들면 `low` 를 쓰세요.

---

## 12. `--log-input` CSV 로 튜닝하기

```bash
python main.py --log-input logs/session1.csv
```
CSV 컬럼: `timestamp, event_type(key|click), finger_or_hand, value(key_or_button), dy_at_fire, hand_size, tracking_confidence`
```
2026-09-26T14:02:11.402,key,left_index,F,24.13,176.2,0.962
2026-09-26T14:02:12.018,click,right_hand,left_click,14.20,181.0,0.955
```
- 키 이벤트의 `dy_at_fire` = 확정 순간 누름 깊이(px), 클릭 이벤트는 핀치 거리(px)
- 로그는 메모리에 버퍼링했다가 백그라운드 스레드가 0.25초마다 파일에 써서 FPS 에 영향이 없습니다.
- 옵션을 안 주면 파일을 만들지 않습니다.

**튜닝 예시:** 의도한 입력만 있었던 세션에서 `dy_at_fire` 분포를 봅니다.
```python
import pandas as pd
df = pd.read_csv("logs/session1.csv")
keys = df[df.event_type == "key"]
print(keys.groupby("finger_or_hand").dy_at_fire.describe())
```
- 대부분 25~40px 인데 오입력(원치 않은 키)이 18~20px 근처에 몰려 있으면 → `PRESS_DISTANCE` 를 21~22 로 올림
- 특정 손가락(예: 약지)의 `dy_at_fire` 가 늘 낮아 입력이 잘 안 되면 → `PRESS_DISTANCE` 를 낮추거나 그 손가락을 더 크게 누르는 습관
- `tracking_confidence` 가 낮을 때 오입력이 몰려 있으면 → `MIN_TRACKING_CONFIDENCE` 를 올리고 조명 개선
- `hand_size` 가 세션마다 크게 다르면 카메라 거리가 변한 것 (임계값은 자동 보정됨)

---

## 13. `--practice` 타자 연습 / 정확도 측정 모드

```bash
python main.py --practice
```
1. 양손 펼침 1초로 ACTIVE 전환
2. 화면 상단 목표 문장을 가상 키보드로 입력 (초록 = 입력 완료, 노랑 = 다음 글자)
3. 3문장이 끝나면 결과 요약이 화면에 표시되고 `practice_result.json` 저장 (`p` 로 다시 시작)

- 실제 OS 입력은 보내지 않고 내부 채점만 합니다. 판정 로직은 실제 입력 경로와 **같은 코드**
  (`FingerTracker` → `KeyboardController` → `KeyEvent`) 를 사용합니다.
- 틀린 키는 커서를 전진시키지 않고 오류로 집계됩니다. Shift/Backspace/Enter 는 채점 제외.

**결과 해석**
| 항목 | 의미 |
|---|---|
| `accuracy` | 정답 키 입력 수 / 전체 키 입력 수 |
| `wpm` | (정답 글자 수 / 5) / 분 — 첫 키 입력부터 마지막 글자까지 시간 기준 |
| `per_finger.<finger>.error_rate` | 그 손가락이 친 키 중 틀린 비율 → 어떤 손가락이 오타를 많이 내는지 |
| `missed_by_expected_finger` | 쳐야 했던 글자의 담당 손가락별 오타 수 (어떤 손가락 자리에서 실수가 나는지) |

**웹캠 없이 검증:** `python main.py --simulate --practice` 는 합성 손 데이터로 활성화 → 3문장 타이핑(의도적 오타 포함)
→ 마우스 클릭 → 추적 실패까지 재생하고 정확도/WPM 을 출력합니다.

---

## 14. 테스트

```bash
python -m pytest
```
| 파일 | 검증 내용 |
|---|---|
| `test_input_state.py` | 손가락 상태 전환, 한 번 누르면 한 번만 입력, 누르고 있어도 반복 없음, 해제 후에만 재입력 |
| `test_cooldown.py` | 손가락 쿨다운, 같은 키 쿨다운(양 엄지 Space), 느린 드리프트 무시, 누름 시작 키 고정 |
| `test_keyboard_layout.py` | 손가락별 담당 키, QWERTY 배치, 히트 테스트 |
| `test_calibration.py` | 캘리브레이션 저장/복원/검증/F·J 맞춤, 이동 평균/One Euro/이상치/겹침/가장자리/손 크기 보정 |
| `test_concurrent.py` | 여러 손가락 동시 입력, FrameQueue/EventQueue 스레드 안전성, 비동기 CSV 로거 |
| `test_mouse_click.py` | 좌클릭/우클릭/더블클릭/드래그, 클릭 쿨다운, 디바운스, 비활성 시 무동작 |
| `test_resolution_scaling.py` | 1280×720 ↔ 4K(×3) / 1080p(×1.5) 좌표 변환, 리사이즈 |
| `test_practice_scoring.py` | 정확도/WPM 공식, 손가락별 오류율, JSON, 웹캠 없는 end-to-end 채점 |
| `test_cli_and_safety.py` | CLI 기본값/오버라이드, 실제 입력 이중 스위치, 비상 정지, ACTIVE 전환 제스처, 추적 실패 |

---

## 15. 조명과 카메라 설정

- 손이 얼굴/배경보다 밝게, **정면에서 고르게** 비추세요. 역광(뒤 창문)은 추적 신뢰도를 크게 떨어뜨립니다.
- 카메라는 손이 화면 아래쪽 절반에 오도록, 약 50~70cm 거리에 두세요 (손 크기 ≈ 150~220px @1280×720).
- 4K 웹캠은 대부분 **MJPG** 에서만 30fps 가 나오므로 코드에서 MJPG 를 요청합니다.
- 자동 노출이 너무 느리면 카메라 설정 앱에서 노출을 고정하면 흔들림이 줄어듭니다.
- 소매/장갑, 손목시계는 랜드마크 오검출의 원인이 될 수 있습니다.

---

## 16. 오류 해결

| 증상 | 해결 |
|---|---|
| `웹캠 연결 실패` (종료 코드 2) | 다른 앱(Zoom, Teams, 카메라 앱)이 카메라를 쓰고 있지 않은지 확인. Windows: 설정 → 개인정보 → **카메라** → 데스크톱 앱 허용. `--camera-index 1` 등 다른 번호 시도 |
| 웹캠 열기가 수십 초 걸린 뒤 실패 | Windows MSMF 드라이버가 사용 중인 카메라를 기다리는 것. 위 항목 확인 |
| `4K 미지원 → ...으로 자동 전환됨` | 정상 동작 (지원하는 해상도로 폴백) |
| `손 랜드마크 모델 초기화 실패` (종료 코드 3) | 인터넷 연결 확인 또는 모델을 직접 받아 `--model-path` 지정 |
| 손이 인식은 되는데 손가락이 `x low_confidence` | 조명 개선, 또는 `MIN_TRACKING_CONFIDENCE` 낮춤 |
| 손가락이 `x overlap` | 손가락 사이를 조금 벌리거나 카메라를 약간 옆에서 비추기 |
| 키가 잘 안 눌림 | 손가락 카드의 누름 게이지가 흰 눈금을 넘는지 확인. 안 넘으면 `--sensitivity high`. 손가락이 빨간 글씨(overlap/edge 등)면 그 원인 해결 |
| 원치 않는 키가 입력됨 | `--log-input` 으로 `dy_at_fire` 확인 후 `PRESS_DISTANCE` 상향 |
| 실제 입력이 안 나감 | `--test-mode false --enable-real-input true` 둘 다 줬는지, 화면이 ACTIVE 인지 확인 |
| ACTIVE 전환이 안 됨 | `Palms` 줄 확인, 손가락을 모두 위로 곧게 펴고 벌리기, 조명 개선. 또는 창에서 `a` 키 |
| 한/영 키를 눌러도 한글이 안 나옴 | Windows 에 한국어 입력기(Microsoft 입력기) 설치 여부 확인. 입력할 창이 포커스를 가져야 함 |
| 오른손이 LEFT 로 표시됨 | 기본값(`--handedness position`)을 쓰고 있는지 확인 |
| Linux 에서 입력 안 됨 | X11 세션인지 확인 (`echo $XDG_SESSION_TYPE`), `libxtst6` 설치 |
| macOS 에서 입력 안 됨 | 손쉬운 사용 권한 허용 후 터미널 재시작 |
| FPS 가 낮음 | `--capture-width 1920 --capture-height 1080`, 다른 무거운 프로그램 종료 |

---

## 17. 참고: C++ 로 다시 옮기고 싶다면

로직 모듈(`input_state`, `keyboard_controller`, `finger_tracker`, `mouse_controller`, `practice_mode` 등)은 OpenCV 에
의존하지 않는 순수 로직이라 1:1 로 C++ 클래스로 옮길 수 있습니다. C++ 에서 MediaPipe Bazel 빌드가 부담되면
ONNX Runtime 에 `palm_detection` + `hand_landmark` ONNX 모델을 올리고 앵커 디코딩/ROI 크롭을 구현하는 방식을
권장합니다. 입력 백엔드는 이 프로젝트와 같은 API(`SendInput`, `XTestFake*Event`, `CGEventPost`)를 그대로 쓰면 됩니다.

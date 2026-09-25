"""메인(렌더링 + 입력 전송) 스레드.

- 캡처/추론 스레드를 띄우고, 최신 프레임과 최신 판정 결과를 받아 화면에 그린다.
- KeyEvent/MouseEvent 는 이 스레드에서만 InputDispatcher(InputBackend) 로 전달한다.
- 웹캠 실패 / 추적 실패 / 창 닫기 / 예외 시 실제 입력을 즉시 중지한다.
"""
from __future__ import annotations

import sys
import time
import traceback
from typing import Optional, Sequence

from .calibration import Calibration
from .cli_options import parse_cli
from .config import (DISPLAY_MAX_WIDTH, FRAME_QUEUE_SIZE, KEY_HIGHLIGHT_TIME, MOUSE_FLASH_TIME, AppConfig)
from .events import KeyEvent, ModeEvent, MouseAction, MouseEvent
from .frame_queue import EventQueue, FrameQueue, LatestValue
from .input_backend import (InputDispatcher, NullInputBackend, create_platform_backend, play_feedback_beep,
                            query_screen_size)
from .input_logger import InputLogger
from .keyboard_layout import Finger
from .practice_mode import PracticeResult, PracticeSession

WINDOW = "Virtual Keyboard"
EXIT_OK, EXIT_ERROR, EXIT_CAMERA, EXIT_MODEL = 0, 1, 2, 3


def _print_banner(cfg: AppConfig) -> None:
    print("=" * 60)
    print(" 웹캠 가상 키보드")
    print(f" 카메라 {cfg.camera_index} / 캡처 {cfg.capture_width}x{cfg.capture_height} "
          f"/ 추론 {cfg.infer_width}x{cfg.infer_height}")
    if cfg.practice:
        print(" 모드: 타자 연습 (로컬 채점, 실제 입력 없음)")
    elif cfg.real_input_allowed:
        print(" 모드: *** 실제 키보드/마우스 입력 활성화 *** (ACTIVE 상태에서만 전송)")
    else:
        print(f" 모드: 테스트 모드 (실제 입력 없음) test_mode={cfg.test_mode} "
              f"enable_real_input={cfg.enable_real_input}")
    print(" 시작 상태: INACTIVE — 양손 손바닥을 펴고 1초 유지하면 ACTIVE")
    print(" ESC: 즉시 종료 / c: 캘리브레이션")
    print("=" * 60)


def run(cfg: AppConfig) -> int:
    if cfg.simulate:
        from .simulation import run_simulation

        report = run_simulation(cfg)
        if cfg.practice and cfg.practice_output and report.practice_result is not None:
            report.practice_result.save(cfg.practice_output)
            print(f"[정보] 연습 결과 저장: {cfg.practice_output}")
        return EXIT_OK

    import cv2

    from .frame_scaler import FrameScaler
    from .hand_tracker import HandTracker
    from .input_processor import InputProcessor
    from .mini_ui import RenderInfo, Renderer
    from .pipeline import CaptureThread, FpsCounter, InferenceThread, display_size, open_camera

    _print_banner(cfg)

    # --- 입력 백엔드: 두 조건이 동시에 만족될 때만 실제 OS 백엔드 생성 ---
    if cfg.real_input_allowed:
        try:
            backend = create_platform_backend()
        except Exception as e:  # noqa: BLE001
            print(f"[오류] 입력 백엔드 초기화 실패 → 테스트 모드로 전환: {e}")
            backend = NullInputBackend(query_screen_size())
            cfg.enable_real_input = False
    else:
        backend = NullInputBackend(query_screen_size())
    dispatcher = InputDispatcher(backend, cfg.real_input_allowed)

    logger: Optional[InputLogger] = InputLogger(cfg.log_input) if cfg.log_input else None
    if logger:
        print(f"[정보] 입력 이벤트 CSV 로깅: {cfg.log_input}")

    capture: Optional[CaptureThread] = None
    inference: Optional[InferenceThread] = None
    tracker = None
    exit_code = EXIT_OK
    practice = PracticeSession() if cfg.practice else None
    practice_result: Optional[PracticeResult] = None
    try:
        # --- 웹캠 ---
        try:
            cap, cw, ch = open_camera(cfg.camera_index, cfg.capture_width, cfg.capture_height)
        except Exception as e:  # noqa: BLE001
            dispatcher.emergency_stop("웹캠 연결 실패")
            print(f"[오류] 웹캠 연결 실패: {e}")
            print("[안전] 실제 입력 중지 상태로 안전 종료합니다.")
            return EXIT_CAMERA
        cfg.capture_width, cfg.capture_height = cw, ch
        scaler = FrameScaler(cw, ch, cfg.infer_width, cfg.infer_height)

        # --- 손 추적 모델 ---
        try:
            tracker = HandTracker(cfg, scaler)
        except Exception as e:  # noqa: BLE001
            cap.release()
            dispatcher.emergency_stop("손 추적 초기화 실패")
            print(f"[오류] 손 랜드마크 모델 초기화 실패: {e}")
            return EXIT_MODEL

        calibration = Calibration.load(cfg.calibration_file, cfg.infer_width, cfg.infer_height)
        processor = InputProcessor(cfg, calibration, backend.screen_size(), mouse_enabled=not cfg.practice)

        frame_queue: FrameQueue = FrameQueue(FRAME_QUEUE_SIZE)
        display_slot: LatestValue = LatestValue()
        snapshot_slot: LatestValue = LatestValue()
        event_queue: EventQueue = EventQueue()
        capture = CaptureThread(cap, frame_queue, display_slot)
        inference = InferenceThread(frame_queue, tracker, scaler, processor, event_queue, snapshot_slot)
        capture.start()
        inference.start()

        renderer = Renderer(scaler)
        fps = FpsCounter()
        info = RenderInfo(now=time.monotonic(), capture_size=(cw, ch), test_mode=cfg.test_mode,
                          real_input=cfg.real_input_allowed, backend_name=backend.name, practice=practice)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        dw, dh = display_size(cw, ch, DISPLAY_MAX_WIDTH)
        cv2.resizeWindow(WINDOW, dw, dh)

        calibrating = False
        cal_edit = calibration
        last_frame_version = -1
        shown = False
        message_until = 0.0

        started = time.monotonic()
        while True:
            now = time.monotonic()
            info.now = now
            if cfg.run_seconds > 0 and now - started >= cfg.run_seconds:
                print(f"[종료] --run-seconds {cfg.run_seconds:g}초 경과")
                break

            # --- 스레드 오류 감시 (웹캠 끊김 / 추론 예외) ---
            if capture.error or inference.error:
                reason = capture.error or inference.error
                dispatcher.emergency_stop(reason)
                print(f"[오류] {reason} → 실제 입력 중지 후 종료")
                exit_code = EXIT_CAMERA if capture.error else EXIT_ERROR
                break

            # --- 이벤트 처리 (입력 전송은 이 스레드에서만) ---
            for ev in event_queue.drain():
                if isinstance(ev, ModeEvent):
                    dispatcher.set_active(ev.active)
                    if not ev.active and ev.reason == "tracking_lost":
                        info.message = "Tracking lost -> INACTIVE (input stopped)"
                        message_until = now + 2.0
                        print("[안전] 손 추적 실패 → INACTIVE 전환, 실제 입력 중지")
                    else:
                        print(f"[상태] {'ACTIVE' if ev.active else 'INACTIVE'}")
                elif isinstance(ev, KeyEvent):
                    info.key_highlights[ev.key] = now + KEY_HIGHLIGHT_TIME
                    if logger:
                        logger.log_key(ev)
                    if cfg.beep:
                        play_feedback_beep()
                    if practice is not None:
                        practice.on_key(ev.key, ev.finger, ev.t)
                        info.last_key = f"{ev.key.label} ({ev.finger.value})"
                        if practice.finished and practice_result is None:
                            practice_result = practice.result(ev.t)
                            info.practice_result = practice_result
                            print("[연습 완료] " + " | ".join(practice_result.summary_lines()))
                            if cfg.practice_output:
                                practice_result.save(cfg.practice_output)
                                print(f"[정보] 연습 결과 저장: {cfg.practice_output}")
                    else:
                        dispatcher.handle_key(ev)
                        info.last_key = f"{dispatcher.last_key_text} ({ev.finger.value})"
                elif isinstance(ev, MouseEvent):
                    if ev.action is not MouseAction.MOVE:
                        info.mouse_flash = (ev.action, now + MOUSE_FLASH_TIME)
                        if logger:
                            logger.log_mouse(ev)
                    dispatcher.handle_mouse(ev)
                    info.last_mouse = dispatcher.last_mouse_text
            if now > message_until:
                info.message = ""

            # --- 렌더링 ---
            item, version = display_slot.get()
            snap, _ = snapshot_slot.get()
            if item is not None and version != last_frame_version:
                last_frame_version = version
                canvas = item.image.copy()   # 추론 스레드가 같은 프레임을 읽는 중일 수 있으므로 복사본에 그림
                info.display_fps = fps.tick(now)
                info.capture_fps = capture.fps
                info.calibration_mode = calibrating
                info.calibration = cal_edit if calibrating else None
                renderer.draw(canvas, snap, info)
                if canvas.shape[1] > dw:
                    canvas = cv2.resize(canvas, (dw, dh), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW, canvas)
                shown = True

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[종료] ESC 입력")
                break
            if shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                dispatcher.emergency_stop("창 닫힘")
                print("[종료] 창 닫힘 → 실제 입력 중지")
                break

            # --- 캘리브레이션 / 연습 모드 키 ---
            if key == ord("c"):
                calibrating = not calibrating
                if not calibrating:
                    processor.set_calibration(cal_edit)
            elif key == ord("p") and practice is not None:
                practice.reset()
                practice_result = None
                info.practice_result = None
            elif calibrating and key != 255:
                step = 8.0
                ch_key = chr(key)
                if ch_key == "i":
                    cal_edit = cal_edit.moved(0, -step)
                elif ch_key == "k":
                    cal_edit = cal_edit.moved(0, step)
                elif ch_key == "j":
                    cal_edit = cal_edit.moved(-step, 0)
                elif ch_key == "l":
                    cal_edit = cal_edit.moved(step, 0)
                elif ch_key in "+=":
                    cal_edit = cal_edit.scaled(1.04, 1.04)
                elif ch_key == "-":
                    cal_edit = cal_edit.scaled(1 / 1.04, 1 / 1.04)
                elif ch_key == "]":
                    cal_edit = cal_edit.scaled(1.0, 1.05)
                elif ch_key == "[":
                    cal_edit = cal_edit.scaled(1.0, 1 / 1.05)
                elif ch_key == "r":
                    cal_edit = Calibration.default(cfg.infer_width, cfg.infer_height)
                elif ch_key == "f" and snap is not None:
                    li = snap.finger_views.get(Finger.LEFT_INDEX)
                    ri = snap.finger_views.get(Finger.RIGHT_INDEX)
                    if li and ri and li.tip and ri.tip:
                        try:
                            cal_edit = Calibration.from_index_fingers(li.tip, ri.tip, cfg.infer_width,
                                                                      cfg.infer_height)
                        except ValueError as e:
                            print(f"[캘리브레이션] {e}")
                    else:
                        print("[캘리브레이션] 양손 검지가 모두 보여야 합니다")
                elif ch_key == "s":
                    cal_edit.save(cfg.calibration_file)
                    processor.set_calibration(cal_edit)
                    print(f"[캘리브레이션] 저장 완료: {cfg.calibration_file}")
                    info.message = "Calibration saved"
                    message_until = now + 2.0
                processor.set_calibration(cal_edit)
    except KeyboardInterrupt:
        print("[종료] Ctrl+C")
    except Exception:  # noqa: BLE001
        dispatcher.emergency_stop("예외 발생")
        traceback.print_exc()
        print("[안전] 예외 발생 → 실제 입력 중지 후 안전 종료")
        exit_code = EXIT_ERROR
    finally:
        dispatcher.emergency_stop("프로그램 종료")
        for th in (capture, inference):
            if th is not None:
                th.stop()
        for th in (capture, inference):
            if th is not None and th.is_alive():
                th.join(timeout=2.0)
        if tracker is not None:
            tracker.close()
        if logger:
            logger.close()
            print(f"[정보] 로그 {logger.rows_written}건 저장: {cfg.log_input}")
        if practice is not None and practice_result is None and practice.total > 0 and cfg.practice_output:
            practice.result(time.monotonic()).save(cfg.practice_output)
            print(f"[정보] (미완료) 연습 결과 저장: {cfg.practice_output}")
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        backend.close()
    return exit_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_cli(argv)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())

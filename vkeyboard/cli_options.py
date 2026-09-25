"""명령줄 인자 파싱 -> AppConfig 기본값 덮어쓰기 (argparse)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Optional, Sequence

from .config import AppConfig


def parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y", "on"):
        return True
    if v in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"true/false 값이 필요합니다: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    d = AppConfig()
    p = argparse.ArgumentParser(
        prog="vkeyboard",
        description="웹캠 가상 키보드/마우스 (MediaPipe HandLandmarker + OpenCV)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera-index", type=int, default=d.camera_index, help="웹캠 번호")
    p.add_argument("--capture-width", type=int, default=d.capture_width, help="캡처 폭")
    p.add_argument("--capture-height", type=int, default=d.capture_height, help="캡처 높이")
    p.add_argument("--test-mode", type=parse_bool, default=d.test_mode, metavar="{true,false}",
                   help="true 면 실제 입력을 절대 보내지 않음")
    p.add_argument("--enable-real-input", type=parse_bool, default=d.enable_real_input, metavar="{true,false}",
                   help="--test-mode false 와 함께 줘야 실제 입력 전송")
    p.add_argument("--calibration-file", default=d.calibration_file, help="캘리브레이션 JSON 경로")
    p.add_argument("--log-input", default=None, metavar="PATH", help="키/클릭 이벤트 CSV 로깅")
    p.add_argument("--practice", action="store_true", help="타자 연습/정확도 측정 모드")
    p.add_argument("--practice-output", default=d.practice_output, metavar="PATH",
                   help="연습 결과 JSON 저장 경로 ('' 이면 저장 안 함)")
    p.add_argument("--model-path", default=d.model_path, help="hand_landmarker.task 경로 (없으면 자동 다운로드)")
    p.add_argument("--simulate", action="store_true",
                   help="웹캠 없이 합성 손 데이터로 판정 파이프라인/연습 채점을 검증 (헤드리스)")
    p.add_argument("--beep", action="store_true", help="키 확정 시 OS 기본 비프음 재생")
    p.add_argument("--handedness", choices=("position", "model"), default=d.handedness,
                   help="좌/우 손 판별: position=화면 위치(손등이 보여도 안정적), model=MediaPipe 라벨")
    p.add_argument("--run-seconds", type=float, default=d.run_seconds,
                   help="0보다 크면 N초 후 자동 종료 (자동 점검용)")
    # 튜닝 값 (재컴파일 없이 실험용)
    p.add_argument("--press-distance", type=float, default=d.press_distance)
    p.add_argument("--release-distance", type=float, default=d.release_distance)
    p.add_argument("--min-down-frames", type=int, default=d.min_down_frames)
    p.add_argument("--key-cooldown", type=float, default=d.key_cooldown)
    return p


def parse_cli(argv: Optional[Sequence[str]] = None) -> AppConfig:
    """argv 를 파싱해 AppConfig 를 만든다. 인자가 없으면 기본값 그대로."""
    args = build_parser().parse_args(argv)
    if args.capture_width <= 0 or args.capture_height <= 0:
        raise SystemExit("캡처 해상도는 양수여야 합니다")
    if args.release_distance >= args.press_distance:
        raise SystemExit("--release-distance 는 --press-distance 보다 작아야 합니다")
    return replace(
        AppConfig(),
        camera_index=args.camera_index,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        test_mode=args.test_mode,
        enable_real_input=args.enable_real_input,
        calibration_file=args.calibration_file,
        log_input=args.log_input,
        practice=args.practice,
        practice_output=args.practice_output or None,
        model_path=args.model_path,
        simulate=args.simulate,
        beep=args.beep,
        handedness=args.handedness,
        run_seconds=args.run_seconds,
        press_distance=args.press_distance,
        release_distance=args.release_distance,
        min_down_frames=args.min_down_frames,
        key_cooldown=args.key_cooldown,
    )

"""추론 해상도(1280x720) <-> 4K 표시 좌표 변환."""
import pytest
from conftest import P

from vkeyboard.config import CAPTURE_HEIGHT, CAPTURE_WIDTH, INFER_HEIGHT, INFER_WIDTH
from vkeyboard.frame_scaler import FrameScaler


def test_config_separates_capture_and_infer_resolution():
    assert (CAPTURE_WIDTH, CAPTURE_HEIGHT) == (3840, 2160)
    assert (INFER_WIDTH, INFER_HEIGHT) == (1280, 720)


def test_4k_scale_factors():
    s = FrameScaler(3840, 2160, 1280, 720)
    assert s.scale_x == pytest.approx(3.0)
    assert s.scale_y == pytest.approx(3.0)


@pytest.mark.parametrize("p, expected", [
    (P(0, 0), P(0, 0)),
    (P(1280, 720), P(3840, 2160)),
    (P(640, 360), P(1920, 1080)),
    (P(100.5, 33.25), P(301.5, 99.75)),
])
def test_infer_to_4k(p, expected):
    s = FrameScaler(3840, 2160)
    q = s.infer_to_display(p)
    assert (q.x, q.y) == pytest.approx((expected.x, expected.y))


def test_roundtrip():
    s = FrameScaler(3840, 2160)
    for p in [P(12.3, 45.6), P(1279, 719), P(640, 1)]:
        q = s.display_to_infer(s.infer_to_display(p))
        assert (q.x, q.y) == pytest.approx((p.x, p.y))


def test_fallback_1080p_scale():
    s = FrameScaler(1920, 1080)
    assert (s.scale_x, s.scale_y) == pytest.approx((1.5, 1.5))
    assert s.infer_to_display_pt(P(100, 100)) == (150, 150)


def test_non_16_9_capture_scales_axes_independently():
    s = FrameScaler(640, 480)
    assert s.scale_x == pytest.approx(0.5)
    assert s.scale_y == pytest.approx(480 / 720)


def test_normalized_landmarks_to_infer_coordinates():
    s = FrameScaler(3840, 2160)
    p = s.normalized_to_infer(0.5, 0.25)
    assert (p.x, p.y) == (640, 180)


def test_scale_length_for_drawing():
    assert FrameScaler(3840, 2160).scale_length(10) == pytest.approx(30)


def test_invalid_resolution_rejected():
    with pytest.raises(ValueError):
        FrameScaler(0, 2160)


def test_thresholds_are_resolution_independent(cfg):
    """판정은 추론 좌표계에서만 하므로 캡처 해상도가 바뀌어도 동일한 이벤트가 나온다."""
    from vkeyboard.simulation import run_simulation

    a = run_simulation(cfg, sentences=("fj",), mistakes=(), verbose=False)
    cfg.capture_width, cfg.capture_height = 1920, 1080
    b = run_simulation(cfg, sentences=("fj",), mistakes=(), verbose=False)
    assert a.typed_text == b.typed_text == "fj"


def test_resize_for_inference_outputs_1280x720():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    small = FrameScaler(3840, 2160).resize_for_inference(frame)
    assert small.shape == (720, 1280, 3)


def test_display_size_fits_small_screen():
    from vkeyboard.pipeline import display_size

    assert display_size(1920, 1080, 1920) == (1920, 1080)
    assert display_size(3840, 2160, 1920) == (1920, 1080)
    w, h = display_size(1920, 1080, 1229, 652)               # 1366x768 노트북 (90% / 85%)
    assert w <= 1229 and h <= 652 and abs(w / h - 16 / 9) < 0.01

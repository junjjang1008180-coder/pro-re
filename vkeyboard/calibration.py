"""키보드 위치/크기 캘리브레이션 (calibration.json 저장/불러오기)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Optional

from .config import INFER_HEIGHT, INFER_WIDTH
from .geometry import Vec2
from .keyboard_layout import UNITS_H, UNITS_W, KeyCode, KeyboardLayout

CALIBRATION_VERSION = 1
MIN_WIDTH = 200.0
MIN_HEIGHT = 60.0


@dataclass(frozen=True)
class Calibration:
    """키보드 사각형 (추론 좌표계 1280x720 기준 픽셀)."""

    x: float
    y: float
    width: float
    height: float
    infer_width: int = INFER_WIDTH
    infer_height: int = INFER_HEIGHT

    @classmethod
    def default(cls, infer_width: int = INFER_WIDTH, infer_height: int = INFER_HEIGHT) -> "Calibration":
        width = infer_width * 0.6
        height = width / UNITS_W * UNITS_H
        x = (infer_width - width) / 2
        y = infer_height * 0.47         # 화면 가운데~아래쪽: 손이 화면 밖으로 잘리지 않는 높이
        return cls(x, y, width, height, infer_width, infer_height)

    def validated(self) -> "Calibration":
        """최소 크기 보장 + 추론 프레임 안으로 클램프."""
        w = min(max(self.width, MIN_WIDTH), float(self.infer_width))
        h = min(max(self.height, MIN_HEIGHT), float(self.infer_height))
        x = min(max(self.x, 0.0), self.infer_width - w)
        y = min(max(self.y, 0.0), self.infer_height - h)
        return replace(self, x=x, y=y, width=w, height=h)

    def moved(self, dx: float, dy: float) -> "Calibration":
        return replace(self, x=self.x + dx, y=self.y + dy).validated()

    def scaled(self, fw: float, fh: float) -> "Calibration":
        """중심을 유지하며 크기 조절."""
        cx, cy = self.x + self.width / 2, self.y + self.height / 2
        w, h = self.width * fw, self.height * fh
        return replace(self, x=cx - w / 2, y=cy - h / 2, width=w, height=h).validated()

    @classmethod
    def from_index_fingers(cls, left_index: Vec2, right_index: Vec2,
                           infer_width: int = INFER_WIDTH,
                           infer_height: int = INFER_HEIGHT) -> "Calibration":
        """왼손 검지 = F 키 중심, 오른손 검지 = J 키 중심이 되도록 키보드를 맞춘다."""
        base = KeyboardLayout(0, 0, UNITS_W, UNITS_H)  # 1 unit = 1px 인 기준 배치
        f, j = base.key_center(KeyCode.F), base.key_center(KeyCode.J)
        unit = (right_index.x - left_index.x) / (j.x - f.x)
        if unit <= 0:
            raise ValueError("왼손 검지가 오른손 검지보다 왼쪽에 있어야 합니다")
        width = UNITS_W * unit
        height = UNITS_H * unit
        x = left_index.x - f.x * unit
        y = (left_index.y + right_index.y) / 2 - f.y * unit
        return cls(x, y, width, height, infer_width, infer_height).validated()

    # --- 직렬화 -----------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["version"] = CALIBRATION_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(float(d["x"]), float(d["y"]), float(d["width"]), float(d["height"]),
                   int(d.get("infer_width", INFER_WIDTH)), int(d.get("infer_height", INFER_HEIGHT)))

    def save(self, path: str) -> None:
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, infer_width: int = INFER_WIDTH,
             infer_height: int = INFER_HEIGHT) -> "Calibration":
        """파일이 없거나 손상되면 기본값을 반환한다."""
        cal = cls.load_optional(path, infer_width, infer_height)
        return cal if cal is not None else cls.default(infer_width, infer_height)

    @classmethod
    def load_optional(cls, path: str, infer_width: int = INFER_WIDTH,
                      infer_height: int = INFER_HEIGHT) -> Optional["Calibration"]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cal = cls.from_dict(json.load(f))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[경고] 캘리브레이션 파일을 읽을 수 없어 기본값을 사용합니다: {path} ({e})")
            return None
        if (cal.infer_width, cal.infer_height) != (infer_width, infer_height):
            sx, sy = infer_width / cal.infer_width, infer_height / cal.infer_height
            cal = Calibration(cal.x * sx, cal.y * sy, cal.width * sx, cal.height * sy,
                              infer_width, infer_height)
        return cal.validated()

"""
schemas.py - Student ID Extraction Pipeline 타입 및 설정 정의

결과/설정 타입을 정의하여 테스트/로그/디버깅 편의성을 제공합니다.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    """파이프라인 설정"""
    conf_threshold: float = 0.6        # OCR confidence threshold
    margin_px: int = 2                 # Header crop margin (위 여백)
    allow_edit_distance_1: bool = True # 편집거리 1 허용 여부
    vlm_timeout_s: float = 10.0        # VLM 호출 timeout (초)


@dataclass
class StudentIdExtractionResult:
    """학번 추출 결과"""
    original_image: Any                # 원본 이미지
    header_image: Any | None           # 헤더 이미지 (레이아웃 실패 시 None)
    student_id: str | None             # 추출된 학번 (실패 시 None)
    meta: dict = field(default_factory=dict)  # 디버깅/로그용 메타정보

    # meta 권장 필드:
    # - stage: "layout" | "ocr" | "vlm" | "match"
    # - reason: "no_table" | "ocr_fail" | "vlm_unavailable" | "ambiguous_match" | ...
    # - ocr_conf: float
    # - used_vlm: bool
    # - vlm_error_type: str | None


@dataclass
class BBox:
    """Bounding Box (x1, y1, x2, y2) 좌표"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def y_top(self) -> float:
        """상단 y 좌표 (crop에 사용)"""
        return self.y1

    @classmethod
    def from_coordinate(cls, coord: list) -> "BBox":
        """PP-DocLayout 결과의 coordinate 리스트에서 생성"""
        # coordinate: [x1, y1, x2, y2]
        return cls(x1=coord[0], y1=coord[1], x2=coord[2], y2=coord[3])

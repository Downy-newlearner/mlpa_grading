"""
schemas.py - 답안 인식 파이프라인 데이터 스키마 정의

입력 메타데이터 및 출력 결과의 데이터 구조를 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import numpy as np


# =============================================================================
# 채점 타입 정의
# =============================================================================

class ScoringType(str, Enum):
    """채점 대상 타입"""
    BINARY = "binary"           # O/X, 체크마크
    OBJECTIVE = "objective"     # 객관식 (1,2,3,4,5)
    SHORT_ANSWER = "short_answer"  # 단답형 텍스트
    OTHERS = "others"           # 서술형/미채점


# =============================================================================
# 입력 메타데이터 스키마
# =============================================================================

@dataclass
class QuestionMeta:
    """개별 문제 메타데이터"""
    question_number: int                    # 문제 번호 (1-indexed)
    sub_question_count: int = 1             # 꼬리문제 개수 (기본 1 = 꼬리문제 없음)
    scoring_type: ScoringType = ScoringType.OBJECTIVE  # 채점 타입
    correct_answer: Optional[List[str]] = None  # 정답 리스트 (sub_question 개수만큼)
    points: Optional[List[float]] = None    # 배점 리스트

    @classmethod
    def from_dict(cls, data: dict) -> "QuestionMeta":
        """딕셔너리에서 생성"""
        scoring_type = data.get("scoring_type", "objective")
        if isinstance(scoring_type, str):
            scoring_type = ScoringType(scoring_type)
        
        return cls(
            question_number=data.get("question_number", 0),
            sub_question_count=data.get("sub_question_count", 1),
            scoring_type=scoring_type,
            correct_answer=data.get("correct_answer"),
            points=data.get("points")
        )


@dataclass
class AnswerSheetMeta:
    """답안지 메타데이터 (정답지 정보)"""
    exam_code: str
    questions: List[QuestionMeta] = field(default_factory=list)
    layout_type: Literal["case1", "case2"] = "case2"  # 레이아웃 타입
    total_questions: int = 0
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnswerSheetMeta":
        """딕셔너리에서 생성"""
        questions = [
            QuestionMeta.from_dict(q) 
            for q in data.get("questions", [])
        ]
        return cls(
            exam_code=data.get("exam_code", ""),
            questions=questions,
            layout_type=data.get("layout_type", "case2"),
            total_questions=len(questions)
        )


# =============================================================================
# 출력 결과 스키마
# =============================================================================

@dataclass
class AnswerRecognitionResult:
    """개별 문제 답안 인식 결과"""
    question_number: int                    # 문제 번호 (1-indexed)
    sub_question_number: Optional[int]      # 꼬리문제 번호 (1-indexed, 없으면 None)
    scoring_type: ScoringType               # 채점 타입
    rec_answer: Optional[str]               # 인식된 답안
    confidence: float                       # 신뢰도 (0~1)
    is_correct: Optional[bool] = None       # 정답 여부 (채점 후)
    points_earned: Optional[float] = None   # 획득 점수
    meta: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터
    
    # S3 연동 및 Fallback용 추가 필드
    s3_key: Optional[str] = None            # S3 업로드 키 (Fallback 시)
    roi_image: Optional[np.ndarray] = field(default=None, repr=False) # ROI 이미지 (업로드용, 직렬화 제외)
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "question_number": self.question_number,
            "sub_question_number": self.sub_question_number,
            "scoring_type": self.scoring_type.value if isinstance(self.scoring_type, ScoringType) else self.scoring_type,
            "rec_answer": self.rec_answer,
            "confidence": round(self.confidence, 4),
            "is_correct": self.is_correct,
            "points_earned": self.points_earned,
            "meta": self.meta,
            "s3_key": self.s3_key
        }


@dataclass
class AnswerSheetResult:
    """전체 답안지 인식 결과"""
    exam_code: str
    student_id: Optional[str] = None
    processed_at: Optional[str] = None      # ISO 8601 형식
    results: List[AnswerRecognitionResult] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    
    # 요약 정보
    total_questions: int = 0
    auto_graded: int = 0
    skipped: int = 0
    correct_count: int = 0
    total_points: float = 0.0
    earned_points: float = 0.0
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "exam_code": self.exam_code,
            "student_id": self.student_id,
            "processed_at": self.processed_at,
            "success": self.success,
            "error_message": self.error_message,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_questions": self.total_questions,
                "auto_graded": self.auto_graded,
                "skipped": self.skipped,
                "correct_count": self.correct_count,
                "total_points": self.total_points,
                "earned_points": self.earned_points
            }
        }
    
    def update_summary(self):
        """결과를 기반으로 요약 정보 갱신"""
        self.total_questions = len(self.results)
        self.auto_graded = sum(
            1 for r in self.results 
            if r.scoring_type != ScoringType.OTHERS
        )
        self.skipped = sum(
            1 for r in self.results 
            if r.scoring_type == ScoringType.OTHERS
        )
        self.correct_count = sum(
            1 for r in self.results 
            if r.is_correct is True
        )
        self.earned_points = sum(
            r.points_earned or 0 
            for r in self.results
        )


# =============================================================================
# ROI 및 중간 결과 스키마
# =============================================================================

@dataclass
class ROI:
    """Region of Interest"""
    x: int
    y: int
    width: int
    height: int
    image: Optional[np.ndarray] = None
    
    @property
    def bbox(self) -> tuple:
        """(x, y, w, h) 튜플 반환"""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def xyxy(self) -> tuple:
        """(x1, y1, x2, y2) 튜플 반환"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class SubQuestionSegment:
    """꼬리문제 세그먼트"""
    sub_number: int             # 꼬리문제 번호 (1-indexed)
    y_start: int                # 시작 Y좌표 (parent row 기준)
    y_end: int                  # 종료 Y좌표
    image: Optional[np.ndarray] = None
    valley_depth: Optional[float] = None  # Valley 깊이 (분리 신뢰도)
    
    @property
    def height(self) -> int:
        return self.y_end - self.y_start

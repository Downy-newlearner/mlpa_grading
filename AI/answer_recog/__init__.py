"""
answer_recog - 답안 인식 모듈

원본 이미지에서 Answer 섹션을 찾고, OCR/VLM을 통해 답안을 추출하여
정답과 비교하는 채점 파이프라인을 제공합니다.

주요 컴포넌트:
- AnswerRecognitionPipeline: 메인 파이프라인 클래스
- find_answer_section: Answer 섹션 추출
- segment_rows: Row 분할
- segment_sub_questions: 꼬리문제 분리 (Y-Projection)
- recognize_answers: 간편 함수
- FallbackStore: Fallback 관리
"""

# Layout & Section Detection
from .find_answer_section import find_answer_section, AnswerSectionResult

# Row Segmentation
from .row_segmentation import (
    segment_rows,
    segment_rows_recursive,
    RowSegment,
    RowSegmentationResult
)

# Sub-Question Segmentation
from .sub_question_segmentation import (
    segment_sub_questions,
    SubQuestionSegment
)

# Schemas
from .schemas import (
    ScoringType,
    QuestionMeta,
    AnswerSheetMeta,
    AnswerRecognitionResult,
    AnswerSheetResult
)

# Main Pipeline
from .pipeline import (
    AnswerRecognitionPipeline,
    PipelineConfig,
    recognize_answers
)

# ROI Extraction & Fallback
from .roi_extraction import (
    AnswerROI,
    FallbackUploadResult,
    FallbackStore,
    extract_roi_from_row,
    create_answer_rois,
    upload_fallback_rois,
    get_fallback_store
)

__all__ = [
    # Pipeline
    "AnswerRecognitionPipeline",
    "PipelineConfig",
    "recognize_answers",
    # Section Detection
    "find_answer_section",
    "AnswerSectionResult",
    # Row Segmentation
    "segment_rows",
    "segment_rows_recursive",
    "RowSegment",
    "RowSegmentationResult",
    # Sub-Question Segmentation
    "segment_sub_questions",
    "SubQuestionSegment",
    # Schemas
    "ScoringType",
    "QuestionMeta",
    "AnswerSheetMeta",
    "AnswerRecognitionResult",
    "AnswerSheetResult",
    # ROI Extraction & Fallback
    "AnswerROI",
    "FallbackUploadResult",
    "FallbackStore",
    "extract_roi_from_row",
    "create_answer_rois",
    "upload_fallback_rois",
    "get_fallback_store",
]


"""
pipeline.py - 답안 인식 메인 파이프라인

전체 답안 인식 흐름을 조율하는 메인 파이프라인입니다.

흐름:
1. 이미지 입력
2. Layout Detection (PP-DocLayout) → Table 탐지
3. Answer Section Crop (X-axis Projection)
4. Row Segmentation:
   - Case 1: 주 문제 가로선 + Y-Projection 꼬리문제 분리
   - Case 2: 전체 가로선 기반
5. Answer Extraction (scoring_type별 처리)
6. 결과 반환

사용법:
    from answer_recog.pipeline import AnswerRecognitionPipeline
    
    pipeline = AnswerRecognitionPipeline(layout_model, ocr_model)
    result = pipeline.process(image, metadata)
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple, Any, Dict
from dataclasses import dataclass

# 상위 디렉토리 import 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 내부 모듈 import
from .schemas import (
    ScoringType, 
    QuestionMeta, 
    AnswerSheetMeta, 
    AnswerRecognitionResult, 
    AnswerSheetResult,
    SubQuestionSegment
)
from .find_answer_section import find_answer_section, AnswerSectionResult
from .row_segmentation import (
    segment_rows, 
    segment_rows_recursive, 
    RowSegment, 
    RowSegmentationResult
)
from .sub_question_segmentation import segment_sub_questions
from .answer_extraction import extract_text_from_row, refined_answer


# =============================================================================
# 파이프라인 설정
# =============================================================================

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # Layout Detection
    enable_deskew: bool = True
    max_skew_angle: float = 5.0
    
    # Row Segmentation
    min_row_height: int = 30
    max_row_height: int = 200
    use_morphological: bool = True
    min_line_length_ratio: float = 0.3
    
    # Sub-question Segmentation
    min_sub_height: int = 15
    
    # Answer Extraction
    min_confidence: float = 0.3  # 이 이하면 "unknown" 처리
    
    # OCR
    ocr_lang: str = "en"
    
    # Debug
    debug_mode: bool = False
    debug_output_dir: Optional[str] = None


# =============================================================================
# 메인 파이프라인 클래스
# =============================================================================

class AnswerRecognitionPipeline:
    """
    답안 인식 메인 파이프라인
    
    전체 답안 인식 흐름을 조율하고, 각 모듈을 호출하여 결과를 반환합니다.
    """
    
    def __init__(
        self,
        layout_model: Any = None,
        ocr_model: Any = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            layout_model: PP-DocLayout 모델 (None이면 내부에서 로드)
            ocr_model: PaddleOCR 모델 (None이면 내부에서 로드)
            config: 파이프라인 설정
        """
        self.layout_model = layout_model
        self.ocr_model = ocr_model
        self.config = config or PipelineConfig()
        
        # 디버그 출력 디렉토리 생성
        if self.config.debug_mode and self.config.debug_output_dir:
            os.makedirs(self.config.debug_output_dir, exist_ok=True)
    
    def process(
        self,
        image: np.ndarray,
        metadata: AnswerSheetMeta,
        student_id: Optional[str] = None
    ) -> AnswerSheetResult:
        """
        답안지 이미지를 처리하여 답안을 인식합니다.
        
        Args:
            image: 답안지 이미지 (BGR)
            metadata: 정답지 메타데이터
            student_id: 학생 학번 (선택)
            
        Returns:
            AnswerSheetResult
        """
        # 결과 초기화
        result = AnswerSheetResult(
            exam_code=metadata.exam_code,
            student_id=student_id,
            processed_at=datetime.now().isoformat()
        )
        
        try:
            # Step 1: Answer Section 추출
            answer_section_result = self._extract_answer_section(image)
            
            if not answer_section_result.success:
                result.success = False
                result.error_message = "Failed to extract answer section"
                return result
            
            answer_image = answer_section_result.answer_section_image
            
            # Step 2: Row Segmentation
            row_result = self._segment_rows(answer_image, metadata)
            
            if not row_result.success:
                result.success = False
                result.error_message = "Failed to segment rows"
                return result
            
            # Step 3: 메타데이터와 Row 매핑 + Answer Extraction
            recognition_results = self._extract_answers_with_metadata(
                row_result.rows,
                metadata
            )
            
            result.results = recognition_results
            
            # Step 4: 채점 (정답과 비교)
            self._grade_answers(result, metadata)
            
            # Step 5: 요약 정보 갱신
            result.update_summary()
            
            # 디버그 저장
            if self.config.debug_mode:
                self._save_debug_output(
                    image, 
                    answer_section_result, 
                    row_result, 
                    result
                )
            
        except Exception as e:
            result.success = False
            result.error_message = f"Pipeline error: {str(e)}"
            import traceback
            if self.config.debug_mode:
                print(f"[Pipeline Error] {traceback.format_exc()}")
        
        return result
    
    def _extract_answer_section(
        self, 
        image: np.ndarray
    ) -> AnswerSectionResult:
        """
        Answer Section 추출
        
        PP-DocLayout으로 Table을 탐지하고, X-axis Projection으로 Answer 컬럼을 추출합니다.
        """
        return find_answer_section(
            image,
            layout_model=self.layout_model,
            enable_deskew=self.config.enable_deskew,
            max_skew_angle=self.config.max_skew_angle
        )
    
    def _segment_rows(
        self,
        answer_image: np.ndarray,
        metadata: AnswerSheetMeta
    ) -> RowSegmentationResult:
        """
        Row Segmentation
        
        Case 1 (주 문제만 가로선): 주 문제 분할 후 꼬리문제 Y-Projection으로 2차 분할
        Case 2 (모든 문제 가로선): 전체 가로선 기반 분할
        """
        # 1차 분할: Morphological 가로선 탐지
        row_result = segment_rows_recursive(
            answer_image,
            min_row_height=self.config.min_row_height,
            max_row_height=self.config.max_row_height
        )
        
        if not row_result.success:
            return row_result
        
        # Case 1 처리: 꼬리문제가 있는 경우 2차 분할
        if metadata.layout_type == "case1":
            row_result = self._apply_sub_question_segmentation(
                row_result,
                metadata,
                answer_image
            )
        
        return row_result
    
    def _apply_sub_question_segmentation(
        self,
        row_result: RowSegmentationResult,
        metadata: AnswerSheetMeta,
        answer_image: np.ndarray
    ) -> RowSegmentationResult:
        """
        Case 1: 꼬리문제 Y-Projection 분할 적용
        
        주 문제 Row와 메타데이터의 question 개수가 일치한다고 가정하고,
        각 Row에 대해 sub_question_count > 1이면 2차 분할을 수행합니다.
        """
        final_rows = []
        
        # Row 수와 Question 수가 맞는지 확인
        if len(row_result.rows) != len(metadata.questions):
            # 불일치 시 경고 로그
            print(f"[Warning] Row count ({len(row_result.rows)}) != "
                  f"Question count ({len(metadata.questions)})")
        
        for i, row in enumerate(row_result.rows):
            # 해당하는 메타데이터 찾기
            if i < len(metadata.questions):
                question_meta = metadata.questions[i]
                sub_count = question_meta.sub_question_count
            else:
                sub_count = 1
            
            if sub_count <= 1:
                # 꼬리문제 없음 → 그대로 사용
                row.row_number = len(final_rows)
                final_rows.append(row)
            else:
                # 꼬리문제 있음 → Y-Projection으로 2차 분할
                sub_segments, sub_meta = segment_sub_questions(
                    row.row_image,
                    expected_count=sub_count,
                    min_sub_height=self.config.min_sub_height,
                    debug=self.config.debug_mode
                )
                
                # SubQuestionSegment → RowSegment 변환
                for seg in sub_segments:
                    # 절대 좌표로 변환
                    abs_y_start = row.y_start + seg.y_start
                    abs_y_end = row.y_start + seg.y_end
                    
                    new_row = RowSegment(
                        row_number=len(final_rows),
                        y_start=abs_y_start,
                        y_end=abs_y_end,
                        row_image=seg.image,
                        valley_depth=seg.valley_depth
                    )
                    final_rows.append(new_row)
        
        # 결과 업데이트
        row_result.rows = final_rows
        row_result.meta["sub_question_segmentation"] = True
        row_result.meta["final_row_count"] = len(final_rows)
        
        return row_result
    
    def _extract_answers_with_metadata(
        self,
        rows: List[RowSegment],
        metadata: AnswerSheetMeta
    ) -> List[AnswerRecognitionResult]:
        """
        Row에서 답안을 추출하고 메타데이터와 매핑합니다.
        
        Row 순서와 문제 순서가 일치한다고 가정합니다.
        각 Row → (question_number, sub_question_number)
        """
        results = []
        row_idx = 0
        
        for q_idx, question in enumerate(metadata.questions):
            question_number = question.question_number
            sub_count = question.sub_question_count
            scoring_type = question.scoring_type
            
            for sub_idx in range(sub_count):
                if row_idx >= len(rows):
                    # Row 부족 → 빈 결과 추가
                    result = AnswerRecognitionResult(
                        question_number=question_number,
                        sub_question_number=sub_idx + 1 if sub_count > 1 else None,
                        scoring_type=scoring_type,
                        rec_answer=None,
                        confidence=0.0,
                        meta={"error": "Row not found"}
                    )
                    results.append(result)
                    continue
                
                row = rows[row_idx]
                row_idx += 1
                
                # 채점 타입별 답안 추출
                rec_answer, confidence, extract_meta = self._extract_answer_by_type(
                    row.row_image,
                    scoring_type
                )
                
                result = AnswerRecognitionResult(
                    question_number=question_number,
                    sub_question_number=sub_idx + 1 if sub_count > 1 else None,
                    scoring_type=scoring_type,
                    rec_answer=rec_answer,
                    confidence=confidence,
                    meta=extract_meta,
                    roi_image=row.row_image.copy()  # ROI 이미지 저장 (복사본)
                )
                results.append(result)
        
        return results
    
    def _extract_answer_by_type(
        self,
        row_image: np.ndarray,
        scoring_type: ScoringType
    ) -> Tuple[Optional[str], float, dict]:
        """
        채점 타입별 답안 추출
        
        Returns:
            (인식된 답안, 신뢰도, 메타데이터)
        """
        meta = {"scoring_type": scoring_type.value}
        
        if scoring_type == ScoringType.OTHERS:
            # 미채점 대상
            return None, 0.0, {"skipped": True, "reason": "Not a scoring target"}
        
        if scoring_type == ScoringType.BINARY:
            # O/X, 체크마크 검출
            answer, confidence = self._detect_binary_mark(row_image)
            meta["mark_type"] = "check" if answer else "empty"
            return answer, confidence, meta
        
        # objective, short_answer: OCR 수행
        raw_text, confidence = extract_text_from_row(row_image)
        meta["raw_ocr_text"] = raw_text
        
        # 답안 정제
        cleaned_answer, is_valid = refined_answer(raw_text)
        
        if not is_valid or confidence < self.config.min_confidence:
            meta["low_confidence"] = True
            return "unknown", confidence, meta
        
        return cleaned_answer, confidence, meta
    
    def _detect_binary_mark(
        self,
        row_image: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        O/X, 체크마크 검출
        
        TODO: 실제 체크마크 검출 로직 구현
        현재는 간단한 픽셀 밀도 기반 판별
        """
        # Grayscale 변환
        if len(row_image.shape) == 3:
            gray = cv2.cvtColor(row_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = row_image.copy()
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 픽셀 밀도 계산
        density = np.sum(binary) / (binary.size * 255)
        
        # 임계값 기반 판별 (추후 개선 필요)
        if density > 0.05:
            return "true", min(density * 5, 1.0)
        else:
            return "false", 1.0 - density
    
    def _grade_answers(
        self,
        result: AnswerSheetResult,
        metadata: AnswerSheetMeta
    ):
        """
        인식된 답안을 정답과 비교하여 채점합니다.
        """
        for rec_result in result.results:
            # 해당 문제 메타데이터 찾기
            question_meta = None
            for q in metadata.questions:
                if q.question_number == rec_result.question_number:
                    question_meta = q
                    break
            
            if question_meta is None:
                continue
            
            # 미채점 대상 스킵
            if rec_result.scoring_type == ScoringType.OTHERS:
                continue
            
            # 정답 비교
            if question_meta.correct_answer is None:
                continue
            
            # sub_question_number에 따른 정답 인덱스
            if rec_result.sub_question_number is not None:
                answer_idx = rec_result.sub_question_number - 1
            else:
                answer_idx = 0
            
            if answer_idx < len(question_meta.correct_answer):
                correct = question_meta.correct_answer[answer_idx]
                
                # 대소문자 무시 비교
                if rec_result.rec_answer and correct:
                    rec_result.is_correct = (
                        rec_result.rec_answer.lower() == correct.lower()
                    )
                else:
                    rec_result.is_correct = False
                
                # 점수 계산
                if question_meta.points and answer_idx < len(question_meta.points):
                    if rec_result.is_correct:
                        rec_result.points_earned = question_meta.points[answer_idx]
                    else:
                        rec_result.points_earned = 0.0
    
    def _save_debug_output(
        self,
        original_image: np.ndarray,
        answer_section_result: AnswerSectionResult,
        row_result: RowSegmentationResult,
        final_result: AnswerSheetResult
    ):
        """
        디버그 출력 저장
        """
        if not self.config.debug_output_dir:
            return
        
        output_dir = self.config.debug_output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 원본 이미지 저장
        cv2.imwrite(
            os.path.join(output_dir, f"{timestamp}_01_original.jpg"),
            original_image
        )
        
        # Answer section 저장
        if answer_section_result.answer_section_image is not None:
            cv2.imwrite(
                os.path.join(output_dir, f"{timestamp}_02_answer_section.jpg"),
                answer_section_result.answer_section_image
            )
        
        # Row 시각화 저장
        from .row_segmentation import visualize_row_segmentation
        visualize_row_segmentation(
            row_result,
            save_path=os.path.join(output_dir, f"{timestamp}_03_row_segmentation.jpg")
        )
        
        # 결과 JSON 저장
        import json
        with open(os.path.join(output_dir, f"{timestamp}_04_result.json"), "w") as f:
            json.dump(final_result.to_dict(), f, indent=2, ensure_ascii=False)


# =============================================================================
# 간편 함수 (Legacy 호환)
# =============================================================================

def recognize_answers(
    image: np.ndarray,
    metadata_dict: dict,
    layout_model: Any = None,
    config: Optional[PipelineConfig] = None
) -> dict:
    """
    답안 인식 간편 함수 (단일 호출용)
    
    Args:
        image: 답안지 이미지
        metadata_dict: 정답지 메타데이터 딕셔너리
        layout_model: PP-DocLayout 모델 (선택)
        config: 파이프라인 설정 (선택)
        
    Returns:
        결과 딕셔너리
    """
    metadata = AnswerSheetMeta.from_dict(metadata_dict)
    pipeline = AnswerRecognitionPipeline(layout_model=layout_model, config=config)
    result = pipeline.process(image, metadata)
    return result.to_dict()


# =============================================================================
# 테스트용 메인
# =============================================================================

if __name__ == "__main__":
    # 테스트 메타데이터
    test_metadata = {
        "exam_code": "AI_2023_MID",
        "layout_type": "case2",
        "questions": [
            {"question_number": 1, "sub_question_count": 1, "scoring_type": "objective", 
             "correct_answer": ["3"], "points": [2]},
            {"question_number": 2, "sub_question_count": 1, "scoring_type": "objective",
             "correct_answer": ["1"], "points": [2]},
            {"question_number": 3, "sub_question_count": 3, "scoring_type": "objective",
             "correct_answer": ["2", "4", "1"], "points": [2, 2, 2]},
        ]
    }
    
    # 테스트 이미지 (실제 경로로 대체 필요)
    test_image_path = "test_answer_sheet.jpg"
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        result = recognize_answers(image, test_metadata)
        print(result)
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid test image path.")

"""
roi_extraction.py - ROI(Region of Interest) 추출 및 Fallback 처리 모듈

Row 이미지에서 답안 영역(ROI)을 추출하고,
낮은 confidence 결과의 ROI 이미지를 S3에 업로드합니다.

핵심 기능:
1. Row 이미지에서 답안 영역 ROI 추출
2. Fallback ROI 이미지 S3 업로드
3. Fallback 결과 관리
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime


# =============================================================================
# ROI 데이터 구조
# =============================================================================

@dataclass
class AnswerROI:
    """답안 영역 ROI"""
    question_number: int            # 문제 번호 (1-indexed)
    sub_question_number: int        # 꼬리문제 번호 (0=없음, 1,2,3...)
    roi_image: np.ndarray           # ROI 이미지
    bbox: Tuple[int, int, int, int] # (x, y, w, h) - Row 내 상대 좌표
    row_index: int                  # Row 인덱스
    
    # 인식 결과
    rec_answer: Optional[str] = None
    confidence: float = 0.0
    scoring_type: str = "objective"
    
    # Fallback 상태
    is_fallback: bool = False       # Fallback 필요 여부
    s3_key: Optional[str] = None    # S3 업로드 키 (업로드된 경우)


@dataclass
class FallbackUploadResult:
    """Fallback 업로드 결과"""
    exam_code: str
    student_id: str
    uploaded_count: int = 0
    failed_count: int = 0
    uploaded_keys: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "exam_code": self.exam_code,
            "student_id": self.student_id,
            "uploaded_count": self.uploaded_count,
            "failed_count": self.failed_count,
            "uploaded_keys": self.uploaded_keys,
            "errors": self.errors
        }


# =============================================================================
# ROI 추출 함수
# =============================================================================

def extract_roi_from_row(
    row_image: np.ndarray,
    padding: int = 5,
    threshold_ratio: float = 0.03, # 기본값을 높여서 확실한 텍스트만 잡도록 (공백 제거 강화)
    margin_crop: int = 5 
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Row 이미지에서 상하좌우 공백을 제거하여 실제 컨텐츠 ROI를 추출합니다.
    높은 Threshold로 1차 시도 후, 실패하면(빈 이미지) 낮은 Threshold로 재시도합니다.
    """
    if row_image is None or row_image.size == 0:
        return row_image, (0, 0, 0, 0)
        
    # 1차 시도 (High Threshold -> Clean Crop)
    roi, bbox = _extract_roi_core(row_image, padding, threshold_ratio, margin_crop)
    
    # 2차 시도 (Low Threshold -> Recover Faint Text)
    if roi.size == 0 or (bbox[2] == 0 or bbox[3] == 0):
        # 10x10 빈 이미지가 반환된 경우 등
        # print(f"    [ROI] Retry with low threshold for {row_image.shape}")
        roi, bbox = _extract_roi_core(row_image, padding, 0.005, margin_crop)
        
    return roi, bbox

def _extract_roi_core(
    row_image: np.ndarray,
    padding: int,
    threshold_ratio: float,
    margin_crop: int
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = row_image.shape[:2]
    
    # 0. 상하단 강제 Crop
    if h > 2 * margin_crop:
        start_y_crop = margin_crop
        end_y_crop = h - margin_crop
        working_img = row_image[start_y_crop:end_y_crop, :].copy()
    else:
        start_y_crop = 0
        end_y_crop = h
        working_img = row_image.copy()
        
    wh, ww = working_img.shape[:2]

    # 1. 전처리
    if len(working_img.shape) == 3:
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = working_img.copy()
        
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # 2. Y-Projection
    y_proj = np.sum(binary, axis=1) / 255
    y_indices = np.where(y_proj > (ww * threshold_ratio))[0]
    
    if len(y_indices) == 0:
        empty_roi = np.zeros((10, 10, 3) if len(row_image.shape)==3 else (10,10), dtype=np.uint8)
        return empty_roi, (0, 0, 0, 0)
        
    y_start_local = max(0, y_indices[0] - padding)
    y_end_local = min(wh, y_indices[-1] + 1 + padding)
    
    # 3. X-Projection (Cropped Y)
    cropped_content = binary[y_start_local:y_end_local, :]
    x_proj = np.sum(cropped_content, axis=0) / 255
    x_indices = np.where(x_proj > ((y_end_local - y_start_local) * threshold_ratio))[0]
    
    if len(x_indices) == 0:
        x_start_local, x_end_local = 0, ww
        # X축으로 아무것도 안 잡히면 -> 빈 이미지로 취급하는 게 나을 수도 있음
        # 하지만 Y축은 잡혔으니 일단 전체 너비 반환
    else:
        x_start_local = max(0, x_indices[0] - padding)
        x_end_local = min(ww, x_indices[-1] + 1 + padding)
        
    roi = working_img[y_start_local:y_end_local, x_start_local:x_end_local].copy()
    
    bbox = (
        x_start_local, 
        start_y_crop + y_start_local, 
        x_end_local - x_start_local, 
        y_end_local - y_start_local
    )
    
    return roi, bbox


def create_answer_rois(
    rows: List[Any],  # List[RowSegment]
    question_metadata: List[dict],
    confidence_threshold: float = 0.7
) -> List[AnswerROI]:
    """
    Row 리스트와 메타데이터를 기반으로 AnswerROI 리스트를 생성합니다.
    
    Args:
        rows: RowSegment 리스트
        question_metadata: 문제별 메타데이터 리스트
        confidence_threshold: Fallback 임계값
        
    Returns:
        AnswerROI 리스트
    """
    rois = []
    row_idx = 0
    
    for q_meta in question_metadata:
        q_num = q_meta.get("question_number", 0)
        sub_count = q_meta.get("sub_question_count", 1)
        scoring_type = q_meta.get("scoring_type", "objective")
        
        for sub_idx in range(sub_count):
            if row_idx >= len(rows):
                break
            
            row = rows[row_idx]
            row_idx += 1
            
            # ROI 추출
            roi_image, bbox = extract_roi_from_row(row.row_image)
            
            roi = AnswerROI(
                question_number=q_num,
                sub_question_number=sub_idx if sub_count > 1 else 0,
                roi_image=roi_image,
                bbox=bbox,
                row_index=row.row_number,
                scoring_type=scoring_type
            )
            rois.append(roi)
    
    return rois


# =============================================================================
# S3 Fallback 업로드
# =============================================================================

def generate_roi_s3_key(
    exam_code: str,
    student_id: str,
    question_number: int,
    sub_question_number: int
) -> str:
    """
    ROI 이미지의 S3 키를 생성합니다.
    
    경로 형식: answer/{exam_code}/{학번}/{문제번호}/{꼬리문제번호}/roi_q{Q}_s{S}.jpg
    """
    filename = f"roi_q{question_number}_s{sub_question_number}.jpg"
    return f"answer/{exam_code}/{student_id}/{question_number}/{sub_question_number}/{filename}"


def upload_fallback_rois(
    rois: List[AnswerROI],
    exam_code: str,
    student_id: str,
    s3_manager: Any,
    confidence_threshold: float = 0.7
) -> FallbackUploadResult:
    """
    낮은 confidence의 ROI 이미지들을 S3에 업로드합니다.
    
    Args:
        rois: AnswerROI 리스트
        exam_code: 시험 코드
        student_id: 학번
        s3_manager: S3 매니저 인스턴스
        confidence_threshold: Fallback 임계값
        
    Returns:
        FallbackUploadResult
    """
    result = FallbackUploadResult(
        exam_code=exam_code,
        student_id=student_id
    )
    
    if s3_manager is None or not s3_manager.is_ready:
        result.errors.append("S3 manager not available")
        return result
    
    for roi in rois:
        # Fallback 조건 체크
        if roi.confidence >= confidence_threshold:
            continue
        
        # scoring_type이 "others"면 skip
        if roi.scoring_type == "others":
            continue
        
        roi.is_fallback = True
        
        try:
            # S3 키 생성
            s3_key = generate_roi_s3_key(
                exam_code,
                student_id,
                roi.question_number,
                roi.sub_question_number
            )
            
            # 이미지를 바이트로 변환
            _, buffer = cv2.imencode('.jpg', roi.roi_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = buffer.tobytes()
            
            # S3 업로드
            s3_manager.upload_bytes(image_bytes, s3_key, content_type="image/jpeg")
            
            roi.s3_key = s3_key
            result.uploaded_keys.append(s3_key)
            result.uploaded_count += 1
            
        except Exception as e:
            result.errors.append(f"Q{roi.question_number}-{roi.sub_question_number}: {str(e)}")
            result.failed_count += 1
    
    return result


# =============================================================================
# Fallback 결과 저장/조회
# =============================================================================

class FallbackStore:
    """
    Fallback ROI 정보를 메모리에 저장하고 관리합니다.
    채점 단계에서 Fallback 수정값을 병합할 때 사용합니다.
    """
    
    def __init__(self):
        # {exam_code: {student_id: [AnswerROI, ...]}}
        self._store: Dict[str, Dict[str, List[AnswerROI]]] = {}
        
        # {exam_code: {student_id: {(q_num, sub_num): corrected_answer}}}
        self._corrections: Dict[str, Dict[str, Dict[Tuple[int, int], str]]] = {}
    
    def add_rois(
        self,
        exam_code: str,
        student_id: str,
        rois: List[AnswerROI]
    ):
        """ROI 리스트 저장"""
        if exam_code not in self._store:
            self._store[exam_code] = {}
        
        self._store[exam_code][student_id] = rois
    
    def get_fallback_rois(
        self,
        exam_code: str,
        student_id: Optional[str] = None
    ) -> Dict[str, List[AnswerROI]]:
        """Fallback이 필요한 ROI 조회"""
        if exam_code not in self._store:
            return {}
        
        if student_id:
            rois = self._store[exam_code].get(student_id, [])
            fallback_rois = [r for r in rois if r.is_fallback]
            return {student_id: fallback_rois} if fallback_rois else {}
        
        # 전체 학생
        result = {}
        for sid, rois in self._store[exam_code].items():
            fallback_rois = [r for r in rois if r.is_fallback]
            if fallback_rois:
                result[sid] = fallback_rois
        
        return result
    
    def apply_corrections(
        self,
        exam_code: str,
        corrections: List[dict]
    ):
        """
        사용자 수정값 적용
        
        corrections 형식:
        [
            {"studentId": "20201234", "questionNumber": 1, "subQuestionNumber": 0, "answer": "3"},
            ...
        ]
        """
        if exam_code not in self._corrections:
            self._corrections[exam_code] = {}
        
        for corr in corrections:
            student_id = corr.get("studentId", corr.get("student_id"))
            q_num = corr.get("questionNumber", corr.get("question_number"))
            sub_num = corr.get("subQuestionNumber", corr.get("sub_question_number", 0))
            answer = corr.get("answer")
            
            if student_id not in self._corrections[exam_code]:
                self._corrections[exam_code][student_id] = {}
            
            self._corrections[exam_code][student_id][(q_num, sub_num)] = answer
    
    def get_corrected_answer(
        self,
        exam_code: str,
        student_id: str,
        question_number: int,
        sub_question_number: int
    ) -> Optional[str]:
        """수정된 답안 조회"""
        if exam_code not in self._corrections:
            return None
        if student_id not in self._corrections[exam_code]:
            return None
        
        return self._corrections[exam_code][student_id].get(
            (question_number, sub_question_number)
        )
    
    def get_fallback_summary(self, exam_code: str) -> dict:
        """Fallback 요약 정보"""
        if exam_code not in self._store:
            return {"exam_code": exam_code, "students": 0, "total_fallbacks": 0}
        
        students = self._store[exam_code]
        total_fallbacks = sum(
            len([r for r in rois if r.is_fallback])
            for rois in students.values()
        )
        
        corrected_count = 0
        if exam_code in self._corrections:
            corrected_count = sum(
                len(corrections)
                for corrections in self._corrections[exam_code].values()
            )
        
        return {
            "exam_code": exam_code,
            "students": len(students),
            "total_fallbacks": total_fallbacks,
            "corrected_count": corrected_count,
            "pending_count": total_fallbacks - corrected_count
        }
    
    def clear_exam(self, exam_code: str):
        """시험 데이터 삭제"""
        if exam_code in self._store:
            del self._store[exam_code]
        if exam_code in self._corrections:
            del self._corrections[exam_code]


# 전역 Fallback 저장소
_fallback_store = None

def get_fallback_store() -> FallbackStore:
    """전역 Fallback 저장소 반환"""
    global _fallback_store
    if _fallback_store is None:
        _fallback_store = FallbackStore()
    return _fallback_store

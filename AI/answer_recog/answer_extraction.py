"""
answer_extraction.py - Row 이미지에서 정답 추출

각 Row 이미지에서 OCR을 수행하여 텍스트를 추출하고,
정규식 등을 사용하여 최종 답안을 정제합니다.

기능:
1. Row 이미지 전처리 (텍스트 강조)
2. PaddleOCR 수행
3. 답안 패턴 매칭 (객관식/단답형)
"""

import cv2
import numpy as np
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from paddleocr import PaddleOCR

# OCR 모델 전역 인스턴스 (Lazy loading)
_ocr_model = None

def get_ocr_model():
    """PaddleOCR 모델 로드 및 반환"""
    global _ocr_model
    if _ocr_model is None:
        # 영문/숫자 인식에 최적화된 모델 사용
        # use_angle_cls=True: 텍스트 방향 보정
        # lang='en': 영문/숫자 위주 (한글이 섞여있어도 인식 가능)
        _ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    return _ocr_model


@dataclass
class AnswerResult:
    """개별 문제의 답안 인식 결과"""
    row_number: int          # 문제 번호 (0-indexed sequence)
    recognized_text: str     # OCR 원본 텍스트
    answser: str             # 정제된 답안
    confidence: float        # 확신도 (0~1)
    is_valid: bool           # 유효한 답안 형식인지 여부
    meta: Dict[str, Any]     # 디버깅용 메타 데이터


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """OCR 인식률을 높이기 위한 이미지 전처리"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Contrast Limited HIM
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # PaddleOCR(Paddlex) 입력은 3채널이어야 함
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


def extract_text_from_row(row_image: np.ndarray) -> Tuple[str, float]:
    """Row 이미지에서 텍스트와 평균 confidence 추출"""
    ocr = get_ocr_model()
    
    # 전처리
    processed_img = preprocess_for_ocr(row_image)
    
    # OCR 실행
    result = ocr.ocr(processed_img)
    
    if not result:
        return "", 0.0
    
    # Paddlex OCR 결과 파싱
    # result = [{'rec_texts': [...], 'rec_scores': [...], ...}]
    try:
        if isinstance(result, list) and len(result) > 0 and 'rec_texts' in result[0]:
            texts = result[0]['rec_texts']
            scores = result[0]['rec_scores']
            
            if not texts:
                return "", 0.0
            
            full_text = " ".join(texts)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            return full_text, avg_conf
            
        elif isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            # 기존 PaddleOCR 구조 호환용 (혹시 모를 대비)
            texts = []
            confidences = []
            for line in result[0]:
                if len(line) >= 2 and len(line[1]) >= 2:
                    texts.append(line[1][0])
                    confidences.append(line[1][1])
            
            full_text = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return full_text, avg_conf
            
    except Exception as e:
        print(f"Error parsing OCR result: {e}")
        return "", 0.0
        
    return "", 0.0


def refined_answer(text: str) -> Tuple[str, bool]:
    """
    OCR 텍스트를 정제하여 최종 답안 형식으로 변환
    """
    if not text:
        return "", False
    
    # 공백 제거
    text = text.replace(" ", "")
    
    # 1. 원문자 변환 매핑
    circle_map = {
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
        '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
        '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
        '❶': '1', '❷': '2', '❸': '3', '❹': '4', '❺': '5',
        'Ⓐ': 'A', 'Ⓑ': 'B', 'Ⓒ': 'C', 'Ⓓ': 'D', 'Ⓔ': 'E'
    }
    
    for circle, number in circle_map.items():
        text = text.replace(circle, number)
        
    # 2. 괄호 내 텍스트 추출
    # 예: "(3)" -> "3", "[정답]3" -> "3"
    match = re.search(r'[\(\[\{]([\d\w]{1,2})[\)\]\}]', text)
    if match:
        return match.group(1), True
    
    # 3. 특수문자 제거 (숫자, 알파벳, 쉼표, 하이픈만 허용)
    cleaned = re.sub(r'[^0-9a-zA-Z,\-]', '', text)
    
    # 4. 유효성 검사
    if len(cleaned) == 0:
        return "", False
        
    return cleaned, True


def extract_answers_from_rows(
    rows: List[Any]  # List[RowSegment]
) -> List[AnswerResult]:
    """모든 Row에서 답안 추출"""
    results = []
    
    for row in rows:
        # OCR 수행
        raw_text, conf = extract_text_from_row(row.row_image)
        
        # 답안 정제
        answer, is_valid = refined_answer(raw_text)
        
        result = AnswerResult(
            row_number=row.row_number,
            recognized_text=raw_text,
            answser=answer,
            confidence=conf,
            is_valid=is_valid,
            meta={}
        )
        results.append(result)
        
    return results

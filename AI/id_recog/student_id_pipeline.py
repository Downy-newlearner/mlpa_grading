"""
student_id_pipeline.py - 학번 추출 파이프라인 오케스트레이션

흐름(순서/분기)만 담당하며, 규칙/정책은 다른 모듈 호출로 위임합니다.

처리 흐름:
1. Layout detect → 모든 bbox 탐지
2. Table 제외한 각 bbox를 crop
3. 각 crop에 PP-OCRv5 수행 → 8자리 숫자 패턴 찾기
4. student_id_list와 매칭
5. (필요 시) VLM fallback
"""

import numpy as np
from PIL import Image
from typing import Any

from schemas import StudentIdExtractionResult, Config, BBox
from layout import (
    detect_all_bboxes,
    get_non_table_boxes,
    get_table_boxes,
    crop_bbox,
    make_header_image,
    LayoutBox
)
from ocr import ppocr_extract, vlm_extract_student_id
from normalize_and_validate import (
    normalize_candidate,
    is_valid_format,
    should_fallback,
    match_to_student_list
)


def extract_student_id(
    original_image: np.ndarray | Image.Image,
    student_id_list: list[str],
    layout_model: Any,
    ocr_model: Any,
    vlm_client: Any = None,
    config: Config | None = None
) -> StudentIdExtractionResult:
    """
    답안지 이미지에서 학번을 추출합니다.
    
    처리 흐름:
    1. Layout detect → 모든 bbox 탐지
    2. Table 제외한 각 bbox를 crop하여 PP-OCRv5 수행
    3. 8자리 숫자 패턴을 찾아 student_id_list와 매칭
    4. (필요 시) VLM fallback (header 이미지 사용)
    5. 결과 반환: {original_image, header_image, student_id}
    
    Args:
        original_image: 원본 답안지 이미지
        student_id_list: 유효한 학번 리스트
        layout_model: PP-DocLayout_plus-L 모델 객체
        ocr_model: PP-OCRv5_mobile_rec 모델 객체 (PaddleX)
        vlm_client: Optional. OpenAI 클라이언트
        config: 파이프라인 설정 (None이면 기본값 사용)
        
    Returns:
        StudentIdExtractionResult
    """
    if config is None:
        config = Config()
    
    meta = {
        "stage": None,
        "reason": None,
        "ocr_conf": None,
        "used_vlm": False,
        "vlm_error_type": None,
        "ocr_candidates": [],  # 각 bbox에서 추출된 후보들
        "matched_from_label": None  # 어느 라벨에서 매칭되었는지
    }
    
    # =========================================================================
    # Step 1) Layout detect - 모든 bbox 탐지
    # =========================================================================
    all_boxes = detect_all_bboxes(original_image, layout_model)
    
    if not all_boxes:
        meta["stage"] = "layout"
        meta["reason"] = "no_boxes_detected"
        return StudentIdExtractionResult(
            original_image=original_image,
            header_image=None,
            student_id=None,
            meta=meta
        )
    
    # Table과 Non-table 분리
    table_boxes = get_table_boxes(all_boxes)
    non_table_boxes = get_non_table_boxes(all_boxes)
    
    # Header 이미지 생성 (사용자 fallback용)
    header_image = None
    if table_boxes:
        # 가장 큰 Table bbox 기준
        largest_table = max(table_boxes, key=lambda b: b.bbox.area)
        header_image = make_header_image(original_image, largest_table.bbox, config.margin_px)
    
    if not non_table_boxes:
        meta["stage"] = "layout"
        meta["reason"] = "no_non_table_boxes"
        return StudentIdExtractionResult(
            original_image=original_image,
            header_image=header_image,
            student_id=None,
            meta=meta
        )
    
    # =========================================================================
    # Step 2 & 3) 각 non-table bbox를 crop하여 PP-OCRv5 수행
    # =========================================================================
    best_match = None
    best_conf = 0.0
    
    for layout_box in non_table_boxes:
        # bbox crop
        cropped = crop_bbox(original_image, layout_box.bbox, padding=2)
        if cropped is None:
            continue
        
        # PP-OCRv5 수행
        raw_text, conf = ppocr_extract(cropped, ocr_model)
        
        # 후보 정규화
        candidate = normalize_candidate(raw_text)
        
        # 결과 기록 (디버깅용)
        meta["ocr_candidates"].append({
            "label": layout_box.label,
            "raw_text": raw_text,
            "normalized": candidate,
            "conf": conf,
            "bbox": [layout_box.bbox.x1, layout_box.bbox.y1, layout_box.bbox.x2, layout_box.bbox.y2]
        })
        
        # 유효성 검사
        if not candidate:
            continue
        
        regex_ok = is_valid_format(candidate)
        length_ok = (len(candidate) == 8)
        
        if not regex_ok or not length_ok:
            continue
        
        # Confidence threshold 체크
        if conf < config.conf_threshold:
            continue
        
        # student_id_list 매칭
        matched = match_to_student_list(
            candidate,
            student_id_list,
            config.allow_edit_distance_1
        )
        
        if matched and conf > best_conf:
            best_match = matched
            best_conf = conf
            meta["matched_from_label"] = layout_box.label
            meta["ocr_conf"] = conf
    
    # =========================================================================
    # Step 4) OCR 성공 시 반환
    # =========================================================================
    if best_match:
        meta["stage"] = "ocr"
        meta["reason"] = "success"
        return StudentIdExtractionResult(
            original_image=original_image,
            header_image=header_image,
            student_id=best_match,
            meta=meta
        )
    
    # =========================================================================
    # Step 5) OCR 실패 시 VLM fallback (header 이미지 사용)
    # =========================================================================
    if vlm_client is not None and header_image is not None:
        meta["used_vlm"] = True
        
        vlm_result = vlm_extract_student_id(
            header_image,
            vlm_client,
            config.vlm_timeout_s
        )
        
        if vlm_result and vlm_result.get("text"):
            vlm_candidate = normalize_candidate(vlm_result["text"])
            vlm_regex_ok = is_valid_format(vlm_candidate) if vlm_candidate else False
            vlm_length_ok = (len(vlm_candidate) == 8) if vlm_candidate else False
            
            if vlm_regex_ok and vlm_length_ok:
                matched = match_to_student_list(
                    vlm_candidate,
                    student_id_list,
                    config.allow_edit_distance_1
                )
                
                if matched:
                    meta["stage"] = "vlm"
                    meta["reason"] = "success"
                    meta["ocr_conf"] = vlm_result.get("confidence", 0.8)
                    return StudentIdExtractionResult(
                        original_image=original_image,
                        header_image=header_image,
                        student_id=matched,
                        meta=meta
                    )
                else:
                    meta["stage"] = "match"
                    meta["reason"] = "vlm_no_match_or_ambiguous"
            else:
                meta["vlm_error_type"] = "invalid_format"
        else:
            meta["vlm_error_type"] = "no_result"
    else:
        if vlm_client is None:
            meta["vlm_error_type"] = "vlm_unavailable"
    
    # =========================================================================
    # 실패: 사용자 fallback으로 이어질 수 있도록 header_image 반환
    # =========================================================================
    if meta["stage"] is None:
        meta["stage"] = "ocr"
        meta["reason"] = "no_valid_student_id_found"
    
    return StudentIdExtractionResult(
        original_image=original_image,
        header_image=header_image,
        student_id=None,
        meta=meta
    )

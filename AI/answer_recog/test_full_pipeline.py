
import os
import sys
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    import paddlex
except ImportError:
    print("Warning: paddlex not installed")

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, AI_DIR)
sys.path.insert(0, os.path.join(AI_DIR, 'id_recog'))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Modules Import
from answer_recog.find_answer_section import find_answer_section, AnswerSectionResult
from answer_recog.row_segmentation import segment_rows, segment_text_lines
from answer_recog.answer_extraction import extract_text_from_row, refined_answer, get_ocr_model
from answer_recog.roi_extraction import extract_roi_from_row

def load_layout_model():
    print("Loading Layout Model...")
    return paddlex.create_model("PP-DocLayout_plus-L")


@dataclass
class QuestionInfo:
    number: int
    sub_number: int  # 0 if no sub-question
    scoring_type: str
    has_sub_questions: bool = False
    sub_count: int = 0

def load_answer_structure(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('answers', [])  # 'questions' -> 'answers'

def parse_questions_flat(questions_data: List[Dict[str, Any]]) -> List[QuestionInfo]:
    """
    Flat한 답변 리스트를 문제(Question) 단위로 그룹핑하여 변환합니다.
    """
    grouped = {}
    
    for q in questions_data:
        q_num = q.get('questionNumber')
        sub_num = q.get('subQuestionNumber', 0)
        
        if q_num not in grouped:
            grouped[q_num] = {
                "sub_questions": [],
                "scoring_type": q.get('answerType', 'objective') # scoringType -> answerType (JSON 키 확인 필요)
            }
        
        if sub_num > 0:
            grouped[q_num]["sub_questions"].append(sub_num)
            
    # 정렬된 순서로 변환
    q_infos = []
    sorted_q_nums = sorted(grouped.keys())
    
    for q_num in sorted_q_nums:
        group = grouped[q_num]
        sub_qs = group["sub_questions"]
        sub_qs.sort()
        
        info = QuestionInfo(
            number=q_num,
            sub_number=0,
            scoring_type=group["scoring_type"],
            has_sub_questions=len(sub_qs) > 0,
            sub_count=len(sub_qs)
        )
        q_infos.append(info)
        
    return q_infos

def main():
    print("======================================================================")
    print("답안 인식 파이프라인 테스트 (Improved Logic)")
    print("======================================================================")
    
    # 1. 설정
    image_path = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/PR 2024 Final_cleaned/PR 2024 Final - 10.jpg"
    json_path = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/answer_recog/answer_structure.json"
    output_dir = "test_output/pipeline_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 데이터 로드
    print("[1/6] 이미지 및 구조 로드...")
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    questions_data = load_answer_structure(json_path)
    q_infos = parse_questions_flat(questions_data)
    print(f"  ✓ {len(q_infos)}개 문제(Parent Question) 정보 로드")
    
    # 3. 모델 로드
    print("[2/6] 모델 로드...")
    layout_model = load_layout_model()
    ocr_model = get_ocr_model() # 미리 로드
    print("  ✓ 모델 로드 완료")
    
    # 4. Answer Section 추출
    print("[3/6] Answer Section 추출...")
    sec_result = find_answer_section(image, layout_model, enable_deskew=True)
    
    if not sec_result.success or sec_result.answer_section_image is None:
        print("  ❌ Answer Section 추출 실패")
        return
        
    answer_image = sec_result.answer_section_image
    cv2.imwrite(os.path.join(output_dir, "01_answer_section.jpg"), answer_image)
    print(f"  ✓ Answer Section 추출 완료: {answer_image.shape}")
    
    # 5. Row Segmentation
    print("[4/6] Row Segmentation (가로선 기준)...")
    # min_line_length_ratio=0.6 (Threshold 60%)
    row_result = segment_rows(
        answer_image, 
        min_row_height=30,
        use_morphological=True,
        min_line_length_ratio=0.6 
    )
    
    if not row_result.success:
        print("  ❌ Row Segmentation 실패")
        return
        
    rows = row_result.rows
    print(f"  ✓ {len(rows)}개 Row 분할 완료")
    
    # **중요**: 첫 번째 Row는 Header이므로 제거
    if len(rows) > 0:
        header_row = rows.pop(0)
        cv2.imwrite(os.path.join(output_dir, "00_header_row.jpg"), header_row.row_image)
        print("  ✓ Header Row 제거 완료 (1개)")
        
    # Row 개수 검증
    if len(rows) != len(q_infos):
        print(f"  ⚠️ Row 개수 불일치: 예상 {len(q_infos)}개, 실제 {len(rows)}개")
        # 개수가 안 맞으면 매칭이 꼬일 수 있음. 일단 진행하되 경고.
    
    # 6. Matching & Extraction
    print("[5/6] Matching, ROI Extraction & OCR...")
    final_results = []
    
    for i, q_info in enumerate(q_infos):
        if i >= len(rows):
            print(f"  ❌ Row 부족: Q{q_info.number} (Skipping)")
            break
            
        row_segment = rows[i]
        row_img = row_segment.row_image
        
        # 디버그: 원본 Row 저장
        cv2.imwrite(os.path.join(output_dir, f"raw_row_q{q_info.number}.jpg"), row_img)
        
        current_rois_imgs = []
        current_sub_indices = []
        
        # Case 0: others type (e.g. Q10 표) -> Skip everything
        if q_info.scoring_type == "others":
            print(f"  Case 0: Q{q_info.number} (Type: others) -> Skipped")
            final_results.append({
                "questionNumber": q_info.number,
                "subQuestionNumber": 0,
                "recAnswer": None,
                "confidence": None,
                "skipped": True
            })
            continue

        # [Step 2] 공백 제거 (X-Projection 등으로 내용만 남김)
        cleaned_row, _ = extract_roi_from_row(row_img)
        
        # 디버그: 공백 제거된 Row 저장
        cv2.imwrite(os.path.join(output_dir, f"cleaned_row_q{q_info.number}.jpg"), cleaned_row)

        # Case 1: 꼬리문제 존재 -> [Step 3] 최종 row 구분 (Sub-segmentation)
        if q_info.has_sub_questions:
            print(f"  Case 1: Q{q_info.number} (Sub-questions: {q_info.sub_count})")
            
            # 공백 제거된 이미지에서 텍스트 라인 분할
            sub_lines = segment_text_lines(
                cleaned_row, 
                expected_count=q_info.sub_count,
                min_height=5 
            )
            
            print(f"    -> Found {len(sub_lines)} sub-lines")
            
            for sub_idx in range(q_info.sub_count):
                if sub_idx < len(sub_lines):
                    sub_img = sub_lines[sub_idx]
                else:
                    # 부족하면 빈 이미지
                    print(f"    ⚠️ Sub-row missing for Q{q_info.number}-{sub_idx+1}")
                    sub_img = np.zeros((10, 10, 3), dtype=np.uint8)
                    
                # 이미 ROI 추출(공백제거)된 상태에서 잘랐으므로 그대로 사용
                # 필요하다면 한번 더 extract_roi_from_row 할 수 있음 (Sub-row 내부의 미세 공백 제거)
                # 여기서는 바로 사용
                current_rois_imgs.append(sub_img)
                current_sub_indices.append(sub_idx + 1) # 1-base
                
                # 저장
                cv2.imwrite(os.path.join(output_dir, f"roi_q{q_info.number}_{sub_idx+1}.jpg"), sub_img)
        
        # Case 2: 단일 문제 -> cleaned_row가 곧 ROI
        else:
            current_rois_imgs.append(cleaned_row)
            current_sub_indices.append(0) # 0 for no sub-question
            
            # 저장
            cv2.imwrite(os.path.join(output_dir, f"roi_q{q_info.number}.jpg"), cleaned_row)
            
        # Recognition (OCR)
        for roi_idx, img in enumerate(current_rois_imgs):
            sub_num = current_sub_indices[roi_idx]
            
            # 이미지 체크
            if img.size == 0 or img.shape[0] < 5 or img.shape[1] < 5:
                # 너무 작으면 Skip
                final_results.append({
                    "questionNumber": q_info.number,
                    "subQuestionNumber": sub_num,
                    "recAnswer": None,
                    "confidence": 0.0,
                    "skipped": False, # 시도는 했으나 실패
                    "error": "Empty ROI"
                })
                continue

            # OCR Run
            text, conf = extract_text_from_row(img)
            ans, _ = refined_answer(text)
            
            final_results.append({
                "questionNumber": q_info.number,
                "subQuestionNumber": sub_num,
                "recAnswer": ans,
                "rawText": text,
                "confidence": conf,
                "skipped": False
            })
            
            print(f"    Q{q_info.number}-{sub_num}: '{ans}' (conf={conf:.2f})")

    # 7. 결과 저장
    result_json = {
        "examCode": "TEST_EXAM",
        "totalQuestions": len(final_results),
        "results": final_results
    }
    
    with open(os.path.join(output_dir, "final_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
        
    print(f"\n[6/6] 완료. 결과 저장됨: {os.path.join(output_dir, 'final_result.json')}")

if __name__ == "__main__":
    main()

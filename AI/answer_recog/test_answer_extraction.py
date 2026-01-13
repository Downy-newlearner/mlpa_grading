"""
test_answer_extraction.py - 통합 답안 추출 테스트

전체 파이프라인을 테스트합니다:
1. Answer Section 추출
2. Row Segmentation (Recursive)
3. OCR 및 답안 정제
4. 결과 검증
"""

import sys
import os
import cv2
import pandas as pd
from pathlib import Path
# 프로젝트 경로 추가
project_root = Path(__file__).resolve().parent.parent
# AI/id_recog 경로 추가 (schemas.py import 해결용)
sys.path.insert(0, str(project_root / "id_recog"))
# AI 경로 추가 (answer_recog 패키지 import 해결용)
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Sys path[0]: {sys.path[0]}")
print(f"Sys path[1]: {sys.path[1]}")
from answer_recog.find_answer_section import find_answer_section
from answer_recog.row_segmentation import segment_rows_recursive
from answer_recog.answer_extraction import extract_answers_from_rows, get_ocr_model


def load_layout_model():
    """PP-DocLayout_plus-L 모델 로드"""
    from paddlex import create_model
    print("Loading PP-DocLayout_plus-L model...")
    model = create_model("PP-DocLayout_plus-L")
    print("Model loaded successfully!")
    return model


def test_answer_extraction(image_path: str, layout_model):
    """단일 이미지에 대한 전체 파이프라인 수행"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print('='*60)
    
    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Failed to load image")
        return False
        
    # 2. Answer Section 추출
    print("Step 1: Extracting Answer Section...")
    sec_result = find_answer_section(image, layout_model)
    if not sec_result.success:
        print("FAILED: Answer section extraction")
        return False
        
    # 3. Row Segmentation
    print("Step 2: Segmenting Rows...")
    row_result = segment_rows_recursive(
        sec_result.answer_section_image,
        min_row_height=30,
        max_row_height=300,
        text_line_height=40
    )
    
    if not row_result.success:
        print("FAILED: Row segmentation")
        return False
        
    print(f"  > Found {len(row_result.rows)} rows")
    
    # 4. OCR & Answer Extraction
    print("Step 3: Extracting Answers (OCR)...")
    try:
        answers = extract_answers_from_rows(row_result.rows)
    except Exception as e:
        print(f"FAILED: OCR extraction - {str(e)}")
        # OCR 모델 로딩 실패 시 pass (로컬 환경 문제 등)
        import traceback
        traceback.print_exc()
        return False
        
    # 결과 출력
    print("\nExtraction Results:")
    print("-" * 50)
    print(f"{'Row':<5} | {'Raw Text':<20} | {'Answer':<10} | {'Conf':<6}")
    print("-" * 50)
    
    valid_count = 0
    for ans in answers:
        print(f"{ans.row_number:<5} | {ans.recognized_text[:20]:<20} | {ans.answser:<10} | {ans.confidence:.2f}")
        if ans.is_valid:
            valid_count += 1
            
    print("-" * 50)
    print(f"Valid Answers: {valid_count}/{len(answers)}")
    
    return True


def main():
    # 모델 로드
    layout_model = load_layout_model()
    
    # OCR 모델 미리 로드 (시간 측정 제외)
    print("Loading OCR model...")
    get_ocr_model()
    
    test_images = [
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned/SaS 2017 Final - 5.jpg",
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/AI 2023 Mid_cleaned/AI 2023 Mid - 5.jpg",
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/PR 2023 Final_cleaned/PR 2023 Final - 5.jpg",
    ]
    
    for img_path in test_images:
        test_answer_extraction(img_path, layout_model)


if __name__ == "__main__":
    main()

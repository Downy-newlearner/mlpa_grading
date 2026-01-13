"""
test_row_segmentation.py - Row Segmentation 테스트

테스트 흐름:
1. Answer section 추출 (find_answer_section 사용)
2. Row segmentation 수행
3. 결과 시각화 및 저장
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "id_recog"))

from answer_recog.find_answer_section import find_answer_section
from answer_recog.row_segmentation import (
    segment_rows,
    visualize_row_segmentation,
    RowSegmentationResult
)


def load_layout_model():
    """PP-DocLayout_plus-L 모델 로드"""
    from paddlex import create_model
    print("Loading PP-DocLayout_plus-L model...")
    model = create_model("PP-DocLayout_plus-L")
    print("Model loaded successfully!")
    return model


def test_row_segmentation(image_path: str, layout_model, output_dir: str):
    """Row segmentation 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print('='*60)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return False
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")
    
    # Step 1: Answer section 추출
    print("\n[Step 1] Extracting answer section...")
    answer_result = find_answer_section(image, layout_model)
    
    if not answer_result.success:
        print(f"ERROR: Failed to extract answer section")
        return False
    
    answer_section = answer_result.answer_section_image
    print(f"Answer section size: {answer_section.shape[1]} x {answer_section.shape[0]}")
    
    # Step 2: Row segmentation
    print("\n[Step 2] Segmenting rows...")
    from answer_recog.row_segmentation import segment_rows_recursive
    
    row_result = segment_rows_recursive(
        answer_section,
        min_row_height=30,
        max_row_height=300,
        text_line_height=25
    )
    
    print(f"Result: {'SUCCESS' if row_result.success else 'FAILED'}")
    print(f"Number of rows: {len(row_result.rows)}")
    print(f"Median row height: {row_result.median_row_height:.1f}" if row_result.median_row_height else "N/A")
    
    if row_result.meta:
        print(f"Meta: {row_result.meta}")
    
    # Row 정보 출력
    print("\nRow details:")
    for row in row_result.rows:
        print(f"  Row {row.row_number}: y={row.y_start}~{row.y_end}, height={row.height}")
    
    # 결과 저장
    base_name = Path(image_path).stem
    
    # 시각화 저장
    viz_path = os.path.join(output_dir, f"{base_name}_row_segmentation.png")
    visualize_row_segmentation(row_result, save_path=viz_path)
    print(f"\nSaved: {viz_path}")
    
    # 각 row 이미지 저장
    row_dir = os.path.join(output_dir, f"{base_name}_rows")
    os.makedirs(row_dir, exist_ok=True)
    
    for row in row_result.rows:
        row_path = os.path.join(row_dir, f"row_{row.row_number:02d}.jpg")
        cv2.imwrite(row_path, row.row_image)
    print(f"Saved {len(row_result.rows)} row images to: {row_dir}")
    
    return True


def main():
    """메인 테스트 함수"""
    # 테스트 이미지 리스트
    test_images = [
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned/SaS 2017 Final - 5.jpg",
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/AI 2023 Mid_cleaned/AI 2023 Mid - 5.jpg",
        "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/PR 2023 Final_cleaned/PR 2023 Final - 5.jpg",
    ]
    
    # 출력 디렉토리 생성
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 모델 로드
    layout_model = load_layout_model()
    
    # 각 이미지 테스트
    results = {}
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"WARNING: Image not found: {image_path}")
            results[image_path] = False
            continue
        
        success = test_row_segmentation(image_path, layout_model, str(output_dir))
        results[image_path] = success
    
    # 최종 결과 요약
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for image_path, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {os.path.basename(image_path)}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

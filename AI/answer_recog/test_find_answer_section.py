"""
test_find_answer_section.py - Answer 섹션 찾기 테스트

테스트 이미지:
- SaS 2017 Final - 5.jpg
- AI 2023 Mid - 5.jpg
- PR 2023 Final - 5.jpg
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

from answer_recog.find_answer_section import (
    find_answer_section,
    visualize_projection_profile,
    compute_x_projection_profile,
    find_column_separators
)


def load_layout_model():
    """PP-DocLayout_plus-L 모델 로드"""
    from paddlex import create_model
    print("Loading PP-DocLayout_plus-L model...")
    model = create_model("PP-DocLayout_plus-L")
    print("Model loaded successfully!")
    return model


def test_single_image(image_path: str, layout_model, output_dir: str):
    """단일 이미지 테스트"""
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print('='*60)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return False
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")
    
    # Answer 섹션 찾기
    result = find_answer_section(image, layout_model)
    
    # 결과 출력
    print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"Meta: {result.meta}")
    
    if result.success:
        print(f"\nAnswer Section Size: {result.answer_section_image.shape[1]} x {result.answer_section_image.shape[0]}")
        print(f"Answer Column X Start: {result.answer_column_x_start}")
        if result.rotation_angle is not None and abs(result.rotation_angle) > 0.01:
            print(f"Rotation Angle: {result.rotation_angle:.2f}° (deskew applied)")
        
        # 결과 이미지 저장
        base_name = Path(image_path).stem
        
        # Answer section 저장
        answer_section_path = os.path.join(output_dir, f"{base_name}_answer_section.jpg")
        cv2.imwrite(answer_section_path, result.answer_section_image)
        print(f"Saved: {answer_section_path}")
        
        # Table 저장 (Answer column 위치 표시)
        table_with_line = result.table_image.copy()
        h = table_with_line.shape[0]
        cv2.line(table_with_line, 
                 (result.answer_column_x_start, 0), 
                 (result.answer_column_x_start, h), 
                 (0, 0, 255), 2)  # Red line
        table_path = os.path.join(output_dir, f"{base_name}_table_marked.jpg")
        cv2.imwrite(table_path, table_with_line)
        print(f"Saved: {table_path}")
        
        # Projection profile 시각화
        profile_path = os.path.join(output_dir, f"{base_name}_projection_profile.png")
        visualize_projection_profile(result.table_image, save_path=profile_path, answer_column_x=result.answer_column_x_start)
        print(f"Saved: {profile_path}")
        
        return True
    else:
        print(f"Reason: {result.meta.get('reason', 'unknown')}")
        return False


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
        
        success = test_single_image(image_path, layout_model, str(output_dir))
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

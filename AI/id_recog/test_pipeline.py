"""
test_pipeline.py - 파이프라인 통합 테스트 스크립트

사용법:
    python test_pipeline.py [이미지_경로]
"""

import os
import sys

# 환경 변수 설정
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
from PIL import Image

# =============================================================================
# 테스트 설정
# =============================================================================
# 테스트할 이미지 경로 (인자로 받거나 기본값 사용)
if len(sys.argv) > 1:
    TEST_IMAGE_PATH = sys.argv[1]
else:
    # 기본 테스트 이미지 경로
    TEST_IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/AI 2023 Mid_cleaned/AI 2023 Mid - 11.jpg"

# 테스트용 학번 리스트 (실제 사용 시 출석부에서 로드)
TEST_STUDENT_IDS = [
    "20231001", "20231002", "20231003", "20231004", "20231005",
    "20231006", "20231007", "20231008", "20231009", "20231010",
    "32217098",  # 테스트 이미지의 실제 학번
]

# =============================================================================
# 모델 로드
# =============================================================================
print("=" * 60)
print("Student ID Extraction Pipeline 테스트")
print("=" * 60)

print("\n[1/2] 모델 로딩 중...")

# Layout 모델
print("  - PP-DocLayout_plus-L 로딩...")
from paddlex import create_model
layout_model = create_model(model_name="PP-DocLayout_plus-L")
print("    ✓ Layout 모델 로드 완료")

# PP-OCRv5 OCR 모델
print("  - PP-OCRv5_mobile_rec 로딩...")
ocr_model = create_model(model_name="PP-OCRv5_mobile_rec")
print("    ✓ PP-OCRv5 OCR 모델 로드 완료")

print("\n모델 로딩 완료!")

# =============================================================================
# 테스트 실행
# =============================================================================
print("\n[2/2] 파이프라인 테스트 실행")
print(f"  테스트 이미지: {TEST_IMAGE_PATH}")

# 이미지 로드
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"\n❌ 오류: 이미지 파일을 찾을 수 없습니다: {TEST_IMAGE_PATH}")
    sys.exit(1)

original_image = np.array(Image.open(TEST_IMAGE_PATH).convert("RGB"))
print(f"  이미지 크기: {original_image.shape}")

# 파이프라인 임포트 및 실행
from schemas import Config
from student_id_pipeline import extract_student_id

config = Config(
    conf_threshold=0.6,
    margin_px=2,
    allow_edit_distance_1=True
)

print("\n  파이프라인 실행 중...")
result = extract_student_id(
    original_image=original_image,
    student_id_list=TEST_STUDENT_IDS,
    layout_model=layout_model,
    ocr_model=ocr_model,
    vlm_client=None,  # VLM 테스트 하려면 OpenAI client 추가
    config=config
)

# =============================================================================
# 결과 출력
# =============================================================================
print("\n" + "=" * 60)
print("테스트 결과")
print("=" * 60)

print(f"\n✓ student_id: {result.student_id}")
print(f"✓ header_image: {'생성됨' if result.header_image is not None else 'None'}")

print(f"\n[Meta 정보]")
print(f"  - stage: {result.meta.get('stage')}")
print(f"  - reason: {result.meta.get('reason')}")
print(f"  - ocr_conf: {result.meta.get('ocr_conf')}")
print(f"  - matched_from_label: {result.meta.get('matched_from_label')}")
print(f"  - used_vlm: {result.meta.get('used_vlm')}")

# 각 bbox에서 추출된 OCR 후보들 출력
ocr_candidates = result.meta.get('ocr_candidates', [])
if ocr_candidates:
    print(f"\n[OCR 후보들] (총 {len(ocr_candidates)}개)")
    for i, cand in enumerate(ocr_candidates):
        print(f"  [{i+1}] label=\"{cand['label']}\" conf={cand['conf']:.2f}")
        print(f"       raw: \"{cand['raw_text'][:50]}...\"" if len(cand.get('raw_text', '')) > 50 else f"       raw: \"{cand['raw_text']}\"")
        print(f"       normalized: \"{cand['normalized']}\"")

# 헤더 이미지 저장 (확인용)
if result.header_image is not None:
    output_dir = "./test_output/"
    os.makedirs(output_dir, exist_ok=True)
    header_path = os.path.join(output_dir, "header_image.jpg")
    Image.fromarray(result.header_image).save(header_path)
    print(f"\n✓ 헤더 이미지 저장됨: {header_path}")

print("\n" + "=" * 60)
if result.student_id:
    print(f"🎉 성공! 추출된 학번: {result.student_id}")
else:
    print("⚠️  학번 추출 실패 - meta 정보를 확인하세요")
print("=" * 60)


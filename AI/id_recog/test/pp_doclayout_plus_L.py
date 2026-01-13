"""
PP-DocLayout_plus-L 모델을 사용한 레이아웃 탐지 스크립트

- create_pipeline 대신 create_model 사용 (레이아웃 탐지 전용, 더 가볍고 빠름)
- PP-DocLayout_plus-L: 23개 클래스 지원 (헤더, 푸터, 수식, 도장 등 포함)
- 디렉토리 내 모든 이미지를 일괄 처리
"""

import os
import glob

# 로그 제어 및 환경 변수 설정
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex import create_model

# =============================================================================
# 1. 모델 생성 (레이아웃 탐지 모델만 로드 - 파이프라인보다 가볍고 빠름)
# =============================================================================
model = create_model(model_name="PP-DocLayout_plus-L")

# =============================================================================
# 2. 입력 디렉토리 설정
# =============================================================================
input_dir = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/AI 2023 Mid_cleaned"
output_base_dir = "./output_doclayout_plus/"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# 출력 디렉토리 생성
os.makedirs(output_base_dir, exist_ok=True)

# 모든 이미지 파일 경로 수집
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))

# 파일명 기준으로 정렬
image_files.sort()

print(f"=" * 60)
print(f"PP-DocLayout_plus-L 레이아웃 탐지")
print(f"=" * 60)
print(f"입력 디렉토리: {input_dir}")
print(f"출력 디렉토리: {output_base_dir}")
print(f"총 {len(image_files)}개의 이미지를 처리합니다.\n")

# =============================================================================
# 3. 각 이미지에 대해 예측 수행
# =============================================================================
for idx, input_path in enumerate(image_files, 1):
    image_name = os.path.basename(input_path)
    image_basename = os.path.splitext(image_name)[0]
    
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(image_files)}] 처리 중: {image_name}")
    print(f"{'='*60}")
    
    # 예측 수행
    outputs = model.predict(input_path, batch_size=1)
    
    for res in outputs:
        # 결과 JSON 구조 확인
        result_data = res.json
        
        print(f"\n--- [Layout Detection Results] ---")
        
        # boxes 정보 출력 (레이아웃 탐지 결과)
        if 'boxes' in result_data:
            boxes = result_data['boxes']
            print(f"탐지된 레이아웃 요소 수: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                label = box.get('label', 'unknown')
                score = box.get('score', 0)
                bbox = box.get('coordinate', [])
                
                print(f"  [{i+1}] {label} (신뢰도: {score:.3f}) - Bbox: {bbox}")
        else:
            # 다른 구조일 경우 전체 키 출력
            print(f"결과 키: {result_data.keys()}")
            print(f"전체 결과: {result_data}")
        
        # 시각적으로 확인하기 위해 결과 저장 (이미지별 서브폴더 생성)
        output_subdir = os.path.join(output_base_dir, image_basename)
        os.makedirs(output_subdir, exist_ok=True)
        res.save_to_img(output_subdir)
        
        # JSON 결과도 저장
        res.save_to_json(output_subdir)

print(f"\n{'='*60}")
print(f"모든 이미지 처리 완료!")
print(f"결과 저장 위치: {output_base_dir}")
print(f"{'='*60}")

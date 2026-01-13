import os
import cv2
import glob
# 로그 제어 및 환경 변수 설정
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
import logging


from paddlex import create_pipeline

# 1. 파이프라인 생성
pipeline = create_pipeline(pipeline="PP-StructureV3")

# 2. 입력 디렉토리 설정
input_dir = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/AI 2023 Mid_cleaned"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]

# 모든 이미지 파일 경로 수집
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_dir, ext)))

# 파일명 기준으로 정렬
image_files.sort()

print(f"총 {len(image_files)}개의 이미지를 처리합니다.\n")

# 3. 각 이미지에 대해 예측 수행
for idx, input_path in enumerate(image_files, 1):
    image_name = os.path.basename(input_path)
    print(f"\n{'='*60}")
    print(f"[{idx}/{len(image_files)}] 처리 중: {image_name}")
    print(f"{'='*60}")
    
    outputs = pipeline.predict(
        input=input_path,
        use_table_recognition=False,
        use_doc_orientation_classify=False
    )

    for res in outputs:
        layout_data = res.json.get('layout', [])
        
        print(f"\n--- [OCR Results (Table Excluded)] ---")
        
        for item in layout_data:
            label = item.get('label')
            
            # 테이블(table) 타입은 제외하고 처리
            if label == 'table':
                continue
                
            bbox = item.get('bbox')
            # 이미 파이프라인 내부에서 OCR이 수행되었다면 text 필드에 값이 있습니다.
            text_content = item.get('text', '').strip()
            
            print(f"[{label}] {text_content} (Bbox: {bbox})")

        # 시각적으로 확인하기 위해 결과 저장 (이미지별 서브폴더 생성)
        output_subdir = f"./output_debug/{os.path.splitext(image_name)[0]}/"
        os.makedirs(output_subdir, exist_ok=True)
        res.save_to_img(output_subdir)

        print(res.json.keys())

print(f"\n{'='*60}")
print(f"모든 이미지 처리 완료!")
print(f"{'='*60}")
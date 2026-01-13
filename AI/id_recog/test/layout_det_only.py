import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex import create_pipeline

# 1. 'PP-StructureV3' 대신 'layout_parsing' 사용
# (버전에 따라 'layout_detection'일 수도 있으나, 최신 3.0은 'layout_parsing' 권장)
pipeline = create_pipeline(pipeline="layout_parsing") 

# 2. 이미지 경로
input_path = "/home/jdh251425/MLPA_auto_grading/Data/processed_data/AI 2023 Mid_cleaned/AI 2023 Mid - 11.jpg"

# 3. 예측 실행
# 레이아웃 분석만 수행하므로 다른 옵션(use_ocr 등)은 필요 없습니다.

outputs = pipeline.predict(
    input=input_path,
    use_table_recognition=True,
    use_layout_detection=True,
    use_doc_orientation_classify=False
)

# 4. 결과 확인
for res in outputs:
    # 시각화 결과 저장 (박스 쳐진 이미지)
    res.save_to_img("./output_layout/")
    
    # JSON 결과 확인
    # 'layout' 키 안에 텍스트 내용은 없고 'bbox'와 'label'만 들어있습니다.
    print(res.json) 

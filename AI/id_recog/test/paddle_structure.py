import os

import os

# 모델 소스 체크 비활성화 (가장 먼저 실행되어야 함)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex import create_pipeline

# 1. 파이프라인 생성 (PP-StructureV3)
# 처음 실행 시 관련 모델(Layout, Table, OCR)을 자동으로 다운로드합니다.
pipeline = create_pipeline(pipeline="PP-StructureV3")

# 2. 이미지 경로 설정
input_path = "/home/jdh251425/MLPA_auto_grading/Data/processed_data/AI 2023 Mid_cleaned/AI 2023 Mid - 10.jpg"

# 3. 예측 실행
# use_table_recognition: 표 내부 구조 분석 활성화
# use_doc_orientation_classify: 문서 기울기 보정 활성화
outputs = pipeline.predict(
    input=input_path,
    use_table_recognition=True,
    use_layout_detection=True,
    use_doc_orientation_classify=False
)

# 4. 결과를 텍스트 파일로 저장
output_txt_path = "inference_result_summary.txt"

with open(output_txt_path, "w", encoding="utf-8") as f:
    for i, res in enumerate(outputs):
        f.write(f"--- Result for Image {i+1} ---\n")
        
        # 전체 구조 파악을 위해 JSON 데이터를 순회합니다.
        # res.json 내의 'layout'은 페이지의 논리적 구조를 담고 있습니다.
        layout_data = res.json.get('layout', [])
        
        for idx, item in enumerate(layout_data):
            label = item.get('label', 'N/A')
            bbox = item.get('bbox', [])
            # 텍스트 내용이 있는 경우 가져오고, 없으면 공백 처리
            text_content = item.get('text', '').replace('\n', ' ')
            
            # 파일 기록 형식: [순서] 타입 | 좌표 | 내용
            line = f"[{idx}] Type: {label} | Bbox: {bbox} | Content: {text_content}\n"
            f.write(line)
            
            # 만약 타입이 Table이라면 추가 정보를 기록할 수 있습니다.
            if label == 'table':
                f.write(f"    -> Table detected at {bbox}. Check output folder for Excel/HTML results.\n")

        # 시각화 이미지 및 엑셀 파일 저장 (분석 보조용)
        res.save_to_img("./output_debug/")
        res.save_to_xlsx("./output_debug/")

print(f"추론이 완료되었습니다. 결과 요약: {output_txt_path}")
print("상세 시각화 및 엑셀 결과는 './output_debug/' 폴더를 확인하세요.")
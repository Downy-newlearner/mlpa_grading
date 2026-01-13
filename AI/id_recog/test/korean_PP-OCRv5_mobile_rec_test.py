"""
PP-OCRv5_mobile_rec 모델 테스트 스크립트

- 모델: PP-OCRv5_mobile_rec (영어/숫자 인식)
- 데이터: multi_digits_test_data
- 결과: 텍스트 파일로 저장
"""

import os
import glob
from datetime import datetime

# 환경 변수 설정
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddlex import create_model
from PIL import Image
import numpy as np

# =============================================================================
# 설정
# =============================================================================
DATA_DIR = "/home/jdh251425/MLPA_auto_grading/Data/multi_digits_test_data"
OUTPUT_DIR = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/id_recog/test/PP-OCRv5_mobile_rec_test"

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 결과 파일 경로
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.join(OUTPUT_DIR, f"ocr_results_{timestamp}.txt")

# =============================================================================
# 모델 로드
# =============================================================================
print("=" * 60)
print("PP-OCRv5_mobile_rec 모델 테스트")
print("=" * 60)

print("\n[1/2] 모델 로딩 중...")
model = create_model(model_name="PP-OCRv5_mobile_rec")
print("    ✓ 모델 로드 완료")

# =============================================================================
# 이미지 파일 수집
# =============================================================================
image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(DATA_DIR, ext)))

# 파일명 기준 정렬 (숫자 순서로)
def sort_key(path):
    filename = os.path.basename(path)
    # multiDigits_1.png -> 1
    try:
        num = int(filename.split('_')[1].split('.')[0])
        return num
    except:
        return filename

image_files.sort(key=sort_key)

print(f"\n총 {len(image_files)}개의 이미지를 처리합니다.")
print(f"  입력: {DATA_DIR}")
print(f"  출력: {result_file}")

# =============================================================================
# OCR 수행
# =============================================================================
print("\n[2/2] OCR 수행 중...")

results = []
results.append(f"PP-OCRv5_mobile_rec OCR 테스트 결과")
results.append(f"테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
results.append(f"데이터 경로: {DATA_DIR}")
results.append(f"총 이미지 수: {len(image_files)}")
results.append("=" * 80)
results.append("")

for idx, img_path in enumerate(image_files, 1):
    filename = os.path.basename(img_path)
    
    try:
        # OCR 수행
        outputs = model.predict(img_path, batch_size=1)
        
        # 결과 추출
        ocr_text = ""
        confidence = 0.0
        
        for res in outputs:
            result_data = res.json
            
            # 결과 구조 확인 및 텍스트 추출
            if 'res' in result_data:
                res_data = result_data['res']
                if isinstance(res_data, dict):
                    ocr_text = res_data.get('rec_text', res_data.get('text', ''))
                    confidence = res_data.get('rec_score', res_data.get('score', 0.0))
                elif isinstance(res_data, list) and len(res_data) > 0:
                    # 리스트인 경우 첫 번째 결과
                    first = res_data[0]
                    if isinstance(first, dict):
                        ocr_text = first.get('rec_text', first.get('text', ''))
                        confidence = first.get('rec_score', first.get('score', 0.0))
                    else:
                        ocr_text = str(first)
            elif 'rec_text' in result_data:
                ocr_text = result_data['rec_text']
                confidence = result_data.get('rec_score', 0.0)
            elif 'text' in result_data:
                ocr_text = result_data['text']
                confidence = result_data.get('score', 0.0)
            else:
                # 알 수 없는 구조면 전체 출력
                ocr_text = str(result_data)
        
        # 결과 기록
        result_line = f"[{idx:03d}] {filename:30s} | OCR: \"{ocr_text}\" | Conf: {confidence:.4f}"
        results.append(result_line)
        print(f"  {result_line}")
        
    except Exception as e:
        error_line = f"[{idx:03d}] {filename:30s} | ERROR: {str(e)}"
        results.append(error_line)
        print(f"  {error_line}")

# =============================================================================
# 결과 저장
# =============================================================================
results.append("")
results.append("=" * 80)
results.append("테스트 완료")

with open(result_file, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n{'=' * 60}")
print(f"✓ 결과 저장됨: {result_file}")
print(f"{'=' * 60}")

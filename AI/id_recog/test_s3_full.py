"""
S3 이미지 업로드 포함 테스트
"""
import requests
import os

BASE_URL = "http://localhost:8000"
ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned/SaS 2017 Final - 9.jpg"

print("=" * 70)
print("S3 이미지 + 메타데이터 업로드 테스트")
print("=" * 70)

# 1. 출석부 업로드
print("\n[Step 1] 출석부 업로드")
with open(ROSTER_PATH, 'rb') as f:
    response = requests.post(f"{BASE_URL}/upload-roster/", files={"file": (os.path.basename(ROSTER_PATH), f)})
print(f"  ✓ {response.json()['student_count']}개 학번 로드")

# 2. 학번 추출
print("\n[Step 2] 학번 추출 + S3 업로드")
with open(IMAGE_PATH, 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/extract-student-id/",
        files={"image": (os.path.basename(IMAGE_PATH), f)}
    )

result = response.json()
    
print(f"\n  📋 추출 결과:")
print(f"     성공: {result['success']}")
print(f"     학번: {result['student_id']}")
print(f"     인식 방식: {'VLM' if result['meta'].get('used_vlm') else 'OCR'}")

print(f"\n  ☁️  S3 업로드:")
print(f"     성공: {result['meta'].get('s3_uploaded', False)}")

if result['meta'].get('s3_uploaded'):
    s3_keys = result['meta'].get('s3_keys', {})
    bucket = result['meta'].get('s3_bucket')
    
    print(f"     버킷: {bucket}")
    print(f"\n     업로드된 파일:")
    for file_type, key in s3_keys.items():
        print(f"       - {file_type:15s}: s3://{bucket}/{key}")

print("\n" + "=" * 70)

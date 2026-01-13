"""
S3 업로드 포함 API 테스트
"""
import requests
import os

BASE_URL = "http://localhost:8000"
ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned/SaS 2017 Final - 9.jpg"

print("=" * 70)
print("S3 업로드 통합 테스트")
print("=" * 70)

# 1. 출석부 업로드
print("\n[Step 1] 출석부 업로드")
with open(ROSTER_PATH, 'rb') as f:
    response = requests.post(f"{BASE_URL}/upload-roster/", files={"file": (os.path.basename(ROSTER_PATH), f)})
print(f"  ✓ {response.json()['student_count']}개 학번 로드")

# 2. 학번 추출 (성공 케이스)
print("\n[Step 2] 학번 추출 + S3 업로드 테스트")
with open(IMAGE_PATH, 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/extract-student-id/",
        files={"image": (os.path.basename(IMAGE_PATH), f)}
    )

result = response.json()
print(f"\n  이미지: {os.path.basename(IMAGE_PATH)}")
print(f"  성공: {result['success']}")
print(f"  학번: {result['student_id']}")
print(f"  방식: {'VLM' if result['meta'].get('used_vlm') else 'OCR'}")

print(f"\n[S3 업로드 정보]")
print(f"  업로드 성공: {result['meta'].get('s3_uploaded', False)}")
if result['meta'].get('s3_uploaded'):
    print(f"  S3 버킷: {result['meta'].get('s3_bucket')}")
    print(f"  S3 키: {result['meta'].get('s3_key')}")
    print(f"  전체 URL: s3://{result['meta'].get('s3_bucket')}/{result['meta'].get('s3_key')}")
else:
    print(f"  업로드 실패 또는 비활성화")

print("\n" + "=" * 70)

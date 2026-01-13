"""
S3 키 형식 검증 테스트
"""
import requests
import os

BASE_URL = "http://localhost:8000"
ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned/SaS 2017 Final - 9.jpg"

print("=" * 70)
print("S3 키 형식 검증 테스트")
print("=" * 70)

# 1. 출석부 업로드
print("\n[1] 출석부 업로드")
with open(ROSTER_PATH, 'rb') as f:
    requests.post(f"{BASE_URL}/upload-roster/", files={"file": (os.path.basename(ROSTER_PATH), f)})
print("  ✓ 완료")

# 2. 학번 추출
print("\n[2] 학번 추출 + S3 업로드")
with open(IMAGE_PATH, 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/extract-student-id/",
        files={"image": (os.path.basename(IMAGE_PATH), f)}
    )

result = response.json()

print(f"\n  학번: {result['student_id']}")
print(f"  S3 업로드: {result['meta'].get('s3_uploaded', False)}")

if result['meta'].get('s3_uploaded'):
    s3_keys = result['meta'].get('s3_keys', {})
    bucket = result['meta'].get('s3_bucket')
    
    print(f"\n  업로드된 파일:")
    for img_type, key in s3_keys.items():
        print(f"    [{img_type:8s}] s3://{bucket}/{key}")
        
    # 키 형식 검증
    print(f"\n  ✓ 키 형식 검증:")
    for img_type, key in s3_keys.items():
        parts = key.split('/')
        print(f"    {img_type}: {parts[0]}/{parts[1]}/{parts[2]}/...")
        expected_prefix = f"{img_type}/SaS_2017_Final/{result['student_id']}"
        actual_prefix = '/'.join(parts[:3])
        if actual_prefix == expected_prefix:
            print(f"      ✓ 올바른 형식")
        else:
            print(f"      ✗ 형식 오류: 예상={expected_prefix}, 실제={actual_prefix}")

print("\n" + "=" * 70)

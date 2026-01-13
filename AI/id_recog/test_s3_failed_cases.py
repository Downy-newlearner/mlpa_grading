"""
실패 케이스 S3 업로드 테스트
"""
import requests
import os

BASE_URL = "http://localhost:8000"
IMAGE_DIR = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned"

print("=" * 70)
print("실패 케이스 S3 업로드 테스트")
print("=" * 70)

# 실패할 가능성이 높은 이미지 선택 (출석부 없이 테스트)
print("\n[1] 출석부 없이 테스트 (빈 리스트)")

# 실패 케이스: 인식 불가능한 이미지
test_images = [
    "SaS 2017 Final - 1.jpg",  # 실패 가능성 높음
    "SaS 2017 Final - 3.jpg",  # 실패 가능성 높음
]

for img_name in test_images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    print(f"\n  테스트: {img_name}")
    
    with open(img_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/extract-student-id/",
            files={"image": (img_name, f)},
            data={"student_id_list": ""}  # 빈 리스트로 의도적으로 실패 유도
        )
    
    result = response.json()
    
    print(f"    성공: {result['success']}")
    print(f"    학번: {result.get('student_id', 'N/A')}")
    print(f"    S3 업로드: {result['meta'].get('s3_uploaded', False)}")
    
    if result['meta'].get('s3_uploaded'):
        s3_keys = result['meta'].get('s3_keys', {})
        for key_type, key in s3_keys.items():
            print(f"    {key_type}: {key}")
            # unknown_id가 포함되어 있는지 확인
            if 'unknown_id' in key:
                print(f"      ✓ 실패 케이스로 올바르게 저장됨!")

print("\n" + "=" * 70)
print("테스트 완료!")
print("=" * 70)

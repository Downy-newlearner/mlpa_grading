"""
백엔드 연동 테스트 스크립트

백엔드 서버 → AI 서버 통신을 시뮬레이션합니다.
"""
import requests
import os
import time

# === 설정 ===
AI_SERVER_URL = "http://localhost:8000"  # AI 서버
BACKEND_URL = "http://localhost:3000"    # 백엔드 서버 (가정)

ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_DIR = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned"

print("=" * 70)
print("백엔드 연동 테스트")
print("=" * 70)

# === 1. AI 서버 헬스 체크 ===
print("\n[1] AI 서버 헬스 체크")
try:
    response = requests.get(f"{AI_SERVER_URL}/health", timeout=5)
    health = response.json()
    
    if health.get("layout_model") and health.get("ocr_model"):
        print("  ✓ AI 서버 정상")
        print(f"    - 모델: Layout ✓, OCR ✓, VLM {'✓' if health.get('vlm_client') else '✗'}")
        print(f"    - S3: {'✓' if health.get('s3_client') else '✗'}")
        print(f"    - 출석부: {health.get('student_count', 0)}명")
    else:
        print("  ✗ AI 서버 모델 로드 실패")
        exit(1)
except Exception as e:
    print(f"  ✗ AI 서버 연결 실패: {e}")
    print("    서버를 시작했는지 확인하세요: cd /path/to/id_recog && python app.py")
    exit(1)

# === 2. 출석부 업로드 시뮬레이션 ===
print("\n[2] 출석부 업로드 (백엔드 → AI)")
with open(ROSTER_PATH, 'rb') as f:
    response = requests.post(
        f"{AI_SERVER_URL}/upload-roster/",
        files={"file": (os.path.basename(ROSTER_PATH), f)}
    )

if response.status_code == 200:
    result = response.json()
    print(f"  ✓ 성공: {result['student_count']}명")
else:
    print(f"  ✗ 실패: {response.status_code}")

# === 3. 학번 추출 시뮬레이션 (단일) ===
print("\n[3] 학번 추출 테스트 (단일 이미지)")
test_image = os.path.join(IMAGE_DIR, "SaS 2017 Final - 9.jpg")

with open(test_image, 'rb') as f:
    start_time = time.time()
    response = requests.post(
        f"{AI_SERVER_URL}/extract-student-id/",
        files={"image": (os.path.basename(test_image), f)}
    )
    elapsed = time.time() - start_time

if response.status_code == 200:
    result = response.json()
    print(f"  ✓ 응답 시간: {elapsed:.2f}초")
    print(f"  학번: {result.get('student_id', 'N/A')}")
    print(f"  성공: {result.get('success')}")
    print(f"  인식 방식: {'VLM' if result['meta'].get('used_vlm') else 'OCR'}")
    print(f"  S3 업로드: {result['meta'].get('s3_uploaded', False)}")
    
    if result['meta'].get('s3_uploaded'):
        print(f"  S3 키:")
        for key_type, key in result['meta'].get('s3_keys', {}).items():
            print(f"    - {key_type}: {key}")
else:
    print(f"  ✗ 실패: {response.status_code}")

# === 4. 배치 처리 시뮬레이션 ===
print("\n[4] 배치 처리 테스트 (3개 이미지)")
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')][:3]
results = []

for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    with open(img_path, 'rb') as f:
        response = requests.post(
            f"{AI_SERVER_URL}/extract-student-id/",
            files={"image": (img_name, f)}
        )
    
    if response.status_code == 200:
        r = response.json()
        status = "✓" if r['success'] else "✗"
        sid = r.get('student_id', '-')
        results.append((img_name, r['success']))
        print(f"  {status} {img_name:30s} | {sid}")

success_rate = sum(1 for _, s in results if s) / len(results) * 100
print(f"\n  성공률: {success_rate:.0f}% ({sum(1 for _, s in results if s)}/{len(results)})")

# === 5. 백엔드 API 연동 테스트 (Optional) ===
print("\n[5] 백엔드 API 연동 테스트")
print("  TODO: 백엔드 서버 준비 후 테스트")
# try:
#     response = requests.get(f"{BACKEND_URL}/health", timeout=5)
#     print(f"  ✓ 백엔드 서버 응답: {response.status_code}")
# except:
#     print(f"  ✗ 백엔드 서버 미실행")

print("\n" + "=" * 70)
print("통합 테스트 완료!")
print("=" * 70)

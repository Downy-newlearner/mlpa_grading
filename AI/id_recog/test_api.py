"""
API 테스트 스크립트
"""
import requests
import os

BASE_URL = "http://localhost:8000"
ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_DIR = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned"

print("=" * 70)
print("Student ID Extraction API 테스트")
print("=" * 70)

# 1. 출석부 업로드
print("\n[Step 1] 출석부 업로드")
with open(ROSTER_PATH, 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/upload-roster/",
        files={"file": (os.path.basename(ROSTER_PATH), f)}
    )
roster_result = response.json()
print(f"  ✓ {roster_result['student_count']}개의 학번 로드됨")
print(f"  학번: {roster_result['student_ids'][:5]}...")

# 2. 이미지 샘플링 (20개)
print("\n[Step 2] 이미지 샘플링")
all_images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')], 
    key=lambda x: int(x.split(' - ')[1].split('.')[0])
)[:20]  # 10개 → 20개로 변경
print(f"  선택: {len(all_images)}개 이미지")

# 3. 테스트
print("\n[Step 3] 학번 추출 테스트")
print("-" * 70)

results = []
for img_name in all_images:
    img_path = os.path.join(IMAGE_DIR, img_name)
    with open(img_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/extract-student-id/",
            files={"image": (img_name, f)}
        )
    
    r = response.json()
    status = "✓" if r["success"] else "✗"
    sid = r["student_id"] or "-"
    conf = f"{r['meta'].get('ocr_conf', 0):.2f}" if r['meta'].get('ocr_conf') else "-"
    
    # OCR 후보 미리보기
    cands = r["meta"].get("ocr_candidates", [])
    preview = "-"
    for c in cands:
        norm = c.get("normalized")
        if norm and len(str(norm)) >= 6:
            preview = str(norm)
            break
    
    # VLM 사용 여부
    used_vlm = "VLM" if r['meta'].get('used_vlm') else "OCR"
    
    results.append({
        "image": img_name,
        "success": r["success"],
        "student_id": r["student_id"],
        "conf": r["meta"].get("ocr_conf"),
        "preview": preview,
        "method": used_vlm
    })
    print(f"  {status} {img_name:30s} | {sid:10s} | conf: {conf:6s} | {used_vlm} | OCR: {preview}")

# 4. 요약
print("\n" + "=" * 70)
print("결과 요약")
print("=" * 70)
success = sum(1 for r in results if r["success"])
fail = len(results) - success
print(f"\n  총 테스트: {len(results)}개")
print(f"  ✓ 성공: {success}개 ({success/len(results)*100:.0f}%)")
print(f"  ✗ 실패: {fail}개 ({fail/len(results)*100:.0f}%)")

if success > 0:
    confs = [r["conf"] for r in results if r["conf"]]
    if confs:
        print(f"  평균 신뢰도: {sum(confs)/len(confs):.2f}")

print("\n" + "=" * 70)

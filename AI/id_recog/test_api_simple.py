import requests
import os
import time

BASE_URL = "http://localhost:8000"
ROSTER_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS_2017.xlsx"
IMAGE_DIR = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/processed_data/SaS 2017 Final_cleaned"

# 1. Upload Roster
with open(ROSTER_PATH, 'rb') as f:
    requests.post(f"{BASE_URL}/upload-roster/", files={"file": (os.path.basename(ROSTER_PATH), f)})

# 2. Test 3 images
all_images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')], 
    key=lambda x: int(x.split(' - ')[1].split('.')[0])
)[:3]

print(f"{'Image':30} | {'Status'} | {'ID'} | {'Method'}")
print("-" * 60)

for img in all_images:
    with open(os.path.join(IMAGE_DIR, img), 'rb') as f:
        start = time.time()
        res = requests.post(f"{BASE_URL}/extract-student-id/", files={"image": (img, f)}).json()
        duration = time.time() - start
        
    status = "SUCCESS" if res['success'] else "FAIL"
    sid = res.get('student_id', '-')
    
    # Determine method
    meta = res.get('meta', {})
    method = "OCR"
    if meta.get('used_vlm'):
        method = "VLM"
    
    sid_str = sid if sid else "-"
    print(f"{img:30} | {status:7} | {sid_str:10} | {method} ({duration:.1f}s)")

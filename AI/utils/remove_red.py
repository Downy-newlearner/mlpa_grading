import cv2
import numpy as np
import os

# Root 경로 설정
RAW_DATA_ROOT = '/home/jdh251425/MLPA_auto_grading/Data/raw_data'
PROCESSED_DATA_ROOT = '/home/jdh251425/MLPA_auto_grading/Data/processed_data'

# 출력 루트 디렉토리 생성
if not os.path.exists(PROCESSED_DATA_ROOT):
    os.makedirs(PROCESSED_DATA_ROOT)

# 빨간색 범위 정의 (HSV)
# 범위 1: 0~10 (붉은색 ~ 주황빛)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])

# 범위 2: 170~180 (자주빛 ~ 붉은색)
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# 이미지 파일 확장자
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# raw_data 내의 모든 하위 디렉토리 탐색
subdirs = sorted([d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))])

print(f"Found {len(subdirs)} directories to process: {subdirs}")

for subdir in subdirs:
    input_dir = os.path.join(RAW_DATA_ROOT, subdir)
    
    # 출력 폴더명 규칙: 원본폴더명_cleaned
    output_dirname = f"{subdir}_cleaned"
    output_dir = os.path.join(PROCESSED_DATA_ROOT, output_dirname)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXTENSIONS)]
    total_files = len(files)
    
    print(f"\nProcessing '{subdir}' -> '{output_dirname}' ({total_files} images)...")
    
    processed_count = 0
    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 이미지 로드
        img = cv2.imread(input_path)
        if img is None:
            print(f"  [Warning] Failed to load: {filename}")
            continue
        
        # BGR -> HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 마스크 생성
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # 빨간색 영역을 흰색으로 변경
        img[mask > 0] = [255, 255, 255]
        
        # 결과 저장
        cv2.imwrite(output_path, img)
        processed_count += 1
        
        if processed_count % 50 == 0:
            print(f"  - Processed {processed_count}/{total_files}...")

    print(f"  Completed '{subdir}': {processed_count} images.")

print("\nAll directories processed successfully.")


import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks

def visualize_q11_projection(image_path: str, output_path: str):
    print(f"이미지 로드: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("이미지 로드 실패")
        return
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    h, w = gray.shape[:2]
    
    # 상하 5px Crop (Margin 적용)
    margin = 5
    if h > 2 * margin:
        gray = gray[margin:h-margin, :]
        print(f"이미지 Crop (Margin {margin}px): {h} -> {gray.shape[0]}")
        h = gray.shape[0] # 높이 업데이트
    
    # 1. Y-Projection 계산 (segment_text_lines 로직과 동일하게)
    # 배경 흰색 가정 -> 반전
    # Adaptive Threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 10
    )
    
    y_profile = np.sum(binary, axis=1) / 255
    
    # 2. Peaks (Lines) & Valleys (Spaces) 찾기
    # segment_text_lines에서 사용한 파라미터: min_distance=10, min_height_ratio=0.1
    # 여기서는 scipy find_peaks로 유사하게 재현하여 시각화
    
    max_val = np.max(y_profile) if np.max(y_profile) > 0 else 1
    # Peak (텍스트 라인)
    peaks, _ = scipy_find_peaks(y_profile, height=max_val * 0.1, distance=10)
    
    # Valley (공백) -> 분할 지점
    # Min depth ratio 0.95 적용 시뮬레이션
    valleys = []
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # 원본 이미지
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    axes[0].imshow(display_img)
    axes[0].set_title(f"Q11 Row Image ({w}x{h})")
    
    # Y-Projection
    y_pos = np.arange(len(y_profile))
    axes[1].plot(y_profile, y_pos, 'b-', label='Projection')
    # Y축 반전 (이미지 좌표계와 일치)
    axes[1].set_ylim(h, 0)
    axes[1].set_title("Y-Projection Profile")
    axes[1].grid(True, alpha=0.3)
    
    # Peak 표시
    axes[1].plot(y_profile[peaks], peaks, "x", color='red', label='Peaks (Text)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"시각화 저장: {output_path}")

if __name__ == "__main__":
    IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/answer_recog/test_output/pipeline_final/raw_row_q11.jpg"
    OUTPUT_PATH = "vis_q11.png"
    
    visualize_q11_projection(IMAGE_PATH, OUTPUT_PATH)

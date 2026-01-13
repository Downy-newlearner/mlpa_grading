
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AI_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, AI_DIR)
sys.path.insert(0, os.path.join(AI_DIR, 'id_recog'))

def visualize_y_projection_final(image_path: str, output_path: str):
    """
    최종 로직: Pure Binary Projection with Threshold 0.4
    """
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
    
    # Adaptive Threshold (Block size 31)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # Pure Projection
    y_pure = np.sum(binary, axis=1) / 255 / w
    
    # Separators 찾기 (Simulation)
    threshold = 0.4
    separators = []
    in_line = False
    line_start = 0
    min_row_height = 30
    
    for y in range(h):
        val = y_pure[y]
        if val >= threshold:
            if not in_line:
                in_line = True
                line_start = y
        else:
            if in_line:
                in_line = False
                line_end = y
                line_center = (line_start + line_end) // 2
                if len(separators) == 0:
                    separators.append(line_center)
                else:
                    if line_center - separators[-1] >= min_row_height:
                        separators.append(line_center)
                    else:
                        separators[-1] = (separators[-1] + line_center) // 2
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    # 1. 원본 + Separator
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
    for sep in separators:
        cv2.line(display_img, (0, sep), (w, sep), (255, 0, 0), 2)
        
    axes[0].imshow(display_img)
    axes[0].set_title(f"Detected Separators ({len(separators)})")
    
    # 2. Projection Profile
    y_pos = np.arange(len(y_pure))
    axes[1].plot(y_pure, y_pos, 'b-', label='Pure Projection')
    axes[1].set_ylim(h, 0)
    axes[1].set_xlim(0, 1.1)
    axes[1].axvline(x=0.4, color='r', linestyle='--', label='Threshold 0.4')
    axes[1].set_title("Pure Binary Projection")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Separator 위치 표시
    for sep in separators:
        axes[1].axhline(y=sep, color='g', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"시각화 저장: {output_path}")

if __name__ == "__main__":
    IMAGE_PATH = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/answer_recog/test_output/pipeline_test/01_answer_section.jpg"
    OUTPUT_PATH = "y_profile_vis_final.png"
    
    visualize_y_projection_final(IMAGE_PATH, OUTPUT_PATH)

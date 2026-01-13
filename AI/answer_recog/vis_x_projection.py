
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_x_projection(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Skipping {img_path} (Not found)")
            continue
            
        print(f"Processing {img_path}...")
        row_image = cv2.imread(img_path)
        h, w = row_image.shape[:2]
        
        # roi_extraction.py의 _extract_roi_core 로직 재현
        margin_crop = 5
        
        # 0. 상하단 강제 Crop
        if h > 2 * margin_crop:
            start_y_crop = margin_crop
            end_y_crop = h - margin_crop
            working_img = row_image[start_y_crop:end_y_crop, :].copy()
        else:
            working_img = row_image.copy()
            
        # 1. 전처리 & 노이즈 제거
        if len(working_img.shape) == 3:
            gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = working_img.copy()
            
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Morph Open
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 2. X-Projection
        x_proj = np.sum(binary_clean, axis=0) / 255
        
        # Threshold Logic Visualization
        max_val = np.max(x_proj) if np.max(x_proj) > 0 else 1
        threshold_ratio = 0.03
        dynamic_threshold = max(working_img.shape[0] * threshold_ratio, max_val * 0.1)
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1) Original Row (Cropped)
        axes[0].imshow(cv2.cvtColor(working_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Cropped Row (w={w}, h={working_img.shape[0]})")
        
        # 2) Binary Clean (Morph Open Result) - 이게 중요함 (노이즈가 어떻게 남았나)
        axes[1].imshow(binary_clean, cmap='gray')
        axes[1].set_title("Binary Clean (Morph Open)")
        
        # 3) X-Projection Profile
        axes[2].plot(x_proj, label='X-Projection')
        axes[2].axhline(y=dynamic_threshold, color='r', linestyle='--', label=f'Threshold ({dynamic_threshold:.1f})')
        axes[2].set_title("X-Projection Profile & Threshold")
        axes[2].set_xlim(0, w)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, f"vis_xproj_{filename}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    base_dir = "/home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/answer_recog/test_output/pipeline_final"
    targets = ["raw_row_q1.jpg", "raw_row_q2.jpg", "raw_row_q4.jpg", "raw_row_q7.jpg", "raw_row_q8.jpg", "raw_row_q9.jpg"]
    
    img_paths = [os.path.join(base_dir, t) for t in targets]
    visualize_x_projection(img_paths, base_dir)

"""
sub_question_segmentation.py - 꼬리문제(Sub-Question) 분리 모듈

주 문제(Main Question) Row 내에서 꼬리문제를 Y축 Projection Profile을 사용하여 분리합니다.
Case 1 레이아웃(주 문제만 가로선 구분)에서 사용됩니다.

핵심 알고리즘:
1. Y축 Projection Profile 계산 (텍스트 밀도)
2. Peak 탐지 (각 sub-question 텍스트 중심)
3. Valley 탐지 (sub-question 경계)
4. expected_count 기반 Valley 선택
5. Final Segment 생성

참고: 가로선 기반 분할은 row_segmentation.py의 segment_rows()를 사용합니다.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .schemas import SubQuestionSegment


# =============================================================================
# Y축 Projection Profile 계산
# =============================================================================

def compute_y_projection_profile(image: np.ndarray) -> np.ndarray:
    """
    이미지의 Y축 Projection Profile을 계산합니다.
    각 y좌표에서 가로 방향으로 검은 픽셀(텍스트)의 합을 계산합니다.
    
    Args:
        image: 입력 이미지 (grayscale 또는 BGR)
        
    Returns:
        y축 projection profile (1D array, 길이 = 이미지 높이)
    """
    # Grayscale로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Y축 projection: 각 y좌표에서 가로 방향 합산
    y_profile = np.sum(binary, axis=1).astype(np.float32)
    
    # 너비로 정규화 (0~1 범위)
    max_val = image.shape[1] * 255
    if max_val > 0:
        y_profile = y_profile / max_val
    
    return y_profile


def smooth_profile(profile: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    1D Profile에 Gaussian Smoothing을 적용합니다.
    
    Args:
        profile: 1D array
        kernel_size: 스무딩 커널 크기 (홀수)
        
    Returns:
        스무딩된 profile
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 1D Gaussian blur 적용
    smoothed = cv2.GaussianBlur(
        profile.reshape(-1, 1), 
        (1, kernel_size), 
        0
    ).flatten()
    
    return smoothed


# =============================================================================
# Peak / Valley 탐지
# =============================================================================

def find_peaks_adaptive(
    profile: np.ndarray,
    min_distance: int = 20,
    min_height_ratio: float = 0.05
) -> List[int]:
    """
    1D Profile에서 Peak(극대점)을 찾습니다.
    
    Peak 조건:
    - 좌우 이웃보다 크거나 같음
    - 최대값의 min_height_ratio 이상
    - 이전 peak와 min_distance 이상 떨어져 있음
    
    Args:
        profile: 1D array
        min_distance: peak 간 최소 거리
        min_height_ratio: 최소 높이 비율 (max 대비)
        
    Returns:
        peak 위치(y좌표) 리스트
    """
    if len(profile) < 3:
        return []
    
    max_val = profile.max()
    min_height = max_val * min_height_ratio
    
    peaks = []
    n = len(profile)
    
    for i in range(1, n - 1):
        # Local maximum 확인
        if profile[i] >= profile[i-1] and profile[i] >= profile[i+1]:
            # 최소 높이 확인
            if profile[i] >= min_height:
                # 최소 거리 확인
                if len(peaks) == 0 or i - peaks[-1] >= min_distance:
                    peaks.append(i)
                elif profile[i] > profile[peaks[-1]]:
                    # 더 높은 peak면 교체
                    peaks[-1] = i
    
    return peaks


def find_valleys_between_peaks(
    profile: np.ndarray,
    peaks: List[int],
    min_depth_ratio: float = 0.7
) -> Tuple[List[int], List[float]]:
    """
    인접한 두 Peak 사이에서 Valley(극소점)를 찾습니다.
    
    Valley 깊이 계산:
    - depth_ratio = valley_val / min(peak_left, peak_right)
    - depth_ratio가 작을수록 깊은 valley (경계가 명확함)
    
    Args:
        profile: 1D array
        peaks: peak 위치 리스트
        min_depth_ratio: valley로 인정하기 위한 최대 depth ratio
                        (높을수록 얕은 valley도 허용)
        
    Returns:
        (valley 위치 리스트, depth ratio 리스트)
    """
    if len(peaks) < 2:
        return [], []
    
    valleys = []
    depths = []
    
    for i in range(len(peaks) - 1):
        y_start = peaks[i]
        y_end = peaks[i + 1]
        
        # 구간 [y_start, y_end]에서 최솟값 찾기
        segment = profile[y_start:y_end+1]
        min_idx = np.argmin(segment)
        valley_y = y_start + min_idx
        valley_val = profile[valley_y]
        
        # Depth ratio 계산 (두 peak 중 작은 값 대비)
        peak_min = min(profile[y_start], profile[y_end])
        
        if peak_min > 0:
            depth_ratio = valley_val / peak_min
        else:
            depth_ratio = 1.0
        
        valleys.append(valley_y)
        depths.append(depth_ratio)
    
    return valleys, depths


def find_local_minimum_near(
    profile: np.ndarray, 
    target_y: int, 
    window: int = 20
) -> int:
    """
    target_y 근처에서 local minimum을 찾습니다.
    
    Args:
        profile: 1D array
        target_y: 탐색 중심 y좌표
        window: 탐색 범위 (±window)
        
    Returns:
        local minimum 위치
    """
    start = max(0, target_y - window)
    end = min(len(profile), target_y + window + 1)
    
    if start >= end:
        return target_y
    
    segment = profile[start:end]
    min_idx = np.argmin(segment)
    
    return start + min_idx


# =============================================================================
# Valley 선택 로직 (expected_count 기반)
# =============================================================================

def select_valleys_by_count(
    valleys: List[int],
    depths: List[float],
    target_count: int,
    profile: np.ndarray
) -> List[int]:
    """
    expected_count에 맞게 Valley를 선택합니다.
    
    전략:
    1. target_count == len(valleys): 전부 사용
    2. target_count < len(valleys): depth가 가장 깊은(값이 작은) N개 선택
    3. target_count > len(valleys): 균등 분할로 보충
    
    Args:
        valleys: valley 위치 리스트
        depths: valley depth ratio 리스트
        target_count: 필요한 valley 개수 (= sub_question_count - 1)
        profile: 원본 profile (균등 분할 시 사용)
        
    Returns:
        선택된 valley 위치 리스트 (정렬됨)
    """
    if target_count <= 0:
        return []
    
    if len(valleys) == target_count:
        return sorted(valleys)
    
    if len(valleys) > target_count:
        # Depth로 정렬 후 상위 N개 선택 (depth가 작을수록 깊은 valley)
        indexed = list(zip(valleys, depths))
        indexed.sort(key=lambda x: x[1])  # depth 오름차순 (가장 깊은 것부터)
        selected = sorted([v for v, d in indexed[:target_count]])
        return selected
    
    # valleys 부족 → 균등 분할로 보충
    total_height = len(profile)
    segment_count = target_count + 1
    
    uniform_valleys = []
    step = total_height / segment_count
    
    for i in range(1, segment_count):
        target_y = int(i * step)
        # 가장 가까운 local minimum 찾기
        local_y = find_local_minimum_near(profile, target_y, window=int(step / 3))
        uniform_valleys.append(local_y)
    
    return sorted(uniform_valleys)


# =============================================================================
# 메인 함수: 꼬리문제 분리
# =============================================================================

def segment_sub_questions(
    row_image: np.ndarray,
    expected_count: int,
    min_sub_height: int = 15,
    smoothing_kernel: Optional[int] = None,
    debug: bool = False
) -> Tuple[List[SubQuestionSegment], dict]:
    """
    Row 이미지에서 꼬리문제(Sub-Question)를 분리합니다.
    
    Y축 Projection Profile을 사용하여 텍스트 영역 간의 경계를 찾고,
    expected_count에 맞게 분할합니다.
    
    Args:
        row_image: 분할할 Row 이미지
        expected_count: 예상 꼬리문제 개수
        min_sub_height: 최소 sub-question 높이 (px)
        smoothing_kernel: 스무딩 커널 크기 (None이면 자동 계산)
        debug: 디버그 정보 반환 여부
        
    Returns:
        (SubQuestionSegment 리스트, 메타데이터 dict)
    """
    h, w = row_image.shape[:2]
    
    meta = {
        "image_height": h,
        "expected_count": expected_count,
        "method": "y_projection"
    }
    
    # 1개 이하면 분할 불필요
    if expected_count <= 1:
        segment = SubQuestionSegment(
            sub_number=1,
            y_start=0,
            y_end=h,
            image=row_image.copy()
        )
        meta["single_segment"] = True
        return [segment], meta
    
    # Step 1: Y축 Projection Profile 계산
    y_profile = compute_y_projection_profile(row_image)
    
    # Step 2: Smoothing
    if smoothing_kernel is None:
        # 예상 sub 높이의 1/5 정도로 스무딩
        expected_sub_height = h / expected_count
        smoothing_kernel = max(3, int(expected_sub_height / 5))
        if smoothing_kernel % 2 == 0:
            smoothing_kernel += 1
    
    y_profile_smooth = smooth_profile(y_profile, smoothing_kernel)
    meta["smoothing_kernel"] = smoothing_kernel
    
    # Step 3: Peak 탐지
    expected_sub_height = h / expected_count
    min_distance = max(10, int(expected_sub_height / 3))
    
    peaks = find_peaks_adaptive(
        y_profile_smooth,
        min_distance=min_distance,
        min_height_ratio=0.03  # 낮은 텍스트도 탐지
    )
    
    meta["peak_count"] = len(peaks)
    meta["peaks"] = peaks
    
    # Step 4: Valley 탐지
    valleys, depths = find_valleys_between_peaks(
        y_profile_smooth,
        peaks,
        min_depth_ratio=0.95  # 얕은 valley도 허용
    )
    
    meta["raw_valley_count"] = len(valleys)
    meta["raw_valleys"] = valleys
    meta["raw_depths"] = depths
    
    # Step 5: expected_count 기반 Valley 선택
    target_valley_count = expected_count - 1
    final_valleys = select_valleys_by_count(
        valleys,
        depths,
        target_valley_count,
        y_profile_smooth
    )
    
    meta["final_valleys"] = final_valleys
    
    # Step 6: Segment 생성
    boundaries = [0] + final_valleys + [h]
    segments = []
    
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        
        # 최소 높이 검증
        if y_end - y_start < min_sub_height:
            # 이전 segment와 병합
            if len(segments) > 0:
                segments[-1].y_end = y_end
                segments[-1].image = row_image[segments[-1].y_start:y_end, :]
                continue
        
        # Valley depth 찾기
        depth = None
        if i < len(final_valleys) and i < len(depths):
            valley_idx = valleys.index(final_valleys[i]) if final_valleys[i] in valleys else -1
            if valley_idx >= 0:
                depth = depths[valley_idx]
        
        segment = SubQuestionSegment(
            sub_number=len(segments) + 1,
            y_start=y_start,
            y_end=y_end,
            image=row_image[y_start:y_end, :].copy(),
            valley_depth=depth
        )
        segments.append(segment)
    
    meta["final_segment_count"] = len(segments)
    
    # 디버그 정보 추가
    if debug:
        meta["y_profile"] = y_profile.tolist()
        meta["y_profile_smooth"] = y_profile_smooth.tolist()
    
    return segments, meta


# =============================================================================
# 시각화 함수 (디버깅용)
# =============================================================================

def visualize_sub_question_segmentation(
    row_image: np.ndarray,
    segments: List[SubQuestionSegment],
    meta: dict,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    꼬리문제 분할 결과를 시각화합니다.
    
    Args:
        row_image: 원본 Row 이미지
        segments: SubQuestionSegment 리스트
        meta: segment_sub_questions()의 메타데이터
        save_path: 저장 경로 (None이면 저장 안 함)
        
    Returns:
        시각화된 이미지
    """
    import matplotlib.pyplot as plt
    
    h, w = row_image.shape[:2]
    
    # Figure 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 이미지 + 분할 경계 표시
    if len(row_image.shape) == 3:
        display_img = cv2.cvtColor(row_image.copy(), cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(row_image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Segment 경계선 그리기
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, seg in enumerate(segments):
        color = colors[i % len(colors)]
        cv2.line(display_img, (0, seg.y_start), (w, seg.y_start), color, 2)
        cv2.line(display_img, (0, seg.y_end), (w, seg.y_end), color, 2)
        # Sub 번호 표시
        cv2.putText(display_img, f"Sub{seg.sub_number}", (5, seg.y_start + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    axes[0].imshow(display_img)
    axes[0].set_title(f"Sub-Question Segmentation ({len(segments)} segments)")
    axes[0].axis('off')
    
    # Y축 Projection Profile 시각화
    if "y_profile" in meta or "y_profile_smooth" in meta:
        y_positions = np.arange(h)
        
        if "y_profile_smooth" in meta:
            profile = np.array(meta["y_profile_smooth"])
            axes[1].plot(profile, y_positions, color='blue', label='Smoothed Profile')
        
        axes[1].set_ylim(h, 0)  # Y축 반전
        axes[1].set_xlabel("Projection value (normalized)")
        axes[1].set_ylabel("Y position")
        axes[1].set_title("Y-axis Projection Profile")
        axes[1].grid(True, alpha=0.3)
        
        # Peak 표시
        if "peaks" in meta:
            for peak in meta["peaks"]:
                if peak < len(profile):
                    axes[1].axhline(y=peak, color='green', linestyle='--', alpha=0.5)
                    axes[1].plot(profile[peak], peak, 'go', markersize=6)
        
        # Valley 표시
        if "final_valleys" in meta:
            for valley in meta["final_valleys"]:
                axes[1].axhline(y=valley, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No profile data available", 
                    ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return cv2.imread(save_path)
    else:
        fig.canvas.draw()
        result_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return result_img

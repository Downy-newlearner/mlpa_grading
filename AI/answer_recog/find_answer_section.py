"""
find_answer_section.py - Answer 섹션 찾기 모듈

원본 이미지에서 PP-DocLayout_plus-L을 사용하여 Table을 찾고,
X축 projection profile을 통해 Answer 컬럼을 추출합니다.

흐름:
1. PP-DocLayout_plus-L로 table bbox 탐지
2. 가장 넓은 면적의 table bbox 선택
3. table crop 이미지에서 x축 projection profile로 column separator 찾기
4. 마지막 column (Answer column)을 crop하여 반환
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import sys
import os

# id_recog 모듈에서 공통 컴포넌트 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from id_recog.schemas import BBox
from id_recog.layout import detect_all_bboxes, get_table_boxes, crop_bbox, LayoutBox


@dataclass
class AnswerSectionResult:
    """Answer 섹션 추출 결과"""
    success: bool
    answer_section_image: Optional[np.ndarray] = None  # Answer 섹션 이미지 (crop)
    table_image: Optional[np.ndarray] = None           # 전체 테이블 이미지
    table_bbox: Optional[BBox] = None                  # 테이블 bbox (원본 이미지 기준)
    answer_column_x_start: Optional[int] = None        # Answer 컬럼 시작 x좌표 (table crop 기준)
    rotation_angle: Optional[float] = None             # 적용된 회전 각도 (도)
    meta: dict = field(default_factory=dict)           # 디버깅/로그용 메타정보


def detect_skew_angle(image: np.ndarray, max_angle: float = 5.0) -> float:
    """
    이미지의 기울기 각도를 탐지합니다.
    Hough Line Transform으로 긴 가로선들을 탐지하고, 그 선들의 평균 각도를 계산합니다.
    
    Args:
        image: 입력 이미지 (BGR 또는 grayscale)
        max_angle: 탐지할 최대 각도 (이보다 큰 각도는 무시)
        
    Returns:
        기울기 각도 (도, 시계 방향 양수). 탐지 실패 시 0.0
    """
    # Grayscale 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection (Canny)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough Line Transform (Probabilistic)
    # minLineLength: 이미지 너비의 30% 이상인 선만 탐지
    min_line_length = int(gray.shape[1] * 0.3)
    max_line_gap = 20
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # 각 선의 각도 계산 (수평선 기준)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 선의 길이
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # 기울기 각도 계산 (라디안 -> 도)
        if x2 - x1 == 0:
            continue  # 수직선은 무시
        
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        
        # 거의 수평인 선만 고려 (|angle| < max_angle)
        if abs(angle) < max_angle:
            # 긴 선에 더 큰 가중치
            angles.append((angle, length))
    
    if len(angles) == 0:
        return 0.0
    
    # 길이 가중 평균 각도 계산
    total_length = sum(length for _, length in angles)
    if total_length == 0:
        return 0.0
    
    weighted_angle = sum(angle * length for angle, length in angles) / total_length
    
    return weighted_angle


def deskew_image(
    image: np.ndarray,
    angle: Optional[float] = None,
    max_angle: float = 5.0,
    border_color: Tuple[int, int, int] = (255, 255, 255)
) -> Tuple[np.ndarray, float]:
    """
    이미지의 기울기를 보정합니다.
    
    Args:
        image: 입력 이미지 (BGR 또는 grayscale)
        angle: 회전 각도 (None이면 자동 탐지)
        max_angle: 자동 탐지 시 최대 각도
        border_color: 회전 후 생기는 빈 공간 색상 (RGB)
        
    Returns:
        (보정된 이미지, 적용된 회전 각도)
    """
    # 각도 탐지
    if angle is None:
        angle = detect_skew_angle(image, max_angle)
    
    # 각도가 너무 작으면 보정하지 않음
    if abs(angle) < 0.1:
        return image.copy(), 0.0
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 회전 행렬 생성 (반시계 방향으로 회전)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 회전 후 이미지 크기 계산 (모든 내용이 포함되도록)
    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    new_w = int(h * sin_angle + w * cos_angle)
    new_h = int(h * cos_angle + w * sin_angle)
    
    # 회전 행렬 조정 (중심 이동)
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    # 이미지 회전
    if len(image.shape) == 3:
        border_value = border_color
    else:
        border_value = int(sum(border_color) / 3)  # grayscale
    
    rotated = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    
    return rotated, angle


def compute_x_projection_profile(image: np.ndarray) -> np.ndarray:
    """
    이미지의 X축 projection profile을 계산합니다.
    각 x좌표에서 세로 방향으로 검은 픽셀(또는 어두운 픽셀)의 합을 계산합니다.
    
    Args:
        image: 입력 이미지 (grayscale 또는 BGR)
        
    Returns:
        x축 projection profile (1D array, 각 x 좌표의 dark pixel 밀도)
    """
    # Grayscale로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # X축 projection: 각 x좌표에서 세로 방향 합산
    x_profile = np.sum(binary, axis=0).astype(np.float32)
    
    # 정규화 (0~1)
    if x_profile.max() > 0:
        x_profile = x_profile / x_profile.max()
    
    return x_profile


def compute_vertical_lines_profile(image: np.ndarray) -> np.ndarray:
    """
    Morphological 연산으로 세로선만 추출한 후 X축 projection profile을 계산합니다.
    테이블의 column separator를 더 정확하게 찾기 위한 함수입니다.
    
    Args:
        image: 입력 이미지 (grayscale 또는 BGR)
        
    Returns:
        x축 projection profile (1D array, 세로선 위치에서 높은 값)
    """
    # Grayscale로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological 연산으로 세로선 추출
    # 세로로 긴 커널 (세로선만 남김)
    vertical_kernel_size = max(h // 30, 15)  # 이미지 높이에 비례
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
    
    # Opening: erosion -> dilation (노이즈 제거 후 세로선만 남김)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # X축 projection: 각 x좌표에서 세로 방향 합산
    x_profile = np.sum(vertical_lines, axis=0).astype(np.float32)
    
    # 정규화 (0~1)
    if x_profile.max() > 0:
        x_profile = x_profile / x_profile.max()
    
    return x_profile


def find_column_separators(
    x_profile: np.ndarray,
    min_gap_width: int = 5,
    threshold: float = 0.1
) -> list[Tuple[int, int]]:
    """
    X축 projection profile에서 column separator (세로선) 위치를 찾습니다.
    세로선이 있는 곳은 projection이 높고, 빈 공간은 낮습니다.
    
    Args:
        x_profile: X축 projection profile
        min_gap_width: 최소 gap 너비 (noise 필터링)
        threshold: separator 판별 임계값
        
    Returns:
        separator 위치 리스트 [(start, end), ...]
    """
    # Profile에서 threshold 이상인 영역 찾기 (세로선 영역)
    is_line = x_profile > threshold
    
    separators = []
    in_separator = False
    start_x = 0
    
    for x, val in enumerate(is_line):
        if val and not in_separator:
            # Separator 시작
            in_separator = True
            start_x = x
        elif not val and in_separator:
            # Separator 종료
            in_separator = False
            if x - start_x >= min_gap_width:
                separators.append((start_x, x))
    
    # 마지막 separator 처리
    if in_separator:
        end_x = len(x_profile)
        if end_x - start_x >= min_gap_width:
            separators.append((start_x, end_x))
    
    return separators


def find_vertical_line_peaks(
    x_profile: np.ndarray,
    min_height: float = 0.3,
    min_prominence: float = 0.1,
    max_width: int = 30
) -> list[int]:
    """
    X축 projection profile에서 세로선(vertical line)에 해당하는 sharp peak를 찾습니다.
    세로선은 좁은 범위에서 높은 peak를 형성합니다.
    
    Args:
        x_profile: X축 projection profile (정규화된 값)
        min_height: 최소 peak 높이 (0~1)
        min_prominence: 주변 대비 최소 prominence
        max_width: peak의 최대 너비 (세로선은 좁아야 함)
        
    Returns:
        peak 위치(x좌표) 리스트
    """
    n = len(x_profile)
    if n < 3:
        return []
    
    peaks = []
    
    for i in range(1, n - 1):
        # Local maximum 확인
        if x_profile[i] > x_profile[i-1] and x_profile[i] > x_profile[i+1]:
            if x_profile[i] < min_height:
                continue
            
            # Peak 너비 측정 (half-height에서)
            half_height = x_profile[i] / 2
            left = i
            right = i
            
            while left > 0 and x_profile[left] > half_height:
                left -= 1
            while right < n - 1 and x_profile[right] > half_height:
                right += 1
            
            width = right - left
            
            # Prominence 계산 (주변 최소값 대비)
            search_range = min(50, i, n - i - 1)
            local_min_left = min(x_profile[max(0, i-search_range):i]) if i > 0 else 0
            local_min_right = min(x_profile[i+1:min(n, i+1+search_range)]) if i < n-1 else 0
            local_baseline = max(local_min_left, local_min_right)
            prominence = x_profile[i] - local_baseline
            
            # 세로선 조건: 좁고, 높고, prominent한 peak
            if width <= max_width and prominence >= min_prominence:
                peaks.append(i)
    
    return peaks


def find_last_column_separator(
    image: np.ndarray,
    min_separator_width: int = 3,
    edge_margin_ratio: float = 0.03,
    right_search_ratio: float = 0.5
) -> Optional[int]:
    """
    테이블 이미지에서 마지막 컬럼(Answer column)의 시작 위치를 찾습니다.
    테이블 우측 경계를 제외하고, 우측에서 가장 가까운 세로선을 찾습니다.
    
    로직:
    1. 모든 sharp peak (세로선 후보) 탐지
    2. 테이블 경계(가장 우측)를 제외
    3. 남은 peak 중 가장 우측 것이 Answer column separator
    
    Args:
        image: 테이블 crop 이미지
        min_separator_width: 최소 separator 너비 (deprecated, 호환성 유지)
        edge_margin_ratio: 이미지 가장자리 margin 비율 (테이블 경계 제외)
        right_search_ratio: 우측에서 검색할 영역 비율
        
    Returns:
        Answer column 시작 x좌표 (table crop 기준), 실패 시 None
    """
    h, w = image.shape[:2]
    
    # Morphological 연산으로 세로선만 추출한 profile 사용
    # 이렇게 하면 텍스트 영역의 노이즈 없이 세로선만 정확히 탐지
    x_profile = compute_vertical_lines_profile(image)
    
    # 경계 margin (픽셀 단위)
    edge_margin = int(w * edge_margin_ratio)
    
    # 세로선 peak 찾기 - morphological profile은 세로선만 남기므로 기준 완화
    vertical_peaks = find_vertical_line_peaks(
        x_profile, 
        min_height=0.3,    # 세로선 profile에서는 threshold 낮춰도 됨
        min_prominence=0.10,
        max_width=30
    )
    
    # 왼쪽 경계 제외한 peak
    valid_peaks = [p for p in vertical_peaks if p > edge_margin]
    
    if len(valid_peaks) == 0:
        # Fallback 1: 덜 엄격한 기준으로 재시도
        vertical_peaks = find_vertical_line_peaks(
            x_profile, 
            min_height=0.2,
            min_prominence=0.05,
            max_width=50
        )
        valid_peaks = [p for p in vertical_peaks if p > edge_margin]
    
    if len(valid_peaks) == 0:
        # Fallback 2: 우측 영역에서 가장 큰 peak 직접 탐색
        search_start = int(w * (1 - right_search_ratio))
        search_end = w - edge_margin
        
        if search_end <= search_start:
            return None
        
        search_region = x_profile[search_start:search_end]
        
        if len(search_region) == 0:
            return None
        
        max_idx = np.argmax(search_region)
        answer_column_x = search_start + max_idx
        
        return answer_column_x
    
    # Peak들을 왼쪽에서 오른쪽 순서로 정렬
    sorted_peaks = sorted(valid_peaks)
    
    # 테이블 경계 제외 전략:
    # - 가장 왼쪽 peak: 테이블 왼쪽 경계일 가능성 높음 (제외)
    # - 가장 오른쪽 peak: 테이블 오른쪽 경계일 가능성 높음 (제외)
    # - 나머지 중 가장 오른쪽이 Answer column separator
    
    if len(sorted_peaks) >= 3:
        # 양쪽 경계 제외하고 내부의 마지막 peak 선택
        # 왼쪽 경계 제외: 첫 번째 peak가 왼쪽 20% 이내에 있으면 제외
        interior_peaks = sorted_peaks.copy()
        
        # 왼쪽 경계 제외 (왼쪽 10% 이내)
        if interior_peaks[0] < w * 0.10:
            interior_peaks = interior_peaks[1:]
        
        # 오른쪽 경계 제외 (오른쪽 5% 이내)
        if len(interior_peaks) > 0 and interior_peaks[-1] > w * 0.95:
            interior_peaks = interior_peaks[:-1]
        
        if len(interior_peaks) > 0:
            # 내부 peak 중 가장 오른쪽 선택
            answer_separator_peak = interior_peaks[-1]
        else:
            # 내부 peak가 없으면 원래 정렬에서 두 번째 오른쪽 선택
            answer_separator_peak = sorted_peaks[-2] if len(sorted_peaks) > 1 else sorted_peaks[-1]
    elif len(sorted_peaks) == 2:
        # Peak가 2개면 오른쪽 것을 선택 (왼쪽은 테이블 경계)
        answer_separator_peak = sorted_peaks[-1]
    else:
        # Peak가 1개면 그것을 사용
        answer_separator_peak = sorted_peaks[0]
    
    # Peak 위치에서 세로선의 오른쪽 끝 찾기
    answer_column_start = answer_separator_peak
    for x in range(answer_separator_peak, min(answer_separator_peak + 30, w)):
        if x_profile[x] < 0.15:  # 세로선 영역 끝
            answer_column_start = x
            break
    
    return answer_column_start


def find_peaks_simple(arr: np.ndarray, min_distance: int = 10) -> list[int]:
    """
    1D array에서 peak 위치를 찾습니다 (간단한 구현).
    
    Args:
        arr: 1D array
        min_distance: peak 사이 최소 거리
        
    Returns:
        peak 위치 리스트
    """
    peaks = []
    n = len(arr)
    
    if n < 3:
        return peaks
    
    for i in range(1, n - 1):
        # Local maximum 확인
        if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
            # 이전 peak와의 거리 확인
            if len(peaks) == 0 or i - peaks[-1] >= min_distance:
                peaks.append(i)
    
    return peaks


def find_answer_section(
    image: np.ndarray,
    layout_model,
    min_table_area_ratio: float = 0.1,
    answer_column_width_ratio: float = 0.15,
    enable_deskew: bool = True,
    max_skew_angle: float = 5.0
) -> AnswerSectionResult:
    """
    원본 이미지에서 Answer 섹션을 찾아 crop합니다.
    
    흐름:
    1. PP-DocLayout_plus-L로 모든 bbox 탐지
    2. Table bbox 중 가장 큰 것 선택
    3. Table crop 후 기울기 보정
    4. X축 projection profile로 Answer column 위치 찾기
    5. Answer column crop하여 반환
    
    Args:
        image: 원본 이미지 (numpy array)
        layout_model: PP-DocLayout_plus-L 모델 객체
        min_table_area_ratio: 최소 테이블 면적 비율 (너무 작은 테이블 무시)
        answer_column_width_ratio: Answer column 최소 너비 비율 (fallback용)
        enable_deskew: 기울기 보정 활성화 여부
        max_skew_angle: 보정할 최대 기울기 각도 (도)
        
    Returns:
        AnswerSectionResult
    """
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    image_area = h * w
    
    meta = {
        "image_size": (w, h),
        "stage": "layout"
    }
    
    rotation_angle = 0.0
    
    # 1. Layout detection
    try:
        all_boxes = detect_all_bboxes(image, layout_model)
        meta["total_boxes"] = len(all_boxes)
    except Exception as e:
        meta["error"] = f"Layout detection failed: {str(e)}"
        return AnswerSectionResult(success=False, meta=meta)
    
    # 2. Table bbox 필터링
    table_boxes = get_table_boxes(all_boxes)
    meta["table_boxes_count"] = len(table_boxes)
    
    if len(table_boxes) == 0:
        meta["reason"] = "no_table_found"
        return AnswerSectionResult(success=False, meta=meta)
    
    # 3. 가장 큰 Table bbox 선택
    largest_table = max(table_boxes, key=lambda b: b.bbox.area)
    table_bbox = largest_table.bbox
    meta["table_bbox"] = {
        "x1": table_bbox.x1,
        "y1": table_bbox.y1,
        "x2": table_bbox.x2,
        "y2": table_bbox.y2,
        "area": table_bbox.area,
        "score": largest_table.score
    }
    
    # 면적 체크
    if table_bbox.area / image_area < min_table_area_ratio:
        meta["reason"] = "table_too_small"
        return AnswerSectionResult(success=False, meta=meta)
    
    # 4. Table crop
    table_image = crop_bbox(image, table_bbox, padding=0)
    if table_image is None:
        meta["reason"] = "table_crop_failed"
        return AnswerSectionResult(success=False, meta=meta)
    
    meta["table_crop_size_original"] = (table_image.shape[1], table_image.shape[0])
    
    # 5. 기울기 보정 (가로선 기준)
    if enable_deskew:
        meta["stage"] = "deskew"
        table_image, rotation_angle = deskew_image(table_image, max_angle=max_skew_angle)
        meta["rotation_angle"] = rotation_angle
        meta["table_crop_size_after_deskew"] = (table_image.shape[1], table_image.shape[0])
    
    meta["table_crop_size"] = (table_image.shape[1], table_image.shape[0])
    meta["stage"] = "column_detection"
    
    # 6. X축 projection profile로 Answer column 위치 찾기
    answer_column_x = find_last_column_separator(table_image)
    
    table_h, table_w = table_image.shape[:2]
    
    if answer_column_x is None:
        # Fallback: 오른쪽 일정 비율을 Answer column으로 가정
        fallback_ratio = answer_column_width_ratio
        answer_column_x = int(table_w * (1 - fallback_ratio))
        meta["answer_column_method"] = "fallback_ratio"
        meta["fallback_ratio"] = fallback_ratio
    else:
        meta["answer_column_method"] = "projection_profile"
    
    meta["answer_column_x_start"] = answer_column_x
    
    # Answer column 너비 검증
    answer_column_width = table_w - answer_column_x
    if answer_column_width < 10:  # 너무 좁으면 실패
        meta["reason"] = "answer_column_too_narrow"
        return AnswerSectionResult(success=False, meta=meta)
    
    meta["answer_column_width"] = answer_column_width
    
    # 7. Answer section crop
    answer_section_image = table_image[:, answer_column_x:].copy()
    
    meta["stage"] = "complete"
    meta["answer_section_size"] = (answer_section_image.shape[1], answer_section_image.shape[0])
    
    return AnswerSectionResult(
        success=True,
        answer_section_image=answer_section_image,
        table_image=table_image,
        table_bbox=table_bbox,
        answer_column_x_start=answer_column_x,
        rotation_angle=rotation_angle,
        meta=meta
    )


def visualize_projection_profile(
    image: np.ndarray,
    save_path: Optional[str] = None,
    answer_column_x: Optional[int] = None
) -> np.ndarray:
    """
    디버깅용: 이미지와 X축 projection profile을 시각화합니다.
    
    Args:
        image: 입력 이미지
        save_path: 저장 경로 (None이면 저장 안 함)
        answer_column_x: Answer column 시작 x좌표 (표시용)
        
    Returns:
        시각화된 이미지
    """
    import matplotlib.pyplot as plt
    
    h, w = image.shape[:2]
    x_profile_original = compute_x_projection_profile(image)
    x_profile_morph = compute_vertical_lines_profile(image)
    
    # Figure 생성 - 3개 서브플롯
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 원본 이미지
    if len(image.shape) == 3:
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Table Image (Deskewed)")
    
    # Answer column 위치 표시 (이미지 위에)
    if answer_column_x is not None:
        axes[0].axvline(x=answer_column_x, color='red', linewidth=2, linestyle='--')
    
    axes[0].axis('off')
    
    # 일반 X축 Projection Profile
    axes[1].plot(x_profile_original, color='blue', alpha=0.7, label='Original Profile')
    axes[1].set_xlim(0, w)
    axes[1].set_title("Original X-axis Projection Profile")
    axes[1].set_xlabel("X position")
    axes[1].set_ylabel("Normalized projection")
    axes[1].grid(True, alpha=0.3)
    if answer_column_x is not None:
        axes[1].axvline(x=answer_column_x, color='red', linewidth=2, linestyle='--', label='Answer Column')
    axes[1].legend(loc='upper right')
    
    # Morphological 세로선 Profile
    axes[2].plot(x_profile_morph, color='green', linewidth=2, label='Vertical Lines Profile')
    axes[2].set_xlim(0, w)
    axes[2].set_title("Morphological Vertical Lines Profile (Used for Column Detection)")
    axes[2].set_xlabel("X position")
    axes[2].set_ylabel("Normalized projection")
    axes[2].grid(True, alpha=0.3)
    
    # Vertical line peaks 표시
    vertical_peaks = find_vertical_line_peaks(x_profile_morph, min_height=0.3, min_prominence=0.10, max_width=30)
    for peak in vertical_peaks:
        axes[2].axvline(x=peak, color='darkgreen', alpha=0.7, linestyle='-', linewidth=1.5)
        axes[2].plot(peak, x_profile_morph[peak], 'go', markersize=10)
    
    # Answer column separator 표시 (빨간 세로선)
    if answer_column_x is not None:
        axes[2].axvline(x=answer_column_x, color='red', linewidth=2, linestyle='--', label='Answer Column')
    
    # Edge margin 표시
    edge_margin = int(w * 0.03)
    axes[2].axvspan(0, edge_margin, alpha=0.15, color='gray', label='Edge margin')
    axes[2].axvspan(w - edge_margin, w, alpha=0.15, color='gray')
    
    # 우측 경계 영역 표시 (92% 이상)
    boundary_start = int(w * 0.92)
    axes[2].axvspan(boundary_start, w, alpha=0.1, color='blue', label='Boundary zone')
    
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 저장된 이미지 반환
        result_img = cv2.imread(save_path)
        return result_img
    else:
        # Figure를 numpy array로 변환
        fig.canvas.draw()
        result_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return result_img

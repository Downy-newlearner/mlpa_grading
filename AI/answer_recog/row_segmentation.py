"""
row_segmentation.py - Answer 섹션 Row 분할 모듈

Answer 섹션 이미지에서 Y축 projection profile을 사용하여
각 문제(row)를 개별적으로 분할합니다.

핵심 로직:
1. Y축 projection 계산
2. Peak 후보 추출 (row 중심)
3. Peak 간 valley 찾기 (row 경계)
4. Valley 깊이 필터링
5. Row 높이 정규화

보완 규칙:
1. Row 높이 제약 (median ± tolerance)
2. Valley 깊이 임계값 (진짜 경계 확인)
3. answer_count 기반 검증 (상위 로직에서 처리)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class RowSegment:
    """개별 Row 정보"""
    row_number: int               # Row 번호 (0-indexed)
    y_start: int                  # Row 시작 y좌표
    y_end: int                    # Row 끝 y좌표
    row_image: Optional[np.ndarray] = None  # Row 이미지
    peak_y: Optional[int] = None  # Peak 위치 (row 중심)
    valley_depth: Optional[float] = None  # Valley 깊이 (상대값)
    
    @property
    def height(self) -> int:
        return self.y_end - self.y_start


@dataclass 
class RowSegmentationResult:
    """Row Segmentation 결과"""
    success: bool
    rows: List[RowSegment] = field(default_factory=list)
    source_image: Optional[np.ndarray] = None  # 원본 answer section
    y_profile: Optional[np.ndarray] = None     # Y축 projection profile
    peaks: List[int] = field(default_factory=list)    # Peak 위치들
    valleys: List[int] = field(default_factory=list)  # Valley 위치들
    median_row_height: Optional[float] = None  # Row 높이 중앙값
    meta: dict = field(default_factory=dict)


def compute_y_projection_profile(image: np.ndarray) -> np.ndarray:
    """
    이미지의 Y축 projection profile을 계산합니다.
    각 y좌표에서 가로 방향으로 검은 픽셀의 합을 계산합니다.
    
    Args:
        image: 입력 이미지 (grayscale 또는 BGR)
        
    Returns:
        y축 projection profile (1D array)
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
    
    return y_profile


def compute_horizontal_lines_profile(
    image: np.ndarray,
    min_line_width_ratio: float = 0.7
) -> np.ndarray:
    """
    Morphological 연산으로 가로선만 추출한 후 Y축 projection profile을 계산합니다.
    테이블의 row separator(가로선)를 더 정확하게 찾기 위한 함수입니다.
    
    **중요**: 전체 너비의 min_line_width_ratio(기본 70%) 이상인 "메인 가로선"만 인식합니다.
    표 내부의 짧은 가로선은 무시됩니다.
    
    Args:
        image: 입력 이미지 (grayscale 또는 BGR)
        min_line_width_ratio: 최소 가로선 길이 비율 (전체 너비 대비)
        
    Returns:
        y축 projection profile (1D array, 메인 가로선 위치에서 높은 값)
    """
    # Grayscale로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    
    # 이진화 (Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological 연산으로 가로선 추출
    # 가로로 긴 커널 (가로선만 남김)
    horizontal_kernel_size = max(w // 30, 15)  # 이미지 너비에 비례
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    
    # Opening: erosion -> dilation (노이즈 제거 후 가로선만 남김)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # =====================================================================
    # 메인 가로선 필터링: 전체 너비의 min_line_width_ratio 이상인 가로선만 남김
    # =====================================================================
    min_line_width = int(w * min_line_width_ratio)
    
    # 각 행(y좌표)에서 가로선 길이 측정
    # 연결된 컴포넌트(가로선 세그먼트)의 길이를 측정
    filtered_lines = np.zeros_like(horizontal_lines)
    
    for y in range(h):
        row = horizontal_lines[y, :]
        
        # 연결된 흰색 픽셀 세그먼트 찾기
        in_segment = False
        segment_start = 0
        
        for x in range(w + 1):  # w+1까지 순회하여 마지막 세그먼트도 처리
            if x < w and row[x] > 0:
                if not in_segment:
                    in_segment = True
                    segment_start = x
            else:
                if in_segment:
                    segment_end = x
                    segment_length = segment_end - segment_start
                    
                    # 메인 가로선 조건: 전체 너비의 min_line_width_ratio 이상
                    if segment_length >= min_line_width:
                        filtered_lines[y, segment_start:segment_end] = 255
                    
                    in_segment = False
    
    # Y축 projection: 각 y좌표에서 가로 방향 합산 (필터링된 가로선만)
    y_profile = np.sum(filtered_lines, axis=1).astype(np.float32)
    
    return y_profile


def find_peaks(profile: np.ndarray, min_distance: int = 10, min_height_ratio: float = 0.1) -> List[int]:
    """
    1D profile에서 peak(극대점)을 찾습니다.
    
    Args:
        profile: 1D array
        min_distance: peak 간 최소 거리
        min_height_ratio: 최소 높이 비율 (max 대비)
        
    Returns:
        peak 위치 리스트
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
    min_depth_ratio: float = 0.3
) -> Tuple[List[int], List[float]]:
    """
    인접한 두 peak 사이에서 valley(극소점)를 찾습니다.
    
    Valley 깊이 조건: valley < min_depth_ratio * min(peak_n, peak_{n+1})
    
    Args:
        profile: 1D array
        peaks: peak 위치 리스트
        min_depth_ratio: valley 깊이 임계값 비율
        
    Returns:
        (valley 위치 리스트, valley 깊이 리스트)
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
        
        # 깊이 계산 (두 peak 중 작은 값 대비)
        peak_min = min(profile[y_start], profile[y_end])
        
        if peak_min > 0:
            relative_depth = valley_val / peak_min
        else:
            relative_depth = 1.0
        
        # 깊이 조건 확인 (valley가 충분히 낮아야 함)
        if relative_depth <= min_depth_ratio:
            valleys.append(valley_y)
            depths.append(relative_depth)
        else:
            # 조건 불충족시에도 중간점을 valley로 사용 (merge 방지)
            # 이 경우 depth를 높게 표시하여 신뢰도가 낮음을 표시
            valleys.append(valley_y)
            depths.append(relative_depth)
    
    return valleys, depths


def normalize_row_heights(
    valleys: List[int], 
    peaks: List[int],
    image_height: int,
    tolerance_ratio: float = 0.3
) -> List[int]:
    """
    Row 높이를 정규화합니다.
    
    - Peak 간 거리의 median을 기준 높이 H로 설정
    - Valley 위치가 H ± tolerance 범위를 벗어나면 조정
    
    Args:
        valleys: valley 위치 리스트
        peaks: peak 위치 리스트
        image_height: 이미지 높이
        tolerance_ratio: 허용 오차 비율
        
    Returns:
        정규화된 valley 위치 리스트
    """
    if len(peaks) < 2:
        return valleys
    
    # Peak 간 거리 계산
    distances = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
    
    if len(distances) == 0:
        return valleys
    
    # Median 높이 계산
    median_height = np.median(distances)
    tolerance = median_height * tolerance_ratio
    
    normalized = []
    for i, valley in enumerate(valleys):
        if i < len(peaks) - 1:
            expected_valley = peaks[i] + int(median_height / 2)
            
            # Valley가 기대 위치에서 너무 벗어나면 조정
            lower_bound = expected_valley - tolerance
            upper_bound = expected_valley + tolerance
            
            # 현재 valley가 범위를 벗어나면 클리핑
            if valley < lower_bound:
                valley = int(lower_bound)
            elif valley > upper_bound:
                valley = int(upper_bound)
                
        # 이미지 범위 내로 제한
        valley = max(0, min(valley, image_height - 1))
        normalized.append(valley)
    
    return normalized


def find_row_separators_morphological(
    image: np.ndarray,
    min_line_length_ratio: float = 0.6,  # 기본값 0.6 (60%) - 안전한 기준
    min_row_height: int = 30
) -> List[int]:
    """
    Morphological 연산(Dilation)으로 가로선을 강화하여 탐지합니다.
    Dilation을 통해 흐린 선도 60% 이상으로 증폭시키고, 
    짧은 선은 60% 미만으로 유지하여 필터링합니다.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    
    # 이진화
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # 1. Morphological Open: 텍스트 노이즈 제거
    open_kernel_len = int(w * 0.2)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_len, 1))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    
    # 2. Dilation: 선 연결 및 강화
    # 메인 가로선(약 100%) -> Dilation 후 60% 충분히 넘음
    # 내부 가로선(약 25%) -> Dilation 후에도 60% 넘기 힘듦
    dilate_len = int(w * 0.05)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_len, 1))
    dilated = cv2.dilate(opened, dilate_kernel, iterations=2)
    
    # Projection
    y_projection = np.sum(dilated, axis=1) / 255
    
    threshold = w * min_line_length_ratio
    
    separators = []
    in_line = False
    line_start = 0
    max_val_in_line = 0
    
    for y in range(h):
        val = y_projection[y]
        
        if val >= threshold:
            if not in_line:
                in_line = True
                line_start = y
                max_val_in_line = val
            else:
                max_val_in_line = max(max_val_in_line, val)
        else:
            if in_line:
                in_line = False
                line_end = y
                
                line_center = (line_start + line_end) // 2
                
                if len(separators) == 0:
                    separators.append(line_center)
                else:
                    dist = line_center - separators[-1]
                    if dist >= min_row_height:
                        separators.append(line_center)
                    else:
                        separators[-1] = (separators[-1] + line_center) // 2
    
    if in_line:
        line_center = (line_start + h) // 2
        if len(separators) == 0 or line_center - separators[-1] >= min_row_height:
            separators.append(line_center)
            
    return separators


def segment_rows(
    image: np.ndarray,
    min_row_height: int = 30,
    max_row_height: int = 200,
    min_depth_ratio: float = 0.3,
    height_tolerance: float = 0.3,
    expected_row_count: Optional[int] = None,
    use_morphological: bool = True,
    min_line_length_ratio: float = 0.6  # 기본값 0.4 -> 0.6으로 상향 (안전한 기준)
) -> RowSegmentationResult:
    """
    Answer 섹션 이미지를 row 단위로 분할합니다.
    
    기본 알고리즘 (morphological):
    1. Morphological 연산으로 가로선(row separator) 직접 탐지
    2. 가로선 기반으로 row 분할
    
    Fallback 알고리즘 (peak/valley):
    1. Y축 projection profile 계산
    2. Peak 추출 (row 중심)
    3. Peak 간 valley 찾기 (row 경계)
    4. Valley 깊이 필터링 + 정규화
    
    Args:
        image: Answer 섹션 이미지
        min_row_height: 최소 row 높이
        max_row_height: 최대 row 높이
        min_depth_ratio: valley 깊이 임계값 (낮을수록 엄격)
        height_tolerance: row 높이 허용 오차 비율
        expected_row_count: 예상 row 개수 (검증용, None이면 무시)
        use_morphological: Morphological 가로선 탐지 사용 여부 (기본 True)
        min_line_length_ratio: 가로선 최소 길이 비율 (morphological 모드용)
        
    Returns:
        RowSegmentationResult
    """
    h, w = image.shape[:2]
    
    meta = {
        "image_size": (w, h),
        "min_row_height": min_row_height,
        "max_row_height": max_row_height
    }
    
    # Y축 projection profile 계산 (시각화용)
    y_profile = compute_y_projection_profile(image)
    
    # 1차 시도: Morphological 가로선 탐지
    if use_morphological:
        separators = find_row_separators_morphological(
            image,
            min_line_length_ratio=min_line_length_ratio,
            min_row_height=min_row_height
        )
        meta["morphological_separators"] = len(separators)
        meta["method"] = "morphological"
        
        # 충분한 separator가 탐지되었는지 확인
        if len(separators) >= 2:
            # Morphological 결과 사용
            rows = []
            boundaries = [0] + separators + [h]
            
            for i in range(len(boundaries) - 1):
                y_start = boundaries[i]
                y_end = boundaries[i + 1]
                row_height = y_end - y_start
                
                # 높이 검증
                if row_height < min_row_height:
                    if len(rows) > 0:
                        rows[-1].y_end = y_end
                        rows[-1].row_image = image[rows[-1].y_start:y_end, :].copy()
                    continue
                
                row = RowSegment(
                    row_number=len(rows),
                    y_start=y_start,
                    y_end=y_end,
                    row_image=image[y_start:y_end, :].copy()
                )
                rows.append(row)
            
            if len(rows) >= 3:  # 최소 3개 row가 있어야 성공으로 판정
                meta["final_row_count"] = len(rows)
                
                # Expected row count 검증
                if expected_row_count is not None and len(rows) != expected_row_count:
                    meta["row_count_mismatch"] = True
                    meta["expected"] = expected_row_count
                    meta["actual"] = len(rows)
                
                return RowSegmentationResult(
                    success=True,
                    rows=rows,
                    source_image=image,
                    y_profile=y_profile,
                    valleys=separators,
                    meta=meta
                )
    
    # Fallback: Peak/Valley 방식
    meta["method"] = "peak_valley_fallback"
    
    # Profile smoothing (노이즈 제거)
    # 텍스트 라인의 미세한 Valley를 보존하기 위해 스무딩 최소화
    kernel_size = max(3, min_row_height // 8)
    if kernel_size % 2 == 0:
        kernel_size += 1
    y_profile_smooth = cv2.GaussianBlur(y_profile.reshape(-1, 1), (1, kernel_size), 0).flatten()
    
    # Peak 추출
    peaks = find_peaks(
        y_profile_smooth, 
        min_distance=min_row_height // 3, # 더 촘촘한 peak 허용
        min_height_ratio=0.05
    )
    
    meta["raw_peak_count"] = len(peaks)
    
    if len(peaks) < 2:
        # Peak가 부족하면 전체를 하나의 row로 처리
        row = RowSegment(
            row_number=0,
            y_start=0,
            y_end=h,
            row_image=image.copy()
        )
        return RowSegmentationResult(
            success=True,
            rows=[row],
            source_image=image,
            y_profile=y_profile,
            peaks=peaks,
            meta={**meta, "reason": "single_row_fallback"}
        )
    
    # 3. Valley 찾기
    valleys, depths = find_valleys_between_peaks(
        y_profile_smooth, 
        peaks, 
        min_depth_ratio=min_depth_ratio
    )
    
    meta["raw_valley_count"] = len(valleys)
    
    # 4. Row 높이 정규화
    valleys_normalized = normalize_row_heights(
        valleys, peaks, h, 
        tolerance_ratio=height_tolerance
    )
    
    # Peak 간 거리로 median row height 계산
    distances = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
    median_height = float(np.median(distances)) if distances else 0
    
    # 5. Row 생성
    # 첫 번째 row: 0 ~ 첫 valley
    # 중간 rows: valley[i] ~ valley[i+1]
    # 마지막 row: 마지막 valley ~ h
    
    rows = []
    
    # Row 경계 결정
    boundaries = [0] + valleys_normalized + [h]
    
    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        row_height = y_end - y_start
        
        # 높이 검증
        if row_height < min_row_height:
            # 너무 작으면 이전 row와 merge (또는 skip)
            if len(rows) > 0:
                rows[-1].y_end = y_end
                rows[-1].row_image = image[rows[-1].y_start:y_end, :].copy()
            continue
        
        if row_height > max_row_height:
            # 너무 크면 분할 시도 (추후 segment_rows_recursive에서 처리)
            pass
        
        # Peak 위치 찾기
        peak_y = None
        if i < len(peaks):
            peak_y = peaks[i]
        
        # Valley depth
        valley_depth = None
        if i < len(depths):
            valley_depth = depths[i]
        
        row = RowSegment(
            row_number=len(rows),
            y_start=y_start,
            y_end=y_end,
            row_image=image[y_start:y_end, :].copy(),
            peak_y=peak_y,
            valley_depth=valley_depth
        )
        rows.append(row)
    
    meta["final_row_count"] = len(rows)
    
    # Expected row count 검증
    if expected_row_count is not None:
        if len(rows) != expected_row_count:
            meta["row_count_mismatch"] = True
            meta["expected"] = expected_row_count
            meta["actual"] = len(rows)
    
    return RowSegmentationResult(
        success=True,
        rows=rows,
        source_image=image,
        y_profile=y_profile,
        peaks=peaks,
        valleys=valleys_normalized,
        median_row_height=median_height,
        meta=meta
    )


def segment_rows_recursive(
    image: np.ndarray,
    min_row_height: int = 30,
    max_row_height: int = 200,
    text_line_height: int = 25,
    enable_recursion: bool = False  # 기본값 False로 변경 (가로선 기준 분할만 수행)
) -> RowSegmentationResult:
    """
    Row Segmentation 함수.
    
    기본적으로 물리적 가로선(morphological) 기반 분할을 수행합니다.
    (enable_recursion=True일 때만 큰 Row에 대해 텍스트 라인 기반 2차 분할 수행)
    
    주의: "메인 가로선" 기준 분할을 위해 enable_recursion=False 추천.
    """
    # 1차 분할: Morphological 가로선 탐지 우선
    # (min_line_length_ratio=0.7은 segment_rows 기본값 사용)
    result = segment_rows(
        image, 
        min_row_height=min_row_height, 
        max_row_height=max_row_height * 10,  # 큰 Row도 1차 분할에선 허용
        use_morphological=True
    )
    
    # 재귀 분할이 비활성화되어 있으면 1차 분할 결과 반환
    if not enable_recursion:
        return result
        
    final_rows = []
    
    for row in result.rows:
        # Row 높이가 매우 큰 경우 (예: max_row_height 초과) -> 2차 분할
        if row.height > max_row_height:
            print(f"Sub-segmenting large row {row.row_number} (height: {row.height})")
            
            # 2차 분할: Peak/Valley 방식 (use_morphological=False)
            sub_result = segment_rows(
                row.row_image,
                min_row_height=text_line_height,
                max_row_height=max_row_height,
                min_depth_ratio=0.9,
                height_tolerance=0.5,
                use_morphological=False
            )
            
            if sub_result.success and len(sub_result.rows) > 1:
                # 분할 성공 시 sub-rows 추가
                for sub_row in sub_result.rows:
                    abs_y_start = row.y_start + sub_row.y_start
                    abs_y_end = row.y_start + sub_row.y_end
                    
                    new_row = RowSegment(
                        row_number=len(final_rows),
                        y_start=abs_y_start,
                        y_end=abs_y_end,
                        row_image=sub_row.row_image,
                        peak_y=(row.y_start + sub_row.peak_y) if sub_row.peak_y else None,
                        valley_depth=sub_row.valley_depth
                    )
                    final_rows.append(new_row)
            else:
                row.row_number = len(final_rows)
                final_rows.append(row)
        else:
            row.row_number = len(final_rows)
            final_rows.append(row)
            
    result.rows = final_rows
    result.meta["recursive_segmentation"] = True
    result.meta["final_row_count"] = len(final_rows)
    
    return result


def segment_text_lines(
    row_image: np.ndarray,
    min_height: int = 10,    # 최소 라인 높이 (너무 작은 노이즈 블록 무시)
    margin: int = 5,         # 상하단 가로선 제거 margin
    expected_count: Optional[int] = None # Fallback용
) -> List[np.ndarray]:
    """
    하나의 Row 이미지에서 텍스트 라인을 분리합니다.
    Peak Finding 대신 Threshold 기반으로 텍스트 덩어리(Blob)를 감지하고, 그 사이의 Valley를 찾아 분할합니다.
    
    Args:
        row_image: Row 이미지
        min_height: 감지할 최소 텍스트 높이
        margin: 상하단 제거 여백
        expected_count: 예상 개수 (Fallback)
        
    Returns:
        분할된 이미지 리스트
    """
    if row_image is None or row_image.size == 0:
        return []

    h, w = row_image.shape[:2]
    
    if h <= 2 * margin:
        return [row_image]

    # 1. 프로파일 계산 (상하단 제거)
    img_for_profile = row_image[margin:h-margin, :]
    y_profile = compute_y_projection_profile(img_for_profile)
    
    # 2. Text Blob 탐지 (Thresholding)
    max_val = np.max(y_profile) if np.max(y_profile) > 0 else 1
    threshold = max_val * 0.05  # 최대값의 5% 이상이면 텍스트 구간으로 간주
    
    is_text = y_profile > threshold
    
    # Morphological Close와 유사하게, 짧은 공백은 메워줌 (텍스트 내 자간/행간 보정)
    # 1D Dilation/Erosion 직접 구현 or 단순 Loop
    # 여기서는 간단히 '연속된 텍스트 구간'을 찾음
    
    text_blocks = [] # (start, end)
    in_block = False
    start = 0
    
    for i, val in enumerate(is_text):
        if val and not in_block:
            in_block = True
            start = i
        elif not val and in_block:
            in_block = False
            # 블록이 너무 작으면 무시 (노이즈)
            if i - start >= min_height:
                text_blocks.append((start, i))
            
    # 마지막 블록 처리
    if in_block:
        if len(y_profile) - start >= min_height:
            text_blocks.append((start, len(y_profile)))
            
    # 3. Valley 찾기 (블록 사이의 최저점)
    cut_points = [0]
    
    for i in range(len(text_blocks) - 1):
        block_end = text_blocks[i][1]
        next_block_start = text_blocks[i+1][0]
        
        # 블록 사이 구간 (Valley Region)
        valley_region = y_profile[block_end:next_block_start]
        
        if len(valley_region) > 0:
            # 구간 내 최저점 인덱스 (relative to block_end)
            min_idx = np.argmin(valley_region)
            cut_point = block_end + min_idx
        else:
            # 붙어있으면 중간 지점
            cut_point = (block_end + next_block_start) // 2
            
        cut_points.append(cut_point)
        
    cut_points.append(h) # 마지막은 보정 전 h가 아니라 프로파일 기준 길이여야 하는데... 
    # 어차피 아래에서 margin 더해서 보정하므로 여기선 len(y_profile) 사용하거나
    # 그냥 cut_points는 profile 기준 좌표임.
    
    # 4. 좌표 보정 및 자르기
    # cut_points는 img_for_profile 기준이므로 margin을 더해서 원본 좌표로 변환
    # 단, 0과 마지막(h)은 예외 처리
    
    final_cuts = [0]
    for cp in cut_points[1:-1]: # 첫 0 제외, 마지막 제외
        final_cuts.append(cp + margin)
    final_cuts.append(h)
    
    # Fallback Check
    detected_count = len(final_cuts) - 1
    sub_rows = []
    
    if expected_count is not None and detected_count != expected_count:
        print(f"    [segment_text_lines] Mismatch: Detected {detected_count} blocks != Expected {expected_count}. Using fallback.")
        
        # Fallback: Equal Split
        if expected_count > 1:
            part_height = h // expected_count
            for i in range(expected_count):
                y1 = i * part_height
                y2 = (i + 1) * part_height if i < expected_count - 1 else h
                sub_rows.append(row_image[y1:y2, :].copy())
        else:
             sub_rows = [row_image]
    else:
        # 정상 분할
        for i in range(len(final_cuts) - 1):
            y1 = final_cuts[i]
            y2 = final_cuts[i+1]
            
            # 너무 얇으면 패스 (혹시나)
            if y2 - y1 < min_height:
                continue
            sub_rows.append(row_image[y1:y2, :].copy())
            
    return sub_rows


def visualize_row_segmentation(
    result: RowSegmentationResult,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Row segmentation 결과를 시각화합니다.
    
    Args:
        result: RowSegmentationResult
        save_path: 저장 경로 (None이면 저장 안 함)
        
    Returns:
        시각화된 이미지
    """
    import matplotlib.pyplot as plt
    
    if result.source_image is None:
        return None
    
    image = result.source_image
    h, w = image.shape[:2]
    
    # Figure 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    
    # 원본 이미지 + row 경계 표시
    if len(image.shape) == 3:
        display_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    else:
        display_img = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    
    # Row 경계선 그리기
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, row in enumerate(result.rows):
        color = colors[i % len(colors)]
        cv2.line(display_img, (0, row.y_start), (w, row.y_start), color, 2)
        cv2.line(display_img, (0, row.y_end), (w, row.y_end), color, 2)
        # Row 번호 표시
        cv2.putText(display_img, f"R{row.row_number}", (10, row.y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    axes[0].imshow(display_img)
    axes[0].set_title(f"Row Segmentation ({len(result.rows)} rows)")
    axes[0].axis('off')
    
    # Y축 Projection Profile
    if result.y_profile is not None:
        # Profile을 수직으로 표시 (Y축과 일치시키기 위해)
        y_positions = np.arange(len(result.y_profile))
        axes[1].plot(result.y_profile, y_positions, color='blue', label='Y Profile')
        axes[1].set_ylim(h, 0)  # Y축 반전
        axes[1].set_xlabel("Projection value")
        axes[1].set_ylabel("Y position")
        axes[1].set_title("Y-axis Projection Profile")
        axes[1].grid(True, alpha=0.3)
        
        # Peak 표시
        for peak in result.peaks:
            axes[1].axhline(y=peak, color='green', linestyle='--', alpha=0.7)
            axes[1].plot(result.y_profile[peak], peak, 'go', markersize=8)
        
        # Valley (row 경계) 표시
        for valley in result.valleys:
            axes[1].axhline(y=valley, color='red', linestyle='-', alpha=0.5, linewidth=2)
        
        axes[1].legend()
    
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

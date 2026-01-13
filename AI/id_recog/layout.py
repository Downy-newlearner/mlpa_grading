"""
layout.py - 레이아웃 탐지 및 bbox crop

PP-DocLayout_plus-L 모델을 사용하여 모든 bbox를 탐지하고,
Table을 제외한 각 bbox를 crop하여 OCR에 전달합니다.
"""

import numpy as np
from PIL import Image
from dataclasses import dataclass

try:
    from .schemas import BBox
except ImportError:
    from schemas import BBox


@dataclass
class LayoutBox:
    """레이아웃 탐지 결과 박스"""
    bbox: BBox
    label: str
    score: float


def detect_all_bboxes(image: np.ndarray | Image.Image, layout_model) -> list[LayoutBox]:
    """
    레이아웃 모델로 모든 bounding box를 검출합니다.
    
    Args:
        image: 입력 이미지 (numpy array 또는 PIL Image)
        layout_model: PP-DocLayout_plus-L 모델 객체
        
    Returns:
        모든 LayoutBox 리스트
    """
    outputs = layout_model.predict(image, batch_size=1)
    
    all_boxes = []
    
    for res in outputs:
        result_data = res.json
        
        # PP-DocLayout_plus-L 모델의 결과 구조: res.json['res']['boxes']
        # 또는 PP-StructureV3 파이프라인: res.json['boxes']
        boxes = None
        if 'res' in result_data and isinstance(result_data['res'], dict):
            boxes = result_data['res'].get('boxes', [])
        elif 'boxes' in result_data:
            boxes = result_data['boxes']
        
        if not boxes:
            continue
            
        for box in boxes:
            label = box.get('label', '')
            score = box.get('score', 0.0)
            coord = box.get('coordinate', [])
            
            if len(coord) == 4:
                bbox = BBox.from_coordinate(coord)
                all_boxes.append(LayoutBox(bbox=bbox, label=label, score=score))
    
    return all_boxes


def get_non_table_boxes(layout_boxes: list[LayoutBox]) -> list[LayoutBox]:
    """
    Table을 제외한 bbox 리스트를 반환합니다.
    
    Args:
        layout_boxes: 전체 LayoutBox 리스트
        
    Returns:
        Table이 아닌 LayoutBox 리스트
    """
    non_table_boxes = []
    for box in layout_boxes:
        label_lower = box.label.lower()
        # table, form 등 테이블 관련 라벨 제외
        if 'table' not in label_lower and 'form' not in label_lower:
            non_table_boxes.append(box)
    
    return non_table_boxes


def get_table_boxes(layout_boxes: list[LayoutBox]) -> list[LayoutBox]:
    """
    Table bbox만 반환합니다.
    
    Args:
        layout_boxes: 전체 LayoutBox 리스트
        
    Returns:
        Table인 LayoutBox 리스트
    """
    table_boxes = []
    for box in layout_boxes:
        label_lower = box.label.lower()
        if 'table' in label_lower:
            table_boxes.append(box)
    
    return table_boxes


def crop_bbox(
    image: np.ndarray | Image.Image,
    bbox: BBox,
    padding: int = 2
) -> np.ndarray | None:
    """
    이미지에서 bbox 영역을 crop합니다.
    
    Args:
        image: 원본 이미지
        bbox: crop할 BBox
        padding: 여백 (기본 2px)
        
    Returns:
        crop된 이미지 (실패 시 None)
    """
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    
    # 좌표 계산 (padding 적용, boundary 체크)
    x1 = max(0, int(bbox.x1) - padding)
    y1 = max(0, int(bbox.y1) - padding)
    x2 = min(w, int(bbox.x2) + padding)
    y2 = min(h, int(bbox.y2) + padding)
    
    # 유효한 영역인지 체크
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Crop
    cropped = image[y1:y2, x1:x2].copy()
    
    return cropped


def make_header_image(
    image: np.ndarray | Image.Image,
    table_bbox: BBox,
    margin_px: int = 2
) -> np.ndarray | None:
    """
    Table bbox 위쪽 영역을 Header 이미지로 crop합니다.
    (사용자가 직접 입력할 때 보여줄 용도)
    
    Args:
        image: 원본 이미지
        table_bbox: 선택된 Table의 BBox
        margin_px: Table 상단과 Header 하단 사이 여백 (기본 2px)
        
    Returns:
        Header 이미지 (crop 불가능하면 None)
    """
    # PIL Image를 numpy array로 변환
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    h, w = image.shape[:2]
    
    # cut_y 계산: Table 상단 - margin
    cut_y = max(0, int(table_bbox.y_top) - margin_px)
    
    # Header 영역이 너무 작으면 None 반환
    if cut_y <= 10:  # 최소 10px 이상
        return None
    
    # Header crop: [0:cut_y, 0:W]
    header_image = image[0:cut_y, 0:w].copy()
    
    return header_image


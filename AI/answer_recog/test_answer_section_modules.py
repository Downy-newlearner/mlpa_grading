"""
test_answer_section_modules.py - Answer 영역 Crop 모듈 단위 테스트

테스트 대상:
1. Table Detection (PP-DocLayout)
2. Column Separator Detection (X축 Projection Profile)
3. Answer Section Crop

실행:
    python test_answer_section_modules.py --image <이미지경로> [--output-dir <출력디렉토리>]
"""

import os
import sys
import cv2
import numpy as np
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports for modules
layout_model = None


@dataclass
class ModuleTestResult:
    """모듈 테스트 결과"""
    module_name: str
    success: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def passed_count(self):
        return sum(1 for v in self.checks.values() if v)
    
    @property
    def total_count(self):
        return len(self.checks)
    
    def to_dict(self):
        return {
            "module_name": self.module_name,
            "success": self.success,
            "passed": f"{self.passed_count}/{self.total_count}",
            "checks": self.checks,
            "metrics": self.metrics,
            "errors": self.errors,
            "warnings": self.warnings
        }


def load_model():
    """Layout 모델 로드 (lazy)"""
    global layout_model
    if layout_model is None:
        from paddlex import create_model
        print("[INFO] Loading PP-DocLayout_plus-L model...")
        layout_model = create_model("PP-DocLayout_plus-L")
        print("[INFO] Model loaded successfully")
    return layout_model


# =============================================================================
# Module 1: Table Detection Test
# =============================================================================

def test_table_detection(image: np.ndarray, output_dir: str) -> ModuleTestResult:
    """
    Module 1: Table Detection 테스트
    
    체크리스트:
    - Table bbox 검출 여부
    - Confidence score (≥ 0.5)
    - Table 면적 비율 (≥ 10%)
    - Table bbox 유효성
    """
    result = ModuleTestResult(module_name="Table Detection")
    
    try:
        from id_recog.layout import detect_all_bboxes, get_table_boxes
        
        model = load_model()
        h, w = image.shape[:2]
        image_area = h * w
        
        # 1. Layout detection
        all_boxes = detect_all_bboxes(image, model)
        result.metrics["total_boxes"] = len(all_boxes)
        
        # 2. Table bbox 필터링
        table_boxes = get_table_boxes(all_boxes)
        result.metrics["table_boxes_count"] = len(table_boxes)
        
        # Check 1: Table이 검출되었는가?
        result.checks["table_detected"] = len(table_boxes) >= 1
        
        if len(table_boxes) == 0:
            result.errors.append("No table detected in the image")
            result.success = False
            return result
        
        # 3. 가장 큰 테이블 선택
        largest_table = max(table_boxes, key=lambda b: b.bbox.area)
        table_bbox = largest_table.bbox
        
        result.metrics["table_bbox"] = {
            "x1": int(table_bbox.x1),
            "y1": int(table_bbox.y1),
            "x2": int(table_bbox.x2),
            "y2": int(table_bbox.y2),
            "width": int(table_bbox.x2 - table_bbox.x1),
            "height": int(table_bbox.y2 - table_bbox.y1),
            "area": int(table_bbox.area)
        }
        result.metrics["table_confidence"] = float(largest_table.score)
        
        # Check 2: Confidence score ≥ 0.5
        result.checks["confidence_adequate"] = largest_table.score >= 0.5
        if largest_table.score < 0.5:
            result.warnings.append(f"Low confidence: {largest_table.score:.3f}")
        
        # Check 3: Table 면적 비율 ≥ 10%
        area_ratio = table_bbox.area / image_area
        result.metrics["table_area_ratio"] = f"{area_ratio * 100:.1f}%"
        result.checks["area_ratio_adequate"] = area_ratio >= 0.10
        if area_ratio < 0.10:
            result.warnings.append(f"Table area too small: {area_ratio * 100:.1f}%")
        
        # Check 4: Table bbox 유효성 (좌표가 이미지 범위 내)
        valid_coords = (
            0 <= table_bbox.x1 < w and
            0 <= table_bbox.y1 < h and
            table_bbox.x1 < table_bbox.x2 <= w and
            table_bbox.y1 < table_bbox.y2 <= h
        )
        result.checks["bbox_valid"] = valid_coords
        
        # 시각화 저장
        vis_image = image.copy()
        cv2.rectangle(
            vis_image,
            (int(table_bbox.x1), int(table_bbox.y1)),
            (int(table_bbox.x2), int(table_bbox.y2)),
            (0, 255, 0), 3
        )
        cv2.putText(
            vis_image,
            f"Table (conf: {largest_table.score:.2f})",
            (int(table_bbox.x1), int(table_bbox.y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        vis_path = os.path.join(output_dir, "m1_table_detection.jpg")
        cv2.imwrite(vis_path, vis_image)
        result.metrics["visualization"] = vis_path
        
        result.success = all(result.checks.values())
        
    except Exception as e:
        result.errors.append(f"Exception: {str(e)}")
        result.success = False
    
    return result


# =============================================================================
# Module 2: Column Separator Detection Test
# =============================================================================

def test_column_separator_detection(image: np.ndarray, output_dir: str) -> ModuleTestResult:
    """
    Module 2: Column Separator Detection 테스트
    
    체크리스트:
    - X축 projection profile 계산 여부
    - 세로선 peak 검출 여부
    - Peak 개수 (최소 2개)
    - Answer column separator 검출 여부
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    result = ModuleTestResult(module_name="Column Separator Detection")
    
    try:
        from find_answer_section import (
            compute_vertical_lines_profile,
            find_vertical_line_peaks,
            find_last_column_separator,
            deskew_image
        )
        from id_recog.layout import detect_all_bboxes, get_table_boxes, crop_bbox
        
        model = load_model()
        
        # Table crop 먼저 수행
        all_boxes = detect_all_bboxes(image, model)
        table_boxes = get_table_boxes(all_boxes)
        
        if len(table_boxes) == 0:
            result.errors.append("Table not detected - cannot test column separator")
            result.success = False
            return result
        
        largest_table = max(table_boxes, key=lambda b: b.bbox.area)
        table_image = crop_bbox(image, largest_table.bbox, padding=0)
        
        # Deskew
        table_image, rotation_angle = deskew_image(table_image)
        result.metrics["rotation_angle"] = f"{rotation_angle:.2f}°"
        
        table_h, table_w = table_image.shape[:2]
        result.metrics["table_size"] = f"{table_w} x {table_h}"
        
        # 1. X축 projection profile 계산
        x_profile = compute_vertical_lines_profile(table_image)
        
        # Check 1: Profile 계산됨
        result.checks["profile_computed"] = len(x_profile) == table_w
        result.metrics["profile_length"] = len(x_profile)
        
        # 2. 세로선 peak 검출
        peaks = find_vertical_line_peaks(x_profile)
        result.metrics["peaks_count"] = len(peaks)
        result.metrics["peaks_positions"] = [int(p) for p in peaks]
        
        # Check 2: Peak 검출됨
        result.checks["peaks_detected"] = len(peaks) >= 1
        
        # Check 3: Peak 개수 적절 (최소 2개 - 좌/우 경계)
        result.checks["peaks_count_adequate"] = len(peaks) >= 2
        if len(peaks) < 2:
            result.warnings.append(f"Low peak count: {len(peaks)}")
        
        # 3. Answer column separator 찾기
        answer_column_x = find_last_column_separator(table_image)
        
        # Check 4: Answer column separator 검출
        result.checks["answer_separator_found"] = answer_column_x is not None
        if answer_column_x is not None:
            result.metrics["answer_column_x"] = int(answer_column_x)
            result.metrics["answer_column_ratio"] = f"{(table_w - answer_column_x) / table_w * 100:.1f}%"
        else:
            result.warnings.append("Answer column separator not found - fallback will be used")
        
        # Check 5: Answer column 너비 적절 (5% ~ 50%)
        if answer_column_x is not None:
            answer_width_ratio = (table_w - answer_column_x) / table_w
            result.checks["answer_width_reasonable"] = 0.05 <= answer_width_ratio <= 0.50
            if not (0.05 <= answer_width_ratio <= 0.50):
                result.warnings.append(f"Unusual answer column width: {answer_width_ratio * 100:.1f}%")
        else:
            result.checks["answer_width_reasonable"] = False
        
        # 시각화 저장
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Table 이미지 + Peaks
        if len(table_image.shape) == 2:
            axes[0].imshow(table_image, cmap='gray')
        else:
            axes[0].imshow(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Table Image (Deskewed by {rotation_angle:.2f}°)")
        
        for i, peak in enumerate(peaks):
            color = 'blue' if i < len(peaks) - 1 else 'red'
            axes[0].axvline(x=peak, color=color, linestyle='--', alpha=0.7, label=f'Peak {i+1}' if i < 3 else None)
        
        if answer_column_x is not None:
            axes[0].axvline(x=answer_column_x, color='green', linewidth=2, label='Answer Column Start')
        
        axes[0].legend(loc='upper right')
        
        # X축 Projection Profile
        axes[1].plot(x_profile, label='Vertical Lines Profile')
        axes[1].set_title("X-axis Projection Profile (Morphological)")
        axes[1].set_xlabel("X coordinate")
        axes[1].set_ylabel("Intensity")
        
        for peak in peaks:
            axes[1].axvline(x=peak, color='blue', linestyle='--', alpha=0.5)
        
        if answer_column_x is not None:
            axes[1].axvline(x=answer_column_x, color='green', linewidth=2, label='Answer Column Start')
        
        axes[1].legend()
        
        plt.tight_layout()
        vis_path = os.path.join(output_dir, "m2_column_separator.jpg")
        plt.savefig(vis_path, dpi=150)
        plt.close()
        result.metrics["visualization"] = vis_path
        
        # Table crop 이미지도 저장
        table_path = os.path.join(output_dir, "m2_table_crop.jpg")
        cv2.imwrite(table_path, table_image)
        result.metrics["table_crop"] = table_path
        
        result.success = all(result.checks.values())
        
    except Exception as e:
        import traceback
        result.errors.append(f"Exception: {str(e)}")
        result.errors.append(traceback.format_exc())
        result.success = False
    
    return result


# =============================================================================
# Module 3: Answer Section Crop Test
# =============================================================================

def test_answer_section_crop(image: np.ndarray, output_dir: str) -> ModuleTestResult:
    """
    Module 3: Answer Section Crop 테스트
    
    체크리스트:
    - Crop 성공 여부
    - Answer section 이미지 유효성
    - 문제번호 컬럼 미포함 (시각적 확인 필요)
    - Crop 방법 (projection_profile vs fallback)
    - 헤더 포함 여부 확인
    """
    result = ModuleTestResult(module_name="Answer Section Crop")
    
    try:
        from find_answer_section import find_answer_section
        
        model = load_model()
        
        # Answer section 찾기
        section_result = find_answer_section(image, model)
        
        # Check 1: Crop 성공
        result.checks["crop_success"] = section_result.success
        
        if not section_result.success:
            result.errors.append(f"Crop failed: {section_result.meta}")
            result.success = False
            return result
        
        answer_image = section_result.answer_section_image
        answer_h, answer_w = answer_image.shape[:2]
        
        result.metrics["answer_section_size"] = f"{answer_w} x {answer_h}"
        result.metrics["answer_column_x_start"] = section_result.answer_column_x_start
        result.metrics["crop_method"] = section_result.meta.get("answer_column_method", "unknown")
        result.metrics["rotation_angle"] = section_result.rotation_angle
        
        # Check 2: Answer section 이미지 유효성
        result.checks["image_valid"] = answer_image is not None and answer_w > 0 and answer_h > 0
        
        # Check 3: Answer column 너비 충분 (≥ 10px)
        result.checks["width_adequate"] = answer_w >= 10
        
        # Check 4: Crop 방법 확인
        crop_method = section_result.meta.get("answer_column_method", "unknown")
        result.checks["method_is_projection"] = crop_method == "projection_profile"
        if crop_method == "fallback_ratio":
            result.warnings.append("Using fallback ratio instead of projection profile")
        
        # Check 5: Answer/Table 너비 비율 합리적 (5% ~ 50%)
        table_w = section_result.table_image.shape[1] if section_result.table_image is not None else 0
        if table_w > 0:
            ratio = answer_w / table_w
            result.metrics["answer_table_ratio"] = f"{ratio * 100:.1f}%"
            result.checks["ratio_reasonable"] = 0.05 <= ratio <= 0.50
        else:
            result.checks["ratio_reasonable"] = False
        
        # 헤더 체크 (첫 번째 행에 "Answer" 텍스트가 있는지 확인)
        # 상단 10% 영역을 헤더 후보로
        header_region_h = int(answer_h * 0.1)
        if header_region_h < 50:
            header_region_h = min(50, answer_h)
        result.metrics["header_region_height"] = header_region_h
        result.warnings.append(f"[MANUAL CHECK] Verify header row ('Answer') is present in top {header_region_h}px")
        
        # 시각화 저장
        # Answer section
        answer_path = os.path.join(output_dir, "m3_answer_section.jpg")
        cv2.imwrite(answer_path, answer_image)
        result.metrics["answer_section_path"] = answer_path
        
        # Table with answer column highlighted
        if section_result.table_image is not None:
            table_vis = section_result.table_image.copy()
            x_start = section_result.answer_column_x_start
            cv2.line(table_vis, (x_start, 0), (x_start, table_vis.shape[0]), (0, 0, 255), 2)
            cv2.putText(
                table_vis,
                f"Answer Column Start (x={x_start})",
                (x_start + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            table_vis_path = os.path.join(output_dir, "m3_table_with_separator.jpg")
            cv2.imwrite(table_vis_path, table_vis)
            result.metrics["table_visualization"] = table_vis_path
        
        result.success = all(result.checks.values())
        
    except Exception as e:
        import traceback
        result.errors.append(f"Exception: {str(e)}")
        result.errors.append(traceback.format_exc())
        result.success = False
    
    return result


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests(image_path: str, output_dir: str) -> Dict[str, Any]:
    """모든 모듈 테스트 실행"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Failed to load image: {image_path}"}
    
    print(f"[INFO] Image loaded: {image_path}")
    print(f"[INFO] Image size: {image.shape[1]} x {image.shape[0]}")
    print(f"[INFO] Output directory: {output_dir}")
    print("=" * 60)
    
    results = {
        "test_timestamp": datetime.now().isoformat(),
        "input_image": image_path,
        "image_size": f"{image.shape[1]} x {image.shape[0]}",
        "output_dir": output_dir,
        "modules": {}
    }
    
    # Module 1: Table Detection
    print("\n[TEST 1/3] Table Detection...")
    m1_result = test_table_detection(image, output_dir)
    results["modules"]["table_detection"] = m1_result.to_dict()
    print(f"  Result: {'✓ PASS' if m1_result.success else '✗ FAIL'}")
    print(f"  Checks: {m1_result.passed_count}/{m1_result.total_count}")
    for check, passed in m1_result.checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
    if m1_result.warnings:
        for w in m1_result.warnings:
            print(f"    ⚠ {w}")
    
    # Module 2: Column Separator Detection
    print("\n[TEST 2/3] Column Separator Detection...")
    m2_result = test_column_separator_detection(image, output_dir)
    results["modules"]["column_separator"] = m2_result.to_dict()
    print(f"  Result: {'✓ PASS' if m2_result.success else '✗ FAIL'}")
    print(f"  Checks: {m2_result.passed_count}/{m2_result.total_count}")
    for check, passed in m2_result.checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
    if m2_result.warnings:
        for w in m2_result.warnings:
            print(f"    ⚠ {w}")
    
    # Module 3: Answer Section Crop
    print("\n[TEST 3/3] Answer Section Crop...")
    m3_result = test_answer_section_crop(image, output_dir)
    results["modules"]["answer_section_crop"] = m3_result.to_dict()
    print(f"  Result: {'✓ PASS' if m3_result.success else '✗ FAIL'}")
    print(f"  Checks: {m3_result.passed_count}/{m3_result.total_count}")
    for check, passed in m3_result.checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
    if m3_result.warnings:
        for w in m3_result.warnings:
            print(f"    ⚠ {w}")
    
    # Summary
    print("\n" + "=" * 60)
    all_passed = all([
        m1_result.success,
        m2_result.success,
        m3_result.success
    ])
    results["overall_success"] = all_passed
    
    total_checks = m1_result.total_count + m2_result.total_count + m3_result.total_count
    passed_checks = m1_result.passed_count + m2_result.passed_count + m3_result.passed_count
    
    print(f"[SUMMARY] Overall: {'✓ ALL PASS' if all_passed else '✗ SOME FAILED'}")
    print(f"[SUMMARY] Total checks: {passed_checks}/{total_checks}")
    
    # 결과 저장
    result_path = os.path.join(output_dir, "module_test_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Results saved to: {result_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Answer 영역 Crop 모듈 단위 테스트")
    parser.add_argument("--image", required=True, help="테스트할 이미지 경로")
    parser.add_argument("--output-dir", default="test_output/module_test", help="출력 디렉토리")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    run_all_tests(args.image, args.output_dir)


if __name__ == "__main__":
    main()

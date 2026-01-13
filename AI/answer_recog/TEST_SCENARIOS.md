# Answer 영역 Crop 테스트 시나리오 및 체크리스트

## 개요

답안인식 파이프라인의 "Answer 영역 crop" 단계를 모듈 단위로 테스트합니다.
이 단계는 3개의 하위 모듈로 구성됩니다:

1. **Table Detection**: PP-DocLayout 모델로 table bbox 검출
2. **Column Separator Detection**: X축 projection profile로 세로선 검출
3. **Answer Section Crop**: Answer 영역 crop

---

## 테스트 모듈 1: Table Detection (PP-DocLayout)

### 목적
원본 답안지 이미지에서 Table 영역을 정확하게 검출하는지 확인

### 테스트 시나리오

| ID | 시나리오 | 입력 | 기대 결과 |
|-----|---------|------|----------|
| T1.1 | 정상 답안지 - 단일 테이블 | 표준 답안지 이미지 | Table bbox 1개 검출, confidence ≥ 0.7 |
| T1.2 | 정상 답안지 - 복수 테이블 | 헤더 테이블 + 답안 테이블 | 가장 큰 테이블(답안 테이블) 선택됨 |
| T1.3 | 기울어진 답안지 (5° 이내) | 살짝 기울어진 스캔 | Table bbox 정상 검출 |
| T1.4 | 저해상도 이미지 | 72dpi 이미지 | Table 검출 or graceful failure |
| T1.5 | 테이블 없는 이미지 | 빈 종이 or 서술형만 | success=False, reason="no_table_found" |
| T1.6 | 노이즈 많은 이미지 | 얼룩/그림자 있는 스캔 | Table bbox 검출, 노이즈 영역 미포함 |

### 체크리스트

```markdown
[ ] Table bbox가 검출되었는가?
[ ] Table bbox의 confidence score가 0.5 이상인가?
[ ] 검출된 bbox가 실제 테이블 영역과 일치하는가? (IoU ≥ 0.8)
[ ] 여러 테이블이 있을 때 가장 큰 테이블이 선택되었는가?
[ ] Table bbox가 이미지 전체 면적의 10% 이상인가?
[ ] 테이블 경계가 정확한가? (±10px 허용)
```

### 테스트 코드 예시

```python
def test_table_detection():
    """Table Detection 테스트"""
    from find_answer_section import find_answer_section
    from id_recog.layout import load_layout_model
    
    # 모델 로드
    layout_model = load_layout_model()
    
    # 테스트 이미지 로드
    test_image = cv2.imread("test_input/normal_answer_sheet.jpg")
    
    # 실행
    result = find_answer_section(test_image, layout_model)
    
    # 체크리스트 검증
    assert result.meta.get("table_boxes_count", 0) >= 1, "Table이 검출되지 않음"
    assert result.table_bbox is not None, "Table bbox가 None"
    assert result.meta["table_bbox"]["score"] >= 0.5, "Table confidence가 낮음"
    
    # 시각화 저장
    save_visualization(test_image, result.table_bbox, "test_output/t1_table_detection.jpg")
```

---

## 테스트 모듈 2: Column Separator Detection (X축 Projection Profile)

### 목적
Table 이미지에서 X축 projection profile을 사용하여 세로선(column separator)을 검출하는지 확인

### 테스트 시나리오

| ID | 시나리오 | 입력 | 기대 결과 |
|-----|---------|------|----------|
| T2.1 | 표준 2-column 테이블 | 문제번호 + 답안 컬럼 | 2개의 세로선(좌/우 경계) + 1개의 separator 검출 |
| T2.2 | 표준 3-column 테이블 | 문제번호 + 내용 + 답안 | 각 컬럼 separator 검출 |
| T2.3 | 세로선이 약한 테이블 | 연한 선 or 점선 | Fallback으로 처리 or 세로선 검출 |
| T2.4 | 세로선 없는 테이블 | 가로선만 있는 테이블 | Fallback (우측 N% crop) |
| T2.5 | 기울기 보정 후 테이블 | Deskew 적용된 이미지 | 정확한 세로선 검출 |
| T2.6 | 텍스트 밀집 영역 | 답안에 긴 텍스트 | 세로선과 텍스트 영역 구분 |

### 체크리스트

```markdown
[ ] X축 projection profile이 계산되었는가?
[ ] Morphological 세로선 추출이 동작하는가?
[ ] 세로선 peak가 정확하게 검출되었는가?
[ ] 검출된 peak 수가 예상과 일치하는가? (테이블 구조에 따라)
[ ] 좌측 테이블 경계가 올바르게 제외되었는가?
[ ] 우측 테이블 경계가 올바르게 제외되었는가?
[ ] Answer column separator (마지막 내부 세로선)가 정확한가?
[ ] Fallback이 발동한 경우, 적절한 비율로 crop 되었는가?
```

### 테스트 코드 예시

```python
def test_column_separator_detection():
    """Column Separator Detection 테스트"""
    from find_answer_section import (
        compute_vertical_lines_profile,
        find_vertical_line_peaks,
        find_last_column_separator,
        visualize_projection_profile
    )
    
    # Table 이미지 로드 (미리 crop된 것)
    table_image = cv2.imread("test_input/table_crop.jpg")
    
    # X축 projection profile 계산
    x_profile = compute_vertical_lines_profile(table_image)
    
    # Peak 검출
    peaks = find_vertical_line_peaks(x_profile)
    
    # Answer column separator 찾기
    answer_column_x = find_last_column_separator(table_image)
    
    # 체크리스트 검증
    assert len(x_profile) == table_image.shape[1], "Profile 길이 불일치"
    assert len(peaks) >= 2, f"Peak 부족: {len(peaks)}개"
    assert answer_column_x is not None, "Answer column x가 None"
    
    # 시각화 저장
    visualize_projection_profile(
        table_image, 
        save_path="test_output/t2_projection_profile.jpg",
        answer_column_x=answer_column_x
    )
    
    print(f"[T2] Peaks: {peaks}")
    print(f"[T2] Answer column X: {answer_column_x}")
```

### 디버깅 포인트

```python
# 디버깅: Profile과 peaks 시각화
import matplotlib.pyplot as plt

def debug_x_profile(table_image, save_path):
    x_profile = compute_vertical_lines_profile(table_image)
    peaks = find_vertical_line_peaks(x_profile)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Table 이미지
    axes[0].imshow(table_image)
    axes[0].set_title("Table Image")
    for peak in peaks:
        axes[0].axvline(x=peak, color='red', linestyle='--', alpha=0.7)
    
    # X축 Projection Profile
    axes[1].plot(x_profile)
    axes[1].set_title("X-axis Projection Profile (Morphological)")
    for peak in peaks:
        axes[1].axvline(x=peak, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## 테스트 모듈 3: Answer Section Crop

### 목적
검출된 column separator를 기준으로 Answer 영역만 정확하게 crop하는지 확인

### 테스트 시나리오

| ID | 시나리오 | 입력 | 기대 결과 |
|-----|---------|------|----------|
| T3.1 | 정상 crop | Answer column separator 검출됨 | Answer 영역만 crop, 문제번호 미포함 |
| T3.2 | 경계 정확도 | 다양한 테이블 | 왼쪽에 세로선 미포함, 오른쪽 여백 적절 |
| T3.3 | Fallback crop | 세로선 미검출 | 우측 15% (기본값) crop |
| T3.4 | 너비 검증 | 정상 테이블 | Answer column 너비 ≥ 10px |
| T3.5 | 헤더 포함 여부 | "Answer" 헤더 있는 테이블 | **헤더 행 포함됨 (현재) → 추후 제외 필요** |
| T3.6 | 이미지 품질 | Crop 후 이미지 | 원본 해상도 유지, blur/artifact 없음 |

### 체크리스트

```markdown
[ ] Answer section crop이 성공했는가? (success=True)
[ ] Crop된 이미지에 문제번호 컬럼이 포함되지 않았는가?
[ ] Crop된 이미지에 답안 내용이 포함되어 있는가?
[ ] 왼쪽 경계에 세로선이 최소화되어 있는가?
[ ] 우측 여백이 적절한가? (테이블 경계 포함 or 제외)
[ ] Answer column 너비가 충분한가? (≥ 10px)
[ ] 메타데이터에 answer_column_x_start가 기록되었는가?
[ ] 메타데이터에 answer_column_method가 기록되었는가?
[ ] Crop 방법이 올바른가? (projection_profile or fallback_ratio)
```

### 테스트 코드 예시

```python
def test_answer_section_crop():
    """Answer Section Crop 테스트"""
    from find_answer_section import find_answer_section
    from id_recog.layout import load_layout_model
    
    layout_model = load_layout_model()
    test_image = cv2.imread("test_input/normal_answer_sheet.jpg")
    
    result = find_answer_section(test_image, layout_model)
    
    # 기본 성공 체크
    assert result.success, f"Crop 실패: {result.meta}"
    assert result.answer_section_image is not None, "Answer section image가 None"
    
    # 크기 체크
    answer_h, answer_w = result.answer_section_image.shape[:2]
    assert answer_w >= 10, f"Answer column 너비 부족: {answer_w}"
    
    # 메타데이터 체크
    assert "answer_column_x_start" in result.meta, "x_start 메타 없음"
    assert "answer_column_method" in result.meta, "method 메타 없음"
    
    # 이미지 저장
    cv2.imwrite("test_output/t3_answer_section.jpg", result.answer_section_image)
    
    print(f"[T3] Answer section size: {answer_w} x {answer_h}")
    print(f"[T3] Crop method: {result.meta['answer_column_method']}")
    print(f"[T3] Answer column X start: {result.answer_column_x_start}")
```

---

## 통합 테스트 시나리오

### End-to-End 테스트

| ID | 시나리오 | 입력 | 기대 결과 |
|-----|---------|------|----------|
| E2E-1 | 전체 파이프라인 | 원본 답안지 이미지 | Answer section crop → Row segmentation → OCR 성공 |
| E2E-2 | 다양한 스캔 환경 | 스캐너/핸드폰 촬영 | 모든 환경에서 crop 성공 |
| E2E-3 | 오류 복구 | Table 미검출 시 | Graceful failure, 상세 에러 메시지 |

---

## 현재 발견된 문제점 (2026-01-14 테스트 결과)

### 📊 테스트 결과 요약

| 모듈 | 결과 | 통과 | 주요 지표 |
|------|------|------|----------|
| Table Detection | ✅ PASS | 4/4 | Confidence: 98.9%, Area: 58.8% |
| Column Separator Detection | ❌ FAIL | 4/5 | Answer width: 81% (비정상) |
| Answer Section Crop | ❌ FAIL | 4/5 | 문제번호 컬럼까지 포함됨 |

---

### 🔴 문제 1: Column Separator 검출 위치 오류 (Critical)

**증상**: 
- Answer column이 테이블의 81%를 차지함 (정상: 10~30%)
- Answer section에 문제번호 컬럼 `(1), (2), (3)...`이 포함됨
- X축 Projection peaks가 테이블 **왼쪽**에서만 검출됨: `[21, 204, 208, 308]`

**원인 분석**:

```
테이블 구조:
┌─────────────────────────────────────────────────────┐
│  문제번호 │ 문제 내용/설명              │ Answer   │
│   (1)    │ ...                        │    0     │
│   (2)    │ ...                        │    0     │
│   ...    │ ...                        │    /     │
└─────────────────────────────────────────────────────┘
    ↑         ↑                            ↑
  x=21      x=308                        x≈1500?
  (검출됨)   (이것을 Answer로 잘못 선택)   (검출 안됨)
```

1. **Morphological 세로선 추출**이 **왼쪽 영역**의 세로선만 검출
2. **오른쪽 Answer 컬럼 separator**(실제 목표)가 검출되지 않음
3. `find_last_column_separator` 함수가 "내부의 마지막 peak" = x=308을 선택
4. 결과적으로 **x=311부터 crop** → 81%가 Answer로 crop됨

**예상 원인**:
- `compute_vertical_lines_profile()`의 morphological 연산이 특정 조건에서만 세로선 검출
- 세로선이 너무 연하거나 스캔 품질 문제
- 테이블 오른쪽 끝의 세로선이 없거나 너무 얇음

**해결 방안**:
1. **우측 영역 검색 강화**: `right_search_ratio` 파라미터를 더 작게 (0.3 → 0.2)
2. **Fallback 전략 개선**: Peak가 오른쪽에 없으면 테이블 우측 N%를 crop
3. **Morphological 파라미터 조정**: kernel 크기, threshold 조정
4. **Direct pixel analysis**: X축 우측 영역에서 직접 수직 밀도 분석

---

### 🟡 문제 2: Row 1이 "Answer" 헤더만 포함

**증상**: 
- `result.json`의 Question 1 결과가 `"AnSVe"`로 인식됨 (실제로 "Answer" 헤더 텍스트)
- 첫 번째 Row가 "Answer" 헤더 행임

**원인**: 
- Row segmentation에서 헤더 행이 답안 행으로 처리됨
- Answer section crop 시 헤더 행이 제외되지 않음

**해결 방안**:
- Row segmentation에서 첫 번째 행 (헤더) 제외
- 또는 Answer section crop 단계에서 상단 N픽셀 제거
- 또는 OCR 전처리에서 "Answer" 텍스트 인식 시 해당 Row skip

---

### 🟡 문제 3: 다수의 빈 답안 인식

**증상**: 
- 대부분의 답안이 `""` (빈 문자열)로 인식
- Confidence가 0.0인 경우 다수

**원인 추정**:
1. Answer section crop이 잘못됨 → 문제번호도 포함되어 OCR 혼란
2. Row segmentation 오류로 빈 영역 crop
3. OCR 모델 자체의 한계

**확인 필요**:
- Row 이미지들 개별 검증
- OCR 모델 출력 raw text 확인

---

### 📋 디버깅 체크리스트

#### Column Separator Detection 디버깅

```bash
# 1. Table crop 이미지 확인
open test_output/module_test/m2_table_crop.jpg

# 2. X축 Profile 시각화 확인
open test_output/module_test/m2_column_separator.jpg

# 3. 수동으로 세로선 위치 확인
# → 테이블 이미지에서 Answer 컬럼을 구분하는 세로선이 어디에 있는가?
# → 그 위치의 x좌표는?
```

#### 핵심 질문

1. **테이블 오른쪽에 세로선이 있는가?**
   - 있다면: morphological 연산 파라미터 조정 필요
   - 없다면: 다른 방식의 column detection 필요 (예: text alignment 분석)

2. **테이블 구조가 2-column인가 3-column인가?**
   - 2-column: 문제번호 | Answer
   - 3-column: 문제번호 | 내용 | Answer

3. **모든 답안지가 동일한 구조인가?**
   - 동일: 고정 비율 fallback 가능
   - 다양: adaptive detection 필요

---

## 테스트 실행 방법

### 개별 모듈 테스트

```bash
cd /home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/answer_recog

# Module 1: Table Detection
python -c "
from test_find_answer_section import test_table_detection
test_table_detection()
"

# Module 2: Column Separator Detection
python -c "
from test_find_answer_section import test_column_detection
test_column_detection()
"

# Module 3: Answer Section Crop
python -c "
from test_find_answer_section import test_answer_crop
test_answer_crop()
"
```

### 통합 테스트

```bash
python test_full_pipeline.py --image test_input/sample.jpg --structure answer_structure.json
```

### 시각화 출력 확인

```bash
# 생성된 디버깅 이미지 확인
ls -la test_output/pipeline_test/
# 01_answer_section.jpg - Answer 영역 crop 결과
# 02_row_*.jpg - 각 Row 분할 결과
```

---

## 개선 우선순위

1. **[HIGH]** 헤더 행 제외 로직 추가
2. **[HIGH]** Row segmentation 정확도 검증
3. **[MEDIUM]** X축 projection profile 파라미터 튜닝
4. **[LOW]** 다양한 테이블 형식 지원

---

## 부록: 테스트 입력 이미지 요구사항

| 항목 | 요구사항 |
|------|---------|
| 해상도 | 최소 150 DPI, 권장 300 DPI |
| 형식 | JPEG, PNG |
| 크기 | 최소 1000x1000 px |
| 기울기 | ±5° 이내 |
| 선명도 | 텍스트가 육안으로 읽을 수 있을 것 |

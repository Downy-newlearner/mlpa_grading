# 답안 인식 파이프라인 설계 문서

## 1. 개요

이 문서는 OCR 기반 **Answer Recognition Pipeline**의 설계를 정의합니다.  
목표: 답안지 이미지에서 각 문제별 답안을 자동으로 추출하고, 정답 메타데이터와 비교하여 채점합니다.

---

## 2. 입력 데이터

### 2.1 답안지 이미지
- **Case 1**: 주 문제(Main Question)만 가로선으로 구분됨
  - 꼬리문제(Sub-Question)는 가로선 없이 같은 Row 내에 존재
  - Y축 Projection Profile로 꼬리문제 분리 필요
  
- **Case 2**: 모든 문제가 가로선으로 구분됨
  - 각 Row = 1개 문제
  - Morphological 가로선 탐지로 충분

### 2.2 정답지 메타데이터 (JSON)
```json
{
  "exam_code": "AI_2023_MID",
  "questions": [
    {
      "question_number": 1,
      "sub_question_count": 3,
      "scoring_type": "objective",  // "binary", "short_answer", "objective", "others"
      "correct_answer": ["1", "2", "3"],
      "points": [2, 2, 2]
    },
    {
      "question_number": 2,
      "sub_question_count": 0,
      "scoring_type": "short_answer",
      "correct_answer": ["CNN"],
      "points": [5]
    }
  ]
}
```

---

## 3. 핵심 로직 및 파이프라인

### 3.1 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              ANSWER RECOGNITION SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────┐                                                 │
│  │            입력 (Input)              │                                                 │
│  ├─────────────────────────────────────┤                                                 │
│  │  • 답안 JSON (정답 메타데이터)        │                                                 │
│  │  • 이미지 from S3                    │                                                 │
│  │    key: original/{exam_code}/       │                                                 │
│  │         {학번}/{파일명}              │                                                 │
│  └──────────────────┬──────────────────┘                                                 │
│                     │                                                                    │
│                     │  N장의 답안지 이미지 (학생 수만큼)                                    │
│                     ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐            │
│  │                    답안인식 파이프라인 (Per Image)                        │            │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │            │
│  │  │  1. Layout Detection    → PP-DocLayout (Table 탐지)             │    │            │
│  │  │  2. Answer Section Crop → X-Axis Projection (Answer Column)     │    │            │
│  │  │  3. Row Segmentation    → Morphological + Y-Projection          │    │            │
│  │  │  4. ROI Extraction      → 각 문제별 답안 영역 Crop               │    │            │
│  │  │  5. Answer Extraction   → OCR / Mark Detection                  │    │            │
│  │  └─────────────────────────────────────────────────────────────────┘    │            │
│  └────────────────────────────────┬────────────────────────────────────────┘            │
│                                   │                                                      │
│          ┌────────────────────────┴────────────────────────┐                             │
│          ▼                                                 ▼                             │
│  ┌───────────────────────┐                    ┌────────────────────────────────┐        │
│  │  [confidence ≥ 0.7]   │                    │  [confidence < 0.7] = Fallback │        │
│  │  정상 인식 결과         │                    │                                │        │
│  │  → 메모리에 저장        │                    │  ROI 이미지 S3 업로드            │        │
│  └───────────────────────┘                    │  key: answer/{exam_code}/      │        │
│                                               │       {학번}/{문제번호}/        │        │
│                                               │       {꼬리문제번호}/{파일명}    │        │
│                                               └────────────────────────────────┘        │
│                                                                                          │
│  ════════════════════════════════════════════════════════════════════════════════════    │
│                         모든 학생 처리 완료 (N장 모두 처리)                                │
│  ════════════════════════════════════════════════════════════════════════════════════    │
│                                                                                          │
│                     ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐            │
│  │                    사용자 Fallback 작업 (BE/Frontend)                    │            │
│  │  • 사용자가 S3의 ROI 이미지를 확인하고 정답 입력                          │            │
│  │  • 수정된 답안을 JSON으로 전송                                           │            │
│  └────────────────────────────────┬────────────────────────────────────────┘            │
│                                   │                                                      │
│                     ▼  "Fallback 완료" 신호 수신 (미정)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐            │
│  │                         채점 단계 (Grading)                              │            │
│  │  • Fallback 수정값 병합                                                  │            │
│  │  • 답안 JSON과 비교하여 채점                                             │            │
│  │  • 점수 계산                                                             │            │
│  └────────────────────────────────┬────────────────────────────────────────┘            │
│                                   │                                                      │
│                                   ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐            │
│  │                         결과 전송 (Output)                               │            │
│  │  • 최종 JSON 생성                                                        │            │
│  │  • 백엔드에 POST (SQS/HTTP)                                              │            │
│  └─────────────────────────────────────────────────────────────────────────┘            │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 단일 이미지 파이프라인 상세

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   SINGLE IMAGE ANSWER RECOGNITION PIPELINE                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  [Image Input] ──► [Layout Detection] ──► [Answer Section Crop]               │
│        │                  │                        │                          │
│        ▼                  ▼                        ▼                          │
│  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────────┐              │
│  │ Deskewing   │   │ PP-DocLayout    │   │ X-Axis Projection   │              │
│  │ (기울기 보정)│   │ (Table Detect)  │   │ (Answer Column Crop)│              │
│  └─────────────┘   └─────────────────┘   └─────────────────────┘              │
│                                                   │                          │
│                                                   ▼                          │
│                           ┌──────────────────────────────────────┐           │
│                           │         ROW SEGMENTATION              │           │
│                           ├──────────────────────────────────────┤           │
│                           │ Case 1: 주 문제 가로선 + Y-Projection │           │
│                           │ Case 2: 전체 가로선 기반              │           │
│                           └──────────────────────────────────────┘           │
│                                                   │                          │
│                                                   ▼                          │
│                           ┌──────────────────────────────────────┐           │
│                           │         ROI EXTRACTION                │           │
│                           ├──────────────────────────────────────┤           │
│                           │ 각 Row에서 답안 영역(ROI)를 Crop       │           │
│                           │ → Fallback용 이미지로 사용            │           │
│                           └──────────────────────────────────────┘           │
│                                                   │                          │
│                                                   ▼                          │
│                           ┌──────────────────────────────────────┐           │
│                           │        ANSWER EXTRACTION              │           │
│                           ├──────────────────────────────────────┤           │
│                           │ binary: ○/×/체크마크 검출             │           │
│                           │ objective: 숫자/원문자 OCR            │           │
│                           │ short_answer: 텍스트 OCR              │           │
│                           │ others: 미채점 (skip)                 │           │
│                           └──────────────────────────────────────┘           │
│                                                   │                          │
│                    ┌──────────────────────────────┴──────────────────────┐   │
│                    ▼                                                     ▼   │
│           [confidence ≥ 0.7]                                [confidence < 0.7]
│                    │                                                     │   │
│                    ▼                                                     ▼   │
│           ┌───────────────┐                              ┌───────────────────┐
│           │ 정상 인식 결과  │                              │ Fallback 처리     │
│           │ rec_answer    │                              │ • ROI 이미지 S3   │
│           │ confidence    │                              │   업로드          │
│           └───────┬───────┘                              │ • rec_answer =   │
│                   │                                      │   "unknown"      │
│                   │                                      └─────────┬─────────┘
│                   └──────────────────┬───────────────────────────────┘       │
│                                      ▼                                       │
│                           ┌──────────────────────────────────────┐           │
│                           │         OUTPUT GENERATION             │           │
│                           │  AnswerRecognitionResult (per Q)      │           │
│                           └──────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 S3 경로 구조

| 유형 | S3 Key 패턴 | 설명 |
|------|------------|------|
| **원본 이미지** | `original/{exam_code}/{학번}/{파일명}` | 학번 인식에서 사용한 원본 |
| **Fallback ROI** | `answer/{exam_code}/{학번}/{문제번호}/{꼬리문제번호}/{파일명}` | 낮은 confidence 답안의 ROI 이미지 |

- **꼬리문제 번호**: 없으면 `0`, 있으면 `1`, `2`, `3`...
- **파일명 포맷**: `roi_q{문제번호}_s{꼬리문제번호}.jpg`

예시:
```
answer/AI_2023_MID/20201234/1/0/roi_q1_s0.jpg     # 1번 문제 (꼬리문제 없음)
answer/AI_2023_MID/20201234/3/1/roi_q3_s1.jpg     # 3번 문제의 첫 번째 꼬리문제
answer/AI_2023_MID/20201234/3/2/roi_q3_s2.jpg     # 3번 문제의 두 번째 꼬리문제
```



### 3.4 모듈별 상세 로직

#### 3.4.1 레이아웃 인식 (Layout Detection)
- **모델**: PP-DocLayout_plus-L
- **가정**: 테이블을 완벽히 인식, 가로선은 항상 명확
- **출력**: Table BBox 좌표

#### 3.4.2 Answer 섹션 추출 (Answer Section Crop)
- **방법**: X축 Projection Profile
- **로직**:
  1. Table crop 후 기울기 보정 (Deskewing)
  2. Morphological 연산으로 세로선 강조
  3. X축 Projection 계산 → Peak 탐지
  4. 마지막 세로선(Answer Column 시작점) 찾기
  5. 해당 위치부터 끝까지 Crop

#### 3.4.3 Row 분할 (Row Segmentation)

**Case 2 (모든 문제가 가로선 구분)**:
```python
def segment_rows_case2(answer_image):
    # Morphological 가로선 탐지
    separators = find_row_separators_morphological(answer_image)
    
    # 가로선 기준으로 Row 생성
    rows = create_rows_from_separators(separators)
    return rows
```

**Case 1 (주 문제만 가로선 구분 + 꼬리문제)**:
```python
def segment_rows_case1(answer_image, metadata):
    # 1단계: Morphological 가로선으로 주 문제(Main Row) 분할
    main_rows = find_row_separators_morphological(answer_image)
    
    final_rows = []
    
    for i, main_row in enumerate(main_rows):
        question_meta = metadata.questions[i]
        sub_count = question_meta.sub_question_count
        
        if sub_count <= 1:
            # 꼬리문제 없음 → 그대로 사용
            final_rows.append(main_row)
        else:
            # 꼬리문제 있음 → Y축 Projection으로 2차 분할
            sub_rows = segment_sub_questions(
                main_row.image, 
                expected_count=sub_count
            )
            final_rows.extend(sub_rows)
    
    return final_rows
```

---

## 4. Y축 Projection 기반 꼬리문제 분리 알고리즘

### 4.1 알고리즘 개요

```python
def segment_sub_questions(row_image, expected_count):
    """
    Y축 Projection Profile을 사용하여 Row 내 꼬리문제를 분리합니다.
    
    핵심 아이디어:
    - 텍스트 영역은 Y-Projection이 높음 (텍스트 픽셀 밀도 높음)
    - 텍스트 사이 공백은 Y-Projection이 낮음 (Valley)
    - expected_count 개의 sub-row로 분할해야 함
    
    알고리즘:
    1. Y축 Projection Profile 계산
    2. Smoothing (노이즈 제거)
    3. Peak 후보 추출 (각 텍스트 라인 중심)
    4. Peak 간 Valley 찾기
    5. expected_count 기반 Valley 선택
    6. Final Row 생성
    """
    
    # Step 1: Y축 Projection Profile 계산
    y_profile = compute_y_projection_profile(row_image)
    
    # Step 2: Smoothing (Gaussian Blur)
    kernel_size = max(3, row_height // 10)
    y_profile_smooth = gaussian_blur_1d(y_profile, kernel_size)
    
    # Step 3: Peak 추출
    # - min_distance: 예상 sub-row 높이의 1/3
    # - min_height_ratio: 최대값 대비 5% 이상
    expected_height = len(y_profile) / expected_count
    peaks = find_peaks(
        y_profile_smooth,
        min_distance=int(expected_height / 3),
        min_height_ratio=0.05
    )
    
    # Step 4: Peak 간 Valley 찾기
    valleys, depths = find_valleys_between_peaks(
        y_profile_smooth,
        peaks,
        min_depth_ratio=0.9  # 얕은 valley도 허용
    )
    
    # Step 5: expected_count 기반 Valley 선택
    # valleys 개수가 expected_count - 1보다 많으면 상위 N개 선택
    # valleys 개수가 부족하면 균등 분할
    final_valleys = select_valleys_by_count(
        valleys, 
        depths, 
        expected_count - 1,
        y_profile_smooth
    )
    
    # Step 6: Final Row 생성
    boundaries = [0] + final_valleys + [len(y_profile)]
    sub_rows = create_rows_from_boundaries(row_image, boundaries)
    
    return sub_rows


def select_valleys_by_count(valleys, depths, target_count, profile):
    """
    expected_count에 맞게 Valley를 선택합니다.
    
    전략:
    1. target_count == len(valleys): 전부 사용
    2. target_count < len(valleys): depth가 가장 깊은(값이 작은) N개 선택
    3. target_count > len(valleys): 균등 분할로 보충
    """
    if len(valleys) == target_count:
        return sorted(valleys)
    
    if len(valleys) > target_count:
        # Depth로 정렬 후 상위 N개 선택 (depth가 작을수록 깊은 valley)
        indexed = list(zip(valleys, depths))
        indexed.sort(key=lambda x: x[1])  # depth 오름차순
        selected = sorted([v for v, d in indexed[:target_count]])
        return selected
    
    # valleys 부족 → 균등 분할
    total_height = len(profile)
    uniform_valleys = []
    step = total_height // (target_count + 1)
    
    for i in range(1, target_count + 1):
        y = i * step
        # 가장 가까운 실제 valley 또는 local minimum 찾기
        local_y = find_local_minimum_near(profile, y, window=step // 2)
        uniform_valleys.append(local_y)
    
    return sorted(uniform_valleys)
```

### 4.2 알고리즘 시각화

```
Y-Projection Profile (가로 = Y좌표, 세로 = 밀도):

        ████                    ← 첫 번째 sub-question 텍스트
       ██████                     (Peak 1)
        ████
          ▼                     ← Valley 1 (sub-question 경계)
        ████
       ██████                   ← 두 번째 sub-question 텍스트
      ████████                    (Peak 2)
        ████
          ▼                     ← Valley 2 (sub-question 경계)
        ████
       ██████                   ← 세 번째 sub-question 텍스트
        ████                      (Peak 3)
    ──────────────────────
    0                    H
```

---

## 5. 채점 대상 타입별 처리

### 5.1 scoring_type 분류

| Type | 설명 | 처리 방법 | 출력 예시 |
|------|------|----------|----------|
| `binary` | O/X, 체크마크 | 체크/비체크 검출 | `true`/`false` |
| `objective` | 객관식 (1,2,3,4,5) | 숫자/원문자 OCR | `"3"`, `"①"` |
| `short_answer` | 단답형 텍스트 | 텍스트 OCR | `"CNN"` |
| `others` | 서술형/미채점 | Skip | `null` |

### 5.2 처리 로직 의사코드

```python
def extract_answer(row_image, scoring_type):
    """채점 타입별 답안 추출"""
    
    if scoring_type == "binary":
        # 체크마크/O/X 검출
        result = detect_binary_mark(row_image)
        return BinaryResult(
            checked=result.is_checked,
            mark_type=result.mark_type,  # "circle", "check", "x"
            confidence=result.confidence
        )
    
    elif scoring_type == "objective":
        # 숫자/원문자 OCR
        text, conf = ocr_extract(row_image)
        answer = normalize_objective_answer(text)  # "①" → "1"
        return ObjectiveResult(
            raw_text=text,
            answer=answer,
            confidence=conf
        )
    
    elif scoring_type == "short_answer":
        # 텍스트 OCR
        text, conf = ocr_extract(row_image)
        answer = clean_short_answer(text)
        return ShortAnswerResult(
            raw_text=text,
            answer=answer,
            confidence=conf
        )
    
    elif scoring_type == "others":
        # 미채점 대상
        return OthersResult(
            skipped=True,
            reason="Not a scoring target"
        )
```

---

## 6. 출력 데이터 구조

### 6.1 JSON 출력 포맷

```json
{
  "exam_code": "AI_2023_MID",
  "student_id": "20201234",
  "processed_at": "2026-01-13T22:51:28+09:00",
  "results": [
    {
      "question_number": 1,
      "sub_question_number": 1,
      "scoring_type": "objective",
      "rec_answer": "3",
      "confidence": 0.95,
      "is_correct": true,
      "points_earned": 2,
      "meta": {
        "raw_ocr_text": "③",
        "roi_bbox": [10, 20, 100, 50]
      }
    },
    {
      "question_number": 1,
      "sub_question_number": 2,
      "scoring_type": "objective",
      "rec_answer": "1",
      "confidence": 0.88,
      "is_correct": false,
      "points_earned": 0,
      "meta": {}
    },
    {
      "question_number": 2,
      "sub_question_number": null,
      "scoring_type": "others",
      "rec_answer": null,
      "confidence": null,
      "is_correct": null,
      "points_earned": null,
      "meta": {
        "skipped": true,
        "reason": "Descriptive question - not auto-graded"
      }
    }
  ],
  "summary": {
    "total_questions": 10,
    "auto_graded": 8,
    "skipped": 2,
    "correct_count": 6,
    "total_points": 50,
    "earned_points": 38
  }
}
```

---

## 7. 성능 최적화

### 7.1 병목 지점 분석

| 병목 지점 | 예상 소요 시간 | 원인 |
|----------|--------------|------|
| **PP-DocLayout 추론** | ~500ms | GPU 모델 추론 |
| **OCR (PaddleOCR)** | ~200ms/row | 다수 Row에 반복 실행 |
| **이미지 전처리** | ~50ms | CV2 연산 |
| **Y-Projection 계산** | ~10ms | 배열 연산 |

### 7.2 최적화 방안

#### 1. 배치 OCR 처리
```python
# Before: 각 Row마다 OCR 호출
for row in rows:
    result = ocr.predict(row.image)  # 200ms * N

# After: 모든 Row를 배치로 처리
row_images = [row.image for row in rows]
results = ocr.predict_batch(row_images)  # 200ms + 50ms * (N-1)
```

#### 2. 이미지 크기 최적화
```python
# Row 이미지 리사이즈 (OCR 정확도 유지하면서 속도 향상)
MAX_HEIGHT = 64
if row.height > MAX_HEIGHT:
    scale = MAX_HEIGHT / row.height
    resized = cv2.resize(row.image, None, fx=scale, fy=scale)
```

#### 3. 캐싱
```python
# Layout 모델은 한 번만 로드
@lru_cache(maxsize=1)
def get_layout_model():
    return load_pp_doclayout()

# Row segmentation 결과 캐싱 (동일 이미지 해시 기준)
row_cache = {}
```

#### 4. 병렬 처리 (Optional)
```python
from concurrent.futures import ThreadPoolExecutor

def extract_answers_parallel(rows, scoring_types):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(extract_answer, row.image, stype)
            for row, stype in zip(rows, scoring_types)
        ]
        return [f.result() for f in futures]
```

---

## 8. 예외 처리 방침

### 8.1 예외 유형 및 처리

| 예외 유형 | 원인 | 처리 방법 |
|----------|------|----------|
| **Table 미탐지** | 레이아웃 변형 | Fallback: 전체 이미지를 Answer로 간주 |
| **가로선 부족** | 스캔 품질 저하 | Y-Projection fallback |
| **OCR 실패** | 필기체, 흐림 | confidence < 0.5일 때 `unknown` 반환 |
| **Row 수 불일치** | 메타데이터 오류 | 경고 로그 + 탐지된 Row 기준 처리 |
| **scoring_type 누락** | JSON 오류 | `others`로 기본 처리 |

### 8.2 예외 처리 코드

```python
def safe_extract(row, metadata):
    try:
        result = extract_answer(row.image, metadata.scoring_type)
        return result
    except OCRException as e:
        logger.warning(f"OCR failed for Q{metadata.question_number}: {e}")
        return FallbackResult(
            rec_answer="unknown",
            confidence=0.0,
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ErrorResult(error=str(e))
```

---

## 9. 추후 통합 계획

### 9.1 현재 구조
```
mlpa_grading/AI/
├── id_recog/
│   └── app.py          # 학번 인식 FastAPI 서버
├── answer_recog/
│   ├── find_answer_section.py
│   ├── row_segmentation.py
│   └── answer_extraction.py
```

### 9.2 목표 구조
```
mlpa_grading/AI/
├── app.py              # 통합 FastAPI 서버 (학번 + 답안 인식)
├── id_recog/
│   └── student_id_pipeline.py
├── answer_recog/
│   ├── pipeline.py     # 메인 파이프라인
│   ├── find_answer_section.py
│   ├── row_segmentation.py
│   └── answer_extraction.py
└── utils/
    └── common.py
```

### 9.3 통합 API 엔드포인트

```python
# POST /process-student-answer
# - 학번 인식 + 답안 인식 동시 수행

# POST /recognize-answer (별도)
# - 답안 인식만 수행 (학번은 입력받음)

# POST /grade (채점)
# - 인식된 답안 + 정답 메타데이터 → 채점 결과
```

---

## 10. 구현 우선순위

1. **[DONE]** `find_answer_section.py` - Answer 섹션 Crop
2. **[DONE]** `row_segmentation.py` - Row 분할 (Morphological + Fallback)
3. **[IN PROGRESS]** Y축 Projection 기반 꼬리문제 분리
4. **[TODO]** 채점 타입별 Answer Extraction 고도화
5. **[TODO]** 메인 파이프라인 통합 (`pipeline.py`)
6. **[TODO]** id_recog와 통합 → 상위 `app.py`로 이동

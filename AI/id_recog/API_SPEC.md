# AI 서버 API 명세서

## 기본 정보
- **Base URL**: `http://AI_SERVER_IP:8000`
- **Content-Type**: `multipart/form-data` (파일 업로드)
- **응답 형식**: `application/json`

---

## 📌 API 엔드포인트

### 1. 출석부 업로드
```
POST /upload-roster/
```

**Request:**
- **Form Data**:
  - `file`: xlsx 파일 (multipart/form-data)

**Response (200 OK):**
```json
{
  "success": true,
  "filename": "SaS_2017.xlsx",
  "student_count": 21,
  "student_ids": ["32161086", "32131798", ...],
  "message": "21개의 학번이 성공적으로 로드되었습니다."
}
```

---

### 2. 학번 추출
```
POST /extract-student-id/
```

**Request:**
- **Form Data**:
  - `image`: 답안지 이미지 파일 (jpg, png)
  - `student_id_list` (optional): 쉼표로 구분된 학번 리스트. 미제공 시 업로드된 출석부 사용
  - `return_images` (optional): `true`면 이미지 base64 포함 (기본: `false`)

**Query Parameters (TODO - 백엔드 추가 필요):**
- `exam_code`: 시험 코드 (예: `SaS_2017_Final`)

**Response (200 OK) - 성공:**
```json
{
  "success": true,
  "student_id": "32141837",
  "header_image_base64": null,
  "original_image_base64": null,
  "meta": {
    "stage": "ocr",
    "reason": "success",
    "ocr_conf": 0.87,
    "used_vlm": false,
    "s3_uploaded": true,
    "s3_keys": {
      "original": "original/SaS_2017_Final/32141837/image.jpg",
      "header": "header/SaS_2017_Final/32141837/image.jpg"
    },
    "s3_bucket": "mlpa-gradi"
  }
}
```

**Response (200 OK) - 실패:**
```json
{
  "success": false,
  "student_id": null,
  "header_image_base64": null,
  "original_image_base64": null,
  "meta": {
    "stage": "ocr",
    "reason": "no_valid_student_id_found",
    "ocr_conf": null,
    "used_vlm": true,
    "ocr_candidates": [...]
  }
}
```

---

### 3. 출석부 조회
```
GET /roster/
```

**Response (200 OK):**
```json
{
  "success": true,
  "filename": "SaS_2017.xlsx",
  "student_count": 21,
  "student_ids": ["32161086", "32131798", ...],
  "message": "현재 21개의 학번이 로드되어 있습니다."
}
```

---

### 4. 출석부 삭제
```
DELETE /roster/
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "21개의 학번이 삭제되었습니다."
}
```

---

### 5. 헬스 체크
```
GET /health
```

**Response (200 OK):**
```json
{
  "layout_model": true,
  "ocr_model": true,
  "vlm_client": true,
  "s3_client": true,
  "s3_credentials": {
    "status": "active",
    "has_session_token": false,
    "expiration": null,
    "is_expired": false
  },
  "roster_loaded": true,
  "roster_filename": "SaS_2017.xlsx",
  "student_count": 21
}
```

---

## ⚙️ S3 저장 구조

```
s3://mlpa-gradi/
├── original/
│   └── {exam_code}/
│       └── {student_id}/
│           └── {filename}.jpg
└── header/
    └── {exam_code}/
        └── {student_id}/
            └── {filename}.jpg
```

**예시:**
- `s3://mlpa-gradi/original/SaS_2017_Final/32141837/test.jpg`
- `s3://mlpa-gradi/header/SaS_2017_Final/32141837/test.jpg`

---

## 🔄 권장 워크플로우

### Option A: 출석부 사전 업로드 방식
```
1. POST /upload-roster/  (xlsx 업로드)
2. POST /extract-student-id/  (이미지 1)
3. POST /extract-student-id/  (이미지 2)
   ...
```

### Option B: 매 요청마다 학번 리스트 전달
```
1. POST /extract-student-id/
   - image: 답안지 이미지
   - student_id_list: "32161086,32131798,..."
```

---

## 🚨 에러 처리

| HTTP Status | 설명 |
|-------------|------|
| `200 OK` | 성공 또는 인식 실패 (응답 JSON의 `success` 필드 확인) |
| `400 Bad Request` | 잘못된 요청 (파일 형식 오류 등) |
| `503 Service Unavailable` | 모델 로드 실패 |

---

## 📝 TODO (백엔드 협의 필요)

1. **exam_code 전달 방식 결정**
   - 현재: 하드코딩 (`"SaS_2017_Final"`)
   - 필요: API 파라미터로 받기

2. **STS 토큰 갱신 API 연동**
   - 엔드포인트: `?`
   - 응답 형식: `?`

3. **인증 방식**
   - API Key? JWT?
   - 헤더 형식?

4. **실패 케이스 S3 저장 여부**
   - 저장 필요 시 `failed/` 폴더 사용

5. **배치 처리 API 필요 여부**
   - 여러 이미지 한 번에 처리?

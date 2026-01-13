# 학번 인식 API 테스트 시나리오

> **서버 주소**  
> - 내부: `http://192.168.0.204:8000`  
> - 외부: `http://220.149.231.136:8000`

---

## 📋 사전 준비

### 필요 파일
1. **출석부 파일**: `출석부.xlsx` (학번 열이 포함된 엑셀 파일)
2. **테스트 이미지**: 답안지 이미지 파일 (jpg/png)

### 서버 실행 확인
```bash
curl -X GET "http://220.149.231.136:8000/health"
```

**예상 응답:**
```json
{
  "layout_model": true,
  "ocr_model": true,
  "vlm_client": false,
  "s3_client": true,
  "sqs_client": true,
  "roster_loaded": false,
  "student_count": 0
}
```

---

## 🧪 테스트 시나리오

### 시나리오 1: 시험 시작 (정상 케이스)

**목적**: exam_code와 출석부를 설정하여 시험을 시작합니다.

```bash
curl -X POST "http://220.149.231.136:8000/start-exam/?exam_code=AI_2024_MID&total_images=10" \
  -F "roster_file=@출석부.xlsx"
```

**예상 응답:**
```json
{
  "success": true,
  "exam_code": "AI_2024_MID",
  "roster_filename": "출석부.xlsx",
  "student_count": 45,
  "message": "시험 'AI_2024_MID' 시작! 45개의 학번이 로드되었습니다."
}
```

---

### 시나리오 2: 현재 시험 정보 조회

**목적**: 현재 진행 중인 시험 상태를 확인합니다.

```bash
curl -X GET "http://220.149.231.136:8000/current-exam/"
```

**예상 응답:**
```json
{
  "success": true,
  "exam_code": "AI_2024_MID",
  "roster_filename": "출석부.xlsx",
  "student_count": 45,
  "current_index": 0,
  "total_images": 10
}
```

---

### 시나리오 3: 학번 추출 (성공 케이스)

**목적**: 답안지 이미지에서 학번을 추출합니다.

```bash
curl -X POST "http://220.149.231.136:8000/extract-student-id/" \
  -F "image=@test_image_01.jpg"
```

**예상 응답 (성공):**
```json
{
  "success": true,
  "student_id": "20211234",
  "header_image_base64": null,
  "original_image_base64": null,
  "meta": {
    "stage": "ocr",
    "reason": "success",
    "ocr_conf": 0.92,
    "s3_uploaded": true,
    "s3_keys": {
      "original": "original/AI_2024_MID/20211234/test_image_01.jpg",
      "header": "header/AI_2024_MID/20211234/test_image_01.jpg"
    },
    "sqs_sent": true,
    "sqs_message_id": "abc123-def456-..."
  }
}
```

**SQS로 전송되는 메시지:**
```json
{
  "exam_code": "AI_2024_MID",
  "student_id": "20211234",
  "filename": "test_image_01.jpg",
  "index": 0,
  "event_type": "id_rec",
  "total": 10
}
```

---

### 시나리오 4: 학번 추출 (실패 케이스 - unknown_id)

**목적**: 학번을 인식하지 못한 경우의 처리를 확인합니다.

```bash
curl -X POST "http://220.149.231.136:8000/extract-student-id/" \
  -F "image=@blurry_image.jpg"
```

**예상 응답 (실패):**
```json
{
  "success": false,
  "student_id": null,
  "header_image_base64": null,
  "original_image_base64": null,
  "meta": {
    "stage": "ocr",
    "reason": "no_valid_student_id_found",
    "s3_uploaded": true,
    "s3_keys": {
      "header": "header/AI_2024_MID/unknown_id/blurry_image.jpg"
    },
    "sqs_sent": true,
    "sqs_message_id": "xyz789-..."
  }
}
```

**SQS로 전송되는 메시지:**
```json
{
  "exam_code": "AI_2024_MID",
  "student_id": "unknown_id",
  "filename": "blurry_image.jpg",
  "index": 1,
  "event_type": "id_rec",
  "total": 10
}
```

---

### 시나리오 5: 이미지 포함 응답

**목적**: 응답에 원본/헤더 이미지를 base64로 포함합니다.

```bash
curl -X POST "http://220.149.231.136:8000/extract-student-id/?return_images=true" \
  -F "image=@test_image_01.jpg"
```

**예상 응답:**
```json
{
  "success": true,
  "student_id": "20211234",
  "header_image_base64": "/9j/4AAQSkZJRg...(base64 데이터)...",
  "original_image_base64": "/9j/4AAQSkZJRg...(base64 데이터)...",
  "meta": { ... }
}
```

---

### 시나리오 6: 시험 종료

**목적**: 시험을 종료하고 상태를 초기화합니다.

```bash
curl -X DELETE "http://220.149.231.136:8000/current-exam/"
```

**예상 응답:**
```json
{
  "success": true,
  "message": "시험 'AI_2024_MID' 종료. 모든 상태가 초기화되었습니다."
}
```

---

### 시나리오 7: 시험 없이 학번 추출 시도 (오류 케이스)

**목적**: exam_code 설정 없이 학번 추출 시 SQS 전송이 실패함을 확인합니다.

```bash
curl -X POST "http://220.149.231.136:8000/extract-student-id/" \
  -F "image=@test_image_01.jpg"
```

**예상 응답:**
```json
{
  "success": true,
  "student_id": "20211234",
  "meta": {
    "s3_uploaded": true,
    "sqs_sent": false,
    "sqs_error": "exam_code 미설정 (/start-exam/ 호출 필요)"
  }
}
```

---

## 📊 E2E 테스트 체크리스트

| # | 테스트 항목 | 예상 결과 | 통과 |
|---|-------------|-----------|------|
| 1 | 서버 헬스체크 | 모든 모델 로드 확인 | ☐ |
| 2 | 시험 시작 (출석부 업로드) | 학번 리스트 로드 | ☐ |
| 3 | 현재 시험 정보 조회 | exam_code, student_count 확인 | ☐ |
| 4 | 학번 추출 (성공) | student_id 반환, S3 업로드, SQS 전송 | ☐ |
| 5 | 학번 추출 (실패) | unknown_id로 S3/SQS 처리 | ☐ |
| 6 | 인덱스 자동 증가 | current_index가 1씩 증가 | ☐ |
| 7 | 시험 종료 | 상태 초기화 확인 | ☐ |
| 8 | SQS 메시지 확인 | 백엔드에서 메시지 수신 확인 | ☐ |

---

## 🔧 디버깅 팁

### 서버 로그 확인
```bash
# 서버 실행 시 로그 출력
cd /home/jdh251425/MLPA_auto_grading/mlpa_grading/AI/id_recog
python app.py
```

### S3 업로드 확인
```bash
aws s3 ls s3://mlpa-gradi/original/AI_2024_MID/ --recursive
aws s3 ls s3://mlpa-gradi/header/AI_2024_MID/ --recursive
```

### SQS 메시지 확인 (AWS Console)
- AWS SQS Console에서 `mlpa-grading-queue.fifo` 확인
- "Messages available" 수 확인
- "Receive messages"로 메시지 내용 확인

---

## 📝 Python 테스트 스크립트

```python
import requests

BASE_URL = "http://220.149.231.136:8000"

# 1. 시험 시작
with open("출석부.xlsx", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/start-exam/",
        params={"exam_code": "TEST_EXAM", "total_images": 5},
        files={"roster_file": f}
    )
    print("시험 시작:", response.json())

# 2. 학번 추출
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/extract-student-id/",
        files={"image": f}
    )
    print("학번 추출:", response.json())

# 3. 현재 상태 확인
response = requests.get(f"{BASE_URL}/current-exam/")
print("현재 시험:", response.json())

# 4. 시험 종료
response = requests.delete(f"{BASE_URL}/current-exam/")
print("시험 종료:", response.json())
```

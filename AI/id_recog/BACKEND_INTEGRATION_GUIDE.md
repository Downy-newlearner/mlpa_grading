# 백엔드 연동 가이드

## 📋 현재 상태

### ✅ 완료된 항목
- [x] AI 서버 API 구현
- [x] S3 업로드 (정적 자격증명)
- [x] 학번 추출 파이프라인
- [x] VLM Fallback
- [x] 실패 케이스 처리

### ⏳ 연동 대기 중
- [ ] JWT 토큰 발급
- [ ] STS 자격증명 연동
- [ ] exam_code 동적 전달

---

## 🔧 백엔드에서 제공한 STS API 정보

```
URL: https://16.184.60.125/storage/sts/upload?folder={section}
Method: PUT
Headers:
  - Content-Type: application/json
  - Authorization: Bearer {jwt_token}
Body: {} (empty JSON)

Response:
{
  "access_key_id": "ASIA46E6DAI2BBQ43BEV",
  "session_token": "IQoJb3Jp...",
  "folder": "section",
  "secret_access_key": "2atRBZrK1Ygk3qerNukGUd35qN3VnSzn+PXdmBwc",
  "expiration": "2025-11-26T17:36:02Z"
}
```

---

## 📝 백엔드 팀에게 필요한 정보

### 1. JWT 토큰 발급 방법
**질문**: JWT 토큰은 어떻게 발급받나요?
- API 엔드포인트?
- 인증 방식? (username/password?)
- 토큰 유효기간?

**현재 필요**: AI 서버가 STS API를 호출하기 위한 JWT 토큰

---

### 2. exam_code (folder) 매핑
**질문**: `folder` 파라미터(=exam_code)는 어떻게 결정되나요?
- 시험마다 고정된 값? (예: `SaS_2017_Final` → `section`)
- 백엔드에서 제공? API 파라미터로 전달?

**현재 가정**: `folder=section` (고정값)

---

### 3. AI 서버 API 호출 방식
**백엔드에서 AI 서버로 요청할 때 보내야 할 정보**:

```python
import requests

AI_SERVER_URL = "http://AI_SERVER_IP:8000"

# 예시 1: 출석부 업로드
with open("roster.xlsx", "rb") as f:
    response = requests.post(
        f"{AI_SERVER_URL}/upload-roster/",
        files={"file": ("roster.xlsx", f)}
    )

# 예시 2: 학번 추출
with open("answer_sheet.jpg", "rb") as f:
    response = requests.post(
        f"{AI_SERVER_URL}/extract-student-id/",
        files={"image": ("answer.jpg", f)},
        data={
            "exam_code": "section",  # TODO: 동적 전달 필요
            "student_id_list": ""  # 출석부 사전 업로드 시 비워둠
        }
    )

result = response.json()
# {
#   "success": true/false,
#   "student_id": "32141837" or null,
#   "meta": {
#     "s3_uploaded": true,
#     "s3_keys": {
#       "original": "original/section/32141837/answer.jpg",
#       "header": "header/section/32141837/answer.jpg"
#     },
#     "s3_bucket": "mlpa-gradi"
#   }
# }
```

---

## 🚀 연동 절차

### Step 1: JWT 토큰 획득
1. 백엔드 팀에서 JWT 토큰 발급 방법 제공
2. AI 서버의 `.env` 파일에 추가:
   ```bash
   JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

### Step 2: STS 자동 갱신 활성화
```bash
# .env 파일 업데이트
STS_API_ENDPOINT=https://16.184.60.125/storage/sts/upload
JWT_TOKEN=<백엔드에서_받은_토큰>
```

### Step 3: exam_code 동적 전달
`app.py`의 `/extract-student-id/` 엔드포인트에 파라미터 추가:
```python
@app.post("/extract-student-id/")
async def extract_student_id_endpoint(
    image: UploadFile,
    exam_code: str = "section",  # 새로 추가
    ...
):
    # exam_code를 S3 업로드 시 사용
```

### Step 4: 테스트
```bash
# 1. AI 서버 시작
cd /path/to/id_recog
python app.py

# 2. Health check
curl http://localhost:8000/health

# 3. 학번 추출 테스트
curl -X POST http://localhost:8000/extract-student-id/ \
  -F "image=@test.jpg" \
  -F "exam_code=section"
```

---

## ⚠️ 현재 알려진 이슈

1. **JWT 토큰 미발급**
   - 현재 정적 AWS 자격증명 사용 중
   - STS 자동 갱신 비활성화됨

2. **exam_code 하드코딩**
   - 현재 `"SaS_2017_Final"` 고정
   - 백엔드 API 파라미터로 받도록 수정 필요

3. **SSL 인증서**
   - 백엔드 URL: `https://16.184.60.125` (자체 인증서일 경우 검증 필요)

---

## 📞 다음 단계

**백엔드 팀과 확인 필요**:
1. ✅ JWT 토큰 발급 방법
2. ✅ exam_code 매핑 규칙
3. ✅ AI 서버 배포 IP/도메인
4. ✅ 통신 테스트 일정

**AI 팀 작업**:
1. JWT 토큰 받으면 `.env` 업데이트
2. `app.py`에 exam_code 파라미터 추가
3. 통합 테스트 진행

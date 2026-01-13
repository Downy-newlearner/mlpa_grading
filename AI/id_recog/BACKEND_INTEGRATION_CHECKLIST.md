# 백엔드 연동 체크리스트

## ✅ 연동 전 준비사항

### 1. AI 서버 준비
- [x] FastAPI 서버 구현 완료
- [x] 모델 로딩 (Layout, OCR, VLM)
- [x] S3 클라이언트 초기화
- [x] API 엔드포인트 구현
- [ ] **exam_code 파라미터 추가** ⚠️ 현재 하드코딩됨
- [ ] **실패 케이스 S3 저장 여부 결정**

### 2. 백엔드 팀과 협의 필요
- [ ] **STS 토큰 발급 API 정보**
  - 엔드포인트 URL: `?`
  - 요청 형식: `?`
  - 응답 형식: `?`
  - 인증 방식: `?`

- [ ] **exam_code 전달 방식**
  - Query parameter? Body? Header?
  - 예: `POST /extract-student-id/?exam_code=SaS_2017_Final`

- [ ] **AI 서버 배포 환경**
  - 서버 IP/도메인: `?`
  - 포트: `8000` (기본값)
  - HTTPS 필요 여부: `?`

- [ ] **인증/보안**
  - AI ↔ 백엔드 간 API Key: `?`
  - CORS 설정: `?`

- [ ] **결과 전달 방식**
  - S3 URL만? 전체 JSON?
  - Webhook? Polling?

---

## 📝 백엔드 팀 요청사항

### 제공해야 할 정보

1. **STS 토큰 발급 API**
   ```
   엔드포인트: POST https://your-backend.com/api/sts/token
   
   요청:
   {
     "service": "student_id_extraction"
   }
   
   응답:
   {
     "accessKeyId": "ASIA...",
     "secretAccessKey": "...",
     "sessionToken": "...",
     "expiration": "2024-01-01T12:00:00Z"
   }
   ```

2. **exam_code 매핑 테이블**
   | 시험명 | exam_code |
   |--------|-----------|
   | 신호및시스템 2017 Final | `SaS_2017_Final` |
   | AI 2023 Mid | `AI_2023_Mid` |
   | ... | ... |

3. **AI 서버 호출 예시 (백엔드 측 코드)**
   ```python
   # 백엔드에서 AI 서버 호출
   import requests
   
   AI_SERVER_URL = "http://ai-server:8000"
   
   # 1. 출석부 업로드
   with open("roster.xlsx", "rb") as f:
       response = requests.post(
           f"{AI_SERVER_URL}/upload-roster/",
           files={"file": ("roster.xlsx", f)}
       )
   
   # 2. 학번 추출
   with open("answer_sheet.jpg", "rb") as f:
       response = requests.post(
           f"{AI_SERVER_URL}/extract-student-id/",
           files={"image": ("answer.jpg", f)},
           params={"exam_code": "SaS_2017_Final"}  # TODO: 구현 필요
       )
   
   result = response.json()
   if result["success"]:
       student_id = result["student_id"]
       s3_keys = result["meta"]["s3_keys"]
       # S3에서 이미지 가져오기 또는 URL 저장
   ```

---

## 🧪 테스트 단계

### Phase 1: 로컬 테스트 ✅
- [x] AI 서버 단독 테스트
- [x] curl/Postman 테스트
- [x] S3 업로드 검증

### Phase 2: 백엔드 연동 테스트
1. **네트워크 연결 확인**
   ```bash
   # 백엔드에서 AI 서버 ping
   curl http://AI_SERVER_IP:8000/health
   ```

2. **출석부 업로드 테스트**
   ```bash
   curl -X POST http://AI_SERVER_IP:8000/upload-roster/ \
     -F "file=@roster.xlsx"
   ```

3. **학번 추출 테스트**
   ```bash
   curl -X POST http://AI_SERVER_IP:8000/extract-student-id/ \
     -F "image=@test.jpg"
   ```

4. **STS 토큰 갱신 테스트**
   - 백엔드 STS API 호출
   - AI 서버에서 토큰 갱신 성공 여부 확인

### Phase 3: 엔드투엔드 테스트
1. 백엔드 → AI 서버 → S3 전체 플로우
2. 100건 이상 대량 처리 테스트
3. 실패 케이스 처리 검증
4. 성능 테스트 (응답 시간)

---

## ⚙️ 배포 전 설정

### AI 서버 (.env)
```bash
# AWS
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=ap-northeast-2
S3_BUCKET=mlpa-gradi

# STS (백엔드 제공)
STS_API_ENDPOINT=https://backend.example.com/api/sts/token

# OpenAI
OPENAI_API_KEY=sk-xxx
```

### 방화벽 설정
- AI 서버 포트 8000 오픈
- 백엔드 → AI 서버 통신 허용

---

## 🚨 트러블슈팅

### 문제: AI 서버 연결 안 됨
- [ ] 서버 실행 확인: `ps aux | grep python`
- [ ] 포트 확인: `netstat -tuln | grep 8000`
- [ ] 방화벽: `sudo ufw status`

### 문제: S3 업로드 실패
- [ ] AWS 자격증명 확인
- [ ] S3 버킷 권한 확인
- [ ] 네트워크 확인

### 문제: 인식률 낮음
- [ ] 이미지 품질 확인
- [ ] 출석부 학번 리스트 확인
- [ ] VLM API 키 확인

---

## 📞 연락처

**문제 발생 시:**
1. AI 서버 로그 확인: `tail -f /path/to/id_recog/app.log`
2. GitHub Issue 등록
3. 백엔드 팀과 협의

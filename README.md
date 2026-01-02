# MLPA Gradi Frontend & Backend 가이드

## 📌 프로젝트 개요
이 프로젝트는 AI를 사용하여 업로드된 답안지 이미지에서 학생 학번을 자동으로 인식하고 채점하는 시스템입니다. **Next.js Frontend**와 **Spring Boot Backend**로 구성되어 있으며, **AWS S3, SQS, Python AI 서비스**와 연동됩니다.

## 🛠️ 아키텍처 및 데이터 흐름

### 1. 학생 학번 인식 프로세스
1.  **업로드**: 사용자가 Frontend에서 답안지(이미지)를 업로드합니다.
2.  **Presigned URL**: Frontend가 Backend(`S3PresignService`)에 Presigned URL을 요청합니다.
3.  **S3 업로드**: Frontend가 이미지를 AWS S3에 직접 업로드합니다.
4.  **이벤트 트리거**: S3 업로드가 AWS Lambda 함수(`lambda_s3_trigger.py`)를 트리거합니다.
5.  **SQS 메시지 전송**: Lambda가 AWS SQS에 `event_type: "STUDENT_ID_RECOGNITION"` 메시지를 전송합니다.
6.  **Backend 처리**:
    *   `SqsListenerService`가 SQS 큐를 폴링(Polling)합니다.
    *   메시지를 파싱하고, 만약 AI 서비스에서 `status` 필드를 누락했다면 자동으로 주입(`processing` 또는 `completed`)합니다.
    *   **SSE (Server-Sent Events)**를 통해 Frontend로 진행 상황을 실시간 전송합니다.
7.  **Frontend 업데이트**: `StudentIdLoading.tsx`가 SSE 이벤트를 수신하여 진행률(예: "학생 4/40명 인식 완료")을 업데이트합니다.

---

##  wrenches: 주요 수정 사항 (Key Fixes)

### ✅ Backend (Spring Boot)
*   **SSE 및 순환 참조 해결**:
    *   `StorageController`와 `SqsListenerService` 간의 순환 의존성을 끊기 위해 SSE 로직을 별도의 `SseService`로 분리했습니다.
*   **SQS 처리 개선**:
    *   `SqsListenerService`를 리팩토링하여 `event_type`에 따라 서로 다른 로직을 수행하도록 `switch` 문을 도입했습니다.
    *   **안전장치**: AI 서비스가 `status` 값을 보내지 않더라도 Backend에서 자동으로 주입하여 Frontend가 멈추지 않도록 조치했습니다.
*   **포트 충돌 해결**:
    *   8080 포트 좀비 프로세스 문제를 해결하고, 최종적으로 **8080 포트**를 사용하도록 설정했습니다.
*   **CORS 설정**:
    *   Frontend(`http://localhost:3000`)에 대해 `allowCredentials(true)`를 포함한 CORS 정책을 적용하여 SSE 연결 안정을 확보했습니다.

### ✅ Frontend (Next.js)
*   **시험 생성 내비게이션 수정**:
    *   로딩 페이지로 이동할 때 `total` 값이 0이거나 문항 수로 잘못 전달되던 버그를 수정했습니다.
    *   이제 업로드된 **실제 파일 개수(`answerSheetFiles.length`)**가 정확히 전달됩니다.
*   **진행률 표시 (Progress Bar)**:
    *   `StudentIdLoading` 컴포넌트가 SSE 이벤트를 수신할 때마다 전체 개수와 현재 진행 개수를 동적으로 업데이트하도록 수정했습니다.
*   **프록시 설정**:
    *   `next.config.ts`에서 `/api/*` 요청을 Backend 포트(`8080`)로 포워딩하도록 설정했습니다.

---

## 🚀 실행 방법 (How to Run)

### 1. Backend (백엔드)
```bash
cd BE
./gradlew bootRun
# 서버가 http://localhost:8080 에서 실행됩니다.
```

### 2. Frontend (프론트엔드)
```bash
cd FE/mlpa-gradi-frontend
npm run dev
# 앱이 http://localhost:3000 에서 실행됩니다.
```

### 3. AI 시뮬레이션 (테스트용)
실제 AWS 업로드 없이 전체 흐름을 테스트하려면 다음 스크립트를 실행하세요:
```bash
python BE/ai/simulate_full_flow.py
# 가짜 SQS 메시지를 전송하여 Frontend가 정상적으로 반응하는지 확인합니다.
```

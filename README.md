# Gradi - AI 기반 답안지 자동 채점 시스템

## 📋 프로젝트 개요

Gradi는 AI를 활용하여 학생들의 답안지를 자동으로 채점하는 웹 애플리케이션입니다. 
학번 인식, 문항 인식, 채점 과정을 실시간으로 진행하며, 사용자에게 피드백을 받아 정확도를 높입니다.

## 🏗️ 기술 스택

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS + Vanilla CSS
- **State Management**: React Hooks (useState, useEffect, useCallback)
- **Real-time Communication**: Server-Sent Events (SSE)

##  wrenches: 주요 수정 사항 (Key Fixes)

### ✅ Backend (Spring Boot)
*   **SSE 및 순환 참조 해결**:
    *   `StorageController`와 `SqsListenerService` 간의 순환 의존성을 끊기 위해 SSE 로직을 별도의 `SseService`로 분리했습니다.
*   **SQS 처리 개선**:
    *   `SqsListenerService`를 리팩토링하여 `event_type`에 따라 서로 다른 로직을 수행하도록 `switch` 문을 도입했습니다.
    *   **안전장치**: AI 서비스가 `status` 값을 보내지 않더라도 Backend에서 자동으로 주입하여 Frontend가 멈추지 않도록 조치했습니다.
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

## 📂 프로젝트 구조

```
mlpa-gradi-frontend/
├── FE/mlpa-gradi-frontend/     # Next.js 프론트엔드
│   ├── app/
│   │   ├── components/         # 공통 컴포넌트
│   │   ├── exam/[examId]/      # 시험 관련 페이지 (Dynamic Routes)
│   │   ├── history/            # 결과 조회 페이지
│   │   ├── hooks/              # 커스텀 훅
│   │   └── services/           # API 서비스
│   └── public/                 # 정적 파일
│
├── BE/                         # Spring Boot 백엔드
│   ├── src/main/java/
│   │   └── com/dankook/mlpa_gradi/
│   │       ├── controller/     # REST Controllers
│   │       ├── service/        # 비즈니스 로직
│   │       ├── repository/     # 데이터 접근 계층
│   │       └── config/         # 설정 클래스
│   └── ai/                     # Lambda 함수 (Python)
│
└── README.md
```

## 🔄 채점 플로우

```
1. 시험 정보 입력 (ExamInput)
   ↓
2. 학번 인식 로딩 (StudentIdLoading) - SSE 실시간 진행 표시
   ↓
3. 학번 인식 완료 (StudentIdRecognitionDone)
   ↓
4. 학번 피드백 (FeedbackPage) - 인식 실패한 학번 수정
   ↓
5. 문항 인식 로딩 (QuestionLoading) - SSE 실시간 진행 표시
   ↓
6. 문항 인식 완료 (QuestionRecognitionDone)
   ↓
7. 문항 피드백 (QuestionFeedbackPage) - 인식 실패한 문항 수정
   ↓
8. 채점 로딩 (GradingLoading) - SSE 실시간 진행 표시
   ↓
9. 채점 완료 (GradingDone) - 🎉 축하 애니메이션
   ↓
10. 결과 조회 (HistoryExamSelect) - 방금 채점된 시험 강조 표시
```

## ✨ 주요 기능 및 구현 사항

### 1. 실시간 진행 상황 표시 (SSE)
- **구현**: `SseService.java`에서 SSE 연결 관리
- **특징**: 
  - 버퍼링 방지를 위한 패딩 전송 (4KB+)
  - 5초마다 heartbeat로 연결 유지
  - 10분 무활동 시 세션 자동 제거

### 2. 중복 메시지 방지 (Deduplication)
- **문제**: SQS에서 동일 메시지가 여러 번 전송되어 카운터가 비정상적으로 증가
- **해결**: `SessionInfo.processedFiles` Set으로 처리된 파일 추적
- **위치**: `SseService.java`, `SqsListenerService.java`

### 3. 동적 애니메이션 (Done 페이지)
- **구현**: `StudentIdRecognitionDone.tsx`, `QuestionRecognitionDone.tsx`, `GradingDone.tsx`
- **효과**:
  - Pop-in 효과 (체크마크)
  - Check path 드로잉 애니메이션
  - 회전 링 효과
  - 파티클 효과
  - Fade-in-up 텍스트 애니메이션

### 4. 피드백 페이지 개선
- **기능**: 인식 실패한 항목 수정, 이미지 확대/이동, 키보드 네비게이션
- **UX**: Fire-and-forget 패턴으로 즉시 페이지 이동 후 백그라운드 API 호출
- **저장**: localStorage에 임시 저장 (draft)

### 5. 삭제 UX 개선
- **구현**: 삭제 중 오버레이 + 스피너 표시
- **안정성**: 멱등성 보장 (이미 삭제된 항목도 에러 없이 처리)
- **위치**: `page.tsx`, `StudentIdLoading.tsx`, `QuestionLoading.tsx`

### 6. 채점 완료 시험 강조 표시
- **구현**: `/history?highlight=EXAMCODE`로 이동 시 해당 시험 강조
- **효과**: 
  - 보라색 그라데이션 테두리
  - "✨ NEW" 뱃지 (bounce 애니메이션)
  - 3초 후 자동 해제
- **위치**: `HistoryExamSelect.tsx`

### 7. 시험 일시 기본값
- **구현**: 오늘 날짜 + 09:00 시간으로 자동 설정
- **형식**: `YYYY-MM-DDTHH:MM` (datetime-local input 호환)
- **위치**: `useExamForm.ts`

## 🔧 환경 설정

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
```

### Backend (application.yml)
```yaml
aws:
  s3:
    bucket: your-bucket-name
  sqs:
    queue-url: https://sqs.ap-northeast-2.amazonaws.com/xxx/your-queue.fifo

ai:
  server:
    url: http://your-ai-server:8000
```

## 🐛 해결된 이슈

### 1. Windows 콘솔 이모지 깨짐
- **원인**: Windows 콘솔의 UTF-8 지원 부족
- **해결**: 로그 메시지에서 이모지를 ASCII 텍스트로 대체

### 2. 네트워크 오류 (CORS)
- **원인**: 프론트엔드에서 직접 AI 서버 호출 시 CORS 차단
- **해결**: 백엔드 프록시를 통해 AI 서버로 요청 전달 (`FeedbackService.java`)

### 3. 채점 계속하기 버튼 느림
- **원인**: API 응답 대기로 인한 UI 블로킹
- **해결**: Fire-and-forget 패턴 (먼저 페이지 이동, 백그라운드에서 API 호출)

### 4. 시험 삭제 시 500 에러
- **원인**: 이미 삭제된 시험 재삭제 시 `NoSuchElementException`
- **해결**: `ExamService.deleteByCode()` 멱등성 보장

### 5. 진행 카운터 비정상 증가 (154/0)
- **원인**: SQS 중복 메시지
- **해결**: 파일명 기반 중복 체크 (`processedFiles` Set)

## ⚠️ 알려진 이슈

### 1. S3 키 경로 불일치
- **증상**: 로그에서 `header/{examCode}/unknown_id/{filename}` 키로 URL 생성 시 실제 URL에 `uploads/...` 경로 포함
- **영향**: 현재 기능에는 영향 없음
- **원인**: S3PresignService 또는 S3 버킷 설정 확인 필요

### 2. 문항 피드백 API 미구현
- **현황**: `/api/question-feedback`, `/api/reports/unknown-questions/{examCode}` 엔드포인트 미구현
- **임시 처리**: 빈 목록 반환으로 기본 동작

## 🚀 실행 방법

### Frontend
```bash
cd FE/mlpa-gradi-frontend
npm install
npm run dev
```

### Backend
```bash
cd BE
./gradlew bootRun
```

## 📝 커밋 컨벤션

```
feat: 새로운 기능 추가
fix: 버그 수정
docs: 문서 수정
style: 코드 포맷팅
refactor: 코드 리팩토링
test: 테스트 코드
chore: 빌드 설정, 패키지 매니저 등
```

## 👥 기여자

- Dankook University MLPA Team

---

*Last Updated: 2026-01-07*

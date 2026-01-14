# 변경 사항 요약 (Idempotency Restoration)

본 작업에서는 커밋 `399de9a7d2a75e9f2d41a034b9681bbf4b55d4ec`에서 구현되었던 **SQS 멱등성 보장 로직**을 복구하고, 서버 구동 환경을 점검하였습니다.

## 주요 변경 사항

### 1. SQS FIFO 및 멱등성 보장 (Lambda Trigger)
- **파일 경로**: `BE/ai/lambda_s3_trigger.py`
- **변경 내용**: 
    - SQS 메시지 전송 시 `MessageGroupId`와 `MessageDeduplicationId`를 추가하였습니다.
    - `examCode`를 GroupId로 설정하여 동일 시험 내 메시지의 순차 처리를 보장합니다.
    - `eventID` 또는 파일의 고유 식별자를 조합한 `dedup_id`를 사용하여 SQS 레벨에서의 중복 제거를 활성화하였습니다.

### 2. 수신 측 중복 처리 방지 (SQS Listener)
- **파일 경로**: `BE/src/main/java/com/dankook/mlpa_gradi/service/SqsListenerService.java`
- **변경 내용**:
    - `handleRecognitionProgress` 메소드에서 `session.processedFiles` (HashSet)를 사용하여 이미 처리된 `filename`인지 확인하는 로직을 복구하였습니다.
    - 이를 통해 네트워크 지연이나 재시도로 인해 동일한 메시지가 중복 수신되더라도 결과가 중복 합산되지 않도록 보장합니다.

### 3. 멀티 큐 환경 지원
- 기존 `application.yml`에서 설정된 두 개의 SQS 큐(`AWS_SQS_QUEUE_URL`, `AWS_SQS_QUEUE_URL_2`)가 각각의 용도(입력/결과)에 맞게 정상적으로 활용될 수 있는 기반을 확인하였습니다.

## 결과 및 후속 조치
- 변경 사항은 `origin main` 브랜치에 커밋 및 푸시되었습니다.
- Frontend (3000), Backend (8080) 서버가 구동 중임을 확인하였습니다.
- AI 서버 구동을 위해 필요한 의존성 확인 루틴을 실행하였습니다.

---
**Commit Message**: `fix: SQS FIFO GroupId/DedupId 적용 및 중복 처리 로직 복구`

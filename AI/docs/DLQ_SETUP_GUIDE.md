# SQS Dead Letter Queue (DLQ) 설정 가이드

## 📋 개요

이 가이드는 MLPA SQS 큐에 DLQ(Dead Letter Queue)를 설정하는 방법을 설명합니다.
DLQ는 일정 횟수 이상 처리 실패한 메시지를 저장하여 무한 재시도 루프를 방지합니다.

## 🔧 AWS 콘솔에서 DLQ 설정하기

### Step 1: DLQ 생성

1. [AWS SQS 콘솔](https://ap-northeast-2.console.aws.amazon.com/sqs/v3/home?region=ap-northeast-2)에 로그인

2. **[대기열 생성]** 클릭

3. 다음 정보 입력:
   - **유형**: FIFO (기존 큐가 FIFO이므로)
   - **이름**: `mlpa-grading-queue-dlq.fifo`
   - **콘텐츠 기반 중복 제거**: ✅ 활성화
   - **메시지 보관 기간**: 14일 (최대)
   - **가시성 제한 시간**: 300초 (5분)

4. **[대기열 생성]** 클릭

### Step 2: 메인 큐에 RedrivePolicy 설정

1. 메인 큐 선택: `mlpa-grading-queue.fifo`

2. 상단 **[편집]** 버튼 클릭

3. 아래로 스크롤하여 **"배달 못한 편지 대기열"** (Dead Letter Queue) 섹션 찾기
   - 또는 **"Redrive policy"** 섹션으로 표시될 수 있음

4. 다음 설정:
   - **활성화됨** 또는 **Enabled**: ✅ 체크
   - **대기열 ARN** 또는 **Queue ARN**: 방금 생성한 DLQ 선택
     - `arn:aws:sqs:ap-northeast-2:309293113458:mlpa-grading-queue-dlq.fifo`
   - **최대 수신** 또는 **Maximum receives**: `3` (3회 실패 후 DLQ로 이동)

5. **[저장]** 클릭

> 💡 **팁**: 섹션을 찾기 어려우면 페이지에서 `Ctrl+F`로 "dead" 또는 "redrive"를 검색해보세요.

### Step 3: 환경변수 설정

`.env` 파일에 다음 추가:

```bash
# DLQ URL (선택사항 - 모니터링 스크립트용)
SQS_DLQ_URL=https://sqs.ap-northeast-2.amazonaws.com/309293113458/mlpa-grading-queue-dlq.fifo
```

## 📊 DLQ 모니터링

### 1. 콘솔에서 확인

AWS SQS 콘솔에서 `mlpa-grading-queue-dlq.fifo` 큐를 선택하면:
- **사용 가능한 메시지**: DLQ에 들어간 실패 메시지 수
- **처리 중 메시지**: 현재 컨슈머가 읽고 있는 메시지 수

### 2. CLI 스크립트로 확인

```bash
# DLQ 상태 조회
python scripts/monitor_dlq.py

# 메시지 미리보기 (삭제 안 함)
python scripts/monitor_dlq.py --peek

# DLQ 비우기 (주의!)
python scripts/monitor_dlq.py --purge

# 메시지를 메인 큐로 재전송
python scripts/monitor_dlq.py --redrive
```

## ⚙️ 코드 레벨 재시도 제한

AWS DLQ와 별개로, `sqs_worker.py`에 **코드 레벨 재시도 제한**이 구현되어 있습니다:

```python
# sqs_worker.py
self._max_nack_count = 5  # 최대 NACK 횟수
```

### 동작 방식

1. 출석부가 로드되지 않은 상태에서 메시지 수신
2. NACK 처리 (메시지 삭제 안 함 → 재시도)
3. 재시도 횟수가 `_max_nack_count`에 도달하면:
   - 에러 결과 메시지를 BE로 전송
   - 메시지를 큐에서 삭제
   - 무한 재시도 방지

### 에러 메시지 형식

```json
{
  "eventType": "STUDENT_ID_RECOGNITION",
  "examCode": "BZPTX8",
  "studentId": "unknown_id",
  "filename": "test.jpg",
  "index": -1,
  "meta": {
    "error": "ATTENDANCE_NOT_LOADED",
    "message": "출석부가 5회 시도 후에도 로드되지 않았습니다.",
    "nack_count": 5
  }
}
```

## 🔄 DLQ 메시지 재처리

DLQ에 쌓인 메시지를 재처리하려면:

### Option 1: 출석부 업로드 후 자동 처리

1. 해당 시험의 출석부 업로드
2. `--redrive` 옵션으로 메시지 재전송
3. 메시지가 자동으로 처리됨

### Option 2: 수동 처리

```bash
# 1. 메시지 확인
python scripts/monitor_dlq.py --peek

# 2. 문제 해결 (예: 출석부 업로드)

# 3. 메시지 재전송
python scripts/monitor_dlq.py --redrive --count 100
```

## 📝 체크리스트

- [ ] AWS 콘솔에서 DLQ 생성
- [ ] 메인 큐에 RedrivePolicy 설정
- [ ] `.env`에 `SQS_DLQ_URL` 추가
- [ ] 서버 재시작하여 코드 레벨 제한 적용
- [ ] 테스트: 출석부 없이 이미지 메시지 전송 → 5회 후 자동 삭제 확인

## ⚠️ 주의사항

1. **DLQ와 코드 레벨 제한의 관계**
   - 코드 레벨 제한 (`_max_nack_count=5`)이 먼저 동작
   - AWS DLQ는 코드 레벨에서 처리되지 않은 예외 상황용

2. **재시도 간격**
   - `VisibilityTimeout=300` (5분)
   - 5번 재시도하면 최대 25분 동안 재시도

3. **메모리 관리**
   - `_nack_tracker`는 메시지별로 카운트 추적
   - 처리 성공/실패 시 자동으로 정리됨

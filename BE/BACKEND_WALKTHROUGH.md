# 백엔드 워크스루 - Gradi (MLPA)

본 문서는 Gradi 프로젝트의 핵심 백엔드 기능 및 API 구현 사항을 요약합니다.

## 🚀 주요 기능

### 1. 강력한 데이터 정리 시스템
- **S3 재귀적 삭제**: `S3PresignService`에 `deleteByExamCode`를 구현하여 `uploads/{examCode}/` 및 `attendance/{examCode}/` 경로의 모든 객체를 `DeleteObjectsRequest`를 통해 효율적으로 일괄 삭제합니다.
- **연관 데이터 정리**: `ExamService`에서 시험 레코드를 삭제하기 전, 해당 시험과 관련된 모든 학생 답안(StudentAnswer)을 데이터베이스에서 먼저 삭제하여 외래키 제약 조건 위반을 방지합니다.
- **트랜잭션 안정성**: 리포지토리에 `@Modifying` 및 `@Transactional` 어노테이션을 적용하여 삭제 작업의 원자성을 보장했습니다.

### 2. PDF 리포트 생성 엔진 (정오표)
- **iText 8 통합**: `PdfService`를 통해 학생별 시험 결과 리포트를 PDF 파일로 생성하는 기능을 구현했습니다.
- **한글 폰트 지원**: 나눔고딕(NanumGothic) 폰트 로딩 설정을 통해 학생 이름 및 시험 정보가 한글로 올바르게 렌더링되도록 했습니다.
- **동적 테이블 생성**: 학생의 응답과 정답을 비교하여 색상(맞음: 하늘색, 틀림: 연분홍)으로 구분한 대조표를 자동으로 생성합니다.

### 3. S3 연동 및 미디어 API
- **배치 Presigned URL**: 여러 장의 답안지 이미지를 업로드하기 위한 Presigned PUT URL을 한 번의 호출로 일괄 요청할 수 있습니다.
- **학생별 이미지 조회**: 파일명에 포함된 학번(`studentId`)을 필터링하여 특정 학생의 채점 이미지만 선별적으로 조회할 수 있는 Presigned GET URL 생성 API를 추가했습니다.
- **프로그래밍 방식의 CORS 설정**: 서버 시작 시 `app.frontend.url` 설정을 읽어 S3 버킷의 CORS 규칙을 자동으로 구성합니다.

### 4. REST API 엔드포인트
- **ReportController**:
    - `GET /api/reports/pdf/{examCode}/{studentId}`: 학생용 정오표 PDF 다운로드
    - `GET /api/reports/images/{examCode}/{studentId}`: 해당 학생의 채점 이미지 URL 목록 조회
- **ExamController**: 시험 코드 및 ID 기반의 CRUD 엔드포인트 제공.
- **StorageController**: SSE 연결 관리 및 출석부 데이터 업로드 처리.

## 🛠 기술 스택
- **언어**: Java 21.
- **프레임워크**: Spring Boot 4.
- **저장소**: AWS S3 (software.amazon.awssdk 활용).
- **PDF 생성**: iText 8 Core.
- **데이터베이스**: JPA/Hibernate 기반 MySQL.

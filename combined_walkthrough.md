# 통합 워크스루: 최종 UI 개선 및 기능 강화

이 문서는 애플리케이션 전반(`page.tsx`, `ExamInput.tsx`, `HistoryExamSelect.tsx`, `StatisticsDownload.tsx`)에 걸쳐 수행된 모든 업데이트 사항을 요약합니다.

## 1. 시각적 개선 (Visual Refinements)
### A. 그라데이션 & 브랜딩
-   **타이포그래피**: **보라색 그라데이션** (`from-[#AC5BF8] to-[#636ACF]`)을 다음 요소에 적용했습니다:
    -   메인 페이지 제목.
    -   히스토리 페이지의 날짜(월별) 그룹 헤더.
    -   특정 "과목 통계" 텍스트 항목.
-   **구분선**: 리스트 항목 내의 수직 구분선(`|`)을 **굵게(Bold)** 처리하고 **보라색 그라데이션**을 적용했습니다.
-   **아이콘**:
    -   **다운로드 아이콘** (Statistics): 그라데이션이 적용된 깔끔한 "화살표(Arrow Down)" 디자인으로 변경했습니다.
    -   **달력 아이콘** (Exam Input): 기본 브라우저 아이콘을 숨기고, 커스텀 **보라색 그라데이션** 달력 아이콘으로 교체했습니다.

### B. 레이아웃 & 간격
-   **Exam Input**: 섹션 헤더와 콘텐츠 박스 사이의 간격을 늘려(`mb-2` → `mb-8`) 가독성을 높였습니다.
-   **Home Page**: 버튼의 위치(`Top: 521px`, `Left: 258px / 677px`)는 유지하되, 디자인을 원래의 "심플 버튼" 스타일로 복구했습니다.
-   **로고 위치**: 모든 페이지에서 Gradi 로고의 위치를 **`Top: 17px, Left: 10px`**로 통일했습니다.

## 2. 상호작용 개선 (Interaction Improvements)
### A. 커서 피드백 (Cursors)
-   **전역 `cursor-pointer` 적용**: 모든 상호작용 가능한 요소에 마우스 오버 시 "손가락 커서"가 나타나도록 했습니다:
    -   **Home**: 메인 내비게이션 버튼.
    -   **Exam Input**: 날짜 선택기, 파일 업로드, 드롭다운, 사이드바 버튼(추가/제거), 드래그 가능한 문제 박스.
    -   **Statistics**: 다운로드 리스트의 전체 행(Row) 클릭 가능 처리.

### B. 내비게이션 & 기능
-   **로고 링크**: 좌측 상단 Gradi 로고를 클릭하면 모든 페이지에서 **홈 화면 (`/`)**으로 이동합니다.
-   **날짜 선택기 (Date Picker)**:
    -   기본 인디케이터를 숨겼습니다.
    -   **어디든 클릭 가능**: 아이콘뿐만 아니라 날짜 입력 박스의 어느 곳을 클릭해도 즉시 달력이 열리도록 구현했습니다.

## 3. 컴포넌트 업데이트 사항
-   **`ExamInput.tsx`**:
    -   "채점 시작하기" 버튼의 높이를 키웠습니다 (`py-4`).
    -   새 문제 추가 시 자동으로 스크롤되는 기능을 구현했습니다.
    -   UI 박스 스타일을 통일하고 헤더 폰트 사이즈를 키웠습니다.
-   **`HistoryExamSelect.tsx`**:
    -   월별 그룹화(YYYY.MM) 기능을 구현했습니다.
    -   `ExamInput`과 동일한 **보라색 테두리 박스 스타일**을 적용했습니다.
    -   **백엔드 데이터 매핑 수정**: `examName`, `examDate` 필드를 올바르게 사용하도록 수정했습니다.
-   **`StatisticsDownload.tsx`**:
    -   학생용 파일과 과목용 파일을 구분하여 스타일링(과목: 보라색 테두리, 학생: 검은색 테두리)을 적용했습니다.

## 4. 백엔드 연동 및 버그 수정 (Backend Integration & Fixes)
### A. 시험 생성 404 에러 해결
- **원인**: Next.js 프록시 설정이 `/api` 경로를 백엔드로 전달할 때 prefix를 누락함.
- **해결**: `next.config.ts`의 `rewrite` 설정을 `http://localhost:8080/api/:path*`로 수정하여 prefix가 유지되도록 함.

### B. 데이터 매핑 수정 (Data Mapping)
- **원인**: 백엔드 DTO(`examName`, `examDate`)와 프론트엔드 요청(`title`, `date`) 불일치로 데이터가 `null`로 저장되는 문제.
- **해결**:
  - `CreateExamRequest`, `ExamHistoryItem` 인터페이스 업데이트.
  - `useExamForm.ts` 및 `useExamHistory.ts` 로직을 수정하여 올바른 필드명(`examName`, `examDate`) 사용.

## 5. 개발 업데이트 워크스루 (Development Updates)
### A. 서버 설정 외부화
- 하드코딩된 서버 주소를 환경 변수를 사용하도록 리팩토링했습니다.
- `next.config.ts` 파일이 `process.env.API_URL`을 사용하도록 업데이트 되었습니다.

```typescript
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.API_URL || "http://localhost:8080/api/:path*",
      },
    ];
  },
};

export default nextConfig;
```

### B. Git 동기화
- **문제 해결**: `unrelated histories` 오류로 인한 병합 불가 문제를 해결했습니다.
- **명령어**: `git pull origin main --allow-unrelated-histories`를 사용하여 로컬과 원격 히스토리를 성공적으로 병합했습니다.

## 6. 최종 UI/UX 및 기능 통합 (Final UI/UX & Feature Integration)

### A. UI 정제 및 브랜딩 통일
-   **상단 로고 통일**: 모든 페이지(`page.tsx`, `ExamInput.tsx`, `HistoryExamSelect.tsx`, `StatisticsDownload.tsx`)의 상단 로고를 **Gradi 로고**로 통일하고, 클릭 시 **홈 화면**으로 이동하도록 통합했습니다.
-   **중앙 이미지 로직**: `ExamInput` 페이지에서 중앙 로고 이미지를 제거하여, 홈 화면에서만 중앙 이미지가 표시되도록 정리했습니다.
-   **레이아웃 최적화**: `ExamInput`의 상단 패딩을 줄여(`pt-480px` → `pt-120px`) 콘텐츠 접근성을 대폭 개선했습니다.

### B. 폰트 변경 (Naver NanumSquare)
-   **CDN 적용**: 기존 구글 폰트(`Geist`, `Nanum Gothic`) 대신 **네이버 나눔스퀘어(NanumSquare)** 폰트를 CDN으로 로드하도록 변경했습니다.
-   **전역 적용**: `app/layout.tsx`와 `globals.css`를 수정하여 애플리케이션 전체에 나눔스퀘어 폰트가 기본 적용되도록 설정했습니다.

### C. 채점 및 피드백 플로우 구현 (Grading & Feedback Flow)
-   **Grading Loading (채점 중)**:
    -   **SSE Mocking**: 채점 로딩 화면(`GradingLoading.tsx`)에 SSE 스트림을 시뮬레이션하는 기능을 추가했습니다. 답안지 이미지가 실시간으로 스캔되는 듯한 UI(Mock Image + Pulse Effect)를 구현했습니다.
-   **Grading Done (채점 완료)**:
    -   **동적 라우팅**: 채점 완료 화면(`GradingDone.tsx`)의 "결과 확인" 버튼이 해당 시험의 **피드백 페이지 (`/history/[examId]`)**로 동적으로 연결되도록 수정했습니다.
-   **Feedback Page (통계/피드백)**:
    -   `StatisticsDownload.tsx`를 피드백 페이지로 활용하며, 상단 로고 및 디자인 일관성을 확보했습니다.

## 검증 체크리스트 (Verification)
1.  **Home**: 심플한 버튼 디자인과 클릭 가능한 로고 확인.
2.  **Input**: 날짜 박스 클릭 시 달력 팝업 확인 (그라데이션 아이콘). 사이드바 버튼 마우스 오버 시 손가락 커서 확인.
3.  **Grading**: 채점 로딩 시 답안지 이미지가 표시되고, 완료 시 결과 확인 버튼이 올바른 히스토리 페이지로 이동하는지 확인.
4.  **Font**: 모든 텍스트가 **나눔스퀘어** 폰트로 렌더링되는지 확인.
5.  **Navigation**: 모든 페이지의 상단 Gradi 로고 클릭 시 홈으로 이동 확인.

# PR Description: 만든 사람들 페이지 구현

## 1. 주요 변경 사항
- **'만든 사람들' (Makers) 페이지 추가 (`/makers`)**
  - 팀원 소개 카드 구현 (정다훈, 정성원, 조성빈)
  - 각 팀원의 프로필 이미지, 역할, 소속 학과, 학번, 연락처(Mobile, Email), Github/IG 링크 표시
  - **팀장(Team Leader) 강조 효과**: 정다훈 (Premium Glow Effect & Crown Badge)
  - 반응형 그리드 레이아웃 적용

- **메인 페이지 (`/`) 업데이트**
  - 우측 상단에 '만든 사람들' 페이지로 이동하는 링크 추가
  - Purple Gradient 텍스트 스타일 적용

- **이미지 리소스 추가**
  - `public/makers/` 경로에 팀원 프로필 이미지 추가

## 2. 세부 구현 내용
- **UI/UX**
  - Purple Gradient (`#AC5BF8` to `#636ACF`) 테마 적용
  - 카드 등장 시 순차적인 Fade-in Animation (`animate-fade-in-up`) 적용
  - Hover 시 쉐도우 및 스케일 효과

## 3. 팀원 정보
- **정다훈 (Team Leader, AI)**: 단국대학교 SW융합대학 소프트웨어학과 (32204041)
- **정성원 (Full Stack)**: 단국대학교 SW융합대학 소프트웨어학과 (32204077)
- **조성빈 (Backend)**: 단국대학교 SW융합대학 소프트웨어학과 (32215110)

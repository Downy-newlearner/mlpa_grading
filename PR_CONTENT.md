# PR Title
feat: Implement Dankook ID Verification, UI Polish & Project Cleanup

# PR Description
## 📋 Summary
본 PR은 단국대학교 포털 학번 검증 기능 구현, 전반적인 UI/UX 개선, 그리고 프로젝트 설정(gitignore, 보안) 정리를 포함합니다.

## ✨ Key Changes

### 1. 🔐 단국대학교 학번 검증 (ID Verification)
- **Backend**: `PortalAuthService`를 구현하여 단국대 포털 API(`ConfirmUserId.eps`)를 프록시 호출합니다. (CORS 해결)
- **Frontend**: 학생 결과 확인 표, 본인 확인 페이지(`StudentVerifyPage`)에서 실시간으로 학번 존재 여부를 검증합니다.
- **Security**: `/api/auth/verify-dku` 엔드포인트를 인증 없이 접근 가능하도록 허용했습니다.

### 2. 🎨 UI/UX 개선 (Branding & Polish)
- **Branding**: MLPA 로고를 제거하고 **Gradi** 및 **단국대학교 엠블럼**으로 리브랜딩했습니다.
- **Favicon**: 단국대 엠블럼 아이콘으로 파비콘을 교체했습니다.
- **Micro-interactions**: 모든 버튼과 인터랙티브 요소에 `cursor-pointer` 및 호버 효과를 적용하여 사용성을 개선했습니다.

### 3. 🐛 Bug Fixes
- **파일 업로드 카운트**: 파일 삭제 시 `input[type="file"]`의 UI 텍스트가 업데이트되지 않던 버그를 수정했습니다. (React State와 Native Input 동기화)

### 4. ⚙️ Project Maintenance
- **.gitignore 업데이트**: `.idea`, `.vscode`, `BE/ai` 등 불필요한 파일 및 로컬 설정 파일이 트래킹되지 않도록 정리했습니다.
- **Secret Removal**: 실수로 포함된 환경 변수 파일 및 AWS 자격 증명을 Git 기록에서 제거하고 안전하게 처리했습니다.
- **Conflict Resolution**: Upstream `main`과의 충돌을 해결하고 최신 상태로 동기화했습니다.

## 🧪 Verification
- [x] `32204077` (유효 학번) 입력 시 결과 페이지로 정상 이동 확인
- [x] `12345678` (무효 학번) 입력 시 에러 메시지 표시 확인
- [x] 파일 업로드 및 삭제 시 카운트 정상 동작 확인
- [x] Github Push 시 Secret Scanning 통과 확인

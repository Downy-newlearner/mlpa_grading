import { CreateExamRequest, ExamHistoryItem } from "../types";

const API_BASE = "/api";

// SSE 연결을 위한 타입
export interface BatchPresignRequest {
    examCode: string;
    total: number;
    images: { index: number; contentType: string; filename: string }[];
}

export interface BatchPresignResponse {
    examCode: string;
    urls: { index: number; filename: string; url: string }[];
}

export interface ExamCreateResponse {
    examId: number;
    examCode: string;
    examName: string;
    examDate: string;
}

export const examService = {
    // Fetch all exams
    async getAll(): Promise<ExamHistoryItem[]> {
        const response = await fetch(`${API_BASE}/exams`);
        if (!response.ok) throw new Error("Failed to fetch exams");
        return response.json();
    },

    // Fetch exam by code
    async getByCode(examCode: string): Promise<ExamHistoryItem> {
        const response = await fetch(`${API_BASE}/exams/code/${examCode}`);
        if (!response.ok) throw new Error("Failed to fetch exam by code");
        return response.json();
    },

    // Create a new exam (returns examCode)
    async create(data: CreateExamRequest): Promise<ExamCreateResponse> {
        const response = await fetch(`${API_BASE}/exams`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("Failed to create exam");
        return response.json();
    },

    // ✅ SSE 연결 (examCode 기반)
    connectSSE(examCode: string): EventSource {
        const eventSource = new EventSource(`${API_BASE}/storage/sse/connect?examCode=${examCode}`);
        return eventSource;
    },

    // ✅ 배치 Presigned URL 요청
    async getBatchPresignedUrls(data: BatchPresignRequest): Promise<BatchPresignResponse> {
        const response = await fetch(`${API_BASE}/storage/presigned-urls/batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("Failed to get batch presigned URLs");
        return response.json();
    },

    // Presigned URL로 이미지 업로드
    async uploadToPresignedUrl(presignedUrl: string, file: File, contentType: string, metadata?: { total?: number; index?: number }): Promise<void> {
        const headers: HeadersInit = {
            "Content-Type": contentType,
        };

        if (metadata) {
            if (metadata.total !== undefined && metadata.total !== null) headers["x-amz-meta-total"] = metadata.total.toString();
            if (metadata.index !== undefined && metadata.index !== null) headers["x-amz-meta-index"] = metadata.index.toString();
        }

        const response = await fetch(presignedUrl, {
            method: "PUT",
            body: file,
            headers: headers,
        });
        if (!response.ok) throw new Error("Failed to upload to presigned URL");
    },

    // ✅ 시험 삭제 (Code 기반) - 롤백용
    async deleteByCode(examCode: string): Promise<void> {
        const response = await fetch(`${API_BASE}/exams/code/${examCode}`, {
            method: "DELETE",
        });
        if (!response.ok) throw new Error("Failed to delete exam");
    },

    // ✅ 출석부 다운로드 URL 가져오기
    async getAttendanceDownloadUrl(examCode: string): Promise<string> {
        const response = await fetch(`${API_BASE}/storage/attendance/download-url?examCode=${examCode}`);
        if (!response.ok) throw new Error("Failed to get attendance download URL");
        const data = await response.json();
        return data.url;
    },

    // ✅ 특정 시험의 답변 현황 조회 (학생 목록 추출용)
    async getAnswersByExamCode(examCode: string): Promise<any[]> {
        const response = await fetch(`${API_BASE}/student-answers/exam/${examCode}`);
        if (!response.ok) throw new Error("Failed to fetch answers");
        return response.json();
    },

    // ✅ 출석部 업로드용 Presigned URL 요청
    async getAttendancePresignedUrl(examCode: string, contentType: string): Promise<string> {
        const response = await fetch(`${API_BASE}/storage/presigned-url/attendance?examCode=${examCode}&contentType=${contentType}`);
        if (!response.ok) throw new Error("Failed to get attendance presigned URL");
        const data = await response.json();
        return data.url;
    },

    // ✅ 현재 진행 중인 채점 프로세스 목록 조회
    async getActiveProcesses(): Promise<{ examCode: string; examName: string; index: number; total: number; status: string; lastUpdateTime: number }[]> {
        const response = await fetch(`${API_BASE}/storage/active-processes`);
        if (!response.ok) throw new Error("Failed to get active processes");
        return response.json();
    },

    // ✅ 채점 프로세스 강제 중단 (목록에서 제거)
    async deleteActiveProcess(examCode: string): Promise<void> {
        const response = await fetch(`${API_BASE}/storage/active-processes/${examCode}`, {
            method: "DELETE",
        });
        if (!response.ok) throw new Error("Failed to stop process");
    }
};

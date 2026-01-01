import { CreateExamRequest, ExamHistoryItem } from "../types";

const API_BASE = "/api";

// SSE 연결을 위한 타입
export interface BatchPresignRequest {
    examCode: string;
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

    // ✅ 출석부 S3 업로드
    async uploadAttendance(file: File, examCode: string): Promise<{ message: string; s3Url: string }> {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("examCode", examCode);

        const response = await fetch(`${API_BASE}/storage/attendance`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error("Failed to upload attendance");
        return response.json();
    },

    // Presigned URL로 이미지 업로드
    async uploadToPresignedUrl(presignedUrl: string, file: File): Promise<void> {
        const response = await fetch(presignedUrl, {
            method: "PUT",
            body: file,
            headers: {
                "Content-Type": file.type,
            },
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
    }
};

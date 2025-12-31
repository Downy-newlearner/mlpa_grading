import { CreateExamRequest, CreateQuestionsRequest, ExamHistoryItem, PresignedUrlRequest } from "../types";

const API_BASE = "/api";

export const examService = {
    // Fetch all exams
    async getAll(): Promise<ExamHistoryItem[]> {
        const response = await fetch(`${API_BASE}/exams`);
        if (!response.ok) throw new Error("Failed to fetch exams");
        return response.json();
    },

    // Create a new exam
    async create(data: CreateExamRequest): Promise<{ id: string }> {
        const response = await fetch(`${API_BASE}/exams`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("Failed to create exam");
        return response.json();
    },

    // Upload attendance file
    async uploadAttendance(file: File, examId: string): Promise<void> {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("examId", examId);
        formData.append("type", "attendance");

        const response = await fetch(`${API_BASE}/forward`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error("Failed to upload attendance");
    },

    // Create questions (Answer Key)
    async createQuestions(data: CreateQuestionsRequest): Promise<void> {
        const response = await fetch(`${API_BASE}/questions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("Failed to create questions");
    },

    // Init SSE Connection (Just a trigger as per request)
    async connectSSE(examId: string): Promise<void> {
        const response = await fetch(`${API_BASE}/sse/connect?examId=${examId}`);
        if (!response.ok) console.warn("SSE connection init warning");
    },

    // Get Presigned URLs for batch images
    async getPresignedUrls(data: PresignedUrlRequest): Promise<any> {
        const response = await fetch(`${API_BASE}/images/presigned-urls`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        if (!response.ok) throw new Error("Failed to get presigned URLs");
        return response.json();
    }
};

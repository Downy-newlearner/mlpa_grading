/**
 * Saga Pattern - Orchestration 방식
 * 
 * 분산 트랜잭션을 여러 단계로 나누고,
 * 실패 시 이전 단계들을 역순으로 보상(Compensate)하는 패턴
 */

import { examService, BatchPresignResponse } from "./examService";
import { BackendQuestion } from "../types";

// ========================
// Saga Step 인터페이스
// ========================
export interface SagaStep<TContext> {
    name: string;
    execute: (ctx: TContext) => Promise<void>;
    compensate: (ctx: TContext) => Promise<void>;
}

// ========================
// Saga Context (상태 공유)
// ========================
export interface ExamSagaContext {
    // Input
    examName: string;
    examDate: string;
    questions: BackendQuestion[];
    attendanceFile: File | null;
    answerSheetFiles: { file: File; name: string }[];

    // Output (steps에서 채워짐)
    examId?: number;
    examCode?: string;
    eventSource?: EventSource;
    attendanceS3Url?: string;
    presignedUrls?: BatchPresignResponse;
    uploadedImageKeys?: string[]; // 롤백용 S3 키 저장

    // Progress callback
    onProgress?: (message: string) => void;
}

// ========================
// Saga Orchestrator
// ========================
export class SagaOrchestrator<TContext> {
    private steps: SagaStep<TContext>[] = [];
    private executedSteps: SagaStep<TContext>[] = [];

    addStep(step: SagaStep<TContext>): this {
        this.steps.push(step);
        return this;
    }

    async execute(context: TContext): Promise<void> {
        this.executedSteps = [];

        for (const step of this.steps) {
            try {
                console.log(`📌 Executing: ${step.name}`);
                await step.execute(context);
                this.executedSteps.push(step);
                console.log(`✅ Completed: ${step.name}`);
            } catch (error) {
                console.error(`❌ Failed: ${step.name}`, error);
                await this.compensate(context);
                throw error; // 원래 에러를 다시 throw
            }
        }
    }

    private async compensate(context: TContext): Promise<void> {
        console.log("🔄 Starting compensation (rollback)...");

        // 역순으로 보상 실행
        for (let i = this.executedSteps.length - 1; i >= 0; i--) {
            const step = this.executedSteps[i];
            try {
                console.log(`🔙 Compensating: ${step.name}`);
                await step.compensate(context);
                console.log(`✅ Compensated: ${step.name}`);
            } catch (compensateError) {
                console.error(`⚠️ Compensation failed: ${step.name}`, compensateError);
                // 보상 실패는 로깅만 하고 계속 진행
            }
        }

        console.log("🔄 Compensation completed");
    }
}

// ========================
// Saga Steps 구현
// ========================

// Step 1: 시험 생성
export const createExamStep: SagaStep<ExamSagaContext> = {
    name: "시험 생성",
    async execute(ctx) {
        ctx.onProgress?.("시험 생성 중...");

        const result = await examService.create({
            examName: ctx.examName,
            examDate: ctx.examDate,
            questions: ctx.questions,
        });

        ctx.examId = result.examId;
        ctx.examCode = result.examCode;
    },
    async compensate(ctx) {
        ctx.onProgress?.("시험 삭제 중...");

        if (ctx.examCode) {
            await examService.deleteByCode(ctx.examCode);
        }
    }
};

// Step 2: SSE 연결
export const connectSSEStep: SagaStep<ExamSagaContext> = {
    name: "서버 연결",
    async execute(ctx) {
        ctx.onProgress?.("서버 연결 중...");

        if (!ctx.examCode) throw new Error("examCode is required");

        ctx.eventSource = examService.connectSSE(ctx.examCode);

        // 연결 확인을 위해 잠시 대기
        await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => resolve(), 2000); // 2초 대기
            ctx.eventSource!.onerror = () => {
                clearTimeout(timeout);
                reject(new Error("SSE connection failed"));
            };
            ctx.eventSource!.onopen = () => {
                clearTimeout(timeout);
                resolve();
            };
        });
    },
    async compensate(ctx) {
        // SSE 연결 해제
        if (ctx.eventSource) {
            ctx.eventSource.close();
            ctx.eventSource = undefined;
        }
    }
};

// Step 3: 출석부 업로드
export const uploadAttendanceStep: SagaStep<ExamSagaContext> = {
    name: "출석부 업로드",
    async execute(ctx) {
        if (!ctx.attendanceFile || !ctx.examCode) return;

        ctx.onProgress?.("출석부 업로드 중...");

        const result = await examService.uploadAttendance(ctx.attendanceFile, ctx.examCode);
        ctx.attendanceS3Url = result.s3Url;
    },
    async compensate(ctx) {
        // TODO: S3에서 출석부 파일 삭제 (현재는 로깅만)
        if (ctx.attendanceS3Url) {
            console.log("⚠️ Attendance S3 cleanup required:", ctx.attendanceS3Url);
        }
    }
};

// Step 4: 이미지 업로드
export const uploadImagesStep: SagaStep<ExamSagaContext> = {
    name: "이미지 업로드",
    async execute(ctx) {
        if (ctx.answerSheetFiles.length === 0 || !ctx.examCode) return;

        ctx.onProgress?.("이미지 URL 생성 중...");
        ctx.uploadedImageKeys = [];

        // Presigned URL 요청
        const batchRequest = {
            examCode: ctx.examCode,
            total: ctx.answerSheetFiles.length,
            images: ctx.answerSheetFiles.map((f, idx) => ({
                index: idx + 1, // 1-based index to match upload metadata
                contentType: f.file.type || "image/jpeg",
                filename: f.name,
            })),
        };

        const presignedResult = await examService.getBatchPresignedUrls(batchRequest);
        ctx.presignedUrls = presignedResult;

        // 각 파일 업로드
        const total = presignedResult.urls.length;
        for (let i = 0; i < presignedResult.urls.length; i++) {
            const urlInfo = presignedResult.urls[i];
            const file = ctx.answerSheetFiles[urlInfo.index];

            if (file) {
                ctx.onProgress?.(`이미지 업로드 중 (${i + 1}/${total})`);

                // Use the SAME content type as requested for presigning
                const contentType = file.file.type || "image/jpeg";

                await examService.uploadToPresignedUrl(urlInfo.url, file.file, contentType, {
                    total: total,
                    idx: i + 1 // 1-based index
                });

                // 성공한 이미지 키 저장 (롤백용)
                ctx.uploadedImageKeys.push(`uploads/${ctx.examCode}/${urlInfo.index}_${urlInfo.filename}`);
            }
        }
    },
    async compensate(ctx) {
        // TODO: S3에서 업로드된 이미지 삭제 (현재는 로깅만)
        if (ctx.uploadedImageKeys && ctx.uploadedImageKeys.length > 0) {
            console.log("⚠️ Image S3 cleanup required:", ctx.uploadedImageKeys);
        }
    }
};

// ========================
// Saga Factory
// ========================
export function createExamSaga(): SagaOrchestrator<ExamSagaContext> {
    return new SagaOrchestrator<ExamSagaContext>()
        .addStep(createExamStep)
        .addStep(connectSSEStep)
        .addStep(uploadAttendanceStep)
        .addStep(uploadImagesStep);
}

export type QuestionType = "multiple" | "short" | "ox";

export interface SubQuestion {
    id: string;
    text: string;
    score: number | string;
    type: QuestionType;
}

export interface Question {
    id: string;
    text: string;
    score: number | string;
    type: QuestionType;
    subQuestions: SubQuestion[];
}

export interface UploadedFile {
    file: File;
    name: string;
}

export interface ExamHistoryItem {
    examId: string | number;
    examName: string;
    examDate: string;
    examCode?: string;
}

// Backend Question format (matches Question.java entity)
export interface BackendQuestion {
    questionNumber: number;
    questionType: string;
    subQuestionNumber: number | null;
    answer: string;
    answerCount: number;
    point: number;
}

export interface CreateExamRequest {
    examName: string;
    examDate: string;
    questions?: BackendQuestion[];
}

export interface CreateQuestionsRequest {
    examId: string;
    questions: Question[];
}

export interface PresignedUrlRequest {
    files: string[];
    examId: string;
}


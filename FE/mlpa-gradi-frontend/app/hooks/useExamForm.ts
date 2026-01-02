import { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Question, SubQuestion, UploadedFile, QuestionType, BackendQuestion } from "../types";
import { createExamSaga, ExamSagaContext } from "../services/examSaga";

// Helper to generate IDs
const createId = () => typeof window !== 'undefined' ? `${Date.now()}-${Math.random().toString(16).slice(2)}` : "server-id";

const createQuestion = (): Question => ({
    id: createId(),
    text: "",
    score: 0,
    type: "multiple",
    subQuestions: [],
});

const createSubQuestion = (inheritedScore: number | string = 0): SubQuestion => ({
    id: createId(),
    text: "",
    score: inheritedScore,
    type: "multiple",
});

export const useExamForm = () => {
    const router = useRouter();

    // State
    const [questions, setQuestions] = useState<Question[]>([{ ...createQuestion(), id: "initial-q-1" }]);
    const [examTitle, setExamTitle] = useState("");
    const [examDate, setExamDate] = useState("");
    const [attendanceFile, setAttendanceFile] = useState<UploadedFile | null>(null);
    const [answerSheetFiles, setAnswerSheetFiles] = useState<UploadedFile[]>([]);

    // 업로드 진행 상태
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState("");

    // Derived State
    const totalScore = useMemo(() => questions.reduce((acc, q) => acc + (Number(q.score) || 0), 0), [questions]);
    const totalSubCount = useMemo(() => questions.reduce((acc, q) => acc + q.subQuestions.length, 0), [questions]);
    const numberingPreview = useMemo(() => {
        return questions.map((q, qIndex) => ({
            qNo: qIndex + 1,
            subNos: q.subQuestions.map((_, subIndex) => `${qIndex + 1}-${subIndex + 1}`),
        }));
    }, [questions]);

    // Question Handlers
    const addQuestion = useCallback(() => {
        setQuestions((prev) => {
            const lastQ = prev[prev.length - 1];
            const newQ = createQuestion();
            if (lastQ) newQ.score = lastQ.score;
            return [...prev, newQ];
        });
    }, []);

    const removeQuestion = useCallback((qId: string) => {
        setQuestions((prev) => prev.filter((q) => q.id !== qId));
    }, []);

    const updateQuestion = useCallback((qId: string, patch: Partial<Omit<Question, "id" | "subQuestions">>) => {
        setQuestions((prev) => prev.map((q) => (q.id === qId ? { ...q, ...patch } : q)));
    }, []);

    // SubQuestion Handlers
    const addSubQuestion = useCallback((qId: string) => {
        setQuestions((prev) =>
            prev.map((q) => {
                if (q.id === qId) {
                    const lastSub = q.subQuestions[q.subQuestions.length - 1];
                    const inheritedScore = lastSub ? lastSub.score : 0;
                    const newSub = createSubQuestion(inheritedScore);
                    const newSubQuestions = [...q.subQuestions, newSub];
                    const totalScore = newSubQuestions.reduce((acc, curr) => acc + (Number(curr.score) || 0), 0);
                    return { ...q, text: "", score: totalScore, subQuestions: newSubQuestions };
                }
                return q;
            })
        );
    }, []);

    const removeSubQuestion = useCallback((qId: string, sqId: string) => {
        setQuestions((prev) =>
            prev.map((q) => {
                if (q.id === qId) {
                    const newSubQuestions = q.subQuestions.filter((sq) => sq.id !== sqId);
                    const totalScore = newSubQuestions.reduce((acc, curr) => acc + (Number(curr.score) || 0), 0);
                    return { ...q, score: totalScore, subQuestions: newSubQuestions };
                }
                return q;
            })
        );
    }, []);

    const updateSubQuestion = useCallback((qId: string, sqId: string, patch: Partial<Omit<SubQuestion, "id">>) => {
        setQuestions((prev) =>
            prev.map((q) => {
                if (q.id !== qId) return q;
                const newSubQuestions = q.subQuestions.map((sq) => (sq.id === sqId ? { ...sq, ...patch } : sq));
                const totalScore = newSubQuestions.reduce((acc, curr) => acc + (Number(curr.score) || 0), 0);
                return { ...q, score: totalScore, subQuestions: newSubQuestions };
            })
        );
    }, []);

    const insertSubQuestion = useCallback((qId: string, index: number) => {
        setQuestions((prev) =>
            prev.map((q) => {
                if (q.id === qId) {
                    const inheritedScore = index >= 0 && q.subQuestions[index] ? q.subQuestions[index].score : 0;
                    const newSub = createSubQuestion(inheritedScore);
                    const newSubQuestions = [...q.subQuestions];
                    newSubQuestions.splice(index + 1, 0, newSub);
                    const totalScore = newSubQuestions.reduce((acc, curr) => acc + (Number(curr.score) || 0), 0);
                    return { ...q, text: "", score: totalScore, subQuestions: newSubQuestions };
                }
                return q;
            })
        );
    }, []);

    // Workflow Handler
    const handleStartGrading = async () => {
        // Validation
        if (!examTitle.trim() || !examDate) {
            alert("시험 이름과 일시를 입력해주세요.");
            return;
        }
        // Temporarily disabled for testing
        // if (!attendanceFile) {
        //     alert("출석부 파일을 업로드해주세요.");
        //     return;
        // }

        let hasInvalidQuestion = false;
        for (const q of questions) {
            if (q.subQuestions.length > 0) {
                for (const sq of q.subQuestions) {
                    if (!sq.text.trim() && (Number(sq.score) === 0 || !sq.score)) {
                        hasInvalidQuestion = true;
                        break;
                    }
                }
            } else {
                if (!q.text.trim() && (Number(q.score) === 0 || !q.score)) {
                    hasInvalidQuestion = true;
                }
            }
            if (hasInvalidQuestion) break;
        }

        if (hasInvalidQuestion) {
            alert("정답이 입력되지 않고 배점이 0점인 문항이 있습니다. 확인해주세요.");
            return;
        }

        // ✅ 이미지 타입 사전 검증
        const ALLOWED_TYPES = ["image/png", "image/jpg", "image/jpeg"];
        if (answerSheetFiles.length > 0) {
            const invalidFiles = answerSheetFiles.filter(
                (f) => !ALLOWED_TYPES.includes(f.file.type)
            );
            if (invalidFiles.length > 0) {
                alert(`지원하지 않는 이미지 형식입니다.\nPNG, JPG, JPEG만 허용됩니다.\n\n문제 파일: ${invalidFiles.map(f => f.name).join(", ")}`);
                return;
            }
        }

        // ✅ Transform frontend questions to backend format
        const backendQuestions: BackendQuestion[] = [];
        questions.forEach((q, qIndex) => {
            if (q.subQuestions.length > 0) {
                q.subQuestions.forEach((sq, sqIndex) => {
                    backendQuestions.push({
                        questionNumber: qIndex + 1,
                        questionType: sq.type,
                        subQuestionNumber: sqIndex + 1,
                        answer: sq.text,
                        answerCount: 1,
                        point: Number(sq.score) || 0,
                    });
                });
            } else {
                backendQuestions.push({
                    questionNumber: qIndex + 1,
                    questionType: q.type,
                    subQuestionNumber: null,
                    answer: q.text,
                    answerCount: 1,
                    point: Number(q.score) || 0,
                });
            }
        });

        // ✅ Saga 패턴으로 실행
        setIsLoading(true);
        setLoadingMessage("준비 중...");

        const sagaContext: ExamSagaContext = {
            examName: examTitle,
            examDate: examDate,
            questions: backendQuestions,
            attendanceFile: attendanceFile?.file || null,
            answerSheetFiles: answerSheetFiles.map(f => ({ file: f.file, name: f.name })),
            onProgress: setLoadingMessage,
        };

        const saga = createExamSaga();

        try {
            await saga.execute(sagaContext);

            setLoadingMessage("완료!");
            setIsLoading(false);

            alert(`시험이 성공적으로 생성되었습니다!\n시험 코드: ${sagaContext.examCode}`);
            router.push(`/exam/${sagaContext.examId}/loading/student-id?examCode=${sagaContext.examCode}&total=${answerSheetFiles.length}`);
        } catch (error) {
            console.error("Saga failed:", error);
            setIsLoading(false);
            setLoadingMessage("");
            alert("오류가 발생했습니다 (자동 롤백 완료): " + String(error));
        }
    };

    return {
        questions,
        setQuestions,
        examTitle,
        setExamTitle,
        examDate,
        setExamDate,
        attendanceFile,
        setAttendanceFile,
        answerSheetFiles,
        setAnswerSheetFiles,
        totalScore,
        totalSubCount,
        numberingPreview,
        addQuestion,
        removeQuestion,
        updateQuestion,
        addSubQuestion,
        removeSubQuestion,
        updateSubQuestion,
        insertSubQuestion,
        handleStartGrading,
        isLoading,
        loadingMessage,
    };
};

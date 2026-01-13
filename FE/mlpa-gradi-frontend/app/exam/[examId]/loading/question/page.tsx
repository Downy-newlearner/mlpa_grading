"use client";

import React, { useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import QuestionLoading from "../../../../QuestionLoading";
import QuestionRecognitionDone from "../../../../components/QuestionRecognitionDone";
import { examService } from "../../../../services/examService";

const QuestionLoadingPage = () => {
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const total = Number(searchParams.get("total")) || 0;

    // examId from URL path is the examCode
    const examCode = (Array.isArray(params.examId) ? params.examId[0] : params.examId) || "UNKNOWN";
    const [status, setStatus] = useState<"loading" | "done">("loading");

    const handleComplete = React.useCallback(() => {
        setStatus("done");
    }, []);

    const handleNext = async () => {
        try {
            // ✅ AI 서버로부터 인식된 문항 데이터를 가져와 DB에 저장하도록 요청
            await examService.triggerQuestionProxy(examCode);

            // Navigate to grading loading page
            router.push(`/exam/${examCode}/loading/grading`);
        } catch (error) {
            console.error("Failed to trigger question proxy:", error);
            // 에러가 발생해도 일단 페이지 이동은 시도함 (사용자 경험 저해 방지)
            router.push(`/exam/${examCode}/loading/grading`);
        }
    };

    if (status === "loading") {
        return <QuestionLoading examCode={examCode} totalStudents={total} onComplete={handleComplete} />;
    }

    return <QuestionRecognitionDone onNext={handleNext} />;
};

export default QuestionLoadingPage;

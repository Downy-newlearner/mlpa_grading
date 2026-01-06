"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import QuestionLoading from "../../../../QuestionLoading";
import QuestionRecognitionDone from "../../../../components/QuestionRecognitionDone";

const QuestionLoadingPage = () => {
    const params = useParams();
    const router = useRouter();

    // examId from URL path is the examCode
    const examCode = (Array.isArray(params.examId) ? params.examId[0] : params.examId) || "UNKNOWN";
    const [status, setStatus] = useState<"loading" | "done">("loading");

    const handleComplete = React.useCallback(() => {
        setStatus("done");
    }, []);

    const handleNext = () => {
        // Navigate to grading loading page
        router.push(`/exam/${examCode}/loading/grading`);
    };

    if (status === "loading") {
        return <QuestionLoading examCode={examCode} onComplete={handleComplete} />;
    }

    return <QuestionRecognitionDone onNext={handleNext} />;
};

export default QuestionLoadingPage;

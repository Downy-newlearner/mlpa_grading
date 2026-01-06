"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";
import QuestionFeedbackPage from "../../../../components/QuestionFeedbackPage";

const ProblemFeedbackPageRoute = () => {
    const params = useParams();
    const examCode = Array.isArray(params.examId) ? params.examId[0] : params.examId;

    return <QuestionFeedbackPage examCode={examCode} />;
};

export default ProblemFeedbackPageRoute;

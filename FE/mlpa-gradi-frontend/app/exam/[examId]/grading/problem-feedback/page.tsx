"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";
import RecognitionFallbackPage from "../../../../components/RecognitionFallbackPage";

const ProblemFeedbackPageRoute = () => {
    const params = useParams();
    // Using examId from path, treating as examCode
    const examCode = Array.isArray(params.examId) ? params.examId[0] : params.examId;

    return <RecognitionFallbackPage examCode={examCode} />;
};

export default ProblemFeedbackPageRoute;

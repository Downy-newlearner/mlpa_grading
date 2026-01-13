"use client";

import React from "react";
import { useParams } from "next/navigation";
import SubjectiveFallbackPage from "../../../../components/SubjectiveFallbackPage";

const OthersFeedbackRoute = () => {
    const params = useParams();
    const examCode = Array.isArray(params.examId) ? params.examId[0] : params.examId;

    return <SubjectiveFallbackPage examCode={examCode} />;
};

export default OthersFeedbackRoute;

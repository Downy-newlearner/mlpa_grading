"use client";

import React from "react";
import { useParams } from "next/navigation";
import FeedbackPage from "../../../../components/FeedbackPage";

const FeedbackPageRoute = () => {
    const params = useParams();
    const examCode = params.examId as string; // Assuming examId matches examCode in this context

    return <FeedbackPage examCode={examCode} />;
};

export default FeedbackPageRoute;

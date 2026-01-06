"use client";

import React from "react";
import { useParams, useRouter } from "next/navigation";
import GradingDone from "../../../../../components/GradingDone";

const GradingDoneTestPage = () => {
    const params = useParams();
    const router = useRouter();

    const examCode = (Array.isArray(params.examId) ? params.examId[0] : params.examId) || "TEST123";

    const handleNext = () => {
        router.push(`/history`);
    };

    return <GradingDone examCode={examCode} onNext={handleNext} />;
};

export default GradingDoneTestPage;

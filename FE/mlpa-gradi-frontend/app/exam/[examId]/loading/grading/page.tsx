"use client";

import React, { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import GradingLoading from "../../../../GradingLoading";
import GradingDone from "../../../../components/GradingDone";

const GradingLoadingPage = () => {
    const params = useParams();
    const router = useRouter();

    const examCode = (Array.isArray(params.examId) ? params.examId[0] : params.examId) || "UNKNOWN";
    const [status, setStatus] = useState<"loading" | "done">("loading");

    const handleComplete = React.useCallback(() => {
        setStatus("done");
    }, []);

    const handleNext = () => {
        // Navigate to results page with highlight param
        router.push(`/history?highlight=${examCode}`);
    };

    if (status === "loading") {
        return <GradingLoading examCode={examCode} onComplete={handleComplete} />;
    }

    return <GradingDone examCode={examCode} onNext={handleNext} />;
};

export default GradingLoadingPage;

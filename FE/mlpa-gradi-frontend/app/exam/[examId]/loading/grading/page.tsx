"use client";

import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import GradingLoading from "../../../../components/GradingLoading";
import GradingDone from "../../../../components/GradingDone";

const GradingLoadingPage = () => {
    const params = useParams();
    const examId = params.examId as string;
    const [status, setStatus] = useState<"loading" | "done">("loading");

    useEffect(() => {
        const timer = setTimeout(() => {
            setStatus("done");
        }, 3000); // 3 seconds loading

        return () => clearTimeout(timer);
    }, []);

    if (status === "loading") {
        return <GradingLoading examCode={examId} />;
    }

    return <GradingDone examId={examId} />;
};

export default GradingLoadingPage;

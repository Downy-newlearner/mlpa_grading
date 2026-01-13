"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import StudentIdLoading from "../../../../StudentIdLoading";
import StudentIdRecognitionDone from "../../../../components/StudentIdRecognitionDone";

const StudentIdLoadingPage = () => {
    const params = useParams();
    const searchParams = useSearchParams();
    const router = useRouter();

    const examCode = searchParams.get("examCode") || "UNKNOWN";
    const examName = searchParams.get("examName") || "Unknown";
    const total = Number(searchParams.get("total")) || 0;

    const [status, setStatus] = useState<"loading" | "done">("loading");

    const handleComplete = React.useCallback(() => {
        setStatus("done");
    }, []);

    const handleNext = () => {
        router.push(`/exam/${examCode}/grading/feedback`);
    };

    if (status === "loading") {
        return <StudentIdLoading examCode={examCode} examName={examName} totalStudents={total} onComplete={handleComplete} />;
    }

    return <StudentIdRecognitionDone onNext={handleNext} />;
};

export default StudentIdLoadingPage;

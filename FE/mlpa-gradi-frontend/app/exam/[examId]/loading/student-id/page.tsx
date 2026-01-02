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
    const total = Number(searchParams.get("total")) || 40;

    const [status, setStatus] = useState<"loading" | "done">("loading");

    const handleComplete = () => {
        setStatus("done");

        // Optional: Auto redirect after some time if needed, or user clicks button in Done component
        // setTimeout(() => router.push(...), 2000);
    };

    const handleNext = () => {
        // Assuming Feedback page is at /exam/[examId]/feedback
        // Or just /feedback if global? But usually specific to exam. 
        // Let's check where the FeedbackPage is used or routed.
        // Based on finding `app/components/FeedbackPage.tsx` and likely used in `app/exam/[examId]/grading/feedback/page.tsx` or similar?
        // Actually I haven't found the route yet, just the component.
        // Let me search for where FeedbackPage is imported.
        router.push(`/exam/${params.examId}/grading/feedback`);
    };

    if (status === "loading") {
        return <StudentIdLoading examCode={examCode} totalStudents={total} onComplete={handleComplete} />;
    }

    return <StudentIdRecognitionDone onNext={handleNext} />;
};

export default StudentIdLoadingPage;

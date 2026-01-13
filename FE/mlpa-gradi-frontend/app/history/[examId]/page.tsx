"use client";

import React, { useEffect, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import StatisticsDownload from "../../StatisticsDownload";
import { examService } from "../../services/examService";

const HistoryDetailPage: React.FC = () => {
    const params = useParams();
    const searchParams = useSearchParams();

    const examId = params.examId as string;
    const examCode = searchParams.get("code") || "";

    const [examName, setExamName] = useState("");

    useEffect(() => {
        if (examCode) {
            examService.getByCode(examCode)
                .then(exam => setExamName(exam.examName))
                .catch(err => console.error("Failed to fetch exam for title:", err));
        }
    }, [examCode]);

    return (
        <StatisticsDownload
            examTitle={examName || "시험 통계"}
            examCode={examCode}
        />
    );
};

export default HistoryDetailPage;

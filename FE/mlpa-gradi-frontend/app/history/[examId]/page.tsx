"use client";

import React from "react";
import { useParams } from "next/navigation";
import StatisticsDownload from "../../StatisticsDownload";

const HistoryDetailPage: React.FC = () => {
    const params = useParams();
    const examId = params.examId as string;

    // In a real app, we would look up the exam title using the ID.
    // For now, we'll just display the ID or a placeholder.
    const title = `시험 통계 (${examId})`;

    return <StatisticsDownload examTitle={title} />;
};

export default HistoryDetailPage;

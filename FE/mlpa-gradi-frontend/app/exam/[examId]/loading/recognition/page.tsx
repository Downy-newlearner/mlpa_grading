"use client";

import React from "react";
import { useParams } from "next/navigation";
import RecognitionLoading from "../../../../components/RecognitionLoading";

const RecognitionLoadingPage = () => {
    const params = useParams();
    const examId = params.examId as string;

    return <RecognitionLoading examCode={examId} />;
};

export default RecognitionLoadingPage;

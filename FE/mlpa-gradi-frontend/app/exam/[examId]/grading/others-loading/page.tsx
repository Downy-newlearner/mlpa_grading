"use client";

import { useParams } from "next/navigation";
import OthersGradingLoading from "../../../../components/OthersGradingLoading";

export default function OthersGradingLoadingPage() {
    const params = useParams();
    const examId = typeof params?.examId === "string" ? params.examId : "UNKNOWN";

    return (
        <div className="bg-gray-50 text-black min-h-screen">
            <OthersGradingLoading examCode={examId} />
        </div>
    );
}

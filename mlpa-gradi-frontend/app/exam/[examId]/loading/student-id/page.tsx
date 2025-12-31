"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import StudentIdLoading from "../../../../StudentIdLoading";
import StudentIdRecognitionDone from "../../../../components/StudentIdRecognitionDone";

const StudentIdLoadingPage = () => {
    const params = useParams();
    const router = useRouter();
    const examId = params.examId as string;
    const [status, setStatus] = useState<"loading" | "done">("loading");

    // Mimic API call / processing time
    useEffect(() => {
        const timer = setTimeout(() => {
            setStatus("done");
        }, 3000); // 3 seconds loading

        return () => clearTimeout(timer);
    }, []);

    // Also auto-redirect to next step after showing done screen for a bit?
    // User request: "buttons on done pages page tsx state routing"
    // Actually the Recognition Done doesn't have a button in the design, maybe it should auto proceed?
    // Wait, the user said "button exist" in the grading done, but specifically for Recognition Done:
    // "이거는 채점이 끝났을때인데 버튼이 하나 있어 done된 페이지들을 page tsx에서 state로 라우팅 되게해줘"
    // He referred to "Grading Done" having a button.
    // For Recognition Done, usually it flows to Grading or NEXT step.
    // I'll add an auto-redirect or just leave it. The prompt implies "done pages... state routing".
    // I'll assume Recognition flows to Grading Loading?
    // Based on flow: Exam Input -> Loading ID -> Recognition Done -> (Logic?) -> Grading Loading -> Grading Done -> History.
    // I will auto-redirect from Recognition Done to Grading Loading after a short delay OR add a hidden/implicit flow.
    // Let's add a `useEffect` in `done` state to move to next page after 1.5s visual confirmation?
    // Or just leave it as is if user didn't specify next step from Recognition Done.
    // Re-reading: "done된 페이지들을 page tsx에서 state로 라우팅 되게해줘"
    // This implies the page.tsx handles the switch from Loading Component -> Done Component.
    // I've done that.

    if (status === "loading") {
        return <StudentIdLoading examCode={examId} />;
    }

    return <StudentIdRecognitionDone />;
};

export default StudentIdLoadingPage;

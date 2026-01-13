"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";

interface QuestionLoadingProps {
    examCode?: string;
    onComplete?: () => void;
}

const QuestionLoading: React.FC<QuestionLoadingProps> = ({
    examCode = "UNKNOWN",
    onComplete
}) => {
    const router = useRouter();
    const [seconds, setSeconds] = useState(0);
    const [progressCount, setProgressCount] = useState(0);
    const [totalCount, setTotalCount] = useState(0);
    const [timedOut, setTimedOut] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const lastMessageTimeRef = useRef<number>(Date.now());
    const eventSourceRef = useRef<EventSource | null>(null);

    // Timer for elapsed seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setSeconds((prev) => prev + 1);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // ✅ Timeout check
    useEffect(() => {
        const timeoutChecker = setInterval(() => {
            const elapsed = Date.now() - lastMessageTimeRef.current;
            if (elapsed > 5 * 60 * 1000) {
                setTimedOut(true);
                if (eventSourceRef.current) {
                    eventSourceRef.current.close();
                }
                clearInterval(timeoutChecker);
            }
        }, 10000);
        return () => clearInterval(timeoutChecker);
    }, []);

    // ✅ Stop/Rollback logic
    const handleStop = async () => {
        if (isDeleting) return;
        if (confirm("정말로 채점을 중단하고 취소하시겠습니까?\n모든 진행 상황이 삭제됩니다.")) {
            setIsDeleting(true);
            try {
                const { examService } = await import("./services/examService");
                try {
                    await examService.deleteActiveProcess(examCode);
                } catch (e) {
                    console.warn("Session already removed", e);
                }
                await examService.deleteByCode(examCode);
                router.push("/");
            } catch (err) {
                console.error(err);
                setIsDeleting(false);
                alert("중단 처리에 실패했습니다.");
            }
        }
    };

    // Real-time SSE Connection
    useEffect(() => {
        let isCancelled = false;

        // ✅ SSE for real-time updates
        const eventSource = new EventSource(
            `http://127.0.0.1:8080/api/storage/sse/connect?examCode=${examCode}`
        );
        eventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
            if (isCancelled) return;
            lastMessageTimeRef.current = Date.now();
            try {
                const payload = JSON.parse(event.data);
                const type = payload.type;
                const data = (type === "connected") ? payload : payload.data;

                // We expect 'question_recognition_update' or similar
                if (type === "connected" || type === "question_recognition_update" || type === "recognition_update") {
                    if (data.index !== undefined) setProgressCount(data.index);
                    if (data.total !== undefined && data.total > 0) setTotalCount(data.total);

                    if (data.status === "completed") {
                        if (onComplete) onComplete();
                    }
                }
            } catch (err) { }
        };

        return () => {
            isCancelled = true;
            eventSource.close();
        };
    }, [examCode, onComplete]);

    // Safety auto-complete
    useEffect(() => {
        if (totalCount > 0 && progressCount >= totalCount) {
            const timer = setTimeout(() => {
                if (onComplete) onComplete();
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [progressCount, totalCount, onComplete]);

    if (timedOut) {
        return (
            <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center">
                <p className="text-black text-[36px] font-bold mb-4">⚠️ 연결 시간 초과</p>
                <button
                    onClick={() => window.location.reload()}
                    className="mt-8 px-6 py-3 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white rounded-lg font-semibold cursor-pointer"
                >
                    다시 시도
                </button>
            </div>
        );
    }

    return (
        <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center">
            {/* Deleting Overlay */}
            {isDeleting && (
                <div className="absolute inset-0 z-50 bg-white/90 flex flex-col items-center justify-center">
                    <svg className="animate-spin h-16 w-16 text-red-500 mb-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <p className="text-2xl font-bold text-red-600">삭제 중...</p>
                    <p className="text-gray-500 mt-2">잠시만 기다려주세요</p>
                </div>
            )}

            <div
                className="absolute top-[30px] left-[30px] w-[120px] h-[32px] cursor-pointer"
                onClick={() => router.push("/")}
                style={{
                    backgroundImage: "url('/Gradi_logo.png')",
                    backgroundSize: "contain",
                    backgroundRepeat: "no-repeat",
                }}
            />

            <div className="relative mb-14">
                <svg className="animate-spin w-32 h-32" viewBox="0 0 100 100">
                    <defs>
                        <linearGradient id="spinner-gradient-q" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#AC5BF8" />
                            <stop offset="100%" stopColor="#636ACF" />
                        </linearGradient>
                    </defs>
                    <circle cx="50" cy="50" r="45" fill="none" stroke="url(#spinner-gradient-q)" strokeWidth="8" strokeLinecap="round" strokeDasharray="200" strokeDashoffset="100" />
                </svg>
            </div>

            <div className="text-center space-y-6">
                <p className="text-black text-[44px] font-bold leading-tight">
                    답안지의 문항별 정답을 인식하고 있어요
                </p>
                <p className="text-gray-600 text-3xl font-semibold">
                    현재{" "}
                    <span className="inline-block font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        {progressCount}
                    </span>
                    개의 문항 인식을 완료했어요!{" "}
                    <span className="inline-block font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        ({progressCount}/{totalCount > 0 ? totalCount : '...'})
                    </span>
                </p>

                <p className="text-4xl font-extrabold mt-8 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                    시험코드 : {examCode}
                </p>

                <p className="text-2xl font-bold text-gray-400 mt-4">
                    현재 <span className="text-[#AC5BF8]">{seconds}</span>초 경과
                </p>

                {/* Stop/Rollback Button */}
                <button
                    onClick={handleStop}
                    disabled={isDeleting}
                    className={`mt-12 px-8 py-3 bg-white border-2 border-red-500 text-red-500 rounded-xl font-bold transition-all cursor-pointer
                        ${isDeleting ? 'opacity-50 cursor-not-allowed' : 'hover:bg-red-50'}`}
                >
                    {isDeleting ? (
                        <span className="flex items-center gap-2">
                            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            삭제 중...
                        </span>
                    ) : (
                        '채점 중단 및 삭제'
                    )}
                </button>
            </div>
        </div>
    );
};

export default QuestionLoading;

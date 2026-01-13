"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";

interface OthersGradingLoadingProps {
    examCode?: string;
    onComplete?: () => void;
}

const OthersGradingLoading: React.FC<OthersGradingLoadingProps> = ({
    examCode = "UNKNOWN",
    onComplete
}) => {
    const router = useRouter();
    const [seconds, setSeconds] = useState(0);
    const [progressCount, setProgressCount] = useState(0);
    const [totalCount, setTotalCount] = useState(0);
    const [timedOut, setTimedOut] = useState(false);
    const lastMessageTimeRef = useRef<number>(Date.now());
    const eventSourceRef = useRef<EventSource | null>(null);

    // Timer for elapsed seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setSeconds((prev) => prev + 1);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // Timeout check
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

    // SSE Connection
    useEffect(() => {
        let isCancelled = false;
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

                // Listen for updates or completion
                if (type === "connected" || type === "grading_update" || type === "recognition_update") {
                    if (data.index !== undefined) setProgressCount(data.index);
                    if (data.total !== undefined && data.total > 0) setTotalCount(data.total);

                    // If it's completed, we can move forward
                    if (data.status === "completed") {
                        if (onComplete) {
                            onComplete();
                        } else {
                            // Default: back to home or success page
                            setTimeout(() => router.push("/"), 1500);
                        }
                    }
                }
            } catch (err) { }
        };

        return () => {
            isCancelled = true;
            eventSource.close();
        };
    }, [examCode, onComplete, router]);

    if (timedOut) {
        return (
            <div className="relative w-full min-h-screen bg-white mx-auto flex flex-col justify-center items-center">
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
        <div className="relative w-full min-h-screen bg-white mx-auto flex flex-col justify-center items-center">
            {/* Logo */}
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
                        <linearGradient id="spinner-gradient-others" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#AC5BF8" />
                            <stop offset="100%" stopColor="#636ACF" />
                        </linearGradient>
                    </defs>
                    <circle cx="50" cy="50" r="45" fill="none" stroke="url(#spinner-gradient-others)" strokeWidth="8" strokeLinecap="round" strokeDasharray="200" strokeDashoffset="100" />
                </svg>
            </div>

            <div className="text-center space-y-6">
                <p className="text-black text-[44px] font-bold leading-tight">
                    표, 그림을 찾아내고 있어요
                </p>
                <p className="text-gray-600 text-3xl font-semibold">
                    교수님의 피드백을 기반으로 정답을 매칭하고 있어요. <br />
                    인식을 반영했고 표, 그림을 찾아내고 있습니다.
                </p>

                <p className="text-2xl font-bold text-gray-400">
                    진행률: <span className="text-[#AC5BF8]">{progressCount}</span> / {totalCount > 0 ? totalCount : '...'}
                </p>

                <p className="text-4xl font-extrabold mt-8 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                    시험코드 : {examCode}
                </p>

                <p className="text-2xl font-bold text-gray-400 mt-4">
                    현재 <span className="text-[#AC5BF8]">{seconds}</span>초 경과
                </p>
            </div>
        </div>
    );
};

export default OthersGradingLoading;

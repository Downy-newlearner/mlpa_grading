"use client";

import React, { useEffect, useState, useRef } from "react";
import { useRouter } from "next/navigation";

interface StudentIdLoadingProps {
    examCode?: string;
    examName?: string;
    totalStudents?: number;
    onComplete?: () => void;
}

const StudentIdLoading: React.FC<StudentIdLoadingProps> = ({
    examCode = "UNKNOWN",
    examName = "Unknown",
    totalStudents = 0,
    onComplete
}) => {
    const router = useRouter();
    const [seconds, setSeconds] = useState(0);
    const [studentCount, setStudentCount] = useState(0);
    const [totalCount, setTotalCount] = useState(totalStudents);
    const [isVisible, setIsVisible] = useState(true);
    const [timedOut, setTimedOut] = useState(false);
    const lastMessageTimeRef = useRef<number>(Date.now());
    const eventSourceRef = useRef<EventSource | null>(null);
    const [isDeleting, setIsDeleting] = useState(false);

    // Timer for elapsed seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setSeconds((prev) => prev + 1);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // ✅ 5-minute timeout check
    useEffect(() => {
        const timeoutChecker = setInterval(() => {
            const elapsed = Date.now() - lastMessageTimeRef.current;
            if (elapsed > 5 * 60 * 1000) { // 5 minutes
                setTimedOut(true);
                if (eventSourceRef.current) {
                    eventSourceRef.current.close();
                    eventSourceRef.current = null;
                }
                clearInterval(timeoutChecker);
            }
        }, 10000); // Check every 10 seconds
        return () => clearInterval(timeoutChecker);
    }, []);

    // ✅ Back navigation protection - navigate to main on confirm
    useEffect(() => {
        const handlePopState = (e: PopStateEvent) => {
            e.preventDefault();
            const shouldLeave = window.confirm(
                "채점이 진행 중입니다. 정말로 이 페이지를 벗어나시겠습니까?\n\n진행 상황은 메인 화면의 '진행 중인 채점' 목록에서 다시 확인할 수 있습니다."
            );
            if (shouldLeave) {
                // Navigate to main page
                router.push("/");
            } else {
                // Push a dummy state to stay on this page
                window.history.pushState(null, "", window.location.href);
            }
        };

        // Push initial state to enable interception
        window.history.pushState(null, "", window.location.href);
        window.addEventListener("popstate", handlePopState);

        return () => {
            window.removeEventListener("popstate", handlePopState);
        };
    }, [router]);

    const [error, setError] = useState<string | null>(null);

    // ✅ Stop/Rollback logic for this specific exam
    const handleStop = async () => {
        if (isDeleting) return;
        if (confirm("정말로 채점을 중단하고 취소하시겠습니까?\n모든 진행 상황이 삭제됩니다.")) {
            setIsDeleting(true);
            try {
                const { examService } = await import("./services/examService");
                // 1. Stop SSE session
                try {
                    await examService.deleteActiveProcess(examCode);
                } catch (e) {
                    console.warn("Session already removed", e);
                }
                // 2. Rollback exam data
                await examService.deleteByCode(examCode);

                router.push("/");
            } catch (err) {
                console.error(err);
                setIsDeleting(false);
                alert("중단 처리에 실패했습니다.");
            }
        }
    };

    // Real-time SSE Connection - pass examName and total
    useEffect(() => {
        console.log("🛠️ StudentIdLoading useEffect fired!", { examCode, examName, totalStudents });
        let isCancelled = false;

        // ✅ 1. Get initial state immediately (robust against SSE buffering/refresh)
        const fetchInitialState = async () => {
            try {
                const res = await fetch(`/api/storage/progress/${examCode}`);
                if (res.ok) {
                    const data = await res.json();
                    console.log("📥 Initial Progress Fetched:", data);
                    if (!isCancelled) {
                        if (data.index !== undefined) setStudentCount(data.index);
                        if (data.total !== undefined && data.total > 0) setTotalCount(data.total);
                        if (data.status === "completed" && onComplete) onComplete();
                    }
                }
            } catch (err) {
                console.warn("⚠️ Failed to fetch initial state, waiting for SSE...");
            }
        };

        fetchInitialState();

        // ✅ 2. Open SSE for real-time updates (Direct to 8080 to bypass proxy buffering)
        const eventSource = new EventSource(
            `http://127.0.0.1:8080/api/storage/sse/connect?examCode=${examCode}&examName=${encodeURIComponent(examName)}&total=${totalStudents}`
        );
        console.log("📡 EventSource created:", eventSource.url);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
            console.log("✅ SSE Connection Opened!");
        };

        eventSource.onmessage = (event) => {
            if (isCancelled) return;
            try {
                const payload = JSON.parse(event.data);

                // Handle both initial connection and real-time updates through 'type' field
                const type = payload.type;
                const data = (type === "connected") ? payload : payload.data;
                console.log(`✉️ SSE [${type}]:`, data);

                if (type === "connected" || type === "recognition_update") {
                    if (data.index !== undefined) {
                        console.log(`🔢 Updating studentCount: ${data.index}`);
                        setStudentCount(data.index);
                    }
                    if (data.total !== undefined && data.total > 0) {
                        setTotalCount(data.total);
                    }

                    if (data.status === "completed") {
                        console.log("🏁 AI Recognition Completed!");
                        if (onComplete) onComplete();
                    }
                } else if (type === "error_occurred") {
                    console.error("❌ AI Process Error:", data);
                    setError(data.message || "오류가 발생했습니다.");
                }
            } catch (err) {
                console.debug("✉️ Non-JSON message received:", event.data);
            }
        };

        eventSource.onerror = (error) => {
            console.warn("⚠️ SSE Connection issues, browser will retry automatically.");
        };

        return () => {
            isCancelled = true;
            eventSource.close();
        };
    }, [examCode, examName, totalStudents, onComplete]);

    // ✅ 3. Auto-complete check (Safety net if status "completed" event is lost)
    useEffect(() => {
        if (totalCount > 0 && studentCount >= totalCount) {
            console.log("🎯 Progress reached 100%. Waiting for final status or auto-completing...");
            const timer = setTimeout(() => {
                console.log("⚡ Auto-completing after progress reached 100%");
                if (onComplete) onComplete();
            }, 1000); // 1 second buffer
            return () => clearTimeout(timer);
        }
    }, [studentCount, totalCount, onComplete]);

    if (error) {
        return (
            <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center p-10 text-center">
                <p className="text-red-500 text-[36px] font-bold leading-tight mb-4">
                    🚨 인식 오류 발생
                </p>
                <div className="bg-red-50 p-6 rounded-xl border border-red-200 mb-8 max-w-2xl">
                    <p className="text-gray-700 text-xl whitespace-pre-wrap">
                        {error}
                    </p>
                </div>
                <div className="flex space-x-4">
                    <button
                        onClick={() => {
                            setError(null);
                            window.location.reload();
                        }}
                        className="px-6 py-3 bg-gradient-to-r from-gray-500 to-gray-700 text-white rounded-lg font-semibold cursor-pointer"
                    >
                        다시 시도
                    </button>
                    <button
                        onClick={handleStop}
                        className="px-6 py-3 bg-red-500 text-white rounded-lg font-semibold cursor-pointer hover:bg-red-600"
                    >
                        채점 취소 및 삭제
                    </button>
                </div>
            </div>
        );
    }

    if (timedOut) {
        return (
            <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center">
                <p className="text-black text-[36px] font-bold leading-tight mb-4">
                    ⚠️ 연결 시간 초과
                </p>
                <p className="text-gray-500 text-xl">
                    5분 동안 AI 서버로부터 응답이 없어 연결이 종료되었습니다.
                </p>
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

            {/* Gradi Logo - Click to go home */}
            <div
                className="absolute top-[30px] left-[30px] w-[120px] h-[32px] cursor-pointer"
                onClick={() => router.push("/")}
                style={{
                    backgroundImage: "url('/Gradi_logo.png')",
                    backgroundSize: "contain",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Large Gradient Spinner Animation */}
            <div className="relative mb-14">
                <svg
                    className="animate-spin w-32 h-32"
                    viewBox="0 0 100 100"
                    xmlns="http://www.w3.org/2000/svg"
                >
                    <defs>
                        <linearGradient id="spinner-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#AC5BF8" />
                            <stop offset="100%" stopColor="#636ACF" />
                        </linearGradient>
                    </defs>
                    <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="url(#spinner-gradient)"
                        strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray="200"
                        strokeDashoffset="100"
                    />
                </svg>
            </div>

            {/* Main Text */}
            <div className="text-center space-y-6">
                <p className="text-black text-[44px] font-bold leading-tight">
                    학생들의 학번을 인식하고 있어요
                </p>
                {/* Animated Paragraph: Always visible with purple gradient */}
                <p className="text-gray-600 text-3xl font-semibold">
                    학생{" "}
                    <span className="inline-block font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        {studentCount}
                    </span>
                    명의 인식을 완료했어요!{" "}
                    <span className="inline-block font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        ({studentCount}/{totalCount > 0 ? totalCount : totalStudents})
                    </span>
                </p>

                {/* Exam Code */}
                <p className="text-4xl font-extrabold mt-8 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                    시험코드 : {examCode}
                </p>

                {/* Timer */}
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

export default StudentIdLoading;

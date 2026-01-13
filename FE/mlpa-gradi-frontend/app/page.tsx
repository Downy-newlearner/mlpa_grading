"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import Button from "@/app/components/Button";
import { examService } from "@/app/services/examService";

interface ActiveProcess {
    examCode: string;
    examName: string;
    index: number;
    total: number;
    status: string;
    lastUpdateTime: number;
}

const Landing: React.FC = () => {
    const [activeProcesses, setActiveProcesses] = useState<ActiveProcess[]>([]);
    const [showProcessList, setShowProcessList] = useState(false);
    const [deletingCode, setDeletingCode] = useState<string | null>(null);

    useEffect(() => {
        const fetchActiveProcesses = async () => {
            try {
                const processes = await examService.getActiveProcesses();
                setActiveProcesses(processes);
            } catch (error) {
                console.error("Failed to fetch active processes:", error);
            }
        };

        fetchActiveProcesses();
        // Poll every 10 seconds
        const interval = setInterval(fetchActiveProcesses, 10000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="relative w-[1152px] min-h-[700px] bg-white mx-auto pb-8">
            {/* Gradi Logo - Redirect to Home */}
            <Link
                href="/"
                className="absolute w-[165px] h-[43px] top-[17px] left-[10px] cursor-pointer"
                style={{
                    backgroundImage: "url(/Gradi_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Center Image */}
            <div
                className="absolute w-[315px] h-[315px] left-[418px] top-[129.04px] animate-fade-in-up"
                style={{
                    backgroundImage: "url(/Dankook_emblem.png)",
                    backgroundSize: "contain",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            ></div>



            {/* Left Button -> /exam-input */}
            <div className="absolute top-[521.52px] left-[258px] animate-fade-in-up">
                <Link href="/exam-input" className="inline-block cursor-pointer">
                    <Button label="시험 채점" />
                </Link>
            </div>

            {/* Right Button -> /history */}
            <div className="absolute top-[521.52px] left-[677px] animate-fade-in-up">
                <Link href="/history" className="inline-block cursor-pointer">
                    <Button label="결과 확인" />
                </Link>
            </div>

            {/* Makers Link */}
            <div className="absolute top-[30px] right-[40px] animate-fade-in-up">
                <Link
                    href="/makers"
                    className="font-extrabold text-xl bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent hover:opacity-80 transition-opacity cursor-pointer"
                >
                    만든 사람들
                </Link>
            </div>

            {/* ✅ Ongoing Grading Button - Only show if there are active processes */}
            {activeProcesses.length > 0 && (
                <div className="absolute top-[640px] left-1/2 transform -translate-x-1/2 animate-fade-in-up">
                    <button
                        onClick={() => setShowProcessList(true)}
                        className="px-6 py-3 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white rounded-lg font-semibold shadow-lg hover:scale-105 transition-transform cursor-pointer"
                    >
                        채점 진행중인 시험 확인하기 ({activeProcesses.length})
                    </button>
                </div>
            )}

            {/* ✅ Modal for Active Processes */}
            {showProcessList && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-xl p-6 w-[500px] max-h-[500px] overflow-y-auto shadow-2xl">
                        <div className="flex justify-between items-center mb-4">
                            <h2 className="text-2xl font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                                진행 중인 채점
                            </h2>
                            <button
                                onClick={() => setShowProcessList(false)}
                                className="text-gray-500 hover:text-gray-700 text-2xl cursor-pointer"
                            >
                                ✕
                            </button>
                        </div>
                        {activeProcesses.length === 0 ? (
                            <p className="text-gray-500 text-center py-8">진행 중인 채점이 없습니다.</p>
                        ) : (
                            <div className="space-y-3">
                                {activeProcesses.map((process) => {
                                    const isDeleting = deletingCode === process.examCode;
                                    return (
                                        <div
                                            key={process.examCode}
                                            className={`relative group block p-4 rounded-xl bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white shadow-lg transition-all ${isDeleting ? 'opacity-60 scale-[0.98]' : 'hover:scale-[1.02]'}`}
                                        >
                                            {/* Deleting Overlay */}
                                            {isDeleting && (
                                                <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-xl z-20">
                                                    <div className="flex items-center gap-2 bg-white/90 px-4 py-2 rounded-full shadow-lg">
                                                        <svg className="animate-spin h-5 w-5 text-red-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                        </svg>
                                                        <span className="text-red-600 font-bold text-sm">삭제 중...</span>
                                                    </div>
                                                </div>
                                            )}

                                            <Link
                                                href={process.status === "completed"
                                                    ? `/exam/${process.examCode}/grading/feedback`
                                                    : `/exam/0/loading/student-id?examCode=${process.examCode}&total=${process.total}`
                                                }
                                                onClick={(e) => {
                                                    if (isDeleting) {
                                                        e.preventDefault();
                                                        return;
                                                    }
                                                    setShowProcessList(false);
                                                }}
                                                className={`block w-full h-full ${isDeleting ? 'pointer-events-none' : ''}`}
                                            >
                                                <div className="flex justify-between items-center">
                                                    <div>
                                                        <p className="font-bold text-lg">{process.examCode}</p>
                                                        <p className="text-sm opacity-90">
                                                            {process.index} / {process.total} 완료
                                                        </p>
                                                    </div>
                                                </div>
                                            </Link>

                                            {/* ⛔ Stop/Rollback Button */}
                                            <button
                                                disabled={isDeleting}
                                                onClick={async (e) => {
                                                    e.preventDefault();
                                                    e.stopPropagation();
                                                    if (isDeleting) return;
                                                    if (confirm("정말로 채점을 중단하고 취소하시겠습니까?\n모든 진행 상황이 삭제됩니다.")) {
                                                        setDeletingCode(process.examCode);
                                                        try {
                                                            // 1. Stop active process (SSE disconnect) - ignore errors
                                                            try {
                                                                await examService.deleteActiveProcess(process.examCode);
                                                            } catch {
                                                                // Session already removed, ignore
                                                            }

                                                            // 2. Rollback exam data (Delete from DB) - ignore errors (idempotent)
                                                            try {
                                                                await examService.deleteByCode(process.examCode);
                                                            } catch {
                                                                // Already deleted, ignore
                                                            }

                                                            // Immediate local state update (this is the important visual feedback)
                                                            setActiveProcesses(prev => prev.filter(p => p.examCode !== process.examCode));

                                                            // Refresh full list from server (optional, ignore errors)
                                                            try {
                                                                const updated = await examService.getActiveProcesses();
                                                                setActiveProcesses(updated);
                                                                if (updated.length === 0) setShowProcessList(false);
                                                            } catch {
                                                                // Refresh failed, but deletion was successful
                                                                if (activeProcesses.length <= 1) setShowProcessList(false);
                                                            }
                                                        } finally {
                                                            setDeletingCode(null);
                                                        }
                                                    }
                                                }}
                                                className={`absolute top-1/2 right-4 transform -translate-y-1/2 bg-white rounded-full w-8 h-8 flex items-center justify-center font-bold shadow-md z-10 transition-all
                                                    ${isDeleting ? 'text-gray-400 cursor-not-allowed' : 'text-red-500 hover:bg-red-100 cursor-pointer'}`}
                                                title="채점 중단 및 삭제"
                                            >
                                                {isDeleting ? (
                                                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                ) : (
                                                    '⛔'
                                                )}
                                            </button>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default Landing;
"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, useParams, useSearchParams } from "next/navigation";
import Image from "next/image";

export default function StudentResultPage() {
    const router = useRouter();
    const params = useParams();
    const examCode = (params?.examCode as string) || "";

    // Mock Data Map (Updated to use ExamCode keys)
    const MOCK_EXAM_DETAILS: Record<string, { title: string; date: string; score: number; total: number; comment: string; imageUrl?: string }> = {
        "AI202501": { title: "인공지능 2025-1학기 중간고사", date: "2025.04.20", score: 85, total: 100, comment: "전반적으로 우수한 성적입니다. 특히 딥러닝 기초 개념에 대한 이해도가 높습니다. 다만 역전파 알고리즘의 유도 과정에서 계산 실수가 있어 아쉽게 감점되었습니다.", imageUrl: "/Gradi_logo.png" }, // Mock image
        "DB202501": { title: "데이터베이스 시스템 기말고사", date: "2025.06.15", score: 92, total: 100, comment: "정규화 과정에 대한 완벽한 이해를 보여주었습니다. SQL 쿼리 최적화 부분도 훌륭합니다." },
        "OS202502": { title: "운영체제 중간고사", date: "2025.10.20", score: 78, total: 100, comment: "프로세스 스케줄링 알고리즘에 대한 개념 정립이 필요합니다. 데드락 회피 기법은 잘 서술하였습니다." },
        "TEST123": { title: "테스트 시험", date: "2026.01.11", score: 90, total: 100, comment: "테스트 시험 코멘트입니다." },
    };

    const examData = MOCK_EXAM_DETAILS[examCode] || { title: "알 수 없는 시험", date: "-", score: 0, total: 100, comment: "시험 정보를 찾을 수 없습니다." };

    // Mock Data State
    const [studentId, setStudentId] = useState("3220****");
    const [score, setScore] = useState(examData.score);
    const [totalScore, setTotalScore] = useState(examData.total);

    // Zoom State
    const [zoomedImage, setZoomedImage] = useState<string | null>(null);
    const [zoomScale, setZoomScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    // Zoom Handlers
    const resetZoom = useCallback(() => {
        setZoomScale(1);
        setPosition({ x: 0, y: 0 });
    }, []);

    const handleZoomIn = (e?: React.MouseEvent) => {
        e?.stopPropagation();
        setZoomScale(prev => Math.min(prev + 0.5, 4));
    };

    const handleZoomOut = (e?: React.MouseEvent) => {
        e?.stopPropagation();
        setZoomScale(prev => {
            const next = Math.max(prev - 0.5, 1);
            if (next === 1) setPosition({ x: 0, y: 0 });
            return next;
        });
    };

    const handleMouseDown = (e: React.MouseEvent) => {
        if (zoomScale <= 1) return;
        setIsDragging(true);
        setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!isDragging || zoomScale <= 1) return;
        setPosition({
            x: e.clientX - dragStart.x,
            y: e.clientY - dragStart.y
        });
    };

    const handleMouseUp = () => {
        setIsDragging(false);
    };

    const handleImageClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        if (zoomScale === 1) {
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const offsetX = (e.clientX - rect.left - rect.width / 2) * -2;
            const offsetY = (e.clientY - rect.top - rect.height / 2) * -2;
            setZoomScale(2);
            setPosition({ x: offsetX, y: offsetY });
        }
    };

    const searchParams = useSearchParams();

    useEffect(() => {
        // Check verification
        const verifiedId = sessionStorage.getItem(`verified_student_id_${examCode}`);
        const forceId = searchParams?.get("force_id");

        if (!verifiedId && !forceId) {
            router.push(`/student/verify/${examCode}`);
            return;
        }

        const finalId = forceId || verifiedId || "";

        // Mask ID (Show first 4, mask middle 4? User said "middle 4 masked")
        // Example: 32204041 -> 3220****
        if (finalId.length >= 8) {
            const masked = finalId.substring(0, 4) + "****";
            setStudentId(masked);
        }
    }, [router, examCode, searchParams]);

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
            <div className="w-full max-w-4xl space-y-8 animate-fade-in-up">

                {/* Header / Nav */}
                <div className="flex justify-between items-center">
                    <div className="w-[165px] h-[43px] relative">
                        <Image
                            src="/Gradi_logo.png"
                            alt="Gradi Logo"
                            fill
                            className="object-cover"
                        />
                    </div>
                    <button
                        onClick={() => {
                            localStorage.removeItem("student_email");
                            sessionStorage.removeItem(`verified_student_id_${examCode}`);
                            router.push("/student/login");
                        }}
                        className="text-gray-500 hover:text-purple-600 font-medium transition-colors cursor-pointer"
                    >
                        로그아웃
                    </button>
                </div>

                {/* Score Card */}
                <div className="bg-white rounded-3xl shadow-xl overflow-hidden border border-purple-100">
                    <div className="bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] p-8 text-white text-center">
                        <h1 className="text-3xl font-bold mb-2">{examData.title}</h1>
                        <p className="opacity-90">시험일시: {examData.date}</p>
                    </div>
                    <div className="p-8 text-center space-y-6">
                        <div className="inline-block p-4 rounded-2xl bg-purple-50">
                            <span className="block text-gray-500 text-sm mb-1 font-semibold">학번</span>
                            <span className="text-2xl font-mono font-bold text-gray-800 tracking-wider">
                                {studentId}
                            </span>
                        </div>

                        <div className="flex justify-center items-end gap-2">
                            <span className="text-6xl font-extrabold text-[#AC5BF8]">{score}</span>
                            <span className="text-3xl font-bold text-gray-400 mb-2">/ {totalScore}점</span>
                        </div>

                        <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                            <div
                                className="bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] h-full rounded-full transition-all duration-1000 ease-out"
                                style={{ width: `${(score / totalScore) * 100}%` }}
                            />
                        </div>
                    </div>
                </div>

                {/* Answer Sheet Preview (Mock) */}
                <div className="bg-white rounded-3xl shadow-xl p-8 border border-purple-100 space-y-6">
                    <h2 className="text-2xl font-bold text-gray-800 border-l-4 border-[#AC5BF8] pl-4">
                        내 답안지 확인
                    </h2>

                    <div
                        className="relative aspect-[3/4] w-full bg-gray-100 rounded-xl overflow-hidden border-2 border-dashed border-gray-300 flex items-center justify-center group cursor-pointer hover:border-purple-400 transition-colors"
                        onClick={() => setZoomedImage(examData.imageUrl || "/Gradi_logo.png")} // Fallback to logo for demo
                    >
                        {examData.imageUrl ? (
                            <>
                                <Image
                                    src={examData.imageUrl}
                                    alt="Answer Sheet"
                                    fill
                                    className="object-contain p-4 group-hover:scale-105 transition-transform duration-300"
                                />
                                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 flex items-center justify-center transition-colors">
                                    <span className="opacity-0 group-hover:opacity-100 bg-black/50 text-white px-4 py-2 rounded-full text-sm font-bold backdrop-blur-sm transform translate-y-2 group-hover:translate-y-0 transition-all">
                                        클릭하여 확대하기
                                    </span>
                                </div>
                            </>
                        ) : (
                            /* Placeholder for Answer Sheet Image */
                            <div className="text-center p-8 group-hover:scale-105 transition-transform">
                                <span className="text-6xl mb-4 block">📄</span>
                                <p className="text-gray-500 font-medium text-lg group-hover:text-purple-600 transition-colors">
                                    답안지 미리보기
                                </p>
                                <p className="text-sm text-gray-400 mt-2">
                                    클릭하면 확대하여 볼 수 있습니다
                                </p>
                                <div className="mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <span className="text-xs bg-purple-100 text-purple-700 px-3 py-1 rounded-full font-bold">
                                        확대하기 🔍
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-green-50 rounded-xl border border-green-100">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="w-2 h-2 rounded-full bg-green-500" />
                                <h3 className="font-bold text-green-700">정답 문항</h3>
                            </div>
                            <p className="text-green-600 font-medium">1, 2, 3, 5, 7, 8, 10</p>
                        </div>
                        <div className="p-4 bg-red-50 rounded-xl border border-red-100">
                            <div className="flex items-center gap-2 mb-2">
                                <span className="w-2 h-2 rounded-full bg-red-500" />
                                <h3 className="font-bold text-red-700">오답 문항</h3>
                            </div>
                            <p className="text-red-600 font-medium">4, 6, 9</p>
                        </div>
                    </div>
                    <div className="mt-6 p-6 bg-purple-50 rounded-2xl border border-purple-100 animate-fade-in-up delay-100">
                        <div className="flex items-start gap-3">
                            <span className="text-2xl">💡</span>
                            <div>
                                <h3 className="font-bold text-purple-800 mb-1">교수님 코멘트</h3>
                                <p className="text-purple-700 leading-relaxed text-sm">
                                    {examData.comment}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="text-center text-gray-400 text-sm">
                    © 2026 Trender Team. All rights reserved.
                </div>
            </div>

            {/* Fixed Bottom Bar for Statistics Download */}
            <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-purple-100 p-4 shadow-[0_-5px_20px_rgba(0,0,0,0.05)] animate-slide-up">
                <div className="max-w-4xl mx-auto flex justify-between items-center">
                    <div className="flex flex-col">
                        <span className="text-sm text-gray-500 font-bold">전체 통계가 궁금하다면?</span>
                        <span className="text-lg font-bold text-gray-800">분반 전체 성적 분포 확인하기</span>
                    </div>
                    <button
                        onClick={() => alert("통계 PDF 다운로드 기능은 준비 중입니다. (Mock Download)")}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white rounded-xl font-bold hover:shadow-lg hover:shadow-purple-200 hover:-translate-y-0.5 transition-all cursor-pointer"
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                            <polyline points="14 2 14 8 20 8"></polyline>
                            <line x1="16" y1="13" x2="8" y2="13"></line>
                            <line x1="16" y1="17" x2="8" y2="17"></line>
                            <polyline points="10 9 9 9 8 9"></polyline>
                        </svg>
                        통계 PDF 다운로드
                    </button>
                </div>
            </div>
            <div className="h-20" /> {/* Spacer for fixed bottom bar */}

            {/* Lightbox / Advanced Zoom Modal */}
            {zoomedImage && (
                <div
                    className="fixed inset-0 z-[100] bg-black/95 flex flex-col items-center justify-center p-4 select-none"
                    onClick={() => setZoomedImage(null)}
                >
                    <div
                        className={`relative w-[90vw] h-[80vh] overflow-hidden rounded-xl border border-white/10 shadow-2xl bg-black/20 flex items-center justify-center
                            ${zoomScale > 1 ? (isDragging ? "cursor-grabbing" : "cursor-grab") : "cursor-crosshair"}`}
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                    >
                        {/* Use next/image or standard img tag. Using img for easier transform manipulation */}
                        <img
                            src={zoomedImage}
                            alt="Answer Sheet Zoomed"
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={handleImageClick}
                            style={{
                                transform: `translate(${position.x}px, ${position.y}px) scale(${zoomScale})`,
                                transition: isDragging ? "none" : "transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)"
                            }}
                            className="max-w-full max-h-full object-contain pointer-events-auto"
                        />
                    </div>

                    {/* Zoom & Navigation Help */}
                    <div className="absolute top-8 left-1/2 -translate-x-1/2 flex gap-4">
                        <div className="bg-white/10 backdrop-blur-md px-6 py-2 rounded-full border border-white/20 text-white/80 text-sm font-medium">
                            💡 {zoomScale > 1 ? "드래그하여 이동 / 클릭 시 닫기" : "원하는 위치를 클릭하여 확대해보세요"}
                        </div>
                    </div>

                    {/* Controls */}
                    <div className="mt-8 flex items-center gap-6 bg-white/10 backdrop-blur-lg px-8 py-4 rounded-full border border-white/20 shadow-2xl scale-110" onClick={(e) => e.stopPropagation()}>
                        <button
                            onClick={handleZoomOut}
                            className="w-12 h-12 flex items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 transition-all text-2xl font-bold"
                        >
                            －
                        </button>
                        <div className="flex flex-col items-center min-w-[80px]">
                            <span className="text-white font-black text-xl">
                                {Math.round(zoomScale * 100)}%
                            </span>
                            <button
                                onClick={resetZoom}
                                className="text-[10px] text-white/50 hover:text-white uppercase tracking-widest mt-1"
                            >
                                Reset
                            </button>
                        </div>
                        <button
                            onClick={handleZoomIn}
                            className="w-12 h-12 flex items-center justify-center rounded-full bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white hover:brightness-110 shadow-lg transition-all text-2xl font-bold"
                        >
                            ＋
                        </button>
                    </div>

                    <button
                        className="absolute top-8 right-8 w-12 h-12 flex items-center justify-center rounded-full bg-white/10 text-white text-2xl hover:bg-red-500 transition-all"
                        onClick={() => setZoomedImage(null)}
                    >
                        ✕
                    </button>
                </div>
            )}
        </div>
    );
}

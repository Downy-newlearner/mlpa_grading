"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";

type FeedbackItem = {
    id: string;
    imageUrl?: string;
    questionNumber: string;  // 문항 번호
    recognizedAnswer: string;  // 인식된 답
    correctAnswer: string;  // 수정된 정답
};

interface QuestionFeedbackPageProps {
    examCode?: string;
}

const QuestionFeedbackPage: React.FC<QuestionFeedbackPageProps> = ({ examCode = "UNKNOWN" }) => {
    const [items, setItems] = useState<FeedbackItem[]>([]);
    const [focusedIndex, setFocusedIndex] = useState(0);
    const [zoomedImage, setZoomedImage] = useState<string | null>(null);
    const [zoomScale, setZoomScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [isSubmitting, setIsSubmitting] = useState(false);

    const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
    const router = useRouter();

    // ✅ Load from API & localStorage draft
    useEffect(() => {
        const fetchUnknownQuestions = async () => {
            try {
                // TODO: Replace with actual API endpoint for unknown questions
                const response = await fetch(`/api/reports/unknown-questions/${examCode}`);
                if (!response.ok) {
                    // If no unknown questions, show empty state
                    setItems([]);
                    return;
                }
                const data = await response.json();

                const savedDraft = localStorage.getItem(`gradi_question_draft_${examCode}`);
                const draftMap = savedDraft ? JSON.parse(savedDraft) : {};

                const initializedItems = data.map((item: any, index: number) => {
                    return {
                        id: String(index),
                        imageUrl: item.imageUrl,
                        questionNumber: item.questionNumber || `Q${index + 1}`,
                        recognizedAnswer: item.recognizedAnswer || "",
                        correctAnswer: draftMap[item.questionNumber] || item.recognizedAnswer || ""
                    };
                });

                setItems(initializedItems);
            } catch (error) {
                console.error("Failed to fetch unknown questions:", error);
                setItems([]);
            }
        };

        fetchUnknownQuestions();
    }, [examCode]);

    // ✅ Keep localStorage in sync
    useEffect(() => {
        if (items.length === 0) return;

        const draft = items.reduce((acc, item) => {
            if (item.correctAnswer) {
                acc[item.questionNumber] = item.correctAnswer;
            }
            return acc;
        }, {} as Record<string, string>);

        localStorage.setItem(`gradi_question_draft_${examCode}`, JSON.stringify(draft));
    }, [items, examCode]);

    const handleKeyDown = (e: React.KeyboardEvent, index: number) => {
        if (e.key === "ArrowDown" || (e.key === "Enter" && !e.shiftKey)) {
            e.preventDefault();
            const nextIndex = Math.min(index + 1, items.length - 1);
            setFocusedIndex(nextIndex);
            inputRefs.current[nextIndex]?.focus();
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            const prevIndex = Math.max(index - 1, 0);
            setFocusedIndex(prevIndex);
            inputRefs.current[prevIndex]?.focus();
        }
    };

    const handleInputChange = (index: number, value: string) => {
        const newItems = [...items];
        newItems[index].correctAnswer = value;
        setItems(newItems);
    };

    // --- Lightbox Zoom & Pan Logic ---
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

    const handleSubmit = async () => {
        if (isSubmitting) return;

        // If no items, navigate to grading loading page
        if (items.length === 0) {
            router.push(`/exam/${examCode}/loading/grading`);
            return;
        }

        setIsSubmitting(true);
        const payload = {
            examCode: examCode,
            questions: items.map(item => ({
                questionNumber: item.questionNumber,
                correctAnswer: item.correctAnswer
            }))
        };

        try {
            // TODO: Replace with actual API endpoint
            const res = await fetch("/api/question-feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                console.log("Question feedback submitted successfully");
                localStorage.removeItem(`gradi_question_draft_${examCode}`);
                // Navigate to grading loading page
                router.push(`/exam/${examCode}/loading/grading`);
            } else {
                const errorText = await res.text();
                console.error("Feedback error:", errorText);
                alert("오류가 발생했습니다: " + errorText);
            }
        } catch (err) {
            console.error(err);
            alert("네트워크 오류가 발생했습니다.");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="relative mx-auto min-h-screen w-[1152px] bg-white pb-32">
            {/* Logo */}
            <div
                className="absolute left-[10px] top-[17px] h-[43px] w-[165px] cursor-pointer z-50"
                onClick={() => router.push("/")}
                style={{
                    backgroundImage: "url('/Gradi_logo.png')",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Title Section */}
            <div className="pt-[100px] pb-6 flex justify-between items-end mb-4 px-6">
                <div className="flex flex-col gap-1">
                    <h1 className="text-[40px] font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        문항 인식 피드백
                    </h1>
                    <p className="text-[18px] font-medium text-[#A0A0A0]">
                        모델이 인식 중 불확실한 문항들을 사용자에게 피드백 받습니다.
                    </p>
                </div>
                <div className="bg-white/90 border border-[#AC5BF8]/20 px-5 py-3 rounded-2xl shadow-sm flex flex-col gap-1.5">
                    <div className="flex items-center gap-2 text-xs font-bold">
                        <span className="text-sm">💡</span>
                        <span className="bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                            이동: 화살표 ↑ ↓ / 다음: Enter
                        </span>
                    </div>
                    <div className="flex items-center gap-2 text-xs font-bold">
                        <span className="text-sm">🔍</span>
                        <span className="bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                            클릭 시 해당 부위 확대 / 드래그로 이동 가능
                        </span>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <section className="w-full bg-[#F8F0FF] rounded-2xl p-8 min-h-[600px] shadow-sm border-[3px] border-[#AC5BF8] mb-12">
                <div className="space-y-4">
                    {items.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-[400px] text-gray-400">
                            <p className="text-2xl font-semibold">피드백이 필요한 문항이 없습니다.</p>
                            <p className="mt-2 text-lg">모든 문항 인식이 성공적으로 완료되었습니다.</p>
                        </div>
                    ) : (
                        items.map((item, index) => (
                            <div
                                key={item.id}
                                className={`relative flex items-center p-8 rounded-xl transition-all duration-300 border-[3px] shadow-md
                                    ${focusedIndex === index
                                        ? "border-[#AC5BF8] bg-white scale-[1.01] ring-4 ring-[#AC5BF8]/10"
                                        : "border-gray-200 bg-white/60"
                                    }`}
                                onClick={() => {
                                    setFocusedIndex(index);
                                    inputRefs.current[index]?.focus();
                                }}
                            >
                                <div
                                    className="w-[450px] h-[220px] bg-white border-2 border-[#D9D9D9] rounded-lg overflow-hidden flex items-center justify-center shadow-inner cursor-zoom-in group"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        if (item.imageUrl) {
                                            setZoomedImage(item.imageUrl);
                                            resetZoom();
                                        }
                                    }}
                                >
                                    {item.imageUrl ? (
                                        <div className="relative w-full h-full">
                                            <img
                                                src={item.imageUrl}
                                                alt={`Question ${index + 1}`}
                                                onError={(e) => {
                                                    (e.currentTarget as HTMLImageElement).src = "";
                                                    (e.currentTarget as HTMLImageElement).alt = "이미지 로드 실패";
                                                }}
                                                className="h-full w-full object-contain transition-transform group-hover:scale-105"
                                            />
                                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 flex items-center justify-center transition-colors">
                                                <span className="opacity-0 group-hover:opacity-100 bg-black/50 text-white px-3 py-1 rounded-full text-xs transition-opacity font-bold">
                                                    클릭하여 확대
                                                </span>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex flex-col items-center gap-2">
                                            <span className="text-3xl">📝</span>
                                            <div className="text-gray-400 font-medium text-sm">문항 {item.questionNumber}</div>
                                        </div>
                                    )}
                                </div>

                                <div className="flex-1 ml-12">
                                    <div className="flex justify-between items-center mb-4">
                                        <h3 className={`text-lg font-bold ${focusedIndex === index ? "bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent" : "text-gray-400"}`}>
                                            문항 #{item.questionNumber}
                                        </h3>
                                        {focusedIndex === index && (
                                            <span className="text-[#AC5BF8] text-xs font-bold animate-pulse">
                                                입력 중...
                                            </span>
                                        )}
                                    </div>
                                    <div className="space-y-3">
                                        <div className="flex items-center gap-4">
                                            <span className="text-sm text-gray-500 w-24">인식된 답:</span>
                                            <span className="text-lg font-semibold text-gray-700">{item.recognizedAnswer || "-"}</span>
                                        </div>
                                        <label className={`text-base font-bold ${focusedIndex === index ? "bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent" : "text-gray-700"}`}>
                                            정답을 수정해주세요
                                        </label>
                                        <input
                                            ref={(el) => { inputRefs.current[index] = el; }}
                                            type="text"
                                            value={item.correctAnswer}
                                            onChange={(e) => handleInputChange(index, e.target.value)}
                                            onKeyDown={(e) => handleKeyDown(e, index)}
                                            onFocus={() => setFocusedIndex(index)}
                                            placeholder="정답을 입력하세요 (ex: 1, 2, 3, 4, 5)"
                                            className={`w-full h-[64px] px-6 text-[24px] font-bold rounded-lg border-2 transition-all 
                                                ${focusedIndex === index
                                                    ? "border-[#AC5BF8] bg-[#FDF8FF] text-black"
                                                    : "border-gray-100 bg-gray-50 text-gray-300 shadow-none"
                                                } focus:outline-none`}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </section>

            {/* Bottom Button */}
            <div className="flex justify-center pt-8">
                <button
                    onClick={handleSubmit}
                    disabled={isSubmitting || (items.length > 0 && items.some(i => !i.correctAnswer.trim()))}
                    className="w-[300px] px-4 py-4 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] rounded-lg text-white text-xl font-bold shadow-xl hover:scale-105 active:scale-95 transition-all cursor-pointer disabled:opacity-50 disabled:grayscale disabled:cursor-not-allowed"
                >
                    {isSubmitting ? (
                        <span className="flex items-center justify-center gap-2">
                            <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            제출 중...
                        </span>
                    ) : (
                        '채점 마무리하기'
                    )}
                </button>
            </div>

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
                        <img
                            src={zoomedImage}
                            alt="Zoomed question"
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
};

export default QuestionFeedbackPage;

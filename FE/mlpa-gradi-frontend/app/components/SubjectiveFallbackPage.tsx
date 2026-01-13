"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";

// Type for Others Feedback Item
type OthersFeedbackItem = {
    id: string;
    imageUrl?: string;
    questionNumber: string;
    score: string; // User input score
    maxScore: number; // Max score for this question (from ExamInput)
    comment: string; // Feedback comment
};

interface SubjectiveFallbackPageProps {
    examCode?: string;
}

const SubjectiveFallbackPage: React.FC<SubjectiveFallbackPageProps> = ({ examCode = "UNKNOWN" }) => {
    const [items, setItems] = useState<OthersFeedbackItem[]>([]);
    const [focusedIndex, setFocusedIndex] = useState(0);
    const [zoomedImage, setZoomedImage] = useState<string | null>(null);
    const [zoomScale, setZoomScale] = useState(1);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Refs for Score Inputs and Comment Textareas
    const scoreInputRefs = useRef<(HTMLInputElement | null)[]>([]);
    const commentRefs = useRef<(HTMLTextAreaElement | null)[]>([]);

    const router = useRouter();

    // Mock Data Loader / API Loader
    useEffect(() => {
        const fetchOthersQuestions = async () => {
            if (examCode === "MOCK_DATA") {
                setItems([
                    {
                        id: "1",
                        imageUrl: "https://via.placeholder.com/600x400?text=Others+Type+Q10",
                        questionNumber: "10",
                        score: "",
                        maxScore: 10, // Mock max score
                        comment: ""
                    },
                    {
                        id: "2",
                        imageUrl: "https://via.placeholder.com/600x400?text=Others+Type+Q11-1",
                        questionNumber: "11-1",
                        score: "",
                        maxScore: 5, // Mock max score
                        comment: ""
                    }
                ]);
                return;
            }

            try {
                // TODO: Replace with actual API endpoint
                const response = await fetch(`/api/reports/others-questions/${examCode}`);
                if (!response.ok) {
                    setItems([]);
                    return;
                }
                const data = await response.json();
                // Map data...
                // Assuming backend returns needed fields
            } catch (error) {
                console.error("Failed to fetch others questions:", error);
                setItems([]);
            }
        };

        fetchOthersQuestions();
    }, [examCode]);

    // Focus & Scroll Logic
    const changeFocus = (index: number, type: 'score' | 'comment' = 'score') => {
        setFocusedIndex(index);
        const element = document.getElementById(`others-item-${index}`);
        element?.scrollIntoView({ behavior: 'smooth', block: 'center' });

        if (type === 'score') {
            scoreInputRefs.current[index]?.focus();
        } else {
            commentRefs.current[index]?.focus();
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent, index: number, type: 'score' | 'comment') => {
        if (e.key === "Enter") {
            e.preventDefault(); // Prevent newline in textarea or form submit
            if (type === 'score') {
                // Move to comment
                changeFocus(index, 'comment');
            } else {
                // Move to next question score
                if (index < items.length - 1) {
                    changeFocus(index + 1, 'score');
                } else {
                    // Last item, maybe focus submit button or nothing
                }
            }
        } else if (e.key === "ArrowDown") {
            e.preventDefault();
            if (index < items.length - 1) {
                changeFocus(index + 1, type);
            }
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            if (index > 0) {
                changeFocus(index - 1, type);
            }
        }
    };

    const handleUpdateItem = (index: number, field: keyof OthersFeedbackItem, value: string) => {
        const newItems = [...items];

        if (field === 'score') {
            // Validation: Check max score
            const numValue = Number(value);
            const max = newItems[index].maxScore;

            if (value !== '' && (isNaN(numValue) || numValue > max)) {
                alert(`점수는 ${max}점을 초과할 수 없습니다.`);
                return;
            }
        }

        (newItems[index] as any)[field] = value;
        setItems(newItems);
    };

    // --- Zoom Logic (Copied from QuestionFeedbackPage) ---
    const resetZoom = useCallback(() => { setZoomScale(1); setPosition({ x: 0, y: 0 }); }, []);
    const handleZoomIn = (e?: React.MouseEvent) => { e?.stopPropagation(); setZoomScale(prev => Math.min(prev + 0.5, 4)); };
    const handleZoomOut = (e?: React.MouseEvent) => { e?.stopPropagation(); setZoomScale(prev => { const next = Math.max(prev - 0.5, 1); if (next === 1) setPosition({ x: 0, y: 0 }); return next; }); };
    const handleMouseDown = (e: React.MouseEvent) => { if (zoomScale <= 1) return; setIsDragging(true); setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y }); };
    const handleMouseMove = (e: React.MouseEvent) => { if (!isDragging || zoomScale <= 1) return; setPosition({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y }); };
    const handleMouseUp = () => { setIsDragging(false); };
    const handleImageClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        if (zoomScale === 1) {
            if (e.currentTarget) {
                const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
                const offsetX = (e.clientX - rect.left - rect.width / 2) * -2;
                const offsetY = (e.clientY - rect.top - rect.height / 2) * -2;
                setZoomScale(2);
                setPosition({ x: offsetX, y: offsetY });
            }
        }
    };

    const handleSubmit = async () => {
        if (isSubmitting) return;
        setIsSubmitting(true);

        const payload = {
            examCode,
            evaluations: items.map(i => ({
                questionNumber: i.questionNumber,
                score: i.score,
                comment: i.comment
            }))
        };
        console.log("Submitting Subjective Evaluation:", payload);

        try {
            const res = await fetch("/api/subjective-feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                alert("채점 및 피드백이 저장되었습니다.");
                router.push(`/exam/${examCode}/grading/others-loading`);
            } else {
                const text = await res.text();
                alert("저장 실패: " + text);
            }
        } catch (error) {
            console.error(error);
            alert("네트워크 오류가 발생했습니다.");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="relative mx-auto min-h-screen w-[1152px] bg-white pb-32 font-sans">
            {/* Logo */}
            <div
                className="absolute left-[10px] top-[17px] h-[43px] w-[165px] cursor-pointer z-50"
                onClick={() => router.push("/")}
                style={{ backgroundImage: "url('/Gradi_logo.png')", backgroundSize: "cover" }}
            />

            {/* Title */}
            <div className="pt-[100px] pb-6 flex justify-between items-end mb-4 px-6">
                <div>
                    <h1 className="text-[40px] font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        채점 피드백
                    </h1>
                    <p className="text-[18px] font-medium text-[#A0A0A0]">
                        인식 안된 문항에 대해 피드백이 필요합니다.
                    </p>
                </div>
            </div>

            {/* Main Content */}
            <div className="w-full bg-[#F8F0FF] rounded-2xl p-8 min-h-[600px] shadow-sm border-[3px] border-[#AC5BF8] space-y-8">
                {items.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-[400px] text-gray-400">
                        <p className="text-2xl font-semibold">채점할 문항이 없습니다.</p>
                    </div>
                ) : (
                    items.map((item, index) => (
                        <div
                            key={item.id}
                            id={`others-item-${index}`}
                            className={`flex gap-8 p-6 rounded-xl border-2 bg-white shadow-sm transition-all
                                ${focusedIndex === index ? "ring-4 ring-[#AC5BF8]/20 border-[#AC5BF8]" : "border-gray-200"}
                            `}
                            onClick={() => changeFocus(index)}
                        >
                            {/* Image Section */}
                            <div className="flex-1 bg-gray-100 rounded-lg border border-gray-300 overflow-hidden relative group cursor-zoom-in"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (item.imageUrl) { setZoomedImage(item.imageUrl); resetZoom(); }
                                }}
                            >
                                {item.imageUrl ? (
                                    <>
                                        <img src={item.imageUrl} alt="Question" className="w-full h-full object-contain" />
                                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 flex items-center justify-center transition-colors">
                                            <span className="opacity-0 group-hover:opacity-100 bg-black/50 text-white px-3 py-1 rounded-full text-xs font-bold">확대</span>
                                        </div>
                                    </>
                                ) : (
                                    <div className="flex items-center justify-center h-64 text-gray-400">이미지 없음</div>
                                )}
                            </div>

                            {/* Scoring & Comment Section */}
                            <div className="w-[350px] flex flex-col gap-6">
                                <div className="flex justify-between items-center border-b-2 border-dashed border-gray-200 pb-4">
                                    <h3 className="text-2xl font-extrabold text-gray-700">문항 {item.questionNumber}</h3>
                                    <div className="flex items-end gap-2">
                                        <input
                                            ref={el => { scoreInputRefs.current[index] = el; }}
                                            type="number"
                                            className="w-20 text-center text-3xl font-bold border-b-2 border-gray-400 focus:border-[#AC5BF8] focus:outline-none bg-transparent"
                                            value={item.score}
                                            onChange={(e) => handleUpdateItem(index, 'score', e.target.value)}
                                            onFocus={() => changeFocus(index, 'score')}
                                            onKeyDown={(e) => handleKeyDown(e, index, 'score')}
                                            placeholder="0"
                                        />
                                        <span className="text-2xl text-gray-400 font-bold">/ {item.maxScore}</span>
                                    </div>
                                </div>

                                <div className="flex-1 flex flex-col gap-2">
                                    <label className="text-sm font-bold text-gray-600">📝 교수자 코멘트 (피드백)</label>
                                    <textarea
                                        ref={el => { commentRefs.current[index] = el; }}
                                        className="w-full flex-1 p-4 rounded-lg border-2 border-[#E0E0E0] focus:border-[#AC5BF8] focus:ring-2 focus:ring-[#AC5BF8]/10 resize-none text-base placeholder-gray-300 transition-all font-medium"
                                        placeholder="학생에게 전달할 피드백을 입력하세요..."
                                        value={item.comment}
                                        onChange={(e) => handleUpdateItem(index, 'comment', e.target.value)}
                                        onFocus={() => changeFocus(index, 'comment')}
                                        onKeyDown={(e) => handleKeyDown(e, index, 'comment')}
                                    />
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>

            {/* Bottom Button */}
            <div className="flex justify-center pt-8">
                <button
                    onClick={handleSubmit}
                    disabled={isSubmitting}
                    className="w-[300px] px-4 py-4 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] rounded-lg text-white text-xl font-bold shadow-xl hover:scale-105 transition-all"
                >
                    {isSubmitting ? "저장 중..." : "계속하기"}
                </button>
            </div>

            {/* Lightbox / Advanced Zoom Modal (Same as RecognitionFallbackPage) */}
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

export default SubjectiveFallbackPage;

import React, { useEffect, useState } from "react";
import { Question } from "../../types";

interface FloatingSidebarProps {
    questions: Question[];
    totalScore: number;
    totalSubCount: number;
    addQuestion: () => void;
    removeQuestion: (id: string) => void;
    addSubQuestion: (qId: string) => void;
    removeSubQuestion: (qId: string, sqId: string) => void;
}

const PlusIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" fill="#AC5BF8" />
        <path d="M12 7V17M7 12H17" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

const MinusIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" fill="#FF5B5B" />
        <path d="M7 12H17" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

export const FloatingSidebar: React.FC<FloatingSidebarProps> = ({
    questions,
    totalScore,
    totalSubCount,
    addQuestion,
    removeQuestion,
    addSubQuestion,
    removeSubQuestion
}) => {
    const [scrollY, setScrollY] = useState(0);

    // Soft follow animation logic
    useEffect(() => {
        let targetY = 0;
        let currentY = 0;
        let animationFrameId: number;

        const handleScroll = () => {
            targetY = window.scrollY;
        };

        const updatePosition = () => {
            const diff = targetY - currentY;
            if (Math.abs(diff) > 0.5) {
                currentY += diff * 0.1;
                setScrollY(currentY);
                animationFrameId = requestAnimationFrame(updatePosition);
            } else {
                animationFrameId = requestAnimationFrame(updatePosition);
            }
        };

        window.addEventListener("scroll", handleScroll);
        animationFrameId = requestAnimationFrame(updatePosition);

        return () => {
            window.removeEventListener("scroll", handleScroll);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    const lastQ = questions[questions.length - 1];

    return (
        <div
            className="absolute left-[calc(50%+580px)] w-[220px] bg-white p-4 rounded-xl shadow-lg border border-gray-200 space-y-3"
            style={{ top: `${96 + scrollY}px` }}
        >
            {/* Statistics */}
            <div className="space-y-2 bg-gray-50 p-3 rounded-lg">
                <h3 className="text-sm font-bold text-gray-800 border-b pb-1 mb-1">시험 통계</h3>
                <div className="flex justify-between text-xs">
                    <span className="text-gray-600">총 문제</span>
                    <span className="font-bold text-purple-600">{questions.length}개</span>
                </div>
                <div className="flex justify-between text-xs">
                    <span className="text-gray-600">세부 문항</span>
                    <span className="font-bold text-purple-600">{totalSubCount}개</span>
                </div>
                <div className="flex justify-between text-xs border-t pt-1 mt-1">
                    <span className="font-bold text-gray-700">총점</span>
                    <span className="font-bold text-base text-purple-700">{totalScore}점</span>
                </div>
            </div>

            {/* Add Question Button */}
            <button
                onClick={addQuestion}
                className="w-full bg-purple-100 hover:bg-purple-200 text-purple-700 font-bold py-2.5 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors text-sm cursor-pointer"
            >
                <PlusIcon />
                <span>문제 추가</span>
            </button>

            {/* Active Controls for Last Question */}
            <div className="space-y-2">
                <h3 className="text-xs font-bold text-gray-500">
                    마지막 문제(<span className="font-bold text-purple-600">{questions.length}</span>번) 관리
                </h3>
                <div className="grid grid-cols-2 gap-2">
                    <button
                        onClick={() => {
                            if (lastQ) removeQuestion(lastQ.id);
                        }}
                        className="col-span-2 flex items-center justify-center p-2 border border-red-100 rounded hover:bg-red-50 transition-colors text-red-600 font-bold text-xs gap-1 cursor-pointer"
                        title="마지막 문제 제거"
                    >
                        <MinusIcon />
                        <span>마지막 문제 제거</span>
                    </button>

                    <button
                        onClick={() => addSubQuestion(lastQ.id)}
                        className="flex flex-col items-center justify-center p-2 border border-purple-100 rounded hover:bg-purple-50 transition-colors cursor-pointer"
                        title="세부 문항 추가"
                    >
                        <PlusIcon />
                        <span className="text-[10px] text-gray-600 mt-1">세부 문항 추가</span>
                    </button>
                    {lastQ?.subQuestions.length > 0 && (
                        <button
                            onClick={() => {
                                if (lastQ.subQuestions.length > 0) {
                                    const lastSubId = lastQ.subQuestions[lastQ.subQuestions.length - 1].id;
                                    removeSubQuestion(lastQ.id, lastSubId);
                                }
                            }}
                            className="flex flex-col items-center justify-center p-2 border border-red-100 rounded hover:bg-red-50 transition-colors cursor-pointer"
                            title="마지막 세부 문항 제거"
                        >
                            <MinusIcon />
                            <span className="text-[10px] text-gray-600 mt-1">세부 문항 제거</span>
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

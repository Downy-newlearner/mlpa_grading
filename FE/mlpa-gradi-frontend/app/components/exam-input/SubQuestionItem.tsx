import React from "react";
import { SubQuestion, QuestionType } from "../../types";

interface SubQuestionItemProps {
    qId: string;
    sub: SubQuestion;
    index: number;
    parentIndex: number; // qIndex (0-based)
    draggingId: string | null;
    setDraggingId: (id: string | null) => void;
    handleDragStart: (e: React.DragEvent, qId: string, index: number, subId: string) => void;
    handleDragOver: (e: React.DragEvent) => void;
    handleDrop: (e: React.DragEvent, targetQId: string, targetIndex: number) => void;
    insertSubQuestion: (qId: string, index: number) => void;
    removeSubQuestion: (qId: string, sqId: string) => void;
    updateSubQuestion: (qId: string, sqId: string, patch: Partial<Omit<SubQuestion, "id">>) => void;
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

export const SubQuestionItem: React.FC<SubQuestionItemProps> = ({
    qId,
    sub,
    index,
    parentIndex,
    draggingId,
    setDraggingId,
    handleDragStart,
    handleDragOver,
    handleDrop,
    insertSubQuestion,
    removeSubQuestion,
    updateSubQuestion
}) => {
    const subLabel = `${parentIndex + 1}-${index + 1}`;
    const isSubDragging = draggingId === sub.id;

    // Inherited score logic for Sub needs access to previous sub. 
    // This logic was "inside" the loop before. Passed down logic via parent or handle inside?
    // Let's rely on placeholder logic or handle it.
    // For simplicity, we won't strictly compute "inherited" score here for placeholder hint unless passed.
    // We can just leave placeholder generic or empty.

    const subDragStyle = isSubDragging
        ? "z-50 transform scale-[1.02] shadow-xl border-purple-500 border-[3px] ring-1 ring-purple-300 bg-purple-50 cursor-pointer"
        : "bg-white border-[#AC5BF8] border-[3px] hover:border-purple-300 cursor-pointer";

    return (
        <div
            id={sub.id}
            className={`rounded border p-3 transition-all duration-200 ${subDragStyle}`}
            draggable
            onDragStart={(e) => handleDragStart(e, qId, index, sub.id)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, qId, index)}
            onDragEnd={() => setDraggingId(null)}
        >
            <div className="flex items-center justify-between gap-2">
                <div className="inline-block text-xl font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">{subLabel}번 정답</div>
                <div className="flex items-center gap-1">
                    <button
                        type="button"
                        onClick={() => insertSubQuestion(qId, index)}
                        title="이 문항 뒤에 추가"
                        className="cursor-pointer"
                    >
                        <PlusIcon />
                    </button>
                    <button
                        type="button"
                        onClick={() => removeSubQuestion(qId, sub.id)}
                        title="삭제"
                        className="cursor-pointer"
                    >
                        <MinusIcon />
                    </button>
                </div>
            </div>

            <div className="mt-2">
                <textarea
                    className="w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300"
                    value={sub.text}
                    placeholder="세부 문항 정답을 입력하세요"
                    onChange={(e) =>
                        updateSubQuestion(qId, sub.id, { text: e.target.value })
                    }
                />
            </div>

            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <label className="block text-sm font-semibold mb-1">배점<span className="text-red-500">*</span></label>
                    <input
                        type="number"
                        min={0}
                        className="w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300 bg-white"
                        value={sub.score}
                        onChange={(e) =>
                            updateSubQuestion(qId, sub.id, {
                                score: e.target.value,
                            })
                        }
                    />
                </div>

                <div>
                    <label className="block text-sm font-semibold mb-1">문제 유형<span className="text-red-500">*</span></label>
                    <select
                        className="w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300 cursor-pointer"
                        value={sub.type}
                        onChange={(e) =>
                            updateSubQuestion(qId, sub.id, {
                                type: e.target.value as QuestionType,
                            })
                        }
                    >
                        <option value="multiple">객관식</option>
                        <option value="short">단답형</option>
                        <option value="ox">OX</option>
                    </select>
                </div>
            </div>
        </div>
    );
};

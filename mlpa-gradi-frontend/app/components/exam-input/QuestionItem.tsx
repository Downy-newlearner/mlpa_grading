import React from "react";
import { Question, SubQuestion, QuestionType } from "../../types";
import { SubQuestionItem } from "./SubQuestionItem";

interface QuestionItemProps {
    question: Question;
    index: number;
    draggingId: string | null;
    setDraggingId: (id: string | null) => void;
    // Main Drag Handlers
    handleMainDragStart: (e: React.DragEvent, index: number) => void;
    handleDragOver: (e: React.DragEvent) => void;
    handleMainDrop: (e: React.DragEvent, targetIndex: number) => void;
    // Actions
    removeQuestion: (id: string) => void;
    updateQuestion: (id: string, patch: Partial<Omit<Question, "id" | "subQuestions">>) => void;
    addSubQuestion: (id: string) => void;
    // Sub Drag & Actions
    handleSubDragStart: (e: React.DragEvent, qId: string, index: number, subId: string) => void;
    handleSubDrop: (e: React.DragEvent, targetQId: string, targetIndex: number) => void;
    insertSubQuestion: (qId: string, index: number) => void;
    removeSubQuestion: (qId: string, sqId: string) => void;
    updateSubQuestion: (qId: string, sqId: string, patch: Partial<Omit<SubQuestion, "id">>) => void;
    questionsLength: number;
    numberingPreview: any;
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

export const QuestionItem: React.FC<QuestionItemProps> = (props) => {
    const {
        question: q,
        index: qIndex,
        draggingId,
        setDraggingId,
        handleMainDragStart,
        handleDragOver,
        handleMainDrop,
        removeQuestion,
        updateQuestion,
        addSubQuestion,
        questionsLength,
        numberingPreview
    } = props;

    const qNo = qIndex + 1;
    const isDragging = draggingId === q.id;

    const dragStyleClass = isDragging
        ? "z-50 transform scale-[1.02] shadow-2xl border-purple-600 border-[3px] ring-2 ring-purple-400 bg-white cursor-pointer"
        : "bg-[#F8F0FF] border-[#AC5BF8] border-[3px] hover:border-purple-300 cursor-pointer";

    return (
        <div
            id={q.id}
            className={`rounded shadow-md space-y-4 transition-all duration-200 ${dragStyleClass} p-4`}
            draggable
            onDragStart={(e) => handleMainDragStart(e, qIndex)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleMainDrop(e, qIndex)}
            onDragEnd={() => setDraggingId(null)}
        >
            <div className="flex items-center justify-between cursor-move group">
                <div className="text-3xl font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">문제 {qNo}</div>
                {questionsLength > 1 && (
                    <button
                        type="button"
                        onClick={() => removeQuestion(q.id)}
                        title="문제 제거"
                        className="cursor-pointer"
                    >
                        <MinusIcon />
                    </button>
                )}
            </div>

            {/* Main Question Inputs */}
            <div className="grid grid-cols-1 gap-4">
                <div>
                    <label className="inline-block text-xl font-extrabold mb-2 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">{qNo}번 정답</label>
                    <textarea
                        className={`w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300 bg-white ${q.subQuestions.length > 0 ? "bg-gray-100 cursor-not-allowed" : ""}`}
                        value={q.text}
                        placeholder={
                            q.subQuestions.length > 0
                                ? "세부 문항이 존재하여 입력할 수 없습니다."
                                : "정답을 입력하세요"
                        }
                        disabled={q.subQuestions.length > 0}
                        onChange={(e) => updateQuestion(q.id, { text: e.target.value })}
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-semibold mb-1">배점<span className="text-red-500">*</span></label>
                        <input
                            type="number"
                            min={0}
                            className={`w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300 bg-white ${q.subQuestions.length > 0 ? "bg-gray-100" : ""}`}
                            value={q.score}
                            readOnly={q.subQuestions.length > 0}
                            onChange={(e) => updateQuestion(q.id, { score: e.target.value })}
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-semibold mb-1">문제 유형<span className="text-red-500">*</span></label>
                        <select
                            className="w-full border border-black p-2 rounded focus:outline-none focus:ring focus:ring-purple-300 bg-white cursor-pointer"
                            value={q.type}
                            onChange={(e) => updateQuestion(q.id, { type: e.target.value as QuestionType })}
                        >
                            <option value="multiple">객관식</option>
                            <option value="short">단답형</option>
                            <option value="ox">OX</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Sub Questions Area */}
            <div className="rounded p-3 bg-white border-[3px] border-[#AC5BF8]">
                <div className="flex items-center justify-between">
                    <div className="text-sm font-semibold text-gray-700">세부 문항</div>
                    <button
                        type="button"
                        onClick={() => addSubQuestion(q.id)}
                        className="cursor-pointer"
                    >
                        <PlusIcon />
                    </button>
                </div>

                {q.subQuestions.length === 0 ? (
                    <div className="mt-3 text-sm text-gray-500">
                        아직 세부 문항이 없습니다. (추가 시 {qNo}-1부터 시작)
                    </div>
                ) : (
                    <div className="mt-3 space-y-3">
                        {q.subQuestions.map((sq, subIndex) => (
                            <SubQuestionItem
                                key={sq.id}
                                qId={q.id}
                                sub={sq}
                                index={subIndex}
                                parentIndex={qIndex}
                                draggingId={draggingId}
                                setDraggingId={setDraggingId}
                                handleDragStart={props.handleSubDragStart}
                                handleDragOver={handleDragOver}
                                handleDrop={props.handleSubDrop}
                                insertSubQuestion={props.insertSubQuestion}
                                removeSubQuestion={props.removeSubQuestion}
                                updateSubQuestion={props.updateSubQuestion}
                            />
                        ))}
                    </div>
                )}
            </div>

            {/* Debug/Preview */}
            <div className="text-xs text-gray-400">
                문제번호: {qNo}
                {numberingPreview[qIndex]?.subNos.length
                    ? ` / 세부 문항: ${numberingPreview[qIndex]?.subNos.join(", ")}`
                    : ""}
            </div>
        </div>
    );
};

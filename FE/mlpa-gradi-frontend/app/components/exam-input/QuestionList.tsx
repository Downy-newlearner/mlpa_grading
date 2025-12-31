import React, { useState } from "react";
import { Question, SubQuestion } from "../../types";
import { QuestionItem } from "./QuestionItem";
import Button from "../../components/Button";

interface QuestionListProps {
    questions: Question[];
    setQuestions: React.Dispatch<React.SetStateAction<Question[]>>;
    numberingPreview: any;
    addQuestion: () => void;
    removeQuestion: (id: string) => void;
    updateQuestion: (id: string, patch: Partial<Omit<Question, "id" | "subQuestions">>) => void;
    addSubQuestion: (id: string) => void;
    insertSubQuestion: (qId: string, index: number) => void;
    removeSubQuestion: (qId: string, sqId: string) => void;
    updateSubQuestion: (qId: string, sqId: string, patch: Partial<Omit<SubQuestion, "id">>) => void;
}

export const QuestionList: React.FC<QuestionListProps> = ({
    questions,
    setQuestions,
    numberingPreview,
    addQuestion,
    removeQuestion,
    updateQuestion,
    addSubQuestion,
    insertSubQuestion,
    removeSubQuestion,
    updateSubQuestion
}) => {
    const [draggingId, setDraggingId] = useState<string | null>(null);

    // Main Drag Handlers
    const handleMainDragStart = (e: React.DragEvent, index: number) => {
        setDraggingId(questions[index].id);
        e.dataTransfer.setData("mainIndex", index.toString());
        e.dataTransfer.effectAllowed = "move";
    };

    const handleMainDrop = (e: React.DragEvent, targetIndex: number) => {
        e.preventDefault();
        setDraggingId(null);
        const sourceIndexStr = e.dataTransfer.getData("mainIndex");
        if (!sourceIndexStr) return;

        const sourceIndex = Number(sourceIndexStr);
        if (sourceIndex === targetIndex) return;

        setQuestions((prev) => {
            const newQuestions = [...prev];
            const [movedQ] = newQuestions.splice(sourceIndex, 1);
            newQuestions.splice(targetIndex, 0, movedQ);
            return newQuestions;
        });
    };

    // Sub Drag Handlers
    const handleSubDragStart = (e: React.DragEvent, qId: string, index: number, subId: string) => {
        e.stopPropagation();
        setDraggingId(subId);
        e.dataTransfer.setData("qId", qId);
        e.dataTransfer.setData("index", index.toString());
    };

    const handleSubDrop = (e: React.DragEvent, targetQId: string, targetIndex: number) => {
        e.preventDefault();
        setDraggingId(null);
        const sourceQId = e.dataTransfer.getData("qId");
        const sourceIndex = Number(e.dataTransfer.getData("index"));

        if (sourceQId !== targetQId || sourceIndex === targetIndex) return;

        setQuestions((prev) =>
            prev.map((q) => {
                if (q.id === targetQId) {
                    const newSubQuestions = [...q.subQuestions];
                    const [movedSub] = newSubQuestions.splice(sourceIndex, 1);
                    newSubQuestions.splice(targetIndex, 0, movedSub);
                    return { ...q, subQuestions: newSubQuestions };
                }
                return q;
            })
        );
    };

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    return (
        <div className="mb-6 pt-12">
            <div className="flex justify-between items-end mb-8">
                <h2 className="text-[44px] font-extrabold leading-[52px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">문제와 정답 입력<span className="text-red-500">*</span></h2>
                <p className="text-[#A0A0A0] text-[22px] font-medium leading-[31px]">채점에 사용하기 위한 정답지를 입력합니다.</p>
            </div>

            <div className="space-y-5">
                {questions.map((q, qIndex) => (
                    <QuestionItem
                        key={q.id}
                        question={q}
                        index={qIndex}
                        draggingId={draggingId}
                        setDraggingId={setDraggingId}
                        handleMainDragStart={handleMainDragStart}
                        handleDragOver={handleDragOver}
                        handleMainDrop={handleMainDrop}
                        removeQuestion={removeQuestion}
                        updateQuestion={updateQuestion}
                        addSubQuestion={addSubQuestion}
                        handleSubDragStart={handleSubDragStart}
                        handleSubDrop={handleSubDrop}
                        insertSubQuestion={insertSubQuestion}
                        removeSubQuestion={removeSubQuestion}
                        updateSubQuestion={updateSubQuestion}
                        questionsLength={questions.length}
                        numberingPreview={numberingPreview}
                    />
                ))}
            </div>

            <div className="mt-4 flex gap-3">
                <Button
                    type="button"
                    label="+ 문제 추가"
                    className="px-4 py-2 text-base shadow cursor-pointer"
                    onClick={addQuestion}
                />
            </div>
        </div>
    );
};

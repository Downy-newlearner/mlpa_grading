"use client";

import React from "react";
import Link from "next/link";
import Button from "../components/Button";
import { useExamForm } from "../hooks/useExamForm";
import { ExamInfoForm } from "../components/exam-input/ExamInfoForm";
import { QuestionList } from "../components/exam-input/QuestionList";
import { FileUploadSection } from "../components/exam-input/FileUploadSection";
import { FloatingSidebar } from "../components/exam-input/FloatingSidebar";

const ExamInput: React.FC = () => {
    const {
        questions,
        setQuestions,
        examTitle,
        setExamTitle,
        examDate,
        setExamDate,
        attendanceFile,
        setAttendanceFile,
        answerSheetFiles,
        setAnswerSheetFiles,
        totalScore,
        totalSubCount,
        numberingPreview,
        addQuestion,
        removeQuestion,
        updateQuestion,
        addSubQuestion,
        removeSubQuestion,
        updateSubQuestion,
        insertSubQuestion,
        handleStartGrading
    } = useExamForm();

    return (
        <div className="relative mx-auto w-[1152px] bg-white font-semibold">
            {/* 로고 영역 */}
            <Link href="/" className="absolute w-[165px] h-[43px] top-[17px] left-[10px] animate-fade-in-up block cursor-pointer z-50">
                <div
                    className="w-full h-full"
                    style={{
                        backgroundImage: "url(/Gradi_logo.png)",
                        backgroundSize: "cover",
                        backgroundPosition: "center",
                        backgroundRepeat: "no-repeat",
                    }}
                />
            </Link>



            <div className="pt-[120px] px-6 pb-24 space-y-24 animate-fade-in-up">

                {/* 1. Exam Info */}
                <ExamInfoForm
                    examTitle={examTitle}
                    setExamTitle={setExamTitle}
                    examDate={examDate}
                    setExamDate={setExamDate}
                />

                {/* 2. Question List */}
                <QuestionList
                    questions={questions}
                    setQuestions={setQuestions}
                    numberingPreview={numberingPreview}
                    addQuestion={addQuestion}
                    removeQuestion={removeQuestion}
                    updateQuestion={updateQuestion}
                    addSubQuestion={addSubQuestion}
                    insertSubQuestion={insertSubQuestion}
                    removeSubQuestion={removeSubQuestion}
                    updateSubQuestion={updateSubQuestion}
                />

                {/* 3. File Uploads */}
                <FileUploadSection
                    attendanceFile={attendanceFile}
                    setAttendanceFile={setAttendanceFile}
                    answerSheetFiles={answerSheetFiles}
                    setAnswerSheetFiles={setAnswerSheetFiles}
                />

                {/* 4. Action Button */}
                <div className="mt-4 flex justify-center">
                    <Button
                        label="채점 시작하기"
                        className="whitespace-nowrap w-[240px] px-4 py-4 text-xl shadow cursor-pointer"
                        onClick={handleStartGrading}
                    />
                </div>

                {/* 5. Floating Sidebar */}
                <FloatingSidebar
                    questions={questions}
                    totalScore={totalScore}
                    totalSubCount={totalSubCount}
                    addQuestion={addQuestion}
                    removeQuestion={removeQuestion}
                    addSubQuestion={addSubQuestion}
                    removeSubQuestion={removeSubQuestion}
                />
            </div>
        </div>
    );
};

export default ExamInput;

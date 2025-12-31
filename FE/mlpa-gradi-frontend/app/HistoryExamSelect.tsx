"use client";

import React, { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ExamHistoryItem } from "./types";
import { useExamHistory } from "./hooks/useExamHistory";

interface HistoryExamSelectProps {
    exams?: ExamHistoryItem[];
}

const HistoryExamSelect: React.FC<HistoryExamSelectProps> = ({
    exams: initialExams,
}) => {
    const router = useRouter();
    const [query, setQuery] = useState("");

    // Usage of custom hook
    const { filterExams, groupExams } = useExamHistory(initialExams);

    // Grouping Logic (Derived from hook helpers)
    const groupedExams = useMemo(() => {
        const filtered = filterExams(query);
        return groupExams(filtered);
    }, [query, filterExams, groupExams]);

    return (
        <div className="relative mx-auto w-[1152px] min-h-[900px] h-auto bg-white font-semibold">
            {/* Top Logo - Redirect to Home */}
            <Link href="/" className="absolute left-[10px] top-[17px] w-[165px] h-[43px] animate-fade-in-up block cursor-pointer z-50">
                <div
                    className="w-full h-full"
                    style={{
                        backgroundImage: "url('/Gradi_logo.png')",
                        backgroundSize: "cover",
                        backgroundPosition: "center",
                        backgroundRepeat: "no-repeat",
                    }}
                />
            </Link>

            {/* Center Logo (Optional, consistent with ExamInput) */}
            <div
                className="absolute w-[315px] h-[315px] left-[418px] top-[129.04px]"
                style={{
                    backgroundImage: "url(/MLPA_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            <div className="pt-[480px] px-6 pb-24 space-y-24 animate-fade-in-up">
                {/* 시험 선택 프레임 */}
                <div>
                    <div className="flex justify-between items-end mb-2">
                        <h2 className="text-[40px] font-semibold leading-[48px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">시험 선택</h2>
                        <p className="text-[#A0A0A0] text-[20px] font-medium leading-[29px]">조회하고 싶은 시험을 선택합니다.</p>
                    </div>

                    <div className="border-[#AC5BF8] border-[3px] rounded-lg p-6 bg-[#F8F0FF] shadow-md space-y-6 font-semibold min-h-[500px]">
                        {/* Search Bar */}
                        <div className="relative h-[60px] w-full">
                            <input
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                className="h-full w-full rounded border border-black px-4 text-xl bg-white focus:outline-none focus:ring focus:ring-purple-300 placeholder:text-gray-400"
                                placeholder="시험명 / 날짜 / 코드로 검색"
                            />
                            {/* Search Icon */}
                            <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                            </div>
                        </div>

                        {/* Divider */}
                        <div className="h-px bg-purple-200" />

                        {/* Exam List Area */}
                        <div className="space-y-6 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
                            {groupedExams.length === 0 ? (
                                <div className="text-center py-10 text-xl text-gray-500">
                                    검색 결과가 없습니다.
                                </div>
                            ) : (
                                groupedExams.map((group) => (
                                    <div key={group.yearMonth} className="space-y-4">
                                        <h3 className="text-2xl font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent ml-1 w-fit">{group.yearMonth}</h3>
                                        <div className="space-y-3">
                                            {group.items.map((exam) => (
                                                <button
                                                    key={exam.examId}
                                                    type="button"
                                                    onClick={() => router.push(`/history/${exam.examId}`)}
                                                    className="relative flex w-full flex-col items-start rounded border border-black p-4 text-left transition-all bg-white hover:bg-purple-50 hover:border-[#AC5BF8]"
                                                >
                                                    <span className="text-xl font-bold text-gray-900 mb-1">{exam.examName}</span>
                                                    <div className="flex gap-4 text-gray-600 font-medium text-sm">
                                                        <span>시험일시: {exam.examDate}</span>
                                                        <span>코드: {exam.code}</span>
                                                    </div>
                                                </button>
                                            ))}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default HistoryExamSelect;

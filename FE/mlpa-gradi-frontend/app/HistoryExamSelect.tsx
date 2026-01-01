"use client";

import React, { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ExamHistoryItem } from "./types";
import { useExamHistory } from "./hooks/useExamHistory";
import { examService } from "./services/examService";
import Button from "./components/Button";

interface HistoryExamSelectProps {
    exams?: ExamHistoryItem[];
}

const HistoryExamSelect: React.FC<HistoryExamSelectProps> = ({
    exams: initialExams,
}) => {
    const router = useRouter();
    const [query, setQuery] = useState("");
    const [isManageMode, setIsManageMode] = useState(false);
    const [selectedCodes, setSelectedCodes] = useState<Set<string>>(new Set());
    const [examsToDelete, setExamsToDelete] = useState<ExamHistoryItem[]>([]);

    // Usage of custom hook
    const { filterExams, groupExams } = useExamHistory(initialExams);

    // Filtered items for Select All
    const allFilteredExams = useMemo(() => filterExams(query), [query, filterExams]);

    // Grouping Logic (Derived from hook helpers)
    const groupedExams = useMemo(() => {
        return groupExams(allFilteredExams);
    }, [allFilteredExams, groupExams]);

    // Navigate to history page with examCode in query params
    const handleExamClick = (exam: ExamHistoryItem) => {
        if (isManageMode) return; // Prevent navigation in manage mode
        if (exam.examCode) {
            router.push(`/history/${exam.examId}?code=${exam.examCode}`);
        } else {
            router.push(`/history/${exam.examId}`);
        }
    };

    const toggleSelect = (examCode: string) => {
        setSelectedCodes(prev => {
            const next = new Set(prev);
            if (next.has(examCode)) next.delete(examCode);
            else next.add(examCode);
            return next;
        });
    };

    const toggleSelectAll = () => {
        if (selectedCodes.size === allFilteredExams.length && allFilteredExams.length > 0) {
            setSelectedCodes(new Set());
        } else {
            const newSelected = new Set(allFilteredExams.map(e => e.examCode).filter((code): code is string => code !== undefined));
            setSelectedCodes(newSelected);
        }
    };

    const handleDeleteSelected = () => {
        const toDelete = allFilteredExams.filter(exam => exam.examCode && selectedCodes.has(exam.examCode));
        if (toDelete.length > 0) {
            setExamsToDelete(toDelete);
        }
    };

    const confirmDelete = async () => {
        if (examsToDelete.length === 0) return;
        try {
            await Promise.all(examsToDelete.map(exam => exam.examCode ? examService.deleteByCode(exam.examCode) : Promise.resolve()));
            window.location.reload();
        } catch (error) {
            console.error("Failed to delete exams:", error);
            alert("시험 삭제 중 오류가 발생했습니다.");
        } finally {
            setExamsToDelete([]);
        }
    };

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


            <div className="pt-[120px] px-6 pb-24 space-y-24 animate-fade-in-up">
                {/* 시험 선택 프레임 */}
                <div>
                    <div className="flex justify-between items-end mb-2">
                        <h2 className="text-[40px] font-semibold leading-[48px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">시험 선택</h2>
                        <div className="flex items-center gap-6">
                            <p className="text-[#A0A0A0] text-[20px] font-medium leading-[29px]">조회하고 싶은 시험을 선택합니다.</p>

                            {/* Management Mode Toggle (Gear Icon) */}
                            <button
                                onClick={() => {
                                    setIsManageMode(!isManageMode);
                                    setSelectedCodes(new Set());
                                }}
                                className={`p-2 rounded-full transition-all hover:bg-purple-100 cursor-pointer ${isManageMode ? 'bg-purple-100 ring-2 ring-purple-300' : ''}`}
                                title="시험 관리"
                            >
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <defs>
                                        <linearGradient id="gearGard" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stopColor="#AC5BF8" />
                                            <stop offset="100%" stopColor="#636ACF" />
                                        </linearGradient>
                                    </defs>
                                    <circle cx="12" cy="12" r="3" stroke="url(#gearGard)" />
                                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="url(#gearGard)" />
                                </svg>
                            </button>
                        </div>
                    </div>

                    {/* Auto-expanding box (no max-height or overflow scroll) */}
                    <div className="border-[#AC5BF8] border-[3px] rounded-lg p-6 bg-[#F8F0FF] shadow-md space-y-6 font-semibold">
                        <div className="flex gap-4">
                            {/* Search Bar */}
                            <div className="relative h-[60px] flex-grow">
                                <input
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    className="h-full w-full rounded border border-black px-4 text-xl bg-white focus:outline-none focus:ring focus:ring-purple-300 placeholder:text-gray-400 font-medium"
                                    placeholder="시험명 / 날짜 / 코드로 검색"
                                />
                                <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                    </svg>
                                </div>
                            </div>

                            {isManageMode && (
                                <div className="flex gap-4 h-[60px]">
                                    <button
                                        onClick={toggleSelectAll}
                                        className="px-6 h-full rounded border-2 border-[#AC5BF8] bg-white text-[#AC5BF8] font-bold text-lg hover:bg-purple-50 transition-all cursor-pointer whitespace-nowrap"
                                    >
                                        {selectedCodes.size === allFilteredExams.length && allFilteredExams.length > 0 ? "전체 해제" : "전체 선택"}
                                    </button>
                                    <button
                                        disabled={selectedCodes.size === 0}
                                        onClick={handleDeleteSelected}
                                        className={`px-6 h-full rounded bg-red-500 text-white font-bold text-lg hover:bg-red-600 transition-all shadow-md cursor-pointer whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed`}
                                    >
                                        삭제하기 ({selectedCodes.size})
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Divider */}
                        <div className="h-px bg-purple-200" />

                        {/* Exam List Area - auto-expanding, no scroll */}
                        <div className="space-y-6">
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
                                                <div key={exam.examId} className="relative flex items-center group">
                                                    {isManageMode && (
                                                        <div
                                                            onClick={() => exam.examCode && toggleSelect(exam.examCode)}
                                                            className={`mr-4 w-8 h-8 rounded border-2 flex items-center justify-center cursor-pointer transition-all ${exam.examCode && selectedCodes.has(exam.examCode) ? 'bg-[#AC5BF8] border-[#AC5BF8]' : 'bg-white border-gray-400 hover:border-[#AC5BF8]'}`}
                                                        >
                                                            {exam.examCode && selectedCodes.has(exam.examCode) && (
                                                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round">
                                                                    <polyline points="20 6 9 17 4 12" />
                                                                </svg>
                                                            )}
                                                        </div>
                                                    )}
                                                    <button
                                                        type="button"
                                                        onClick={() => handleExamClick(exam)}
                                                        disabled={isManageMode}
                                                        className={`relative flex flex-grow flex-col items-start rounded border border-black p-4 text-left transition-all bg-white font-semibold ${isManageMode ? 'opacity-70 cursor-default' : 'hover:bg-purple-50 hover:border-[#AC5BF8] cursor-pointer'}`}
                                                    >
                                                        <span className="text-xl font-bold text-gray-900 mb-1">{exam.examName}</span>
                                                        <div className="flex gap-4 text-gray-600 font-medium text-sm">
                                                            <span>시험일시: {exam.examDate}</span>
                                                            <span className="bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent font-bold">
                                                                코드: {exam.examCode || "-"}
                                                            </span>
                                                        </div>
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Deletion Confirmation Modal */}
            {examsToDelete.length > 0 && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
                    <div className="bg-white rounded-lg p-8 w-[500px] shadow-2xl border-t-8 border-red-500 flex flex-col items-center text-center animate-scale-up">
                        <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mb-4">
                            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#EF4444" strokeWidth="2.5">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                                <line x1="12" y1="9" x2="12" y2="13" />
                                <line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-bold text-gray-900 mb-2">시험 삭제 경고</h3>
                        <div className="text-gray-600 mb-6 font-medium max-h-[150px] overflow-y-auto w-full text-left px-4">
                            <p className="mb-2 text-center">선택한 <span className="text-red-600 font-bold">{examsToDelete.length}개</span>의 시험을 삭제하시겠습니까?</p>
                            <ul className="list-disc list-inside text-sm">
                                {examsToDelete.map(exam => (
                                    <li key={exam.examId} className="truncate">{exam.examName}</li>
                                ))}
                            </ul>
                            <p className="mt-4 text-red-500 font-bold text-center italic">삭제된 데이터는 복구할 수 없습니다.</p>
                        </div>
                        <div className="flex gap-4 w-full">
                            <button
                                onClick={() => setExamsToDelete([])}
                                className="flex-grow h-12 rounded-lg border-2 border-[#AC5BF8] text-[#AC5BF8] font-bold text-lg hover:bg-purple-50 transition-all cursor-pointer"
                            >
                                취소
                            </button>
                            <button
                                onClick={confirmDelete}
                                className="flex-grow h-12 rounded-lg bg-red-500 text-white font-bold text-lg hover:bg-red-600 transition-all shadow-md cursor-pointer"
                            >
                                삭제하기
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default HistoryExamSelect;


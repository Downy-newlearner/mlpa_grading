import React, { useMemo, useState, useEffect } from "react";
import Button from "./components/Button";
import Link from "next/link";
import { examService } from "./services/examService";

type DownloadKind = "course-pdf" | "student-zip" | "attendance-xlsx";

type StatisticsItem = {
    id: string;
    label: string;
    kind: DownloadKind;
};

interface StatisticsDownloadProps {
    items?: StatisticsItem[];
    onDownload?: (item: StatisticsItem) => void;
    examTitle?: string;
    examCode?: string;
}

const StatisticsDownload: React.FC<StatisticsDownloadProps> = ({
    items,
    onDownload,
    examTitle = "시험 통계",
    examCode = "ABC123",
}) => {
    const [query, setQuery] = useState("");
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [displayItems, setDisplayItems] = useState<StatisticsItem[]>([]);

    // Student Access State
    const [isStudentAccess, setIsStudentAccess] = useState(false);

    useEffect(() => {
        if (examCode) {
            const saved = localStorage.getItem(`student_access_${examCode}`);
            setIsStudentAccess(saved === "true");
        }
    }, [examCode]);

    const toggleStudentAccess = () => {
        const newState = !isStudentAccess;
        setIsStudentAccess(newState);
        localStorage.setItem(`student_access_${examCode}`, String(newState));
    };

    const studentPageUrl = typeof window !== 'undefined' ? `${window.location.origin}/student/verify/${items ? '' : examCode}` : ''; // Using examCode for mock URL

    useEffect(() => {
        if (items && items.length > 0) {
            setDisplayItems(items);
            return;
        }

        const loadContent = async () => {
            if (!examCode) return;
            try {
                const answers = await examService.getAnswersByExamCode(examCode);
                const uniqueIds = Array.from(new Set(answers.map(a => a.studentId))).filter(Boolean);

                const generated: StatisticsItem[] = [
                    {
                        id: "course-pdf-1",
                        label: `${examTitle} 과목 통계 PDF 다운로드`,
                        kind: "course-pdf",
                    },
                    {
                        id: "attendance-xlsx-1",
                        label: `${examTitle} 출석부 XLSX 다운로드`,
                        kind: "attendance-xlsx",
                    },
                    ...uniqueIds.map(sid => ({
                        id: `student-zip-${sid}`,
                        label: `${examTitle} | ${sid} (학생 답안지, 정오표)`,
                        kind: "student-zip" as const,
                    }))
                ];
                setDisplayItems(generated);
            } catch (error) {
                console.error("Failed to load students for StatisticsDownload:", error);
            }
        };

        loadContent();
    }, [items, examCode, examTitle]);

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        let result = displayItems;
        if (q) {
            result = displayItems.filter((it) => it.label.toLowerCase().includes(q));
        }
        return result.sort((a, b) => {
            const order = { 'course-pdf': 0, 'attendance-xlsx': 1, 'student-zip': 2 };
            return (order[a.kind] ?? 99) - (order[b.kind] ?? 99);
        });
    }, [displayItems, query]);

    const toggleSelect = (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        const next = new Set(selectedIds);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        setSelectedIds(next);
    };

    const toggleAll = () => {
        if (selectedIds.size === filtered.length) {
            setSelectedIds(new Set());
        } else {
            setSelectedIds(new Set(filtered.map((it) => it.id)));
        }
    };

    const handleDownload = async (item: StatisticsItem) => {
        if (item.kind === "attendance-xlsx") {
            try {
                const url = await examService.getAttendanceDownloadUrl(examCode);
                window.open(url, "_blank");
            } catch (error) {
                console.error("Attendance download failed:", error);
                alert("출석부 다운로드에 실패했습니다.");
            }
        } else {
            onDownload?.(item);
        }
    };

    const handleDownloadSelected = () => {
        const selectedItems = filtered.filter(it => selectedIds.has(it.id));
        selectedItems.forEach(it => handleDownload(it));
    };

    const statGroup = filtered.filter(it => it.kind === 'course-pdf' || it.kind === 'attendance-xlsx');
    const studentGroup = filtered.filter(it => it.kind === 'student-zip');
    const selectedStudentCount = studentGroup.filter(it => selectedIds.has(it.id)).length;

    return (
        <div className="relative mx-auto min-h-screen w-[1152px] bg-white pb-24 font-semibold">
            {/* Gradi Logo */}
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

            {/* Header Content Wrapper */}
            <div className="pt-[120px] px-6 relative z-10 animate-fade-in-up">
                <div className="flex justify-between items-end mb-2">
                    <div>
                        <h1 className="text-[40px] font-semibold leading-[48px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                            {examTitle}({examCode})
                        </h1>
                        <p className="text-[#A0A0A0] text-[20px] font-medium leading-[29px]">
                            과목 통계와 학생별 ZIP(학생이 응답한 답안지, 정오표)를 다운로드합니다.
                        </p>
                    </div>

                    {/* Student Access Control Panel */}
                    <div className="flex flex-col items-end gap-2">
                        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${isStudentAccess ? 'bg-purple-50 border-purple-200' : 'bg-gray-50 border-gray-200'}`}>
                            <div className={`w-3 h-3 rounded-full ${isStudentAccess ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
                            <span className={`text-sm font-bold ${isStudentAccess ? 'text-purple-700' : 'text-gray-500'}`}>
                                {isStudentAccess ? '학생 결과 페이지 공개 중' : '학생 결과 페이지 비공개'}
                            </span>
                            <label className="relative inline-flex items-center cursor-pointer ml-2">
                                <input type="checkbox" checked={isStudentAccess} onChange={toggleStudentAccess} className="sr-only peer" />
                                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#AC5BF8]"></div>
                            </label>
                        </div>
                        {isStudentAccess && (
                            <a
                                href={`/student/verify/${examCode}`}
                                target="_blank"
                                className="text-xs text-[#AC5BF8] underline hover:text-purple-700"
                            >
                                학생 페이지 링크 열기 ↗
                            </a>
                        )}
                    </div>
                </div>

                {/* Main Content Area (Frame) */}
                <div className="border-[#AC5BF8] border-[3px] rounded-lg p-6 bg-[#F8F0FF] shadow-md space-y-6">



                    {/* Search Bar */}
                    <div className="relative h-[60px] w-full">
                        <input
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="h-full w-full rounded border border-black px-4 text-xl bg-white focus:outline-none focus:ring focus:ring-purple-300 placeholder:text-gray-400 font-medium"
                            placeholder="과목명 / 학번으로 검색"
                        />
                        <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                        </div>
                    </div>

                    <div className="h-px bg-purple-200" />

                    {/* Selection Action Buttons */}
                    <div className="flex justify-between items-center px-2">
                        <div className="flex items-center gap-4">
                            <Button
                                label={selectedIds.size === filtered.length && filtered.length > 0 ? "전체 해제" : "전체 선택"}
                                className="w-[140px] h-[40px] text-[18px]"
                                onClick={toggleAll}
                            />
                            <span className="text-[18px] text-[#707070] font-medium">
                                선택됨: <span className="text-[#AC5BF8] font-bold">{selectedIds.size}</span> / {filtered.length}
                            </span>
                        </div>
                        <Button
                            label="선택 다운로드"
                            className="w-[180px] h-[45px] text-[20px] font-bold"
                            onClick={handleDownloadSelected}
                            disabled={selectedIds.size === 0}
                            style={{ opacity: selectedIds.size === 0 ? 0.5 : 1 }}
                        />
                    </div>

                    {/* Download List Grouped */}
                    <div className="space-y-12">
                        {/* 1. Statistics Group */}
                        <div className="space-y-4">
                            <div className="flex justify-between items-end border-b border-purple-200 pb-2">
                                <h2 className="text-2xl font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent ml-1">시험 통계 데이터</h2>
                                <span className="text-gray-500 font-medium">총 {statGroup.length}개</span>
                            </div>
                            <div className="space-y-3">
                                {statGroup.map((item) => (
                                    <DownloadItem
                                        key={item.id}
                                        item={item}
                                        selected={selectedIds.has(item.id)}
                                        onToggle={(id, e) => toggleSelect(id, e)}
                                        onDownload={() => handleDownload(item)}
                                    />
                                ))}
                            </div>
                        </div>

                        {/* 2. Individual Students Group */}
                        <div className="space-y-4">
                            <div className="flex justify-between items-end border-b border-purple-200 pb-2">
                                <h2 className="text-2xl font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent ml-1">학생 개별 결과</h2>
                                <span className="text-gray-500 font-medium">선택됨 <span className="text-purple-600 font-bold">{selectedStudentCount}</span> / {studentGroup.length}개</span>
                            </div>
                            <div className="space-y-3">
                                {studentGroup.length > 0 ? (
                                    studentGroup.map((item) => (
                                        <DownloadItem
                                            key={item.id}
                                            item={item}
                                            selected={selectedIds.has(item.id)}
                                            onToggle={(id, e) => toggleSelect(id, e)}
                                            onDownload={() => handleDownload(item)}
                                        />
                                    ))
                                ) : (
                                    <div className="text-center py-10 text-xl text-gray-400 italic bg-white rounded border border-gray-200">
                                        열람할 수 있는 학생 데이터가 없습니다.
                                    </div>
                                )}
                            </div>
                        </div>

                        {filtered.length === 0 && (
                            <div className="text-center py-20 text-2xl text-gray-400 italic font-medium">
                                검색 결과가 없습니다.
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const DownloadItem: React.FC<{
    item: StatisticsItem;
    selected: boolean;
    onToggle: (id: string, e: React.MouseEvent) => void;
    onDownload: () => void;
}> = ({ item, selected, onToggle, onDownload }) => {
    const isHighlighted = item.kind === 'course-pdf' || item.kind === 'attendance-xlsx';
    const borderClass = selected ? "border-[#AC5BF8] ring-2 ring-[#AC5BF8]/20" : "border-black";
    const textClass = isHighlighted
        ? "bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent font-bold"
        : "text-gray-900 font-bold";

    const labelParts = item.label.split('|');

    return (
        <div
            className={`relative flex min-h-[64px] w-full items-center rounded border p-4 transition-all bg-white hover:bg-purple-50 cursor-pointer ${borderClass}`}
            onClick={onDownload}
        >
            <div
                className={`mr-4 w-6 h-6 border-2 rounded flex-shrink-0 flex items-center justify-center transition-colors ${selected ? "bg-[#AC5BF8] border-[#AC5BF8]" : "bg-white border-gray-300"}`}
                onClick={(e) => onToggle(item.id, e)}
            >
                {selected && (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="4">
                        <polyline points="20 6 9 17 4 12" />
                    </svg>
                )}
            </div>

            <div className="flex-grow flex items-center text-xl overflow-hidden mr-4">
                <span className={textClass}>
                    {labelParts.map((part, index) => (
                        <React.Fragment key={index}>
                            {part.trim()}
                            {index < labelParts.length - 1 && (
                                <span className="mx-2 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent font-extrabold">|</span>
                            )}
                        </React.Fragment>
                    ))}
                </span>
            </div>

            <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onDownload(); }}
                className="ml-auto w-10 h-10 rounded-full border border-purple-200 flex items-center justify-center bg-white hover:bg-purple-100 transition-colors shadow-sm"
            >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#AC5BF8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 5v14M19 12l-7 7-7-7" />
                </svg>
            </button>
        </div>
    );
};

export default StatisticsDownload;
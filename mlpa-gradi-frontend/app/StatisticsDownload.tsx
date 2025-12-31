import React, { useMemo, useState } from "react";

type DownloadKind = "course-pdf" | "student-zip";

type StatisticsItem = {
    id: string;
    label: string;
    kind: DownloadKind;
};

const MOCK_ITEMS: StatisticsItem[] = [
    {
        id: "course-pdf-1",
        label: "인공지능 2025-2분반 중간고사 과목 통계 PDF 다운로드",
        kind: "course-pdf",
    },
    {
        id: "student-zip-32204077",
        label: "인공지능 2025-2분반 중간고사 | 32204077 (학생 답안지, 정오표)",
        kind: "student-zip",
    },
    {
        id: "student-zip-32201959",
        label: "인공지능 2025-2분반 중간고사 | 32201959 (학생 답안지, 정오표)",
        kind: "student-zip",
    },
];

interface StatisticsDownloadProps {
    items?: StatisticsItem[];
    onDownload?: (item: StatisticsItem) => void;
    examTitle?: string; // Add this
}

const StatisticsDownload: React.FC<StatisticsDownloadProps> = ({
    items = MOCK_ITEMS,
    onDownload,
    examTitle = "시험 통계", // Default title
}) => {
    const [query, setQuery] = useState("");

    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return items;
        return items.filter((it) => it.label.toLowerCase().includes(q));
    }, [items, query]);

    return (
        <div className="relative mx-auto h-[700px] w-[1152px] bg-white">
            {/* KakaoTalk Logo */}
            <div
                className="absolute left-[10px] top-[17px] h-[43px] w-[165px]"
                style={{
                    backgroundImage: "url('/KakaoTalk_20251125_001618855.png')",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Title */}
            {/* Title */}
            <h1 className="absolute left-0 top-[96px] w-fit text-[48px] font-semibold leading-[57px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent animate-fade-in-up">
                {examTitle}
            </h1>

            {/* Divider Removed */}

            {/* Description */}
            <p className="absolute right-0 top-[124px] text-[20px] font-medium leading-[29px] text-[#A0A0A0] animate-fade-in-up">
                과목 통계와 학생별 ZIP(학생이 응답한 답안지, 정오표)를 다운로드합니다.
            </p>

            {/* Frame */}
            <section className="absolute left-0 top-[159px] h-[459px] w-[1152px] bg-[#F1E2FF] rounded-lg shadow-md animate-fade-in-up border-[3px] border-[#AC5BF8]">
                {/* Search Bar */}
                <div className="absolute left-1/2 top-[27px] h-[72px] w-[700px] -translate-x-1/2">
                    <div className="relative h-full w-full">
                        <input
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            className="h-full w-full rounded-[50px] border border-[#E1E7ED] bg-[#F8F9FA] pl-[32px] pr-[72px] text-[24px] font-medium leading-[29px] text-black placeholder:text-[#C8D3DE] focus:outline-none focus:ring-2 focus:ring-[#AC5BF8]/35"
                            placeholder="과목명 / 학번으로 검색"
                        />
                        {/* simple search icon */}
                        <div className="pointer-events-none absolute right-[22px] top-1/2 h-[46px] w-[46px] -translate-y-1/2">
                            <div className="relative h-full w-full">
                                <div className="absolute left-[12px] top-[12px] h-[20px] w-[20px] rounded-full border-[2px] border-[#C8D3DE]" />
                                <div className="absolute left-[28px] top-[28px] h-[14px] w-[2px] rotate-45 bg-[#C8D3DE]" />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Download List */}
                <div className="absolute left-1/2 top-[136px] w-[1099px] -translate-x-1/2 space-y-[22px] h-[300px] overflow-y-auto pr-2">
                    {filtered.sort((a, b) => (a.kind === 'course-pdf' ? -1 : 1)).map((item) => {
                        const isCourse = item.kind === 'course-pdf';
                        // All boxes have black border as per latest request
                        const borderClass = "border border-black";
                        // Course text is Gradient Purple, others Black. Added w-fit for gradient visibility.
                        const textClass = isCourse
                            ? "bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent font-semibold w-fit block"
                            : "text-black";

                        // Parse label for pipes
                        const labelParts = item.label.split('|');

                        return (
                            <div
                                key={item.id}
                                className={`relative flex h-[62px] w-full items-center rounded-[5px] bg-white px-6 ${borderClass} cursor-pointer hover:bg-gray-50`}
                            >
                                <span className={`text-[24px] font-normal leading-[29px] truncate w-full ${textClass}`}>
                                    {labelParts.map((part, index) => (
                                        <React.Fragment key={index}>
                                            {part}
                                            {index < labelParts.length - 1 && (
                                                <span className="mx-2 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent inline-block font-extrabold">|</span>
                                            )}
                                        </React.Fragment>
                                    ))}
                                </span>

                                <button
                                    type="button"
                                    aria-label="download"
                                    onClick={() => onDownload?.(item)}
                                    className="absolute right-[18px] top-1/2 h-[40px] w-[40px] -translate-y-1/2 flex items-center justify-center rounded transition-colors hover:bg-purple-50 cursor-pointer"
                                >
                                    {/* New Download Icon - Simple Arrow Down + Line - Gradient */}
                                    <div className="w-[32px] h-[32px] rounded-full p-[2px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF]">
                                        <div className="w-full h-full rounded-full bg-white flex items-center justify-center">
                                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                                <defs>
                                                    <linearGradient id={`grad-${item.id}`} x1="0%" y1="0%" x2="100%" y2="0%">
                                                        <stop offset="0%" stopColor="#AC5BF8" />
                                                        <stop offset="100%" stopColor="#636ACF" />
                                                    </linearGradient>
                                                </defs>
                                                <path d="M12 5V19" stroke={`url(#grad-${item.id})`} />
                                                <path d="M19 12L12 19L5 12" stroke={`url(#grad-${item.id})`} />
                                            </svg>
                                        </div>
                                    </div>
                                </button>
                            </div>
                        );
                    })}
                    {filtered.length === 0 && (
                        <div className="rounded-[5px] bg-white px-6 py-5 text-[24px] font-medium leading-[29px] text-[#A0A0A0]">
                            검색 결과가 없습니다.
                        </div>
                    )}
                </div>
            </section>
        </div>
    );
};

export default StatisticsDownload;
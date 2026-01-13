import React from "react";

interface ExamInfoFormProps {
    examTitle: string;
    setExamTitle: (value: string) => void;
    examDate: string;
    setExamDate: (value: string) => void;
    isStudentResultEnabled: boolean;
    setIsStudentResultEnabled: (value: boolean) => void;
}

export const ExamInfoForm: React.FC<ExamInfoFormProps> = ({
    examTitle,
    setExamTitle,
    examDate,
    setExamDate,
    isStudentResultEnabled,
    setIsStudentResultEnabled
}) => {
    return (
        <div>
            <div className="flex justify-between items-end mb-8">
                <h2 className="text-[44px] font-extrabold leading-[52px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">시험 정보</h2>
                <p className="text-[#A0A0A0] text-[22px] font-medium leading-[31px]">시험에 대한 기본 정보를 입력합니다.</p>
            </div>
            <div className="border-[#AC5BF8] border-[3px] rounded-lg p-6 bg-[#F8F0FF] shadow-md space-y-6 font-semibold">
                {/* Exam Title */}
                <div>
                    <label className="block text-lg font-semibold mb-2">시험 이름</label>
                    <input
                        type="text"
                        value={examTitle}
                        onChange={(e) => setExamTitle(e.target.value)}
                        placeholder="시험의 이름을 입력해주세요 ex) 인공지능 2025-2분반 중간고사"
                        className="w-full border border-black p-3 rounded focus:outline-none focus:ring focus:ring-purple-300 bg-white"
                    />
                </div>

                {/* Exam Date & Student Result Option */}
                <div className="flex flex-col gap-6">
                    <div className="w-1/2">
                        <label className="block text-lg font-semibold mb-2">시험 일시</label>
                        <div className="relative w-full">
                            <style jsx>{`
                                input[type="datetime-local"]::-webkit-calendar-picker-indicator {
                                    display: none;
                                }
                            `}</style>
                            <input
                                type="datetime-local"
                                value={examDate}
                                onChange={(e) => setExamDate(e.target.value)}
                                className="w-full border border-black p-3 rounded focus:outline-none focus:ring focus:ring-purple-300 text-gray-500 bg-white cursor-pointer"
                                onClick={(e) => e.currentTarget.showPicker()}
                            />
                            {/* Custom Gradient Calendar Icon */}
                            <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                                    <defs>
                                        <linearGradient id="cal-grad" x1="0%" y1="0%" x2="100%" y2="0%">
                                            <stop offset="0%" stopColor="#AC5BF8" />
                                            <stop offset="100%" stopColor="#636ACF" />
                                        </linearGradient>
                                    </defs>
                                    <path d="M19 4H5C3.89543 4 3 4.89543 3 6V20C3 21.1046 3.89543 22 5 22H19C20.1046 22 21 21.1046 21 20V6C21 4.89543 20.1046 4 19 4Z" stroke="url(#cal-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M16 2V6" stroke="url(#cal-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M8 2V6" stroke="url(#cal-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                    <path d="M3 10H21" stroke="url(#cal-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </div>
                        </div>
                    </div>

                    {/* Student Result Page Checkbox */}
                    <div className="flex items-center gap-3 p-4 bg-white/50 rounded-lg border border-purple-100">
                        <div className="relative flex items-center">
                            <input
                                type="checkbox"
                                id="student-result-check"
                                checked={isStudentResultEnabled}
                                onChange={(e) => setIsStudentResultEnabled(e.target.checked)}
                                className="peer h-6 w-6 cursor-pointer appearance-none rounded border border-gray-300 shadow-sm checked:border-[#AC5BF8] checked:bg-[#AC5BF8] hover:border-[#AC5BF8] focus:outline-none focus:ring-2 focus:ring-[#AC5BF8]/50"
                            />
                            <svg
                                className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-white opacity-0 peer-checked:opacity-100"
                                width="16"
                                height="16"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="3"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            >
                                <polyline points="20 6 9 17 4 12" />
                            </svg>
                        </div>
                        <label htmlFor="student-result-check" className="cursor-pointer select-none text-lg font-semibold text-gray-700">
                            학생들에게 보여줄 결과확인 페이지 생성하기
                        </label>
                    </div>
                </div>
            </div>
        </div>
    );
};

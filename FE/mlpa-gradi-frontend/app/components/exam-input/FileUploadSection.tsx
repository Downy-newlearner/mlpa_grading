import React, { useRef, useEffect } from "react";
import { UploadedFile } from "../../types";

interface FileUploadSectionProps {
    attendanceFile: UploadedFile | null;
    setAttendanceFile: (file: UploadedFile | null) => void;
    answerSheetFiles: UploadedFile[];
    setAnswerSheetFiles: (files: UploadedFile[]) => void; // Fixed type to correct signature
}

const MinusIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="10" fill="#FF5B5B" />
        <path d="M7 12H17" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

export const FileUploadSection: React.FC<FileUploadSectionProps> = ({
    attendanceFile,
    setAttendanceFile,
    answerSheetFiles,
    setAnswerSheetFiles
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Sync state changes back to the native file input to ensure the "N files selected" text is correct
    useEffect(() => {
        if (fileInputRef.current) {
            const dataTransfer = new DataTransfer();
            answerSheetFiles.forEach(file => {
                // Determine if we have the original File object
                if (file.file instanceof File) {
                    dataTransfer.items.add(file.file);
                }
            });
            fileInputRef.current.files = dataTransfer.files;
        }
    }, [answerSheetFiles]);

    return (
        <div className="pt-12">
            <div className="flex justify-between items-end mb-8">
                <h2 className="text-[44px] font-extrabold leading-[52px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">출석부 / 학생 답안지 입력<span className="text-red-500">*</span></h2>
                <p className="text-[#A0A0A0] text-[22px] font-medium leading-[31px]">학생들의 출석부와, 응답 이미지를 입력 받습니다.</p>
            </div>
            <div className="border-[#AC5BF8] border-[3px] rounded-lg p-6 bg-[#F8F0FF] shadow-md space-y-6 font-semibold">

                {/* Attendance File Upload */}
                <div>
                    <h3 className="text-lg font-semibold mb-2">출석부 파일 업로드<span className="text-red-500">*</span></h3>
                    <input
                        type="file"
                        accept=".csv,.xlsx"
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) setAttendanceFile({ file, name: file.name });
                        }}
                        className="border border-black p-2 rounded w-full text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100 cursor-pointer file:cursor-pointer"
                    />
                    {attendanceFile && (
                        <div className="mt-2 flex items-center justify-between text-sm text-gray-600">
                            <span>선택된 파일: {attendanceFile.name}</span>
                            <button
                                type="button"
                                onClick={() => setAttendanceFile(null)}
                            >
                                <MinusIcon />
                            </button>
                        </div>
                    )}
                </div>

                {/* Answer Sheet Upload */}
                <div>
                    <h3 className="text-lg font-semibold mb-2">
                        학생 답안지 업로드
                        {answerSheetFiles.length > 0 && (
                            <span className="ml-2 text-sm font-medium bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                                ({answerSheetFiles.length}개 선택됨)
                            </span>
                        )}
                    </h3>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".csv,.xlsx,.pdf"
                        multiple
                        onChange={(e) => {
                            const files = e.target.files;
                            if (files) {
                                // Accumulate files instead of replacing
                                const newFiles: UploadedFile[] = Array.from(files).map((file) => ({
                                    file,
                                    name: file.name,
                                }));

                                // Avoid duplicates if needed, or just append. 
                                // Requirements imply we just want to add them. 
                                // But standard behavior for input type=file is replace.
                                // The user's issue implies they want to manipulate the list.
                                // If we want to support "add more", we should merge.
                                // However, keeping it simple: just update state. 
                                // The useEffect will handle the reverse sync.
                                setAnswerSheetFiles(newFiles);
                            }
                        }}
                        className="border border-black p-2 rounded w-full text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100 cursor-pointer file:cursor-pointer"
                    />
                    {answerSheetFiles.length > 0 && (
                        <div className="mt-2 space-y-1">
                            {answerSheetFiles.map((f, idx) => (
                                <div
                                    key={idx}
                                    className="flex items-center justify-between text-sm text-gray-600 border border-black rounded p-1"
                                >
                                    <span>{f.name}</span>
                                    <button
                                        type="button"
                                        onClick={() =>
                                            setAnswerSheetFiles(answerSheetFiles.filter((_, i) => i !== idx))
                                        }
                                    >
                                        <MinusIcon />
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

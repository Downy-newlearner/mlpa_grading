"use client";

import React, { useEffect, useState } from "react";

interface GradingLoadingProps {
    examCode?: string;
}

const GradingLoading: React.FC<GradingLoadingProps> = ({ examCode = "ND1FHG" }) => {
    const [currentImage, setCurrentImage] = useState<string | null>(null);
    const [seconds, setSeconds] = useState(0);
    const [studentCount, setStudentCount] = useState(14);
    const [isVisible, setIsVisible] = useState(true);

    useEffect(() => {
        // Mock SSE Connection
        const timer = setTimeout(() => {
            setCurrentImage("/mock_answer_sheet.png");
        }, 1000); // Simulate receiving first image after 1s
        return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            setSeconds((prev) => prev + 1);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            if (studentCount >= 40) return;
            setIsVisible(false);
            setTimeout(() => {
                setStudentCount((prev) => (prev < 40 ? prev + 1 : prev));
                setIsVisible(true);
            }, 500);
        }, 4000);
        return () => clearInterval(interval);
    }, [studentCount]);

    return (
        <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center">
            <div
                className="absolute top-[30px] left-[30px] w-[120px] h-[32px]"
                style={{
                    backgroundImage: "url('/Gradi_logo.png')",
                    backgroundSize: "contain",
                    backgroundRepeat: "no-repeat",
                }}
            />

            <div className="flex gap-12 items-center mb-10">
                {/* Spinner */}
                <div className="relative">
                    <svg
                        className="animate-spin w-32 h-32"
                        viewBox="0 0 100 100"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <defs>
                            <linearGradient id="spinner-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" stopColor="#AC5BF8" />
                                <stop offset="100%" stopColor="#636ACF" />
                            </linearGradient>
                        </defs>
                        <circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke="url(#spinner-gradient)"
                            strokeWidth="8"
                            strokeLinecap="round"
                            strokeDasharray="200"
                            strokeDashoffset="100"
                        />
                    </svg>
                </div>

                {/* Live Image Display */}
                {currentImage ? (
                    <div className="w-[300px] h-[400px] border-4 border-[#AC5BF8] rounded-lg overflow-hidden shadow-xl animate-pulse">
                        <img
                            src={currentImage}
                            alt="Grading Stream"
                            className="w-full h-full object-contain bg-gray-100"
                        />
                        <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
                            LIVE
                        </div>
                    </div>
                ) : (
                    <div className="w-[300px] h-[400px] border-4 border-gray-200 rounded-lg flex items-center justify-center bg-gray-50 text-gray-400">
                        이미지 수신 대기중...
                    </div>
                )}
            </div>

            <div className="text-center space-y-4">
                <p className="text-black text-[40px] font-bold leading-tight">
                    교수님의 피드백을 기반으로 채점을 하고 있어요
                </p>
                <p
                    className={`text-gray-500 text-3xl font-semibold transition-opacity duration-500 ${isVisible ? "opacity-100" : "opacity-0"
                        }`}
                >
                    학생{" "}
                    <span className="inline-block font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        {studentCount}
                    </span>
                    명의 채점을 완료했어요!{" "}
                    <span className="inline-block font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        ({studentCount}/40)
                    </span>
                </p>

                <p className="text-3xl font-extrabold mt-4 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                    시험코드 : {examCode}
                </p>

                <p className="text-xl font-bold text-gray-400">
                    현재 <span className="text-[#AC5BF8]">{seconds}</span>초 경과
                </p>
            </div>
        </div>
    );
};

export default GradingLoading;

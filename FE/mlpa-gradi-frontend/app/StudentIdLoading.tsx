"use client";

import React, { useEffect, useState } from "react";

interface StudentIdLoadingProps {
    examCode?: string;
}

const StudentIdLoading: React.FC<StudentIdLoadingProps> = ({ examCode = "ND1FHG" }) => {
    const [seconds, setSeconds] = useState(0);
    const [studentCount, setStudentCount] = useState(14);
    const [isVisible, setIsVisible] = useState(true);

    // Timer for elapsed seconds
    useEffect(() => {
        const interval = setInterval(() => {
            setSeconds((prev) => prev + 1);
        }, 1000);
        return () => clearInterval(interval);
    }, []);

    // Simulation of student recognition (14 -> 15 -> 16... up to 40)
    // This logic demonstrates the fade animation requested by the user.
    useEffect(() => {
        const interval = setInterval(() => {
            if (studentCount >= 40) return;

            // Start fade out
            setIsVisible(false);

            // Wait for fade out, then update and fade in
            setTimeout(() => {
                setStudentCount((prev) => (prev < 40 ? prev + 1 : prev));
                setIsVisible(true);
            }, 500); // 500ms matches the transition duration
        }, 4000); // Change count every 4 seconds

        return () => clearInterval(interval);
    }, [studentCount]);

    return (
        <div className="relative w-[1152px] h-[700px] bg-white mx-auto flex flex-col justify-center items-center">
            {/* Gradi Logo */}
            <div
                className="absolute top-[30px] left-[30px] w-[120px] h-[32px]"
                style={{
                    backgroundImage: "url('/Gradi_logo.png')",
                    backgroundSize: "contain",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Large Gradient Spinner Animation */}
            <div className="relative mb-14">
                <svg
                    className="animate-spin w-32 h-32" // Much larger size
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
                        strokeDasharray="200" // Creates the "gap" or "segment" effect
                        strokeDashoffset="100"
                    />
                </svg>
            </div>

            {/* Main Text */}
            <div className="text-center space-y-6">
                <p className="text-black text-[44px] font-bold leading-tight">
                    학생들의 학번을 인식하고 있어요
                </p>
                {/* Animated Paragraph: Entire line fades */}
                <p
                    className={`text-gray-500 text-3xl font-semibold transition-opacity duration-500 ${isVisible ? "opacity-100" : "opacity-0"
                        }`}
                >
                    학생{" "}
                    <span className="inline-block font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        {studentCount}
                    </span>
                    명의 인식을 완료했어요!{" "}
                    <span className="inline-block font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                        ({studentCount}/40)
                    </span>
                </p>

                {/* Exam Code */}
                <p className="text-4xl font-extrabold mt-8 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                    시험코드 : {examCode}
                </p>

                {/* Timer */}
                <p className="text-2xl font-bold text-gray-400 mt-4">
                    현재 <span className="text-[#AC5BF8]">{seconds}</span>초 경과
                </p>
            </div>
        </div>
    );
};

export default StudentIdLoading;

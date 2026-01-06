"use client";

import React from "react";
import { useRouter } from "next/navigation";

interface QuestionRecognitionDoneProps {
    onNext: () => void;
}

const QuestionRecognitionDone: React.FC<QuestionRecognitionDoneProps> = ({ onNext }) => {
    const router = useRouter();

    return (
        <div className="relative mx-auto w-[1152px] h-[700px] bg-white overflow-hidden">
            {/* Gradi Logo - Click to go home */}
            <div
                className="absolute w-[165px] h-[43px] left-[10px] top-[17px] cursor-pointer hover:opacity-80 transition-opacity z-10"
                onClick={() => router.push("/")}
                style={{
                    backgroundImage: "url(/Gradi_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Animated Circle Background */}
            <div
                className="absolute w-[286px] h-[286px] left-[calc(50%-286px/2)] top-[170px] rounded-full animate-pulse-ring"
                style={{
                    background: "linear-gradient(121.67deg, #AC5BF8 19.64%, #636ACF 77.54%)",
                    opacity: 0.3,
                }}
            />

            {/* Rotating Ring Effect */}
            <div className="absolute w-[260px] h-[260px] left-[calc(50%-130px)] top-[183px] animate-spin-slow">
                <svg viewBox="0 0 260 260" className="w-full h-full">
                    <circle
                        cx="130"
                        cy="130"
                        r="120"
                        fill="none"
                        stroke="url(#gradientQ)"
                        strokeWidth="4"
                        strokeDasharray="60 30"
                        strokeLinecap="round"
                    />
                    <defs>
                        <linearGradient id="gradientQ" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#AC5BF8" />
                            <stop offset="100%" stopColor="#636ACF" />
                        </linearGradient>
                    </defs>
                </svg>
            </div>

            {/* Check Icon Container with Pop Animation */}
            <div
                className="absolute flex flex-col items-center justify-center gap-[10px] w-[212.67px] h-[217.46px] left-[calc(50%-212.67px/2+0.33px)] top-[calc(50%-217.46px/2-37.27px)] rounded-full shadow-2xl animate-pop-in"
                style={{
                    background: "linear-gradient(121.67deg, #AC5BF8 19.64%, #636ACF 77.54%)",
                }}
            >
                {/* Animated Check Mark */}
                <div className="w-[112.67px] h-[77.46px] flex items-center justify-center">
                    <svg width="113" height="78" viewBox="0 0 113 78" fill="none" xmlns="http://www.w3.org/2000/svg" className="animate-draw-check">
                        <path
                            d="M5 43.5L38.5 72.5L107.5 5"
                            stroke="white"
                            strokeWidth="13"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="check-path"
                        />
                    </svg>
                </div>
            </div>

            {/* Particle Effects */}
            <div className="absolute inset-0 pointer-events-none">
                {[...Array(8)].map((_, i) => (
                    <div
                        key={i}
                        className="absolute w-3 h-3 rounded-full animate-particle"
                        style={{
                            background: i % 2 === 0 ? "#AC5BF8" : "#636ACF",
                            left: `calc(50% + ${Math.cos((i * Math.PI) / 4) * 180}px)`,
                            top: `calc(50% - 37px + ${Math.sin((i * Math.PI) / 4) * 180}px)`,
                            animationDelay: `${i * 0.1}s`,
                        }}
                    />
                ))}
            </div>

            {/* Result Message with Fade In */}
            <div className="absolute w-[400px] left-[calc(50%-400px/2)] top-[487px] font-['Pretendard'] font-semibold text-[36px] leading-tight text-center text-[#5C5C5C] whitespace-pre-wrap animate-fade-in-up">
                문항 인식 완료!
            </div>

            {/* Next Button with Slide Up */}
            <div className="absolute top-[580px] left-[calc(50%-150px)] animate-slide-up">
                <button
                    onClick={onNext}
                    className="w-[300px] h-[60px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] rounded-[10px] text-white text-[24px] font-bold shadow-lg hover:scale-105 hover:shadow-xl active:scale-95 transition-all cursor-pointer"
                >
                    채점 계속하기
                </button>
            </div>

            <style jsx>{`
                @keyframes pop-in {
                    0% {
                        transform: scale(0);
                        opacity: 0;
                    }
                    50% {
                        transform: scale(1.15);
                    }
                    70% {
                        transform: scale(0.95);
                    }
                    100% {
                        transform: scale(1);
                        opacity: 1;
                    }
                }
                .animate-pop-in {
                    animation: pop-in 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55) forwards;
                }

                @keyframes draw-check {
                    0% {
                        stroke-dashoffset: 200;
                    }
                    100% {
                        stroke-dashoffset: 0;
                    }
                }
                .check-path {
                    stroke-dasharray: 200;
                    stroke-dashoffset: 200;
                    animation: draw-check 0.6s ease-out 0.5s forwards;
                }

                @keyframes spin-slow {
                    from {
                        transform: rotate(0deg);
                    }
                    to {
                        transform: rotate(360deg);
                    }
                }
                .animate-spin-slow {
                    animation: spin-slow 8s linear infinite;
                }

                @keyframes pulse-ring {
                    0%, 100% {
                        transform: scale(1);
                        opacity: 0.3;
                    }
                    50% {
                        transform: scale(1.1);
                        opacity: 0.5;
                    }
                }
                .animate-pulse-ring {
                    animation: pulse-ring 2s ease-in-out infinite;
                }

                @keyframes particle {
                    0% {
                        transform: scale(0) translate(0, 0);
                        opacity: 0;
                    }
                    30% {
                        transform: scale(1.2) translate(0, 0);
                        opacity: 1;
                    }
                    100% {
                        transform: scale(0) translate(calc(var(--x, 0) * 30px), calc(var(--y, 0) * 30px));
                        opacity: 0;
                    }
                }
                .animate-particle {
                    animation: particle 1.5s ease-out forwards;
                }

                @keyframes fade-in-up {
                    0% {
                        transform: translateY(20px);
                        opacity: 0;
                    }
                    100% {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                .animate-fade-in-up {
                    animation: fade-in-up 0.6s ease-out 0.8s forwards;
                    opacity: 0;
                }

                @keyframes slide-up {
                    0% {
                        transform: translateY(40px);
                        opacity: 0;
                    }
                    100% {
                        transform: translateY(0);
                        opacity: 1;
                    }
                }
                .animate-slide-up {
                    animation: slide-up 0.6s ease-out 1s forwards;
                    opacity: 0;
                }
            `}</style>
        </div>
    );
};

export default QuestionRecognitionDone;

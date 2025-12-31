import React from "react";

import Link from "next/link";
import Button from "./Button";

interface GradingDoneProps {
    examId?: string;
}

const GradingDone: React.FC<GradingDoneProps> = ({ examId }) => {
    return (
        <div className="relative mx-auto w-[1152px] h-[700px] bg-white">
            {/* Logo */}
            <div
                className="absolute w-[165px] h-[43px] left-[10px] top-[17px]"
                style={{
                    backgroundImage: "url(/Gradi_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Circle Background */}
            <div
                className="absolute w-[286px] h-[286px] left-[calc(50%-286px/2)] top-[170px] rounded-full opacity-50"
                style={{
                    background: "linear-gradient(121.67deg, #AC5BF8 19.64%, #636ACF 77.54%)",
                }}
            />

            {/* Check Icon Container */}
            <div
                className="absolute flex flex-col items-start p-[70px_50px] gap-[10px] w-[212.67px] h-[217.46px] left-[calc(50%-212.67px/2+0.33px)] top-[calc(50%-217.46px/2-37.27px)] rounded-[130px]"
                style={{
                    background: "linear-gradient(121.67deg, #AC5BF8 19.64%, #636ACF 77.54%)",
                }}
            >
                <div className="w-[112.67px] h-[77.46px] flex items-center justify-center">
                    <svg width="113" height="78" viewBox="0 0 113 78" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M5 43.5L38.5 72.5L107.5 5" stroke="white" strokeWidth="13" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </div>
            </div>

            {/* Result Message */}
            <div className="absolute w-[215px] h-[43px] left-[calc(50%-215px/2+0.5px)] top-[470px] font-['Pretendard'] font-semibold text-[36px] leading-[43px] text-center text-[#5C5C5C]">
                채점 완료!
            </div>

            {/* Button */}
            <div className="absolute left-[calc(50%-240px/2)] top-[550px]">
                <Link href={`/history/${examId}`}>
                    <Button
                        label="결과 확인"
                        className="w-[240px] px-4 py-2 text-2xl shadow cursor-pointer"
                    />
                </Link>
            </div>
        </div>
    );
};

export default GradingDone;

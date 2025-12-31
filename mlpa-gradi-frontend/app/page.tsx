import React from "react";
import Link from "next/link";
import Button from "@/app/components/Button";

const Landing: React.FC = () => {
    return (
        <div className="relative w-[1152px] h-[700px] bg-white mx-auto">
            {/* KakaoTalk Logo - Redirect to Home */}
            <Link
                href="/"
                className="absolute w-[165px] h-[43px] top-[17px] left-[10px] cursor-pointer"
                style={{
                    backgroundImage: "url(/Gradi_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            />

            {/* Center Image */}
            <div
                className="absolute w-[315px] h-[315px] left-[418px] top-[129.04px] animate-fade-in-up"
                style={{
                    backgroundImage: "url(/MLPA_logo.png)",
                    backgroundSize: "cover",
                    backgroundPosition: "center",
                    backgroundRepeat: "no-repeat",
                }}
            ></div>

            {/* Left Button -> /exam-input */}
            <div className="absolute top-[521.52px] left-[258px] animate-fade-in-up">
                <Link href="/exam-input" className="inline-block cursor-pointer">
                    <Button label="시험 채점" />
                </Link>
            </div>

            {/* Right Button -> /history */}
            <div className="absolute top-[521.52px] left-[677px] animate-fade-in-up">
                <Link href="/history" className="inline-block cursor-pointer">
                    <Button label="결과 확인" />
                </Link>
            </div>
        </div>
    );
};

export default Landing;
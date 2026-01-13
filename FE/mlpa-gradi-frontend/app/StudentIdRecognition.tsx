import React from "react";

const StudentIdRecognitionDone: React.FC = () => {
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

            {/* Gradient circle (Ellipse 133) */}
            <div className="absolute left-1/2 top-[170px] h-[286px] w-[286px] -translate-x-1/2 rounded-full bg-gradient-to-br from-[#AC5BF8] to-[#636ACF] opacity-50" />

            {/* Check icon container */}
            <div className="absolute left-1/2 top-[206px] h-[217.46px] w-[212.67px] -translate-x-1/2 rounded-[130px] bg-gradient-to-br from-[#AC5BF8] to-[#636ACF]">
                {/* White check mark (vector-like) */}
                <div className="absolute left-1/2 top-1/2 h-[77.46px] w-[112.67px] -translate-x-1/2 -translate-y-1/2">
                    <div className="absolute left-[18px] top-[34px] h-[22px] w-[44px] rotate-45 rounded-[12px] bg-white" />
                    <div className="absolute left-[44px] top-[24px] h-[22px] w-[74px] -rotate-45 rounded-[12px] bg-white" />
                </div>
            </div>

            {/* Result Message */}
            <p className="absolute left-1/2 top-[487px] w-[215px] -translate-x-1/2 text-center text-[36px] font-semibold leading-[43px] text-[#5C5C5C]">
                인식 완료!
            </p>
        </div>
    );
};

export default StudentIdRecognitionDone;
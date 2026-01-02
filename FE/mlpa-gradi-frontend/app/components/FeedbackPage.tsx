import React, { useEffect, useState } from "react";

type FeedbackItem = {
    id: string;
    imageUrl?: string; // 실제로는 서버/업로드에서 내려오는 이미지 URL
    value: string;
};

const mockItems: FeedbackItem[] = [
    { id: "1", imageUrl: undefined, value: "" },
    { id: "2", imageUrl: undefined, value: "" },
];

const FeedbackPage: React.FC<{ examCode?: string }> = ({ examCode = "ND1FHG" }) => {
    const [items, setItems] = useState<FeedbackItem[]>([]);

    useEffect(() => {
        const fetchUnknownImages = async () => {
            try {
                const response = await fetch(`http://localhost:8080/api/reports/unknown-images/${examCode}`);
                const urls: string[] = await response.json();
                setItems(urls.map((url, index) => ({
                    id: String(index),
                    imageUrl: url,
                    value: ""
                })));
            } catch (error) {
                console.error("Failed to fetch unknown images:", error);
            }
        };

        fetchUnknownImages();
    }, [examCode]);

    return (
        <div className="relative mx-auto h-[1453px] w-[1152px] bg-white">
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
            <h1 className="absolute left-0 top-[96px] text-[48px] font-semibold leading-[57px] text-black">
                학번 피드백
            </h1>

            {/* Divider */}
            <div className="absolute left-0 top-[159.48px] h-px w-[1152px] bg-black/20" />

            {/* Description */}
            <p className="absolute right-0 top-[124px] text-[24px] font-medium leading-[29px] text-[#A0A0A0]">
                모델이 인식 중 불확실한 개체들을 사용자에게 피드백 받습니다.
            </p>

            {/* Main Frame */}
            <section className="absolute left-0 top-[159px] h-[912px] w-[1152px] bg-[#F1E2FF]">
                {/* 카드 리스트 영역(피그마는 1개 예시처럼 보이지만, 확장 가능하게 구성) */}
                <div className="absolute left-[17px] top-[170px] w-[1120px] space-y-6">
                    {items.map((item) => (
                        <div
                            key={item.id}
                            className="relative h-[278px] w-[1120px] rounded-[5px] bg-white"
                        >
                            {/* Left: Image box */}
                            <div className="absolute left-[56px] top-[18px] h-[242px] w-[424px] rounded-[5px] border border-[#D9D9D9] bg-white">
                                {item.imageUrl ? (
                                    // eslint-disable-next-line @next/next/no-img-element
                                    <img
                                        src={item.imageUrl}
                                        alt="feedback target"
                                        className="h-full w-full rounded-[5px] object-cover"
                                    />
                                ) : (
                                    <div className="flex h-full w-full items-center justify-center text-[16px] font-medium text-[#A0A0A0]">
                                        이미지 영역
                                    </div>
                                )}
                            </div>

                            {/* Right: Input */}
                            <div className="absolute right-[60px] top-[88px] w-[575px]">
                                <input
                                    defaultValue={item.value}
                                    placeholder="이미지에 쓰여있는 답을 입력해주세요"
                                    className="h-[62px] w-full rounded-[5px] border border-[#D9D9D9] bg-white px-5 text-[24px] font-medium leading-[29px] text-black placeholder:text-[#A0A0A0] focus:outline-none focus:ring-2 focus:ring-[#AC5BF8]/40"
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Continue Button */}
            <button
                type="button"
                className="absolute left-[387px] top-[1130px] h-[127px] w-[379px] rounded-[5px] bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-[48px] font-semibold leading-[57px] text-white shadow-lg"
            >
                채점 계속하기
            </button>
        </div>
    );
};

export default FeedbackPage;
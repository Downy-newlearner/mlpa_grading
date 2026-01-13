"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import Image from "next/image";

export default function StudentVerifyPage() {
    const router = useRouter();
    const params = useParams();
    const examCode = (params?.examCode as string) || "";

    const [studentId, setStudentId] = useState("");
    const [error, setError] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        const email = localStorage.getItem("student_email");
        if (!email) {
            router.push(`/student/login?redirect=/student/verify/${examCode}`);
        }
    }, [router, examCode]);

    const handleVerify = async (e: React.FormEvent) => {
        e.preventDefault();
        setError("");
        setIsLoading(true);

        try {
            const response = await fetch("/api/auth/verify-dku", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ studentId }),
            });

            const data = await response.json();

            if (data.success) {
                // Verification Success
                sessionStorage.setItem(`verified_student_id_${examCode}`, studentId);
                router.push(`/student/result/${examCode}`);
            } else {
                // Verification Failed
                setError(data.message || "학번 정보가 일치하지 않습니다.");
                setIsLoading(false);
            }
        } catch (err) {
            console.error("Verification error:", err);
            setError("서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.");
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8 md:p-12 space-y-8 animate-fade-in-up">
                <div className="text-center space-y-2">
                    <div className="flex justify-center mb-6">
                        <div className="w-[165px] h-[43px] relative">
                            <Image
                                src="/Gradi_logo.png"
                                alt="Gradi Logo"
                                fill
                                className="object-cover"
                                priority
                            />
                        </div>
                    </div>
                    <h1 className="text-2xl font-bold text-gray-900">본인 확인</h1>
                    <p className="text-gray-500">
                        시험 결과를 확인하기 위해<br />학번을 입력해주세요.
                    </p>
                </div>

                <form onSubmit={handleVerify} className="space-y-6">
                    <div className="space-y-2">
                        <label htmlFor="studentId" className="block text-sm font-medium text-gray-700">
                            학번
                        </label>
                        <input
                            id="studentId"
                            type="text"
                            value={studentId}
                            onChange={(e) => setStudentId(e.target.value)}
                            placeholder="32210000"
                            maxLength={8}
                            className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#AC5BF8] focus:border-[#AC5BF8] outline-none transition-all placeholder:text-gray-400 tracking-wider font-mono text-lg"
                        />
                        {error && (
                            <p className="text-sm text-red-500 font-medium animate-shake">
                                {error}
                            </p>
                        )}
                    </div>

                    <button
                        type="submit"
                        disabled={isLoading}
                        className={`w-full py-3 px-4 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white font-bold rounded-lg shadow-md hover:shadow-lg transform transition-all duration-200 cursor-pointer ${isLoading ? 'opacity-70 cursor-wait' : 'hover:-translate-y-0.5'}`}
                    >
                        {isLoading ? "확인 중..." : "결과 확인하기"}
                    </button>

                    <p className="text-xs text-center text-gray-400">
                        본인의 학번을 정확히 입력해주세요.
                    </p>
                </form>
            </div>
        </div>
    );
}

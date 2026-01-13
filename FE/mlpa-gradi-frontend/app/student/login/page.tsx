"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";

export default function StudentLoginPage() {
    const router = useRouter();
    const searchParams = useSearchParams();
    const redirectUrl = searchParams.get("redirect") || "/student/dashboard";
    const [email, setEmail] = useState("");
    const [error, setError] = useState("");

    const handleLogin = (e: React.FormEvent) => {
        e.preventDefault();
        setError("");

        if (!email) {
            setError("이메일을 입력해주세요.");
            return;
        }

        if (!email.endsWith("@dankook.ac.kr")) {
            setError("단국대학교 이메일(@dankook.ac.kr)만 사용할 수 있습니다.");
            return;
        }

        // Mock Login Success
        localStorage.setItem("student_email", email);
        router.push(redirectUrl);
    };

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-4">
            <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-8 md:p-12 space-y-8 animate-fade-in-up">
                {/* Logo and Header */}
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
                    <h1 className="text-2xl font-bold text-gray-900">학생 로그인</h1>
                    <p className="text-gray-500">
                        시험 결과를 확인하려면<br />단국대학교 이메일로 로그인해주세요.
                    </p>
                </div>

                {/* Mock Login Form */}
                <form onSubmit={handleLogin} className="space-y-6">
                    <div className="space-y-2">
                        <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                            학교 이메일
                        </label>
                        <input
                            id="email"
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="example@dankook.ac.kr"
                            className="w-full px-4 py-3 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#AC5BF8] focus:border-[#AC5BF8] outline-none transition-all placeholder:text-gray-400"
                        />
                        {error && (
                            <p className="text-sm text-red-500 font-medium animate-shake">
                                {error}
                            </p>
                        )}
                    </div>

                    <button
                        type="submit"
                        className="w-full py-3 px-4 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white font-bold rounded-lg shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200 cursor-pointer"
                    >
                        로그인하기
                    </button>

                    <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t border-gray-200" />
                        </div>
                        <div className="relative flex justify-center text-sm">
                            <span className="bg-white px-2 text-gray-500">
                                This is a mock login page
                            </span>
                        </div>
                    </div>
                </form>
            </div>
        </div>
    );
}

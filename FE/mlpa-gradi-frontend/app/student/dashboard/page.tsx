"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

// Mock Data for Student's Exam History
const MOCK_STUDENT_EXAMS = [
    {
        examId: "exam_001",
        examCode: "AI202501",
        title: "인공지능 2025-1학기 중간고사",
        date: "2025-04-20",
        score: 85,
        totalScore: 100,
        studentId: "32204041"
    },
    {
        examId: "exam_002",
        examCode: "DB202501",
        title: "데이터베이스 시스템 기말고사",
        date: "2025-06-15",
        score: 92,
        totalScore: 100,
        studentId: "32204041"
    },
    {
        examId: "exam_003",
        examCode: "OS202502",
        title: "운영체제 중간고사",
        date: "2025-10-20",
        score: 78,
        totalScore: 100,
        studentId: "32204041"
    },
    // Exams for other students (should be filtered out)
    {
        examId: "exam_004",
        examCode: "AI202501",
        title: "인공지능 2025-1학기 중간고사",
        date: "2025-04-20",
        score: 88,
        totalScore: 100,
        studentId: "32204077"
    }
];

export default function StudentDashboardPage() {
    const router = useRouter();
    const [studentId, setStudentId] = useState("");
    const [isSearched, setIsSearched] = useState(false);
    const [myExams, setMyExams] = useState<typeof MOCK_STUDENT_EXAMS>([]);

    useEffect(() => {
        const email = localStorage.getItem("student_email");
        if (!email) {
            router.push("/student/login?redirect=/student/dashboard");
        }

        // Auto-fill student ID if previously verified
        // For now, let's keep it manual to demonstrate the "filtering" clearly
    }, [router]);

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();

        if (!/^\d{8}$/.test(studentId)) {
            alert("학번 8자리를 정확히 입력해주세요.");
            return;
        }

        // Filter exams for this student
        const filtered = MOCK_STUDENT_EXAMS.filter(exam => exam.studentId === studentId);
        setMyExams(filtered);
        setIsSearched(true);

        // Save verification for result pages
        MOCK_STUDENT_EXAMS.filter(exam => exam.studentId === studentId).forEach(exam => {
            sessionStorage.setItem(`verified_student_id_${exam.examCode}`, studentId);
        });
    };

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4">
            <div className="w-full max-w-4xl space-y-8 animate-fade-in-up">
                {/* Header */}
                <div className="flex justify-between items-center">
                    <div className="w-[165px] h-[43px] relative cursor-pointer" onClick={() => router.push("/")}>
                        <Image
                            src="/Gradi_logo.png"
                            alt="Gradi Logo"
                            fill
                            className="object-cover"
                        />
                    </div>
                    <button
                        onClick={() => {
                            localStorage.removeItem("student_email");
                            router.push("/student/login");
                        }}
                        className="text-gray-500 hover:text-purple-600 font-medium transition-colors"
                    >
                        로그아웃
                    </button>
                </div>

                {/* Search Section */}
                <div className="bg-white rounded-3xl shadow-xl p-8 border border-purple-100 text-center space-y-6">
                    <div>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent mb-2">
                            나의 시험 결과 조회
                        </h1>
                        <p className="text-gray-500">
                            학번을 입력하여 응시한 모든 시험의 결과를 확인하세요.
                        </p>
                    </div>

                    <form onSubmit={handleSearch} className="max-w-md mx-auto flex gap-3">
                        <input
                            type="text"
                            value={studentId}
                            onChange={(e) => setStudentId(e.target.value)}
                            placeholder="학번 입력 (ex. 32204041)"
                            maxLength={8}
                            className="flex-grow px-6 py-3 rounded-xl border border-gray-300 focus:ring-2 focus:ring-[#AC5BF8] focus:border-[#AC5BF8] outline-none transition-all text-lg font-mono tracking-wider"
                        />
                        <button
                            type="submit"
                            className="px-8 py-3 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white font-bold rounded-xl shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all text-lg whitespace-nowrap"
                        >
                            조회하기
                        </button>
                    </form>
                </div>

                {/* Results List */}
                {isSearched && (
                    <div className="space-y-4 animate-slide-up">
                        <div className="flex items-center justify-between px-2">
                            <h2 className="text-xl font-bold text-gray-800">
                                조회 결과 <span className="text-[#AC5BF8]">{myExams.length}</span>건
                            </h2>
                        </div>

                        {myExams.length > 0 ? (
                            <div className="grid grid-cols-1 gap-4">
                                {myExams.map((exam) => (
                                    <div
                                        key={exam.examId}
                                        onClick={() => router.push(`/student/result/${exam.examCode}`)}
                                        className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 hover:border-[#AC5BF8] hover:shadow-md transition-all cursor-pointer group flex justify-between items-center"
                                    >
                                        <div>
                                            <h3 className="text-xl font-bold text-gray-900 group-hover:text-[#AC5BF8] transition-colors mb-1">
                                                {exam.title}
                                            </h3>
                                            <div className="flex gap-4 text-sm text-gray-500 font-medium">
                                                <span>{exam.date}</span>
                                                <span className="text-purple-400">Code: {exam.examCode}</span>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <div className="text-right">
                                                <span className="block text-xs text-gray-400 font-bold">내 점수</span>
                                                <span className="text-2xl font-extrabold text-[#AC5BF8]">{exam.score}</span>
                                                <span className="text-sm text-gray-400 font-bold"> / {exam.totalScore}</span>
                                            </div>
                                            <div className="w-10 h-10 rounded-full bg-purple-50 flex items-center justify-center group-hover:bg-[#AC5BF8] transition-colors">
                                                <svg className="w-6 h-6 text-[#AC5BF8] group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                                                </svg>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="bg-white rounded-2xl p-12 text-center border border-gray-200">
                                <span className="text-4xl mb-4 block">📭</span>
                                <p className="text-lg font-medium text-gray-600">
                                    해당 학번으로 조회된 시험 결과가 없습니다.
                                </p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

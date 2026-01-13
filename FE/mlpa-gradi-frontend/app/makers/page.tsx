"use client";

import React from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";

interface Maker {
    name: string;
    role: string;
    department: string;
    studentId: string;
    mobile: string;
    email: string;
    ig: string;
    githubId: string;
    image: string;
    objectPosition?: string;
    isLeader?: boolean;
}

const makers: Maker[] = [
    {
        name: "정다훈",
        role: "AI",
        department: "단국대학교 SW융합대학 소프트웨어학과",
        studentId: "32204041",
        mobile: "+82 10-2391-8501",
        email: "jdh251425142514@gmail.com",
        ig: "@scentfuldowny_",
        githubId: "Downy-newlearner",
        image: "/makers/jeong_dahoon.png",
        objectPosition: "center 35%",
        isLeader: true,
    },
    {
        name: "정성원",
        role: "Full Stack",
        department: "단국대학교 SW융합대학 소프트웨어학과",
        studentId: "32204077",
        mobile: "+82 10-7511-3564",
        email: "woniwory@gmail.com",
        ig: "@_swan_1206",
        githubId: "woniwory",
        image: "/makers/jeong_sungwon.jpg",
        objectPosition: "center 25%",
    },
    {
        name: "조성빈",
        role: "Backend",
        department: "단국대학교 SW융합대학 소프트웨어학과",
        studentId: "32215110",
        mobile: "+82 10-5059-6484",
        email: "comicricky@naver.com",
        ig: "@rixxybin",
        githubId: "ricky00",
        image: "/makers/jo_seongbin.png",
        objectPosition: "center 15%",
    },
];

export default function MakersPage() {
    const router = useRouter();

    return (
        <div className="flex flex-col items-center min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="w-full max-w-7xl">
                <div className="flex justify-between items-center mb-12 animate-fade-in-up">
                    <h1 className="text-4xl font-extrabold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent tracking-tight">
                        만든 사람들
                    </h1>
                    <button
                        onClick={() => router.push("/")}
                        className="px-6 py-2 bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] text-white font-semibold rounded-full shadow-md hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200 cursor-pointer"
                    >
                        메인으로 돌아가기
                    </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-10">
                    {makers.map((maker, index) => (
                        <div
                            key={maker.name}
                            className={`
                bg-white rounded-2xl overflow-hidden transition-all duration-500 flex flex-col
                ${maker.isLeader ? 'shadow-[0_0_30px_rgba(172,91,248,0.3)] scale-[1.02] border-2 border-[#AC5BF8]/10' : 'shadow-xl hover:shadow-2xl border border-purple-100'}
                animate-fade-in-up
              `}
                            style={{ animationDelay: `${index * 150}ms` }}
                        >
                            <div className="relative h-96 w-full bg-gray-100">
                                <Image
                                    src={maker.image}
                                    alt={maker.name}
                                    fill
                                    className="object-cover"
                                    style={{ objectPosition: maker.objectPosition || "top" }}
                                />
                                {maker.isLeader && (
                                    <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1.5 z-10 border border-[#AC5BF8]/20">
                                        <span className="text-lg">👑</span>
                                        <span className="text-xs font-bold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">
                                            TEAM LEADER
                                        </span>
                                    </div>
                                )}
                            </div>
                            <div className="p-8 flex-1 flex flex-col">
                                <div className="flex justify-between items-end mb-4">
                                    <div className="flex items-center gap-2">
                                        <h2 className="text-3xl font-bold text-gray-900">{maker.name}</h2>
                                    </div>
                                    <span className="text-lg font-semibold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent bg-purple-50 px-3 py-1 rounded-lg">
                                        {maker.role}
                                    </span>
                                </div>


                                <div className="space-y-3 text-gray-600 text-sm flex-1">
                                    <p className="flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">Department</span>
                                        <span className="flex-1 truncate" title={maker.department}>{maker.department}</span>
                                    </p>
                                    <p className="flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">Student ID</span>
                                        <span className="flex-1">{maker.studentId}</span>
                                    </p>
                                    <p className="flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">Mobile</span>
                                        <span className="flex-1">{maker.mobile}</span>
                                    </p>
                                    <p className="flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">E-mail</span>
                                        <span className="flex-1 truncate" title={maker.email}>{maker.email}</span>
                                    </p>
                                    <p className="flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">IG</span>
                                        <span className="flex-1 font-semibold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent">{maker.ig}</span>
                                    </p>
                                    <div className="pt-2 mt-2 border-t border-gray-100 flex items-center">
                                        <span className="font-semibold w-24 text-gray-500">Github</span>
                                        <a
                                            href={`https://github.com/${maker.githubId}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex-1 font-semibold bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] bg-clip-text text-transparent hover:opacity-80 transition-opacity"
                                        >
                                            {maker.githubId}
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="mt-16 text-center text-gray-400 text-sm animate-fade-in-up" style={{ animationDelay: '500ms' }}>
                    <p>© 2026 Trender Team. All rights reserved.</p>
                </div>
            </div>
        </div>
    );
}

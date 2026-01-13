import React from "react";
import { QuestionType } from "../../types";

interface TypeSelectorProps {
    value: QuestionType;
    onChange: (value: QuestionType) => void;
}

export const TypeSelector: React.FC<TypeSelectorProps> = ({ value, onChange }) => {
    const options: { value: QuestionType; label: string; desc: string }[] = [
        { value: "binary", label: "OX", desc: "참/거짓" },
        { value: "short_answer", label: "단답형", desc: "텍스트/숫자" },
        { value: "objective", label: "객관식", desc: "선택형 정답" },
        { value: "others", label: "기타", desc: "피드백 기반" },
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {options.map((opt) => {
                const isSelected = value === opt.value;
                return (
                    <button
                        key={opt.value}
                        type="button"
                        onClick={() => onChange(opt.value)}
                        className={`
                            relative flex flex-col items-center justify-center p-2 rounded-lg border-2 transition-all duration-200
                            ${isSelected
                                ? "bg-[#AC5BF8] border-[#AC5BF8] text-white shadow-md transform scale-[1.02]"
                                : "bg-white border-gray-200 text-gray-600 hover:border-purple-200 hover:bg-purple-50"
                            }
                        `}
                    >
                        <span className={`text-base font-bold ${isSelected ? "text-white" : "text-gray-800"}`}>
                            {opt.label}
                        </span>
                        <span className={`text-xs mt-1 ${isSelected ? "text-purple-100" : "text-gray-400"}`}>
                            {opt.desc}
                        </span>
                    </button>
                );
            })}
        </div>
    );
};

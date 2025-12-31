// components/Button.tsx
import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    label: string;
}

const Button: React.FC<ButtonProps> = ({ label, className, ...props }) => {
    return (
        <button
            className={`bg-gradient-to-r from-[#AC5BF8] to-[#636ACF] rounded-[5px] flex justify-center items-center text-white font-semibold cursor-pointer ${className || "w-[216px] h-[102px] text-[40px]"
                }`}
            {...props}
        >
            {label}
        </button>
    );
};

export default Button;
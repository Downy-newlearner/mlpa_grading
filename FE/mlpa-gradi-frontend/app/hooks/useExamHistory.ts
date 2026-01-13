import { useState, useEffect, useMemo } from "react";
import { ExamHistoryItem } from "../types";
import { examService } from "../services/examService";

export const useExamHistory = (initialExams?: ExamHistoryItem[]) => {
    const [exams, setExams] = useState<ExamHistoryItem[]>(initialExams || []);
    const [loading, setLoading] = useState(!initialExams);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchExams = async () => {
            setLoading(true);
            try {
                const data = await examService.getAll();
                setExams(data);
            } catch (err) {
                setError(String(err));
            } finally {
                setLoading(false);
            }
        };

        fetchExams();
    }, []);

    const filterExams = (query: string) => {
        const q = query.trim().toLowerCase();
        if (!q) return exams;
        return exams.filter((e) => {
            const hay = `${e.examName} ${e.examDate} ${e.code}`.toLowerCase();
            return hay.includes(q);
        });
    };

    const groupExams = (filtered: ExamHistoryItem[]) => {
        const groups: Record<string, ExamHistoryItem[]> = {};
        filtered.forEach((exam) => {
            const dateStr = exam.examDate || "";
            const yearMonth = dateStr.slice(0, 7); // "2025.09" or "2025-12"
            if (!groups[yearMonth]) groups[yearMonth] = [];
            groups[yearMonth].push(exam);
        });
        const sortedKeys = Object.keys(groups).sort((a, b) => b.localeCompare(a));
        return sortedKeys.map(key => ({
            yearMonth: key,
            items: groups[key]
        }));
    };

    return { exams, loading, error, filterExams, groupExams };
};

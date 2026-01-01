package com.dankook.mlpa_gradi.mapper;

import com.dankook.mlpa_gradi.dto.ExamDto;
import com.dankook.mlpa_gradi.entity.Exam;

public class ExamMapper {

    public static ExamDto toDto(Exam exam) {
        ExamDto dto = new ExamDto();
        dto.setExamId(exam.getExamId());
        dto.setExamName(exam.getExamName());
        dto.setExamCode(exam.getExamCode()); // ✅ 시험 코드 매핑
        dto.setExamDate(exam.getExamDate());
        dto.setCreatedAt(exam.getCreatedAt());
        dto.setUpdatedAt(exam.getUpdatedAt());

        dto.setQuestions(
                exam.getQuestions().stream()
                        .map(QuestionMapper::toDto)
                        .toList());

        return dto;
    }
}

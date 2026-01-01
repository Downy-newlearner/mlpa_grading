package com.dankook.mlpa_gradi.mapper;

import com.dankook.mlpa_gradi.dto.StudentAnswerDto;
import com.dankook.mlpa_gradi.entity.StudentAnswer;

public class StudentAnswerMapper {

    public static StudentAnswerDto toDto(StudentAnswer a) {
        StudentAnswerDto dto = new StudentAnswerDto();
        dto.setStudentAnswerId(a.getStudentAnswerId());
        dto.setQuestionNumber(a.getQuestionNumber());
        dto.setSubQuestionNumber(a.getSubQuestionNumber());
        dto.setStudentAnswer(a.getStudentAnswer());
        dto.setAnswerCount(a.getAnswerCount());
        dto.setConfidence(a.getConfidence());
        dto.setCorrect(a.isCorrect());
        dto.setScore(a.getScore());
        return dto;
    }
}

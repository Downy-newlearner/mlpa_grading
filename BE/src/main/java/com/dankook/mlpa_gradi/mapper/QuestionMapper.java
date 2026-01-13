package com.dankook.mlpa_gradi.mapper;

import com.dankook.mlpa_gradi.dto.QuestionDto;
import com.dankook.mlpa_gradi.entity.Question;

public class QuestionMapper {

    public static QuestionDto toDto(Question q) {
        QuestionDto dto = new QuestionDto();
        dto.setQuestionId(q.getQuestionId());
        if (q.getExam() != null) {
            dto.setExamCode(q.getExam().getExamCode());
        }
        dto.setQuestionNumber(q.getQuestionNumber());
        dto.setQuestionType(q.getQuestionType());
        dto.setSubQuestionNumber(q.getSubQuestionNumber());
        dto.setAnswer(q.getAnswer());
        dto.setAnswerCount(q.getAnswerCount());
        dto.setPoint(q.getPoint());
        return dto;
    }

    public static Question toEntity(QuestionDto dto) {
        Question q = new Question();
        q.setQuestionNumber(dto.getQuestionNumber());
        q.setQuestionType(dto.getQuestionType());
        q.setSubQuestionNumber(dto.getSubQuestionNumber());
        q.setAnswer(dto.getAnswer());
        q.setAnswerCount(dto.getAnswerCount());
        q.setPoint(dto.getPoint());
        return q;
    }
}

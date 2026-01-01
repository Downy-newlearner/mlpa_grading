package com.dankook.mlpa_gradi.mapper;

import com.dankook.mlpa_gradi.dto.QuestionDto;
import com.dankook.mlpa_gradi.entity.Question;

public class QuestionMapper {

    public static QuestionDto toDto(Question q) {
        QuestionDto dto = new QuestionDto();
        dto.setQuestionId(q.getQuestionId());
        dto.setQuestionNumber(q.getQuestionNumber());
        dto.setQuestionType(q.getQuestionType());
        dto.setSubQuestionNumber(q.getSubQuestionNumber());
        dto.setAnswer(q.getAnswer());
        dto.setAnswerCount(q.getAnswerCount());
        dto.setPoint(q.getPoint());
        return dto;
    }
}

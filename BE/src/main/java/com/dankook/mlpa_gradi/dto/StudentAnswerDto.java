package com.dankook.mlpa_gradi.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class StudentAnswerDto {

    private Long studentAnswerId;

    private int questionNumber;
    private int subQuestionNumber;
    private String studentAnswer;

    private int answerCount;
    private float confidence;
    private boolean correct;
    private float score;
    private Long studentId;
}

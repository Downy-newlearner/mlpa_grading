package com.dankook.mlpa_gradi.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class QuestionDto {

    private Long questionId;
    private int questionNumber;
    private String questionType;
    private Integer subQuestionNumber;

    private String answer;
    private int answerCount;
    private float point;
}

//
//    // ✅ 외래 키 설정
//    @ManyToOne(fetch = FetchType.LAZY)
//    @JoinColumn(name = "student_response_id")  // 이 컬럼이 foreign key가 됨
//    private StudentResponse studentResponse;




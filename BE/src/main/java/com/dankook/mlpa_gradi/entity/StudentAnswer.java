package com.dankook.mlpa_gradi.entity;


import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@Entity
public class StudentAnswer {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long studentAnswerId;  // 데이터베이스에서 식별할 ID
    private int questionNumber;
    private int subQuestionNumber;
    private String studentAnswer;
    private int answerCount;
    private float confidence;
    private boolean isCorrect;
    private float score;
    private String examCode;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "student_id") // DB 컬럼명
    private Student student;
}
package com.dankook.mlpa_gradi.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class ExamDto {

    private Long examId;
    private String examName;
    private String examCode; // ✅ 시험 코드
    private LocalDateTime examDate;

    // 시험에 포함된 문제 목록
    private List<QuestionDto> questions;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
}

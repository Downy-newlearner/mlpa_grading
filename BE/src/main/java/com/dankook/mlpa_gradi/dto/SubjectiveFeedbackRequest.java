package com.dankook.mlpa_gradi.dto;

import lombok.Data;
import java.util.List;

@Data
public class SubjectiveFeedbackRequest {
    private String examCode;
    private List<Evaluation> evaluations;

    @Data
    public static class Evaluation {
        private String questionNumber; // "10", "11-1" 등
        private String score;
        private String comment;
    }
}

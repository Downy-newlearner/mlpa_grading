package com.dankook.mlpa_gradi.dto;

import lombok.Data;
import java.util.List;

@Data
public class FeedbackRequest {
    private String examCode;
    private List<FeedbackImage> images;

    @Data
    public static class FeedbackImage {
        private String fileName;
        @com.fasterxml.jackson.annotation.JsonProperty("student_id")
        private String studentId;
    }
}

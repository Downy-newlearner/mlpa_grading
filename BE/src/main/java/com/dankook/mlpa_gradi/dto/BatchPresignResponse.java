package com.dankook.mlpa_gradi.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class BatchPresignResponse {
    private String examCode;
    private List<PresignedUrl> urls;

    public BatchPresignResponse(String examCode, List<PresignedUrl> urls) {
        this.examCode = examCode;
        this.urls = urls;
    }

    @Getter
    @AllArgsConstructor
    public static class PresignedUrl {
        private int index;
        private String filename;
        private String url;
    }
}

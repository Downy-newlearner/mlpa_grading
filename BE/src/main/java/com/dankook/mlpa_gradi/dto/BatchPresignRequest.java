package com.dankook.mlpa_gradi.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
public class BatchPresignRequest {
    private String examCode;
    private int total; // 전체 학생/페이지 수
    // studentId 제거 (처음에 없음)
    private List<ImageInfo> images;

    @Getter
    @Setter
    @NoArgsConstructor
    public static class ImageInfo {
        private int index;
        private String contentType; // image/png or image/jpeg
        private String filename; // 파일명
    }
}

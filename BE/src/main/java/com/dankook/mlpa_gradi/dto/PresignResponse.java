package com.dankook.mlpa_gradi.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public class PresignResponse {
    private String examCode;
    private Long studentId;
    private Integer totalIndex;
    private Integer index;
    private String url; // presigned PUT url
}

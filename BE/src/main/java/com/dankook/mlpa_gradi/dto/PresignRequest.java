package com.dankook.mlpa_gradi.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class PresignRequest {

    private String examCode;
    private Long studentId;
    private int totalIndex;
    private int index;

    private String contentType; // image/png or image/jpeg
}

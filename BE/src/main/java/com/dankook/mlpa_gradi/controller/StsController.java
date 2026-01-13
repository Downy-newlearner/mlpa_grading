package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.service.StsService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/sts")
@RequiredArgsConstructor
public class StsController {

    private final StsService stsService;

    @GetMapping("/token")
    public Map<String, String> getStsToken(
            @RequestParam("examCode") String examCode,
            @RequestParam("studentId") String studentId) {

        return stsService.getTemporaryCredentials(examCode, studentId);
    }

    @GetMapping("/ai-token")
    public Map<String, String> getAiStsToken() {
        return stsService.getAiServerCredentials();
    }
}

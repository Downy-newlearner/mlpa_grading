package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.FeedbackRequest;
import com.dankook.mlpa_gradi.service.FeedbackService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/feedback")
@RequiredArgsConstructor
public class FeedbackController {

    private final FeedbackService feedbackService;
    private final com.dankook.mlpa_gradi.repository.memory.InMemoryReportRepository inMemoryReportRepository;

    @PostMapping
    public ResponseEntity<String> receiveFeedback(@RequestBody FeedbackRequest request) {
        feedbackService.sendFeedback(request);
        // 제출 완료 후 메모리 캐시 비우기 (새로고침 시 중복 제거 등 관리 목적)
        inMemoryReportRepository.clear(request.getExamCode());
        return ResponseEntity.ok("Feedback received and forwarded.");
    }
}

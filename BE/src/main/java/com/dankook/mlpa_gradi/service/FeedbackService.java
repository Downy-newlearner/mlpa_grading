package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.FeedbackRequest;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

@Service
@RequiredArgsConstructor
@Slf4j
public class FeedbackService {

    private final WebClient aiWebClient;

    public void sendFeedback(FeedbackRequest request) {
        log.info("[FEEDBACK] Forwarding to AI Server /fallback/...");
        log.info("[PAYLOAD] examCode={}, images={}", request.getExamCode(), request.getImages().size());

        try {
            String response = aiWebClient.post()
                    .uri("/fallback/")
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            log.info("[SUCCESS] AI Server response: {}", response);
        } catch (Exception e) {
            log.error("[ERROR] Failed to forward feedback to AI Server: {}", e.getMessage());
            throw new RuntimeException("AI Server communication failed: " + e.getMessage(), e);
        }
    }
}

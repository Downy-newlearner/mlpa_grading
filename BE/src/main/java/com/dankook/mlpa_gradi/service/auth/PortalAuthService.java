package com.dankook.mlpa_gradi.service.auth;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class PortalAuthService {

    private final ObjectMapper objectMapper;

    public boolean verifyStudentId(String studentId) {
        WebClient webClient = WebClient.builder()
                .baseUrl("https://portal.dankook.ac.kr")
                .defaultHeader("Origin", "https://portal.dankook.ac.kr")
                .defaultHeader("Referer", "https://portal.dankook.ac.kr/proc/Login.eps")
                .build();

        MultiValueMap<String, String> formData = new LinkedMultiValueMap<>();
        formData.add("user_id_cert", studentId);
        formData.add("user_id_p", "");
        formData.add("user_id_e", "");
        formData.add("reset_type", "niceCert");

        try {
            String response = webClient.post()
                    .uri("/dku/ConfirmUserId.eps")
                    .contentType(MediaType.APPLICATION_FORM_URLENCODED)
                    .bodyValue(formData)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block(); // Blocking for simplicity in this synchronous flow, or change to return Mono

            log.info("Portal Auth Response for {}: {}", studentId, response);

            if (response == null)
                return false;

            JsonNode jsonNode = objectMapper.readTree(response);
            return "Y".equalsIgnoreCase(jsonNode.path("resultYn").asText());

        } catch (Exception e) {
            log.error("Failed to verify student ID: {}", studentId, e);
            return false;
        }
    }
}

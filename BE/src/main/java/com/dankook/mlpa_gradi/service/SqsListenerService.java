package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.controller.StorageController;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.DeleteMessageRequest;
import software.amazon.awssdk.services.sqs.model.Message;
import software.amazon.awssdk.services.sqs.model.ReceiveMessageRequest;

import java.util.List;
import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class SqsListenerService {

    private final SqsClient sqsClient;
    private final SseService sseService;
    private final S3PresignService s3PresignService;
    private final ObjectMapper objectMapper;
    private final com.dankook.mlpa_gradi.repository.memory.InMemoryReportRepository inMemoryReportRepository;

    private int failureCount = 0;
    private static final int MAX_FAILURES = 10;

    @Value("${aws.sqs.queue-url}")
    private String queueUrl;

    @PostConstruct
    public void init() {
        log.info("🚀 SqsListenerService initialized with Queue URL: {}", queueUrl);
    }

    @Scheduled(fixedDelay = 1000) // Poll every second
    public void pollMessages() {
        if (failureCount >= MAX_FAILURES) {
            log.error("SQS polling suspended due to {} consecutive failures. Please check configuration.",
                    failureCount);
            return;
        }

        // log.info("Polling SQS messages from URL: {}", queueUrl); // Too noisy
        ReceiveMessageRequest receiveRequest = ReceiveMessageRequest.builder()
                .queueUrl(queueUrl)
                .maxNumberOfMessages(10)
                .waitTimeSeconds(5) // Long polling
                .build();

        try {
            List<Message> messages = sqsClient.receiveMessage(receiveRequest).messages();
            if (!messages.isEmpty()) {
                log.info("Successfully polled {} messages from SQS", messages.size());
            }
            failureCount = 0; // Reset on success

            for (Message message : messages) {
                try {
                    processMessage(message.body());
                    deleteMessage(message.receiptHandle());
                } catch (Exception e) {
                    log.error("Error processing SQS message body: {}. Content: {}", e.getMessage(), message.body());
                }
            }
        } catch (Exception e) {
            failureCount++;
            log.error("CRITICAL: Failed to poll SQS from URL: {}. Failure count: {}/{}",
                    queueUrl, failureCount, MAX_FAILURES);
            log.error("Detailed Exception: ", e);
            if (e.getMessage().contains("QueueDoesNotExist")) {
                log.error("Queue does not exist. Please check the URL and AWS Region.");
            }
        }
    }

    private void processMessage(String body) throws Exception {
        Map<String, Object> event = objectMapper.readValue(body, Map.class);

        String eventType = (String) event.getOrDefault("event_type", "STUDENT_ID_RECOGNITION");

        switch (eventType) {
            case "STUDENT_ID_RECOGNITION":
                handleStudentIdRecognition(event);
                break;
            default:
                log.warn("⚠️ Received unknown event type: {}", eventType);
        }
    }

    private void handleStudentIdRecognition(Map<String, Object> event) throws Exception {
        // Handle snake_case and camelCase for compatibility
        String examCode = (String) event.getOrDefault("examCode", event.get("exam_code"));
        String studentId = (String) event.getOrDefault("studentId", event.get("student_id"));

        // Handle index (can be integer or string)
        Object indexObj = event.getOrDefault("idx", event.get("index"));
        int idx = 0;
        if (indexObj instanceof Integer) {
            idx = (Integer) indexObj;
        } else if (indexObj instanceof String) {
            try {
                idx = Integer.parseInt((String) indexObj);
            } catch (NumberFormatException ignored) {
            }
        }

        // Map to standardized keys for frontend
        event.put("examCode", examCode);
        event.put("idx", idx);

        // ✅ Inject 'status' for Frontend if missing
        if (!event.containsKey("status")) {
            int total = 40; // Default
            Object totalObj = event.getOrDefault("total", 40);
            if (totalObj instanceof Integer) {
                total = (Integer) totalObj;
            } else if (totalObj instanceof String) {
                try {
                    total = Integer.parseInt((String) totalObj);
                } catch (NumberFormatException ignored) {
                }
            }
            // Ensure total is in the event for frontend
            event.put("total", total);

            if (idx >= total && total > 0) {
                event.put("status", "completed");
            } else {
                event.put("status", "processing");
            }
        }

        if (examCode != null) {
            // Case 1: "unknown_id" -> Generate URL from filename and save to memory
            if ("unknown_id".equals(studentId)) {
                String filename = (String) event.get("filename");
                if (filename != null) {
                    // Start with basic presignedUrls if sent, otherwise generate
                    List<String> urls = new java.util.ArrayList<>();

                    // Assume S3 key structure: uploads/{examCode}/{filename}
                    // Note: You might need to adjust the path if strictly defined elsewhere
                    String s3Key = String.format("uploads/%s/%s", examCode, filename);
                    String generatedUrl = s3PresignService.generatePresignedGetUrl(s3Key);

                    if (generatedUrl != null) {
                        urls.add(generatedUrl);
                        inMemoryReportRepository.saveUnknownImages(examCode, urls);
                        log.info("💾 Generated & Saved URL for unknown_id: {} -> {}", filename, generatedUrl);

                        // Add to event for frontend
                        event.put("presignedUrls", urls);
                    }
                }
            }
            // Case 2: Explicit presignedUrls (Backward compatibility)
            else if (event.containsKey("presignedUrls")) {
                Object urlsObj = event.get("presignedUrls");
                if (urlsObj instanceof List) {
                    List<String> urls = (List<String>) urlsObj;
                    inMemoryReportRepository.saveUnknownImages(examCode, urls);
                    log.info("💾 Saved {} unknown images to memory from list", urls.size());
                }
            }

            // Forward event to SSE emitters via SseService
            sseService.sendEvent(examCode, "recognition_update", event);
            log.info("📢 Broadcasted SSE event for exam {}: idx={}, studentId={}", examCode, idx, studentId);
        }
    }

    private void deleteMessage(String receiptHandle) {
        DeleteMessageRequest deleteRequest = DeleteMessageRequest.builder()
                .queueUrl(queueUrl)
                .receiptHandle(receiptHandle)
                .build();
        sqsClient.deleteMessage(deleteRequest);
    }
}

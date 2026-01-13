package com.dankook.mlpa_gradi.service;

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
    private final StudentAnswerService studentAnswerService;

    private int failureCount = 0;
    private static final int MAX_FAILURES = 10;

    @Value("${aws.sqs.id-result-queue-url}")
    private String idResultQueueUrl;

    @Value("${aws.sqs.answer-result-queue-url}")
    private String answerResultQueueUrl;

    @PostConstruct
    public void init() {
        log.info("🚀 SqsListenerService initialized.");
        log.info("📡 ID Result Queue: {}", idResultQueueUrl);
        log.info("📡 Answer Result Queue: {}", answerResultQueueUrl);
    }

    @Scheduled(fixedDelay = 1000)
    public void pollMessages() {
        if (failureCount >= MAX_FAILURES)
            return;

        // Poll ID Result Queue
        pollQueue(idResultQueueUrl);

        // Poll Answer Result Queue
        pollQueue(answerResultQueueUrl);
    }

    private void pollQueue(String queueUrl) {
        if (queueUrl == null || queueUrl.isEmpty())
            return;

        ReceiveMessageRequest receiveRequest = ReceiveMessageRequest.builder()
                .queueUrl(queueUrl)
                .maxNumberOfMessages(10)
                .waitTimeSeconds(2)
                .build();

        try {
            List<Message> messages = sqsClient.receiveMessage(receiveRequest).messages();
            for (Message message : messages) {
                try {
                    processMessage(message.body());
                    deleteMessage(queueUrl, message.receiptHandle());
                } catch (Exception e) {
                    log.error("Error processing SQS message from {}: {}", queueUrl, e.getMessage());
                }
            }
            failureCount = 0;
        } catch (Exception e) {
            failureCount++;
            log.error("Failed to poll SQS [{}]: {}", queueUrl, e.getMessage());
        }
    }

    private void deleteMessage(String queueUrl, String receiptHandle) {
        DeleteMessageRequest deleteRequest = DeleteMessageRequest.builder()
                .queueUrl(queueUrl)
                .receiptHandle(receiptHandle)
                .build();
        sqsClient.deleteMessage(deleteRequest);
    }

    private void processMessage(String body) throws Exception {
        Map<String, Object> event = objectMapper.readValue(body, Map.class);

        String eventType = (String) event.getOrDefault("event_type",
                event.getOrDefault("eventType", "STUDENT_ID_RECOGNITION"));

        switch (eventType) {
            case "STUDENT_ID_RECOGNITION":
            case "QUESTION_RECOGNITION":
                handleRecognitionProgress(event);
                break;
            case "ATTENDANCE_UPLOAD":
                log.info("📂 Attendance file upload event received. ExamCode: {}, URL: {}", event.get("examCode"),
                        event.get("downloadUrl"));
                String attExamCode = (String) event.get("examCode");
                if (attExamCode != null) {
                    sseService.sendEvent(attExamCode, "attendance_uploaded", event);
                }
                break;
            case "ERROR":
                log.error("🚨 Error event received from AI Server: {}", event.get("message"));
                String errorCode = (String) event.get("examCode");
                sseService.sendEvent(errorCode, "error_occurred", event);
                break;
            default:
                log.warn("[WARN] Received unknown event type: {}", eventType);
        }
    }

    private void handleRecognitionProgress(Map<String, Object> event) throws Exception {
        log.info("[SQS] Raw Event: {}", event);

        // Handle snake_case and camelCase for compatibility
        String rawExamCode = (String) event.getOrDefault("examCode", event.get("exam_code"));
        String examCode = rawExamCode != null ? rawExamCode.trim().toUpperCase() : null;
        String studentId = (String) event.getOrDefault("studentId", event.get("student_id"));
        String filename = (String) event.getOrDefault("filename", event.get("fileName"));

        // Get or create session
        SseService.SessionInfo session = sseService.getSession(examCode);
        if (session == null) {
            log.warn("[WARN] No session found for examCode: {}", examCode);
            return;
        }

        // ==== DEDUPLICATION: Check if this file was already processed ====
        if (filename != null && !filename.isEmpty()) {
            if (session.processedFiles.contains(filename)) {
                log.info("[SKIP] Duplicate file ignored: {} (already processed)", filename);
                return; // Skip duplicate
            }
            session.processedFiles.add(filename);
        }

        // Use the count of unique processed files as the actual progress
        int currentProgress = session.processedFiles.size();

        // Total 처리
        int total = 0;
        Object totalObj = event.getOrDefault("total", null);
        if (totalObj != null) {
            try {
                total = (int) Double.parseDouble(totalObj.toString());
            } catch (Exception ignored) {
            }
        }

        if (total <= 0) {
            total = session.total;
        } else {
            session.total = total; // Update session total if provided
        }

        if (total > 0 && currentProgress > total) {
            log.warn("[WARN] Progress ({}) exceeded total ({}). Clamping.", currentProgress, total);
            currentProgress = total;
        }

        // Status 결정
        String status = (String) event.getOrDefault("status", "processing");
        if (total > 0 && currentProgress >= total) {
            status = "completed";
        }

        // Update session
        session.index = currentProgress;
        session.status = status;
        session.lastUpdateTime = System.currentTimeMillis();

        // 데이터 표준화
        event.put("examCode", examCode);
        event.put("index", currentProgress);
        event.put("total", total);
        event.put("status", status);

        log.info("[SYNC] {} -> {}/{} ({}) [unique files: {}]", examCode, currentProgress, total, status,
                session.processedFiles.size());

        if (examCode != null) {
            // Case 1: "unknown_id" -> Generate URL from filename and save to memory
            if ("unknown_id".equals(studentId)) {
                if (filename != null) {
                    String s3Key = String.format("header/%s/unknown_id/%s", examCode, filename);
                    String generatedUrl = s3PresignService.generatePresignedGetUrl(s3Key);

                    if (generatedUrl != null) {
                        List<String> urls = new java.util.ArrayList<>();
                        urls.add(generatedUrl);
                        inMemoryReportRepository.saveUnknownImages(examCode, urls);
                        log.info("[SAVED] unknown_id: {} -> {}", filename, generatedUrl);
                        event.put("presignedUrls", urls);
                    }
                }
            } else if (event.containsKey("presignedUrls")) {
                Object urlsObj = event.get("presignedUrls");
                if (urlsObj instanceof List) {
                    List<String> urls = (List<String>) urlsObj;
                    inMemoryReportRepository.saveUnknownImages(examCode, urls);
                }
            }

            // Forward event to SSE emitters via SseService
            log.info("[SSE] Broadcasting: {} - {}/{}", examCode, currentProgress, total);
            sseService.updateProgress(examCode, currentProgress, total);
            sseService.sendEvent(examCode, "recognition_update", event);

            // Save answers to DB if present
            try {
                if (event.containsKey("answers")) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> answers = (List<Map<String, Object>>) event.get("answers");
                    if (studentId != null && answers != null && !answers.isEmpty()) {
                        studentAnswerService.saveRecognitionResults(examCode, studentId, answers);
                        log.info("[DB] Saved {} answers for student {}", answers.size(), studentId);
                    }
                }
            } catch (Exception e) {
                log.error("[DB] Failed to save answers for student {}: {}", studentId, e.getMessage());
            }
        }
    }
}

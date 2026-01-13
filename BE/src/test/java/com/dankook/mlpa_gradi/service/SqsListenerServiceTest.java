package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.repository.memory.InMemoryReportRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;
import software.amazon.awssdk.services.sqs.SqsClient;
import software.amazon.awssdk.services.sqs.model.*;

import java.util.List;
import java.util.Map;
import java.util.HashSet;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class SqsListenerServiceTest {

    @Mock
    private SqsClient sqsClient;
    @Mock
    private SseService sseService;
    @Mock
    private S3PresignService s3PresignService;
    @Mock
    private ObjectMapper objectMapper;
    @Mock
    private InMemoryReportRepository inMemoryReportRepository;
    @Mock
    private StudentAnswerService studentAnswerService;

    @InjectMocks
    private SqsListenerService sqsListenerService;

    @BeforeEach
    void setUp() {
        ReflectionTestUtils.setField(sqsListenerService, "queueUrl", "test-queue-url");
    }

    @Test
    @DisplayName("멱등성 테스트: 동일한 파일명의 메시지가 중복 수신될 경우 한 번만 처리되어야 한다")
    void verifyIdempotency() throws Exception {
        // Given
        String duplicateBody = """
                    {
                        "event_type": "STUDENT_ID_RECOGNITION",
                        "examCode": "TESTCODE",
                        "studentId": "12345",
                        "filename": "duplicate_file.jpg",
                        "status": "processing"
                    }
                """;

        // Mock ObjectMapper to return a Map
        Map<String, Object> eventMap = new java.util.HashMap<>();
        eventMap.put("event_type", "STUDENT_ID_RECOGNITION");
        eventMap.put("examCode", "TESTCODE");
        eventMap.put("studentId", "12345");
        eventMap.put("filename", "duplicate_file.jpg");
        eventMap.put("status", "processing");

        // Mock return for both calls
        lenient().when(objectMapper.readValue(eq(duplicateBody), eq(Map.class))).thenReturn(eventMap);

        // Mock SQS Messages (2 identical messages)
        Message msg1 = Message.builder().body(duplicateBody).receiptHandle("h1").build();
        Message msg2 = Message.builder().body(duplicateBody).receiptHandle("h2").build();

        ReceiveMessageResponse response = ReceiveMessageResponse.builder()
                .messages(msg1, msg2)
                .build();

        when(sqsClient.receiveMessage(any(ReceiveMessageRequest.class))).thenReturn(response);

        // Mock Session with CORRECT Constructor
        SseService.SessionInfo sessionInfo = new SseService.SessionInfo("TESTCODE", "Test Exam", 100);
        // sessionInfo.processedFiles is initialized in the class definition

        when(sseService.getSession("TESTCODE")).thenReturn(sessionInfo);

        // When
        sqsListenerService.pollMessages();

        // Then
        // 1. processMessage called twice (logic inside handles deduplication)
        // 2. handleRecognitionProgress -> logic checks "processedFiles"

        // Verify that updateProgress (or sendEvent) was only called ONCE
        // Because the second time, it should hit the "processedFiles.contains" check
        // and return.
        verify(sseService, times(1)).sendEvent(eq("TESTCODE"), eq("recognition_update"), any());

        // Verify deleteMessage was called for both (since we consume them)
        verify(sqsClient, times(2)).deleteMessage(any(DeleteMessageRequest.class));
    }
}

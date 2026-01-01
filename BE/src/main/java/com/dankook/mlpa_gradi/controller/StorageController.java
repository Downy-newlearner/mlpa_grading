package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.BatchPresignRequest;
import com.dankook.mlpa_gradi.dto.BatchPresignResponse;
import com.dankook.mlpa_gradi.dto.PresignRequest;
import com.dankook.mlpa_gradi.dto.PresignResponse;
import com.dankook.mlpa_gradi.service.S3PresignService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/storage")
public class StorageController {

    private final S3PresignService s3PresignService;

    // SSE Emitters 저장소 (examCode -> Emitter)
    private final Map<String, SseEmitter> sseEmitters = new ConcurrentHashMap<>();

    // ✅ SSE 연결 (프론트가 먼저 연결)
    @GetMapping(value = "/sse/connect", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter connectSSE(@RequestParam("examCode") String examCode) {
        SseEmitter emitter = new SseEmitter(60 * 60 * 1000L); // 1시간 타임아웃

        sseEmitters.put(examCode, emitter);

        emitter.onCompletion(() -> sseEmitters.remove(examCode));
        emitter.onTimeout(() -> sseEmitters.remove(examCode));
        emitter.onError((e) -> sseEmitters.remove(examCode));

        // 연결 확인 메시지
        try {
            emitter.send(SseEmitter.event()
                    .name("connected")
                    .data(Map.of("examCode", examCode, "message", "SSE connected")));
        } catch (IOException e) {
            emitter.completeWithError(e);
        }

        return emitter;
    }

    // ✅ 배치 이미지 Presigned URL 생성 (examCode 기반)
    @PostMapping("/presigned-urls/batch")
    public BatchPresignResponse createBatchPresignedUrls(@RequestBody BatchPresignRequest request) {
        return s3PresignService.createBatchPutUrls(request);
    }

    // ✅ 단일 이미지 Presigned URL 생성
    @PostMapping("/presigned-url")
    public PresignResponse createPresignedUrl(@RequestBody PresignRequest request) {
        return s3PresignService.createPutUrl(request);
    }

    @PostMapping(value = "/attendance", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<Map<String, String>> uploadAttendance(
            @RequestPart("file") MultipartFile file,
            @RequestParam("examCode") String examCode) {
        String s3Url = s3PresignService.uploadAttendance(file, examCode);
        return ResponseEntity.ok(Map.of(
                "message", "Attendance uploaded successfully",
                "s3Url", s3Url));
    }

    // ✅ 출석부 다운로드용 Presigned URL 생성
    @GetMapping("/attendance/download-url")
    public ResponseEntity<Map<String, String>> getAttendanceDownloadUrl(@RequestParam("examCode") String examCode) {
        String downloadUrl = s3PresignService.getAttendanceDownloadUrl(examCode);
        return ResponseEntity.ok(Map.of("url", downloadUrl));
    }

    // ✅ SSE로 이벤트 전송 (채점 진행 등)
    public void sendSSEEvent(String examCode, String eventName, Object data) {
        SseEmitter emitter = sseEmitters.get(examCode);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event().name(eventName).data(data));
            } catch (IOException e) {
                sseEmitters.remove(examCode);
            }
        }
    }
}

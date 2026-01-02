package com.dankook.mlpa_gradi.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
@RequiredArgsConstructor
public class SseService {

    // SSE Emitters repository (examCode -> Emitter)
    private final Map<String, SseEmitter> sseEmitters = new ConcurrentHashMap<>();

    public SseEmitter connect(String examCode) {
        SseEmitter emitter = new SseEmitter(60 * 60 * 1000L); // 1 hour timeout

        sseEmitters.put(examCode, emitter);

        emitter.onCompletion(() -> sseEmitters.remove(examCode));
        emitter.onTimeout(() -> sseEmitters.remove(examCode));
        emitter.onError((e) -> sseEmitters.remove(examCode));

        // Send connection confirmation
        try {
            emitter.send(SseEmitter.event()
                    .name("connected")
                    .data(Map.of("examCode", examCode, "message", "SSE connected")));
        } catch (IOException e) {
            emitter.completeWithError(e);
        }

        log.info("SSE Connected for examCode: {}", examCode);
        return emitter;
    }

    public void sendEvent(String examCode, String eventName, Object data) {
        SseEmitter emitter = sseEmitters.get(examCode);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event().name(eventName).data(data));
            } catch (IOException e) {
                sseEmitters.remove(examCode);
                log.warn("Failed to send SSE event to {}, removing emitter.", examCode);
            }
        }
    }
}

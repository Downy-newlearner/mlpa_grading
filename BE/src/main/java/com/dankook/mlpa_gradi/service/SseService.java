package com.dankook.mlpa_gradi.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
@RequiredArgsConstructor
public class SseService {

    private final ObjectMapper objectMapper;
    private final Map<String, SessionInfo> sessions = new ConcurrentHashMap<>();

    public static class SessionInfo {
        public SseEmitter emitter;
        public final String examCode;
        public String examName = "Unknown";
        public int index = 0;
        public int total = 0;
        public String status = "processing";
        public long lastUpdateTime;
        // Deduplication: Track processed filenames to prevent counting duplicates
        public final Set<String> processedFiles = new HashSet<>();

        public SessionInfo(String examCode, String examName, int total) {
            this.examCode = examCode;
            this.examName = examName;
            this.total = total;
            this.lastUpdateTime = System.currentTimeMillis();
        }
    }

    private SessionInfo getOrCreateSession(String examCode, String examName, int total) {
        String normalizedCode = (examCode != null) ? examCode.trim().toUpperCase() : "UNKNOWN";
        return sessions.compute(normalizedCode, (key, existing) -> {
            if (existing != null) {
                if (total > 0)
                    existing.total = total;
                if (examName != null && !"Unknown".equals(examName))
                    existing.examName = examName;
                existing.lastUpdateTime = System.currentTimeMillis();
                return existing;
            }
            return new SessionInfo(normalizedCode, examName, total);
        });
    }

    public SseEmitter connect(String examCode, String examName, int total) {
        SessionInfo session = getOrCreateSession(examCode, examName, total);

        // Timeout 1 hour
        SseEmitter emitter = new SseEmitter(3600_000L);
        session.emitter = emitter;
        session.lastUpdateTime = System.currentTimeMillis();

        emitter.onCompletion(() -> cleanupEmitter(session, emitter));
        emitter.onTimeout(() -> cleanupEmitter(session, emitter));
        emitter.onError((e) -> cleanupEmitter(session, emitter));

        try {
            // ✅ 버퍼링 방지: 연결 시 충분한 양의 패딩 전송 (4KB 이상 권장)
            String padding = " ".repeat(4096);
            emitter.send(SseEmitter.event().comment("init").comment(padding));

            Map<String, Object> initData = Map.of(
                    "type", "connected",
                    "index", session.index,
                    "total", session.total,
                    "status", session.status);
            sendDirect(emitter, initData);
            log.info("📡 [SseService] SSE Connected & Flushed for {}", session.examCode);
        } catch (IOException e) {
            log.warn("❌ [SseService] Failed to send initial SSE to {}", session.examCode);
        }

        return emitter;
    }

    private void cleanupEmitter(SessionInfo session, SseEmitter emitter) {
        if (session.emitter == emitter) {
            log.info("🔌 [SseService] Emitter disconnected for {}", session.examCode);
            session.emitter = null;
        }
    }

    public void updateProgress(String examCode, int index, int total) {
        SessionInfo s = getOrCreateSession(examCode, null, total);
        s.index = index;
        if (total > 0)
            s.total = total;
        // 보정: index가 total을 넘지 않게 강제 (5/4 방지)
        if (s.total > 0 && s.index > s.total)
            s.index = s.total;

        if (s.total > 0 && s.index >= s.total)
            s.status = "completed";
        s.lastUpdateTime = System.currentTimeMillis();

        log.info("📈 [SseService] Internal Update {}: {}/{} ({})", s.examCode, s.index, s.total, s.status);
    }

    public void sendEvent(String examCode, String eventName, Object data) {
        String code = (examCode != null) ? examCode.trim().toUpperCase() : "";
        SessionInfo s = sessions.get(code);
        if (s != null && s.emitter != null) {
            try {
                Map<String, Object> payload = Map.of("type", eventName, "data", data);
                sendDirect(s.emitter, payload);
                s.lastUpdateTime = System.currentTimeMillis();
                log.info("📤 [SseService] Sent Event: {} to {}", eventName, code);
            } catch (Exception e) {
                log.warn("⚠️ [SseService] Send failed, clearing emitter: {}", code);
                s.emitter = null;
            }
        }
    }

    // ✅ 데이터를 보낼 때 패딩을 섞어서 강제로 버퍼를 비움
    private void sendDirect(SseEmitter emitter, Object data) throws IOException {
        String json = objectMapper.writeValueAsString(data);
        // 데이터 뒤에 공백 패딩을 붙여서 전송 (Next.js/Nginx 등의 버퍼를 강제로 밀어냄)
        emitter.send(SseEmitter.event().data(json).comment(" ".repeat(1024)));
    }

    public SessionInfo getSession(String examCode) {
        return (examCode != null) ? sessions.get(examCode.trim().toUpperCase()) : null;
    }

    public List<Map<String, Object>> getActiveProcesses() {
        List<Map<String, Object>> result = new ArrayList<>();
        sessions.forEach((k, s) -> {
            Map<String, Object> m = new java.util.HashMap<>();
            m.put("examCode", s.examCode);
            m.put("examName", s.examName);
            m.put("index", s.index);
            m.put("total", s.total);
            m.put("status", s.status);
            result.add(m);
        });
        return result;
    }

    public void removeSession(String examCode) {
        String code = (examCode != null) ? examCode.trim().toUpperCase() : "";
        SessionInfo s = sessions.remove(code);
        if (s != null && s.emitter != null) {
            try {
                s.emitter.complete();
            } catch (Exception ignored) {
            }
        }
    }

    @org.springframework.scheduling.annotation.Scheduled(fixedRate = 5000)
    public void heartbeat() {
        long now = System.currentTimeMillis();
        // 10분 무활동 시 세션 제거
        sessions.entrySet().removeIf(entry -> (now - entry.getValue().lastUpdateTime) > 600000);

        sessions.forEach((code, s) -> {
            if (s.emitter != null) {
                try {
                    // 심박수 측정 시에도 패딩을 보내 연결 유지 강제
                    s.emitter.send(SseEmitter.event().comment("heartbeat").comment(" ".repeat(512)));
                } catch (IOException e) {
                    s.emitter = null;
                }
            }
        });
    }
}

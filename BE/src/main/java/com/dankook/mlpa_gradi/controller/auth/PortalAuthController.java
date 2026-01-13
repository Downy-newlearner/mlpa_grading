package com.dankook.mlpa_gradi.controller.auth;

import com.dankook.mlpa_gradi.service.auth.PortalAuthService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
@Slf4j
public class PortalAuthController {

    private final PortalAuthService portalAuthService;

    @PostMapping("/verify-dku")
    public ResponseEntity<Map<String, Object>> verifyDankookId(@RequestBody Map<String, String> request) {
        String studentId = request.get("studentId");
        log.info("Received verification request for studentId: {}", studentId);

        if (studentId == null || studentId.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("success", false, "message", "Student ID is required"));
        }

        boolean isValid = portalAuthService.verifyStudentId(studentId);

        if (isValid) {
            return ResponseEntity.ok(Map.of("success", true, "message", "Verified successfully"));
        } else {
            return ResponseEntity.status(401).body(Map.of("success", false, "message", "Invalid student ID"));
        }
    }
}

package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.EmailRequest;
import com.dankook.mlpa_gradi.dto.VerificationRequest;
import com.dankook.mlpa_gradi.service.EmailVerificationService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/email")
@RequiredArgsConstructor
public class EmailVerificationController {

    private final EmailVerificationService emailService;

    // 1. 인증 코드 발송
    @PostMapping("/verification-code")
    public ResponseEntity<?> sendVerificationCode(@RequestBody EmailRequest req) {
        emailService.sendVerificationCode(req.getStudentId(), req.getEmail());
        return ResponseEntity.ok(Map.of("message", "Verification code sent to " + req.getEmail()));
    }

    // 2. 인증 코드 검증
    @PostMapping("/verify/{studentId}")
    public ResponseEntity<?> verifyCode(@PathVariable String studentId, @RequestBody VerificationRequest req) {
        boolean isVerified = emailService.verifyCode(studentId, req.getCode());

        if (isVerified) {
            return ResponseEntity.ok(Map.of("message", "Verification successful", "verified", true));
        } else {
            return ResponseEntity.badRequest().body(Map.of("message", "Verification failed", "verified", false));
        }
    }
}

package com.dankook.mlpa_gradi.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class EmailVerificationService {

    private final JavaMailSender mailSender;
    private final Map<String, String> verificationStore = new ConcurrentHashMap<>();
    private final Map<String, java.time.LocalDateTime> blackListStore = new ConcurrentHashMap<>();

    @Value("${spring.mail.username}")
    private String fromAddress;

    public void sendVerificationCode(String studentId, String email) {
        String code = createRandomCode();
        verificationStore.put(studentId, code);
        log.info("Generated code for student {}: {}", studentId, code);

        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(fromAddress);
        message.setTo(email);
        message.setSubject("[Gradi] 이메일 인증 코드입니다.");
        message.setText("안녕하세요,\n\n요청하신 인증 코드는 [" + code + "] 입니다.\n\n감사합니다.");

        try {
            mailSender.send(message);
            log.info("Verification email sent to {}", email);
        } catch (Exception e) {
            log.error("Failed to send email to {}", email, e);
            throw new RuntimeException("이메일 발송에 실패했습니다.");
        }
    }

    public boolean verifyCode(String studentId, String inputCode) {
        String storedCode = verificationStore.get(studentId);

        if (storedCode == null) {
            return false;
        }

        if (storedCode.equals(inputCode)) {
            verificationStore.remove(studentId);
            // 인증 성공한 코드를 블랙리스트에 추가 (하루 동안 사용 불가)
            blackListStore.put(inputCode, java.time.LocalDateTime.now());
            log.info("Code {} added to blacklist.", inputCode);
            return true;
        }

        return false;
    }

    private String createRandomCode() {
        Random random = new Random();
        String code;
        String regex = "^\\d{6}$";

        while (true) {
            int num = random.nextInt(1000000); // 0 ~ 999999
            code = String.format("%06d", num);

            // 1. 정규표현식 검증 (6자리 숫자)
            if (code.matches(regex)) {
                // 2. 블랙리스트 확인
                if (isNotBlackListed(code)) {
                    break;
                }
            }
        }
        return code;
    }

    private boolean isNotBlackListed(String code) {
        java.time.LocalDateTime blackListTime = blackListStore.get(code);
        if (blackListTime == null) {
            return true;
        }

        // 하루(24시간)가 지났는지 확인
        if (blackListTime.isBefore(java.time.LocalDateTime.now().minusDays(1))) {
            blackListStore.remove(code); // 만료된 블랙리스트 제거
            return true;
        }

        return false;
    }
}

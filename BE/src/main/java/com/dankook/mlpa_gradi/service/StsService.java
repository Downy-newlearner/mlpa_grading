package com.dankook.mlpa_gradi.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import software.amazon.awssdk.services.sts.StsClient;
import software.amazon.awssdk.services.sts.model.GetFederationTokenRequest;
import software.amazon.awssdk.services.sts.model.GetFederationTokenResponse;

import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class StsService {

    private final StsClient stsClient;

    @Value("${aws.s3.bucket}")
    private String bucket;

    public Map<String, String> getTemporaryCredentials(String examCode, String studentId) {
        // 1. Define allowed paths
        String unknownAnswerResource = String.format("arn:aws:s3:::%s/header/%s/%s/unknown_answer/*", bucket, examCode,
                studentId);
        String knownAnswerResource = String.format("arn:aws:s3:::%s/header/%s/%s/known_answer/*", bucket, examCode,
                studentId);

        // 2. Create Policy JSON
        String policy = String.format("""
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": ["s3:PutObject", "s3:GetObject"],
                            "Resource": [
                                "%s",
                                "%s"
                            ]
                        }
                    ]
                }
                """, unknownAnswerResource, knownAnswerResource);

        // 3. Request STS Token (GetFederationToken)
        try {
            GetFederationTokenRequest request = GetFederationTokenRequest.builder()
                    .name("User-" + studentId)
                    .policy(policy)
                    .durationSeconds(3600) // 1 Hour
                    .build();

            GetFederationTokenResponse response = stsClient.getFederationToken(request);

            log.info("✅ Generated STS token for exam: {}, student: {}", examCode, studentId);

            return Map.of(
                    "accessKey", response.credentials().accessKeyId(),
                    "secretKey", response.credentials().secretAccessKey(),
                    "sessionToken", response.credentials().sessionToken());

        } catch (Exception e) {
            log.error("❌ Failed to generate STS token", e);
            throw new RuntimeException("Failed to generate AWS temporary credentials", e);
        }
    }
}

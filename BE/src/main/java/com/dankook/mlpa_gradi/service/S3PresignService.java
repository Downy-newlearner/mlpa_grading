package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.PresignRequest;
import com.dankook.mlpa_gradi.dto.PresignResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectsRequest;
import software.amazon.awssdk.services.s3.model.ObjectIdentifier;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Request;
import software.amazon.awssdk.services.s3.model.ListObjectsV2Response;
import software.amazon.awssdk.services.s3.model.CORSRule;
import software.amazon.awssdk.services.s3.model.CORSConfiguration;
import software.amazon.awssdk.services.s3.model.PutBucketCorsRequest;
import software.amazon.awssdk.services.s3.presigner.S3Presigner;
import software.amazon.awssdk.services.s3.presigner.model.PutObjectPresignRequest;
import software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest;
import jakarta.annotation.PostConstruct;

import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.NoSuchElementException;

@Service
@RequiredArgsConstructor
public class S3PresignService {

        private final S3Presigner presigner;
        private final S3Client s3Client;

        @Value("${aws.s3.bucket}")
        private String bucket;

        @Value("${aws.s3.prefix:uploads}")
        private String prefix;

        @Value("${app.frontend.url}")
        private String frontendUrl;

        /**
         * ✅ 서버 시작 시 S3 버킷의 CORS 설정을 자동으로 업데이트합니다.
         * 프론트엔드에서 Presigned URL로 직접 업로드할 때 발생하는 CORS 오류를 방지합니다.
         */
        @PostConstruct
        public void initBucketCors() {
                try {
                        CORSRule corsRule = CORSRule.builder()
                                        .allowedHeaders("*")
                                        .allowedMethods("GET", "PUT", "POST", "DELETE", "HEAD")
                                        .allowedOrigins(frontendUrl) // 환경변수에서 가져온 주소
                                        .exposeHeaders("ETag")
                                        .maxAgeSeconds(3000)
                                        .build();

                        CORSConfiguration corsConfiguration = CORSConfiguration.builder()
                                        .corsRules(List.of(corsRule))
                                        .build();

                        PutBucketCorsRequest putBucketCorsRequest = PutBucketCorsRequest.builder()
                                        .bucket(bucket)
                                        .corsConfiguration(corsConfiguration)
                                        .build();

                        s3Client.putBucketCors(putBucketCorsRequest);
                        System.out.println("✅ S3 CORS configuration updated for bucket: " + bucket);
                } catch (Exception e) {
                        System.err.println("❌ Failed to set S3 CORS: " + e.getMessage());
                        // 애플리케이션 실행을 멈추지는 않지만 경고를 출력합니다.
                }
        }

        // ✅ 이미지 Presigned URL 생성 (기존 기능)
        public PresignResponse createPutUrl(PresignRequest req) {
                String contentType = req.getContentType();

                if (contentType == null ||
                                !(contentType.equals("image/png")
                                                || contentType.equals("image/jpg")
                                                || contentType.equals("image/jpeg"))) {
                        throw new IllegalArgumentException("Only PNG/JPG/JPEG allowed");
                }

                String ext = contentType.equals("image/png") ? "png" : "jpg";

                // S3 Key 규칙: uploads/{examCode}/{studentId}/{index}.{ext}
                String key = String.format("%s/%s/%d/%d.%s",
                                prefix,
                                req.getExamCode(),
                                req.getStudentId(),
                                req.getIndex(),
                                ext);

                PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                                .bucket(bucket)
                                .key(key)
                                .contentType(contentType.equals("image/jpg") ? "image/jpeg" : contentType)
                                .build();

                PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
                                .signatureDuration(Duration.ofMinutes(10))
                                .putObjectRequest(putObjectRequest)
                                .build();

                String url = presigner.presignPutObject(presignRequest)
                                .url()
                                .toString();

                return new PresignResponse(
                                req.getExamCode(),
                                req.getStudentId(),
                                req.getTotalIndex(),
                                req.getIndex(),
                                url);
        }

        // ✅ 출석부 파일 S3 직접 업로드
        public String uploadAttendance(MultipartFile file, String examCode) {
                String originalFilename = file.getOriginalFilename();
                String ext = "";
                if (originalFilename != null && originalFilename.contains(".")) {
                        ext = originalFilename.substring(originalFilename.lastIndexOf("."));
                }

                // S3 Key 규칙: attendance/{examCode}/attendance{ext}
                String key = String.format("attendance/%s/attendance%s", examCode, ext);

                try {
                        PutObjectRequest putRequest = PutObjectRequest.builder()
                                        .bucket(bucket)
                                        .key(key)
                                        .contentType(file.getContentType())
                                        .build();

                        s3Client.putObject(putRequest, RequestBody.fromBytes(file.getBytes()));

                        return String.format("s3://%s/%s", bucket, key);
                } catch (IOException e) {
                        throw new RuntimeException("Failed to upload attendance file to S3", e);
                }
        }

        // ✅ 배치 이미지 Presigned URL 생성
        public com.dankook.mlpa_gradi.dto.BatchPresignResponse createBatchPutUrls(
                        com.dankook.mlpa_gradi.dto.BatchPresignRequest req) {
                java.util.List<com.dankook.mlpa_gradi.dto.BatchPresignResponse.PresignedUrl> urls = new java.util.ArrayList<>();

                for (com.dankook.mlpa_gradi.dto.BatchPresignRequest.ImageInfo img : req.getImages()) {
                        String contentType = img.getContentType();
                        if (contentType == null ||
                                        !(contentType.equals("image/png")
                                                        || contentType.equals("image/jpg")
                                                        || contentType.equals("image/jpeg"))) {
                                throw new IllegalArgumentException("Only PNG/JPG/JPEG allowed");
                        }

                        String ext = contentType.equals("image/png") ? "png" : "jpg";
                        String filename = img.getFilename() != null ? img.getFilename() : (img.getIndex() + "." + ext);

                        // S3 Key: uploads/{examCode}/{index}_{filename}
                        String key = String.format("%s/%s/%d_%s",
                                        prefix,
                                        req.getExamCode(),
                                        img.getIndex(),
                                        filename);

                        PutObjectRequest putObjectRequest = PutObjectRequest.builder()
                                        .bucket(bucket)
                                        .key(key)
                                        .contentType(contentType.equals("image/jpg") ? "image/jpeg" : contentType)
                                        .metadata(java.util.Map.of(
                                                        "total", String.valueOf(req.getTotal()),
                                                        "idx", String.valueOf(img.getIndex())))
                                        .build();

                        PutObjectPresignRequest presignRequest = PutObjectPresignRequest.builder()
                                        .signatureDuration(Duration.ofMinutes(10))
                                        .putObjectRequest(putObjectRequest)
                                        .build();

                        String url = presigner.presignPutObject(presignRequest).url().toString();
                        urls.add(new com.dankook.mlpa_gradi.dto.BatchPresignResponse.PresignedUrl(img.getIndex(),
                                        filename, url));
                }

                return new com.dankook.mlpa_gradi.dto.BatchPresignResponse(req.getExamCode(), urls);
        }

        /**
         * ✅ 출석부 다운로드 용 Presigned URL 생성 (GET)
         */
        public String getAttendanceDownloadUrl(String examCode) {
                // attendance/{examCode}/ 내의 파일을 찾음
                String prefix = String.format("attendance/%s/attendance", examCode);

                ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                                .bucket(bucket)
                                .prefix(prefix)
                                .maxKeys(1)
                                .build();

                ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);

                if (listResponse.contents().isEmpty()) {
                        throw new NoSuchElementException("Attendance file not found for exam: " + examCode);
                }

                String key = listResponse.contents().get(0).key();

                GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                                .bucket(bucket)
                                .key(key)
                                .build();

                GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
                                .signatureDuration(Duration.ofMinutes(10))
                                .getObjectRequest(getObjectRequest)
                                .build();

                return presigner.presignGetObject(presignRequest).url().toString();
        }

        /**
         * ✅ 특정 시험의 모든 S3 데이터 삭제 (이미지 + 출석부)
         */
        public void deleteByExamCode(String examCode) {
                // 1. 이미지 삭제 (uploads/{examCode}/)
                deleteObjectsWithPrefix(String.format("%s/%s/", prefix, examCode));

                // 2. 출석부 삭제 (attendance/{examCode}/)
                deleteObjectsWithPrefix(String.format("attendance/%s/", examCode));
        }

        private void deleteObjectsWithPrefix(String prefix) {
                try {
                        ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                                        .bucket(bucket)
                                        .prefix(prefix)
                                        .build();

                        ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);

                        if (!listResponse.contents().isEmpty()) {
                                List<ObjectIdentifier> identifiers = listResponse.contents().stream()
                                                .map(obj -> ObjectIdentifier.builder().key(obj.key()).build())
                                                .toList();

                                DeleteObjectsRequest deleteRequest = DeleteObjectsRequest.builder()
                                                .bucket(bucket)
                                                .delete(d -> d.objects(identifiers))
                                                .build();

                                s3Client.deleteObjects(deleteRequest);
                                System.out.println("✅ Deleted S3 objects with prefix: " + prefix);
                        }
                } catch (Exception e) {
                        System.err.println(
                                        "❌ Failed to delete S3 objects with prefix " + prefix + ": " + e.getMessage());
                }
        }

        /**
         * ✅ 특정 학생의 채점 이미지 Presigned URL 목록 조회
         */
        public java.util.List<String> getStudentImageUrls(String examCode, String studentId) {
                String folderPrefix = String.format("%s/%s/", prefix, examCode);

                ListObjectsV2Request listRequest = ListObjectsV2Request.builder()
                                .bucket(bucket)
                                .prefix(folderPrefix)
                                .build();

                ListObjectsV2Response listResponse = s3Client.listObjectsV2(listRequest);

                return listResponse.contents().stream()
                                .filter(obj -> obj.key().contains(studentId)) // 파일명에 학번이 포함된 경우 필터링
                                .map(obj -> {
                                        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                                                        .bucket(bucket)
                                                        .key(obj.key())
                                                        .build();

                                        GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
                                                        .signatureDuration(Duration.ofMinutes(10))
                                                        .getObjectRequest(getObjectRequest)
                                                        .build();

                                        return presigner.presignGetObject(presignRequest).url().toString();
                                })
                                .toList();
        }

        /**
         * ✅ 특정 시험의 인식되지 않은 학번 이미지(unknown_id) Presigned URL 목록 조회
         */
        public java.util.List<String> getUnknownIdImageUrls(String examCode) {
                String folderPrefix = String.format("header/%s/unknown_id/", examCode);

                software.amazon.awssdk.services.s3.model.ListObjectsV2Request listRequest = software.amazon.awssdk.services.s3.model.ListObjectsV2Request
                                .builder()
                                .bucket(bucket)
                                .prefix(folderPrefix)
                                .build();

                software.amazon.awssdk.services.s3.model.ListObjectsV2Response listResponse = s3Client
                                .listObjectsV2(listRequest);

                return listResponse.contents().stream()
                                .map(obj -> {
                                        software.amazon.awssdk.services.s3.model.GetObjectRequest getObjectRequest = software.amazon.awssdk.services.s3.model.GetObjectRequest
                                                        .builder()
                                                        .bucket(bucket)
                                                        .key(obj.key())
                                                        .build();

                                        software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest presignRequest = software.amazon.awssdk.services.s3.presigner.model.GetObjectPresignRequest
                                                        .builder()
                                                        .signatureDuration(java.time.Duration.ofMinutes(10))
                                                        .getObjectRequest(getObjectRequest)
                                                        .build();

                                        return presigner.presignGetObject(presignRequest).url().toString();
                                })
                                .toList();
        }

        /**
         * ✅ 특정 키에 대한 Presigned GET URL 생성 (AI 서버 또는 내부 확인용)
         */
        public String generatePresignedGetUrl(String key) {
                if (key == null || key.isEmpty()) {
                        return null;
                }

                try {
                        GetObjectRequest getObjectRequest = GetObjectRequest.builder()
                                        .bucket(bucket)
                                        .key(key)
                                        .build();

                        GetObjectPresignRequest presignRequest = GetObjectPresignRequest.builder()
                                        .signatureDuration(Duration.ofMinutes(10))
                                        .getObjectRequest(getObjectRequest)
                                        .build();

                        return presigner.presignGetObject(presignRequest).url().toString();
                } catch (Exception e) {
                        System.err.println("❌ Failed to generate presigned URL for key: " + key);
                        return null;
                }
        }
}

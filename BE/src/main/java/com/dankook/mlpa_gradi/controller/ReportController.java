package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.entity.Exam;
import com.dankook.mlpa_gradi.repository.ExamRepository;
import com.dankook.mlpa_gradi.service.AiPdfClientService;
import com.dankook.mlpa_gradi.service.PdfService;
import com.dankook.mlpa_gradi.service.S3PresignService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import lombok.extern.slf4j.Slf4j;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/reports")
@Slf4j
public class ReportController {

    private final PdfService pdfService;
    private final AiPdfClientService aiPdfClientService;
    private final S3PresignService s3PresignService;
    private final ExamRepository examRepository;
    private final com.dankook.mlpa_gradi.repository.memory.InMemoryReportRepository inMemoryReportRepository;

    /**
     * ✅ 학생 정오표 PDF 다운로드 (BE에서 iText로 직접 생성)
     */
    @GetMapping("/pdf/{examCode}/{studentId}")
    public ResponseEntity<ByteArrayResource> downloadPdf(
            @PathVariable String examCode,
            @PathVariable Long studentId) {

        byte[] pdfBytes = pdfService.generateStudentReport(examCode, studentId);
        ByteArrayResource resource = new ByteArrayResource(pdfBytes);

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.attachment()
                                .filename(examCode + "_" + studentId + "_report.pdf")
                                .build().toString())
                .contentType(MediaType.APPLICATION_PDF)
                .contentLength(pdfBytes.length)
                .body(resource);
    }

    /**
     * ✅ 과목 통계 PDF 다운로드
     * examCode → DB 검증 → examName(subject) 매핑 → FastAPI PDF 호출
     *
     * 테스트:
     * /api/reports/course-stats.pdf?examCode=ABC123
     */
    @GetMapping("/course-stats.pdf")
    public ResponseEntity<ByteArrayResource> downloadCourseStatsPdf(
            @RequestParam String examCode
    ) {
        // 1️⃣ examCode 존재 여부 검증
        Exam exam = examRepository.findByExamCode(examCode)
                .orElseThrow(() ->
                        new ResponseStatusException(
                                HttpStatus.NOT_FOUND,
                                "Invalid examCode: " + examCode
                        )
                );

        // 2️⃣ examCode → subject(examName) 매핑
        String subject = exam.getExamName();

        // 3️⃣ AI 서버(FastAPI) 호출
        byte[] pdfBytes = aiPdfClientService.fetchCourseStatsPdf(subject);
        ByteArrayResource resource = new ByteArrayResource(pdfBytes);

        // 4️⃣ PDF 다운로드 응답
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        ContentDisposition.attachment()
                                .filename("course-stats-" + examCode + ".pdf")
                                .build().toString())
                .contentType(MediaType.APPLICATION_PDF)
                .contentLength(pdfBytes.length)
                .body(resource);
    }

    /**
     * ✅ 학생 채점 이미지 S3 URL 목록 조회
     */
    @GetMapping("/images/{examCode}/{studentId}")
    public List<String> getStudentImages(
            @PathVariable String examCode,
            @PathVariable String studentId) {
        return s3PresignService.getStudentImageUrls(examCode, studentId);
    }

    @GetMapping("/unknown-images/{examCode}")
    public List<String> getUnknownImages(@PathVariable String examCode) {
        String normalizedCode = examCode.trim().toUpperCase();
        log.info("📥 Request for unknown images: examCode={}", normalizedCode);

        // 1. Get URLs from both sources
        // Get a copy to avoid ConcurrentModificationException
        List<String> memoryUrls = new ArrayList<>(inMemoryReportRepository.getUnknownImages(normalizedCode));
        List<String> s3Urls = s3PresignService.getUnknownIdImageUrls(normalizedCode);

        log.info("📊 Found {} from memory, {} from S3 for {}", memoryUrls.size(), s3Urls.size(), normalizedCode);

        // 2. Deduplicate by decoded filename
        Map<String, String> dedupMap = new LinkedHashMap<>();

        // Add memory URLs first
        for (String url : memoryUrls) {
            String filename = extractAndDecodeFilename(url);
            if (filename != null) {
                dedupMap.put(filename, url);
            }
        }

        // Add S3 URLs last (overwriting with fresh URLs if names match)
        for (String url : s3Urls) {
            String filename = extractAndDecodeFilename(url);
            if (filename != null) {
                dedupMap.put(filename, url);
            }
        }

        List<String> result = new ArrayList<>(dedupMap.values());
        log.info("✅ Returning {} unique unknown images for {}", result.size(), normalizedCode);
        return result;
    }

    private String extractAndDecodeFilename(String url) {
        if (url == null)
            return null;
        try {
            String path = url.split("\\?")[0];
            String filename = path.substring(path.lastIndexOf('/') + 1);
            return URLDecoder.decode(filename, StandardCharsets.UTF_8);
        } catch (Exception e) {
            log.warn("⚠️ Failed to extract/decode filename from URL: {}", url);
            return null;
        }
    }
}

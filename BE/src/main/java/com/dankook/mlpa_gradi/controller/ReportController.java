package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.service.PdfService;
import com.dankook.mlpa_gradi.service.S3PresignService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

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
    private final S3PresignService s3PresignService;
    private final com.dankook.mlpa_gradi.repository.memory.InMemoryReportRepository inMemoryReportRepository;

    /**
     * ✅ 학생 정오표 PDF 다운로드
     */
    @GetMapping("/pdf/{examCode}/{studentId}")
    public ResponseEntity<ByteArrayResource> downloadPdf(
            @PathVariable String examCode,
            @PathVariable Long studentId) {

        byte[] pdfBytes = pdfService.generateStudentReport(examCode, studentId);
        ByteArrayResource resource = new ByteArrayResource(pdfBytes);

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, ContentDisposition.attachment()
                        .filename(examCode + "_" + studentId + "_report.pdf")
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

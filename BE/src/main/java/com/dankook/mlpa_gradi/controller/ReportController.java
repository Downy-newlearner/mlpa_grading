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

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/reports")
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

    /**
     * ✅ 인식되지 않은 학번 이미지 S3 URL 목록 조회 (In-Memory SQS Data)
     */
    @GetMapping("/unknown-images/{examCode}")
    public List<String> getUnknownImages(@PathVariable String examCode) {
        // 기존 S3 폴더 조회 로직과 메모리 데이터 병합 또는 메모리 우선 사용
        // 여기서는 SQS로 받은 실시간 데이터를 우선합니다.
        return inMemoryReportRepository.getUnknownImages(examCode);
    }
}

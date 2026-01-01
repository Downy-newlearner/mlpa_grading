package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.entity.Exam;
import com.dankook.mlpa_gradi.entity.Question;
import com.dankook.mlpa_gradi.entity.Student;
import com.dankook.mlpa_gradi.entity.StudentAnswer;
import com.dankook.mlpa_gradi.repository.ExamRepository;
import com.dankook.mlpa_gradi.repository.StudentAnswerRepository;
import com.itextpdf.io.font.FontProgram;
import com.itextpdf.io.font.FontProgramFactory;
import com.itextpdf.io.font.PdfEncodings;
import com.itextpdf.kernel.colors.ColorConstants;
import com.itextpdf.kernel.colors.DeviceRgb;
import com.itextpdf.kernel.font.PdfFont;
import com.itextpdf.kernel.font.PdfFontFactory;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Cell;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.element.Table;
import com.itextpdf.layout.properties.TextAlignment;
import com.itextpdf.layout.properties.UnitValue;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.List;
import java.util.NoSuchElementException;

@Service
@RequiredArgsConstructor
public class PdfService {

    private final ResourceLoader resourceLoader;
    private final ExamRepository examRepository;
    private final StudentAnswerRepository studentAnswerRepository;

    /**
     * ✅ 학생 정오표 PDF 생성
     */
    public byte[] generateStudentReport(String examCode, Long studentId) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();

        try (PdfWriter writer = new PdfWriter(out);
                PdfDocument pdf = new PdfDocument(writer);
                Document document = new Document(pdf)) {

            // 1. 폰트 로드 (NanumGothic)
            PdfFont font = null;
            try {
                Resource fontResource = resourceLoader.getResource("classpath:/static/fonts/NanumGothic.ttf");
                byte[] fontBytes;
                try (InputStream is = fontResource.getInputStream()) {
                    fontBytes = is.readAllBytes();
                }
                FontProgram fontProgram = FontProgramFactory.createFont(fontBytes);
                font = PdfFontFactory.createFont(fontProgram, PdfEncodings.IDENTITY_H);
                document.setFont(font);
            } catch (Exception e) {
                System.err.println("⚠️ Font load failed, using fallback: " + e.getMessage());
            }

            // 2. 데이터 조회
            Exam exam = examRepository.findByExamCode(examCode)
                    .orElseThrow(() -> new NoSuchElementException("Exam not found: " + examCode));

            List<StudentAnswer> answers = studentAnswerRepository.findByExamCode(examCode).stream()
                    .filter(a -> a.getStudent().getStudentId().equals(studentId))
                    .toList();

            if (answers.isEmpty()) {
                throw new NoSuchElementException("No answers found for student ID: " + studentId);
            }

            Student student = answers.get(0).getStudent();

            // 3. 제목 및 기본 정보
            document.add(new Paragraph(
                    exam.getExamName() + " - " + student.getStudentName() + "(" + student.getStudentId() + ") 시험 리포트")
                    .setFontSize(18)
                    .setTextAlignment(TextAlignment.CENTER)
                    .setMarginBottom(20));

            // 4. 결과 테이블 생성
            Table table = new Table(UnitValue.createPercentArray(new float[] { 2, 3, 3, 2 }))
                    .useAllAvailableWidth();

            // 헤더 추가
            String[] headers = { "문항", "학생 응답", "정답", "취득 점수" };
            for (String h : headers) {
                table.addHeaderCell(new Cell()
                        .add(new Paragraph(h).setBold())
                        .setBackgroundColor(ColorConstants.LIGHT_GRAY)
                        .setTextAlignment(TextAlignment.CENTER));
            }

            // 색상 정의
            com.itextpdf.kernel.colors.Color correctColor = new DeviceRgb(173, 216, 230); // 하늘색
            com.itextpdf.kernel.colors.Color wrongColor = new DeviceRgb(255, 204, 203); // 연분홍

            float totalScore = 0;

            for (StudentAnswer answer : answers) {
                // 원본 문제 정보 찾기 (정답 확인용)
                Question question = exam.getQuestions().stream()
                        .filter(q -> q.getQuestionNumber() == answer.getQuestionNumber()
                                && q.getSubQuestionNumber() == answer.getSubQuestionNumber())
                        .findFirst()
                        .orElse(null);

                String questionKey = answer.getQuestionNumber()
                        + (answer.getSubQuestionNumber() > 0 ? "-" + answer.getSubQuestionNumber() : "");
                String correctAnswer = (question != null) ? question.getAnswer() : "-";

                com.itextpdf.kernel.colors.Color bgColor = answer.isCorrect() ? correctColor : wrongColor;

                table.addCell(new Cell().add(new Paragraph(questionKey)).setBackgroundColor(bgColor));
                table.addCell(new Cell().add(new Paragraph(answer.getStudentAnswer())).setBackgroundColor(bgColor));
                table.addCell(new Cell().add(new Paragraph(correctAnswer)).setBackgroundColor(bgColor));
                table.addCell(
                        new Cell().add(new Paragraph(String.valueOf(answer.getScore()))).setBackgroundColor(bgColor));

                totalScore += answer.getScore();
            }

            document.add(table);

            // 5. 총점 요약
            document.add(new Paragraph("\n총점: " + totalScore)
                    .setTextAlignment(TextAlignment.RIGHT)
                    .setFontSize(24)
                    .setBold());

        } catch (Exception e) {
            e.printStackTrace();
        }

        return out.toByteArray();
    }
}

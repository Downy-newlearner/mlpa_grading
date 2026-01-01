package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.ExamDto;
import com.dankook.mlpa_gradi.entity.Exam;
import com.dankook.mlpa_gradi.mapper.ExamMapper;
import com.dankook.mlpa_gradi.repository.ExamRepository;
import com.dankook.mlpa_gradi.repository.StudentAnswerRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.security.SecureRandom;
import java.util.List;
import java.util.NoSuchElementException;

@Service
@RequiredArgsConstructor
@Transactional
public class ExamService {

    private final ExamRepository examRepository;
    private final StudentAnswerRepository studentAnswerRepository;
    private final S3PresignService s3PresignService;

    // ✅ 혼동되는 문자 제외 (I, l, 1, O, 0)
    private static final String CODE_CHARACTERS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
    private static final int CODE_LENGTH = 6;
    private static final SecureRandom RANDOM = new SecureRandom();

    // ✅ 6자리 랜덤 코드 생성
    private String generateUniqueCode() {
        String code;
        do {
            StringBuilder sb = new StringBuilder(CODE_LENGTH);
            for (int i = 0; i < CODE_LENGTH; i++) {
                sb.append(CODE_CHARACTERS.charAt(RANDOM.nextInt(CODE_CHARACTERS.length())));
            }
            code = sb.toString();
        } while (examRepository.existsByExamCode(code)); // 중복 검사
        return code;
    }

    // ✅ 전체 시험 조회
    @Transactional(readOnly = true)
    public List<ExamDto> getAll() {
        return examRepository.findAll(Sort.by(Sort.Direction.DESC, "examDate"))
                .stream()
                .map(ExamMapper::toDto)
                .toList();
    }

    // ✅ 단일 시험 조회 (ID 기반)
    @Transactional(readOnly = true)
    public ExamDto getOne(Long examId) {
        Exam exam = examRepository.findById(examId)
                .orElseThrow(() -> new NoSuchElementException("Exam not found: " + examId));
        return ExamMapper.toDto(exam);
    }

    // ✅ 단일 시험 조회 (Code 기반)
    @Transactional(readOnly = true)
    public ExamDto getByCode(String examCode) {
        Exam exam = examRepository.findByExamCode(examCode)
                .orElseThrow(() -> new NoSuchElementException("Exam not found with code: " + examCode));
        return ExamMapper.toDto(exam);
    }

    // ✅ 시험 생성 (Code 자동 발급)
    public ExamDto create(Exam exam) {
        exam.setExamCode(generateUniqueCode()); // 6자리 코드 자동 생성
        Exam saved = examRepository.save(exam);
        return ExamMapper.toDto(saved);
    }

    // ✅ 시험 수정 (ID 기반)
    public ExamDto update(Long examId, Exam req) {
        Exam exam = examRepository.findById(examId)
                .orElseThrow(() -> new NoSuchElementException("Exam not found: " + examId));

        exam.setExamName(req.getExamName());
        exam.setExamDate(req.getExamDate());

        Exam saved = examRepository.save(exam);
        return ExamMapper.toDto(saved);
    }

    // ✅ 시험 수정 (Code 기반)
    public ExamDto updateByCode(String examCode, Exam req) {
        Exam exam = examRepository.findByExamCode(examCode)
                .orElseThrow(() -> new NoSuchElementException("Exam not found with code: " + examCode));

        exam.setExamName(req.getExamName());
        exam.setExamDate(req.getExamDate());

        Exam saved = examRepository.save(exam);
        return ExamMapper.toDto(saved);
    }

    // ✅ 시험 삭제 (ID 기반)
    public void delete(Long examId) {
        if (!examRepository.existsById(examId)) {
            throw new NoSuchElementException("Exam not found: " + examId);
        }
        examRepository.deleteById(examId);
    }

    // ✅ 시험 삭제 (Code 기반)
    public void deleteByCode(String examCode) {
        String trimmedCode = examCode.trim();

        // 1. S3 데이터 삭제 (이미지 + 출석부)
        s3PresignService.deleteByExamCode(trimmedCode);

        // 2. 해당 시험의 답안들 먼저 삭제 (Casacade가 안되어있으므로 명시적 삭제)
        studentAnswerRepository.deleteByExamCode(trimmedCode);

        // 3. 시험 삭제
        Exam exam = examRepository.findByExamCode(trimmedCode)
                .orElseThrow(() -> new NoSuchElementException("Exam not found with code: " + trimmedCode));
        examRepository.delete(exam);
    }
}

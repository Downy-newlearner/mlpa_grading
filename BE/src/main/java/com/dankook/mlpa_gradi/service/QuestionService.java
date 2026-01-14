package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.QuestionDto;
import com.dankook.mlpa_gradi.entity.Question;
import com.dankook.mlpa_gradi.entity.Exam;
import com.dankook.mlpa_gradi.mapper.QuestionMapper;
import com.dankook.mlpa_gradi.repository.ExamRepository;
import com.dankook.mlpa_gradi.repository.QuestionRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class QuestionService {

    private final QuestionRepository questionRepository;
    private final ExamRepository examRepository;

    public List<QuestionDto> getAll() {
        return questionRepository.findAll().stream()
                .map(QuestionMapper::toDto)
                .toList();
    }

    public QuestionDto getOne(Long questionId) {
        Question q = questionRepository.findById(questionId)
                .orElseThrow(() -> new RuntimeException("Question not found: " + questionId));
        return QuestionMapper.toDto(q);
    }

    public QuestionDto create(Question question) {
        Question saved = questionRepository.save(question);
        return QuestionMapper.toDto(saved);
    }

    private final PdfService pdfService;

    public String proxyQuestionsToAi(String examCode) {
        // 1. "정답지가 뭐야?" -> AI 서버에 인식된 정답지 요청
        log.info("[PROXY] Asking AI for answer key: {}", examCode);
        List<QuestionDto> aiQuestions = pdfService.getAnswerKeyFromAi(examCode);

        // 2. 받은 정답지를 로컬 DB에 저장
        Exam exam = examRepository.findByExamCode(examCode)
                .orElseThrow(() -> new RuntimeException("Exam not found: " + examCode));

        // 기존 문항 삭제 (새로운 인식 결과로 갱신하기 위해)
        List<Question> existing = questionRepository.findByExam_ExamCode(examCode);
        if (!existing.isEmpty()) {
            questionRepository.deleteAll(existing);
        }

        List<Question> newQuestions = aiQuestions.stream().map(dto -> {
            Question q = QuestionMapper.toEntity(dto);
            q.setExam(exam);
            return q;
        }).toList();
        questionRepository.saveAll(newQuestions);

        log.info("[PROXY] Forwarding fetched questions to AI start recognition: {}", examCode);

        // 3. "FastAPI 서버로 보내야해" -> POST /recognition/answer/start
        return pdfService.sendQuestions(aiQuestions);
    }
}

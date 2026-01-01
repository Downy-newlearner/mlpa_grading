package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.QuestionDto;
import com.dankook.mlpa_gradi.entity.Question;
import com.dankook.mlpa_gradi.mapper.QuestionMapper;
import com.dankook.mlpa_gradi.repository.QuestionRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class QuestionService {

    private final QuestionRepository questionRepository;

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
}

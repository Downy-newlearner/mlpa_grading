package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.StudentAnswerDto;
import com.dankook.mlpa_gradi.entity.StudentAnswer;
import com.dankook.mlpa_gradi.mapper.StudentAnswerMapper;
import com.dankook.mlpa_gradi.repository.StudentAnswerRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class StudentAnswerService {

    private final StudentAnswerRepository studentAnswerRepository;

    public List<StudentAnswerDto> getAll() {
        return studentAnswerRepository.findAll()
                .stream()
                .map(StudentAnswerMapper::toDto)
                .toList();
    }

    public List<StudentAnswerDto> getByExamCode(String examCode) {
        return studentAnswerRepository.findByExamCode(examCode)
                .stream()
                .map(StudentAnswerMapper::toDto)
                .toList();
    }

    public StudentAnswerDto create(StudentAnswer studentAnswer) {
        return StudentAnswerMapper.toDto(
                studentAnswerRepository.save(studentAnswer)
        );
    }
}

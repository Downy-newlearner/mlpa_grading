package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.StudentAnswerDto;
import com.dankook.mlpa_gradi.entity.StudentAnswer;
import com.dankook.mlpa_gradi.service.StudentAnswerService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/student-answers")
public class StudentAnswerController {

    private final StudentAnswerService studentAnswerService;

    @GetMapping
    public List<StudentAnswerDto> getAll() {
        return studentAnswerService.getAll();
    }

    @GetMapping("/exam/{examCode}")
    public List<StudentAnswerDto> getByExamCode(@PathVariable String examCode) {
        return studentAnswerService.getByExamCode(examCode);
    }

    @PostMapping
    public StudentAnswerDto create(@RequestBody StudentAnswer studentAnswer) {
        return studentAnswerService.create(studentAnswer);
    }
}

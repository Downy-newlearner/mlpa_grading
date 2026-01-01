package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.QuestionDto;
import com.dankook.mlpa_gradi.entity.Question;
import com.dankook.mlpa_gradi.service.QuestionService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/questions")
public class QuestionController {

    private final QuestionService questionService;

    @GetMapping
    public List<QuestionDto> getAll() {
        return questionService.getAll();
    }

    @GetMapping("/{questionId}")
    public QuestionDto getOne(@PathVariable Long questionId) {
        return questionService.getOne(questionId);
    }

    @PostMapping
    public QuestionDto create(@RequestBody Question question) {
        return questionService.create(question);
    }
}

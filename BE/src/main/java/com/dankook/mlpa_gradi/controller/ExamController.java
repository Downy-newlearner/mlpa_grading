package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.ExamDto;
import com.dankook.mlpa_gradi.entity.Exam;
import com.dankook.mlpa_gradi.service.ExamService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/exams")
public class ExamController {

    private final ExamService examService;

    // ✅ 전체 시험 조회
    @GetMapping
    public List<ExamDto> getAll() {
        return examService.getAll();
    }

    // ✅ 단일 시험 조회 (ID 기반)
    @GetMapping("/{examId}")
    public ExamDto getOne(@PathVariable Long examId) {
        return examService.getOne(examId);
    }

    // ✅ 단일 시험 조회 (Code 기반)
    @GetMapping("/code/{examCode}")
    public ExamDto getByCode(@PathVariable String examCode) {
        return examService.getByCode(examCode);
    }

    // ✅ 시험 생성 (Code 자동 발급)
    @PostMapping
    public ExamDto create(@RequestBody Exam exam) {
        return examService.create(exam);
    }

    // ✅ 시험 수정 (ID 기반)
    @PutMapping("/{examId}")
    public ExamDto update(@PathVariable Long examId, @RequestBody Exam req) {
        return examService.update(examId, req);
    }

    // ✅ 시험 수정 (Code 기반)
    @PutMapping("/code/{examCode}")
    public ExamDto updateByCode(@PathVariable String examCode, @RequestBody Exam req) {
        return examService.updateByCode(examCode, req);
    }

    // ✅ 시험 삭제 (ID 기반)
    @DeleteMapping("/{examId}")
    public void delete(@PathVariable Long examId) {
        examService.delete(examId);
    }

    // ✅ 시험 삭제 (Code 기반)
    @DeleteMapping("/code/{examCode}")
    public void deleteByCode(@PathVariable String examCode) {
        examService.deleteByCode(examCode);
    }
}

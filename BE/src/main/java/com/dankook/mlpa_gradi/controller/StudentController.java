package com.dankook.mlpa_gradi.controller;

import com.dankook.mlpa_gradi.dto.StudentDto;
import com.dankook.mlpa_gradi.entity.Student;
import com.dankook.mlpa_gradi.service.StudentService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/students")
public class StudentController {

    private final StudentService studentService;

    @GetMapping
    public List<StudentDto> getAll() {
        return studentService.getAll();
    }

    @GetMapping("/{studentId}")
    public StudentDto getOne(@PathVariable Long studentId) {
        return studentService.getOne(studentId);
    }

    @PostMapping
    public StudentDto create(@RequestBody Student student) {
        return studentService.create(student);
    }
}

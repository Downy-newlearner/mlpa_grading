package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.StudentDto;
import com.dankook.mlpa_gradi.entity.Student;
import com.dankook.mlpa_gradi.mapper.StudentMapper;
import com.dankook.mlpa_gradi.repository.StudentRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class StudentService {

    private final StudentRepository studentRepository;

    public List<StudentDto> getAll() {
        return studentRepository.findAll().stream()
                .map(StudentMapper::toDto)
                .toList();
    }

    public StudentDto getOne(Long studentId) {
        Student s = studentRepository.findById(String.valueOf(studentId))
                .orElseThrow(() -> new RuntimeException("Student not found: " + studentId));
        return StudentMapper.toDto(s);
    }

    public StudentDto create(Student student) {
        Student saved = studentRepository.save(student);
        return StudentMapper.toDto(saved);
    }
}

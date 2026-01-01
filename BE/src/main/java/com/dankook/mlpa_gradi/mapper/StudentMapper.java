package com.dankook.mlpa_gradi.mapper;

import com.dankook.mlpa_gradi.dto.StudentDto;
import com.dankook.mlpa_gradi.entity.Student;

public class StudentMapper {

    public static StudentDto toDto(Student s) {
        StudentDto dto = new StudentDto();
        dto.setStudentId(s.getStudentId());
        dto.setStudentName(s.getStudentName());
        return dto;
    }
}

package com.dankook.mlpa_gradi.repository;

import com.dankook.mlpa_gradi.entity.Student;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StudentRepository extends JpaRepository<Student, String> {
    // studentId가 PK라 String
}

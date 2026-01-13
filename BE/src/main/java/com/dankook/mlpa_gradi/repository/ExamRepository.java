package com.dankook.mlpa_gradi.repository;

import com.dankook.mlpa_gradi.entity.Exam;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface ExamRepository extends JpaRepository<Exam, Long> {
    Optional<Exam> findByExamCode(String examCode);

    boolean existsByExamCode(String examCode);
}

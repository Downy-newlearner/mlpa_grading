package com.dankook.mlpa_gradi.repository;

import com.dankook.mlpa_gradi.entity.Question;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface QuestionRepository extends JpaRepository<Question, Long> {

    // 특정 시험에 속한 문제들
    List<Question> findByExam_ExamId(Long examId);
}

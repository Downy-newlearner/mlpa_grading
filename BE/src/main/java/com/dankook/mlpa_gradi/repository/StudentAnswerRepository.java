package com.dankook.mlpa_gradi.repository;

import com.dankook.mlpa_gradi.entity.StudentAnswer;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

public interface StudentAnswerRepository extends JpaRepository<StudentAnswer, Long> {

    // 시험 코드 기준 전체 답안 조회
    List<StudentAnswer> findByExamCode(String examCode);

    // 시험 코드 + 문제 번호
    List<StudentAnswer> findByExamCodeAndQuestionNumber(
            String examCode,
            int questionNumber);

    // 시험 코드 기준 삭제
    @Modifying
    @Transactional
    void deleteByExamCode(String examCode);
}

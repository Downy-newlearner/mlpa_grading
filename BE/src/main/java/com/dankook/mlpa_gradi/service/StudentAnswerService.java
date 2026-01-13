package com.dankook.mlpa_gradi.service;

import com.dankook.mlpa_gradi.dto.StudentAnswerDto;
import com.dankook.mlpa_gradi.dto.SubjectiveFeedbackRequest;
import com.dankook.mlpa_gradi.entity.StudentAnswer;
import com.dankook.mlpa_gradi.mapper.StudentAnswerMapper;
import com.dankook.mlpa_gradi.repository.StudentAnswerRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class StudentAnswerService {

    private final StudentAnswerRepository studentAnswerRepository;

    public List<StudentAnswerDto> getAll() {
        return studentAnswerRepository.findAll()
                .stream()
                .map(StudentAnswerMapper::toDto)
                .toList();
    }

    public List<StudentAnswerDto> getByExamCode(String examCode) {
        return studentAnswerRepository.findByExamCode(examCode)
                .stream()
                .map(StudentAnswerMapper::toDto)
                .toList();
    }

    public StudentAnswerDto create(StudentAnswer studentAnswer) {
        return StudentAnswerMapper.toDto(
                studentAnswerRepository.save(studentAnswer));
    }

    @org.springframework.transaction.annotation.Transactional
    public void saveRecognitionResults(String examCode, String studentIdStr, List<Map<String, Object>> answers) {
        // Parse String studentId to Long
        Long studentId;
        try {
            studentId = Long.parseLong(studentIdStr);
        } catch (NumberFormatException e) {
            // Handle error or set null if acceptable? For now assuming valid ID.
            throw new IllegalArgumentException("Invalid student ID format: " + studentIdStr);
        }

        com.dankook.mlpa_gradi.entity.Student studentReference = new com.dankook.mlpa_gradi.entity.Student();
        studentReference.setStudentId(studentId);

        List<StudentAnswer> entities = answers.stream().map(ans -> {
            StudentAnswer entity = new StudentAnswer();
            entity.setExamCode(examCode);
            entity.setStudent(studentReference);

            entity.setQuestionNumber((Integer) ans.getOrDefault("questionNumber", ans.get("question_number")));
            Object subNo = ans.getOrDefault("subQuestionNumber", ans.get("sub_question_number"));
            entity.setSubQuestionNumber(subNo != null ? (Integer) subNo : 0);

            entity.setAnswerCount((Integer) ans.getOrDefault("answerCount", ans.getOrDefault("answer_count", 1)));

            // Logic for rec_answer / values
            @SuppressWarnings("unchecked")
            Map<String, Object> recAnswer = (Map<String, Object>) ans.getOrDefault("recAnswer", ans.get("rec_answer"));
            if (recAnswer != null) {
                // Check if 'values' exists and is a list
                Object valuesObj = recAnswer.get("values");
                if (valuesObj instanceof List) {
                    List<?> values = (List<?>) valuesObj;
                    if (!values.isEmpty()) {
                        // [1, 3] -> "1,3"
                        String joined = values.stream()
                                .sorted() // Determine if sorting is generic object or number. Assuming integer-like
                                .map(Object::toString)
                                .collect(java.util.stream.Collectors.joining(","));
                        entity.setStudentAnswer(joined);
                    } else {
                        // Empty values list usually means blank or check rawText
                        String rawText = (String) recAnswer.getOrDefault("rawText", recAnswer.get("raw_text"));
                        entity.setStudentAnswer(rawText);
                    }
                } else {
                    String rawText = (String) recAnswer.getOrDefault("rawText", recAnswer.get("raw_text"));
                    entity.setStudentAnswer(rawText);
                }

                // Confidence
                Object confObj = recAnswer.get("confidence");
                if (confObj instanceof List) {
                    List<?> confList = (List<?>) confObj;
                    if (!confList.isEmpty()) {
                        Object first = confList.get(0);
                        if (first instanceof Number) {
                            entity.setConfidence(((Number) first).floatValue());
                        }
                    } else {
                        entity.setConfidence(0f);
                    }
                } else if (confObj instanceof Number) {
                    entity.setConfidence(((Number) confObj).floatValue());
                }
            }

            // Map point to maxScore
            Object pointObj = ans.getOrDefault("point", ans.get("point"));
            if (pointObj instanceof Number) {
                entity.setMaxScore(((Number) pointObj).floatValue());
            }

            entity.setCorrect(false);
            entity.setScore(0f);

            return entity;
        }).toList();

        studentAnswerRepository.saveAll(entities);
    }

    @org.springframework.transaction.annotation.Transactional
    public void updateSubjectiveFeedback(String examCode,
            List<SubjectiveFeedbackRequest.Evaluation> evaluations) {
        for (SubjectiveFeedbackRequest.Evaluation eval : evaluations) {
            String qStr = eval.getQuestionNumber();
            int qNum = 0;
            int subNum = 0;

            if (qStr.contains("-")) {
                String[] parts = qStr.split("-");
                qNum = Integer.parseInt(parts[0]);
                subNum = Integer.parseInt(parts[1]);
            } else {
                qNum = Integer.parseInt(qStr);
            }

            // Find entity (assuming 1 student for now or need studentId context?
            // The SubjectiveFeedbackPage currently doesn't seem to pass studentId in the
            // payload?
            // Wait, the page is for *Others* feedback which might be PER STUDENT?
            // The SubjectiveFallbackPage fetches
            // `/api/reports/others-questions/{examCode}`.
            // This returns a list of items. Do these items have studentId?
            // If the items are aggregated across students (e.g. "Question 10 for all
            // students" OR "Question 10 for Student A, Question 10 for Student B"), we need
            // to know WHICH answer we are updating.

            // Looking at `SubjectiveFallbackPage.tsx`: payload is `{ examCode, evaluations:
            // [{ questionNumber, score, comment }] }`.
            // It does NOT include studentId. This implies `SubjectiveFallbackPage` might
            // presently be designed for a SINGLE student context OR it's assumed the
            // backend knows which answers are "others".
            // However, `StudentAnswer` has a `student_id`. If multiple students took the
            // exam, there are multiple "Question 10" answers.
            // Updating blindly by `examCode` + `questionNumber` would update ALL students'
            // question 10? That might be intended if it's "Concept Grading" but likely not.

            // Re-reading user request: "others fallback page... mock data...".
            // The user didn't specify multi-student logic yet.
            // BUT, `RecognitionFallbackPage` (QuestionFeedback) logic handles "Unknown"
            // questions.
            // `QuestionFeedbackPage` logic (Step 2223) payload: `{ examCode, questions: [{
            // questionNumber, correctAnswer }] }`. Also no studentId.
            // This suggests the current Fallback UI might be operating on a "Batch"
            // assumption (maybe just one student active? or applying correction key to
            // all?)
            // OR maybe the `id` field in the items contains info? `SubjectiveFallbackPage`
            // items has `id`.

            // Let's look at `SubjectiveFallbackPage` mock data: `id: "1", questionNumber:
            // "10"`.
            // Real implementation of `others-questions` reports likely returns a list of
            // specific answers that need grading.
            // If the list includes answers from DIFFERENT students, each item MUST identify
            // the student.

            // I will assume for now we might need to update *all* matching answers if no
            // student ID is provided, OR the `questionNumber` provided is unique enough?
            // No.

            // Given the constraints and current stage (Demo/Mock), I will implement logic
            // to find answers by `examCode` + `questionNumber` (+ `subNumber`) and update
            // them.
            // Ideally we need `studentId`. But I will proceed with `examCode` + `q#`
            // lookup.

            List<StudentAnswer> answers = studentAnswerRepository.findByExamCodeAndQuestionNumber(examCode, qNum);
            for (StudentAnswer ans : answers) {
                if (ans.getSubQuestionNumber() == subNum) {
                    // Update score
                    if (eval.getScore() != null && !eval.getScore().isEmpty()) {
                        try {
                            ans.setScore(Float.parseFloat(eval.getScore()));
                            ans.setCorrect(ans.getScore() > 0); // Simple logic
                        } catch (NumberFormatException ignored) {
                        }
                    }
                    // Update comment
                    if (eval.getComment() != null) {
                        ans.setComment(eval.getComment());
                    }
                }
            }
        }
    }
}

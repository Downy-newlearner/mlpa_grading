"""
sqs_schemas.py - SQS 메시지 스키마 정의

BE ↔ AI 서버 간 SQS 메시지 포맷을 정의합니다.
모든 JSON 필드는 camelCase를 사용합니다.

Event Types:
- ATTENDANCE_UPLOAD: 출석부 업로드 알림
- STUDENT_ID_RECOGNITION: 이미지 학번 추출 요청/결과
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import json


# =============================================================================
# Constants
# =============================================================================
EVENT_ATTENDANCE_UPLOAD = "ATTENDANCE_UPLOAD"
EVENT_STUDENT_ID_RECOGNITION = "STUDENT_ID_RECOGNITION"
EVENT_ANSWER_METADATA_UPLOAD = "ANSWER_METADATA_UPLOAD"  # 정답 메타데이터 업로드
EVENT_ANSWER_RECOGNITION = "ANSWER_RECOGNITION"  # 답안 인식 요청
EVENT_GRADING_COMPLETE = "GRADING_COMPLETE"  # 채점 완료 요청
EVENT_GRADING_RESULT = "GRADING_RESULT"  # 채점 결과 (AI → BE)
UNKNOWN_ID = "unknown_id"


def to_camel_case(snake_str: str) -> str:
    """snake_case를 camelCase로 변환"""
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def dict_to_camel_case(d: dict) -> dict:
    """딕셔너리의 모든 키를 camelCase로 변환"""
    return {to_camel_case(k): v for k, v in d.items() if v is not None}


# =============================================================================
# BE → AI: 입력 메시지 (SQS에서 수신)
# =============================================================================
@dataclass
class SQSInputMessage:
    """
    백엔드에서 AI 서버로 전송되는 SQS 메시지
    
    Event Types:
    - ATTENDANCE_UPLOAD: 출석부 다운로드 요청
      {
        "eventType": "ATTENDANCE_UPLOAD",
        "examCode": "TEST",
        "downloadUrl": "presigned_url",
        "filename": "출석부.xlsx"
      }
      
    - STUDENT_ID_RECOGNITION: 이미지 학번 추출 요청
      {
        "eventType": "STUDENT_ID_RECOGNITION",
        "examCode": "AI_2024_MID",
        "downloadUrl": "presigned_url",
        "filename": "page_001.jpg"
      }
    """
    event_type: str
    exam_code: str
    filename: str
    download_url: str
    receipt_handle: Optional[str] = None  # SQS 메시지 삭제용 (내부 사용)
    
    @classmethod
    def from_sqs_message(cls, body: dict, receipt_handle: str = None) -> "SQSInputMessage":
        """
        SQS 메시지 Body(camelCase JSON)에서 객체 생성
        """
        return cls(
            event_type=body.get("eventType", ""),
            exam_code=body.get("examCode", ""),
            filename=body.get("filename", ""),
            download_url=body.get("downloadUrl", ""),
            receipt_handle=receipt_handle
        )


# =============================================================================
# AI → BE: 출력 메시지 (SQS로 전송)
# =============================================================================
@dataclass
class SQSOutputMessage:
    """
    AI 서버에서 백엔드로 전송하는 SQS 결과 메시지
    
    Case 1: 학번 인식 성공
    {
      "eventType": "STUDENT_ID_RECOGNITION",
      "examCode": "AI_2024_MID",
      "studentId": "20211234",
      "filename": "page_001.jpg",
      "index": 0
    }
    
    Case 2: 학번 인식 실패
    {
      "eventType": "STUDENT_ID_RECOGNITION",
      "examCode": "AI_2024_MID",
      "studentId": "unknown_id",
      "filename": "page_001.jpg",
      "index": 0
    }
    """
    event_type: str
    exam_code: str
    student_id: str  # 실패 시 "unknown_id"
    filename: str
    index: int
    
    def to_dict(self) -> dict:
        """camelCase 딕셔너리로 변환"""
        return {
            "eventType": self.event_type,
            "examCode": self.exam_code,
            "studentId": self.student_id,
            "filename": self.filename,
            "index": self.index,
        }
    
    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def create(
        cls,
        exam_code: str,
        student_id: Optional[str],
        filename: str,
        index: int,
        event_type: str = EVENT_STUDENT_ID_RECOGNITION
    ) -> "SQSOutputMessage":
        """
        팩토리 메서드: 결과 메시지 생성
        
        Args:
            exam_code: 시험 코드
            student_id: 추출된 학번 (None이면 unknown_id)
            filename: 파일명
            index: AI 서버가 카운트한 순차 인덱스
            event_type: 이벤트 타입
        """
        effective_student_id = student_id if student_id else UNKNOWN_ID
        
        return cls(
            event_type=event_type,
            exam_code=exam_code,
            student_id=effective_student_id,
            filename=filename,
            index=index
        )


# =============================================================================
# 답안 인식 관련 메시지 스키마
# =============================================================================
@dataclass
class AnswerRecognitionInputMessage:
    """
    답안 인식 요청 메시지 (BE → AI)
    
    {
        "eventType": "ANSWER_RECOGNITION",
        "examCode": "AI_2024_MID",
        "downloadUrl": "presigned_url",
        "filename": "page_001.jpg",
        "studentId": "20201234"  # 이미 인식된 학번
    }
    """
    event_type: str
    exam_code: str
    filename: str
    download_url: str
    student_id: str  # 학번 인식 단계에서 이미 확정된 학번
    receipt_handle: Optional[str] = None
    
    @classmethod
    def from_sqs_message(cls, body: dict, receipt_handle: str = None) -> "AnswerRecognitionInputMessage":
        return cls(
            event_type=body.get("eventType", ""),
            exam_code=body.get("examCode", ""),
            filename=body.get("filename", ""),
            download_url=body.get("downloadUrl", ""),
            student_id=body.get("studentId", ""),
            receipt_handle=receipt_handle
        )


@dataclass
class AnswerRecognitionResultItem:
    """개별 문제 답안 인식 결과"""
    question_number: int
    sub_question_number: int  # 0 = 꼬리문제 없음
    rec_answer: Optional[str]
    confidence: float
    is_fallback: bool = False
    s3_key: Optional[str] = None  # Fallback 시 ROI 이미지 S3 키
    
    def to_dict(self) -> dict:
        return {
            "questionNumber": self.question_number,
            "subQuestionNumber": self.sub_question_number,
            "recAnswer": self.rec_answer,
            "confidence": round(self.confidence, 4),
            "isFallback": self.is_fallback,
            "s3Key": self.s3_key
        }


@dataclass
class AnswerRecognitionOutputMessage:
    """
    답안 인식 결과 메시지 (AI → BE)
    
    {
        "eventType": "ANSWER_RECOGNITION",
        "examCode": "AI_2024_MID",
        "studentId": "20201234",
        "filename": "page_001.jpg",
        "success": true,
        "results": [
            {"questionNumber": 1, "subQuestionNumber": 0, "recAnswer": "3", "confidence": 0.95, "isFallback": false},
            {"questionNumber": 2, "subQuestionNumber": 1, "recAnswer": null, "confidence": 0.3, "isFallback": true, "s3Key": "answer/..."}
        ],
        "fallbackCount": 1
    }
    """
    event_type: str
    exam_code: str
    student_id: str
    filename: str
    success: bool
    results: list  # List[AnswerRecognitionResultItem]
    fallback_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "eventType": self.event_type,
            "examCode": self.exam_code,
            "studentId": self.student_id,
            "filename": self.filename,
            "success": self.success,
            "results": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.results],
            "fallbackCount": self.fallback_count,
            "errorMessage": self.error_message
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def create(
        cls,
        exam_code: str,
        student_id: str,
        filename: str,
        results: list,
        success: bool = True,
        error_message: str = None
    ) -> "AnswerRecognitionOutputMessage":
        fallback_count = sum(1 for r in results if getattr(r, 'is_fallback', False))
        return cls(
            event_type=EVENT_ANSWER_RECOGNITION,
            exam_code=exam_code,
            student_id=student_id,
            filename=filename,
            success=success,
            results=results,
            fallback_count=fallback_count,
            error_message=error_message
        )


# 이벤트 타입 상수 추가
EVENT_ANSWER_FALLBACK = "ANSWER_FALLBACK"

@dataclass
class AnswerFallbackMessage:
    """
    답안 인식 Fallback 알림 메시지 (AI → BE, SQS_QUEUE_URL3)
    
    Low confidence ROI를 S3에 업로드하고, 그 정보를 백엔드에 알립니다.
    
    {
        "eventType": "ANSWER_FALLBACK",
        "examCode": "AI_2024_MID",
        "studentId": "20201234",
        "questionNumber": 2,
        "subQuestionNumber": 1,
        "recAnswer": null,
        "confidence": 0.3,
        "s3Key": "answer/AI_2024_MID/20201234/2/1/roi.jpg"
    }
    """
    event_type: str
    exam_code: str
    student_id: str
    question_number: int
    sub_question_number: int
    rec_answer: Optional[str]
    confidence: float
    s3_key: str
    
    def to_dict(self) -> dict:
        return {
            "eventType": self.event_type,
            "examCode": self.exam_code,
            "studentId": self.student_id,
            "questionNumber": self.question_number,
            "subQuestionNumber": self.sub_question_number,
            "recAnswer": self.rec_answer,
            "confidence": round(self.confidence, 4),
            "s3Key": self.s3_key
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def create(
        cls,
        exam_code: str,
        student_id: str,
        question_number: int,
        sub_question_number: int,
        rec_answer: Optional[str],
        confidence: float,
        s3_key: str
    ) -> "AnswerFallbackMessage":
        return cls(
            event_type=EVENT_ANSWER_FALLBACK,
            exam_code=exam_code,
            student_id=student_id,
            question_number=question_number,
            sub_question_number=sub_question_number,
            rec_answer=rec_answer,
            confidence=confidence,
            s3_key=s3_key
        )


@dataclass
class GradingResultMessage:
    """
    채점 결과 메시지 (AI → BE)
    
    {
        "eventType": "GRADING_RESULT",
        "examCode": "AI_2024_MID",
        "totalStudents": 30,
        "gradedStudents": 28,
        "pendingFallbacks": 2,
        "results": [
            {"studentId": "20201234", "totalPoints": 50, "earnedPoints": 42, "correctCount": 8},
            ...
        ]
    }
    """
    event_type: str
    exam_code: str
    total_students: int
    graded_students: int
    pending_fallbacks: int
    results: list  # List[StudentGradeResult]
    
    def to_dict(self) -> dict:
        return {
            "eventType": self.event_type,
            "examCode": self.exam_code,
            "totalStudents": self.total_students,
            "gradedStudents": self.graded_students,
            "pendingFallbacks": self.pending_fallbacks,
            "results": self.results
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# =============================================================================
# Fallback API 스키마
# =============================================================================
@dataclass
class FallbackImageItem:
    """Fallback 요청의 개별 이미지 항목"""
    filename: str
    student_id: str
    
    @classmethod
    def from_dict(cls, d: dict) -> "FallbackImageItem":
        return cls(
            filename=d.get("filename", ""),
            student_id=d.get("studentId", "")
        )


@dataclass
class FallbackRequest:
    """
    사용자 Fallback 요청 스키마
    
    Request Body (camelCase):
    {
        "examCode": "ABCDEF",
        "images": [
            {"filename": "image1.jpg", "studentId": "32204077"},
            ...
        ]
    }
    """
    exam_code: str
    images: list[FallbackImageItem] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, d: dict) -> "FallbackRequest":
        exam_code = d.get("examCode", "")
        images = [FallbackImageItem.from_dict(img) for img in d.get("images", [])]
        return cls(exam_code=exam_code, images=images)


# =============================================================================
# 테스트 코드
# =============================================================================
if __name__ == "__main__":
    print("=== SQS 입력 메시지 (BE → AI) ===")
    
    # ATTENDANCE_UPLOAD 예시
    attendance_body = {
        "eventType": "ATTENDANCE_UPLOAD",
        "examCode": "TEST",
        "downloadUrl": "https://presigned-url...",
        "filename": "출석부.xlsx"
    }
    attendance_msg = SQSInputMessage.from_sqs_message(attendance_body, receipt_handle="test-handle")
    print(f"[ATTENDANCE_UPLOAD]")
    print(f"  event_type: {attendance_msg.event_type}")
    print(f"  exam_code: {attendance_msg.exam_code}")
    print(f"  download_url: {attendance_msg.download_url}")
    print(f"  filename: {attendance_msg.filename}")
    print()
    
    # STUDENT_ID_RECOGNITION 예시
    recognition_body = {
        "eventType": "STUDENT_ID_RECOGNITION",
        "examCode": "AI_2024_MID",
        "downloadUrl": "https://presigned-url.../page_001.jpg",
        "filename": "page_001.jpg"
    }
    recognition_msg = SQSInputMessage.from_sqs_message(recognition_body, receipt_handle="test-handle")
    print(f"[STUDENT_ID_RECOGNITION]")
    print(f"  event_type: {recognition_msg.event_type}")
    print(f"  exam_code: {recognition_msg.exam_code}")
    print(f"  download_url: {recognition_msg.download_url}")
    print(f"  filename: {recognition_msg.filename}")
    print()
    
    print("=== SQS 출력 메시지 (AI → BE) ===")
    output_msg = SQSOutputMessage.create(
        exam_code="AI_2024_MID",
        student_id="20211234",
        filename="page_001.jpg",
        index=0  # AI 서버가 카운트한 값
    )
    print(output_msg.to_json())
    print()
    
    print("=== Fallback 요청 ===")
    fallback_body = {
        "examCode": "ABCDEF",
        "images": [
            {"filename": "image1.jpg", "studentId": "32204077"},
            {"filename": "image2.jpg", "studentId": "32204078"}
        ]
    }
    fallback_req = FallbackRequest.from_dict(fallback_body)
    print(f"exam_code: {fallback_req.exam_code}")
    print(f"images: {[(img.filename, img.student_id) for img in fallback_req.images]}")


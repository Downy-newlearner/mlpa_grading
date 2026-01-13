"""
sqs_worker.py - SQS Consumer Worker

백그라운드에서 SQS 메시지를 수신하고 처리합니다.

Event Types:
- ATTENDANCE_UPLOAD: presigned URL에서 출석부 다운로드 → 파싱
- STUDENT_ID_RECOGNITION: S3에서 이미지 다운로드 → 학번 추출 → 결과 전송
"""

import os
import io
import json
import time
import logging
import threading
import tempfile
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

import boto3
import requests
import numpy as np
from PIL import Image
from botocore.exceptions import ClientError

from id_recog.sqs_schemas import (
    SQSInputMessage, 
    SQSOutputMessage,
    AnswerRecognitionInputMessage,
    AnswerRecognitionOutputMessage,
    AnswerRecognitionResultItem,
    AnswerFallbackMessage,
    GradingResultMessage,
    EVENT_ATTENDANCE_UPLOAD,
    EVENT_STUDENT_ID_RECOGNITION,
    EVENT_ANSWER_METADATA_UPLOAD,
    EVENT_ANSWER_RECOGNITION,
    EVENT_ANSWER_FALLBACK,
    EVENT_GRADING_COMPLETE,
    EVENT_GRADING_RESULT,
    UNKNOWN_ID
)

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQSWorker:
    """
    SQS Consumer Worker
    
    백그라운드에서 SQS로부터 메시지를 수신하고 처리합니다.
    - ATTENDANCE_UPLOAD: 출석부 다운로드 및 파싱
    - STUDENT_ID_RECOGNITION: 이미지 학번 추출
    - ANSWER_METADATA_UPLOAD: 정답 메타데이터 업로드
    - ANSWER_RECOGNITION: 답안 인식
    - GRADING_COMPLETE: 채점 완료 요청
    """
    
    def __init__(
        self,
        queue_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "ap-northeast-2",
        s3_bucket: str = "mlpa-gradi",
        result_queue_url: str = None,  # AI → BE 결과 전송용 큐 (None이면 queue_url 사용)
        fallback_queue_url: str = None  # AI → BE Fallback 알림용 큐
    ):
        self.queue_url = queue_url  # BE → AI 입력 큐
        self.result_queue_url = result_queue_url if result_queue_url else queue_url  # AI → BE 결과 큐
        self.fallback_queue_url = fallback_queue_url  # AI → BE Fallback 알림 큐
        self.s3_bucket = s3_bucket
        
        # SQS 클라이언트
        self.sqs = boto3.client(
            'sqs',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # S3 클라이언트 (이미지 다운로드/업로드용)
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # 워커 상태
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # 콜백 함수
        self._student_id_callback: Optional[Callable] = None
        self._attendance_callback: Optional[Callable] = None
        
        # ExamCode별 학번 리스트 저장소
        self._student_id_lists: Dict[str, List[str]] = {}
        
        # ExamCode별 index 카운터 (AI 서버에서 0부터 카운트)
        self._index_counters: Dict[str, int] = {}
        
        # ExamCode별 정답 메타데이터 저장소 (답안 인식용)
        self._answer_metadata: Dict[str, dict] = {}
        
        # =====================================================================
        # NACK 추적 (무한 재시도 방지용)
        # =====================================================================
        # 키: f"{exam_code}:{filename}", 값: NACK 횟수
        self._nack_tracker: Dict[str, int] = {}
        # 최대 NACK 횟수 (이후 메시지 삭제 및 에러 로깅)
        self._max_nack_count: int = 5
        
        logger.info(f"SQS Worker 초기화 완료: 입력={queue_url}, 결과={self.result_queue_url}")
    
    def set_student_id_callback(self, callback: Callable[[np.ndarray, List[str]], dict]):
        """
        학번 추출 콜백 함수 설정
        
        Args:
            callback: (image, student_id_list) -> {"student_id": str | None, "meta": dict}
        """
        self._student_id_callback = callback
    
    def set_attendance_callback(self, callback: Callable[[str], List[str]]):
        """
        출석부 파싱 콜백 함수 설정
        
        Args:
            callback: (file_path) -> [student_id, ...]
        """
        self._attendance_callback = callback
    
    def set_answer_recognition_callback(self, callback: Callable[[np.ndarray, str, dict], dict]):
        """
        답안 인식 콜백 함수 설정
        
        Args:
            callback: (image, student_id, metadata) -> {
                "results": List[AnswerRecognitionResult],
                "fallback_rois": List[AnswerROI]
            }
        """
        self._answer_recognition_callback = callback
    
    def get_answer_metadata(self, exam_code: str) -> Optional[dict]:
        """특정 시험의 정답 메타데이터 반환"""
        return self._answer_metadata.get(exam_code)
    
    def get_student_list(self, exam_code: str) -> List[str]:
        """특정 시험의 학번 리스트 반환"""
        return self._student_id_lists.get(exam_code, [])
    
    def get_next_index(self, exam_code: str) -> int:
        """특정 시험의 다음 index 반환 (1부터 시작, 호출 시 자동 증가)"""
        if exam_code not in self._index_counters:
            self._index_counters[exam_code] = 0
        self._index_counters[exam_code] += 1
        return self._index_counters[exam_code]
    
    def reset_index(self, exam_code: str):
        """특정 시험의 index 카운터 리셋 (출석부 업로드 시 호출)"""
        self._index_counters[exam_code] = 0
        logger.info(f"[INDEX_RESET] {exam_code} index 카운터 리셋")
    
    # =========================================================================
    # 이미지 다운로드
    # =========================================================================
    def download_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        이미지 경로에서 이미지 다운로드
        
        지원 형식:
        - s3://bucket/key
        - S3 키 (bucket은 기본값 사용)
        - http/https URL
        """
        try:
            if image_path.startswith("s3://"):
                # s3://bucket/key 형식
                parts = image_path[5:].split("/", 1)
                bucket = parts[0]
                key = parts[1] if len(parts) > 1 else ""
                response = self.s3.get_object(Bucket=bucket, Key=key)
            elif image_path.startswith("http://") or image_path.startswith("https://"):
                # HTTP URL (presigned URL 등)
                resp = requests.get(image_path, timeout=60)
                resp.raise_for_status()
                pil_image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                return np.array(pil_image)
            else:
                # S3 키로 간주
                response = self.s3.get_object(Bucket=self.s3_bucket, Key=image_path)
            
            image_data = response['Body'].read()
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return np.array(pil_image)
            
        except Exception as e:
            logger.error(f"이미지 다운로드 실패 ({image_path}): {e}")
            return None
    
    def download_file_from_url(self, url: str, suffix: str = ".xlsx") -> Optional[str]:
        """
        URL에서 파일 다운로드하여 임시 파일로 저장
        
        Returns:
            임시 파일 경로 (실패 시 None)
        """
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(resp.content)
                return tmp.name
        except Exception as e:
            logger.error(f"파일 다운로드 실패 ({url}): {e}")
            return None
    
    # =========================================================================
    # S3 업로드
    # =========================================================================
    def upload_image_to_s3(
        self, 
        image: np.ndarray, 
        s3_key: str,
        quality: int = 95
    ) -> bool:
        """이미지를 S3에 업로드"""
        try:
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='image/jpeg'
            )
            logger.info(f"S3 업로드 성공: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"S3 업로드 실패: {e}")
            return False
    
    # =========================================================================
    # SQS 메시지 처리
    # =========================================================================
    def receive_message(self, wait_time_seconds: int = 20) -> Optional[SQSInputMessage]:
        """SQS에서 메시지 하나를 수신 (Long Polling + VisibilityTimeout 최적화)"""
        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=wait_time_seconds,
                # ✅ 중요: AI 처리 시간(모델 로딩 및 추론)을 고려하여 5분(300초) 설정
                # 이 시간 동안은 다른 컨슈머가 이 메시지를 가져가지 못해 중복 수신을 방지합니다.
                VisibilityTimeout=300,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            if not messages:
                return None
            
            msg = messages[0]
            raw_body = msg['Body']
            
            body = json.loads(raw_body)
            # 디버깅: 수신된 모든 메시지 로깅 (Raw body 포함)
            print(f"[SQS_RECEIVE] ✅ 메시지 수신 성공")
            print(f"[SQS_RAW] {raw_body[:500]}")  # 처음 500자만
            logger.info(f"[SQS_RECEIVED] Raw body: {raw_body}")
            print(f"[SQS_RECEIVE] eventType={body.get('eventType')}, examCode={body.get('examCode')}, filename={body.get('filename')}")
            
            # 자신이 보낸 결과 메시지인지 확인 (결과 메시지에는 studentId가 있음)
            if "studentId" in body and body.get("eventType") == EVENT_STUDENT_ID_RECOGNITION:
                logger.info(f"[SQS_DROP] AI가 생성한 결과 메시지를 무시합니다: {body.get('studentId')}")
                print(f"[SQS_DROP] Ignoring own result message for {body.get('studentId')}")
                # ⚠️ 중요: 결과 메시지도 큐에서 삭제해야 FIFO 큐가 블로킹되지 않음
                self.delete_message(msg['ReceiptHandle'])
                print(f"[SQS_DROP] ✅ 결과 메시지 삭제 완료")
                return None

            return SQSInputMessage.from_sqs_message(body, msg['ReceiptHandle'])
            
        except Exception as e:
            print(f"[SQS_RECEIVE] ❌ 메시지 수신 실패: {e}")
            logger.error(f"SQS 메시지 수신 실패: {e}")
            return None
    
    def send_result_message(self, message: SQSOutputMessage, group_id: str = "default") -> Optional[str]:
        """결과 메시지를 결과 큐(AI → BE)로 전송"""
        import uuid
        
        try:
            body = message.to_json()
            # 디버깅: 전송 메시지 로그 (print로 터미널에 직접 출력)
            print(f"[SQS_SEND] 결과 큐로 전송할 JSON:\n{body}")
            logger.info(f"[SQS_SEND] Sending result to {self.result_queue_url}: {body}")
            
            response = self.sqs.send_message(
                QueueUrl=self.result_queue_url,  # ✅ 결과 전용 큐 사용
                MessageBody=body,
                MessageGroupId=group_id,
                MessageDeduplicationId=str(uuid.uuid4())
            )
            msg_id = response.get('MessageId')
            print(f"[SQS_SEND] ✅ 결과 전송 완료 (MessageId: {msg_id})")
            logger.info(f"SQS 결과 전송 완료: {msg_id}")
            return msg_id
        except ClientError as e:
            print(f"[SQS_SEND] ❌ 결과 전송 실패: {e}")
            logger.error(f"SQS 메시지 전송 실패: {e}")
            return None
    
    def send_fallback_message(self, message: AnswerFallbackMessage, group_id: str = "fallback") -> Optional[str]:
        """Fallback 알림 메시지를 Fallback 큐(AI → BE)로 전송"""
        import uuid
        
        if not self.fallback_queue_url:
            print("[SQS_FALLBACK] ⚠️ Fallback 큐 URL이 설정되지 않아 전송 생략")
            return None
        
        try:
            body = message.to_json()
            print(f"[SQS_FALLBACK] Fallback 큐로 전송할 JSON:\n{body}")
            logger.info(f"[SQS_FALLBACK] Sending to {self.fallback_queue_url}: {body}")
            
            response = self.sqs.send_message(
                QueueUrl=self.fallback_queue_url,
                MessageBody=body,
                MessageGroupId=group_id,
                MessageDeduplicationId=str(uuid.uuid4())
            )
            msg_id = response.get('MessageId')
            print(f"[SQS_FALLBACK] ✅ Fallback 전송 완료 (MessageId: {msg_id})")
            logger.info(f"SQS Fallback 전송 완료: {msg_id}")
            return msg_id
        except ClientError as e:
            print(f"[SQS_FALLBACK] ❌ Fallback 전송 실패: {e}")
            logger.error(f"SQS Fallback 메시지 전송 실패: {e}")
            return None
    
    def delete_message(self, receipt_handle: str) -> bool:
        """처리 완료된 메시지 삭제 (입력 큐에서)"""
        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            print(f"[SQS_DELETE] ✅ 입력 큐에서 메시지 삭제 완료")
            return True
        except ClientError as e:
            print(f"[SQS_DELETE] ❌ 메시지 삭제 실패: {e}")
            logger.error(f"SQS 메시지 삭제 실패: {e}")
            return False
    
    # =========================================================================
    # 이벤트 핸들러
    # =========================================================================
    def handle_attendance_upload(self, msg: SQSInputMessage) -> bool:
        """출석부 업로드 이벤트 처리"""
        logger.info(f"[ATTENDANCE_UPLOAD] exam={msg.exam_code}, file={msg.filename}")
        
        if not msg.download_url:
            logger.error(f"[ATTENDANCE_UPLOAD ERROR] downloadUrl이 누락되었습니다. 이 메시지를 큐에서 삭제합니다. 메시지: {msg}")
            return True  # True를 반환하여 큐에서 메시지를 삭제하도록 함
        
        # 1. 파일 다운로드
        tmp_path = self.download_file_from_url(msg.download_url, suffix=".xlsx")
        if not tmp_path:
            return False  # 다운로드 실패는 재시도 가치가 있으므로 False
        
        try:
            # 2. 출석부 파싱
            if self._attendance_callback:
                student_ids = self._attendance_callback(tmp_path)
            else:
                # 기본 파싱
                from id_recog.parsing_xlsx import parsing_xlsx
                student_ids = parsing_xlsx(tmp_path)
            
            # 3. 메모리에 저장
            self._student_id_lists[msg.exam_code] = student_ids
            logger.info(f"[ATTENDANCE_UPLOAD] {msg.exam_code}: {len(student_ids)}명 로드 완료")
            
            # 4. 해당 시험의 index 카운터 리셋 (새 시험 시작)
            self.reset_index(msg.exam_code)
            
            return True
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def handle_student_id_recognition(self, msg: SQSInputMessage) -> bool:
        """이미지 학번 추출 이벤트 처리"""
        
        # =====================================================================
        # NACK 로직 (재시도 횟수 제한 추가)
        # =====================================================================
        student_list = self.get_student_list(msg.exam_code)
        if not student_list:
            # NACK 추적 키 생성
            nack_key = f"{msg.exam_code}:{msg.filename}"
            self._nack_tracker[nack_key] = self._nack_tracker.get(nack_key, 0) + 1
            nack_count = self._nack_tracker[nack_key]
            
            print(f"[NACK] ⏳ 출석부가 아직 로드되지 않음 (exam={msg.exam_code})")
            print(f"[NACK] 재시도 횟수: {nack_count}/{self._max_nack_count}")
            logger.warning(f"[NACK] 출석부 미로드 상태에서 이미지 도착: {msg.exam_code}/{msg.filename} (시도 {nack_count}/{self._max_nack_count})")
            
            # 최대 재시도 횟수 초과 시 메시지 삭제 및 에러 처리
            if nack_count >= self._max_nack_count:
                print(f"[NACK_LIMIT] ❌ 최대 재시도 횟수 초과! 메시지를 DLQ 처리합니다.")
                logger.error(f"[NACK_LIMIT] 최대 재시도 횟수 초과 (출석부 미로드): {msg.exam_code}/{msg.filename}")
                
                # 에러 결과 메시지 전송 (BE에 알림)
                error_result = SQSOutputMessage.create(
                    exam_code=msg.exam_code,
                    student_id=None,  # 실패
                    filename=msg.filename,
                    index=-1  # 에러 표시
                )
                error_result.meta = {
                    "error": "ATTENDANCE_NOT_LOADED",
                    "message": f"출석부가 {self._max_nack_count}회 시도 후에도 로드되지 않았습니다.",
                    "nack_count": nack_count
                }
                self.send_result_message(error_result, group_id=msg.exam_code)
                
                # 추적에서 제거
                del self._nack_tracker[nack_key]
                
                # True 반환 → 메시지 삭제 (더 이상 재시도 안 함)
                return True
            
            # False 반환 → delete_message()가 호출되지 않음 → VisibilityTimeout 후 재시도
            print(f"[NACK] 메시지를 삭제하지 않고 재시도 대기 (VisibilityTimeout 후 자동 재시도)")
            return False
        
        current_index = self.get_next_index(msg.exam_code)
        print(f"[STEP 0/4] 이미지 처리 시작: exam={msg.exam_code}, file={msg.filename}, index={current_index}")
        logger.info(f"[STUDENT_ID_RECOGNITION] exam={msg.exam_code}, file={msg.filename}, index={current_index}")
        
        if not msg.download_url:
            print(f"[ERROR] downloadUrl 누락! 메시지 삭제 처리")
            logger.error(f"[STUDENT_ID_RECOGNITION ERROR] downloadUrl이 누락되었습니다. 이 메시지를 큐에서 삭제합니다. 메시지: {msg}")
            return True  # True를 반환하여 큐에서 메시지를 삭제하도록 함
        
        # 1. 이미지 다운로드 (downloadUrl 사용)
        print(f"[STEP 1/4] 이미지 다운로드 중... URL: {msg.download_url[:100]}...")
        image = self.download_image(msg.download_url)
        if image is None:
            print(f"[STEP 1/4] ❌ 이미지 다운로드 실패!")
            # 실패해도 결과는 전송
            result_msg = SQSOutputMessage.create(
                exam_code=msg.exam_code,
                student_id=None,
                filename=msg.filename,
                index=current_index
            )
            self.send_result_message(result_msg, group_id=msg.exam_code)
            return False
        print(f"[STEP 1/4] ✅ 이미지 다운로드 완료! shape={image.shape}")
        
        # 2. 학번 추출
        print(f"[STEP 2/4] AI 학번 추출 중...")
        student_id = None
        header_image = None
        if self._student_id_callback:
            # student_list는 NACK 체크에서 이미 조회됨
            print(f"[STEP 2/4] 학번 리스트 {len(student_list)}명 로드됨")
            result = self._student_id_callback(image, student_list)
            student_id = result.get("student_id")
            header_image = result.get("header_image")  # 헤더 이미지 추출
        print(f"[STEP 2/4] ✅ AI 추출 완료! student_id={student_id}")
        
        # 3. 결과 메시지 전송
        print(f"[STEP 3/4] SQS 결과 메시지 전송 중...")
        result_msg = SQSOutputMessage.create(
            exam_code=msg.exam_code,
            student_id=student_id,
            filename=msg.filename,
            index=current_index
        )
        self.send_result_message(result_msg, group_id=msg.exam_code)
        print(f"[STEP 3/4] ✅ 결과 전송 완료!")
        
        # 4. S3 업로드
        # - 성공 시: original/{exam_code}/{student_id}/{filename} (원본 이미지)
        # - 실패 시: 
        #    1. header/{exam_code}/unknown_id/{filename} (헤더 확인용)
        #    2. original/{exam_code}/unknown_id/{filename} (나중에 답안 인식 Fallback용 원본)
        
        if student_id:
            s3_key = f"original/{msg.exam_code}/{student_id}/{msg.filename}"
            print(f"[STEP 4/4] S3 업로드 중 (original)... key={s3_key}")
            self.upload_image_to_s3(image, s3_key)
        else:
            # 1. 헤더 이미지 업로드 (프론트엔드 확인용)
            header_key = f"header/{msg.exam_code}/{UNKNOWN_ID}/{msg.filename}"
            upload_header = header_image if header_image is not None else image
            print(f"[STEP 4/4] S3 업로드 중 (header)... key={header_key}")
            self.upload_image_to_s3(upload_header, header_key)
            
            # 2. 원본 이미지 업로드 (unknown_id 폴더에 저장 -> 추후 Fallback 시 사용)
            original_unknown_key = f"original/{msg.exam_code}/{UNKNOWN_ID}/{msg.filename}"
            print(f"[STEP 4/4] S3 업로드 중 (original_unknown)... key={original_unknown_key}")
            self.upload_image_to_s3(image, original_unknown_key)
        
        print(f"[STEP 4/4] ✅ S3 업로드 완료!")
        
        print(f"[DONE] 이미지 처리 완료: {msg.filename} → {student_id or 'unknown_id'}")
        
        # 성공 시 NACK 트래커에서 제거 (메모리 정리)
        nack_key = f"{msg.exam_code}:{msg.filename}"
        if nack_key in self._nack_tracker:
            del self._nack_tracker[nack_key]
        
        return True
    
    def process_message(self, msg: SQSInputMessage) -> bool:
        """메시지 타입에 따라 적절한 핸들러 호출"""
        print(f"[SQS_PROCESSING] event_type={msg.event_type}, exam_code={msg.exam_code}")
        if msg.event_type == EVENT_ATTENDANCE_UPLOAD:
            return self.handle_attendance_upload(msg)
        elif msg.event_type == EVENT_STUDENT_ID_RECOGNITION:
            return self.handle_student_id_recognition(msg)
        elif msg.event_type == EVENT_ANSWER_METADATA_UPLOAD:
            return self.handle_answer_metadata_upload(msg)
        elif msg.event_type == EVENT_ANSWER_RECOGNITION:
            return self.handle_answer_recognition(msg)
        elif msg.event_type == EVENT_GRADING_COMPLETE:
            return self.handle_grading_complete(msg)
        else:
            print(f"[SQS_WARNING] 알 수 없는 이벤트 타입: {msg.event_type}")
            logger.warning(f"알 수 없는 이벤트 타입: {msg.event_type}")
            return False
    
    # =========================================================================
    # 답안 인식 이벤트 핸들러
    # =========================================================================
    # =========================================================================
    # 답안 인식 이벤트 핸들러
    # =========================================================================
    def handle_answer_metadata_upload(self, msg: SQSInputMessage) -> bool:
        """정답 메타데이터 업로드 이벤트 처리 + 배치 답안 인식 실행"""
        logger.info(f"[ANSWER_METADATA_UPLOAD] exam={msg.exam_code}, file={msg.filename}")
        print(f"[ANSWER_METADATA_UPLOAD] 정답 메타데이터 다운로드 중...")
        
        if not msg.download_url:
            logger.error(f"[ANSWER_METADATA_UPLOAD ERROR] downloadUrl 누락")
            return True  # 삭제 처리
        
        try:
            # 1. JSON 파일 다운로드
            import requests
            resp = requests.get(msg.download_url, timeout=60)
            resp.raise_for_status()
            
            metadata = resp.json()
            
            # 2. 메모리에 저장
            self._answer_metadata[msg.exam_code] = metadata
            logger.info(f"[ANSWER_METADATA_UPLOAD] {msg.exam_code}: 메타데이터 로드 완료")
            print(f"[ANSWER_METADATA_UPLOAD] ✅ 메타데이터 로드 완료: {len(metadata.get('questions', []))}개 문제")
            
            # 3. 배치 답안 인식 시작 (비동기 권장이지만, 현재는 동기 처리)
            print(f"[ANSWER_METADATA_UPLOAD] 🚀 배치 답안 인식 트러거됨 (exam={msg.exam_code})")
            threading.Thread(
                target=self.process_batch_answer_recognition,
                args=(msg.exam_code, metadata),
                daemon=True
            ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"[ANSWER_METADATA_UPLOAD ERROR] {e}")
            print(f"[ANSWER_METADATA_UPLOAD] ❌ 실패: {e}")
            return False

    def process_batch_answer_recognition(self, exam_code: str, metadata: dict):
        """
        [배치 처리] 해당 시험의 모든 이미지를 S3에서 가져와 답안 인식 수행
        
        Input S3: original/{exam_code}/{student_id}/{filename}
        
        Logic:
        1. original/{exam_code}/ 하위 모든 이미지 순회
        2. student_id 폴더가 'unknown_id'인 경우:
           - 메타데이터의 'images' 리스트(fallback info)를 확인
           - filename이 매칭되면 해당 studentId로 인식 수행
           - 매칭 안 되면 스킵
        3. 그 외 (정상 student_id)인 경우:
           - 해당 student_id로 인식 수행
        
        Output S3:
          - Result: answer/{exam_code}/{student_id}/result.json
          - Fallback IMG: answer/{exam_code}/{student_id}/{q}/{sub_q}/{filename}
        """
        print(f"[BATCH] 🏁 배치 작업 시작: {exam_code}")
        
        # 0. Fallback 매핑 정보 생성 (unknown_id 처리용)
        # metadata = { "examCode": "...", "images": [ {"fileName": "...", "studentId": "..."}, ... ] }
        fallback_map = {} # filename -> studentId
        if "images" in metadata and isinstance(metadata["images"], list):
            for img_info in metadata["images"]:
                fname = img_info.get("fileName")
                sid = img_info.get("studentId")
                if fname and sid:
                    fallback_map[fname] = sid
        
        prefix = f"original/{exam_code}/"
        paginator = self.s3.get_paginator('list_objects_v2')
        
        processed_count = 0
        error_count = 0
        
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # key format: original/{exam_code}/{student_id}/{filename}
                    parts = key.split('/')
                    if len(parts) < 4:
                        continue
                        
                    folder_student_id = parts[2]
                    filename = parts[3]
                    
                    target_student_id = folder_student_id
                    
                    # Unknown ID 처리 로직
                    if folder_student_id == "unknown_id":
                        if filename in fallback_map:
                            target_student_id = fallback_map[filename]
                            print(f"[BATCH] 🔄 Fallback 매핑: {filename} -> {target_student_id}")
                        else:
                            # 매핑 정보가 없으면 스킵 (혹은 로그)
                            # print(f"[BATCH] ⚠️ Unknown image skipped (no mapping): {filename}")
                            continue
                    
                    # 이미지 다운로드
                    image = self.download_image(key)
                    if image is None:
                        print(f"[BATCH] ❌ 이미지 다운로드 실패: {key}")
                        error_count += 1
                        continue
                        
                    # 콜백 실행 (답안 인식 + Fallback 업로드)
                    if self._answer_recognition_callback:
                        try:
                            # filename 인자 추가 전달
                            # 주의: target_student_id를 전달해야 함
                            result = self._answer_recognition_callback(image, target_student_id, metadata, filename)
                            
                            # 결과 포맷팅 및 S3 업로드 (result.json)
                            self._format_and_upload_result(exam_code, target_student_id, result, metadata)
                            processed_count += 1
                            
                            if processed_count % 10 == 0:
                                print(f"[BATCH] 진행 중... {processed_count}건 완료")
                                
                        except Exception as e:
                            print(f"[BATCH] ❌ 처리 중 에러 ({key}): {e}")
                            import traceback
                            traceback.print_exc()
                            error_count += 1
            
            print(f"[BATCH] ✅ 배치 작업 완료: 성공 {processed_count}, 실패 {error_count}")
            
        except Exception as e:
            print(f"[BATCH] ❌ 배치 루프 에러: {e}")
            import traceback
            traceback.print_exc()

    def _format_and_upload_result(self, exam_code: str, student_id: str, result_data: dict, metadata: dict):
        """
        결과 JSON 포맷팅 및 S3 업로드
        
        Metadata 필드: questionId, questionNumber, questionType, answer, answerCount, point
        Result 필드: questionNumber, subQuestionNumber, recAnswer, confidence, rawText
        """
        raw_results = result_data.get("results", [])
        
        formatted_answers = []
        
        # 메타데이터 questions 리스트 (검색 최적화를 위해 dict로 변환하면 좋지만, 개별 순회도 무방)
        meta_questions = metadata.get("questions", [])
        
        for item in raw_results:
            # item is AnswerRecognitionResult object from schemas.py
            
            # 1. 해당 문제의 메타 정보 찾기
            q_meta = next((q for q in meta_questions if q['questionNumber'] == item.question_number), None)
            
            # 메타데이터가 없는 문제는 스킵할지 포함할지 결정 (여기선 포함하되 기본값 사용)
            question_id = q_meta.get("questionId", 0) if q_meta else 0
            question_type = q_meta.get("questionType", "objective") if q_meta else item.scoring_type.value
            
            # answer (정답 값): 메타에서 가져옴
            answer_val = q_meta.get("answer", "") if q_meta else ""
            
            # point (배점): 메타에서
            point = q_meta.get("point", 0.0) if q_meta else 0.0
            
            # answerCount: 메타에서
            answer_count = q_meta.get("answerCount", 1) if q_meta else 1
            
            # 2. 인식 결과 Parsing
            rec_str = item.rec_answer or ""
            values = []
            
            # 객관식인 경우 숫자 추출, 주관식인 경우 텍스트 그대로 등 처리
            # (요청 예시에는 values: [6] 처럼 숫자 리스트로 되어 있음 -> 객관식 가정)
            # 만약 questionType이 SUBJECTIVE라면 values 처리가 다를 수 있음
            
            if rec_str:
                import re
                nums = re.findall(r'\d+', rec_str)
                values = [int(n) for n in nums]
            
            raw_text = item.meta.get("raw_ocr_text", "")
            if not raw_text and rec_str:
                raw_text = rec_str
            
            formatted_item = {
                "questionNumber": item.question_number,
                "subQuestionNumber": item.sub_question_number or 0,
                "point": point,             # 메타 그대로
                "answerCount": answer_count, # 메타 그대로
                "answerType": question_type, # 메타의 questionType 사용 (필드명은 answerType 유지, 요청 예시 따름)
                "recAnswer": {
                    "values": values,
                    "confidence": [round(item.confidence, 4)] if item.confidence else [],
                    "rawText": raw_text
                }
            }
            formatted_answers.append(formatted_item)
        
        final_json = {
            "examCode": exam_code,
            "studentId": student_id,
            "total": 100, # TODO: 실제 총점 계산 로직 필요 시 추가
            "status": "completed",
            "eventType": "ANSWER_RECOGNITION",
            "answers": formatted_answers
        }
        
        # S3 업로드: answer/{exam code}/{학번}/result.json
        s3_key = f"answer/{exam_code}/{student_id}/result.json"
        
        try:
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(final_json, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            # print(f"  [UPLOAD] 결과 JSON 업로드: {s3_key}")
        except Exception as e:
            print(f"  [UPLOAD FAIL] 결과 JSON 업로드 실패: {s3_key}, {e}")

    def handle_answer_recognition(self, msg: SQSInputMessage) -> bool:
        """답안 인식 이벤트 처리 (개별 메시지)"""
        body = {
            "eventType": msg.event_type,
            "examCode": msg.exam_code,
            "filename": msg.filename,
            "downloadUrl": msg.download_url,
            "studentId": ""  # SQSInputMessage에는 없음, 추후 수정 필요
        }
        answer_msg = AnswerRecognitionInputMessage.from_sqs_message(body, msg.receipt_handle)
        
        logger.info(f"[ANSWER_RECOGNITION] exam={msg.exam_code}, file={msg.filename}")
        print(f"[ANSWER_RECOGNITION] 답안 인식 시작: {msg.filename}")
        
        # 메타데이터 로드 확인
        metadata = self.get_answer_metadata(msg.exam_code)
        if not metadata:
            print(f"[NACK] ⏳ 정답 메타데이터가 아직 로드되지 않음 (exam={msg.exam_code})")
            return False  # NACK → 재시도
        
        # 이미지 다운로드
        image = self.download_image(msg.download_url)
        if image is None:
            print(f"[ANSWER_RECOGNITION] ❌ 이미지 다운로드 실패")
            return False
        
        # 콜백 호출
        if hasattr(self, '_answer_recognition_callback') and self._answer_recognition_callback:
            try:
                result = self._answer_recognition_callback(
                    image, 
                    answer_msg.student_id,
                    metadata
                )
                
                # 결과 메시지 생성
                results = result.get("results", [])
                fallback_rois = result.get("fallback_rois", [])
                
                # AnswerRecognitionResultItem 리스트로 변환
                result_items = []
                for r in results:
                    item = AnswerRecognitionResultItem(
                        question_number=r.question_number,
                        sub_question_number=r.sub_question_number or 0,
                        rec_answer=r.rec_answer,
                        confidence=r.confidence,
                        is_fallback=r.confidence < 0.7,
                        s3_key=getattr(r, 's3_key', None)
                    )
                    result_items.append(item)
                
                # 결과 메시지 전송
                output_msg = AnswerRecognitionOutputMessage.create(
                    exam_code=msg.exam_code,
                    student_id=answer_msg.student_id,
                    filename=msg.filename,
                    results=result_items
                )
                
                self.send_result_message_generic(output_msg, group_id=msg.exam_code)
                print(f"[ANSWER_RECOGNITION] ✅ 완료: {len(result_items)}개 문제 인식, {output_msg.fallback_count}개 Fallback")
                
                return True
                
            except Exception as e:
                logger.error(f"[ANSWER_RECOGNITION ERROR] {e}")
                print(f"[ANSWER_RECOGNITION] ❌ 콜백 실행 실패: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"[ANSWER_RECOGNITION] ⚠️ 콜백이 설정되지 않음")
            return True  # 콜백 없으면 그냥 통과
    
    def handle_grading_complete(self, msg: SQSInputMessage) -> bool:
        """채점 완료 요청 처리"""
        logger.info(f"[GRADING_COMPLETE] exam={msg.exam_code}")
        print(f"[GRADING_COMPLETE] 채점 요청 수신: {msg.exam_code}")
        
        # TODO: 실제 채점 로직 구현
        # 1. Fallback 수정값 병합
        # 2. 정답 메타데이터와 비교
        # 3. 점수 계산
        # 4. 결과 메시지 전송
        
        print(f"[GRADING_COMPLETE] ⚠️ 채점 로직 미구현 (TODO)")
        return True
    
    def send_result_message_generic(self, message, group_id: str = "default") -> Optional[str]:
        """범용 결과 메시지 전송 (AnswerRecognitionOutputMessage 등)"""
        import uuid
        
        try:
            body = message.to_json()
            print(f"[SQS_SEND] 결과 전송: {message.event_type}")
            
            response = self.sqs.send_message(
                QueueUrl=self.result_queue_url,
                MessageBody=body,
                MessageGroupId=group_id,
                MessageDeduplicationId=str(uuid.uuid4())
            )
            msg_id = response.get('MessageId')
            print(f"[SQS_SEND] ✅ 전송 완료 (MessageId: {msg_id})")
            return msg_id
        except ClientError as e:
            print(f"[SQS_SEND] ❌ 전송 실패: {e}")
            logger.error(f"SQS 메시지 전송 실패: {e}")
            return None

    
    # =========================================================================
    # 워커 루프
    # =========================================================================
    def _get_queue_status(self) -> tuple:
        """큐의 현재 상태 조회 (대기, 처리중)"""
        try:
            attrs = self.sqs.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )['Attributes']
            available = int(attrs['ApproximateNumberOfMessages'])
            in_flight = int(attrs['ApproximateNumberOfMessagesNotVisible'])
            return (available, in_flight)
        except Exception as e:
            print(f"[SQS_STATUS_ERROR] 큐 상태 조회 실패: {e}")
            return (-1, -1)
    
    def _worker_loop(self):
        """워커 메인 루프"""
        import datetime
        
        with open("debug_worker.log", "a") as f:
            f.write(f"[{time.ctime()}] SQS Worker Loop Started\n")
        print(f"[SQS_LOOP] SQS Worker 시작 - 메시지 폴링 대기 중...")
        print(f"[SQS_LOOP] 입력 큐: {self.queue_url}")
        print(f"[SQS_LOOP] 결과 큐: {self.result_queue_url}")
        logger.info(f"SQS Worker 시작 - 입력={self.queue_url}, 결과={self.result_queue_url}")
        
        poll_count = 0
        
        while self._running:
            try:
                poll_count += 1
                timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # =========================================================
                # [POLL_START] 폴링 시작 전 상태
                # =========================================================
                before_available, before_in_flight = self._get_queue_status()
                print(f"\n{'='*60}")
                print(f"[POLL #{poll_count}] {timestamp} 폴링 시작")
                print(f"[POLL_BEFORE] 대기: {before_available}, 처리중: {before_in_flight}")
                
                # =========================================================
                # [POLL_WAIT] Long Polling 수행 (최대 20초)
                # =========================================================
                print(f"[POLL_WAIT] Long Polling 대기 중... (최대 20초)")
                msg = self.receive_message(wait_time_seconds=20)
                
                poll_end_timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                
                # =========================================================
                # [POLL_RESULT] 폴링 결과
                # =========================================================
                if msg is None:
                    print(f"[POLL_RESULT] {poll_end_timestamp} 메시지 없음 (타임아웃 또는 빈 큐)")
                else:
                    print(f"[POLL_RESULT] {poll_end_timestamp} ✅ 메시지 수신!")
                    print(f"[POLL_RESULT] event={msg.event_type}, exam={msg.exam_code}, file={msg.filename}")
                
                # =========================================================
                # [POLL_AFTER] 폴링 후 상태 비교
                # =========================================================
                after_available, after_in_flight = self._get_queue_status()
                delta_available = after_available - before_available
                delta_in_flight = after_in_flight - before_in_flight
                
                print(f"[POLL_AFTER] 대기: {after_available} ({delta_available:+d}), 처리중: {after_in_flight} ({delta_in_flight:+d})")
                
                # ⚠️ 이상 감지: AI가 메시지를 안 받았는데 처리중이 증가?
                if msg is None and delta_in_flight > 0:
                    print(f"[⚠️ ANOMALY] AI가 receive 안 했는데 처리중이 +{delta_in_flight} 증가!")
                    print(f"[⚠️ ANOMALY] 다른 컨슈머(BE/Lambda)가 폴링 중일 가능성 높음")
                
                # AI가 1개 받았는데 처리중이 2개 이상 증가?
                if msg is not None and delta_in_flight > 1:
                    print(f"[⚠️ ANOMALY] AI가 1개 receive 했는데 처리중이 +{delta_in_flight} 증가!")
                    print(f"[⚠️ ANOMALY] 동시에 다른 컨슈머도 receive 했을 가능성")
                
                print(f"{'='*60}")
                
                if msg is None:
                    continue
                
                # =========================================================
                # 메시지 처리
                # =========================================================
                success = self.process_message(msg)
                
                # 처리 완료 시 메시지 삭제 (ACK), 실패 시 삭제 안 함 (NACK → 재시도)
                if success and msg.receipt_handle:
                    print(f"[SQS_ACK] 처리 성공 → 메시지 삭제 진행")
                    self.delete_message(msg.receipt_handle)
                elif not success:
                    print(f"[SQS_NACK] 처리 실패/보류 → 메시지 삭제 안 함 (VisibilityTimeout 후 재시도)")
                    
            except Exception as e:
                print(f"[SQS_WORKER_ERROR] Worker 에러: {e}")
                logger.error(f"Worker 에러: {e}")
                time.sleep(5)
        
        logger.info("SQS Worker 종료")
    
    def start(self):
        """워커 백그라운드 실행 시작"""
        if self._running:
            logger.warning("Worker가 이미 실행 중입니다.")
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="SQS-Worker-Thread"
        )
        self._worker_thread.start()
        logger.info("SQS Worker가 백그라운드에서 시작되었습니다.")
    
    def stop(self):
        """워커 종료"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=25)
        logger.info("SQS Worker가 종료되었습니다.")
    
    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================
_worker_instance: Optional[SQSWorker] = None


def get_sqs_worker() -> Optional[SQSWorker]:
    """SQS Worker 싱글톤 인스턴스 반환"""
    global _worker_instance
    return _worker_instance


def init_sqs_worker(
    queue_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str = "ap-northeast-2",
    s3_bucket: str = "mlpa-gradi",
    result_queue_url: str = None,
    fallback_queue_url: str = None
) -> SQSWorker:
    """SQS Worker 초기화 및 싱글톤 설정"""
    global _worker_instance
    _worker_instance = SQSWorker(
        queue_url=queue_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        s3_bucket=s3_bucket,
        result_queue_url=result_queue_url,
        fallback_queue_url=fallback_queue_url
    )
    return _worker_instance

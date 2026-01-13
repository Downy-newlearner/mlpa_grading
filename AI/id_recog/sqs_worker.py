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

from sqs_schemas import (
    SQSInputMessage, 
    SQSOutputMessage,
    EVENT_ATTENDANCE_UPLOAD,
    EVENT_STUDENT_ID_RECOGNITION,
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
    """
    
    def __init__(
        self,
        queue_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "ap-northeast-2",
        s3_bucket: str = "mlpa-gradi",
        result_queue_url: str = None  # AI → BE 결과 전송용 큐 (None이면 queue_url 사용)
    ):
        self.queue_url = queue_url  # BE → AI 입력 큐
        self.result_queue_url = result_queue_url if result_queue_url else queue_url  # AI → BE 결과 큐
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
                from parsing_xlsx import parsing_xlsx
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
        # NACK 로직: 출석부가 아직 로드되지 않았으면 재시도
        # =====================================================================
        student_list = self.get_student_list(msg.exam_code)
        if not student_list:
            print(f"[NACK] ⏳ 출석부가 아직 로드되지 않음 (exam={msg.exam_code})")
            print(f"[NACK] 메시지를 삭제하지 않고 재시도 대기 (VisibilityTimeout 후 자동 재시도)")
            logger.warning(f"[NACK] 출석부 미로드 상태에서 이미지 도착: {msg.exam_code}/{msg.filename}")
            # False 반환 → delete_message()가 호출되지 않음 → 5분 후 재시도
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
        # - 실패 시: header/{exam_code}/unknown_id/{filename} (헤더 이미지)
        if student_id:
            s3_key = f"original/{msg.exam_code}/{student_id}/{msg.filename}"
            upload_image = image  # 원본 이미지 업로드
            print(f"[STEP 4/4] S3 업로드 중 (original)... key={s3_key}")
        else:
            s3_key = f"header/{msg.exam_code}/{UNKNOWN_ID}/{msg.filename}"
            # 헤더 이미지가 있으면 헤더를, 없으면 원본을 업로드
            upload_image = header_image if header_image is not None else image
            print(f"[STEP 4/4] S3 업로드 중 (header)... key={s3_key}")
        
        self.upload_image_to_s3(upload_image, s3_key)
        print(f"[STEP 4/4] ✅ S3 업로드 완료!")
        
        print(f"[DONE] 이미지 처리 완료: {msg.filename} → {student_id or 'unknown_id'}")
        return True
    
    def process_message(self, msg: SQSInputMessage) -> bool:
        """메시지 타입에 따라 적절한 핸들러 호출"""
        print(f"[SQS_PROCESSING] event_type={msg.event_type}, exam_code={msg.exam_code}")
        if msg.event_type == EVENT_ATTENDANCE_UPLOAD:
            return self.handle_attendance_upload(msg)
        elif msg.event_type == EVENT_STUDENT_ID_RECOGNITION:
            return self.handle_student_id_recognition(msg)
        else:
            print(f"[SQS_WARNING] 알 수 없는 이벤트 타입: {msg.event_type}")
            logger.warning(f"알 수 없는 이벤트 타입: {msg.event_type}")
            return False
    
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
    result_queue_url: str = None
) -> SQSWorker:
    """SQS Worker 초기화 및 싱글톤 설정"""
    global _worker_instance
    _worker_instance = SQSWorker(
        queue_url=queue_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
        s3_bucket=s3_bucket,
        result_queue_url=result_queue_url
    )
    return _worker_instance

"""
sqs_client.py - AWS SQS 클라이언트

SQS 메시지 전송 및 수신을 담당합니다.
환경 변수(.env)에서 AWS 자격 증명을 로드합니다.
"""

import os
import json
import logging
import boto3
import uuid
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

from sqs_schemas import SQSImageUploadMessage

# 로거 설정
logger = logging.getLogger(__name__)

# .env 파일 로드 (같은 디렉토리 기준)
load_dotenv()

# =============================================================================
# AWS 설정 (환경변수에서 로드)
# =============================================================================
# 주의: 실제 배포 시에는 반드시 환경 변수나 IAM Role을 사용하세요.
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")


class SQSClient:
    """AWS SQS 클라이언트"""
    
    def __init__(self):
        try:
            self.sqs = boto3.client(
                'sqs',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            self.queue_url = SQS_QUEUE_URL
            logger.info(f"SQS 클라이언트 초기화 완료: {self.queue_url}")
        except Exception as e:
            logger.error(f"SQS 클라이언트 초기화 실패: {e}")
            raise

    def send_result_message(self, message: SQSImageUploadMessage, group_id: str = "default") -> Optional[str]:
        """
        처리 결과 메시지를 SQS로 전송합니다.
        
        Args:
            message: 전송할 메시지 객체 (SQSImageUploadMessage)
            group_id: FIFO 큐용 메시지 그룹 ID
            
        Returns:
            MessageId (성공 시), None (실패 시)
        """
        if not self.queue_url:
            logger.error("SQS Queue URL이 설정되지 않았습니다.")
            return None

        try:
            # FIFO 큐 요구사항: MessageDeduplicationId 필요 (ContentBasedDeduplication 미설정 시)
            deduplication_id = str(uuid.uuid4())
            
            response = self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=message.to_json(),
                MessageGroupId=group_id,
                MessageDeduplicationId=deduplication_id
            )
            
            msg_id = response.get('MessageId')
            logger.info(f"SQS 메시지 전송 성공: {msg_id}")
            return msg_id
            
        except ClientError as e:
            logger.error(f"SQS 메시지 전송 실패: {e}")
            return None

    def receive_messages(self, max_number: int = 1, wait_time_seconds: int = 10) -> List[Dict[str, Any]]:
        """
        SQS에서 메시지를 수신합니다 (Long Polling).
        
        Args:
            max_number: 최대 수신 메시지 수 (1~10)
            wait_time_seconds: Long Polling 대기 시간 (0~20)
            
        Returns:
            수신된 메시지 리스트 (없으면 빈 리스트)
        """
        if not self.queue_url:
            return []

        try:
            response = self.sqs.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_number,
                WaitTimeSeconds=wait_time_seconds,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            if messages:
                logger.info(f"{len(messages)}개의 메시지 수신")
            
            return messages
            
        except ClientError as e:
            logger.error(f"SQS 메시지 수신 실패: {e}")
            return []

    def delete_message(self, receipt_handle: str) -> bool:
        """
        처리가 완료된 메시지를 큐에서 삭제합니다.
        
        Args:
            receipt_handle: 메시지 수신 시 받은 핸들
            
        Returns:
            성공 여부
        """
        try:
            self.sqs.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            logger.info("SQS 메시지 삭제 성공")
            return True
        except ClientError as e:
            logger.error(f"SQS 메시지 삭제 실패: {e}")
            return False


# =============================================================================
# 테스트 코드
# =============================================================================
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    print("=== SQS 연결 및 전송 테스트 ===")
    
    # 1. 클라이언트 생성
    try:
        client = SQSClient()
        
        # 2. 테스트 메시지 생성
        msg = SQSImageUploadMessage.create(
            exam_code="TEST_EXAM_SQS_NEW",
            student_id="20211234",
            filename="test_page_001.jpg",
            index=0,
            event_type="id_rec",
            total=40
        )
        
        print(f"전송할 메시지: {msg.to_json()}")
        
        # 3. 메시지 전송
        # 주의: 실제 큐에 메시지가 쌓이게 됩니다.
        msg_id = client.send_result_message(msg, group_id="test-group")
        
        if msg_id:
            print(f"✅ 테스트 성공! Message ID: {msg_id}")
        else:
            print("❌ 테스트 실패")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

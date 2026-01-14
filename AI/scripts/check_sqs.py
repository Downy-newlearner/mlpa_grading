#!/usr/bin/env python3
"""
check_sqs.py - SQS 큐 상태 확인 스크립트

사용법: python check_sqs.py
"""

import os
import json
from dotenv import load_dotenv
import boto3

load_dotenv()

# AWS 자격증명
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")
QUEUE_URL = os.getenv("SQS_QUEUE_URL")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, QUEUE_URL]):
    print("❌ .env 파일에 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, SQS_QUEUE_URL 설정 필요")
    exit(1)

sqs = boto3.client(
    'sqs',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print("=" * 60)
print("SQS 큐 상태 확인")
print("=" * 60)
print(f"Queue URL: {QUEUE_URL}")
print()

# 1. 큐 속성 조회
try:
    attrs = sqs.get_queue_attributes(
        QueueUrl=QUEUE_URL,
        AttributeNames=['All']
    )['Attributes']
    
    available = int(attrs.get('ApproximateNumberOfMessages', 0))
    in_flight = int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0))
    delayed = int(attrs.get('ApproximateNumberOfMessagesDelayed', 0))
    
    print(f"📊 큐 상태:")
    print(f"   - 대기 중 (Available): {available}개")
    print(f"   - 처리 중 (In-Flight): {in_flight}개")
    print(f"   - 지연 중 (Delayed): {delayed}개")
    print()
except Exception as e:
    print(f"❌ 큐 상태 조회 실패: {e}")
    exit(1)

# 2. 대기 중인 메시지 미리보기 (최대 10개)
if available > 0:
    print("=" * 60)
    print(f"📨 대기 중인 메시지 미리보기 (최대 10개)")
    print("=" * 60)
    
    try:
        # VisibilityTimeout=0으로 하면 메시지를 가져오지만 바로 다시 보이게 됨
        # 하지만 FIFO 큐에서는 이게 위험할 수 있으므로, 짧은 시간(1초)으로 설정
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=1,
            VisibilityTimeout=1,  # 1초 후 다시 보이게
            MessageAttributeNames=['All']
        )
        
        messages = response.get('Messages', [])
        
        if not messages:
            print("   (메시지 없음)")
        else:
            for i, msg in enumerate(messages, 1):
                body = json.loads(msg['Body'])
                print(f"\n[{i}] eventType: {body.get('eventType')}")
                print(f"    examCode: {body.get('examCode')}")
                print(f"    filename: {body.get('filename')}")
                print(f"    timestamp: {body.get('timestamp')}")
                
    except Exception as e:
        print(f"❌ 메시지 조회 실패: {e}")
else:
    print("📭 대기 중인 메시지 없음")

print()
print("=" * 60)
print("✅ 확인 완료")
print("=" * 60)
print()
print("💡 참고: '처리 중(In-Flight)' 메시지는 다른 컨슈머가 가져간 상태라")
print("   내용을 직접 확인할 수 없습니다. VisibilityTimeout 후 다시 나타납니다.")

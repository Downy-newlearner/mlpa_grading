"""
debug_queue.py - SQS 큐 메시지 디버깅 스크립트

큐에 있는 메시지들의 내용과 MessageGroupId를 확인합니다.
⚠️ 주의: 이 스크립트는 메시지를 receive하므로 VisibilityTimeout(300초) 동안 
        다른 컨슈머에게 안 보이게 됩니다!
"""

import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

queue_url = os.environ.get("SQS_QUEUE_URL")
aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")

sqs = boto3.client(
    'sqs',
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=region
)

print("=" * 70)
print("SQS 큐 메시지 디버깅")
print("=" * 70)

# 큐 상태 확인
attrs = sqs.get_queue_attributes(
    QueueUrl=queue_url,
    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
)['Attributes']

print(f"\n📊 큐 상태:")
print(f"   대기 (Available): {attrs['ApproximateNumberOfMessages']}")
print(f"   처리중 (In-flight): {attrs['ApproximateNumberOfMessagesNotVisible']}")

# 메시지 peek (VisibilityTimeout=5초로 짧게 설정)
print(f"\n📨 메시지 Peek (VisibilityTimeout=5초):")
print("-" * 70)

for i in range(10):  # 최대 10개까지 확인
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=1,  # 1초만 대기
        VisibilityTimeout=5,  # 5초 후 다시 보임
        AttributeNames=['All'],
        MessageAttributeNames=['All']
    )
    
    messages = response.get('Messages', [])
    if not messages:
        print(f"\n✅ 더 이상 메시지 없음 (총 {i}개 확인)")
        break
    
    msg = messages[0]
    body = json.loads(msg['Body'])
    
    # MessageGroupId 추출
    msg_group_id = msg.get('Attributes', {}).get('MessageGroupId', 'N/A')
    
    print(f"\n[메시지 #{i+1}]")
    print(f"  MessageGroupId: {msg_group_id}")
    print(f"  eventType: {body.get('eventType')}")
    print(f"  examCode: {body.get('examCode')}")
    print(f"  filename: {body.get('filename')}")
    
    if body.get('eventType') == 'ATTENDANCE_UPLOAD':
        print(f"  📋 출석부 메시지!")
    elif body.get('eventType') == 'STUDENT_ID_RECOGNITION':
        print(f"  🖼️ 이미지 메시지!")

print("\n" + "=" * 70)
print("⚠️ 위 메시지들은 5초 후 다시 '대기' 상태로 돌아갑니다.")
print("=" * 70)

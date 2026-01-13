#!/usr/bin/env python3
"""
monitor_dlq.py - DLQ 모니터링 및 메시지 조회 스크립트

DLQ에 쌓인 실패 메시지를 확인하고 분석합니다.

사용법:
    python scripts/monitor_dlq.py              # 상태 조회
    python scripts/monitor_dlq.py --peek       # 메시지 미리보기 (삭제 안 함)
    python scripts/monitor_dlq.py --purge      # DLQ 비우기 (주의!)
    python scripts/monitor_dlq.py --redrive    # 메시지를 메인 큐로 재전송
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import boto3
from botocore.exceptions import ClientError


def get_dlq_status(sqs_client, dlq_url: str) -> dict:
    """DLQ 상태를 조회합니다."""
    response = sqs_client.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=[
            'ApproximateNumberOfMessages',
            'ApproximateNumberOfMessagesNotVisible',
            'ApproximateNumberOfMessagesDelayed',
            'CreatedTimestamp',
            'LastModifiedTimestamp'
        ]
    )
    
    attrs = response.get('Attributes', {})
    
    return {
        "queue_url": dlq_url,
        "messages_available": int(attrs.get('ApproximateNumberOfMessages', 0)),
        "messages_in_flight": int(attrs.get('ApproximateNumberOfMessagesNotVisible', 0)),
        "messages_delayed": int(attrs.get('ApproximateNumberOfMessagesDelayed', 0)),
        "created": datetime.fromtimestamp(int(attrs.get('CreatedTimestamp', 0))).isoformat(),
        "last_modified": datetime.fromtimestamp(int(attrs.get('LastModifiedTimestamp', 0))).isoformat()
    }


def peek_messages(sqs_client, dlq_url: str, max_messages: int = 10) -> list:
    """DLQ 메시지를 미리봅니다 (삭제하지 않음)."""
    messages = []
    
    # VisibilityTimeout을 0으로 설정하면 다른 컨슈머도 볼 수 있음
    # 하지만 AWS에서는 최소 1초이므로 짧게 설정
    response = sqs_client.receive_message(
        QueueUrl=dlq_url,
        MaxNumberOfMessages=min(max_messages, 10),
        WaitTimeSeconds=1,
        VisibilityTimeout=1,  # 1초 후 다시 visible
        AttributeNames=['All'],
        MessageAttributeNames=['All']
    )
    
    for msg in response.get('Messages', []):
        try:
            body = json.loads(msg['Body'])
        except json.JSONDecodeError:
            body = msg['Body']
        
        messages.append({
            'message_id': msg['MessageId'],
            'body': body,
            'receive_count': msg.get('Attributes', {}).get('ApproximateReceiveCount', 'N/A'),
            'first_receive': msg.get('Attributes', {}).get('ApproximateFirstReceiveTimestamp', 'N/A')
        })
    
    return messages


def purge_dlq(sqs_client, dlq_url: str) -> bool:
    """DLQ의 모든 메시지를 삭제합니다."""
    try:
        sqs_client.purge_queue(QueueUrl=dlq_url)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'AWS.SimpleQueueService.PurgeQueueInProgress':
            print("⚠️ 이미 Purge 작업이 진행 중입니다. 60초 후 다시 시도하세요.")
            return False
        raise


def redrive_messages(sqs_client, dlq_url: str, main_queue_url: str, max_messages: int = 100) -> int:
    """DLQ 메시지를 메인 큐로 재전송합니다."""
    redrive_count = 0
    
    while redrive_count < max_messages:
        response = sqs_client.receive_message(
            QueueUrl=dlq_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=1,
            VisibilityTimeout=30
        )
        
        messages = response.get('Messages', [])
        if not messages:
            break
        
        for msg in messages:
            try:
                # 메인 큐로 재전송
                # FIFO 큐인 경우 MessageGroupId와 MessageDeduplicationId 필요
                send_params = {
                    'QueueUrl': main_queue_url,
                    'MessageBody': msg['Body']
                }
                
                if main_queue_url.endswith('.fifo'):
                    import uuid
                    # 원본 메시지의 GroupId를 사용하거나 기본값 사용
                    body = json.loads(msg['Body'])
                    group_id = body.get('examCode', 'redrive')
                    send_params['MessageGroupId'] = group_id
                    send_params['MessageDeduplicationId'] = str(uuid.uuid4())
                
                sqs_client.send_message(**send_params)
                
                # DLQ에서 삭제
                sqs_client.delete_message(
                    QueueUrl=dlq_url,
                    ReceiptHandle=msg['ReceiptHandle']
                )
                
                redrive_count += 1
                
            except Exception as e:
                print(f"⚠️ 메시지 재전송 실패: {e}")
    
    return redrive_count


def main():
    parser = argparse.ArgumentParser(description='DLQ 모니터링 도구')
    parser.add_argument('--peek', action='store_true', help='메시지 미리보기')
    parser.add_argument('--purge', action='store_true', help='DLQ 비우기')
    parser.add_argument('--redrive', action='store_true', help='메인 큐로 재전송')
    parser.add_argument('--count', type=int, default=10, help='조회할 메시지 수')
    args = parser.parse_args()
    
    # 환경변수 로드
    main_queue_url = os.environ.get("SQS_QUEUE_URL")
    dlq_url = os.environ.get("SQS_DLQ_URL")
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
    
    if not all([aws_key, aws_secret]):
        print("❌ AWS 자격 증명이 설정되지 않았습니다.")
        sys.exit(1)
    
    # DLQ URL이 없으면 메인 큐에서 추론
    if not dlq_url and main_queue_url:
        queue_name = main_queue_url.split('/')[-1]
        if queue_name.endswith('.fifo'):
            dlq_name = queue_name.replace('.fifo', '') + '-dlq.fifo'
        else:
            dlq_name = f"{queue_name}-dlq"
        
        # DLQ URL 구성
        base_url = '/'.join(main_queue_url.split('/')[:-1])
        dlq_url = f"{base_url}/{dlq_name}"
    
    if not dlq_url:
        print("❌ DLQ URL을 찾을 수 없습니다. SQS_DLQ_URL 환경변수를 설정하세요.")
        sys.exit(1)
    
    print("=" * 60)
    print("📊 DLQ 모니터링")
    print("=" * 60)
    
    # SQS 클라이언트 생성
    sqs = boto3.client(
        'sqs',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=region
    )
    
    try:
        # 상태 조회
        status = get_dlq_status(sqs, dlq_url)
        
        print(f"\n📋 DLQ 상태:")
        print(f"   - URL: {status['queue_url']}")
        print(f"   - 대기 메시지: {status['messages_available']}")
        print(f"   - 처리 중: {status['messages_in_flight']}")
        print(f"   - 지연: {status['messages_delayed']}")
        print(f"   - 마지막 수정: {status['last_modified']}")
        
        total = status['messages_available'] + status['messages_in_flight']
        
        if total == 0:
            print(f"\n✅ DLQ가 비어있습니다. 실패 메시지가 없습니다.")
        else:
            print(f"\n⚠️ DLQ에 {total}개의 실패 메시지가 있습니다.")
        
        # 메시지 미리보기
        if args.peek and total > 0:
            print(f"\n🔍 메시지 미리보기 (최대 {args.count}개):")
            messages = peek_messages(sqs, dlq_url, args.count)
            
            for i, msg in enumerate(messages, 1):
                print(f"\n--- 메시지 #{i} ---")
                print(f"    ID: {msg['message_id']}")
                print(f"    수신 횟수: {msg['receive_count']}")
                if isinstance(msg['body'], dict):
                    print(f"    이벤트: {msg['body'].get('eventType', 'N/A')}")
                    print(f"    시험코드: {msg['body'].get('examCode', 'N/A')}")
                    print(f"    파일명: {msg['body'].get('filename', 'N/A')}")
                else:
                    print(f"    Body: {str(msg['body'])[:200]}")
        
        # DLQ 비우기
        if args.purge and total > 0:
            confirm = input(f"\n⚠️ DLQ의 모든 메시지({total}개)를 삭제하시겠습니까? (yes/no): ")
            if confirm.lower() == 'yes':
                success = purge_dlq(sqs, dlq_url)
                if success:
                    print("✅ DLQ가 비워졌습니다.")
            else:
                print("❌ 취소되었습니다.")
        
        # 메인 큐로 재전송
        if args.redrive and total > 0:
            if not main_queue_url:
                print("❌ 메인 큐 URL이 필요합니다. SQS_QUEUE_URL 환경변수를 설정하세요.")
                sys.exit(1)
            
            confirm = input(f"\n⚠️ DLQ 메시지를 메인 큐로 재전송하시겠습니까? (yes/no): ")
            if confirm.lower() == 'yes':
                count = redrive_messages(sqs, dlq_url, main_queue_url, args.count)
                print(f"✅ {count}개 메시지가 재전송되었습니다.")
            else:
                print("❌ 취소되었습니다.")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
            print(f"❌ DLQ가 존재하지 않습니다: {dlq_url}")
            print("   먼저 python scripts/setup_dlq.py를 실행하세요.")
        else:
            raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
setup_dlq.py - SQS Dead Letter Queue 설정 스크립트

이 스크립트는 기존 SQS 큐에 DLQ(Dead Letter Queue)를 설정합니다.
DLQ는 일정 횟수 이상 처리 실패한 메시지를 저장하여 무한 루프를 방지합니다.

사용법:
    python scripts/setup_dlq.py

환경변수:
    - SQS_QUEUE_URL: 메인 큐 URL
    - AWS_ACCESS_KEY_ID: AWS 액세스 키
    - AWS_SECRET_ACCESS_KEY: AWS 시크릿 키
    - AWS_DEFAULT_REGION: AWS 리전 (기본: ap-northeast-2)
"""

import os
import sys
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

import boto3
from botocore.exceptions import ClientError


def get_queue_arn(sqs_client, queue_url: str) -> str:
    """큐 URL에서 ARN을 가져옵니다."""
    response = sqs_client.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['QueueArn']
    )
    return response['Attributes']['QueueArn']


def create_dlq(sqs_client, main_queue_url: str, max_receive_count: int = 3) -> dict:
    """
    DLQ를 생성하고 메인 큐에 연결합니다.
    
    Args:
        sqs_client: boto3 SQS 클라이언트
        main_queue_url: 메인 큐 URL
        max_receive_count: 최대 수신 횟수 (이후 DLQ로 이동)
    
    Returns:
        dict: DLQ 정보 (url, arn, redrive_policy)
    """
    
    # 1. 메인 큐 이름 추출
    # URL 형식: https://sqs.{region}.amazonaws.com/{account-id}/{queue-name}
    queue_name = main_queue_url.split('/')[-1]
    dlq_name = f"{queue_name}-dlq"
    
    print(f"[1/4] DLQ 이름: {dlq_name}")
    
    # 2. DLQ가 FIFO인지 확인 (메인 큐가 .fifo로 끝나면 DLQ도 FIFO)
    is_fifo = queue_name.endswith('.fifo')
    if is_fifo:
        dlq_name = dlq_name.replace('.fifo', '') + '-dlq.fifo'
    
    print(f"[2/4] FIFO 큐 여부: {is_fifo}")
    
    # 3. DLQ 생성 (이미 존재하면 기존 것 사용)
    try:
        # 먼저 존재 여부 확인
        response = sqs_client.get_queue_url(QueueName=dlq_name)
        dlq_url = response['QueueUrl']
        print(f"[3/4] 기존 DLQ 발견: {dlq_url}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'AWS.SimpleQueueService.NonExistentQueue':
            # DLQ 생성
            create_attrs = {
                'MessageRetentionPeriod': '1209600',  # 14일 (최대값)
                'VisibilityTimeout': '300'  # 5분
            }
            
            if is_fifo:
                create_attrs['FifoQueue'] = 'true'
                create_attrs['ContentBasedDeduplication'] = 'true'
            
            response = sqs_client.create_queue(
                QueueName=dlq_name,
                Attributes=create_attrs
            )
            dlq_url = response['QueueUrl']
            print(f"[3/4] ✅ DLQ 생성 완료: {dlq_url}")
        else:
            raise
    
    # 4. DLQ ARN 가져오기
    dlq_arn = get_queue_arn(sqs_client, dlq_url)
    print(f"[3/4] DLQ ARN: {dlq_arn}")
    
    # 5. 메인 큐에 RedrivePolicy 설정
    redrive_policy = {
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": str(max_receive_count)
    }
    
    sqs_client.set_queue_attributes(
        QueueUrl=main_queue_url,
        Attributes={
            'RedrivePolicy': json.dumps(redrive_policy)
        }
    )
    print(f"[4/4] ✅ RedrivePolicy 설정 완료 (maxReceiveCount={max_receive_count})")
    
    return {
        "dlq_url": dlq_url,
        "dlq_arn": dlq_arn,
        "dlq_name": dlq_name,
        "max_receive_count": max_receive_count,
        "redrive_policy": redrive_policy
    }


def verify_dlq_setup(sqs_client, main_queue_url: str) -> dict:
    """DLQ 설정이 올바른지 확인합니다."""
    response = sqs_client.get_queue_attributes(
        QueueUrl=main_queue_url,
        AttributeNames=['RedrivePolicy', 'ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    )
    
    attrs = response.get('Attributes', {})
    
    redrive = attrs.get('RedrivePolicy')
    if redrive:
        redrive = json.loads(redrive)
    
    return {
        "main_queue_url": main_queue_url,
        "redrive_policy": redrive,
        "messages_available": attrs.get('ApproximateNumberOfMessages', 'N/A'),
        "messages_in_flight": attrs.get('ApproximateNumberOfMessagesNotVisible', 'N/A')
    }


def main():
    print("=" * 60)
    print("🔧 SQS Dead Letter Queue (DLQ) 설정")
    print("=" * 60)
    
    # 환경변수 로드
    queue_url = os.environ.get("SQS_QUEUE_URL")
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
    
    if not all([queue_url, aws_key, aws_secret]):
        print("❌ 필수 환경변수가 설정되지 않았습니다:")
        print("   - SQS_QUEUE_URL")
        print("   - AWS_ACCESS_KEY_ID")
        print("   - AWS_SECRET_ACCESS_KEY")
        sys.exit(1)
    
    print(f"\n📋 설정 정보:")
    print(f"   - 메인 큐: {queue_url}")
    print(f"   - 리전: {region}")
    
    # SQS 클라이언트 생성
    sqs = boto3.client(
        'sqs',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=region
    )
    
    # 최대 수신 횟수 (사용자 입력 또는 기본값)
    max_receive_count = int(os.environ.get("DLQ_MAX_RECEIVE_COUNT", "3"))
    print(f"   - 최대 재시도 횟수: {max_receive_count}")
    
    print(f"\n🚀 DLQ 설정 시작...\n")
    
    try:
        # DLQ 생성 및 설정
        result = create_dlq(sqs, queue_url, max_receive_count)
        
        print(f"\n" + "=" * 60)
        print("✅ DLQ 설정 완료!")
        print("=" * 60)
        print(f"\n📊 결과:")
        print(f"   - DLQ URL: {result['dlq_url']}")
        print(f"   - DLQ ARN: {result['dlq_arn']}")
        print(f"   - 최대 수신 횟수: {result['max_receive_count']}")
        
        # 설정 검증
        print(f"\n🔍 설정 검증 중...")
        verify = verify_dlq_setup(sqs, queue_url)
        print(f"   - 메인 큐 메시지 수: {verify['messages_available']}")
        print(f"   - 처리 중 메시지 수: {verify['messages_in_flight']}")
        print(f"   - RedrivePolicy: {verify['redrive_policy']}")
        
        # 환경변수 안내
        print(f"\n📝 환경변수에 추가하세요 (.env):")
        print(f"   SQS_DLQ_URL={result['dlq_url']}")
        
        # DLQ 모니터링 안내
        print(f"\n💡 DLQ 모니터링 방법:")
        print(f"   python scripts/monitor_dlq.py")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

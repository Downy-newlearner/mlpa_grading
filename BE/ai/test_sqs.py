import boto3
import json

import os

sqs = boto3.client(
    'sqs',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='ap-northeast-2'
)

queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/309293113458/mlpa-grading-queue.fifo'

message = {
    'event_type': 'STUDENT_ID_RECOGNITION',
    'exam_code': 'TEST_EXAM_SQS_NEW',
    'student_id': 'unknown_id',
    'filename': 'page_001.jpg',
    'index': 15,
    'total': 40
}

response = sqs.send_message(
    QueueUrl=queue_url,
    MessageBody=json.dumps(message),
    MessageGroupId='test-group',
    MessageDeduplicationId='test-dedup-1'
)

print(f"Message sent! ID: {response['MessageId']}")

import boto3
import json
import time

# Setup
QUEUE_URL = 'https://sqs.ap-northeast-2.amazonaws.com/309293113458/mlpa-grading-queue.fifo'
EXAM_CODE = "TEST_EXAM_FLOW"

import os

sqs = boto3.client(
    'sqs',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name='ap-northeast-2'
)

def send_sqs_message(status=None, idx=0, total=40, student_id="processing", filename=None):
    message = {
        'event_type': 'STUDENT_ID_RECOGNITION',
        'exam_code': EXAM_CODE,
        'student_id': student_id,
        'idx': idx,
        'total': total
    }
    
    if status:
        message['status'] = status
    
    if filename:
        message['filename'] = filename

    print(f"📤 Sending SQS: {message}")
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps(message),
        MessageGroupId='test-group',
        MessageDeduplicationId=str(time.time())
    )

def simulate_flow():
    print("🚀 Starting Full Flow Simulation...")
    
    # 1. Loading Phase (0 ~ 100%)
    for i in range(1, 6):
        send_sqs_message(status="RECOGNITION_UPDATE", idx=i*2, total=10)
        time.sleep(1) # Simulate processing time

    # 2. Unknown ID found (Trigger Feedback Page Data)
    print("⚠️  Simulating Unknown ID detection...")
    send_sqs_message(
        student_id="unknown_id",
        filename="unknown_sample.jpg", # Backend will generate URL for this
        idx=8, 
        total=10
    )
    time.sleep(1)

    # 3. Completion
    print("✅ Simulating Completion...")
    send_sqs_message(status="completed", idx=10, total=10)

if __name__ == "__main__":
    simulate_flow()

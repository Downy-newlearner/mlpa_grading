
import os
import json
import boto3
import time
from dotenv import load_dotenv

load_dotenv()

queue_url = os.environ.get("SQS_QUEUE_URL")
aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
region = "ap-northeast-2"

sqs = boto3.client(
    'sqs',
    aws_access_key_id=aws_key,
    aws_secret_access_key=aws_secret,
    region_name=region
)

print(f"Polling from {queue_url}...")

while True:
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20,
        AttributeNames=['All'],
        MessageAttributeNames=['All']
    )
    
    messages = response.get('Messages', [])
    if messages:
        for msg in messages:
            print(f"[{time.ctime()}] Received: {msg['Body']}")
            # sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
    else:
        print(f"[{time.ctime()}] No messages...")

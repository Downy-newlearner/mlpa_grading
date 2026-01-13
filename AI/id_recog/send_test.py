
import os
import json
import boto3
import uuid
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

test_msg = {
    "eventType": "ATTENDANCE_UPLOAD",
    "examCode": "TEST_AUTO",
    "downloadUrl": "https://example.com/test.xlsx",
    "filename": "test.xlsx"
}

response = sqs.send_message(
    QueueUrl=queue_url,
    MessageBody=json.dumps(test_msg),
    MessageGroupId="test_group",
    MessageDeduplicationId=str(uuid.uuid4())
)

print(f"Message sent! ID: {response.get('MessageId')}")

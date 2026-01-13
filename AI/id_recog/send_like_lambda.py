
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

# Lambda와 동일한 형식의 메시지
test_msg = {
    "eventType": "STUDENT_ID_RECOGNITION",
    "examCode": "UC56A5",
    "downloadUrl": "https://mlpa-gradi.s3.ap-northeast-2.amazonaws.com/uploads/UC56A5/test.png",
    "filename": "test.png"
}

response = sqs.send_message(
    QueueUrl=queue_url,
    MessageBody=json.dumps(test_msg),
    MessageGroupId="UC56A5_test",
    MessageDeduplicationId=str(uuid.uuid4())
)

print(f"✅ Message sent! ID: {response.get('MessageId')}")
print(f"Queue: {queue_url}")


import os
import json
import boto3
import uuid
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

test_msg = {"test": str(uuid.uuid4())}

sqs.send_message(
    QueueUrl=queue_url,
    MessageBody=json.dumps(test_msg),
    MessageGroupId="test_group",
    MessageDeduplicationId=str(uuid.uuid4())
)

time_start = time.time()
found = False
for i in range(10):
    attributes = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
    )
    v = int(attributes['Attributes']['ApproximateNumberOfMessages'])
    nv = int(attributes['Attributes']['ApproximateNumberOfMessagesNotVisible'])
    print(f"[{i}] Visible: {v}, In-flight: {nv}")
    if v > 0 or nv > 0:
        found = True
    time.sleep(0.5)

if not found:
    print("Message DISAPPEARED immediately!")
else:
    print("Message stayed for a bit.")

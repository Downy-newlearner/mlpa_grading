
import os
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

attributes = sqs.get_queue_attributes(
    QueueUrl=queue_url,
    AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
)
print(f"Available: {attributes['Attributes']['ApproximateNumberOfMessages']}")
print(f"In flight: {attributes['Attributes']['ApproximateNumberOfMessagesNotVisible']}")

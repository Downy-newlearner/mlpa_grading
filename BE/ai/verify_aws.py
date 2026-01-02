import boto3
import os
try:
    sts = boto3.client(
        'sts',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name='ap-northeast-2'
    )
    identity = sts.get_caller_identity()
    print(f"Account: {identity['Account']}")
    print(f"Arn: {identity['Arn']}")
    
    sqs = boto3.client(
        'sqs',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name='ap-northeast-2'
    )
    queues = sqs.list_queues(QueueNamePrefix='mlpa-grading-queue')
    print(f"Queues found: {queues.get('QueueUrls', [])}")
except Exception as e:
    print(f"Error: {e}")

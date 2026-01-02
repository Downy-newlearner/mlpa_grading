import json
import boto3
import os
import urllib.parse

# Initialize clients outside handler for reuse
s3 = boto3.client('s3')
sqs = boto3.client('sqs')

# Environment Variable: URL of the SQS queue that the AI server polls
# Example: https://sqs.ap-northeast-2.amazonaws.com/1234567890/ai-input-queue
QUEUE_URL = os.environ.get('AI_INPUT_QUEUE_URL')

def lambda_handler(event, context):
    """
    Triggered by S3 Object Created Event.
    Generates a Presigned GET URL for the new object and sends it to SQS.
    """
    print(f"Received event: {json.dumps(event)}")
    
    if not QUEUE_URL:
        print("Error: AI_INPUT_QUEUE_URL environment variable is not set.")
        return {
            'statusCode': 500,
            'body': json.dumps('Configuration Error: Missing Queue URL')
        }

    for record in event['Records']:
        try:
            # 1. Extract Bucket and Key
            bucket = record['s3']['bucket']['name']
            key = urllib.parse.unquote_plus(record['s3']['object']['key'], encoding='utf-8')
            
            print(f"Processing object: s3://{bucket}/{key}")

            # 2. Extract Metadata (total, idx)
            try:
                head_object = s3.head_object(Bucket=bucket, Key=key)
                metadata = head_object.get('Metadata', {})
                # S3 metadata keys are lowercased
                total_count = int(metadata.get('total', 40)) 
                current_idx = int(metadata.get('idx', 0))
            except Exception as meta_error:
                print(f"⚠️ Failed to extract metadata: {meta_error}")
                total_count = 40
                current_idx = 0

            # 3. Extract Exam Code (Assuming key format: uploads/{examCode}/{filename})
            # Adjust this logic based on your actual S3 path structure
            parts = key.split('/')
            exam_code = "unknown"
            if len(parts) > 1 and parts[0] == "uploads":
                exam_code = parts[1]
            
            # 3. Generate Presigned GET URL (Valid for 1 hour)
            presigned_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600
            )

            # 4. Construct Message Payload for AI Server
            message_body = {
                "exam_code": exam_code,
                "s3_key": key,
                "filename": parts[-1],
                "download_url": presigned_url,
                "event_type": "STUDENT_ID_RECOGNITION",
                "idx": current_idx,
                "total": total_count
            }

            # 5. Send to SQS
            response = sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(message_body)
            )
            
            print(f"Message sent to SQS! MessageId: {response['MessageId']}")

        except Exception as e:
            print(f"Error processing record {record['eventID']}: {e}")
            # Raising exception marks the batch as failed (if configured)
            raise e

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully processed S3 event')
    }

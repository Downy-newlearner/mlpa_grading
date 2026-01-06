import json
import boto3
import os
import urllib.parse

# Initialize clients outside handler for reuse
s3 = boto3.client('s3')
sqs = boto3.client('sqs')

# Environment Variable: URL of the SQS queue that the AI server polls
QUEUE_URL = os.environ.get('AI_INPUT_QUEUE_URL')

def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "local-test"
    print(f"--- [Request ID: {request_id}] Lambda Handler Start ---")
    print(f"Received event: {json.dumps(event)}")
    print(f"Using QUEUE_URL: {QUEUE_URL}")
    
    if not QUEUE_URL:
        print(f"❌ [{request_id}] Error: AI_INPUT_QUEUE_URL is not set.")
        return {
            'statusCode': 500,
            'body': json.dumps('Configuration Error: Missing Queue URL')
        }

    records = event.get('Records', [])
    print(f"[{request_id}] Total Records in this event: {len(records)}")
    
    processed_count = 0
    for i, record in enumerate(records):
        try:
            # 1. Extract Bucket and Key
            bucket = record['s3']['bucket']['name']
            raw_key = record['s3']['object']['key']
            key = urllib.parse.unquote_plus(raw_key, encoding='utf-8')
            
            print(f"--- [{request_id}] Processing Record {i + 1}/{len(records)} ---")
            print(f"Bucket: {bucket}")
            print(f"Key: {key}")

            # 2. Extract Exam Code and Determine Event Type
            clean_key = key.lstrip('/')
            parts = clean_key.split('/')
            print(f"Key Parts: {parts}")
            
            exam_code = "unknown"
            event_type = "STUDENT_ID_RECOGNITION" # Default

            # Robust searching for 'uploads' or 'attendance'
            if "uploads" in parts:
                idx = parts.index("uploads")
                if len(parts) > idx + 1:
                    exam_code = parts[idx + 1]
                    event_type = "STUDENT_ID_RECOGNITION"
            elif "attendance" in parts:
                idx = parts.index("attendance")
                if len(parts) > idx + 1:
                    exam_code = parts[idx + 1]
                    event_type = "ATTENDANCE_UPLOAD"
            
            print(f"Detected Exam Code: {exam_code}")
            print(f"Detected Event Type: {event_type}")

            # 3. Generate Presigned GET URL (Valid for 1 hour)
            print(f"Generating presigned URL for {key}...")
            presigned_url = s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600
            )

            # 4. Construct Message Payload for AI Server
            message_body = {
                "examCode": exam_code,
                "filename": parts[-1],
                "downloadUrl": presigned_url,
                "eventType": event_type
            }
            print(f"SQS Message Body: {json.dumps(message_body)}")

            # 5. Send to SQS
            safe_group_id = exam_code.replace(" ", "_")
            
            # Use eventID or a unique combination as deduplication ID
            dedup_id = record.get('eventID') or f"{bucket}-{key}-{record['s3']['object'].get('sequencer', 'default')}"
            dedup_id = dedup_id.replace(" ", "_").replace("/", "_")

            print(f"Sending to SQS (GroupId: {safe_group_id}, DedupId: {dedup_id})...")

            response = sqs.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(message_body),
                MessageGroupId=safe_group_id,
                MessageDeduplicationId=dedup_id
            )
            
            print(f"✅ [{request_id}] Message sent! MessageId: {response['MessageId']}")
            processed_count += 1

        except Exception as e:
            print(f"❌ [{request_id}] Error processing record: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue to next record if one fails, but keep track of error
            continue

    print(f"--- [{request_id}] Lambda Handler End (Processed {processed_count}/{len(records)}) ---")
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully processed {processed_count} S3 event(s)')
    }

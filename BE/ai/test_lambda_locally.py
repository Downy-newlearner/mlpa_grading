import os
import json
import yaml
from lambda_s3_trigger import lambda_handler

# 1. Load AWS Credentials and Queue URL from application-local.yml if available
# This helps running the test locally without manual env setup
def load_env_from_yaml(path='../application-local.yml'):
    if not os.path.exists(path):
        print(f"⚠️ {path} not found. Using system environment variables.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
        # Mapping YAML keys to Env variables expected by boto3/lambda
        os.environ['AWS_ACCESS_KEY_ID'] = config.get('AWS_ACCESS_KEY', os.environ.get('AWS_ACCESS_KEY_ID'))
        os.environ['AWS_SECRET_ACCESS_KEY'] = config.get('AWS_SECRET_KEY', os.environ.get('AWS_SECRET_ACCESS_KEY'))
        os.environ['AWS_DEFAULT_REGION'] = config.get('AWS_REGION_STATIC', 'ap-northeast-2')
        os.environ['AI_INPUT_QUEUE_URL'] = config.get('AWS_SQS_QUEUE_URL', os.environ.get('AI_INPUT_QUEUE_URL'))

def create_s3_event(bucket_name, object_key):
    return {
        "Records": [
            {
                "eventVersion": "2.1",
                "eventSource": "aws:s3",
                "awsRegion": "ap-northeast-2",
                "eventTime": "2026-01-03T00:00:00.000Z",
                "eventName": "ObjectCreated:Put",
                "s3": {
                    "bucket": {
                        "name": bucket_name
                    },
                    "object": {
                        "key": object_key,
                        "size": 1024,
                        "eTag": "test-etag"
                    }
                },
                "eventID": "test-event-id"
            }
        ]
    }

if __name__ == "__main__":
    # Initialize Environment
    load_env_from_yaml()
    
    # Verify environment
    if not os.environ.get('AI_INPUT_QUEUE_URL'):
        print("❌ Error: AI_INPUT_QUEUE_URL (AWS_SQS_QUEUE_URL in YAML) is not set.")
        exit(1)

    print(f"🚀 Testing Lambda with Queue: {os.environ['AI_INPUT_QUEUE_URL']}")

    # --- Case 1: Student ID Recognition Upload ---
    print("\n--- Testing Case 1: Student ID Recognition (uploads/...) ---")
    event_id = create_s3_event("mlpa-gradi", "uploads/TEST_EXAM/student_001.jpg")
    try:
        response = lambda_handler(event_id, None)
        print(f"Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"Caught Expected/Unexpected error (Metadata might fail if file doesn't exist): {e}")

    # --- Case 2: Attendance File Upload ---
    print("\n--- Testing Case 2: Attendance Upload (attendance/...) ---")
    event_attendance = create_s3_event("mlpa-gradi", "attendance/TEST_EXAM/attendance.xlsx")
    try:
        response = lambda_handler(event_attendance, None)
        print(f"Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"Caught error: {e}")

    print("\n✅ Test execution completed.")

import json
import boto3
import os
import urllib.request
import urllib.parse
from urllib.error import URLError, HTTPError

# Configuration
API_BASE_URL = "http://localhost:8080/api"
EXAM_CODE = "TEST_EXAM_FLOW"
STUDENT_ID = "20241234"

def get_sts_token():
    """Step 1: Request Temporary Credentials from Backend"""
    print(f"🔑 Requesting STS Token for {EXAM_CODE}/{STUDENT_ID}...")
    url = f"{API_BASE_URL}/sts/token?examCode={EXAM_CODE}&studentId={STUDENT_ID}"
    
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print("✅ Received Credentials")
                return data
            else:
                print(f"❌ Failed to get STS token. Status: {response.status}")
                return None
    except HTTPError as e:
        print(f"❌ HTTP Error: {e.code} {e.reason}")
        return None
    except URLError as e:
        print(f"❌ URL Error: {e.reason}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None

def upload_file_with_sts(creds, filename, content):
    """Step 2: Upload File to S3 using STS Credentials"""
    print(f"📤 Uploading {filename} to S3...")
    
    if not creds:
        return

    # Initialize S3 Client with Temporary Credentials
    s3 = boto3.client(
        's3',
        aws_access_key_id=creds['accessKey'],
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=creds['sessionToken'],
        region_name='ap-northeast-2'
    )

    # Construct Key (Must match Policy Resource: header/{examCode}/{studentId}/unknown_answer/*)
    key = f"header/{EXAM_CODE}/{STUDENT_ID}/unknown_answer/1/1/{filename}"
    bucket = "mlpa-gradi" # Replace with actual bucket name

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType='image/jpeg'
        )
        print(f"✅ Upload Successful: s3://{bucket}/{key}")
        return key
    except Exception as e:
        print(f"❌ Upload Failed: {e}")
        return None

if __name__ == "__main__":
    # Create a dummy image content
    dummy_content = b"fake_image_content"
    
    creds = get_sts_token()
    if creds:
        upload_file_with_sts(creds, "test_image.jpg", dummy_content)

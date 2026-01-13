
import os
from dotenv import load_dotenv

load_dotenv()

print(f"SQS_QUEUE_URL={os.environ.get('SQS_QUEUE_URL')}")
print(f"AWS_DEFAULT_REGION={os.environ.get('AWS_DEFAULT_REGION')}")
print(f"S3_BUCKET={os.environ.get('S3_BUCKET')}")

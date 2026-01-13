
import os
from dotenv import load_dotenv

load_dotenv(".env.production")

print(f"SQS_QUEUE_URL={os.environ.get('SQS_QUEUE_URL')}")

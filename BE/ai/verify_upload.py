import requests
import json

BASE_URL = "http://localhost:8080/api"

def test_upload():
    print("🚀 Starting Upload Verification...")

    # 1. Request Batch Presigned URLs
    exam_code = "TEST_UPLOAD_CODE"
    
    # Payload matching Frontend logic (1-based index)
    payload = {
        "examCode": exam_code,
        "total": 1,
        "images": [
            {
                "index": 1, # 1-based
                "contentType": "image/jpeg",
                "filename": "test.jpg"
            }
        ]
    }

    print(f"📡 Requesting Presigned URLs with payload: {json.dumps(payload, indent=2)}")
    
    try:
        resp = requests.post(
            f"{BASE_URL}/storage/presigned-urls/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Response Status: {resp.status_code}")
        
        if resp.status_code != 200:
            print(f"❌ Failed to get presigned URLs: {resp.text}")
            return

        data = resp.json()
        print(f"✅ Got Presigned URLs: {json.dumps(data, indent=2)}")
        
        url_info = data['urls'][0]
        upload_url = url_info['url']
        
        # 2. Attempt Upload
        print(f"📤 Attempting Upload to: {upload_url}")
        
        # Headers matching Frontend logic
        headers = {
            "Content-Type": "image/jpeg",
            "x-amz-meta-total": "1",
            "x-amz-meta-idx": "1"
        }
        
        # Dummy content
        content = b"dummy content"
        
        upload_resp = requests.put(
            upload_url,
            data=content,
            headers=headers
        )
        
        print(f"Upload Status: {upload_resp.status_code}")
        
        if upload_resp.status_code in [200, 201]:
            print("✅ Upload SUCCESS!")
        else:
            print(f"❌ Upload FAILED. Status: {upload_resp.status_code}")
            with open("upload_error.xml", "w") as f:
                f.write(upload_resp.text)
            print("Error details saved to upload_error.xml")
            # print headers for debug
            print("Response Headers:", upload_resp.headers)

    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_upload()

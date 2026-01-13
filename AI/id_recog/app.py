"""
app.py - FastAPI 서버 구현

Student ID Extraction Pipeline API 서버입니다.
SQS Worker를 통해 백그라운드에서 메시지를 처리합니다.
"""

import os
import io
import base64
from contextlib import asynccontextmanager
from typing import Optional, List

# .env 파일 자동 로드
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import boto3

# 환경 변수 설정 (모델 로드 전에 설정)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from schemas import Config
from student_id_pipeline import extract_student_id


# =============================================================================
# Global Model Storage
# =============================================================================
class ModelStore:
    layout_model = None
    ocr_model = None
    vlm_client = None
    s3_manager = None
    sqs_worker = None  # SQS Worker


# =============================================================================
# Lifespan (모델 로드 및 SQS Worker 시작)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 모델을 로드하고 SQS Worker를 시작합니다."""
    print("=" * 60)
    print("모델 로딩 시작...")
    print("=" * 60)
    
    # 1. PP-DocLayout_plus-L 모델 로드
    print("[1/3] PP-DocLayout_plus-L 모델 로딩...")
    try:
        from paddlex import create_model
        ModelStore.layout_model = create_model(model_name="PP-DocLayout_plus-L")
        print("  ✓ Layout 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ Layout 모델 로드 실패: {e}")
    
    # 2. PP-OCRv5_mobile_rec 모델 로드
    print("[2/3] PP-OCRv5_mobile_rec 모델 로딩...")
    try:
        ModelStore.ocr_model = create_model(model_name="PP-OCRv5_mobile_rec")
        print("  ✓ PP-OCRv5 OCR 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ PP-OCRv5 OCR 모델 로드 실패: {e}")
    
    # 3. VLM Client (OpenAI)
    print("[3/3] VLM Client 설정...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            ModelStore.vlm_client = OpenAI(api_key=api_key)
            print("  ✓ VLM Client 설정 완료")
        except Exception as e:
            print(f"  ✗ VLM Client 설정 실패: {e}")
            ModelStore.vlm_client = None
    else:
        print("  - OPENAI_API_KEY 없음, VLM fallback 비활성화")
        ModelStore.vlm_client = None
    
    print("=" * 60)
    print("모델 로딩 완료!")
    print("=" * 60)
    
    # 4. S3 클라이언트 초기화
    print("\n[S3] S3 클라이언트 초기화...")
    try:
        from s3_client import get_s3_manager
        ModelStore.s3_manager = get_s3_manager()
        if ModelStore.s3_manager.is_ready:
            print("  ✓ S3 클라이언트 준비 완료")
        else:
            print("  - S3 자격증명 미설정")
    except Exception as e:
        print(f"  ✗ S3 클라이언트 초기화 실패: {e}")
        ModelStore.s3_manager = None
    
    # 5. SQS Worker 초기화 및 시작
    print("\n[SQS] SQS Worker 초기화...")
    try:
        from sqs_worker import init_sqs_worker
        
        queue_url = os.environ.get("SQS_QUEUE_URL")  # BE → AI 입력 큐
        result_queue_url = os.environ.get("SQS_QUEUE_URL2")  # AI → BE 결과 큐
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        if queue_url and aws_key and aws_secret:
            worker = init_sqs_worker(
                queue_url=queue_url,
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"),
                s3_bucket=os.environ.get("S3_BUCKET", "mlpa-gradi"),
                result_queue_url=result_queue_url  # 결과 전송용 큐 (없으면 입력 큐 사용)
            )
            
            # 학번 추출 콜백 설정
            def student_id_callback(image: np.ndarray, student_list: list) -> dict:
                config = Config()
                result = extract_student_id(
                    original_image=image,
                    student_id_list=student_list,
                    layout_model=ModelStore.layout_model,
                    ocr_model=ModelStore.ocr_model,
                    vlm_client=ModelStore.vlm_client,
                    config=config
                )
                return {
                    "student_id": result.student_id,
                    "header_image": result.header_image,  # 헤더 이미지 추가
                    "meta": result.meta
                }
            
            worker.set_student_id_callback(student_id_callback)
            worker.start()
            ModelStore.sqs_worker = worker
            print(f"  ✓ SQS Worker 시작됨")
            print(f"    - 입력 큐: {queue_url}")
            print(f"    - 결과 큐: {result_queue_url or queue_url}")
        else:
            print("  - SQS 환경변수 미설정 (SQS_QUEUE_URL, AWS_ACCESS_KEY_ID)")
    except Exception as e:
        print(f"  ✗ SQS Worker 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("서버 종료...")
    if ModelStore.sqs_worker:
        ModelStore.sqs_worker.stop()


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Student ID Extraction API",
    description="답안지 이미지에서 학번을 자동 추출하는 API (SQS 기반)",
    version="2.0.0",
    lifespan=lifespan
)


# =============================================================================
# Request/Response Models (camelCase)
# =============================================================================
class FallbackImageItem(BaseModel):
    """Fallback 이미지 항목"""
    fileName: str
    studentId: str


class FallbackRequest(BaseModel):
    """사용자 Fallback 요청"""
    examCode: str
    images: List[FallbackImageItem]


class FallbackResponse(BaseModel):
    """Fallback 응답"""
    success: bool
    uploadedCount: int = 0
    s3Keys: List[str] = []
    message: str = ""


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/")
async def root():
    """헬스 체크"""
    return {"status": "ok", "message": "Student ID Extraction API v2.0 (SQS-based)"}


@app.get("/health")
async def health():
    """서비스 상태 확인"""
    worker_status = {}
    if ModelStore.sqs_worker:
        worker_status = {
            "running": ModelStore.sqs_worker.is_running,
            "loadedExams": list(ModelStore.sqs_worker._student_id_lists.keys())
        }
    
    return {
        "layoutModel": ModelStore.layout_model is not None,
        "ocrModel": ModelStore.ocr_model is not None,
        "vlmClient": ModelStore.vlm_client is not None,
        "s3Client": ModelStore.s3_manager is not None and ModelStore.s3_manager.is_ready,
        "sqsWorker": worker_status
    }


@app.post("/fallback/", response_model=FallbackResponse)
async def fallback(request: FallbackRequest):
    """
    사용자 Fallback 처리
    
    - unknown_id 폴더의 이미지를 올바른 학번 폴더로 이동
    - header → original 경로 복사
    """
    # 디버깅용: 요청 데이터 로깅
    print(f"[API_REQUEST] /fallback/ body: {request.json()}")
    
    if not ModelStore.s3_manager or not ModelStore.s3_manager.is_ready:
        raise HTTPException(status_code=503, detail="S3 클라이언트가 준비되지 않았습니다.")
    
    exam_code = request.examCode
    uploaded_keys = []
    errors = []
    
    s3_bucket = os.environ.get("S3_BUCKET", "mlpa-gradi")
    s3_client = ModelStore.s3_manager._s3_client
    
    for item in request.images:
        try:
            # 원본 이미지 경로 (unknown_id에 저장된 이미지)
            source_key = f"header/{exam_code}/unknown_id/{item.fileName}"
            
            # 새 경로 (올바른 학번)
            dest_key = f"original/{exam_code}/{item.studentId}/{item.fileName}"
            
            # S3 복사 (header/unknown_id → original/studentId)
            s3_client.copy_object(
                Bucket=s3_bucket,
                CopySource={"Bucket": s3_bucket, "Key": source_key},
                Key=dest_key
            )
            
            uploaded_keys.append(dest_key)
            
        except Exception as e:
            errors.append(f"{item.fileName}: {str(e)}")
    
    if errors:
        return FallbackResponse(
            success=False,
            uploadedCount=len(uploaded_keys),
            s3Keys=uploaded_keys,
            message=f"일부 실패: {'; '.join(errors)}"
        )
    
    return FallbackResponse(
        success=True,
        uploadedCount=len(uploaded_keys),
        s3Keys=uploaded_keys,
        message=f"{len(uploaded_keys)}개 이미지 이동 완료"
    )


@app.get("/exams/")
async def list_loaded_exams():
    """
    현재 로드된 시험 목록 조회
    (SQS Worker에 의해 ATTENDANCE_UPLOAD로 로드된 시험들)
    """
    if not ModelStore.sqs_worker:
        return {"exams": [], "message": "SQS Worker가 실행 중이 아닙니다."}
    
    exams = []
    for exam_code, student_list in ModelStore.sqs_worker._student_id_lists.items():
        exams.append({
            "examCode": exam_code,
            "studentCount": len(student_list)
        })
    
    return {"exams": exams}


# =============================================================================
# Run (개발용)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
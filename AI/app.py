"""
app.py - 통합 AI 서버 (학번 인식 + 답안 인식)

Student ID Extraction과 Answer Recognition 파이프라인을 
통합하여 제공하는 FastAPI 서버입니다.

기능:
1. 학번 인식 (SQS STUDENT_ID_RECOGNITION 이벤트)
2. 답안 인식 (SQS ANSWER_RECOGNITION 이벤트)
3. Fallback 처리 (학번/답안 모두)
4. 채점 (GRADING_COMPLETE 이벤트)
"""

import os
import sys
import io
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime

# .env 파일 자동 로드
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import cv2

# 환경 변수 설정 (모델 로드 전에 설정)
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)


# =============================================================================
# Global Model Storage
# =============================================================================
class ModelStore:
    """전역 모델 저장소"""
    layout_model = None
    ocr_model = None
    vlm_client = None
    s3_manager = None
    sqs_worker = None
    attendance_worker = None
    
    # 답안 인식용
    answer_pipeline = None
    fallback_store = None


# =============================================================================
# Lifespan (모델 로드 및 SQS Worker 시작)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 모델을 로드하고 SQS Worker를 시작합니다."""
    print("=" * 60)
    print("AI 통합 서버 시작 - 모델 로딩...")
    print("=" * 60)
    
    # 1. PP-DocLayout_plus-L 모델 로드
    print("[1/4] PP-DocLayout_plus-L 모델 로딩...")
    try:
        from paddlex import create_model
        ModelStore.layout_model = create_model(model_name="PP-DocLayout_plus-L")
        print("  ✓ Layout 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ Layout 모델 로드 실패: {e}")
    
    # 2. PP-OCRv5_mobile_rec 모델 로드
    print("[2/4] PP-OCRv5_mobile_rec 모델 로딩...")
    try:
        from paddlex import create_model
        ModelStore.ocr_model = create_model(model_name="PP-OCRv5_mobile_rec")
        print("  ✓ PP-OCRv5 OCR 모델 로드 완료")
    except Exception as e:
        print(f"  ✗ PP-OCRv5 OCR 모델 로드 실패: {e}")
    
    # 3. VLM Client (OpenAI)
    print("[3/4] VLM Client 설정...")
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
    
    # 4. Answer Recognition Pipeline
    print("[4/4] Answer Recognition Pipeline 초기화...")
    try:
        from answer_recog.pipeline import AnswerRecognitionPipeline, PipelineConfig
        from answer_recog.roi_extraction import get_fallback_store
        
        config = PipelineConfig(
            debug_mode=os.environ.get("DEBUG_MODE", "").lower() == "true",
            debug_output_dir=os.path.join(CURRENT_DIR, "debug_output")
        )
        
        ModelStore.answer_pipeline = AnswerRecognitionPipeline(
            layout_model=ModelStore.layout_model,
            ocr_model=ModelStore.ocr_model,
            config=config
        )
        ModelStore.fallback_store = get_fallback_store()
        print("  ✓ Answer Recognition Pipeline 초기화 완료")
    except Exception as e:
        print(f"  ✗ Answer Recognition Pipeline 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("모델 로딩 완료!")
    print("=" * 60)
    
    # 5. S3 클라이언트 초기화
    print("\n[S3] S3 클라이언트 초기화...")
    try:
        from id_recog.s3_client import get_s3_manager
        ModelStore.s3_manager = get_s3_manager()
        if ModelStore.s3_manager.is_ready:
            print("  ✓ S3 클라이언트 준비 완료")
        else:
            print("  - S3 자격증명 미설정")
    except Exception as e:
        print(f"  ✗ S3 클라이언트 초기화 실패: {e}")
        ModelStore.s3_manager = None
    
    # 6. SQS Worker 초기화 및 시작
    print("\n[SQS] SQS Worker 초기화...")
    try:
        from id_recog.sqs_worker import init_sqs_worker
        from id_recog.schemas import Config
        from id_recog.student_id_pipeline import extract_student_id
        
        queue_url = os.environ.get("SQS_QUEUE_URL")
        attendance_queue_url = os.environ.get("SQS_ATTENDANCE_QUEUE_URL")
        result_queue_url = os.environ.get("SQS_QUEUE_URL2")
        fallback_queue_url = os.environ.get("SQS_QUEUE_URL3")
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        
        region = os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2")
        bucket = os.environ.get("S3_BUCKET", "mlpa-gradi")
        
        if queue_url and aws_key and aws_secret:
            # 1. 메인 워커 (학번/답안 인식 전용)
            worker = init_sqs_worker(
                queue_url=queue_url,
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=region,
                s3_bucket=bucket,
                result_queue_url=result_queue_url,
                fallback_queue_url=fallback_queue_url
            )
            
            # 2. 출석부 전용 워커 (있을 경우)
            if attendance_queue_url:
                print(f"  ✓ 출석부 전용 워커 초기화 중... ({attendance_queue_url})")
                from id_recog.sqs_worker import SQSWorker
                att_worker = SQSWorker(
                    queue_url=attendance_queue_url,
                    aws_access_key_id=aws_key,
                    aws_secret_access_key=aws_secret,
                    region_name=region,
                    s3_bucket=bucket,
                    result_queue_url=result_queue_url
                )
                # 중요: 두 워커가 학번 리스트 메모리를 공유하도록 설정
                att_worker._student_id_lists = worker._student_id_lists
                att_worker._index_counters = worker._index_counters
                
                # 출석부 워커 전용 콜백 설정
                def attendance_callback(file_path: str) -> list:
                    from id_recog.parsing_xlsx import parsing_xlsx
                    ids = parsing_xlsx(file_path)
                    print(f"  ✓ [ATTENDANCE_WORKER] 출석부 파싱 완료: {len(ids)}명 로드됨")
                    return ids
                
                att_worker.set_attendance_callback(attendance_callback)
                att_worker.start()
                ModelStore.attendance_worker = att_worker
                print(f"  ✓ 출석부 전용 워커 시작됨 (Callback 설정 완료)")
            
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
                    "header_image": result.header_image,
                    "meta": result.meta
                }
            
            worker.set_student_id_callback(student_id_callback)
            
            # 답안 인식 콜백 설정
            def answer_recognition_callback(image: np.ndarray, student_id: str, metadata_dict: dict, filename: str = "unknown.jpg") -> dict:
                from answer_recog.schemas import AnswerSheetMeta
                
                metadata = AnswerSheetMeta.from_dict(metadata_dict)
                result = ModelStore.answer_pipeline.process(image, metadata, student_id)
                
                if not result.success:
                    print(f"  ✗ 답안 인식 실패: {result.error_message}")
                    return {"results": [], "fallback_rois": []}
                
                # Fallback 처리 (Low Confidence)
                for res in result.results:
                    # 신뢰도가 낮거나 미채점이면 ROI 업로드 (필요 시)
                    is_low_conf = res.confidence < ModelStore.answer_pipeline.config.min_confidence
                    
                    if is_low_conf and res.roi_image is not None and ModelStore.sqs_worker:
                        # User Request path: answer/{exam code}/{학번}/{문제 번호}/{꼬리문제 번호}/{파일명}
                        s3_key = f"answer/{metadata.exam_code}/{student_id}/{res.question_number}/{res.sub_question_number or 0}/{filename}"
                        
                        # S3 업로드 (sqs_worker의 메서드 활용)
                        success = ModelStore.sqs_worker.upload_image_to_s3(res.roi_image, s3_key)
                        if success:
                            res.s3_key = s3_key
                
                return {
                    "results": result.results,
                    "fallback_rois": []  # sqs_worker에서 results를 순회하므로 빈 리스트 반환
                }
            
            worker.set_answer_recognition_callback(answer_recognition_callback)
            
            worker.start()
            ModelStore.sqs_worker = worker
            print(f"  ✓ SQS Worker 시작됨")
            print(f"    - 입력 큐: {queue_url}")
            print(f"    - 결과 큐: {result_queue_url or queue_url}")
        else:
            print("  - SQS 환경변수 미설정")
    except Exception as e:
        print(f"  ✗ SQS Worker 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("서버 종료...")
    if ModelStore.sqs_worker:
        ModelStore.sqs_worker.stop()
    if ModelStore.attendance_worker:
        ModelStore.attendance_worker.stop()


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="MLPA AI Server",
    description="학번 인식 + 답안 인식 통합 AI 서버",
    version="3.0.0",
    lifespan=lifespan
)


# =============================================================================
# Request/Response Models
# =============================================================================

# --- 학번 인식 Fallback ---
class StudentIdFallbackItem(BaseModel):
    """학번 Fallback 항목"""
    fileName: str
    studentId: str


class StudentIdFallbackRequest(BaseModel):
    """학번 Fallback 요청"""
    examCode: str
    images: List[StudentIdFallbackItem]


# --- 답안 인식 Fallback ---
class AnswerFallbackItem(BaseModel):
    """답안 Fallback 항목"""
    studentId: str
    questionNumber: int
    subQuestionNumber: int = 0
    answer: str


class AnswerFallbackRequest(BaseModel):
    """답안 Fallback 요청"""
    examCode: str
    corrections: List[AnswerFallbackItem]


# --- 채점 요청 ---
class GradingRequest(BaseModel):
    """채점 요청"""
    examCode: str


# --- 공통 응답 ---
class GenericResponse(BaseModel):
    """범용 응답"""
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


# =============================================================================
# Endpoints - 헬스체크
# =============================================================================

@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "message": "MLPA AI Server v3.0 (Student ID + Answer Recognition)",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health():
    """서비스 상태 확인"""
    worker_status = {}
    if ModelStore.sqs_worker:
        worker_status = {
            "running": ModelStore.sqs_worker.is_running,
            "loadedExams": list(ModelStore.sqs_worker._student_id_lists.keys())
        }
    
    att_worker_status = {}
    if ModelStore.attendance_worker:
        att_worker_status = {
            "running": ModelStore.attendance_worker.is_running,
            "queue": ModelStore.attendance_worker.queue_url
        }
    
    return {
        "layoutModel": ModelStore.layout_model is not None,
        "ocrModel": ModelStore.ocr_model is not None,
        "vlmClient": ModelStore.vlm_client is not None,
        "s3Client": ModelStore.s3_manager is not None and ModelStore.s3_manager.is_ready,
        "sqsWorker": worker_status,
        "attendanceWorker": att_worker_status,
        "answerPipeline": ModelStore.answer_pipeline is not None
    }


# =============================================================================
# Endpoints - 학번 인식
# =============================================================================

@app.post("/fallback/student-id/", response_model=GenericResponse)
async def fallback_student_id(request: StudentIdFallbackRequest):
    """
    학번 인식 Fallback 처리
    
    - unknown_id 폴더의 이미지를 올바른 학번 폴더로 이동
    """
    if not ModelStore.s3_manager or not ModelStore.s3_manager.is_ready:
        raise HTTPException(status_code=503, detail="S3 클라이언트가 준비되지 않았습니다.")
    
    exam_code = request.examCode
    uploaded_keys = []
    errors = []
    
    s3_bucket = os.environ.get("S3_BUCKET", "mlpa-gradi")
    s3_client = ModelStore.s3_manager._s3_client
    
    for item in request.images:
        try:
            source_key = f"header/{exam_code}/unknown_id/{item.fileName}"
            dest_key = f"original/{exam_code}/{item.studentId}/{item.fileName}"
            
            s3_client.copy_object(
                Bucket=s3_bucket,
                CopySource={"Bucket": s3_bucket, "Key": source_key},
                Key=dest_key
            )
            
            uploaded_keys.append(dest_key)
            
        except Exception as e:
            errors.append(f"{item.fileName}: {str(e)}")
    
    # 레거시 호환성을 위해 flat 구조로 반환 (GenericResponse 모델 사용하되 data 필드 활용 대신 직접 구성하거나 GenericResponse 구조 무시)
    # 하지만 response_model=GenericResponse로 되어 있으므로, GenericResponse 자체가 유연하거나, 
    # 호환성을 위해 dict를 반환하고 response_model을 제거/변경해야 함.
    # 기존 코드 호환을 위해 response_model을 제거하고 dict를 반환합니다.
    
    if errors:
        return {
            "success": False,
            "message": f"일부 실패: {'; '.join(errors)}",
            "uploadedCount": len(uploaded_keys),
            "s3Keys": uploaded_keys
        }
    
    return {
        "success": True,
        "message": f"{len(uploaded_keys)}개 이미지 이동 완료",
        "uploadedCount": len(uploaded_keys),
        "s3Keys": uploaded_keys
    }


@app.get("/exams/")
async def list_loaded_exams():
    """현재 로드된 시험 목록 조회"""
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
# Endpoints - 답안 인식
# =============================================================================

@app.post("/fallback/answer/", response_model=GenericResponse)
async def fallback_answer(request: AnswerFallbackRequest):
    """
    답안 인식 Fallback 처리
    
    - 사용자가 수정한 답안을 저장
    - 채점 시 수정값 병합
    """
    print("=" * 60)
    print(f"[API] 답안 Fallback 요청 수신: /fallback/answer/")
    print(f"Payload: {request.json(indent=2, ensure_ascii=False)}")
    print("=" * 60)

    if not ModelStore.fallback_store:
        raise HTTPException(status_code=503, detail="Fallback store가 초기화되지 않았습니다.")
    
    exam_code = request.examCode
    corrections = [c.dict() for c in request.corrections]
    
    ModelStore.fallback_store.apply_corrections(exam_code, corrections)
    
    return GenericResponse(
        success=True,
        message=f"{len(corrections)}개 답안 수정값 저장 완료",
        data={"examCode": exam_code, "correctionCount": len(corrections)}
    )


@app.get("/fallback/answer/{exam_code}")
async def get_fallback_status(exam_code: str):
    """
    답안 Fallback 상태 조회
    
    - Fallback이 필요한 ROI 목록과 현재 수정 현황
    """
    print("=" * 60)
    print(f"[API] 답안 Fallback 상태 조회: /fallback/answer/{exam_code}")
    print("=" * 60)

    if not ModelStore.fallback_store:
        raise HTTPException(status_code=503, detail="Fallback store가 초기화되지 않았습니다.")
    
    summary = ModelStore.fallback_store.get_fallback_summary(exam_code)
    
    # Fallback ROI 목록 생성 (S3 키 포함)
    fallback_rois = ModelStore.fallback_store.get_fallback_rois(exam_code)
    
    roi_list = []
    for student_id, rois in fallback_rois.items():
        for roi in rois:
            roi_list.append({
                "studentId": student_id,
                "questionNumber": roi.question_number,
                "subQuestionNumber": roi.sub_question_number,
                "s3Key": roi.s3_key,
                "recAnswer": roi.rec_answer,
                "confidence": roi.confidence
            })
    
    return {
        "examCode": exam_code,
        "summary": summary,
        "fallbackRois": roi_list
    }


from fastapi import BackgroundTasks

@app.post("/recognition/answer/start", response_model=GenericResponse)
async def start_answer_recognition(
    metadata: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    답안 인식 시작 및 배치 처리 트리거
    
    - 답안지 메타데이터(JSON)를 받아 배치 처리를 시작합니다.
    - S3의 original/{examCode}/ 이미지들을 찾아 답안 인식을 수행합니다.
    """
    print("=" * 60)
    print(f"[API] 답안 인식 시작 요청: /recognition/answer/start")
    print(f"Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
    print("=" * 60)

    if not ModelStore.sqs_worker:
        raise HTTPException(status_code=503, detail="Worker가 초기화되지 않았습니다.")
    
    # 1. examCode 추출
    exam_code = metadata.get("examCode") or metadata.get("exam_code")
    if not exam_code:
        raise HTTPException(status_code=400, detail="examCode가 누락되었습니다.")
    
    # 2. 메타데이터 저장 (Worker 메모리)
    ModelStore.sqs_worker._answer_metadata[exam_code] = metadata
    
    # 3. 배치 작업 시작 (Background)
    # process_batch_answer_recognition는 긴 작업이므로 백그라운드에서 실행
    background_tasks.add_task(
        ModelStore.sqs_worker.process_batch_answer_recognition,
        exam_code,
        metadata
    )
    
    return GenericResponse(
        success=True,
        message=f"답안 인식 배치 작업이 시작되었습니다. (Target: {exam_code})",
        data={"examCode": exam_code}
    )


# =============================================================================
# Endpoints - 채점
# =============================================================================

@app.post("/grade/", response_model=GenericResponse)
async def grade_exam(request: GradingRequest):
    """
    시험 채점 수행
    
    - Fallback 수정값 병합
    - 답안 JSON과 비교하여 채점
    - 결과 반환 (및 백엔드 전송)
    """
    exam_code = request.examCode
    
    if not ModelStore.fallback_store:
        raise HTTPException(status_code=503, detail="Fallback store가 초기화되지 않았습니다.")
    
    # TODO: 실제 채점 로직 구현
    # 1. Fallback 수정값 병합
    # 2. 정답 메타데이터 로드
    # 3. 각 학생별 채점 수행
    # 4. 결과 집계
    
    summary = ModelStore.fallback_store.get_fallback_summary(exam_code)
    
    if summary.get("pending_count", 0) > 0:
        return GenericResponse(
            success=False,
            message=f"아직 처리되지 않은 Fallback이 {summary['pending_count']}개 있습니다.",
            data=summary
        )
    
    return GenericResponse(
        success=True,
        message="채점 요청이 접수되었습니다. (TODO: 실제 채점 로직 구현)",
        data={"examCode": exam_code, "status": "pending"}
    )


# =============================================================================
# Endpoints - 테스트 (개발용)
# =============================================================================

@app.post("/test/recognize-answer/")
async def test_recognize_answer(
    exam_code: str = "TEST_EXAM",
    student_id: str = "20201234"
):
    """
    답안 인식 테스트 (개발용)
    
    - 로컬 테스트 이미지로 파이프라인 테스트
    """
    if not ModelStore.answer_pipeline:
        raise HTTPException(status_code=503, detail="Answer pipeline이 초기화되지 않았습니다.")
    
    # 테스트 이미지 경로
    test_image_path = os.path.join(CURRENT_DIR, "answer_recog", "test_output", "test_image.jpg")
    
    if not os.path.exists(test_image_path):
        return GenericResponse(
            success=False,
            message=f"테스트 이미지 없음: {test_image_path}"
        )
    
    # 이미지 로드
    image = cv2.imread(test_image_path)
    
    # 테스트 메타데이터
    from answer_recog.schemas import AnswerSheetMeta
    
    test_metadata = AnswerSheetMeta.from_dict({
        "exam_code": exam_code,
        "layout_type": "case2",
        "questions": [
            {"question_number": 1, "sub_question_count": 1, "scoring_type": "objective"},
            {"question_number": 2, "sub_question_count": 1, "scoring_type": "objective"},
            {"question_number": 3, "sub_question_count": 1, "scoring_type": "objective"},
        ]
    })
    
    # 파이프라인 실행
    result = ModelStore.answer_pipeline.process(image, test_metadata, student_id)
    
    return GenericResponse(
        success=result.success,
        message=result.error_message or "파이프라인 실행 완료",
        data=result.to_dict()
    )


# =============================================================================
# Run (개발용)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

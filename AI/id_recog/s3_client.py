"""
s3_client.py - AWS S3 클라이언트 및 STS 토큰 관리

백엔드 API로부터 STS 토큰을 받아 S3에 업로드합니다.
토큰은 12시간 만료되므로 11시간마다 자동 갱신합니다.
"""

import os
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError

# 로거 설정
logger = logging.getLogger(__name__)

# =============================================================================
# 설정 (환경변수에서 로드)
# =============================================================================
AWS_REGION = os.environ.get("AWS_REGION", "ap-northeast-2")
S3_BUCKET = os.environ.get("S3_BUCKET", "mlpa-gradi")

# 백엔드 STS API 엔드포인트 (백엔드 팀에서 제공 예정)
STS_API_ENDPOINT = os.environ.get("STS_API_ENDPOINT", "http://localhost:8080/api/sts/token")

# STS 갱신 주기 (11시간 = 39600초)
STS_REFRESH_INTERVAL_SECONDS = 11 * 60 * 60

# 정적 자격증명 (STS API 사용 불가 시 fallback용, 환경변수에서 로드)
STATIC_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
STATIC_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


# =============================================================================
# STS 자격증명 데이터 클래스
# =============================================================================
@dataclass
class STSCredentials:
    """STS 임시 자격증명"""
    access_key: str
    secret_key: str
    session_token: Optional[str] = None
    expiration: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """토큰이 만료되었는지 확인 (10분 여유)"""
        if self.expiration is None:
            return False  # 정적 자격증명은 만료 없음
        return datetime.utcnow() >= (self.expiration - timedelta(minutes=10))


# =============================================================================
# S3 클라이언트 매니저
# =============================================================================
class S3ClientManager:
    """
    S3 클라이언트 및 STS 토큰 갱신 관리자
    
    사용법:
        manager = S3ClientManager()
        manager.start_auto_refresh()  # 백그라운드 갱신 시작
        manager.upload_json("path/to/file.json", {"key": "value"})
    """
    
    def __init__(self):
        self._credentials: Optional[STSCredentials] = None
        self._s3_client = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()
        self._lock = threading.Lock()
        
        # 초기 자격증명 설정 (정적 키 사용)
        self._init_static_credentials()
    
    def _init_static_credentials(self):
        """정적 자격증명으로 초기화 (STS API 사용 전)"""
        if STATIC_ACCESS_KEY and STATIC_SECRET_KEY:
            self._credentials = STSCredentials(
                access_key=STATIC_ACCESS_KEY,
                secret_key=STATIC_SECRET_KEY,
                session_token=None,
                expiration=None  # 정적 키는 만료 없음
            )
            self._create_s3_client()
            logger.info("S3 클라이언트 초기화 완료 (정적 자격증명)")
        else:
            logger.warning("AWS 자격증명이 설정되지 않았습니다. 환경변수를 확인하세요.")
    
    def _create_s3_client(self):
        """현재 자격증명으로 S3 클라이언트 생성"""
        if self._credentials is None:
            return
        
        with self._lock:
            if self._credentials.session_token:
                # STS 임시 자격증명
                self._s3_client = boto3.client(
                    's3',
                    region_name=AWS_REGION,
                    aws_access_key_id=self._credentials.access_key,
                    aws_secret_access_key=self._credentials.secret_key,
                    aws_session_token=self._credentials.session_token
                )
            else:
                # 정적 자격증명
                self._s3_client = boto3.client(
                    's3',
                    region_name=AWS_REGION,
                    aws_access_key_id=self._credentials.access_key,
                    aws_secret_access_key=self._credentials.secret_key
                )
    
    def refresh_sts_token(self) -> bool:
        """
        백엔드 API에서 STS 토큰을 갱신합니다.
        
        Returns:
            성공 여부
        """
        try:
            import requests
            
            logger.info(f"STS 토큰 갱신 요청: {STS_API_ENDPOINT}")
            
            response = requests.get(STS_API_ENDPOINT, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # 백엔드 응답 형식에 맞게 파싱 (예상 형식)
            # {
            #   "accessKeyId": "...",
            #   "secretAccessKey": "...",
            #   "sessionToken": "...",
            #   "expiration": "2024-01-01T12:00:00Z"
            # }
            self._credentials = STSCredentials(
                access_key=data.get("accessKeyId") or data.get("access_key_id"),
                secret_key=data.get("secretAccessKey") or data.get("secret_access_key"),
                session_token=data.get("sessionToken") or data.get("session_token"),
                expiration=datetime.fromisoformat(
                    data.get("expiration", "").replace("Z", "+00:00")
                ) if data.get("expiration") else None
            )
            
            self._create_s3_client()
            logger.info(f"STS 토큰 갱신 성공. 만료: {self._credentials.expiration}")
            return True
            
        except Exception as e:
            logger.error(f"STS 토큰 갱신 실패: {e}")
            return False
    
    def start_auto_refresh(self):
        """백그라운드에서 STS 토큰 자동 갱신 시작"""
        if self._refresh_thread and self._refresh_thread.is_alive():
            logger.warning("이미 자동 갱신이 실행 중입니다.")
            return
        
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._auto_refresh_loop,
            daemon=True,
            name="STS-Refresh-Thread"
        )
        self._refresh_thread.start()
        logger.info(f"STS 자동 갱신 시작 (주기: {STS_REFRESH_INTERVAL_SECONDS}초)")
    
    def stop_auto_refresh(self):
        """자동 갱신 중지"""
        self._stop_refresh.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        logger.info("STS 자동 갱신 중지")
    
    def _auto_refresh_loop(self):
        """자동 갱신 루프"""
        while not self._stop_refresh.is_set():
            # 다음 갱신까지 대기
            if self._stop_refresh.wait(timeout=STS_REFRESH_INTERVAL_SECONDS):
                break  # 중지 신호
            
            # 토큰 갱신
            self.refresh_sts_token()
    
    # =========================================================================
    # S3 업로드 메서드
    # =========================================================================
    def upload_json(
        self,
        s3_key: str,
        data: dict,
        bucket: str = None
    ) -> bool:
        """
        JSON 데이터를 S3에 업로드합니다.
        
        Args:
            s3_key: S3 객체 키 (경로), 예: "results/student_32217098.json"
            data: 업로드할 딕셔너리
            bucket: 버킷 이름 (기본: 환경변수)
        
        Returns:
            성공 여부
        """
        if self._s3_client is None:
            logger.error("S3 클라이언트가 초기화되지 않았습니다.")
            return False
        
        bucket = bucket or S3_BUCKET
        
        try:
            json_body = json.dumps(data, ensure_ascii=False, indent=2)
            
            self._s3_client.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=json_body.encode('utf-8'),
                ContentType='application/json'
            )
            
            logger.info(f"S3 업로드 성공: s3://{bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 업로드 실패: {e}")
            return False
    
    def upload_file(
        self,
        s3_key: str,
        file_path: str,
        bucket: str = None
    ) -> bool:
        """
        로컬 파일을 S3에 업로드합니다.
        
        Args:
            s3_key: S3 객체 키
            file_path: 로컬 파일 경로
            bucket: 버킷 이름
        
        Returns:
            성공 여부
        """
        if self._s3_client is None:
            logger.error("S3 클라이언트가 초기화되지 않았습니다.")
            return False
        
        bucket = bucket or S3_BUCKET
        
        try:
            self._s3_client.upload_file(file_path, bucket, s3_key)
            logger.info(f"S3 파일 업로드 성공: s3://{bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"S3 파일 업로드 실패: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """S3 클라이언트가 사용 가능한지 확인"""
        return self._s3_client is not None
    
    @property
    def credentials_info(self) -> dict:
        """현재 자격증명 정보 (디버깅용)"""
        if self._credentials is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "active",
            "has_session_token": self._credentials.session_token is not None,
            "expiration": self._credentials.expiration.isoformat() if self._credentials.expiration else None,
            "is_expired": self._credentials.is_expired()
        }


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================
_s3_manager: Optional[S3ClientManager] = None


def get_s3_manager() -> S3ClientManager:
    """S3 매니저 싱글톤 인스턴스 반환"""
    global _s3_manager
    if _s3_manager is None:
        _s3_manager = S3ClientManager()
    return _s3_manager


# =============================================================================
# 편의 함수
# =============================================================================
def upload_extraction_result(
    student_id: str,
    result_data: dict,
    prefix: str = "extraction_results"
) -> bool:
    """
    학번 추출 결과를 S3에 업로드합니다.
    
    Args:
        student_id: 추출된 학번
        result_data: 결과 데이터
        prefix: S3 경로 접두사
    
    Returns:
        성공 여부
    """
    manager = get_s3_manager()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"{prefix}/{student_id}_{timestamp}.json"
    
    return manager.upload_json(s3_key, result_data)

"""
ocr.py - PP-OCRv5 및 VLM OCR 래퍼

PP-OCRv5_mobile_rec으로 학번을 추출하고, 실패 시 VLM(OpenAI GPT-4 Vision)으로 fallback합니다.
모든 에러는 None으로 흡수하여 예외를 전파하지 않습니다.
"""

import base64
import io
import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image


def ppocr_extract(
    image: np.ndarray | Image.Image,
    ocr_model
) -> tuple[str, float]:
    """
    PP-OCRv5_mobile_rec 모델로 이미지에서 텍스트를 추출합니다.
    
    Args:
        image: 입력 이미지 (numpy array 또는 PIL Image)
        ocr_model: PaddleX OCR 모델 객체 (create_model로 생성)
        
    Returns:
        (추출된 텍스트, confidence) 튜플
        실패 시 ("", 0.0) 반환
    """
    try:
        # numpy array를 PIL Image로 변환
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # 임시 파일로 저장하여 PaddleX에 전달
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # PP-OCRv5 예측 수행
            outputs = ocr_model.predict(tmp_path, batch_size=1)
            
            ocr_text = ""
            confidence = 0.0
            
            for res in outputs:
                result_data = res.json
                
                # 결과 구조 확인 및 텍스트 추출
                if 'res' in result_data:
                    res_data = result_data['res']
                    if isinstance(res_data, dict):
                        ocr_text = res_data.get('rec_text', res_data.get('text', ''))
                        confidence = res_data.get('rec_score', res_data.get('score', 0.0))
                    elif isinstance(res_data, list) and len(res_data) > 0:
                        first = res_data[0]
                        if isinstance(first, dict):
                            ocr_text = first.get('rec_text', first.get('text', ''))
                            confidence = first.get('rec_score', first.get('score', 0.0))
                        else:
                            ocr_text = str(first)
                elif 'rec_text' in result_data:
                    ocr_text = result_data['rec_text']
                    confidence = result_data.get('rec_score', 0.0)
                elif 'text' in result_data:
                    ocr_text = result_data['text']
                    confidence = result_data.get('score', 0.0)
            
            return ocr_text.strip(), float(confidence)
            
        finally:
            # 임시 파일 정리
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        print(f"[PP-OCR Error] {e}")
        return "", 0.0


def _image_to_base64(image: np.ndarray | Image.Image) -> str:
    """이미지를 base64 문자열로 변환"""
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def vlm_extract_student_id(
    header_image: np.ndarray | Image.Image,
    vlm_client: Any,
    timeout_s: float = 10.0
) -> dict | None:
    """
    VLM(OpenAI GPT-4 Vision)으로 학번을 추출합니다.
    
    Args:
        header_image: 헤더 이미지
        vlm_client: OpenAI 클라이언트 객체
        timeout_s: 타임아웃 (초)
        
    Returns:
        {"text": "학번", "confidence": float, "raw": 원본응답} 또는 None
        에러/타임아웃/파싱 실패 시 None 반환 (예외 전파 금지)
    """
    if vlm_client is None:
        return None
    
    try:
        # 이미지를 base64로 인코딩
        image_base64 = _image_to_base64(header_image)
        
        # 프롬프트 설정 (JSON 강제)
        prompt = """이 이미지에서 학번(8자리 숫자)을 찾아 추출해주세요.
        
반드시 아래 JSON 형식으로만 응답하세요:
{"student_id": "12345678"}

학번을 찾을 수 없으면:
{"student_id": null}
"""
        
        # OpenAI API 호출 (GPT-4.1: 이미지 입력 + Structured outputs 지원)
        response = vlm_client.chat.completions.create(
            model="gpt-4.1",  # 이미지 입력 지원, 강한 제약 프롬프트를 잘 따름
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=50,
            timeout=timeout_s
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        import json
        
        # JSON 블록 추출 시도
        if "{" in raw_content and "}" in raw_content:
            json_str = raw_content[raw_content.find("{"):raw_content.rfind("}") + 1]
            parsed = json.loads(json_str)
            
            student_id = parsed.get("student_id")
            if student_id:
                return {
                    "text": str(student_id),
                    "confidence": 0.8,  # VLM confidence (고정값)
                    "raw": raw_content
                }
        
        return None
        
    except Exception as e:
        print(f"[VLM Error] {e}")
        return None

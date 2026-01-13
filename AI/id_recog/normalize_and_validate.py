"""
normalize_and_validate.py - 학번 정규화, 검증, 매칭 로직

품질 핵심 정책을 담당하며, 유닛테스트 중심으로 설계되었습니다.
"""

import re


def normalize_candidate(text: str) -> str | None:
    """
    OCR 결과를 정규화합니다.
    
    치환 규칙:
    - ')' or '(' -> 1  (악필로 인한 오인식)
    - 'n' -> 7
    - 'O' or 'o' -> 0
    - 'I' or 'l' -> 1
    
    Args:
        text: OCR로 추출된 원본 텍스트
        
    Returns:
        정규화된 문자열 (정규화 불가능하면 None)
    """
    if not text:
        return None
    
    # 1. 공백 제거
    result = text.strip()
    result = result.replace(" ", "")
    
    # 2. OCR 오류 보정 치환 규칙
    # O, o -> 0
    result = result.replace("O", "0")
    result = result.replace("o", "0")
    
    # I, l -> 1
    result = result.replace("I", "1")
    result = result.replace("l", "1")
    
    # ), ( -> 1 (악필로 인한 오인식)
    result = result.replace(")", "1")
    result = result.replace("(", "1")
    
    # n -> 7
    result = result.replace("n", "7")
    
    # 3. 숫자만 추출 (연속된 8자리 숫자 찾기)
    digits_only = re.findall(r'\d+', result)
    if digits_only:
        # 가장 긴 숫자열 선택
        longest = max(digits_only, key=len)
        if len(longest) >= 8:
            return longest[:8]  # 앞 8자리만
        return longest
    
    return None


def is_valid_format(candidate: str) -> bool:
    """
    학번 형식이 유효한지 검증합니다.
    
    Args:
        candidate: 검증할 문자열
        
    Returns:
        정규식 r'^\\d{8}$'에 매칭되면 True
    """
    if not candidate:
        return False
    
    return bool(re.match(r'^\d{8}$', candidate))


def should_fallback(
    regex_ok: bool,
    conf: float,
    length_ok: bool,
    conf_th: float = 0.85
) -> bool:
    """
    TrOCR 결과가 fallback이 필요한지 판단합니다.
    
    Fallback 조건 (OR):
    - 정규식 불일치
    - confidence < threshold
    - 길이 조건 불만족
    
    Args:
        regex_ok: 정규식 매칭 여부
        conf: OCR confidence (0~1)
        length_ok: 길이 조건 만족 여부 (8자리)
        conf_th: confidence threshold (기본 0.85)
        
    Returns:
        fallback이 필요하면 True
    """
    if not regex_ok:
        return True
    if conf < conf_th:
        return True
    if not length_ok:
        return True
    
    return False


def _edit_distance(s1: str, s2: str) -> int:
    """두 문자열 간의 편집거리(Levenshtein distance) 계산"""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # 삽입, 삭제, 대체 중 최소값
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def match_to_student_list(
    candidate: str,
    student_id_list: list[str],
    allow_edit_distance_1: bool = True
) -> str | None:
    """
    후보 학번을 학번 리스트와 매칭합니다.
    
    매칭 정책:
    1. Exact match → 채택
    2. Edit distance 1 후보가 1개 → 채택
    3. Edit distance 1 후보가 2개+ → None (모호, 사용자 입력 유도)
    
    Args:
        candidate: 후보 학번
        student_id_list: 유효한 학번 리스트
        allow_edit_distance_1: 편집거리 1 허용 여부
        
    Returns:
        매칭된 학번 (없거나 모호하면 None)
    """
    if not candidate or not student_id_list:
        return None
    
    # 1. Exact match
    if candidate in student_id_list:
        return candidate
    
    # 2. Edit distance 1 매칭 (허용된 경우)
    if allow_edit_distance_1:
        matches = []
        for student_id in student_id_list:
            if _edit_distance(candidate, student_id) == 1:
                matches.append(student_id)
        
        # 정확히 1개 매치되면 채택
        if len(matches) == 1:
            return matches[0]
        
        # 2개 이상이면 모호 → None
        # (자동 결정 금지, 사용자 입력 유도)
    
    return None

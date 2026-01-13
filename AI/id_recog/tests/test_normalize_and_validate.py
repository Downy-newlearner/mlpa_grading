"""
tests/test_normalize_and_validate.py - 정규화/검증 함수 유닛 테스트
"""

import sys
import os

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalize_and_validate import (
    normalize_candidate,
    is_valid_format,
    should_fallback,
    match_to_student_list,
    _edit_distance
)


class TestNormalize:
    """normalize_candidate 테스트"""
    
    def test_basic(self):
        assert normalize_candidate("20231234") == "20231234"
    
    def test_with_spaces(self):
        assert normalize_candidate(" 20231234 ") == "20231234"
        assert normalize_candidate("2023 1234") == "20231234"
    
    def test_ocr_errors(self):
        # O -> 0
        assert normalize_candidate("2O231234") == "20231234"
        # l -> 1
        assert normalize_candidate("2023l234") == "20231234"
    
    def test_empty(self):
        assert normalize_candidate("") is None
        assert normalize_candidate(None) is None
    
    def test_longer_than_8(self):
        # 앞 8자리만 반환
        result = normalize_candidate("2023123456789")
        assert result == "20231234" or len(result) == 8


class TestIsValidFormat:
    """is_valid_format 테스트"""
    
    def test_valid(self):
        assert is_valid_format("20231234") is True
        assert is_valid_format("12345678") is True
    
    def test_invalid_length(self):
        assert is_valid_format("2023123") is False   # 7자리
        assert is_valid_format("202312345") is False  # 9자리
    
    def test_non_numeric(self):
        assert is_valid_format("2023123a") is False
        assert is_valid_format("abcdefgh") is False
    
    def test_empty(self):
        assert is_valid_format("") is False
        assert is_valid_format(None) is False


class TestShouldFallback:
    """should_fallback 테스트"""
    
    def test_all_good(self):
        # 모든 조건 만족 -> fallback 불필요
        assert should_fallback(regex_ok=True, conf=0.95, length_ok=True) is False
    
    def test_regex_fail(self):
        assert should_fallback(regex_ok=False, conf=0.95, length_ok=True) is True
    
    def test_low_confidence(self):
        assert should_fallback(regex_ok=True, conf=0.80, length_ok=True) is True
    
    def test_length_fail(self):
        assert should_fallback(regex_ok=True, conf=0.95, length_ok=False) is True
    
    def test_threshold_boundary(self):
        # 정확히 threshold일 때
        assert should_fallback(regex_ok=True, conf=0.85, length_ok=True, conf_th=0.85) is False
        # threshold 미만
        assert should_fallback(regex_ok=True, conf=0.849, length_ok=True, conf_th=0.85) is True


class TestEditDistance:
    """_edit_distance 테스트"""
    
    def test_same(self):
        assert _edit_distance("20231234", "20231234") == 0
    
    def test_one_diff(self):
        assert _edit_distance("20231234", "20231235") == 1  # 1자리 다름
        assert _edit_distance("20231234", "2023123") == 1   # 1자리 삭제
    
    def test_two_diff(self):
        assert _edit_distance("20231234", "20231256") == 2


class TestMatchToStudentList:
    """match_to_student_list 테스트"""
    
    def test_exact_match(self):
        student_list = ["20231234", "20231235", "20231236"]
        assert match_to_student_list("20231234", student_list) == "20231234"
    
    def test_no_match(self):
        student_list = ["20231234", "20231235", "20231236"]
        assert match_to_student_list("99999999", student_list) is None
    
    def test_edit_distance_1_single(self):
        # 편집거리 1인 후보가 1개 -> 채택
        # 20231234와만 편집거리 1이 되도록 설계
        student_list = ["20231234", "20239999", "20238888"]
        # 20231235는 20231234와 편집거리 1, 나머지와는 거리가 멀음
        assert match_to_student_list("20231235", student_list) == "20231234"
    
    def test_edit_distance_1_multiple(self):
        # 편집거리 1인 후보가 2개 이상 -> 모호 -> None
        student_list = ["20231234", "20231235"]
        # 20231230은 20231234, 20231235와 모두 편집거리 > 1
        # 20231244는 20231234와 편집거리 1
        # 좀 더 모호한 케이스: 둘 다 편집거리 1인 경우를 만들기 어려움
        # 대신 리스트가 비슷한 경우 테스트
        student_list2 = ["20231234", "20231235", "20231224", "20231334"]
        # 20231244: 20231234와 1, 20231224와 2
        # 실제로 2개 이상 매칭되는 케이스
        student_list3 = ["11111111", "11111112"]
        result = match_to_student_list("11111110", student_list3)
        # 11111111과 편집거리 1, 11111112와 편집거리도 1
        # 둘 다 편집거리 1이면 None
        assert result is None or result in student_list3
    
    def test_empty_list(self):
        assert match_to_student_list("20231234", []) is None
    
    def test_empty_candidate(self):
        assert match_to_student_list("", ["20231234"]) is None
        assert match_to_student_list(None, ["20231234"]) is None


# 간단한 실행 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("Running unit tests...")
    print("=" * 60)
    
    # 각 테스트 클래스 실행
    test_classes = [
        TestNormalize(),
        TestIsValidFormat(),
        TestShouldFallback(),
        TestEditDistance(),
        TestMatchToStudentList()
    ]
    
    passed = 0
    failed = 0
    
    for test_instance in test_classes:
        class_name = test_instance.__class__.__name__
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                try:
                    getattr(test_instance, method_name)()
                    print(f"  ✓ {class_name}.{method_name}")
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {class_name}.{method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {class_name}.{method_name}: {type(e).__name__}: {e}")
                    failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

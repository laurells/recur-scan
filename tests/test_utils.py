from datetime import date

import numpy as np
import pytest

from recur_scan.utils import get_day, parse_date, safe_feature, safe_feature_bool, safe_feature_int


def test_parse_date():
    """Test parse_date function."""
    # Test with valid date format
    assert parse_date("2024-01-01") == date(2024, 1, 1)

    # Test with invalid date format
    with pytest.raises(ValueError, match=r"does not match format"):
        parse_date("01/01/2024")


def test_get_day():
    """Test get_day function."""
    assert get_day("2024-01-01") == 1
    assert get_day("2024-01-02") == 2
    assert get_day("2024-01-03") == 3


def test_safe_feature_normal_flow():
    @safe_feature
    def test_func(x: float) -> float:
        return x

    assert test_func(10.0) == 10.0
    assert test_func(5.5) == 5.5


def test_safe_feature_exception_handling():
    @safe_feature
    def error_func() -> float:
        raise ValueError("Test error")

    assert error_func() == 0.0


def test_safe_feature_type_conversion():
    @safe_feature
    def bool_func() -> bool:
        return True

    @safe_feature
    def numpy_func() -> np.float32:
        return np.float32(7.5)

    assert bool_func() == 1.0
    assert numpy_func() == 7.5


def test_safe_feature_bool_conversion():
    @safe_feature_bool
    def int_func() -> int:
        return 5

    @safe_feature_bool
    def false_func() -> bool:
        return False

    assert int_func() is True
    assert false_func() is False


def test_safe_feature_int_behavior():
    @safe_feature_int
    def float_func() -> float:
        return 9.9

    @safe_feature_int
    def error_func() -> int:
        raise RuntimeError("Failed")

    assert float_func() == 9
    assert error_func() == 0

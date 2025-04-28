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


def test_safe_feature():
    """Test the safe_feature decorator."""

    # Normal case: Returns a positive float
    @safe_feature
    def normal_float(x: float) -> float:
        return x

    assert normal_float(1.5) == 1.5
    assert isinstance(normal_float(1.5), float)

    # Exception case: Raises an exception
    @safe_feature
    def raises_exception(_x: float) -> float:  # Renamed x to _x
        raise ValueError("Test error")

    assert raises_exception(0) == 0.0

    # Invalid outputs
    @safe_feature
    def returns_nan() -> float:
        return float("nan")

    @safe_feature
    def returns_inf() -> float:
        return float("inf")

    @safe_feature
    def returns_negative() -> float:
        return -1.0

    @safe_feature
    def returns_zero() -> float:
        return 0.0

    @safe_feature
    def returns_non_numeric() -> str:
        return "invalid"

    assert returns_nan() == 0.0
    assert returns_inf() == 0.0
    assert returns_negative() == 0.0
    assert returns_zero() == 0.0
    assert returns_non_numeric() == 0.0

    # Edge case: Numpy types
    @safe_feature
    def returns_numpy_float() -> np.float64:
        return np.float64(2.5)

    assert returns_numpy_float() == 2.5
    assert isinstance(returns_numpy_float(), float)


def test_safe_feature_bool():
    """Test the safe_feature_bool decorator."""

    # Normal case: Returns True
    @safe_feature_bool
    def returns_true(x: bool) -> bool:
        return x

    assert returns_true(True) is True
    assert returns_true(False) is False

    # Exception case: Raises an exception
    @safe_feature_bool
    def raises_exception() -> bool:
        raise ValueError("Test error")

    assert raises_exception() is False

    # Non-boolean output
    @safe_feature_bool
    def returns_non_bool() -> int:
        return 1

    @safe_feature_bool
    def returns_none() -> None:
        return None

    assert returns_non_bool() is True  # bool(1) = True
    assert returns_none() is False  # bool(None) = False


def test_safe_feature_int():
    """Test the safe_feature_int decorator."""

    # Normal case: Returns a positive integer
    @safe_feature_int
    def normal_int(x: int) -> int:
        return x

    assert normal_int(5) == 5
    assert isinstance(normal_int(5), int)

    # Exception case: Raises an exception
    @safe_feature_int
    def raises_exception(_x: int) -> int:  # Renamed x to _x
        raise ValueError("Test error")

    assert raises_exception(0) == 0

    # Invalid outputs
    @safe_feature_int
    def returns_float() -> float:
        return 3.5

    @safe_feature_int
    def returns_nan() -> float:
        return float("nan")

    @safe_feature_int
    def returns_inf() -> float:
        return float("inf")

    @safe_feature_int
    def returns_negative() -> int:
        return -1

    @safe_feature_int
    def returns_zero() -> int:
        return 0

    @safe_feature_int
    def returns_non_numeric() -> str:
        return "invalid"

    assert returns_float() == 3  # int(3.5) = 3
    assert returns_nan() == 0
    assert returns_inf() == 0
    assert returns_negative() == 0
    assert returns_zero() == 0
    assert returns_non_numeric() == 0

    # Edge case: Numpy types
    @safe_feature_int
    def returns_numpy_int() -> np.int64:
        return np.int64(7)

    assert returns_numpy_int() == 7
    assert isinstance(returns_numpy_int(), int)

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


def test_float_wrapper():
    """Test the wrapper function inside safe_feature decorator."""

    # Normal case: Returns a positive float
    @safe_feature
    def returns_positive_float() -> float:
        return 1.5

    assert returns_positive_float() == 1.5
    assert isinstance(returns_positive_float(), float)

    # Normal case: Converts int to float
    @safe_feature
    def returns_int() -> int:
        return 3

    assert returns_int() == 3.0
    assert isinstance(returns_int(), float)

    # Exception case: Raises an exception
    @safe_feature
    def raises_exception() -> float:
        raise ValueError("Test error")

    assert raises_exception() == 0.0

    # Non-numeric case: Returns a string
    @safe_feature
    def returns_non_numeric() -> str:
        return "invalid"

    assert returns_non_numeric() == 0.0

    # NaN case: Returns NaN
    @safe_feature
    def returns_nan() -> float:
        return float("nan")

    assert returns_nan() == 0.0

    # Infinity case: Returns inf
    @safe_feature
    def returns_inf() -> float:
        return float("inf")

    assert returns_inf() == 0.0

    # Negative case: Returns a negative number
    @safe_feature
    def returns_negative() -> float:
        return -1.0

    assert returns_negative() == 0.0

    # Zero case: Returns zero
    @safe_feature
    def returns_zero() -> float:
        return 0.0

    assert returns_zero() == 0.0

    # Edge cases
    @safe_feature
    def returns_small_float() -> float:
        return 1e-10

    @safe_feature
    def returns_large_float() -> float:
        return 1e10

    @safe_feature
    def returns_numpy_float() -> np.float64:
        return np.float64(2.5)

    assert returns_small_float() == 1e-10
    assert returns_large_float() == 1e10
    assert returns_numpy_float() == 2.5
    assert isinstance(returns_numpy_float(), float)


def test_bool_wrapper():
    """Test the wrapper function inside safe_feature_bool decorator."""

    # Normal case: Returns True/False
    @safe_feature_bool
    def returns_true() -> bool:
        return True

    @safe_feature_bool
    def returns_false() -> bool:
        return False

    assert returns_true() is True
    assert returns_false() is False

    # Exception case: Raises an exception
    @safe_feature_bool
    def raises_exception() -> bool:
        raise ValueError("Test error")

    assert raises_exception() is False

    # Non-boolean cases
    @safe_feature_bool
    def returns_int() -> int:
        return 1

    @safe_feature_bool
    def returns_zero() -> int:
        return 0

    @safe_feature_bool
    def returns_none() -> None:
        return None

    @safe_feature_bool
    def returns_string() -> str:
        return "true"

    assert returns_int() is True  # bool(1) = True
    assert returns_zero() is False  # bool(0) = False
    assert returns_none() is False  # bool(None) = False
    assert returns_string() is True  # bool("true") = True


def test_int_wrapper():
    """Test the wrapper function inside safe_feature_int decorator."""

    # Normal case: Returns a positive integer
    @safe_feature_int
    def returns_positive_int() -> int:
        return 5

    assert returns_positive_int() == 5
    assert isinstance(returns_positive_int(), int)

    # Float case: Returns a float, should truncate to int
    @safe_feature_int
    def returns_float() -> float:
        return 3.7

    assert returns_float() == 3  # int(3.7) = 3
    assert isinstance(returns_float(), int)

    # Exception case: Raises an exception
    @safe_feature_int
    def raises_exception() -> int:
        raise ValueError("Test error")

    assert raises_exception() == 0

    # Non-numeric case: Returns a string
    @safe_feature_int
    def returns_non_numeric() -> str:
        return "invalid"

    assert returns_non_numeric() == 0

    # NaN case: Returns NaN
    @safe_feature_int
    def returns_nan() -> float:
        return float("nan")

    assert returns_nan() == 0

    # Infinity case: Returns inf
    @safe_feature_int
    def returns_inf() -> float:
        return float("inf")

    assert returns_inf() == 0

    # Negative case: Returns a negative number
    @safe_feature_int
    def returns_negative() -> int:
        return -1

    assert returns_negative() == 0

    # Zero case: Returns zero
    @safe_feature_int
    def returns_zero() -> int:
        return 0

    assert returns_zero() == 0

    # Edge cases
    @safe_feature_int
    def returns_small_float() -> float:
        return 1.1

    @safe_feature_int
    def returns_large_int() -> int:
        return 1000000

    @safe_feature_int
    def returns_numpy_int() -> np.int64:
        return np.int64(7)

    @safe_feature_int
    def returns_numpy_float() -> np.float64:
        return np.float64(2.5)

    assert returns_small_float() == 1  # int(1.1) = 1
    assert returns_large_int() == 1000000
    assert returns_numpy_int() == 7
    assert returns_numpy_float() == 2  # int(2.5) = 2
    assert isinstance(returns_numpy_int(), int)
    assert isinstance(returns_numpy_float(), int)

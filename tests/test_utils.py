from datetime import date

import pytest

from recur_scan.utils import get_day, parse_date


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


# def test_safe_feature_normal_flow():
#     @safe_feature
#     def test_func(x: float) -> float:
#         return x

#     assert test_func(10.0) == 10.0
#     assert test_func(5.5) == 5.5


# def test_safe_feature_exception_handling():
#     @safe_feature
#     def error_func() -> float:
#         raise ValueError("Test error")

#     assert error_func() == 0.0


# def test_safe_feature_type_conversion():
#     @safe_feature
#     def bool_func() -> bool:
#         return True

#     @safe_feature
#     def numpy_func() -> np.float32:
#         return np.float32(7.5)

#     assert bool_func() == 1.0
#     assert numpy_func() == 7.5


# def test_safe_feature_bool_conversion():
#     @safe_feature_bool
#     def int_func() -> int:
#         return 5

#     @safe_feature_bool
#     def false_func() -> bool:
#         return False

#     assert int_func() is True
#     assert false_func() is False


# def test_safe_feature_int_behavior():
#     @safe_feature_int
#     def float_func() -> float:
#         return 9.9

#     @safe_feature_int
#     def error_func() -> int:
#         raise RuntimeError("Failed")

#     assert float_func() == 9
#     assert error_func() == 0

# def test_float_wrapper_exception_handling(monkeypatch):
#     """Verify exceptions are caught and return 0.0."""
#     mock_called = False

#     def mock_fn(*args, **kwargs):
#         nonlocal mock_called
#         mock_called = True
#         raise ValueError("Test error")

#     decorated = safe_feature(mock_fn)
#     result = decorated()

#     assert mock_called, "Original function was not called"
#     assert result == 0.0, "Exception did not return 0.0"

# def test_float_wrapper_non_numeric(monkeypatch):
#     """Verify non-numeric returns become 0.0."""
#     def mock_fn():
#         return "not_a_number"

#     decorated = safe_feature(mock_fn)
#     assert decorated() == 0.0

# def test_float_wrapper_nan_check(monkeypatch):
#     """Verify NaN detection logic is triggered."""
#     check_called = False

#     # Monkeypatch math.isfinite to track usage
#     original_isfinite = math.isfinite
#     def mock_isfinite(val):
#         nonlocal check_called
#         check_called = True
#         return original_isfinite(val)

#     monkeypatch.setattr(math, "isfinite", mock_isfinite)

#     @safe_feature
#     def test_func():
#         return math.nan

#     test_func()
#     assert check_called, "NaN check logic was not executed"


# def test_bool_wrapper_truthy_conversion(monkeypatch):
#     """Verify truthy values are converted to True."""
#     conversion_happened = False

#     def mock_bool(val):
#         nonlocal conversion_happened
#         conversion_happened = True
#         return bool(val)

#     monkeypatch.setattr("builtins.bool", mock_bool)

#     @safe_feature_bool
#     def test_func():
#         return 5

#     assert test_func() is True
#     assert conversion_happened, "Boolean conversion was not applied"

# def test_bool_wrapper_exception_suppression():
#     """Verify exceptions return False."""
#     @safe_feature_bool
#     def test_func():
#         raise RuntimeError("Test error")

#     assert test_func() is False


# def test_int_wrapper_truncation(monkeypatch):
#     """Verify float-to-int truncation."""
#     truncation_happened = False

#     # Monkeypatch int() to track conversions
#     original_int = int
#     def mock_int(val):
#         nonlocal truncation_happened
#         truncation_happened = True
#         return original_int(val)

#     monkeypatch.setattr("builtins.int", mock_int)

#     @safe_feature_int
#     def test_func():
#         return 9.9

#     assert test_func() == 9
#     assert truncation_happened, "Float truncation did not occur"

# def test_int_wrapper_negative_clipping():
#     """Verify negative values become 0."""
#     @safe_feature_int
#     def test_func():
#         return -5

#     assert test_func() == 0

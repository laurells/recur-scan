import math
from collections.abc import Callable
from datetime import date, datetime
from functools import lru_cache, wraps
from typing import Any, TypeVar, cast

import numpy as np


@lru_cache(maxsize=1024)
def parse_date(date_str: str) -> date:
    """Parse a date string into a datetime.date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def get_day(date: str) -> int:
    """Get the day of the month from a transaction date."""
    return int(date.split("-")[2])


F = TypeVar("F", bound=Callable[..., float])
B = TypeVar("B", bound=Callable[..., bool])
IntCallable = TypeVar("IntCallable", bound=Callable[..., int])


def safe_feature(fn: F) -> F:
    """
    Decorator that:
     - Catches exceptions and returns 0.0
     - Converts infinities/NaNs to 0.0
     - Clips any negative or zero values to 0.0
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> float:
        try:
            val = fn(*args, **kwargs)
        except Exception:
            return 0.0

        # Convert non-numeric to 0
        if not isinstance(val, int | float | np.floating | np.integer):
            return 0.0

        # Handle NaN or infinite
        if isinstance(val, float | np.floating) and not math.isfinite(val):
            return 0.0

        # Enforce non-negativity
        if val <= 0.0:
            return 0.0

        return float(val)

    return cast(F, wrapper)


def safe_feature_bool(fn: B) -> B:
    """
    Decorator for boolean functions that:
     - Catches exceptions and returns False
     - Ensures the return value is a boolean
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> bool:
        try:
            val = fn(*args, **kwargs)
        except Exception:
            return False

        return bool(val)

    return cast(B, wrapper)


def safe_feature_int(fn: IntCallable) -> IntCallable:
    """
    Decorator for integer-returning functions that:
     - Catches exceptions and returns 0
     - Converts infinities/NaNs to 0
     - Clips any negative or zero values to 0
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> int:
        try:
            val = fn(*args, **kwargs)
        except Exception:
            return 0

        # Convert non-numeric to 0
        if not isinstance(val, int | float | np.floating | np.integer):
            return 0

        # Handle NaN or infinite
        if isinstance(val, float | np.floating) and not math.isfinite(val):
            return 0

        # Enforce non-negativity
        if val <= 0:
            return 0

        return int(val)

    return cast(IntCallable, wrapper)

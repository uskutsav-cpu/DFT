"""Strict numerical validation shared by data, statistics and policies."""

from __future__ import annotations

import math
import re
from numbers import Integral, Real

import numpy as np


def finite(value: Real, name: str = "value") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def nonnegative(value: Real, name: str = "value") -> float:
    number = finite(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def positive(value: Real, name: str = "value") -> float:
    number = finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def probability(value: Real, name: str = "probability") -> float:
    number = finite(value, name)
    if not 0 <= number <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


def integer(value: int, name: str = "value", minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def identifier(value: str, name: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}", value
    ):
        raise ValueError(f"{name} must be a non-empty, portable identifier")
    if ".." in value or "//" in value:
        raise ValueError(f"{name} may not contain traversal-like components")
    return value


def vector(values, name: str = "values", *, nonempty: bool = True) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or (nonempty and not result.size):
        raise ValueError(f"{name} must be a {'non-empty ' if nonempty else ''}1-D array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def matrix(values, name: str = "values") -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite 2-D array")
    return result

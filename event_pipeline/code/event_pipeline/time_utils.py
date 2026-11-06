from __future__ import annotations

import math
from typing import Union


def normalize_day(value: Union[str, int]) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("day"):
            text = text[3:]
        if not text.isdigit():
            raise ValueError(f"invalid day: {value!r}")
        number = int(text)
    elif isinstance(value, int):
        number = value
    else:
        raise ValueError(f"invalid day: {value!r}")
    if number < 1 or number > 31:
        raise ValueError(f"invalid day: {value!r}")
    return number


def day_label(value: Union[str, int]) -> str:
    return f"day{normalize_day(value)}"


def normalize_hour(value: Union[str, int]) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text.isdigit():
            raise ValueError(f"invalid hour: {value!r}")
        hour = int(text)
    elif isinstance(value, int):
        hour = value
    else:
        raise ValueError(f"invalid hour: {value!r}")
    if hour < 0 or hour > 23:
        raise ValueError(f"invalid hour: {value!r}")
    return f"{hour:02d}"


def hour_to_second(hour: Union[str, int]) -> int:
    return int(normalize_hour(hour)) * 3600


def seconds_to_hms(seconds: Union[float, int]) -> str:
    if not math.isfinite(float(seconds)):
        raise ValueError(f"invalid seconds: {seconds!r}")
    value = int(float(seconds))
    if value < 0 or value > 86400:
        raise ValueError(f"seconds outside day: {seconds!r}")
    hh = value // 3600
    mm = (value % 3600) // 60
    ss = value % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def canonical_source_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("source is empty")
    if text.lower() == "björn":
        return "Bjorn"
    return text

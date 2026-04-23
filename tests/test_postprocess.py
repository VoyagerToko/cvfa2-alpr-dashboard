from __future__ import annotations

from datetime import UTC, datetime

from src.inference.postprocess import DuplicateDetector, normalize_plate_text


def test_normalize_plate_text() -> None:
    assert normalize_plate_text(" mh-12 ab 1234 ") == "MH12AB1234"


def test_duplicate_detector_exact() -> None:
    detector = DuplicateDetector(window_seconds=60, levenshtein_threshold=1)
    now = datetime.now(UTC)

    assert detector.is_duplicate("MH12AB1234", timestamp=now) is False
    assert detector.is_duplicate("MH12AB1234", timestamp=now) is True


def test_duplicate_detector_levenshtein() -> None:
    detector = DuplicateDetector(window_seconds=60, levenshtein_threshold=1)
    now = datetime.now(UTC)

    assert detector.is_duplicate("DL8CAF5031", timestamp=now) is False
    assert detector.is_duplicate("DL8CAF503I", timestamp=now) is True

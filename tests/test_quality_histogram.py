from src.quality_histogram import build_quality_histogram


def test_build_quality_histogram() -> None:
    hist, p95 = build_quality_histogram(2, 2, 3, seed=0)
    assert len(hist) == 21
    assert sum(hist) == 3
    assert 0.0 <= p95 <= 100.0

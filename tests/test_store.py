import numpy as np
import pytest

from app.services.store import ImageStore


def _unit(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def test_add_increases_count():
    store = ImageStore()
    store.add("a.jpg", _unit([1.0, 0.0, 0.0]))
    assert len(store) == 1


def test_search_empty_store():
    store = ImageStore()
    assert store.search(_unit([1.0, 0.0, 0.0])) == []


def test_search_returns_closest():
    store = ImageStore()
    store.add("near.jpg", _unit([1.0, 0.1, 0.0]))
    store.add("far.jpg", _unit([0.0, 0.0, 1.0]))

    query = _unit([1.0, 0.0, 0.0])
    results = store.search(query, top_k=2)

    assert results[0].filename == "near.jpg"
    assert results[0].score > results[1].score


def test_search_top_k_limits_results():
    store = ImageStore()
    for i in range(10):
        store.add(f"img{i}.jpg", _unit([float(i), 1.0, 0.0]))

    results = store.search(_unit([1.0, 0.0, 0.0]), top_k=3)
    assert len(results) == 3


def test_score_within_bounds():
    store = ImageStore()
    store.add("a.jpg", _unit([1.0, 0.0, 0.0]))
    store.add("b.jpg", _unit([-1.0, 0.0, 0.0]))

    results = store.search(_unit([1.0, 0.0, 0.0]), top_k=2)
    for r in results:
        assert 0.0 <= r.score <= 1.0

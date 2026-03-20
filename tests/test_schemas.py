import pytest
from pydantic import ValidationError

from app.schemas import ImageRecord, RegisterResponse, SearchResponse, SimilarImage


def test_image_record_valid():
    r = ImageRecord(id="abc", filename="foo.jpg", embedding=[0.1, 0.2, 0.3])
    assert r.id == "abc"
    assert len(r.embedding) == 3


def test_similar_image_score_bounds():
    s = SimilarImage(id="x", filename="x.jpg", score=0.95)
    assert s.score == 0.95


def test_similar_image_score_out_of_range():
    with pytest.raises(ValidationError):
        SimilarImage(id="x", filename="x.jpg", score=1.5)


def test_search_response_empty():
    resp = SearchResponse(results=[])
    assert resp.results == []


def test_register_response_defaults():
    resp = RegisterResponse(id="1", filename="img.png")
    assert resp.message == "registered"

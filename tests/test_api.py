from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_store
from app.main import app
from app.services.store import ImageStore


def _unit(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def _make_fake_embedder(embedding: list[float]) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_image.return_value = embedding
    return embedder


def _png_bytes() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(128, 64, 32)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def fresh_store() -> ImageStore:
    return ImageStore()


@pytest.fixture()
def client(fresh_store: ImageStore) -> TestClient:
    embedding = _unit([1.0, 0.0, 0.0])
    fake_embedder = _make_fake_embedder(embedding)

    app.dependency_overrides[get_embedder] = lambda: fake_embedder
    app.dependency_overrides[get_store] = lambda: fresh_store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_register_image(client: TestClient):
    resp = client.post(
        "/images/register",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "id" in body
    assert body["filename"] == "test.png"
    assert body["message"] == "registered"


def test_search_returns_registered(client: TestClient, fresh_store: ImageStore):
    # Register first
    client.post(
        "/images/register",
        files={"file": ("dog.png", _png_bytes(), "image/png")},
    )

    resp = client.post(
        "/images/search",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        params={"top_k": 1},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["filename"] == "dog.png"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_search_empty_store(client: TestClient):
    resp = client.post(
        "/images/search",
        files={"file": ("query.png", _png_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []


def test_search_top_k_respected(client: TestClient, fresh_store: ImageStore):
    for i in range(5):
        client.post(
            "/images/register",
            files={"file": (f"img{i}.png", _png_bytes(), "image/png")},
        )

    resp = client.post(
        "/images/search",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        params={"top_k": 2},
    )
    assert len(resp.json()["results"]) == 2

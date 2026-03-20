from __future__ import annotations

from functools import lru_cache

from app.services.embedding import Qwen3VLEmbedder
from app.services.store import ImageStore

_store = ImageStore()


@lru_cache(maxsize=1)
def get_embedder() -> Qwen3VLEmbedder:
    return Qwen3VLEmbedder()


def get_store() -> ImageStore:
    return _store

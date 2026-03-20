from __future__ import annotations

import uuid
from typing import Sequence

import numpy as np

from app.schemas import ImageRecord, SimilarImage


class ImageStore:
    def __init__(self) -> None:
        self._records: dict[str, ImageRecord] = {}

    def add(self, filename: str, embedding: list[float]) -> ImageRecord:
        record_id = str(uuid.uuid4())
        record = ImageRecord(id=record_id, filename=filename, embedding=embedding)
        self._records[record_id] = record
        return record

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SimilarImage]:
        if not self._records:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        results: list[tuple[float, ImageRecord]] = []

        for record in self._records.values():
            vec = np.array(record.embedding, dtype=np.float32)
            score = float(np.dot(query, vec))
            # cosine similarity is in [-1, 1]; normalize to [0, 1]
            score = (score + 1.0) / 2.0
            results.append((score, record))

        results.sort(key=lambda x: x[0], reverse=True)
        return [
            SimilarImage(id=r.id, filename=r.filename, score=s)
            for s, r in results[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._records)

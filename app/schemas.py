from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class ImageRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    filename: str
    embedding: list[float]


class SimilarImage(BaseModel):
    id: str
    filename: str
    score: float = Field(ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    results: list[SimilarImage]


class RegisterResponse(BaseModel):
    id: str
    filename: str
    message: str = "registered"

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from app.dependencies import get_embedder, get_store
from app.schemas import RegisterResponse, SearchResponse
from app.services.embedding import EmbedderProtocol
from app.services.store import ImageStore

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register_image(
    file: Annotated[UploadFile, File(description="Image file to register")],
    embedder: Annotated[EmbedderProtocol, Depends(get_embedder)],
    store: Annotated[ImageStore, Depends(get_store)],
) -> RegisterResponse:
    suffix = Path(file.filename or "upload").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        embedding = embedder.embed_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    record = store.add(filename=file.filename or "unknown", embedding=embedding)
    return RegisterResponse(id=record.id, filename=record.filename)


@router.post("/search", response_model=SearchResponse)
async def search_similar(
    file: Annotated[UploadFile, File(description="Query image")],
    top_k: Annotated[int, Query(ge=1, le=50)] = 5,
    embedder: Annotated[EmbedderProtocol, Depends(get_embedder)] = ...,
    store: Annotated[ImageStore, Depends(get_store)] = ...,
) -> SearchResponse:
    suffix = Path(file.filename or "query").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        embedding = embedder.embed_image(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    results = store.search(embedding, top_k=top_k)
    return SearchResponse(results=results)

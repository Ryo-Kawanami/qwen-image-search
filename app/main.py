from fastapi import FastAPI

from app.routers.images import router as images_router

app = FastAPI(
    title="Qwen Image Search",
    description="Similar image search powered by Qwen3-VL-Embedding",
    version="0.1.0",
)

app.include_router(images_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

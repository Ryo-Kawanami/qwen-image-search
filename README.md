# Qwen Image Search

Qwen3-VL-Embedding-2B を使った類似画像検索 API。

## セットアップ

```bash
uv sync --extra dev
```

## 起動

```bash
uv run uvicorn app.main:app --reload
```

API ドキュメント: http://localhost:8000/docs

## API

### 画像登録

```bash
curl -X POST http://localhost:8000/images/register \
  -F "file=@/path/to/image.jpg"
```

```json
{"id": "uuid", "filename": "image.jpg", "message": "registered"}
```

### 類似画像検索

```bash
curl -X POST "http://localhost:8000/images/search?top_k=5" \
  -F "file=@/path/to/query.jpg"
```

```json
{
  "results": [
    {"id": "uuid", "filename": "image.jpg", "score": 0.98}
  ]
}
```

`score` は 0〜1（1が最も類似）。

## テスト

```bash
uv run pytest -v
```

## 備考

- 初回起動時に `Qwen/Qwen3-VL-Embedding-2B` が Hugging Face から自動ダウンロードされます（約5GB）
- GPU 推奨（CPU でも動きますが低速）
- 画像ストアはインメモリのためプロセス再起動でリセットされます

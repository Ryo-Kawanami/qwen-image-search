from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class EmbedderProtocol(Protocol):
    def embed_image(self, image_path: str | Path) -> list[float]: ...


class Qwen3VLEmbedder:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B") -> None:
        self._model_name = model_name_or_path
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor  # type: ignore

        self._model = AutoModel.from_pretrained(self._model_name, trust_remote_code=True)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self._model_name, trust_remote_code=True
        )

    def embed_image(self, image_path: str | Path) -> list[float]:
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        self._load()

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": str(image_path)}],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1][:, -1, :].squeeze(0)
            embedding = embedding / embedding.norm()
        return embedding.cpu().float().tolist()

# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional local text embeddings through Hugging Face Transformers."""

from __future__ import annotations

import asyncio
import math
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ._sync import run_sync_callback
from .embeddings import _MAX_VECTOR_DIMENSIONS, EmbeddingError

_DEFAULT_INPUT_BYTES = 4 * 1024 * 1024
_DEFAULT_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_BATCH_SIZE = 64
_MAX_INPUT_TOKENS = 8192
_MAX_TEXTS = 10_000


@dataclass
class TransformersEmbeddingProvider:
    """Embed text locally with a safe-weight Transformers encoder.

    The model loads on first use and remains resident until ``aclose``. Mean pooling respects the
    tokenizer attention mask; CLS pooling selects the first token. Vectors are L2-normalized by
    default for cosine search. The model's own retrieval prompts or task prefixes are not inferred;
    configure them in a separate provider when a model requires special formatting.
    """

    model_id: str
    token_env: str = "HF_TOKEN"
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    device: str = "cpu"
    pooling: str = "mean"
    normalize: bool = True
    max_input_tokens: int = 512
    batch_size: int = 16
    max_texts: int = _MAX_TEXTS
    max_input_bytes: int = _DEFAULT_INPUT_BYTES
    max_output_bytes: int = _DEFAULT_OUTPUT_BYTES
    timeout_seconds: float = 120.0
    name: str = "transformers_embeddings"
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _model_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _call_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate resource bounds before importing or loading optional model libraries."""
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty Hugging Face model ID or local path")
        self.model_id = self.model_id.strip()
        if (
            not isinstance(self.token_env, str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.token_env) is None
        ):
            raise ValueError("token_env must be an environment variable name")
        for setting_name, setting_value in (
            ("revision", self.revision),
            ("cache_dir", self.cache_dir),
        ):
            if setting_value is not None and (
                not isinstance(setting_value, str) or not setting_value.strip()
            ):
                raise ValueError(f"{setting_name} must be a non-empty string or None")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty PyTorch device")
        self.device = self.device.strip()
        if self.pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a boolean")
        for setting_name, numeric_value, maximum in (
            ("max_input_tokens", self.max_input_tokens, _MAX_INPUT_TOKENS),
            ("batch_size", self.batch_size, _MAX_BATCH_SIZE),
            ("max_texts", self.max_texts, _MAX_TEXTS),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or not 1 <= numeric_value <= maximum
            ):
                raise ValueError(f"{setting_name} must be an integer from 1 through {maximum}")
        for limit_name, limit_value in (
            ("max_input_bytes", self.max_input_bytes),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if isinstance(limit_value, bool) or not isinstance(limit_value, int) or limit_value < 1:
                raise ValueError(f"{limit_name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one bounded finite vector per input text, preserving input order."""
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if len(texts) > self.max_texts:
            raise EmbeddingError(f"Embedding input exceeded max_texts={self.max_texts}")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must contain non-empty strings")
        input_bytes = 0
        for text in texts:
            try:
                input_bytes += len(text.encode("utf-8"))
            except UnicodeEncodeError:
                raise ValueError("texts must contain valid UTF-8 encodable characters") from None
            if input_bytes > self.max_input_bytes:
                raise EmbeddingError(
                    f"Embedding input exceeded max_input_bytes={self.max_input_bytes}"
                )
        if not texts:
            return []

        stop_event = threading.Event()
        async with self._call_lock:
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    result = await run_sync_callback(self._embed_sync, list(texts), stop_event)
            except TimeoutError:
                stop_event.set()
                raise EmbeddingError("Local Transformers embedding exceeded its timeout") from None
            except asyncio.CancelledError:
                stop_event.set()
                raise
        if not isinstance(result, list) or len(result) != len(texts):
            raise EmbeddingError("Local Transformers embedding returned an invalid result")
        return result

    def _embed_sync(self, texts: list[str], stop_event: threading.Event) -> list[list[float]]:
        """Run bounded inference outside the event loop while serializing model access."""
        with self._model_lock:
            try:
                tokenizer, model, torch = self._load_model()
                outputs: list[list[float]] = []
                expected_dimensions: int | None = None
                output_bytes = 0
                for offset in range(0, len(texts), self.batch_size):
                    if stop_event.is_set():
                        raise EmbeddingError("Local Transformers embedding was cancelled")
                    batch = texts[offset : offset + self.batch_size]
                    inputs = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_tokens,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(self.device)
                    with torch.inference_mode():
                        encoded = model(**inputs)
                    hidden = encoded.last_hidden_state
                    dimensions = int(hidden.shape[-1])
                    if not 1 <= dimensions <= _MAX_VECTOR_DIMENSIONS:
                        raise EmbeddingError("Local Transformers returned an invalid vector size")
                    batch_output_bytes = len(batch) * dimensions * 8
                    if output_bytes + batch_output_bytes > self.max_output_bytes:
                        raise EmbeddingError(
                            "Local Transformers embeddings exceeded "
                            f"max_output_bytes={self.max_output_bytes}"
                        )
                    expected_dimensions = expected_dimensions or dimensions
                    if dimensions != expected_dimensions:
                        raise EmbeddingError(
                            "Local Transformers returned inconsistent vector dimensions"
                        )
                    pooled = self._pool(hidden, inputs)
                    if self.normalize:
                        pooled = pooled.to(dtype=torch.float32)
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    rows = pooled.detach().to(device="cpu", dtype=torch.float32).tolist()
                    vectors = self._validate_vectors(rows, len(batch), dimensions)
                    output_bytes += batch_output_bytes
                    outputs.extend(vectors)
                return outputs
            except EmbeddingError:
                raise
            except Exception:
                raise EmbeddingError("Local Transformers embedding failed") from None

    def _pool(self, hidden: Any, inputs: Any) -> Any:
        if self.pooling == "cls":
            return hidden[:, 0, :]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise EmbeddingError("Local Transformers tokenizer did not return an attention mask")
        weights = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def _validate_vectors(
        self, rows: Any, expected_count: int, expected_dimensions: int
    ) -> list[list[float]]:
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise EmbeddingError("Local Transformers returned an unexpected number of vectors")
        result: list[list[float]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) != expected_dimensions:
                raise EmbeddingError("Local Transformers returned an invalid embedding vector")
            vector: list[float] = []
            squared_norm = 0.0
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingError("Local Transformers returned a non-numeric vector value")
                converted = float(value)
                if not math.isfinite(converted):
                    raise EmbeddingError("Local Transformers returned a non-finite vector value")
                vector.append(converted)
                squared_norm += converted * converted
            if not math.isfinite(squared_norm) or squared_norm == 0:
                raise EmbeddingError("Local Transformers returned a zero or invalid vector")
            result.append(vector)
        return result

    def _load_model(self) -> tuple[Any, Any, Any]:
        """Load only standard Transformers architectures and safetensors weights."""
        if self._tokenizer is not None and self._model is not None:
            try:
                import torch
            except ImportError:
                raise EmbeddingError(
                    "Install Gabby's optional 'transformers' dependencies to use this provider"
                ) from None
            return self._tokenizer, self._model, torch
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            token = os.environ.get(self.token_env)
            options: dict[str, Any] = {
                "local_files_only": self.local_files_only,
                "trust_remote_code": False,
            }
            if token:
                options["token"] = token
            if self.revision:
                options["revision"] = self.revision
            if self.cache_dir:
                options["cache_dir"] = self.cache_dir
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
            model = AutoModel.from_pretrained(self.model_id, use_safetensors=True, **options)
            model.to(self.device)
            model.eval()
        except ImportError:
            raise EmbeddingError(
                "Install Gabby's optional 'transformers' dependencies to use this provider"
            ) from None
        except Exception:
            raise EmbeddingError("Local Transformers embedding model could not be loaded") from None
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model, torch

    async def aclose(self) -> None:
        """Release model and tokenizer references after active embedding calls finish."""
        async with self._call_lock:
            await run_sync_callback(self._clear_model)

    def _clear_model(self) -> None:
        with self._model_lock:
            self._model = None
            self._tokenizer = None

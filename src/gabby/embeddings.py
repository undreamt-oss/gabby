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
"""Model-independent adapters for remote text embedding services."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from .config import (
    DEFAULT_MAX_MODEL_REQUEST_BYTES,
    DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    validate_model_credentials,
)

_MAX_BATCH_ITEMS = 100
_MAX_VECTOR_DIMENSIONS = 65_536


class EmbeddingError(RuntimeError):
    """The embedding provider failed to return one valid vector per input."""


class EmbeddingResponseSizeError(EmbeddingError):
    """An embedding provider response exceeded the configured byte limit."""


@dataclass
class OpenAICompatibleEmbeddingProvider:
    """Call a provider implementing ``POST /embeddings`` and normalize its indexed results.

    The model is fixed on this adapter so one `EmbeddingProvider` instance represents one
    embedding space. Texts are divided into bounded batches, then returned in input order even if
    a compatible server reorders the response entries. The service still owns token limits and
    model-specific input requirements.
    """

    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    dimensions: int | None = None
    batch_size: int = 64
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    name: str = "openai_compatible_embeddings"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if not isinstance(self.extra_headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise TypeError("extra_headers must map strings to strings")
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or not 1 <= self.dimensions <= _MAX_VECTOR_DIMENSIONS
        ):
            raise ValueError(f"dimensions must be from 1 through {_MAX_VECTOR_DIMENSIONS}")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= _MAX_BATCH_ITEMS
        ):
            raise ValueError(f"batch_size must be from 1 through {_MAX_BATCH_ITEMS}")
        for name, limit in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_response_bytes", self.max_response_bytes),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        validate_model_credentials({"base_url": self.base_url}, provider="openai_compatible")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OpenAICompatibleEmbeddingProvider:
        """Construct from a host-managed embedding configuration without inline secrets."""
        validate_model_credentials(config, provider="openai_compatible")
        return cls(
            model=config.get("model", ""),
            base_url=config.get("base_url", "https://api.openai.com/v1").rstrip("/"),
            api_key_env=config.get("api_key_env", "OPENAI_API_KEY"),
            extra_headers=dict(config.get("headers", {})),
            dimensions=config.get("dimensions"),
            batch_size=config.get("batch_size", 64),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one finite, consistently sized vector for each non-empty input text."""
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if not texts:
            return []

        key = self.api_key or os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers)

        outputs: list[list[float]] = []
        expected_dimensions = self.dimensions
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            payload: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            self._ensure_request_size(payload)
            try:
                async with client.stream(
                    "POST",
                    "/embeddings",
                    json=payload,
                    timeout=httpx.Timeout(self.timeout_seconds),
                ) as response:
                    if response.status_code >= 400:
                        raise EmbeddingError(
                            f"Embedding provider returned HTTP {response.status_code}"
                        )
                    body = await self._read_response_limited(response)
                    data = json.loads(body)
            except EmbeddingError:
                raise
            except (httpx.HTTPError, ValueError):
                raise EmbeddingError("Embedding provider request failed") from None

            vectors = self._parse_response(data, len(batch))
            for vector in vectors:
                if expected_dimensions is None:
                    expected_dimensions = len(vector)
                elif len(vector) != expected_dimensions:
                    raise EmbeddingError("Embedding provider returned inconsistent dimensions")
            outputs.extend(vectors)
        return outputs

    async def aclose(self) -> None:
        """Close the shared HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _ensure_request_size(self, payload: dict[str, Any]) -> None:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        byte_count = 0
        try:
            for part in encoder.iterencode(payload):
                byte_count += len(part.encode("utf-8"))
                if byte_count > self.max_request_bytes:
                    raise EmbeddingError(
                        "Serialized embedding request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError("Embedding request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise EmbeddingResponseSizeError(
                    f"Embedding response exceeded max_response_bytes={self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @classmethod
    def _parse_response(cls, data: Any, expected_count: int) -> list[list[float]]:
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise EmbeddingError("Embedding provider returned an invalid response object")
        items = data["data"]
        if len(items) != expected_count:
            raise EmbeddingError("Embedding provider returned an unexpected number of vectors")
        ordered: list[list[float] | None] = [None] * expected_count
        dimensions: int | None = None
        for item in items:
            if not isinstance(item, dict):
                raise EmbeddingError("Embedding provider returned an invalid vector entry")
            index = item.get("index")
            vector = item.get("embedding")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < expected_count
            ):
                raise EmbeddingError("Embedding provider returned an invalid vector index")
            if ordered[index] is not None:
                raise EmbeddingError("Embedding provider returned a duplicate vector index")
            if not isinstance(vector, list) or not 1 <= len(vector) <= _MAX_VECTOR_DIMENSIONS:
                raise EmbeddingError("Embedding provider returned an invalid vector")
            normalized: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingError("Embedding provider returned a non-numeric vector value")
                try:
                    converted = float(value)
                except OverflowError:
                    raise EmbeddingError(
                        "Embedding provider returned a non-finite vector value"
                    ) from None
                if not math.isfinite(converted):
                    raise EmbeddingError("Embedding provider returned a non-finite vector value")
                normalized.append(converted)
            if dimensions is None:
                dimensions = len(normalized)
            elif len(normalized) != dimensions:
                raise EmbeddingError("Embedding provider returned inconsistent dimensions")
            ordered[index] = normalized
        if any(vector is None for vector in ordered):
            raise EmbeddingError("Embedding provider response omitted vector indexes")
        return [vector for vector in ordered if vector is not None]


@dataclass
class HuggingFaceFeatureExtractionProvider:
    """Call Hugging Face Inference Providers' feature-extraction task endpoint.

    The task API may return token vectors rather than one sentence vector. The adapter applies
    configured ``mean`` or ``cls`` pooling in that case, validates all numeric values, and keeps
    the transport and response size bounded. The model is fixed for each provider instance.
    """

    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "HF_TOKEN"
    base_url: str = "https://router.huggingface.co/hf-inference/models"
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    pooling: str = "mean"
    batch_size: int = 32
    truncate: bool | None = None
    normalize: bool | None = None
    prompt_name: str | None = None
    truncation_direction: str | None = None
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    name: str = "huggingface_feature_extraction"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self._is_model_id(self.model):
            raise ValueError("model must be a Hugging Face model ID such as org/model")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if not isinstance(self.extra_headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise TypeError("extra_headers must map strings to strings")
        validate_model_credentials({"base_url": self.base_url}, provider="huggingface")
        if self.pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= _MAX_BATCH_ITEMS
        ):
            raise ValueError(f"batch_size must be from 1 through {_MAX_BATCH_ITEMS}")
        for name, value in (("truncate", self.truncate), ("normalize", self.normalize)):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or None")
        if self.prompt_name is not None and (
            not isinstance(self.prompt_name, str) or not self.prompt_name.strip()
        ):
            raise ValueError("prompt_name must be a non-empty string or None")
        if self.truncation_direction not in (None, "left", "right"):
            raise ValueError("truncation_direction must be 'left', 'right', or None")
        if self.truncation_direction is not None and self.truncate is not True:
            raise ValueError("truncation_direction requires truncate=True")
        for name, limit in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_response_bytes", self.max_response_bytes),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")

    @staticmethod
    def _is_model_id(value: str) -> bool:
        segments = value.split("/")
        return len(segments) >= 2 and all(
            segment
            and segment not in (".", "..")
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", segment) is not None
            for segment in segments
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> HuggingFaceFeatureExtractionProvider:
        """Construct from host-managed model configuration without inline secrets."""
        validate_model_credentials(config, provider="huggingface")
        headers = config.get("headers", {})
        base_url = config.get("base_url", "https://router.huggingface.co/hf-inference/models")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        return cls(
            model=config.get("model", ""),
            api_key_env=config.get("api_key_env", "HF_TOKEN"),
            base_url=base_url.rstrip("/"),
            extra_headers=dict(headers),
            pooling=config.get("pooling", "mean"),
            batch_size=config.get("batch_size", 32),
            truncate=config.get("truncate"),
            normalize=config.get("normalize"),
            prompt_name=config.get("prompt_name"),
            truncation_direction=config.get("truncation_direction"),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one pooled feature vector per input, preserving the input order."""
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if not texts:
            return []

        key = self.api_key or os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, follow_redirects=False
            )
        endpoint = f"/{quote(self.model, safe='/')}/pipeline/feature-extraction"
        outputs: list[list[float]] = []
        expected_dimensions: int | None = None
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            payload: dict[str, Any] = {"inputs": batch}
            for name in ("truncate", "normalize", "prompt_name", "truncation_direction"):
                value = getattr(self, name)
                if value is not None:
                    payload[name] = value
            self._ensure_request_size(payload)
            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    timeout=httpx.Timeout(self.timeout_seconds),
                ) as response:
                    if response.status_code >= 400:
                        raise EmbeddingError(
                            f"Hugging Face feature extraction returned HTTP {response.status_code}"
                        )
                    body = await self._read_response_limited(response)
                    data = json.loads(body)
            except EmbeddingError:
                raise
            except (httpx.HTTPError, ValueError):
                raise EmbeddingError("Hugging Face feature extraction request failed") from None

            vectors = self._parse_response(data, len(batch), pooling=self.pooling)
            for vector in vectors:
                if expected_dimensions is None:
                    expected_dimensions = len(vector)
                elif len(vector) != expected_dimensions:
                    raise EmbeddingError("Hugging Face returned inconsistent vector dimensions")
            outputs.extend(vectors)
        return outputs

    def _ensure_request_size(self, payload: dict[str, Any]) -> None:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        byte_count = 0
        try:
            for part in encoder.iterencode(payload):
                byte_count += len(part.encode("utf-8"))
                if byte_count > self.max_request_bytes:
                    raise EmbeddingError(
                        "Serialized Hugging Face request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError("Hugging Face request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise EmbeddingResponseSizeError(
                    f"Hugging Face response exceeded max_response_bytes={self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @classmethod
    def _parse_response(cls, data: Any, expected_count: int, *, pooling: str) -> list[list[float]]:
        if not isinstance(data, list):
            raise EmbeddingError("Hugging Face returned an invalid feature response")
        if len(data) == expected_count:
            samples = data
        elif expected_count == 1:
            samples = [data]
        else:
            raise EmbeddingError("Hugging Face returned an unexpected number of vectors")

        vectors: list[list[float]] = []
        for sample in samples:
            if not isinstance(sample, list) or not sample:
                raise EmbeddingError("Hugging Face returned an invalid feature vector")
            if isinstance(sample[0], list):
                rows = sample
                normalized_rows = [cls._normalize_vector(row) for row in rows]
                dimensions = len(normalized_rows[0])
                if any(len(row) != dimensions for row in normalized_rows):
                    raise EmbeddingError("Hugging Face returned ragged token vectors")
                if pooling == "cls":
                    vector = normalized_rows[0]
                else:
                    vector = [
                        sum(row[column] for row in normalized_rows) / len(normalized_rows)
                        for column in range(dimensions)
                    ]
            else:
                vector = cls._normalize_vector(sample)
            vectors.append(cls._normalize_vector(vector))
        return vectors

    @staticmethod
    def _normalize_vector(vector: Any) -> list[float]:
        if not isinstance(vector, list) or not 1 <= len(vector) <= _MAX_VECTOR_DIMENSIONS:
            raise EmbeddingError("Hugging Face returned an invalid feature vector")
        normalized: list[float] = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError("Hugging Face returned a non-numeric feature value")
            try:
                converted = float(value)
            except OverflowError:
                raise EmbeddingError("Hugging Face returned a non-finite feature value") from None
            if not math.isfinite(converted):
                raise EmbeddingError("Hugging Face returned a non-finite feature value")
            normalized.append(converted)
        return normalized

    async def aclose(self) -> None:
        """Close the shared HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


@dataclass
class GeminiEmbeddingProvider:
    """Call Google's Gemini text embedding API through bounded batch requests.

    ``gemini-embedding-001`` supports API task types such as ``RETRIEVAL_QUERY`` and
    ``RETRIEVAL_DOCUMENT``. ``gemini-embedding-2`` uses task instructions in the input text
    instead, so this adapter rejects ``task_type`` for that model. The provider instance fixes
    one embedding space; rebuild stored vectors when changing model or dimensions.
    """

    model: str = "gemini-embedding-2"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    extra_headers: dict[str, str] = field(default_factory=dict, repr=False)
    task_type: str | None = None
    query_task_type: str | None = None
    document_task_type: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    title: str | None = None
    dimensions: int | None = None
    batch_size: int = 32
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    name: str = "gemini_embeddings"
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    _TASK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "RETRIEVAL_QUERY",
            "RETRIEVAL_DOCUMENT",
            "SEMANTIC_SIMILARITY",
            "CLASSIFICATION",
            "CLUSTERING",
            "CODE_RETRIEVAL_QUERY",
            "QUESTION_ANSWERING",
            "FACT_VERIFICATION",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model, str):
            raise ValueError("model must be a Gemini embedding model name")
        model = self.model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model):
            raise ValueError("model must be a Gemini embedding model name")
        self.model = model
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if not isinstance(self.extra_headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.extra_headers.items()
        ):
            raise TypeError("extra_headers must map strings to strings")
        validate_model_credentials({"base_url": self.base_url}, provider="gemini")
        for name in ("task_type", "query_task_type", "document_task_type"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or value not in self._TASK_TYPES):
                raise ValueError(f"{name} is not a supported Gemini embedding task type")
        if self.model == "gemini-embedding-2" and any(
            getattr(self, name) is not None
            for name in ("task_type", "query_task_type", "document_task_type")
        ):
            raise ValueError(
                "gemini-embedding-2 does not support task_type; include task instructions in text"
            )
        for name in ("query_prefix", "document_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            if len(value.encode("utf-8")) > 4096:
                raise ValueError(f"{name} exceeds 4096 UTF-8 bytes")
        document_task_type = self.document_task_type or self.task_type
        if self.title is not None and (
            not isinstance(self.title, str)
            or not self.title.strip()
            or document_task_type != "RETRIEVAL_DOCUMENT"
        ):
            raise ValueError(
                "title requires a non-empty value and document_task_type=RETRIEVAL_DOCUMENT"
            )
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or not 128 <= self.dimensions <= 3072
        ):
            raise ValueError("dimensions must be from 128 through 3072")
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or not 1 <= self.batch_size <= _MAX_BATCH_ITEMS
        ):
            raise ValueError(f"batch_size must be from 1 through {_MAX_BATCH_ITEMS}")
        for name, limit in (
            ("max_request_bytes", self.max_request_bytes),
            ("max_response_bytes", self.max_response_bytes),
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GeminiEmbeddingProvider:
        """Construct from host-managed configuration without inline secrets."""
        validate_model_credentials(config, provider="gemini")
        base_url = config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        return cls(
            model=config.get("model", "gemini-embedding-2"),
            api_key_env=config.get("api_key_env", "GEMINI_API_KEY"),
            base_url=base_url.rstrip("/"),
            extra_headers=dict(config.get("headers", {})),
            task_type=config.get("task_type"),
            query_task_type=config.get("query_task_type"),
            document_task_type=config.get("document_task_type"),
            query_prefix=config.get("query_prefix", ""),
            document_prefix=config.get("document_prefix", ""),
            title=config.get("title"),
            dimensions=config.get("dimensions"),
            batch_size=config.get("batch_size", 32),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed role-neutral inputs using the configured generic task type."""
        return await self._embed(texts, task_type=self.task_type, title=None, prefix="")

    async def embed_queries(
        self, texts: Sequence[str], *, prefix: str | None = None
    ) -> list[list[float]]:
        """Embed retrieval queries with query-specific task type and prefix settings."""
        return await self._embed(
            texts,
            task_type=self.query_task_type or self.task_type,
            title=None,
            prefix=self.query_prefix if prefix is None else prefix,
        )

    async def embed_documents(
        self, texts: Sequence[str], *, prefix: str | None = None
    ) -> list[list[float]]:
        """Embed indexed documents with document-specific task type and optional title."""
        return await self._embed(
            texts,
            task_type=self.document_task_type or self.task_type,
            title=self.title,
            prefix=self.document_prefix if prefix is None else prefix,
        )

    async def _embed(
        self,
        texts: Sequence[str],
        *,
        task_type: str | None,
        title: str | None,
        prefix: str,
    ) -> list[list[float]]:
        """Return one finite, consistently sized vector for each non-empty input string."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if not texts:
            return []
        if prefix:
            texts = [prefix + text for text in texts]

        headers = {"Content-Type": "application/json", **self.extra_headers}
        key = self.api_key or os.environ.get(self.api_key_env)
        if key:
            headers["x-goog-api-key"] = key
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url, headers=headers, follow_redirects=False
            )

        outputs: list[list[float]] = []
        expected_dimensions = self.dimensions
        endpoint = f"/models/{quote(self.model, safe='')}:batchEmbedContents"
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            requests: list[dict[str, Any]] = []
            for text in batch:
                item: dict[str, Any] = {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                }
                config: dict[str, Any] = {}
                if task_type is not None:
                    config["taskType"] = task_type
                if title is not None:
                    config["title"] = title
                if self.dimensions is not None:
                    config["outputDimensionality"] = self.dimensions
                if config:
                    item["embedContentConfig"] = config
                requests.append(item)
            payload = {"requests": requests}
            self._ensure_request_size(payload)
            try:
                async with client.stream(
                    "POST",
                    endpoint,
                    json=payload,
                    timeout=httpx.Timeout(self.timeout_seconds),
                ) as response:
                    if response.status_code >= 400:
                        raise EmbeddingError(
                            f"Gemini embedding provider returned HTTP {response.status_code}"
                        )
                    body = await self._read_response_limited(response)
                    data = json.loads(body)
            except EmbeddingError:
                raise
            except (httpx.HTTPError, ValueError):
                raise EmbeddingError("Gemini embedding provider request failed") from None

            vectors = self._parse_response(data, len(batch))
            for vector in vectors:
                if expected_dimensions is None:
                    expected_dimensions = len(vector)
                elif len(vector) != expected_dimensions:
                    raise EmbeddingError("Gemini returned inconsistent vector dimensions")
            outputs.extend(vectors)
        return outputs

    def _ensure_request_size(self, payload: dict[str, Any]) -> None:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        byte_count = 0
        try:
            for part in encoder.iterencode(payload):
                byte_count += len(part.encode("utf-8"))
                if byte_count > self.max_request_bytes:
                    raise EmbeddingError(
                        "Serialized Gemini embedding request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError("Gemini embedding request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise EmbeddingResponseSizeError(
                    "Gemini embedding response exceeded max_response_bytes="
                    f"{self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @classmethod
    def _parse_response(cls, data: Any, expected_count: int) -> list[list[float]]:
        if not isinstance(data, dict) or not isinstance(data.get("embeddings"), list):
            raise EmbeddingError("Gemini returned an invalid embedding response object")
        items = data["embeddings"]
        if len(items) != expected_count:
            raise EmbeddingError("Gemini returned an unexpected number of vectors")
        vectors: list[list[float]] = []
        dimensions: int | None = None
        for item in items:
            values = item.get("values") if isinstance(item, dict) else None
            if not isinstance(values, list) or not 1 <= len(values) <= _MAX_VECTOR_DIMENSIONS:
                raise EmbeddingError("Gemini returned an invalid embedding vector")
            vector: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingError("Gemini returned a non-numeric embedding value")
                try:
                    converted = float(value)
                except OverflowError:
                    raise EmbeddingError("Gemini returned a non-finite embedding value") from None
                if not math.isfinite(converted):
                    raise EmbeddingError("Gemini returned a non-finite embedding value")
                vector.append(converted)
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise EmbeddingError("Gemini returned inconsistent vector dimensions")
            vectors.append(vector)
        return vectors

    async def aclose(self) -> None:
        """Close the shared HTTP connection pool."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

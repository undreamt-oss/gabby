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
"""Bounded, host-configured reranking provider adapters."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._sync import run_sync_callback
from .config import (
    DEFAULT_MAX_MODEL_REQUEST_BYTES,
    DEFAULT_MAX_MODEL_RESPONSE_BYTES,
    validate_model_credentials,
    validate_model_endpoint,
)
from .knowledge import MAX_RETRIEVAL_DOCUMENTS, Document


class RerankingError(RuntimeError):
    """A reranking provider failed or returned an invalid result."""


class RerankingResponseSizeError(RerankingError):
    """A reranking provider response exceeded the configured byte limit."""


@dataclass
class CohereReranker:
    """Rerank a bounded set of document texts with Cohere's v2 rerank endpoint.

    Only document text is sent to the provider. Gabby maps returned indexes back to the original
    ``Document`` objects, preserving local source metadata and identifiers. Credentials are read
    from the host environment unless an application injects ``api_key`` directly.
    """

    model: str = "rerank-v4.0-fast"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "COHERE_API_KEY"
    base_url: str = "https://api.cohere.com/v2"
    max_tokens_per_doc: int = 4096
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty Cohere rerank model name")
        self.model = self.model.strip()
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if (
            isinstance(self.max_tokens_per_doc, bool)
            or not isinstance(self.max_tokens_per_doc, int)
            or not 1 <= self.max_tokens_per_doc <= 32_768
        ):
            raise ValueError("max_tokens_per_doc must be an integer from 1 through 32768")
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
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        validate_model_credentials({"base_url": self.base_url}, provider="cohere")
        validate_model_endpoint(self.base_url, provider="Cohere reranking")
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CohereReranker:
        """Create an adapter from non-secret configuration and host environment credentials."""
        validate_model_credentials(config, provider="cohere")
        base_url = config.get("base_url", "https://api.cohere.com/v2")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        return cls(
            model=config.get("model", "rerank-v4.0-fast"),
            api_key_env=config.get("api_key_env", "COHERE_API_KEY"),
            base_url=base_url.rstrip("/"),
            max_tokens_per_doc=config.get("max_tokens_per_doc", 4096),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return the provider-ranked subset, mapping only validated indexes to input documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of Document values")
        if len(documents) > MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError(f"documents must contain at most {MAX_RETRIEVAL_DOCUMENTS} values")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if limit == 0 or not query.strip() or not documents:
            return []
        for document in documents:
            if not isinstance(document, Document) or not isinstance(document.text, str):
                raise TypeError("documents must contain Document values with string text")

        payload = {
            "model": self.model,
            "query": query,
            "documents": [document.text for document in documents],
            "top_n": min(limit, len(documents)),
            "max_tokens_per_doc": self.max_tokens_per_doc,
        }
        self._ensure_request_size(payload)
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RerankingError(f"Cohere API credential is missing from {self.api_key_env}")
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )

        try:
            async with client.stream(
                "POST",
                "/rerank",
                json=payload,
                timeout=httpx.Timeout(self.timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    raise RerankingError(f"Cohere reranking returned HTTP {response.status_code}")
                body = await self._read_response_limited(response)
                data = json.loads(body)
        except RerankingError:
            raise
        except (httpx.HTTPError, ValueError):
            raise RerankingError("Cohere reranking request failed") from None

        return self._parse_response(data, documents, limit=min(limit, len(documents)))

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
                    raise RerankingError(
                        "Serialized Cohere reranking request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            if isinstance(exc, RerankingError):
                raise
            raise RerankingError("Cohere reranking request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise RerankingResponseSizeError(
                    "Cohere reranking response exceeded max_response_bytes="
                    f"{self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @staticmethod
    def _parse_response(data: Any, documents: Sequence[Document], *, limit: int) -> list[Document]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RerankingError("Cohere reranking returned an invalid response object")
        results = data["results"]
        if len(results) > limit:
            raise RerankingError("Cohere reranking returned more documents than requested")
        ranked: list[Document] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise RerankingError("Cohere reranking returned an invalid result entry")
            index = result.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen
            ):
                raise RerankingError("Cohere reranking returned an invalid document index")
            score = result.get("relevance_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankingError("Cohere reranking returned an invalid relevance score")
            try:
                finite_score = math.isfinite(float(score))
            except OverflowError:
                finite_score = False
            if not finite_score:
                raise RerankingError("Cohere reranking returned an invalid relevance score")
            seen.add(index)
            ranked.append(documents[index])
        return ranked


@dataclass
class JinaReranker:
    """Rerank a bounded set of document texts with Jina's hosted rerank API."""

    model: str = "jina-reranker-v3.5"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "JINA_API_KEY"
    base_url: str = "https://api.jina.ai/v1"
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty Jina rerank model name")
        self.model = self.model.strip()
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
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
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        validate_model_credentials({"base_url": self.base_url}, provider="jina")
        validate_model_endpoint(self.base_url, provider="Jina reranking")
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JinaReranker:
        """Create an adapter from non-secret configuration and host environment credentials."""
        validate_model_credentials(config, provider="jina")
        base_url = config.get("base_url", "https://api.jina.ai/v1")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        return cls(
            model=config.get("model", "jina-reranker-v3.5"),
            api_key_env=config.get("api_key_env", "JINA_API_KEY"),
            base_url=base_url.rstrip("/"),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return the provider-ranked subset while preserving original document objects."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of Document values")
        if len(documents) > MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError(f"documents must contain at most {MAX_RETRIEVAL_DOCUMENTS} values")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if limit == 0 or not query.strip() or not documents:
            return []
        for document in documents:
            if not isinstance(document, Document) or not isinstance(document.text, str):
                raise TypeError("documents must contain Document values with string text")

        payload = {
            "model": self.model,
            "query": query,
            "documents": [document.text for document in documents],
            "top_n": min(limit, len(documents)),
            "return_documents": False,
        }
        self._ensure_request_size(payload)
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RerankingError(f"Jina API credential is missing from {self.api_key_env}")
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        try:
            async with client.stream(
                "POST", "/rerank", json=payload, timeout=httpx.Timeout(self.timeout_seconds)
            ) as response:
                if response.status_code >= 400:
                    raise RerankingError(f"Jina reranking returned HTTP {response.status_code}")
                body = await self._read_response_limited(response)
                data = json.loads(body)
        except RerankingError:
            raise
        except (httpx.HTTPError, ValueError):
            raise RerankingError("Jina reranking request failed") from None
        return self._parse_response(data, documents, limit=min(limit, len(documents)))

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
                    raise RerankingError(
                        "Serialized Jina reranking request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError):
            raise RerankingError("Jina reranking request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise RerankingResponseSizeError(
                    f"Jina reranking response exceeded max_response_bytes={self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @staticmethod
    def _parse_response(data: Any, documents: Sequence[Document], *, limit: int) -> list[Document]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RerankingError("Jina reranking returned an invalid response object")
        results = data["results"]
        if len(results) > limit:
            raise RerankingError("Jina reranking returned more documents than requested")
        ranked: list[Document] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise RerankingError("Jina reranking returned an invalid result entry")
            index = result.get("index")
            score = result.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen
            ):
                raise RerankingError("Jina reranking returned an invalid document index")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankingError("Jina reranking returned an invalid relevance score")
            try:
                finite_score = math.isfinite(float(score))
            except OverflowError:
                finite_score = False
            if not finite_score:
                raise RerankingError("Jina reranking returned an invalid relevance score")
            seen.add(index)
            ranked.append(documents[index])
        return ranked


@dataclass
class VoyageReranker:
    """Rerank a bounded set of document texts with Voyage AI's hosted rerank API."""

    model: str = "rerank-2.5-lite"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "VOYAGE_API_KEY"
    base_url: str = "https://api.voyageai.com/v1"
    truncation: bool = False
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty Voyage rerank model name")
        self.model = self.model.strip()
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
        if not isinstance(self.truncation, bool):
            raise ValueError("truncation must be a boolean")
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
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        validate_model_credentials({"base_url": self.base_url}, provider="voyage")
        validate_model_endpoint(self.base_url, provider="Voyage reranking")
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> VoyageReranker:
        """Create an adapter from non-secret settings and a host environment credential."""
        validate_model_credentials(config, provider="voyage")
        base_url = config.get("base_url", "https://api.voyageai.com/v1")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        return cls(
            model=config.get("model", "rerank-2.5-lite"),
            api_key_env=config.get("api_key_env", "VOYAGE_API_KEY"),
            base_url=base_url.rstrip("/"),
            truncation=config.get("truncation", False),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return provider-ranked original documents after validating result indexes."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of Document values")
        if len(documents) > MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError(f"documents must contain at most {MAX_RETRIEVAL_DOCUMENTS} values")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if limit == 0 or not query.strip() or not documents:
            return []
        for document in documents:
            if not isinstance(document, Document) or not isinstance(document.text, str):
                raise TypeError("documents must contain Document values with string text")

        payload = {
            "model": self.model,
            "query": query,
            "documents": [document.text for document in documents],
            "top_k": min(limit, len(documents)),
            "return_documents": False,
            "truncation": self.truncation,
        }
        self._ensure_request_size(payload)
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RerankingError(f"Voyage API credential is missing from {self.api_key_env}")
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        try:
            async with client.stream(
                "POST", "/rerank", json=payload, timeout=httpx.Timeout(self.timeout_seconds)
            ) as response:
                if response.status_code >= 400:
                    raise RerankingError(f"Voyage reranking returned HTTP {response.status_code}")
                body = await self._read_response_limited(response)
                data = json.loads(body)
        except RerankingError:
            raise
        except (httpx.HTTPError, ValueError):
            raise RerankingError("Voyage reranking request failed") from None
        return self._parse_response(data, documents, limit=min(limit, len(documents)))

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
                    raise RerankingError(
                        "Serialized Voyage reranking request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError):
            raise RerankingError("Voyage reranking request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise RerankingResponseSizeError(
                    "Voyage reranking response exceeded max_response_bytes="
                    f"{self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @staticmethod
    def _parse_response(data: Any, documents: Sequence[Document], *, limit: int) -> list[Document]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise RerankingError("Voyage reranking returned an invalid response object")
        results = data["results"]
        if len(results) > limit:
            raise RerankingError("Voyage reranking returned more documents than requested")
        ranked: list[Document] = []
        seen: set[int] = set()
        for result in results:
            if not isinstance(result, dict):
                raise RerankingError("Voyage reranking returned an invalid result entry")
            index = result.get("index")
            score = result.get("relevance_score")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen
            ):
                raise RerankingError("Voyage reranking returned an invalid document index")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RerankingError("Voyage reranking returned an invalid relevance score")
            try:
                finite_score = math.isfinite(float(score))
            except OverflowError:
                finite_score = False
            if not finite_score:
                raise RerankingError("Voyage reranking returned an invalid relevance score")
            seen.add(index)
            ranked.append(documents[index])
        return ranked


@dataclass
class NvidiaReranker:
    """Rerank candidates with NVIDIA's hosted NeMo retrieval reranking API."""

    model: str = "nvidia/rerank-qa-mistral-4b"
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str = "NVIDIA_API_KEY"
    base_url: str = "https://ai.api.nvidia.com"
    max_request_bytes: int = DEFAULT_MAX_MODEL_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_MODEL_RESPONSE_BYTES
    timeout_seconds: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 128:
            raise ValueError("model must contain from 1 through 128 characters")
        self.model = self.model.strip()
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a non-empty string when provided")
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
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        validate_model_credentials({"base_url": self.base_url}, provider="nvidia")
        validate_model_endpoint(self.base_url, provider="NVIDIA reranking")
        self.base_url = self.base_url.rstrip("/")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> NvidiaReranker:
        """Create an adapter from non-secret settings and host environment credentials."""
        validate_model_credentials(config, provider="nvidia")
        base_url = config.get("base_url", "https://ai.api.nvidia.com")
        if not isinstance(base_url, str):
            raise ValueError("base_url must be an HTTPS URL or loopback HTTP URL")
        return cls(
            model=config.get("model", "nvidia/rerank-qa-mistral-4b"),
            api_key_env=config.get("api_key_env", "NVIDIA_API_KEY"),
            base_url=base_url.rstrip("/"),
            max_request_bytes=config.get("max_request_bytes", DEFAULT_MAX_MODEL_REQUEST_BYTES),
            max_response_bytes=config.get("max_response_bytes", DEFAULT_MAX_MODEL_RESPONSE_BYTES),
            timeout_seconds=config.get("timeout_seconds", 120.0),
        )

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return the provider-ranked top subset after validating result indexes and scores."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of Document values")
        if len(documents) > MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError(f"documents must contain at most {MAX_RETRIEVAL_DOCUMENTS} values")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if limit == 0 or not query.strip() or not documents:
            return []
        for document in documents:
            if not isinstance(document, Document) or not isinstance(document.text, str):
                raise TypeError("documents must contain Document values with string text")

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": document.text} for document in documents],
            "truncate": "NONE",
        }
        self._ensure_request_size(payload)
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RerankingError(f"NVIDIA API credential is missing from {self.api_key_env}")
        client = self._client
        if client is None or client.is_closed:
            client = self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        try:
            async with client.stream(
                "POST",
                "/v1/retrieval/nvidia/reranking",
                json=payload,
                timeout=httpx.Timeout(self.timeout_seconds),
            ) as response:
                if response.status_code >= 400:
                    raise RerankingError(f"NVIDIA reranking returned HTTP {response.status_code}")
                body = await self._read_response_limited(response)
                data = json.loads(body)
        except RerankingError:
            raise
        except (httpx.HTTPError, ValueError):
            raise RerankingError("NVIDIA reranking request failed") from None
        return self._parse_response(data, documents, limit=min(limit, len(documents)))

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
                    raise RerankingError(
                        "Serialized NVIDIA reranking request exceeded max_request_bytes="
                        f"{self.max_request_bytes}"
                    )
        except (TypeError, ValueError, UnicodeEncodeError):
            raise RerankingError("NVIDIA reranking request is not valid JSON") from None

    async def _read_response_limited(self, response: httpx.Response) -> bytearray:
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > self.max_response_bytes:
                raise RerankingResponseSizeError(
                    "NVIDIA reranking response exceeded max_response_bytes="
                    f"{self.max_response_bytes}"
                )
            body.extend(chunk)
        return body

    @staticmethod
    def _parse_response(data: Any, documents: Sequence[Document], *, limit: int) -> list[Document]:
        if not isinstance(data, dict) or not isinstance(data.get("rankings"), list):
            raise RerankingError("NVIDIA reranking returned an invalid response object")
        rankings = data["rankings"]
        if len(rankings) > len(documents):
            raise RerankingError("NVIDIA reranking returned more passages than provided")
        ranked: list[Document] = []
        seen: set[int] = set()
        for result in rankings:
            if not isinstance(result, dict):
                raise RerankingError("NVIDIA reranking returned an invalid ranking entry")
            index = result.get("index")
            logit = result.get("logit")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(documents)
                or index in seen
            ):
                raise RerankingError("NVIDIA reranking returned an invalid passage index")
            if isinstance(logit, bool) or not isinstance(logit, (int, float)):
                raise RerankingError("NVIDIA reranking returned an invalid logit")
            try:
                finite_logit = math.isfinite(float(logit))
            except OverflowError:
                finite_logit = False
            if not finite_logit:
                raise RerankingError("NVIDIA reranking returned an invalid logit")
            seen.add(index)
            if len(ranked) < limit:
                ranked.append(documents[index])
        return ranked


@dataclass
class TransformersReranker:
    """Rerank documents with a local Hugging Face sequence-classification model.

    Weights load lazily and remote model code is never executed. Inputs are bounded by UTF-8 bytes,
    tokens, and batch size; scoring runs in Gabby's bounded sync callback pool.
    """

    model_id: str
    token_env: str = "HF_TOKEN"
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    device: str = "cpu"
    max_length: int = 512
    batch_size: int = 8
    max_input_bytes: int = 4 * 1024 * 1024
    timeout_seconds: float = 120.0
    relevance_label_index: int | None = None
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _torch: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty Hugging Face model ID or local path")
        self.model_id = self.model_id.strip()
        if not isinstance(self.token_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.token_env
        ):
            raise ValueError("token_env must be an environment variable name")
        for field_name, value in (("revision", self.revision), ("cache_dir", self.cache_dir)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty PyTorch device")
        self.device = self.device.strip()
        for field_name, numeric_value, upper_bound in (
            ("max_length", self.max_length, 8192),
            ("batch_size", self.batch_size, MAX_RETRIEVAL_DOCUMENTS),
            ("max_input_bytes", self.max_input_bytes, 64 * 1024 * 1024),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or not 1 <= numeric_value <= upper_bound
            ):
                raise ValueError(f"{field_name} must be an integer from 1 through {upper_bound}")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if self.relevance_label_index is not None and (
            isinstance(self.relevance_label_index, bool)
            or not isinstance(self.relevance_label_index, int)
            or self.relevance_label_index < 0
        ):
            raise ValueError("relevance_label_index must be a non-negative integer or None")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TransformersReranker:
        """Build local reranking options from host-owned, non-secret configuration."""
        validate_model_credentials(config, provider="transformers")
        model_id = config.get("model")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model must be a non-empty Hugging Face model ID or path")
        return cls(
            model_id=model_id,
            token_env=config.get("api_key_env", "HF_TOKEN"),
            revision=config.get("revision"),
            cache_dir=config.get("cache_dir"),
            local_files_only=config.get("local_files_only", False),
            device=config.get("device", "cpu"),
            max_length=config.get("max_length", 512),
            batch_size=config.get("batch_size", 8),
            max_input_bytes=config.get("max_input_bytes", 4 * 1024 * 1024),
            timeout_seconds=config.get("timeout_seconds", 120.0),
            relevance_label_index=config.get("relevance_label_index"),
        )

    async def rerank(
        self, query: str, documents: Sequence[Document], *, limit: int
    ) -> list[Document]:
        """Return documents ordered by a local model's finite relevance logits."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of Document values")
        if len(documents) > MAX_RETRIEVAL_DOCUMENTS:
            raise ValueError(f"documents must contain at most {MAX_RETRIEVAL_DOCUMENTS} values")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError("limit must be an integer from 0 through 100")
        if limit == 0 or not query.strip() or not documents:
            return []
        if any(
            not isinstance(document, Document) or not isinstance(document.text, str)
            for document in documents
        ):
            raise TypeError("documents must contain Document values with string text")
        byte_count = 0
        try:
            for text in (query, *(document.text for document in documents)):
                for offset in range(0, len(text), 64 * 1024):
                    byte_count += len(text[offset : offset + 64 * 1024].encode("utf-8"))
                    if byte_count > self.max_input_bytes:
                        raise RerankingError(
                            "Local Transformers reranking input exceeded max_input_bytes="
                            f"{self.max_input_bytes}"
                        )
        except UnicodeEncodeError:
            raise RerankingError(
                "Local Transformers reranking input contains invalid Unicode"
            ) from None

        stop_event = threading.Event()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                ranked = await run_sync_callback(
                    self._rerank_sync,
                    query,
                    list(documents),
                    min(limit, len(documents)),
                    stop_event,
                )
                if not isinstance(ranked, list) or any(
                    not isinstance(document, Document) for document in ranked
                ):
                    raise RerankingError("Local reranker worker returned an invalid result")
                return ranked
        except TimeoutError:
            stop_event.set()
            raise RerankingError("Local Transformers reranking exceeded its timeout") from None
        except asyncio.CancelledError:
            stop_event.set()
            raise

    def _rerank_sync(
        self,
        query: str,
        documents: list[Document],
        limit: int,
        stop_event: threading.Event,
    ) -> list[Document]:
        """Tokenize and score document batches while serializing model access."""
        with self._lock:
            try:
                tokenizer, model, torch = self._load_model()
                scores: list[float] = []
                for start in range(0, len(documents), self.batch_size):
                    if stop_event.is_set():
                        raise RerankingError("Local Transformers reranking was cancelled")
                    batch = documents[start : start + self.batch_size]
                    encoded = tokenizer(
                        [query] * len(batch),
                        [document.text for document in batch],
                        truncation=True,
                        max_length=self.max_length,
                        padding=True,
                        return_tensors="pt",
                    )
                    encoded = encoded.to(model.device)
                    with torch.inference_mode():
                        logits = model(**encoded).logits
                    if logits.ndim != 2 or logits.shape[0] != len(batch):
                        raise RerankingError("Local reranker returned an invalid score matrix")
                    if self.relevance_label_index is not None:
                        label_index = self.relevance_label_index
                    elif logits.shape[1] == 1:
                        label_index = 0
                    elif logits.shape[1] == 2:
                        label_index = 1
                    else:
                        raise RerankingError(
                            "Local reranker needs relevance_label_index for multi-class output"
                        )
                    if label_index >= logits.shape[1]:
                        raise RerankingError("Local reranker relevance_label_index is out of range")
                    batch_scores = logits[:, label_index].detach().to("cpu").tolist()
                    if not isinstance(batch_scores, list) or len(batch_scores) != len(batch):
                        raise RerankingError("Local reranker returned an invalid score vector")
                    for score in batch_scores:
                        if isinstance(score, bool) or not isinstance(score, (int, float)):
                            raise RerankingError("Local reranker returned a non-numeric score")
                        finite_score = float(score)
                        if not math.isfinite(finite_score):
                            raise RerankingError("Local reranker returned a non-finite score")
                        scores.append(finite_score)
                ranked_indexes = sorted(
                    range(len(scores)), key=lambda index: (-scores[index], index)
                )
                return [documents[index] for index in ranked_indexes[:limit]]
            except RerankingError:
                raise
            except Exception:
                raise RerankingError("Local Transformers reranking failed") from None

    def _load_model(self) -> tuple[Any, Any, Any]:
        """Lazily load tokenizer and safetensors-only sequence-classification weights."""
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise RerankingError(
                "Install Gabby's optional 'transformers' dependencies to use this reranker"
            ) from None
        options: dict[str, Any] = {
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "token": os.environ.get(self.token_env),
            "trust_remote_code": False,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **options)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                use_safetensors=True,
                **options,
            )
            model.to(self.device)
            model.eval()
        except Exception:
            raise RerankingError("Local Transformers reranker could not be loaded") from None
        self._tokenizer, self._model, self._torch = tokenizer, model, torch
        return tokenizer, model, torch

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
"""Opt-in acceptance for a real local Transformers embedding model."""

from __future__ import annotations

import math
import os

import pytest

from gabby import TransformersEmbeddingProvider


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_transformers_embedding_model() -> None:
    run_flag = "GABBY_RUN_TRANSFORMERS_EMBEDDINGS_INTEGRATION"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to load a local Transformers embedding model")
    model_id = os.environ.get("GABBY_TRANSFORMERS_EMBEDDINGS_MODEL")
    if not model_id:
        pytest.fail("set GABBY_TRANSFORMERS_EMBEDDINGS_MODEL to a model path or Hugging Face ID")

    provider = TransformersEmbeddingProvider(
        model_id=model_id,
        revision=os.environ.get("GABBY_TRANSFORMERS_EMBEDDINGS_REVISION"),
        device=os.environ.get("GABBY_TRANSFORMERS_EMBEDDINGS_DEVICE", "cpu"),
        local_files_only=os.environ.get("GABBY_TRANSFORMERS_EMBEDDINGS_LOCAL_FILES_ONLY", "1")
        != "0",
        max_input_tokens=128,
        batch_size=2,
        timeout_seconds=300,
    )
    try:
        vectors = await provider.embed(
            [
                "Authentication uses short-lived access tokens.",
                "A database index speeds up exact document lookup.",
            ]
        )
    finally:
        await provider.aclose()

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])
    for vector in vectors:
        assert all(math.isfinite(value) for value in vector)
        assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, abs_tol=1e-5)

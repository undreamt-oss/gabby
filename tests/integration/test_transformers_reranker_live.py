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
"""Opt-in acceptance against a locally installed Transformers sequence classifier."""

from __future__ import annotations

import os

import pytest

from gabby import Document, TransformersReranker


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_transformers_reranker_contract() -> None:
    run_flag = "GABBY_RUN_TRANSFORMERS_RERANKER_INTEGRATION"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to load a local Transformers reranker")
    model_id = os.environ.get("GABBY_TRANSFORMERS_RERANKER_MODEL")
    if not model_id:
        pytest.fail("set GABBY_TRANSFORMERS_RERANKER_MODEL to a local or Hub sequence classifier")

    reranker = TransformersReranker(
        model_id=model_id,
        revision=os.environ.get("GABBY_TRANSFORMERS_RERANKER_REVISION"),
        local_files_only=os.environ.get("GABBY_TRANSFORMERS_RERANKER_LOCAL_FILES_ONLY", "1") != "0",
        device=os.environ.get("GABBY_TRANSFORMERS_DEVICE", "cpu"),
    )
    documents = [
        Document(
            "Reset a forgotten password from Account Settings by choosing Security, "
            "then Reset password.",
            source="support-guide.md",
            id="password-help",
        ),
        Document(
            "The data analysis workspace has a CSV preview and column statistics.",
            source="data-guide.md",
            id="data-analysis",
        ),
        Document(
            "If the account email is unavailable, verify ownership with the support team.",
            source="support-guide.md",
            id="account-recovery",
        ),
    ]

    result = await reranker.rerank("How do I reset my account password?", documents, limit=2)

    candidates = {document.id for document in documents}
    assert 0 < len(result) <= 2
    assert len({document.id for document in result}) == len(result)
    assert all(document.id in candidates for document in result)

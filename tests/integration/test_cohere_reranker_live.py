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
"""Opt-in live contract acceptance for Cohere's reranking API."""

from __future__ import annotations

import os

import pytest

from gabby import CohereReranker, Document


@pytest.mark.asyncio
async def test_cohere_reranker_live_contract() -> None:
    if os.environ.get("GABBY_RUN_COHERE_RERANK_INTEGRATION") != "1":
        pytest.skip("set GABBY_RUN_COHERE_RERANK_INTEGRATION=1 to call the Cohere reranking API")
    if not os.environ.get("COHERE_API_KEY"):
        pytest.fail("COHERE_API_KEY must be configured for live Cohere acceptance")

    model = os.environ.get("GABBY_COHERE_RERANK_MODEL", "rerank-v4.0-fast")
    reranker = CohereReranker(model=model)
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
    try:
        result = await reranker.rerank("How do I reset my account password?", documents, limit=2)
    finally:
        await reranker.aclose()

    candidate_ids = {document.id for document in documents}
    assert 0 < len(result) <= 2
    assert all(document.id in candidate_ids for document in result)

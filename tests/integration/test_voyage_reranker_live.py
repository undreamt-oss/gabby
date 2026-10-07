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
"""Opt-in live acceptance for Voyage AI reranking."""

from __future__ import annotations

import os

import pytest

from gabby import Document, VoyageReranker


@pytest.mark.integration
@pytest.mark.asyncio
async def test_voyage_reranker_live_contract() -> None:
    run_flag = "GABBY_RUN_VOYAGE_RERANK_INTEGRATION"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to call the Voyage reranking API")
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.fail("set VOYAGE_API_KEY in the host environment for live Voyage acceptance")

    reranker = VoyageReranker(model=os.environ.get("GABBY_VOYAGE_RERANK_MODEL", "rerank-2.5-lite"))
    documents = [
        Document("Password reset: open account settings and choose Reset password.", id="reset"),
        Document("The support team is available Monday through Friday.", id="hours"),
    ]
    try:
        result = await reranker.rerank("How do I reset my account password?", documents, limit=2)
    finally:
        await reranker.aclose()

    assert len(result) == 2
    assert {document.id for document in result} == {"reset", "hours"}
    assert result[0].id == "reset"

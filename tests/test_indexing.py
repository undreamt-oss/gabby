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
"""Durable hybrid index generation coordination."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from gabby.indexing import (
    GenerationBusyError,
    GenerationLease,
    HybridIndexCoordinator,
    SQLiteGenerationManifestStore,
)
from gabby.knowledge import Document, KnowledgeStoreError, SQLiteFTS5Store


class Embeddings:
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [(float(len(text)), 1.0) for text in texts]


class SlowEmbeddings(Embeddings):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        await asyncio.sleep(2.5)
        return await super().embed(texts)


class Vectors:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, ...], Document] = {}
        self.fences: dict[str, int] = {}
        self.fail_stage = False

    async def upsert(self, documents: Sequence[Document], embeddings: Any) -> int:
        return len(documents)

    async def replace_source(
        self, source: str, documents: Sequence[Document], embeddings: Any
    ) -> int:
        await self.delete_source(source)
        for document in documents:
            self.rows[(source, document.generation or "")] = document
        return len(documents)

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
        embeddings: Any,
    ) -> int:
        if self.fail_stage:
            raise RuntimeError("simulated vector backend failure")
        if fencing_token < self.fences.get(source, 0):
            raise KnowledgeStoreError("stale vector fencing token")
        self.fences[source] = fencing_token
        for document in documents:
            self.rows[(source, generation, document.id or document.text)] = Document(
                document.text,
                source,
                dict(document.metadata),
                document.id,
                generation,
            )
        return len(documents)

    async def delete_source(self, source: str) -> int:
        old = len(self.rows)
        self.rows = {key: value for key, value in self.rows.items() if key[0] != source}
        return old - len(self.rows)

    async def delete_generation(self, source: str, generation: str) -> int:
        old = len(self.rows)
        self.rows = {
            key: value
            for key, value in self.rows.items()
            if not (key[0] == source and key[1] == generation)
        }
        return old - len(self.rows)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        if fencing_token < self.fences.get(source, 0):
            raise KnowledgeStoreError("stale vector fencing token")
        self.fences[source] = fencing_token
        return await self.delete_generation(source, generation)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        if fencing_token < self.fences.get(source, 0):
            raise KnowledgeStoreError("stale vector fencing token")
        self.fences[source] = fencing_token

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        matches = [
            document
            for document in self.rows.values()
            if generations is None or generations.get(document.source) == document.generation
        ]
        return matches[:limit]


class DurableVectors(Vectors):
    """Small SQLite vector fake whose staged rows survive a writer process exit."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript(
                """CREATE TABLE IF NOT EXISTS test_vectors (
                       source TEXT NOT NULL,
                       generation TEXT NOT NULL,
                       document_id TEXT NOT NULL,
                       text TEXT NOT NULL,
                       metadata TEXT NOT NULL,
                       embedding TEXT NOT NULL,
                       PRIMARY KEY(source, generation, document_id)
                   );
                   CREATE TABLE IF NOT EXISTS test_vector_fences (
                       source TEXT PRIMARY KEY,
                       fencing_token INTEGER NOT NULL
                   );"""
            )

    async def stage_source(
        self,
        source: str,
        generation: str,
        fencing_token: int,
        documents: Sequence[Document],
        embeddings: Any,
    ) -> int:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                current = connection.execute(
                    "SELECT fencing_token FROM test_vector_fences WHERE source = ?",
                    (source,),
                ).fetchone()
                if current is not None and fencing_token < int(current[0]):
                    raise KnowledgeStoreError("stale vector fencing token")
                connection.execute(
                    "INSERT INTO test_vector_fences(source, fencing_token) VALUES (?, ?) "
                    "ON CONFLICT(source) DO UPDATE SET fencing_token = excluded.fencing_token",
                    (source, fencing_token),
                )
                for document, embedding in zip(documents, embeddings, strict=True):
                    connection.execute(
                        "INSERT OR REPLACE INTO test_vectors "
                        "(source, generation, document_id, text, metadata, embedding) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            source,
                            generation,
                            document.id or document.text,
                            document.text,
                            json.dumps(dict(document.metadata)),
                            json.dumps(list(embedding)),
                        ),
                    )
        return len(documents)

    async def discard_generation(self, source: str, generation: str, fencing_token: int) -> int:
        await self.advance_fence(source, fencing_token)
        return await self.delete_generation(source, generation)

    async def advance_fence(self, source: str, fencing_token: int) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                current = connection.execute(
                    "SELECT fencing_token FROM test_vector_fences WHERE source = ?",
                    (source,),
                ).fetchone()
                if current is not None and fencing_token < int(current[0]):
                    raise KnowledgeStoreError("stale vector fencing token")
                connection.execute(
                    "INSERT INTO test_vector_fences(source, fencing_token) VALUES (?, ?) "
                    "ON CONFLICT(source) DO UPDATE SET fencing_token = excluded.fencing_token",
                    (source, fencing_token),
                )

    async def delete_generation(self, source: str, generation: str) -> int:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            cursor = connection.execute(
                "DELETE FROM test_vectors WHERE source = ? AND generation = ?",
                (source, generation),
            )
            return cursor.rowcount

    async def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        filters: dict[str, Any] | None = None,
        generations: Mapping[str, str] | None = None,
    ) -> list[Document]:
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT source, generation, document_id, text, metadata "
                "FROM test_vectors ORDER BY rowid"
            ).fetchall()
        documents = [
            Document(
                text=row[3],
                source=row[0],
                metadata=json.loads(row[4]),
                id=row[2],
                generation=row[1],
            )
            for row in rows
            if generations is None or generations.get(row[0]) == row[1]
        ]
        if filters:
            documents = [
                document
                for document in documents
                if all(document.metadata.get(key) == value for key, value in filters.items())
            ]
        return documents[:limit]


def make_coordinator(tmp_path: Path, vectors: Vectors | None = None) -> HybridIndexCoordinator:
    return HybridIndexCoordinator(
        SQLiteFTS5Store(tmp_path / "lexical.db"),
        Embeddings(),
        vectors or Vectors(),
        SQLiteGenerationManifestStore(tmp_path / "manifest.db"),
    )


async def begin(
    manifest: SQLiteGenerationManifestStore,
    source: str,
    generation: str,
    count: int,
    *,
    lease_ttl_seconds: float = 60,
) -> GenerationLease:
    return await manifest.begin_source(
        source,
        generation,
        count,
        owner_id="test-owner",
        lease_ttl_seconds=lease_ttl_seconds,
    )


async def mark_ready(
    manifest: SQLiteGenerationManifestStore,
    lease: GenerationLease,
    backend: str,
) -> None:
    await manifest.mark_ready(
        lease.source,
        lease.generation,
        backend,  # type: ignore[arg-type]
        owner_id=lease.owner_id,
        fencing_token=lease.fencing_token,
    )


async def activate(manifest: SQLiteGenerationManifestStore, lease: GenerationLease) -> None:
    await manifest.activate_source(
        lease.source,
        lease.generation,
        owner_id=lease.owner_id,
        fencing_token=lease.fencing_token,
    )


def test_manifest_rejects_nonpersistent_paths_and_invalid_timeouts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="persistent database"):
        SQLiteGenerationManifestStore(":memory:")
    for timeout in (True, 0, float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            SQLiteGenerationManifestStore(tmp_path / "manifest.db", busy_timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_manifest_validates_generation_identity_and_readiness(tmp_path: Path) -> None:
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    with pytest.raises(ValueError, match="source"):
        await manifest.begin_source(" ", "g", 0, owner_id="owner")
    with pytest.raises(ValueError, match="generation"):
        await manifest.begin_source("source", " ", 0, owner_id="owner")
    with pytest.raises(TypeError, match="document_count"):
        await manifest.begin_source("source", "g", True, owner_id="owner")
    with pytest.raises(ValueError, match="document_count"):
        await manifest.begin_source("source", "g", -1, owner_id="owner")
    with pytest.raises(ValueError, match="lease_ttl_seconds"):
        await manifest.begin_source("source", "bad-ttl", 0, owner_id="owner", lease_ttl_seconds=0)
    with pytest.raises(ValueError, match="owner_id"):
        await manifest.begin_source("source", "bad-owner", 0, owner_id=" ")
    lease = await begin(manifest, "source", "g", 1)
    with pytest.raises(GenerationBusyError, match="pending generation"):
        await begin(manifest, "source", "g2", 1)
    with pytest.raises(ValueError, match="backend"):
        await manifest.mark_ready(
            "source",
            "g",
            cast(Any, "other"),
            owner_id=lease.owner_id,
            fencing_token=lease.fencing_token,
        )
    with pytest.raises(KnowledgeStoreError, match="missing"):
        await manifest.mark_ready("source", "missing", "lexical", owner_id="owner", fencing_token=1)
    with pytest.raises(KnowledgeStoreError, match="missing"):
        await manifest.activate_source(
            "source", "missing", owner_id=lease.owner_id, fencing_token=lease.fencing_token
        )


@pytest.mark.asyncio
async def test_manifest_activation_is_idempotent_and_retired_removal_is_safe(
    tmp_path: Path,
) -> None:
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    lease1 = await begin(manifest, "source", "g1", 1)
    await mark_ready(manifest, lease1, "lexical")
    await mark_ready(manifest, lease1, "vector")
    await activate(manifest, lease1)
    await activate(manifest, lease1)
    lease2 = await begin(manifest, "source", "g2", 0)
    await mark_ready(manifest, lease2, "lexical")
    await mark_ready(manifest, lease2, "vector")
    await activate(manifest, lease2)

    assert await manifest.retired_generations() == [("source", "g1")]
    await manifest.remove_retired("source", "g1")
    await manifest.remove_retired("source", "g1")
    assert await manifest.retired_generations() == []
    assert await manifest.active_document_count("missing") == 0


@pytest.mark.asyncio
async def test_recovery_does_not_delete_generation_with_live_writer_lease(
    tmp_path: Path,
) -> None:
    vectors = Vectors()
    coordinator = make_coordinator(tmp_path, vectors)
    lease = await coordinator.manifest.begin_source(
        "guide.md", "live-generation", 1, owner_id="live-writer", lease_ttl_seconds=60
    )
    document = Document("writer is still staging", "guide.md", id="live")
    await coordinator.lexical.stage_source(
        lease.source, lease.generation, lease.fencing_token, [document]
    )
    await vectors.stage_source(
        lease.source, lease.generation, lease.fencing_token, [document], [[1.0, 1.0]]
    )

    await coordinator.reconcile()

    pending = await coordinator.manifest.pending_generations()
    assert len(pending) == 1
    assert pending[0].owner_id == "live-writer"
    assert ("guide.md", "live-generation", "live") in vectors.rows
    with pytest.raises(GenerationBusyError, match="pending generation"):
        await coordinator.replace_source("guide.md", [Document("new", "guide.md")])


@pytest.mark.asyncio
async def test_next_write_recovers_expired_lease_before_retrying_source(
    tmp_path: Path,
) -> None:
    coordinator = make_coordinator(tmp_path)
    await coordinator.retrieve("empty startup reconciliation")
    stale = await coordinator.manifest.begin_source(
        "guide.md", "abandoned", 1, owner_id="crashed-writer", lease_ttl_seconds=60
    )
    await coordinator.manifest.release_lease(
        stale.source,
        stale.generation,
        owner_id=stale.owner_id,
        fencing_token=stale.fencing_token,
    )

    await coordinator.replace_source("guide.md", [Document("replacement", "guide.md")])

    assert await coordinator.manifest.pending_generations() == []
    assert await coordinator.manifest.active_document_count("guide.md") == 1


@pytest.mark.asyncio
async def test_expired_lease_fences_out_stale_writer_and_backend_stages(
    tmp_path: Path,
) -> None:
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    lexical = SQLiteFTS5Store(tmp_path / "lexical.db")
    vectors = Vectors()
    stale = await manifest.begin_source(
        "guide.md", "generation-1", 1, owner_id="old-writer", lease_ttl_seconds=0.01
    )
    stale_document = Document("stale staged content", "guide.md", id="stale")
    await lexical.stage_source(
        stale.source, stale.generation, stale.fencing_token, [stale_document]
    )
    await vectors.stage_source(
        stale.source, stale.generation, stale.fencing_token, [stale_document], [[1.0, 1.0]]
    )
    await asyncio.sleep(0.03)
    recovery = await manifest.claim_expired(
        stale.source,
        stale.generation,
        owner_id="recovery-writer",
        lease_ttl_seconds=60,
    )
    assert recovery is not None
    assert recovery.fencing_token > stale.fencing_token
    await lexical.discard_generation(stale.source, stale.generation, recovery.fencing_token)
    await vectors.discard_generation(stale.source, stale.generation, recovery.fencing_token)

    with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
        await lexical.stage_source(
            stale.source, stale.generation, stale.fencing_token, [stale_document]
        )
    with pytest.raises(KnowledgeStoreError, match="stale vector fencing token"):
        await vectors.stage_source(
            stale.source, stale.generation, stale.fencing_token, [stale_document], [[1.0, 1.0]]
        )
    with pytest.raises(KnowledgeStoreError, match="missing index generation ready"):
        await manifest.mark_ready(
            stale.source,
            stale.generation,
            "lexical",
            owner_id=stale.owner_id,
            fencing_token=stale.fencing_token,
        )


@pytest.mark.asyncio
async def test_live_lease_renews_and_expired_lease_can_be_claimed(tmp_path: Path) -> None:
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    lease = await manifest.begin_source(
        "source", "generation", 1, owner_id="writer", lease_ttl_seconds=0.3
    )
    assert (
        await manifest.claim_expired(
            lease.source,
            lease.generation,
            owner_id="too-early",
            lease_ttl_seconds=1,
        )
        is None
    )
    assert await manifest.renew_lease(
        lease.source,
        lease.generation,
        lease.owner_id,
        lease.fencing_token,
        0.3,
    )
    await asyncio.sleep(0.35)
    assert not await manifest.renew_lease(
        lease.source,
        lease.generation,
        lease.owner_id,
        lease.fencing_token,
        0.3,
    )

    takeover = await manifest.claim_expired(
        lease.source,
        lease.generation,
        owner_id="recovery",
        lease_ttl_seconds=1,
    )
    assert takeover is not None
    assert takeover.fencing_token > lease.fencing_token


@pytest.mark.asyncio
async def test_only_one_process_can_claim_an_expired_generation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.db"
    manifest = SQLiteGenerationManifestStore(manifest_path)
    stale = await manifest.begin_source(
        "source", "generation", 1, owner_id="crashed", lease_ttl_seconds=0.01
    )
    await asyncio.sleep(0.03)
    script = textwrap.dedent(
        """
        import asyncio
        import sys
        from gabby.indexing import SQLiteGenerationManifestStore

        async def main():
            manifest = SQLiteGenerationManifestStore(sys.argv[1])
            lease = await manifest.claim_expired(
                "source", "generation", owner_id=sys.argv[2], lease_ttl_seconds=5
            )
            print("none" if lease is None else str(lease.fencing_token))

        asyncio.run(main())
        """
    )
    project_root = Path(__file__).resolve().parents[1]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(project_root / "src")

    process_specs = [
        ("recovery-a", tmp_path / "claim-a.log"),
        ("recovery-b", tmp_path / "claim-b.log"),
    ]
    processes: list[subprocess.Popen[bytes]] = []
    with (
        process_specs[0][1].open("wb") as first_output,
        process_specs[1][1].open("wb") as second_output,
    ):
        for (owner, _), output in zip(process_specs, (first_output, second_output), strict=True):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", script, str(manifest_path), owner],
                    cwd=project_root,
                    env=child_env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            )
        try:
            return_codes = [process.wait(timeout=10) for process in processes]
        except subprocess.TimeoutExpired:
            for process in processes:
                process.kill()
            for process in processes:
                process.wait()
            raise

    outputs = [
        (tmp_path / "claim-a.log").read_text().strip(),
        (tmp_path / "claim-b.log").read_text().strip(),
    ]
    assert return_codes == [0, 0], outputs
    tokens = [output for output in outputs if output != "none"]
    assert len(tokens) == 1
    assert int(tokens[0]) > stale.fencing_token


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_point",
    ["during_staging", "after_lexical_cleanup", "after_ready", "after_recovery_fences"],
)
async def test_process_crash_is_recovered(tmp_path: Path, crash_point: str) -> None:
    manifest_path = tmp_path / "crash-manifest.db"
    lexical_path = tmp_path / "crash-lexical.db"
    source = "guide.md"
    generation = "crashed-generation"
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import sqlite3
        import sys
        from contextlib import closing
        from gabby.indexing import SQLiteGenerationManifestStore
        from gabby.knowledge import Document, SQLiteFTS5Store
        from test_indexing import DurableVectors

        async def main():
            manifest = SQLiteGenerationManifestStore(sys.argv[1])
            lease = await manifest.begin_source(
                "guide.md", "crashed-generation", 1,
                owner_id="crashing-writer", lease_ttl_seconds=60,
            )
            await SQLiteFTS5Store(sys.argv[2]).stage_source(
                "guide.md", "crashed-generation", lease.fencing_token,
                [Document("partially committed lexical text", "guide.md", id="crashed")],
            )
            vectors = DurableVectors(sys.argv[3])
            if sys.argv[4] in {"after_ready", "after_recovery_fences"}:
                await vectors.stage_source(
                    "guide.md", "crashed-generation", lease.fencing_token,
                    [Document("partially committed lexical text", "guide.md", id="crashed")],
                    [[1.0, 1.0]],
                )
                await manifest.mark_ready(
                    "guide.md", "crashed-generation", "lexical",
                    owner_id=lease.owner_id, fencing_token=lease.fencing_token,
                )
                await manifest.mark_ready(
                    "guide.md", "crashed-generation", "vector",
                    owner_id=lease.owner_id, fencing_token=lease.fencing_token,
                )
            if sys.argv[4] in {"after_recovery_fences", "after_lexical_cleanup"}:
                with closing(sqlite3.connect(sys.argv[1])) as connection:
                    connection.execute(
                        "UPDATE gabby_index_generations SET lease_expires_at = 0 "
                        "WHERE source = ? AND generation = ?",
                        ("guide.md", "crashed-generation"),
                    )
                    connection.commit()
                recovery_lease = await manifest.claim_expired(
                    "guide.md", "crashed-generation", owner_id="recovery-crasher",
                    lease_ttl_seconds=60,
                )
                if recovery_lease is None:
                    raise RuntimeError("expected to claim expired generation")
                lexical = SQLiteFTS5Store(sys.argv[2])
                if sys.argv[4] == "after_recovery_fences":
                    await lexical.advance_fence("guide.md", recovery_lease.fencing_token)
                    await vectors.advance_fence("guide.md", recovery_lease.fencing_token)
                else:
                    await lexical.discard_generation(
                        "guide.md", "crashed-generation", recovery_lease.fencing_token
                    )
                print(lease.fencing_token, recovery_lease.fencing_token, flush=True)
            else:
                print(lease.fencing_token, flush=True)
            os._exit(0)

        asyncio.run(main())
        """
    )
    project_root = Path(__file__).resolve().parents[1]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root / "src"), str(project_root / "tests")]
    )
    vector_path = tmp_path / "crash-vectors.db"
    output_path = tmp_path / "crash-output.log"
    with output_path.open("wb") as output:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(manifest_path),
                    str(lexical_path),
                    str(vector_path),
                    crash_point,
                ],
                cwd=project_root,
                env=child_env,
                stdout=output,
                stderr=output,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(f"Crash-recovery helper timed out at {crash_point!r}: {exc}")
    child_output = output_path.read_text(encoding="utf-8")
    assert completed.returncode == 0, child_output
    stale_fences = [int(value) for value in child_output.split()]
    with closing(sqlite3.connect(manifest_path)) as connection:
        connection.execute(
            "UPDATE gabby_index_generations SET lease_expires_at = 0 "
            "WHERE source = ? AND generation = ?",
            (source, generation),
        )
        connection.commit()

    lexical = SQLiteFTS5Store(lexical_path)
    manifest = SQLiteGenerationManifestStore(manifest_path)
    vectors = DurableVectors(vector_path)
    coordinator = HybridIndexCoordinator(lexical, Embeddings(), vectors, manifest)
    results = await coordinator.retrieve("partially committed")
    assert await manifest.pending_generations() == []
    if crash_point in {"after_ready", "after_recovery_fences"}:
        assert (await manifest.active_generations()) == {source: generation}
        assert [item.id for item in results] == ["crashed"]
    else:
        assert results == []
        assert await manifest.retired_generations() == [(source, generation)]
    for index, stale_fence in enumerate(stale_fences):
        with pytest.raises(KnowledgeStoreError, match="Stale fencing token"):
            await lexical.stage_source(
                source,
                generation,
                stale_fence,
                [Document("late stale write", source, id=f"late-{index}")],
            )
        with pytest.raises(KnowledgeStoreError, match="stale vector fencing token"):
            await vectors.stage_source(
                source,
                generation,
                stale_fence,
                [Document("late stale vector write", source, id=f"late-vector-{index}")],
                [[1.0, 1.0]],
            )


@pytest.mark.asyncio
async def test_coordinator_renews_lease_during_slow_embedding(tmp_path: Path) -> None:
    coordinator = HybridIndexCoordinator(
        SQLiteFTS5Store(tmp_path / "lexical.db"),
        SlowEmbeddings(),
        Vectors(),
        SQLiteGenerationManifestStore(tmp_path / "manifest.db"),
        lease_ttl_seconds=2,
    )

    await coordinator.replace_source("guide.md", [Document("slow embedded text", "guide.md")])

    assert await coordinator.manifest.active_document_count("guide.md") == 1


@pytest.mark.asyncio
async def test_coordinator_activates_only_complete_generation_and_retires_old(
    tmp_path: Path,
) -> None:
    vectors = Vectors()
    coordinator = make_coordinator(tmp_path, vectors)
    await coordinator.replace_source(
        "guide.md", [Document("old authentication guide", "guide.md", id="same")]
    )
    old_generation = (await coordinator.manifest.active_generations())["guide.md"]

    await coordinator.replace_source(
        "guide.md", [Document("new authentication guide", "guide.md", id="same")]
    )

    active = await coordinator.manifest.active_generations()
    assert active["guide.md"] != old_generation
    assert [doc.text for doc in await coordinator.retrieve("new authentication")] == [
        "new authentication guide"
    ]
    lexical_results = await coordinator.lexical.retrieve("old", generations=active)
    assert lexical_results == []
    assert await coordinator.manifest.retired_generations() == [("guide.md", old_generation)]
    with pytest.raises(ValueError, match="readers_quiescent"):
        await coordinator.prune_retired()
    assert await coordinator.prune_retired(readers_quiescent=True) == 1
    assert not any(key[0] == "guide.md" and key[1] == old_generation for key in vectors.rows)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "crash_point",
    ["after_lexical_delete", "after_vector_delete", "after_backend_cleanup"],
)
async def test_process_crash_during_retired_generation_pruning_is_recovered(
    tmp_path: Path, crash_point: str
) -> None:
    lexical_path = tmp_path / "lexical.db"
    manifest_path = tmp_path / "manifest.db"
    vector_path = tmp_path / "vectors.db"
    vectors = DurableVectors(vector_path)
    coordinator = HybridIndexCoordinator(
        SQLiteFTS5Store(lexical_path),
        Embeddings(),
        vectors,
        SQLiteGenerationManifestStore(manifest_path),
    )
    await coordinator.replace_source("guide.md", [Document("old guide", "guide.md", id="old")])
    old_generation = (await coordinator.manifest.active_generations())["guide.md"]
    await coordinator.replace_source(
        "guide.md", [Document("current guide", "guide.md", id="current")]
    )
    active_generation = (await coordinator.manifest.active_generations())["guide.md"]
    assert active_generation != old_generation
    assert await coordinator.manifest.retired_generations() == [("guide.md", old_generation)]

    # Simulate process exits after each durable cleanup boundary, before the
    # manifest record is removed.
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import sqlite3
        import sys
        from contextlib import closing
        from gabby.knowledge import SQLiteFTS5Store

        async def main():
            crash_point = sys.argv[4]
            if crash_point in {"after_lexical_delete", "after_backend_cleanup"}:
                await SQLiteFTS5Store(sys.argv[1]).delete_generation(sys.argv[2], sys.argv[3])
            if crash_point in {"after_vector_delete", "after_backend_cleanup"}:
                with closing(sqlite3.connect(sys.argv[5])) as connection, connection:
                    connection.execute(
                        "DELETE FROM test_vectors WHERE source = ? AND generation = ?",
                        (sys.argv[2], sys.argv[3]),
                    )
            os._exit(0)

        asyncio.run(main())
        """
    )
    project_root = Path(__file__).resolve().parents[1]
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(lexical_path),
            "guide.md",
            old_generation,
            crash_point,
            str(vector_path),
        ],
        cwd=project_root,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    restarted = HybridIndexCoordinator(
        SQLiteFTS5Store(lexical_path),
        Embeddings(),
        DurableVectors(vector_path),
        SQLiteGenerationManifestStore(manifest_path),
    )
    assert await restarted.prune_retired(readers_quiescent=True) == 1
    assert await restarted.manifest.retired_generations() == []
    assert [document.id for document in await restarted.retrieve("current")] == ["current"]
    with closing(sqlite3.connect(lexical_path)) as connection:
        lexical_rows = connection.execute(
            "SELECT COUNT(*) FROM gabby_documents WHERE source = ? AND generation = ?",
            ("guide.md", old_generation),
        ).fetchone()
    with closing(sqlite3.connect(vector_path)) as connection:
        vector_rows = connection.execute(
            "SELECT COUNT(*) FROM test_vectors WHERE source = ? AND generation = ?",
            ("guide.md", old_generation),
        ).fetchone()
    assert lexical_rows == (0,)
    assert vector_rows == (0,)


@pytest.mark.asyncio
async def test_coordinator_recovers_incomplete_generation_by_discarding_staging(
    tmp_path: Path,
) -> None:
    vectors = Vectors()
    coordinator = make_coordinator(tmp_path, vectors)
    vectors.fail_stage = True

    with pytest.raises(KnowledgeStoreError, match="Hybrid source generation failed"):
        await coordinator.replace_source(
            "guide.md", [Document("partially staged text", "guide.md", id="staged")]
        )
    assert len(await coordinator.manifest.pending_generations()) == 1
    vectors.fail_stage = False

    assert await coordinator.retrieve("partially staged") == []
    assert await coordinator.manifest.pending_generations() == []
    assert not vectors.rows
    await coordinator.replace_source(
        "guide.md", [Document("new committed text", "guide.md", id="committed")]
    )
    assert [doc.id for doc in await coordinator.retrieve("committed text")] == ["committed"]


@pytest.mark.asyncio
async def test_manifest_reopens_durably_and_enforces_readiness(tmp_path: Path) -> None:
    path = tmp_path / "manifest.db"
    manifest = SQLiteGenerationManifestStore(path)
    lease = await begin(manifest, "source", "gen-1", 2)
    with pytest.raises(KnowledgeStoreError, match="incompletely staged"):
        await activate(manifest, lease)
    await mark_ready(manifest, lease, "lexical")
    await mark_ready(manifest, lease, "vector")
    await activate(SQLiteGenerationManifestStore(path), lease)

    reopened = SQLiteGenerationManifestStore(path)
    assert await reopened.active_generations() == {"source": "gen-1"}
    assert await reopened.active_document_count("source") == 2


@pytest.mark.asyncio
async def test_manifest_rejects_schema_newer_than_supported(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "future-manifest.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version = 3")

    with pytest.raises(KnowledgeStoreError, match="newer than supported"):
        await SQLiteGenerationManifestStore(path).active_generations()


@pytest.mark.asyncio
async def test_manifest_migrates_legacy_pending_generation_as_expired(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "legacy-manifest.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """CREATE TABLE gabby_index_generations (
                   source TEXT NOT NULL,
                   generation TEXT NOT NULL,
                   state TEXT NOT NULL,
                   document_count INTEGER NOT NULL,
                   lexical_ready INTEGER NOT NULL,
                   vector_ready INTEGER NOT NULL,
                   error_type TEXT,
                   created_at REAL NOT NULL,
                   PRIMARY KEY(source, generation)
               );
               CREATE UNIQUE INDEX gabby_one_pending_generation
                   ON gabby_index_generations(source) WHERE state = 'pending';
               CREATE UNIQUE INDEX gabby_one_active_generation
                   ON gabby_index_generations(source) WHERE state = 'active';
               INSERT INTO gabby_index_generations
                   (source, generation, state, document_count, lexical_ready, vector_ready,
                    created_at)
                   VALUES ('legacy-source', 'legacy-generation', 'pending', 1, 0, 0, 0);
               PRAGMA user_version = 1;"""
        )

    manifest = SQLiteGenerationManifestStore(path)
    pending = await manifest.pending_generations()
    assert pending[0].lease_expires_at == 0
    lease = await manifest.claim_expired(
        "legacy-source",
        "legacy-generation",
        owner_id="migration-recovery",
        lease_ttl_seconds=1,
    )
    assert lease is not None
    assert lease.fencing_token == 1


@pytest.mark.asyncio
async def test_coordinator_recovers_fully_staged_generation_after_restart(
    tmp_path: Path,
) -> None:
    vectors = Vectors()
    lexical = SQLiteFTS5Store(tmp_path / "lexical.db")
    manifest = SQLiteGenerationManifestStore(tmp_path / "manifest.db")
    document = Document("recovered generation content", "guide.md", id="recovered")
    generation = "complete-pending"
    lease = await begin(manifest, "guide.md", generation, 1)
    await lexical.stage_source("guide.md", generation, lease.fencing_token, [document])
    await vectors.stage_source(
        "guide.md", generation, lease.fencing_token, [document], [[1.0, 1.0]]
    )
    await mark_ready(manifest, lease, "lexical")
    await mark_ready(manifest, lease, "vector")
    await manifest.release_lease(
        lease.source,
        lease.generation,
        owner_id=lease.owner_id,
        fencing_token=lease.fencing_token,
    )
    coordinator = HybridIndexCoordinator(lexical, Embeddings(), vectors, manifest)

    results = await coordinator.retrieve("recovered generation")

    assert (await manifest.active_generations())["guide.md"] == generation
    assert [item.id for item in results] == ["recovered"]


@pytest.mark.asyncio
async def test_coordinator_delete_uses_empty_active_generation(tmp_path: Path) -> None:
    coordinator = make_coordinator(tmp_path)
    await coordinator.replace_source("guide.md", [Document("retire me", "guide.md")])

    assert await coordinator.delete_source("guide.md") == 1
    assert await coordinator.retrieve("retire me") == []
    assert "guide.md" in await coordinator.manifest.active_generations()


@pytest.mark.asyncio
async def test_coordinator_validates_source_documents_and_lease_ttl(tmp_path: Path) -> None:
    coordinator = make_coordinator(tmp_path)
    with pytest.raises(ValueError, match="source"):
        await coordinator.replace_source(" ", [])
    with pytest.raises(ValueError, match="requested source"):
        await coordinator.replace_source("one", [Document("wrong source", "two")])
    with pytest.raises(ValueError, match="must not set a generation"):
        await coordinator.replace_source(
            "one", [Document("pre-tagged", "one", generation="external")]
        )
    with pytest.raises(ValueError, match="lease_ttl_seconds"):
        HybridIndexCoordinator(
            SQLiteFTS5Store(tmp_path / "other-lexical.db"),
            Embeddings(),
            Vectors(),
            SQLiteGenerationManifestStore(tmp_path / "other-manifest.db"),
            lease_ttl_seconds=float("nan"),
        )

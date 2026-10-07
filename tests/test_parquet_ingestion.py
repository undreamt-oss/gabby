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
"""Contract tests for bounded optional Parquet knowledge ingestion."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest

arrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")

from gabby import FileIngestor, ParquetTextParser, SQLiteFTS5Store  # noqa: E402


def _parquet_bytes(table: object) -> bytes:
    output = BytesIO()
    parquet.write_table(table, output)
    return output.getvalue()


def test_parquet_parser_creates_ordered_bounded_pages_with_row_citations() -> None:
    table = arrow.table(
        {
            "Account": ["Ada", "Grace", "Linus"],
            "active": [True, False, True],
            "balance": [Decimal("12.50"), Decimal("0.00"), Decimal("99.95")],
            "joined": [date(2020, 1, 2), date(2021, 2, 3), date(2022, 3, 4)],
        }
    )

    pages = ParquetTextParser(max_rows_per_page=2).parse(_parquet_bytes(table))

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].metadata == {"row_start": 1, "row_end": 2}
    assert '"Account": "Ada"' in pages[0].text
    assert '"balance": "12.50"' in pages[0].text
    assert '"joined": "2020-01-02"' in pages[0].text
    assert pages[1].metadata == {"row_start": 3, "row_end": 3}
    assert '"Account": "Linus"' in pages[1].text


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_rows": 1}, "max_rows=1"),
        ({"max_columns": 1}, "max_columns=1"),
        ({"max_uncompressed_bytes": 1}, "max_uncompressed_bytes=1"),
        ({"max_row_bytes": 1}, "max_row_bytes=1"),
        ({"max_output_bytes": 1}, "max_output_bytes=1"),
        ({"max_cell_bytes": 1}, "max_cell_bytes=1"),
        ({"max_pages": 1, "max_rows_per_page": 1}, "max_pages=1"),
    ],
)
def test_parquet_parser_enforces_configured_bounds(options: dict[str, int], message: str) -> None:
    data = _parquet_bytes(arrow.table({"name": ["Ada", "Grace"], "age": [36, 85]}))

    with pytest.raises(ValueError, match=message):
        ParquetTextParser(**options).parse(data)


def test_parquet_parser_rejects_nested_and_binary_fields() -> None:
    nested = _parquet_bytes(arrow.table({"items": [[1, 2]]}))
    binary = _parquet_bytes(arrow.table({"payload": [b"secret"]}))

    with pytest.raises(ValueError, match="unsupported nested or binary"):
        ParquetTextParser().parse(nested)
    with pytest.raises(ValueError, match="unsupported nested or binary"):
        ParquetTextParser().parse(binary)


def test_parquet_parser_rejects_invalid_and_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="could not read Parquet"):
        ParquetTextParser().parse(b"not a parquet file")
    with pytest.raises(ValueError, match="max_input_bytes=4"):
        ParquetTextParser(max_input_bytes=4).parse(b"PAR1toolong")


@pytest.mark.asyncio
async def test_file_ingestor_indexes_parquet_records(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    (root / "customers.parquet").write_bytes(
        _parquet_bytes(arrow.table({"name": ["Ada", "Grace"], "city": ["London", "New York"]}))
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    assert ".parquet" in ingestor.supported_extensions
    assert await ingestor.ingest_file("customers.parquet") == 1
    documents = await store.retrieve("Grace New York", limit=5)
    assert len(documents) == 1
    assert '"name": "Grace"' in documents[0].text
    assert '"city": "New York"' in documents[0].text
    assert documents[0].metadata["row_start"] == 1

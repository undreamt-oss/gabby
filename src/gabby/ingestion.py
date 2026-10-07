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
"""Root-confined page-aware file ingestion and deterministic text chunking."""

from __future__ import annotations

import codecs
import csv
import datetime
import decimal
import hashlib
import json
import math
import posixpath
import re
import tomllib
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from email.message import Message
from email.parser import BytesParser
from email.policy import default as default_email_policy
from html.parser import HTMLParser
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

import yaml

from ._sync import run_sync_callback
from .knowledge import Document, SourceKnowledgeWriter


class TextChunker(Protocol):
    """Synchronous, replaceable text-to-chunks strategy."""

    def chunk(self, text: str) -> Sequence[str]:
        """Return deterministic, non-empty chunks in source order."""
        ...


@dataclass(frozen=True)
class ParsedPage:
    """Text extracted from one logical page of a source document."""

    text: str
    page_number: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRPageResult:
    """OCR output and decoded-pixel usage for one raster frame."""

    text: str
    pixel_count: int


@dataclass(frozen=True)
class PDFRenderedPage:
    """A bounded rasterized PDF page and its decoded pixel count."""

    image: bytes
    pixel_count: int


class FileParser(Protocol):
    """Replaceable synchronous parser for a bounded source file."""

    extensions: frozenset[str]

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Extract source-ordered pages from the provided bytes."""
        ...


class _EmailInputError(ValueError):
    """A bounded email message violates the parser's supported input contract."""


class OCRBackend(Protocol):
    """Synchronous, replaceable OCR engine for bounded raster image sources."""

    def recognize(
        self,
        content: bytes,
        *,
        max_pages: int,
        max_image_pixels: int,
        timeout_seconds: float,
    ) -> Sequence[OCRPageResult]:
        """Return source-ordered OCR text and pixel usage for each image frame."""
        ...


class PDFPageRenderer(Protocol):
    """Synchronous, replaceable PDF-page rasterizer for opt-in OCR."""

    def open_document(self, content: bytes) -> PDFPageRenderSession:
        """Open one run-scoped renderer session for the PDF bytes."""
        ...


class PDFPageRenderSession(Protocol):
    """Run-scoped renderer session reused across pages of one PDF."""

    def render_page(
        self, page_number: int, *, max_image_pixels: int, max_image_bytes: int
    ) -> PDFRenderedPage:
        """Render one-based page ``page_number`` within both image bounds."""
        ...

    def close(self) -> None:
        """Release native rendering resources."""
        ...


class PDFiumPageRenderer:
    """Rasterize vector-only PDF pages using the optional ``pdf-ocr`` extra."""

    def __init__(self, *, scale: float = 2.0, jpeg_quality: int = 85) -> None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError("scale must be a positive number")
        if not math.isfinite(scale) or not 0.1 <= scale <= 4.0:
            raise ValueError("scale must be from 0.1 through 4.0")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise TypeError("jpeg_quality must be an integer")
        if not 30 <= jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be from 30 through 95")
        self.scale = float(scale)
        self.jpeg_quality = jpeg_quality

    def open_document(self, content: bytes) -> PDFPageRenderSession:
        """Open a reusable in-memory PDFium document session."""
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise _MissingPDFOCRDependencyError(
                "vector-only PDF OCR requires the optional dependency; "
                "install gabby-agent-runtime[pdf-ocr]"
            ) from exc
        if not isinstance(content, bytes):
            raise TypeError("PDF content must be bytes")
        return _PDFiumPageRenderSession(pdfium.PdfDocument(content), self.scale, self.jpeg_quality)


class _PDFiumPageRenderSession:
    def __init__(self, document: Any, scale: float, jpeg_quality: int) -> None:
        self.document = document
        self.scale = scale
        self.jpeg_quality = jpeg_quality
        self.closed = False

    def render_page(
        self, page_number: int, *, max_image_pixels: int, max_image_bytes: int
    ) -> PDFRenderedPage:
        """Rasterize one page and close its native page and bitmap handles."""
        for name, value in (
            ("page_number", page_number),
            ("max_image_pixels", max_image_pixels),
            ("max_image_bytes", max_image_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.closed:
            raise ValueError("PDF renderer session is closed")
        page = None
        bitmap = None
        rendered = None
        image = None
        try:
            if page_number > len(self.document):
                raise ValueError("PDF page number is out of range")
            page = self.document[page_number - 1]
            width, height = page.get_size()
            if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                raise ValueError("PDF page dimensions are invalid")
            page_scale = min(self.scale, math.sqrt(max_image_pixels / (width * height)) * 0.999)
            if not math.isfinite(page_scale) or page_scale <= 0:
                raise ValueError("PDF page cannot be rendered within the configured pixel limit")
            bitmap = page.render(scale=page_scale, fill_color=(255, 255, 255, 255))
            pixel_width, pixel_height = bitmap.width, bitmap.height
            pixel_count = pixel_width * pixel_height
            if pixel_count < 1 or pixel_count > max_image_pixels:
                raise ValueError("PDF render exceeded max_image_pixels")
            image = bitmap.to_pil().convert("RGB")
            rendered = _BoundedBytesIO(max_image_bytes)
            image.save(rendered, format="JPEG", quality=self.jpeg_quality, optimize=True)
            data = rendered.getvalue()
            return PDFRenderedPage(data, pixel_count)
        finally:
            with suppress(Exception):
                if image is not None:
                    image.close()
            with suppress(Exception):
                if rendered is not None:
                    rendered.close()
            with suppress(Exception):
                if bitmap is not None:
                    bitmap.close()
            with suppress(Exception):
                if page is not None:
                    page.close()

    def close(self) -> None:
        """Close the underlying PDFium document exactly once."""
        if not self.closed:
            self.closed = True
            self.document.close()


class _BoundedBytesIO:
    """In-memory output stream that enforces a byte cap during image encoding."""

    def __init__(self, max_bytes: int) -> None:
        self._buffer = BytesIO()
        self.max_bytes = max_bytes

    def write(self, data: bytes) -> int:
        if max(self.tell() + len(data), self._buffer.getbuffer().nbytes) > self.max_bytes:
            raise _PDFInputError(f"PDF render exceeds max_ocr_image_bytes={self.max_bytes}")
        return self._buffer.write(data)

    def tell(self) -> int:
        return self._buffer.tell()

    def getvalue(self) -> bytes:
        return self._buffer.getvalue()

    def close(self) -> None:
        self._buffer.close()


class TesseractOCRBackend:
    """Run Tesseract against bounded images using the optional ``ocr`` extra."""

    def __init__(self, *, language: str = "eng") -> None:
        if not isinstance(language, str) or not re.fullmatch(r"[A-Za-z0-9_+.\-]{1,128}", language):
            raise ValueError("language must be a valid Tesseract language code list")
        self.language = language

    def recognize(
        self,
        content: bytes,
        *,
        max_pages: int,
        max_image_pixels: int,
        timeout_seconds: float,
    ) -> Sequence[OCRPageResult]:
        """OCR each frame without extracting or writing image data to disk."""
        try:
            import pytesseract
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise RuntimeError(
                "image OCR requires the optional dependency; install gabby-agent-runtime[ocr]"
            ) from exc

        try:
            with Image.open(BytesIO(content)) as image:
                frame_count = getattr(image, "n_frames", 1)
                if frame_count > max_pages:
                    raise ValueError(f"image exceeds max_pages={max_pages}")
                results: list[OCRPageResult] = []
                total_pixels = 0
                for index in range(frame_count):
                    image.seek(index)
                    width, height = image.size
                    pixels = width * height
                    total_pixels += pixels
                    if total_pixels > max_image_pixels:
                        raise ValueError(f"image exceeds max_image_pixels={max_image_pixels}")
                    image.load()
                    rgb_image = image.convert("RGB")
                    try:
                        results.append(
                            OCRPageResult(
                                text=pytesseract.image_to_string(
                                    rgb_image,
                                    lang=self.language,
                                    timeout=timeout_seconds,
                                ),
                                pixel_count=pixels,
                            )
                        )
                    finally:
                        rgb_image.close()
                return results
        except ValueError:
            raise
        except (UnidentifiedImageError, OSError, EOFError):
            raise ValueError("could not decode image for OCR") from None
        except Exception as exc:
            if type(exc).__name__ == "TesseractNotFoundError":
                raise RuntimeError("Tesseract executable is not installed or not on PATH") from None
            if type(exc).__name__ == "TimeoutExpired":
                raise TimeoutError("Tesseract OCR exceeded timeout_seconds") from None
            raise ValueError("Tesseract OCR failed") from None


class ImageOCRParser:
    """Extract bounded, page-attributed text from raster images using OCR."""

    extensions = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})

    def __init__(
        self,
        *,
        backend: OCRBackend | None = None,
        max_pages: int = 100,
        max_image_pixels: int = 40_000_000,
        max_extracted_chars: int = 10_000_000,
        timeout_seconds: float = 30.0,
    ) -> None:
        for name, value in (
            ("max_pages", max_pages),
            ("max_image_pixels", max_image_pixels),
            ("max_extracted_chars", max_extracted_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise TypeError("timeout_seconds must be a positive number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive number")
        self.backend = backend or TesseractOCRBackend()
        self.max_pages = max_pages
        self.max_image_pixels = max_image_pixels
        self.max_extracted_chars = max_extracted_chars
        self.timeout_seconds = float(timeout_seconds)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """OCR each bounded image frame and preserve one-based page attribution."""
        texts = self.backend.recognize(
            content,
            max_pages=self.max_pages,
            max_image_pixels=self.max_image_pixels,
            timeout_seconds=self.timeout_seconds,
        )
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise ValueError("OCR backend must return a sequence of text pages")
        if not texts or len(texts) > self.max_pages:
            raise ValueError(f"OCR result must contain between 1 and {self.max_pages} pages")
        pages: list[ParsedPage] = []
        extracted_chars = 0
        decoded_pixels = 0
        for index, result in enumerate(texts, start=1):
            if (
                not isinstance(result, OCRPageResult)
                or not isinstance(result.text, str)
                or "\x00" in result.text
            ):
                raise ValueError("OCR pages must contain valid text without NUL characters")
            if (
                isinstance(result.pixel_count, bool)
                or not isinstance(result.pixel_count, int)
                or result.pixel_count < 1
            ):
                raise ValueError("OCR pages must report a positive pixel_count")
            decoded_pixels += result.pixel_count
            if decoded_pixels > self.max_image_pixels:
                raise ValueError(f"OCR exceeds max_image_pixels={self.max_image_pixels}")
            extracted_chars += len(result.text)
            if extracted_chars > self.max_extracted_chars:
                raise ValueError(f"OCR exceeds max_extracted_chars={self.max_extracted_chars}")
            pages.append(ParsedPage(result.text, page_number=index))
        return pages


_MAX_MARKDOWN_FRONT_MATTER_BYTES = 16 * 1024
_MAX_FRONT_MATTER_NODES = 10_000
_MAX_FRONT_MATTER_DEPTH = 32


class _UniqueSafeYamlLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueSafeYamlLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(
                None, None, "YAML merge keys are not supported", key_node.start_mark
            )
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError:
            raise yaml.constructor.ConstructorError(
                None, None, "YAML mapping keys must be scalar", key_node.start_mark
            ) from None
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate YAML mapping key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _normalize_structured_value(
    value: Any,
    *,
    ancestors: set[int],
    count: list[int],
    depth: int = 0,
    max_nodes: int = _MAX_FRONT_MATTER_NODES,
    max_depth: int = _MAX_FRONT_MATTER_DEPTH,
    source_label: str = "structured document",
) -> Any:
    """Normalize parsed structured data into a bounded JSON-compatible tree."""
    count[0] += 1
    if count[0] > max_nodes or depth > max_depth:
        raise ValueError(f"{source_label} exceeds structural limits")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{source_label} must contain finite numbers")
        return value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{source_label} must not contain cyclic references")
        ancestors.add(identity)
        try:
            if isinstance(value, list):
                return [
                    _normalize_structured_value(
                        item,
                        ancestors=ancestors,
                        count=count,
                        depth=depth + 1,
                        max_nodes=max_nodes,
                        max_depth=max_depth,
                        source_label=source_label,
                    )
                    for item in value
                ]
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{source_label} mapping keys must be strings")
                normalized[key] = _normalize_structured_value(
                    item,
                    ancestors=ancestors,
                    count=count,
                    depth=depth + 1,
                    max_nodes=max_nodes,
                    max_depth=max_depth,
                    source_label=source_label,
                )
            return normalized
        finally:
            ancestors.remove(identity)
    raise ValueError(f"{source_label} must contain only JSON-compatible values")


def _parse_markdown_front_matter(text: str) -> tuple[str, dict[str, Any]]:
    """Remove and validate an optional bounded YAML front matter block."""
    lines = text.splitlines(keepends=True)
    if not lines or not lines[0].endswith(("\n", "\r")) or lines[0].rstrip("\r\n").strip() != "---":
        return text, {}
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n").strip() in {"---", "..."}
        ),
        None,
    )
    if closing_index is None:
        raise ValueError("Markdown front matter is missing its closing delimiter")
    front_matter_text = "".join(lines[1:closing_index])
    if len(front_matter_text.encode("utf-8")) > _MAX_MARKDOWN_FRONT_MATTER_BYTES:
        raise ValueError(f"Markdown front matter exceeds {_MAX_MARKDOWN_FRONT_MATTER_BYTES} bytes")
    try:
        loaded = yaml.load(front_matter_text, Loader=_UniqueSafeYamlLoader)
    except yaml.YAMLError:
        raise ValueError("Markdown front matter is invalid YAML") from None
    if loaded is None:
        metadata: dict[str, Any] = {}
    elif not isinstance(loaded, dict):
        raise ValueError("Markdown front matter must be a mapping")
    else:
        normalized = _normalize_structured_value(
            loaded, ancestors=set(), count=[0], source_label="Markdown front matter"
        )
        assert isinstance(normalized, dict)
        metadata = normalized
    return "".join(lines[closing_index + 1 :]), metadata


class Utf8TextParser:
    """Decode UTF-8 Markdown, text, and logs with bounded leading YAML metadata."""

    extensions = frozenset({".md", ".markdown", ".txt", ".log"})

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Decode one UTF-8 source into a single page."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("text files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("text files must not contain NUL characters")
        body, metadata = _parse_markdown_front_matter(text)
        return (ParsedPage(body, metadata=metadata),)


class MarkupTextParser:
    """Index bounded reStructuredText and AsciiDoc source without executing directives."""

    extensions = frozenset({".adoc", ".asciidoc", ".rst"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Decode markup source as searchable text, preserving markup and line order."""
        if not isinstance(content, bytes):
            raise TypeError("markup content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"markup exceeds max_input_bytes={self.max_input_bytes}")
        try:
            text = content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            raise ValueError("markup files must contain valid UTF-8") from None
        if "\x00" in text:
            raise ValueError("markup files must not contain NUL characters")
        if len(text) > self.max_extracted_chars:
            raise ValueError(f"markup exceeds max_extracted_chars={self.max_extracted_chars}")
        return (ParsedPage(text),)


class RTFTextParser:
    """Extract bounded visible text from common Rich Text Format documents."""

    extensions = frozenset({".rtf"})
    _SKIPPED_DESTINATIONS = frozenset(
        {
            "annotation",
            "colortbl",
            "datastore",
            "filetbl",
            "fonttbl",
            "footer",
            "footerf",
            "footerl",
            "footerr",
            "header",
            "headerf",
            "headerl",
            "headerr",
            "info",
            "listoverridetable",
            "listtable",
            "object",
            "pict",
            "revtbl",
            "rsidtbl",
            "stylesheet",
            "themedata",
            "xmlnstbl",
        }
    )
    _SPECIAL_TEXT = {
        "emdash": "—",
        "endash": "–",
        "emspace": " ",
        "enspace": " ",
        "qmspace": " ",
        "bullet": "•",
        "lquote": "‘",
        "rquote": "’",
        "ldblquote": "“",
        "rdblquote": "”",
    }

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_group_depth: int = 128,
        max_control_words: int = 1_000_000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_group_depth", max_group_depth, 4096),
            ("max_control_words", max_control_words, 10_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_group_depth = max_group_depth
        self.max_control_words = max_control_words
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Parse RTF control syntax while discarding binary and non-body destinations."""
        if not isinstance(content, bytes):
            raise TypeError("RTF content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"RTF exceeds max_input_bytes={self.max_input_bytes}")
        if not content.lstrip().startswith(b"{\\rtf"):
            raise ValueError("RTF document has no valid header")

        output: list[str] = []
        output_chars = 0
        raw_text = bytearray()
        state: dict[str, Any] | None = None
        stack: list[dict[str, Any] | None] = []
        root_closed = False
        control_words = 0
        index = 0

        def append_text(text: str) -> None:
            nonlocal output_chars
            if not text or state is None or state["skip"] or state["hidden"]:
                return
            fallback = state["fallback"]
            if fallback:
                skipped = min(fallback, len(text))
                state["fallback"] = fallback - skipped
                text = text[skipped:]
            if not text:
                return
            if "\x00" in text:
                raise ValueError("RTF visible text must not contain NUL characters")
            output_chars += len(text)
            if output_chars > self.max_extracted_chars:
                raise ValueError(f"RTF exceeds max_extracted_chars={self.max_extracted_chars}")
            output.append(text)

        def flush_raw() -> None:
            if not raw_text:
                return
            assert state is not None
            try:
                text = bytes(raw_text).decode(state["codepage"], errors="strict")
            except (LookupError, UnicodeDecodeError):
                raise ValueError(
                    "RTF contains invalid text for its declared ANSI code page"
                ) from None
            raw_text.clear()
            append_text(text)

        def emit_control_text(text: str) -> None:
            append_text(text)

        try:
            while index < len(content):
                byte = content[index]
                if byte == 0x7B:  # {
                    flush_raw()
                    if root_closed:
                        raise ValueError("RTF contains data after its root group")
                    if len(stack) >= self.max_group_depth:
                        raise ValueError(f"RTF exceeds max_group_depth={self.max_group_depth}")
                    stack.append(state)
                    state = (
                        {
                            "skip": False,
                            "hidden": False,
                            "uc": 1,
                            "fallback": 0,
                            "codepage": "cp1252",
                            "at_start": True,
                            "ignorable": False,
                        }
                        if state is None
                        else {**state, "at_start": True, "ignorable": False}
                    )
                    index += 1
                    continue
                if byte == 0x7D:  # }
                    flush_raw()
                    if not stack:
                        raise ValueError("RTF contains an unmatched closing group")
                    state = stack.pop()
                    if state is None:
                        root_closed = True
                    index += 1
                    continue
                if state is None:
                    if byte not in b" \t\r\n":
                        raise ValueError("RTF contains data outside its root group")
                    index += 1
                    continue
                if byte != 0x5C:  # backslash
                    if byte in (0x0A, 0x0D):  # source line formatting is insignificant
                        index += 1
                        continue
                    raw_text.append(byte)
                    state["at_start"] = False
                    index += 1
                    continue

                if not (index + 1 < len(content) and content[index + 1] == 0x27):
                    flush_raw()
                index += 1
                if index >= len(content):
                    raise ValueError("RTF ends with an incomplete control sequence")
                next_byte = content[index]
                if (65 <= next_byte <= 90) or (97 <= next_byte <= 122):
                    start = index
                    while index < len(content) and (
                        65 <= content[index] <= 90 or 97 <= content[index] <= 122
                    ):
                        index += 1
                    if index - start > 64:
                        raise ValueError("RTF contains an oversized control word")
                    word = content[start:index].decode("ascii").lower()
                    number: int | None = None
                    negative = False
                    if index < len(content) and content[index] == 0x2D:
                        negative = True
                        index += 1
                    number_start = index
                    while index < len(content) and 48 <= content[index] <= 57:
                        index += 1
                    if index - number_start > 9:
                        raise ValueError("RTF contains an oversized control parameter")
                    if negative and index == number_start:
                        raise ValueError("RTF contains an invalid signed control parameter")
                    if index > number_start:
                        number = int(content[number_start:index])
                        if negative:
                            number = -number
                    if index < len(content) and content[index] == 0x20:
                        index += 1
                    control_words += 1
                    if control_words > self.max_control_words:
                        raise ValueError(f"RTF exceeds max_control_words={self.max_control_words}")
                    if state["at_start"]:
                        if word in self._SKIPPED_DESTINATIONS or state["ignorable"]:
                            state["skip"] = True
                        state["at_start"] = False
                        state["ignorable"] = False
                    if word == "uc" and number is not None:
                        if not 0 <= number <= 32:
                            raise ValueError("RTF Unicode fallback count must be from 0 through 32")
                        state["uc"] = number
                    elif word == "u" and number is not None:
                        if not -32768 <= number <= 32767:
                            raise ValueError(
                                "RTF Unicode control value is outside signed 16-bit range"
                            )
                        emit_control_text(chr(number & 0xFFFF))
                        state["fallback"] = state["uc"]
                    elif word == "ansicpg" and number is not None:
                        codepage = f"cp{number}"
                        try:
                            codecs.lookup(codepage)
                        except LookupError:
                            raise ValueError("RTF declares an unsupported ANSI code page") from None
                        state["codepage"] = codepage
                    elif word == "bin" and number is not None:
                        if number < 0 or index + number > len(content):
                            raise ValueError("RTF contains an invalid binary payload length")
                        index += number
                    elif word in {"par", "line", "page", "row"}:
                        emit_control_text("\n")
                    elif word in {"tab", "cell"}:
                        emit_control_text("\t")
                    elif word == "v":
                        state["hidden"] = number != 0
                    elif word in self._SPECIAL_TEXT:
                        emit_control_text(self._SPECIAL_TEXT[word])
                    continue

                index += 1
                if next_byte == 0x27:  # \'x hex byte
                    if index + 2 > len(content):
                        raise ValueError("RTF contains an incomplete hexadecimal character")
                    digits = content[index : index + 2]
                    try:
                        raw_text.append(int(digits.decode("ascii"), 16))
                    except (UnicodeDecodeError, ValueError):
                        raise ValueError("RTF contains an invalid hexadecimal character") from None
                    index += 2
                    state["at_start"] = False
                elif next_byte in (0x5C, 0x7B, 0x7D):  # \\, \{, \}
                    raw_text.append(next_byte)
                    state["at_start"] = False
                elif next_byte == 0x2A:  # \* marks an ignorable destination
                    state["ignorable"] = True
                elif next_byte == 0x7E:  # \~
                    append_text("\u00a0")
                elif next_byte == 0x5F:  # \_
                    append_text("\u2011")
                elif next_byte in (0x2D, 0x0A, 0x0D):  # optional hyphen or source newline
                    pass
                else:
                    state["at_start"] = False

            flush_raw()
        except ValueError as exc:
            message = str(exc)
            if message.startswith("RTF "):
                raise
            raise ValueError("RTF document is malformed or contains unsupported text") from None

        if stack or not root_closed:
            raise ValueError("RTF document has unbalanced groups")
        try:
            text = "".join(output).encode("utf-16-le", errors="surrogatepass").decode("utf-16-le")
        except UnicodeDecodeError:
            raise ValueError("RTF contains an unpaired Unicode surrogate") from None
        return (ParsedPage(text),)


class EmailTextParser:
    """Extract selected headers and bounded text bodies from RFC 5322 ``.eml`` files."""

    extensions = frozenset({".eml"})
    _HEADER_NAMES = ("From", "To", "Cc", "Date", "Subject", "Message-ID")

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_parts: int = 1000,
        max_depth: int = 32,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_parts", max_parts, 100_000),
            ("max_depth", max_depth, 256),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_parts = max_parts
        self.max_depth = max_depth
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Extract headers and text/plain bodies, falling back to visible HTML text."""
        if not isinstance(content, bytes):
            raise TypeError("email content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"email exceeds max_input_bytes={self.max_input_bytes}")
        try:
            message = BytesParser(policy=default_email_policy).parsebytes(content)
            return (ParsedPage(self._extract(message)),)
        except _EmailInputError as exc:
            raise ValueError(str(exc)) from None
        except Exception:
            raise ValueError("could not parse email message") from None

    def _extract(self, message: Message) -> str:
        fragments: list[str] = []
        extracted_chars = 0

        def append(value: str) -> None:
            nonlocal extracted_chars
            if not value:
                return
            if "\x00" in value:
                raise _EmailInputError("email text must not contain NUL characters")
            separator_size = 2 if fragments else 0
            extracted_chars += separator_size + len(value)
            if extracted_chars > self.max_extracted_chars:
                raise _EmailInputError(
                    f"email exceeds max_extracted_chars={self.max_extracted_chars}"
                )
            fragments.append(value)

        for name in self._HEADER_NAMES:
            values = message.get_all(name, [])
            for value in values:
                normalized = " ".join(str(value).replace("\x00", "").splitlines()).strip()
                if normalized:
                    append(f"{name}: {normalized}")

        plain_bodies: list[str] = []
        html_bodies: list[str] = []
        stack: list[tuple[Message, int]] = [(message, 0)]
        part_count = 0
        while stack:
            part, depth = stack.pop()
            part_count += 1
            if part_count > self.max_parts:
                raise _EmailInputError(f"email exceeds max_parts={self.max_parts}")
            if depth > self.max_depth:
                raise _EmailInputError(f"email exceeds max_depth={self.max_depth}")
            if part.get_content_disposition() == "attachment" or part.get_filename() is not None:
                continue
            payload = part.get_payload()
            if part.is_multipart():
                if not isinstance(payload, list) or any(
                    not isinstance(child, Message) for child in payload
                ):
                    raise _EmailInputError("email contains a malformed multipart body")
                stack.extend((child, depth + 1) for child in reversed(payload))
                continue
            content_type = part.get_content_type().casefold()
            if content_type not in ("text/plain", "text/html"):
                continue
            body = self._decode_text_part(part, payload)
            if content_type == "text/html":
                body = HTMLTextParser().parse(body.encode("utf-8"))[0].text
                html_bodies.append(body)
            else:
                plain_bodies.append(body)

        # MIME alternatives commonly carry the same content as both plain text and HTML.
        selected_bodies = plain_bodies or html_bodies
        for body in selected_bodies:
            append(body)
        return "\n\n".join(fragments)

    @staticmethod
    def _decode_text_part(part: Message, payload: Any) -> str:
        decoded = part.get_payload(decode=True)
        if decoded is None:
            if isinstance(payload, str):
                return payload
            raise _EmailInputError("email text part has an invalid transfer encoding")
        if not isinstance(decoded, bytes):
            raise _EmailInputError("email text part has an invalid transfer encoding")
        charset = part.get_content_charset() or "us-ascii"
        try:
            return decoded.decode(charset, errors="replace")
        except LookupError:
            return decoded.decode("utf-8", errors="replace")


class MboxTextParser:
    """Extract bounded separator-framed mbox messages using the email MIME policy."""

    extensions = frozenset({".mbox"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_messages: int = 1000,
        max_lines: int = 100_000,
        max_extracted_chars: int = 10_000_000,
        max_parts_per_message: int = 1000,
        max_mime_depth: int = 32,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_messages", max_messages, 100_000),
            ("max_lines", max_lines, 1_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
            ("max_parts_per_message", max_parts_per_message, 100_000),
            ("max_mime_depth", max_mime_depth, 256),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_messages = max_messages
        self.max_lines = max_lines
        self.max_extracted_chars = max_extracted_chars
        self.email_parser = EmailTextParser(
            max_input_bytes=max_input_bytes,
            max_parts=max_parts_per_message,
            max_depth=max_mime_depth,
            max_extracted_chars=max_extracted_chars,
        )

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Return one cited page per envelope separator; attachments remain excluded."""
        if not isinstance(content, bytes):
            raise TypeError("mbox content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"mbox exceeds max_input_bytes={self.max_input_bytes}")
        if b"\x00" in content:
            raise ValueError("mbox must not contain NUL characters")

        pages: list[ParsedPage] = []
        current_lines: list[bytes] | None = None
        total_chars = 0

        def finish_message() -> None:
            nonlocal current_lines, total_chars
            if current_lines is None:
                return
            if len(pages) >= self.max_messages:
                raise ValueError(f"mbox exceeds max_messages={self.max_messages}")
            message = b"\n".join(current_lines)
            if not message.strip():
                raise ValueError("mbox contains an empty message")
            header_bytes = message.partition(b"\n\n")[0]
            if any(
                header.lower().startswith(b"content-length:")
                for header in header_bytes.splitlines()
            ):
                raise ValueError("mbox Content-Length framing is not supported")
            try:
                parsed = self.email_parser.parse(message)
            except ValueError:
                raise ValueError("mbox contains an invalid email message") from None
            if len(parsed) != 1:
                raise ValueError("mbox message parser returned an invalid page count")
            total_chars += len(parsed[0].text)
            if total_chars > self.max_extracted_chars:
                raise ValueError(f"mbox exceeds max_extracted_chars={self.max_extracted_chars}")
            page_index = len(pages) + 1
            metadata = dict(parsed[0].metadata)
            metadata["mailbox_message_index"] = page_index
            pages.append(ParsedPage(parsed[0].text, page_number=page_index, metadata=metadata))
            current_lines = None

        for line_count, raw_line in enumerate(BytesIO(content), start=1):
            if line_count > self.max_lines:
                raise ValueError(f"mbox exceeds max_lines={self.max_lines}")
            line = raw_line.removesuffix(b"\n").removesuffix(b"\r")
            if b"\r" in line:
                raise ValueError("mbox contains an invalid line ending")
            if line.startswith(b"From "):
                if len(line) > 4096:
                    raise ValueError("mbox envelope line exceeds 4096 bytes")
                finish_message()
                current_lines = []
            elif current_lines is None:
                if line.strip():
                    raise ValueError("mbox content appears before the first From separator")
            else:
                current_lines.append(line)
        finish_message()
        if not pages:
            raise ValueError("mbox must contain at least one From separator and message")
        return tuple(pages)


class ICalendarTextParser:
    """Extract bounded VEVENT, VTODO, and VJOURNAL records from iCalendar files."""

    extensions = frozenset({".ics"})
    _TEXT_PROPERTIES = frozenset({"SUMMARY", "DESCRIPTION", "LOCATION", "UID", "STATUS"})
    _DISPLAY_PROPERTIES = (
        ("SUMMARY", "Summary"),
        ("DTSTART", "Start"),
        ("DTEND", "End"),
        ("LOCATION", "Location"),
        ("DESCRIPTION", "Description"),
        ("ORGANIZER", "Organizer"),
        ("ATTENDEE", "Attendee"),
        ("STATUS", "Status"),
    )
    _TASK_DISPLAY_PROPERTIES = (
        ("SUMMARY", "Summary"),
        ("DTSTART", "Start"),
        ("DUE", "Due"),
        ("COMPLETED", "Completed"),
        ("DESCRIPTION", "Description"),
        ("STATUS", "Status"),
        ("PRIORITY", "Priority"),
        ("LOCATION", "Location"),
        ("ORGANIZER", "Organizer"),
        ("ATTENDEE", "Attendee"),
    )
    _JOURNAL_DISPLAY_PROPERTIES = (
        ("SUMMARY", "Summary"),
        ("DTSTART", "Start"),
        ("DESCRIPTION", "Description"),
        ("STATUS", "Status"),
        ("ORGANIZER", "Organizer"),
        ("ATTENDEE", "Attendee"),
    )
    _INDEXED_PROPERTIES = frozenset(
        {
            "SUMMARY",
            "DTSTART",
            "DTEND",
            "LOCATION",
            "DESCRIPTION",
            "ORGANIZER",
            "ATTENDEE",
            "STATUS",
            "UID",
            "DUE",
            "COMPLETED",
            "PRIORITY",
        }
    )

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_events: int = 1000,
        max_tasks: int = 1000,
        max_journals: int = 1000,
        max_attendees: int = 1000,
        max_lines: int = 100_000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_events", max_events, 100_000),
            ("max_tasks", max_tasks, 100_000),
            ("max_journals", max_journals, 100_000),
            ("max_attendees", max_attendees, 10_000),
            ("max_lines", max_lines, 1_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_events = max_events
        self.max_tasks = max_tasks
        self.max_journals = max_journals
        self.max_attendees = max_attendees
        self.max_lines = max_lines
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Return source-ordered event, task, and journal pages; alarms are not indexed."""
        if not isinstance(content, bytes):
            raise TypeError("iCalendar content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"iCalendar exceeds max_input_bytes={self.max_input_bytes}")
        if b"\x00" in content:
            raise ValueError("iCalendar must not contain NUL characters")
        try:
            source = content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise ValueError("iCalendar must be valid UTF-8") from None
        if "\r" in source.replace("\r\n", ""):
            raise ValueError("iCalendar contains an invalid line ending")
        physical_lines = source.replace("\r\n", "\n").split("\n")
        if len(physical_lines) > self.max_lines:
            raise ValueError(f"iCalendar exceeds max_lines={self.max_lines}")
        lines: list[str] = []
        for line in physical_lines:
            if line.startswith((" ", "\t")):
                if not lines:
                    raise ValueError("iCalendar starts with a folded line")
                lines[-1] += line[1:]
            else:
                lines.append(line)

        components: list[str] = []
        records: list[tuple[str, dict[str, list[str]]]] = []
        current: dict[str, list[str]] | None = None
        record_depth: int | None = None
        event_count = 0
        task_count = 0
        journal_count = 0
        saw_calendar = False
        for line in lines:
            if not line:
                continue
            property_name, raw_value = self._split_content_line(line)
            if not components and property_name != "BEGIN":
                raise ValueError("iCalendar content appears outside its VCALENDAR root")
            if property_name == "BEGIN":
                component = raw_value.strip().upper()
                if not component or len(components) >= 32:
                    raise ValueError("iCalendar has an invalid component nesting depth")
                if not components:
                    if component != "VCALENDAR" or saw_calendar:
                        raise ValueError("iCalendar must have one VCALENDAR root")
                    saw_calendar = True
                elif component == "VCALENDAR":
                    raise ValueError("iCalendar has a nested VCALENDAR")
                elif component in {"VEVENT", "VTODO", "VJOURNAL"} and components != ["VCALENDAR"]:
                    raise ValueError(f"iCalendar {component} must be a direct child of VCALENDAR")
                components.append(component)
                if component in {"VEVENT", "VTODO", "VJOURNAL"}:
                    if current is not None or components.count(component) != 1:
                        raise ValueError(f"iCalendar has an invalid nested {component}")
                    current = {}
                    record_depth = len(components)
                continue
            if property_name == "END":
                component = raw_value.strip().upper()
                if not components or components[-1] != component:
                    raise ValueError("iCalendar component boundaries are invalid")
                if component in {"VEVENT", "VTODO", "VJOURNAL"}:
                    if current is None:
                        raise ValueError(f"iCalendar has an invalid {component}")
                    records.append((component, current))
                    if component == "VEVENT":
                        event_count += 1
                        if event_count > self.max_events:
                            raise ValueError(f"iCalendar exceeds max_events={self.max_events}")
                    elif component == "VTODO":
                        task_count += 1
                        if task_count > self.max_tasks:
                            raise ValueError(f"iCalendar exceeds max_tasks={self.max_tasks}")
                    else:
                        journal_count += 1
                        if journal_count > self.max_journals:
                            raise ValueError(f"iCalendar exceeds max_journals={self.max_journals}")
                    current = None
                    record_depth = None
                components.pop()
                continue
            if current is None or record_depth != len(components):
                continue
            if property_name not in self._INDEXED_PROPERTIES:
                continue
            if (
                property_name == "ATTENDEE"
                and len(current.get("ATTENDEE", ())) >= self.max_attendees
            ):
                raise ValueError(f"iCalendar exceeds max_attendees={self.max_attendees}")
            value = (
                self._unescape_text(raw_value)
                if property_name in self._TEXT_PROPERTIES
                else raw_value
            )
            current.setdefault(property_name, []).append(value)

        if components or current is not None or not saw_calendar:
            raise ValueError("iCalendar component boundaries are incomplete")
        pages: list[ParsedPage] = []
        extracted_chars = 0
        event_index = 0
        task_index = 0
        journal_index = 0
        for page_number, (component, record) in enumerate(records, start=1):
            if component == "VEVENT":
                event_index += 1
                index = event_index
                text = self._render_event(record)
            elif component == "VTODO":
                task_index += 1
                index = task_index
                text = self._render_task(record)
            else:
                journal_index += 1
                index = journal_index
                text = self._render_journal(record)
            extracted_chars += len(text)
            if extracted_chars > self.max_extracted_chars:
                raise ValueError(
                    f"iCalendar exceeds max_extracted_chars={self.max_extracted_chars}"
                )
            metadata: dict[str, Any]
            if component == "VEVENT":
                metadata = {"calendar_event_index": index}
            elif component == "VTODO":
                metadata = {"calendar_task_index": index}
            else:
                metadata = {"calendar_journal_index": index}
            property_metadata: tuple[tuple[str, str], ...]
            if component == "VEVENT":
                property_metadata = (
                    ("UID", "calendar_uid"),
                    ("DTSTART", "calendar_start"),
                    ("DTEND", "calendar_end"),
                    ("LOCATION", "calendar_location"),
                    ("ORGANIZER", "calendar_organizer"),
                    ("STATUS", "calendar_status"),
                )
            elif component == "VTODO":
                property_metadata = (
                    ("UID", "calendar_task_uid"),
                    ("DTSTART", "calendar_task_start"),
                    ("DUE", "calendar_task_due"),
                    ("COMPLETED", "calendar_task_completed"),
                    ("STATUS", "calendar_task_status"),
                    ("PRIORITY", "calendar_task_priority"),
                )
            else:
                property_metadata = (
                    ("UID", "calendar_journal_uid"),
                    ("DTSTART", "calendar_journal_start"),
                    ("STATUS", "calendar_journal_status"),
                )
            for source_name, metadata_name in property_metadata:
                values = record.get(source_name)
                if values:
                    metadata[metadata_name] = values[0]
            attendees = record.get("ATTENDEE")
            if attendees:
                attendee_key = {
                    "VEVENT": "calendar_attendees",
                    "VTODO": "calendar_task_attendees",
                    "VJOURNAL": "calendar_journal_attendees",
                }[component]
                metadata[attendee_key] = attendees
            pages.append(ParsedPage(text, page_number=page_number, metadata=metadata))
        return tuple(pages)

    @staticmethod
    def _split_content_line(line: str) -> tuple[str, str]:
        """Split a content line at its first colon outside quoted parameters."""
        quoted = False
        escaped = False
        delimiter = -1
        for index, char in enumerate(line):
            if char == '"' and not escaped:
                quoted = not quoted
            elif char == ":" and not quoted:
                delimiter = index
                break
            escaped = char == "\\" and not escaped
        if delimiter < 1:
            raise ValueError("iCalendar contains a malformed content line")
        head = line[:delimiter].split(";", 1)[0]
        property_name = head.rsplit(".", 1)[-1].upper()
        if not property_name or any(not (char.isalnum() or char == "-") for char in property_name):
            raise ValueError("iCalendar contains an invalid property name")
        return property_name, line[delimiter + 1 :]

    @staticmethod
    def _unescape_text(value: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                escaped = value[index + 1]
                if escaped in ("n", "N"):
                    output.append("\n")
                elif escaped in ("\\", ";", ","):
                    output.append(escaped)
                else:
                    output.extend((char, escaped))
                index += 2
            else:
                output.append(char)
                index += 1
        return "".join(output)

    @classmethod
    def _render_event(cls, event: dict[str, list[str]]) -> str:
        fragments: list[str] = []
        for property_name, label in cls._DISPLAY_PROPERTIES:
            values = event.get(property_name, ())
            for value in values:
                if value.strip():
                    fragments.append(f"{label}: {value.strip()}")
        return "\n".join(fragments)

    @classmethod
    def _render_task(cls, task: dict[str, list[str]]) -> str:
        fragments: list[str] = []
        for property_name, label in cls._TASK_DISPLAY_PROPERTIES:
            values = task.get(property_name, ())
            for value in values:
                if value.strip():
                    fragments.append(f"{label}: {value.strip()}")
        return "\n".join(fragments)

    @classmethod
    def _render_journal(cls, journal: dict[str, list[str]]) -> str:
        fragments: list[str] = []
        for property_name, label in cls._JOURNAL_DISPLAY_PROPERTIES:
            values = journal.get(property_name, ())
            for value in values:
                if value.strip():
                    fragments.append(f"{label}: {value.strip()}")
        return "\n".join(fragments)


class VCardTextParser:
    """Extract bounded vCard contact records into cited, filterable pages."""

    extensions = frozenset({".vcf"})
    _INDEXED_PROPERTIES = frozenset(
        {
            "VERSION",
            "FN",
            "N",
            "NICKNAME",
            "ORG",
            "TITLE",
            "EMAIL",
            "TEL",
            "ADR",
            "URL",
            "NOTE",
            "UID",
            "KIND",
            "BDAY",
        }
    )
    _DISPLAY_PROPERTIES = (
        ("FN", "Name"),
        ("N", "Name components"),
        ("NICKNAME", "Nickname"),
        ("ORG", "Organization"),
        ("TITLE", "Title"),
        ("EMAIL", "Email"),
        ("TEL", "Telephone"),
        ("ADR", "Address"),
        ("URL", "URL"),
        ("NOTE", "Note"),
    )

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_cards: int = 1000,
        max_lines: int = 100_000,
        max_properties_per_card: int = 1000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_input_bytes", max_input_bytes, 256 * 1024 * 1024),
            ("max_cards", max_cards, 100_000),
            ("max_lines", max_lines, 1_000_000),
            ("max_properties_per_card", max_properties_per_card, 100_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        self.max_input_bytes = max_input_bytes
        self.max_cards = max_cards
        self.max_lines = max_lines
        self.max_properties_per_card = max_properties_per_card
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Return one source-ordered page per vCard without fetching referenced URLs."""
        if not isinstance(content, bytes):
            raise TypeError("vCard content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"vCard exceeds max_input_bytes={self.max_input_bytes}")
        if b"\x00" in content:
            raise ValueError("vCard must not contain NUL characters")
        try:
            source = content.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError:
            raise ValueError("vCard must be valid UTF-8") from None
        if "\r" in source.replace("\r\n", ""):
            raise ValueError("vCard contains an invalid line ending")
        physical_lines = source.replace("\r\n", "\n").split("\n")
        if len(physical_lines) > self.max_lines:
            raise ValueError(f"vCard exceeds max_lines={self.max_lines}")
        lines: list[str] = []
        for line in physical_lines:
            if line.startswith((" ", "\t")):
                if not lines:
                    raise ValueError("vCard starts with a folded line")
                lines[-1] += line[1:]
            else:
                lines.append(line)

        cards: list[dict[str, list[str]]] = []
        current: dict[str, list[str]] | None = None
        property_count = 0
        for line in lines:
            if not line:
                continue
            property_name, value = self._split_content_line(line)
            if property_name == "BEGIN":
                if value.strip().upper() != "VCARD" or current is not None:
                    raise ValueError("vCard has an invalid or nested BEGIN component")
                current = {}
                property_count = 0
                continue
            if property_name == "END":
                if value.strip().upper() != "VCARD" or current is None:
                    raise ValueError("vCard component boundaries are invalid")
                versions = current.get("VERSION", ())
                if len(versions) != 1 or versions[0].strip() not in {"2.1", "3.0", "4.0"}:
                    raise ValueError("vCard requires one VERSION of 2.1, 3.0, or 4.0")
                if not current.get("FN") and not current.get("N"):
                    raise ValueError("vCard requires a formatted or structured name")
                cards.append(current)
                if len(cards) > self.max_cards:
                    raise ValueError(f"vCard exceeds max_cards={self.max_cards}")
                current = None
                continue
            if current is None:
                raise ValueError("vCard property appears outside a VCARD component")
            if ";encoding=quoted-printable" in line.casefold():
                raise ValueError("vCard quoted-printable properties are not supported")
            property_count += 1
            if property_count > self.max_properties_per_card:
                raise ValueError(
                    f"vCard exceeds max_properties_per_card={self.max_properties_per_card}"
                )
            if property_name in self._INDEXED_PROPERTIES:
                current.setdefault(property_name, []).append(self._unescape_text(value))
        if current is not None or not cards:
            raise ValueError("vCard component boundaries are incomplete")

        pages: list[ParsedPage] = []
        extracted_chars = 0
        for index, card in enumerate(cards, start=1):
            text = self._render_card(card)
            extracted_chars += len(text)
            if extracted_chars > self.max_extracted_chars:
                raise ValueError(f"vCard exceeds max_extracted_chars={self.max_extracted_chars}")
            metadata: dict[str, Any] = {"vcard_index": index}
            for property_name, metadata_name in (
                ("UID", "vcard_uid"),
                ("VERSION", "vcard_version"),
                ("FN", "vcard_formatted_name"),
                ("KIND", "vcard_kind"),
                ("ORG", "vcard_organization"),
                ("TITLE", "vcard_title"),
                ("BDAY", "vcard_birthday"),
            ):
                values = card.get(property_name)
                if values:
                    metadata[metadata_name] = values[0]
            for property_name, metadata_name in (
                ("EMAIL", "vcard_emails"),
                ("TEL", "vcard_telephones"),
                ("URL", "vcard_urls"),
            ):
                values = card.get(property_name)
                if values:
                    metadata[metadata_name] = values
            pages.append(ParsedPage(text, page_number=index, metadata=metadata))
        return tuple(pages)

    @staticmethod
    def _split_content_line(line: str) -> tuple[str, str]:
        """Split at the first colon outside quoted parameters and normalize the property name."""
        quoted = False
        escaped = False
        delimiter = -1
        for index, char in enumerate(line):
            if char == '"' and not escaped:
                quoted = not quoted
            elif char == ":" and not quoted:
                delimiter = index
                break
            escaped = char == "\\" and not escaped
        if delimiter < 1:
            raise ValueError("vCard contains a malformed content line")
        head = line[:delimiter].split(";", 1)[0]
        property_name = head.rsplit(".", 1)[-1].upper()
        if not property_name or any(not (char.isalnum() or char == "-") for char in property_name):
            raise ValueError("vCard contains an invalid property name")
        return property_name, line[delimiter + 1 :]

    @staticmethod
    def _unescape_text(value: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                escaped = value[index + 1]
                if escaped in ("n", "N"):
                    output.append("\n")
                elif escaped in ("\\", ";", ","):
                    output.append(escaped)
                else:
                    output.extend((char, escaped))
                index += 2
            else:
                output.append(char)
                index += 1
        return "".join(output)

    @classmethod
    def _render_card(cls, card: dict[str, list[str]]) -> str:
        fragments: list[str] = []
        for property_name, label in cls._DISPLAY_PROPERTIES:
            for value in card.get(property_name, ()):
                if value.strip():
                    fragments.append(f"{label}: {value.strip()}")
        return "\n".join(fragments)


class JSONTextParser:
    """Validate JSON documents and extract JSON Feed items when the version marker matches."""

    extensions = frozenset({".json"})
    _FEED_VERSIONS = frozenset(
        {"https://jsonfeed.org/version/1", "https://jsonfeed.org/version/1.1"}
    )

    def __init__(
        self,
        *,
        max_depth: int = 64,
        max_tokens: int = 200_000,
        max_feed_items: int = 10_000,
        max_output_bytes: int = 12 * 1024 * 1024,
    ) -> None:
        for name, value in (
            ("max_depth", max_depth),
            ("max_tokens", max_tokens),
            ("max_feed_items", max_feed_items),
            ("max_output_bytes", max_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.max_feed_items = max_feed_items
        self.max_output_bytes = max_output_bytes

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Validate one strict JSON value with duplicate-key and complexity limits."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("JSON files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("JSON files must not contain NUL characters")
        self._check_complexity(text)

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("JSON objects must not contain duplicate keys")
                result[key] = value
            return result

        try:
            value = json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_constant=self._reject_constant,
                # Preserve the syntax of arbitrary precision numbers while validating structure.
                parse_int=str,
                parse_float=str,
            )
        except (ValueError, RecursionError):
            raise ValueError("JSON files must contain one valid JSON value") from None
        if isinstance(value, dict) and value.get("version") in self._FEED_VERSIONS:
            return self._parse_feed(value)
        return (ParsedPage(text),)

    def _parse_feed(self, feed: Mapping[str, Any]) -> Sequence[ParsedPage]:
        """Turn recognized JSON Feed items into bounded, independently cited pages."""
        items = feed.get("items")
        if not isinstance(items, list):
            raise ValueError("JSON Feed items must be an array")
        if len(items) > self.max_feed_items:
            raise ValueError(f"JSON Feed exceeds max_feed_items={self.max_feed_items}")
        feed_title = feed.get("title")
        if feed_title is not None and not isinstance(feed_title, str):
            raise ValueError("JSON Feed title must be a string")
        pages: list[ParsedPage] = []
        output_bytes = 0
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"JSON Feed item {index} must be an object")
            identifier = item.get("id")
            if isinstance(identifier, (str, int)) and not isinstance(identifier, bool):
                identifier = str(identifier)
            else:
                raise ValueError(f"JSON Feed item {index} must have a string or numeric id")
            title = item.get("title", "")
            content_text = item.get("content_text")
            content_html = item.get("content_html")
            summary = item.get("summary", "")
            if not isinstance(title, str) or not isinstance(summary, str):
                raise ValueError(f"JSON Feed item {index} title and summary must be strings")
            if content_text is not None and not isinstance(content_text, str):
                raise ValueError(f"JSON Feed item {index} content_text must be a string")
            if content_html is not None and not isinstance(content_html, str):
                raise ValueError(f"JSON Feed item {index} content_html must be a string")
            if content_text is None and content_html is None and not summary:
                raise ValueError(f"JSON Feed item {index} must contain text, HTML, or summary")
            body = content_text
            if body is None and content_html is not None:
                body = HTMLTextParser().parse(content_html.encode("utf-8"))[0].text
            body = body or summary
            item_author = item.get("author")
            feed_author = feed.get("author")
            author_value = item_author if item_author is not None else feed_author
            author = author_value.get("name") if isinstance(author_value, dict) else None
            if author_value is not None and (
                not isinstance(author_value, dict) or not isinstance(author, str)
            ):
                raise ValueError(f"JSON Feed item {index} author must have a string name")
            tags = item.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
                raise ValueError(f"JSON Feed item {index} tags must be an array of strings")
            published = item.get("date_published")
            if published is not None and not isinstance(published, str):
                raise ValueError(f"JSON Feed item {index} date_published must be a string")
            link = item.get("url")
            if link is not None and not isinstance(link, str):
                raise ValueError(f"JSON Feed item {index} url must be a string")
            safe_link = RSSAtomTextParser._safe_link(link) if link else None
            lines = [
                f"Title: {title}" if title else "",
                f"Feed: {feed_title}" if feed_title else "",
                f"Published: {published}" if published else "",
                f"Author: {author}" if author else "",
                f"Tags: {', '.join(tags)}" if tags else "",
                f"Link: {safe_link}" if safe_link else "",
            ]
            header = "\n".join(line for line in lines if line)
            page_text = f"{header}\n\n{body}" if header else body
            metadata: dict[str, Any] = {
                "feed_item_number": index,
                "feed_item_id": identifier,
            }
            if feed_title:
                metadata["feed_title"] = feed_title
            if safe_link:
                metadata["url"] = safe_link
            if published:
                metadata["published_at"] = published
            if author:
                metadata["author"] = author
            if tags:
                metadata["categories"] = tags
            output_bytes += len(page_text.encode("utf-8"))
            output_bytes += len(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
            if output_bytes > self.max_output_bytes:
                raise ValueError(f"JSON Feed exceeds max_output_bytes={self.max_output_bytes}")
            pages.append(ParsedPage(page_text, page_number=index, metadata=metadata))
        return tuple(pages)

    def _check_complexity(self, text: str) -> None:
        depth = 0
        tokens = 0
        in_string = False
        escaped = False
        previous: str | None = None
        value_starts = "-0123456789tfn"

        for character in text:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                tokens += 1
                in_string = True
            elif character in "[{":
                depth += 1
                tokens += 1
                if depth > self.max_depth:
                    raise ValueError(f"JSON exceeds max_depth={self.max_depth}")
            elif character in "]}":
                depth -= 1
            elif character in value_starts and (previous is None or previous in "[:,[{"):
                tokens += 1

            if tokens > self.max_tokens:
                raise ValueError(f"JSON exceeds max_tokens={self.max_tokens}")
            if not character.isspace():
                previous = character

    @staticmethod
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")


def _render_structured_value(value: Any, *, max_output_bytes: int, format_name: str) -> str:
    """Render a normalized mapping/list tree with searchable key paths and bounded UTF-8 output."""
    fragments: list[str] = []
    output_bytes = 0

    def append_fragment(fragment: str) -> None:
        nonlocal output_bytes
        try:
            fragment_bytes = len(fragment.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError(f"{format_name} values must contain valid Unicode text") from None
        candidate_bytes = output_bytes + fragment_bytes + (1 if fragments else 0)
        if candidate_bytes > max_output_bytes:
            raise ValueError(f"{format_name} exceeds max_output_bytes={max_output_bytes}")
        fragments.append(fragment)
        output_bytes = candidate_bytes

    def emit(path: str, item: Any) -> None:
        if isinstance(item, dict):
            if not item:
                append_fragment(f"{path or '$'}: {{}}")
            else:
                for key, child in item.items():
                    safe_key = (
                        key
                        if re.fullmatch(r"[A-Za-z0-9_-]+", key)
                        else json.dumps(key, ensure_ascii=False)
                    )
                    emit(f"{path}.{safe_key}" if path else safe_key, child)
            return
        if isinstance(item, list):
            if not item:
                append_fragment(f"{path or '$'}: []")
            else:
                for index, child in enumerate(item):
                    emit(f"{path}[{index}]", child)
            return
        if isinstance(item, str):
            scalar = item if "\n" not in item and "\r" not in item else json.dumps(item)
        else:
            scalar = json.dumps(item, ensure_ascii=False, allow_nan=False)
        append_fragment(f"{path or '$'}: {scalar}")

    emit("", value)
    rendered = "\n".join(fragments)
    if not rendered.strip():
        raise ValueError(f"{format_name} document must not be empty")
    return rendered


class YAMLTextParser:
    """Validate bounded YAML and render values as searchable dotted-path text."""

    extensions = frozenset({".yaml", ".yml"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 10 * 1024 * 1024,
        max_nodes: int = 10_000,
        max_depth: int = 32,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_nodes", max_nodes),
            ("max_depth", max_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Render one safe YAML document without allowing duplicate keys or alias cycles."""
        if len(content) > self.max_input_bytes:
            raise ValueError(f"YAML exceeds max_input_bytes={self.max_input_bytes}")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("YAML files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("YAML files must not contain NUL characters")
        self._check_token_complexity(text)
        try:
            root = yaml.compose(text, Loader=_UniqueSafeYamlLoader)
        except yaml.YAMLError:
            raise ValueError("YAML files must contain one valid YAML document") from None
        if root is None:
            raise ValueError("YAML files must contain one non-empty document")
        self._check_node_complexity(root)
        try:
            loaded = yaml.load(text, Loader=_UniqueSafeYamlLoader)
            normalized = _normalize_structured_value(
                loaded,
                ancestors=set(),
                count=[0],
                max_nodes=self.max_nodes,
                max_depth=self.max_depth,
                source_label="YAML",
            )
        except (yaml.YAMLError, ValueError, RecursionError):
            raise ValueError(
                "YAML must contain unique keys and bounded JSON-compatible values"
            ) from None

        return (
            ParsedPage(
                _render_structured_value(
                    normalized,
                    max_output_bytes=self.max_output_bytes,
                    format_name="YAML",
                )
            ),
        )

    def _check_token_complexity(self, text: str) -> None:
        """Bound YAML collection depth and scanner work before constructing a node graph."""
        token_count = 0
        depth = 0
        collection_starts = (
            yaml.tokens.BlockMappingStartToken,
            yaml.tokens.BlockSequenceStartToken,
            yaml.tokens.FlowMappingStartToken,
            yaml.tokens.FlowSequenceStartToken,
        )
        collection_ends = (
            yaml.tokens.BlockEndToken,
            yaml.tokens.FlowMappingEndToken,
            yaml.tokens.FlowSequenceEndToken,
        )
        try:
            for token in yaml.scan(text, Loader=_UniqueSafeYamlLoader):
                token_count += 1
                if token_count > self.max_nodes * 4 + 32:
                    raise ValueError(
                        f"YAML exceeds max_nodes={self.max_nodes} scanner token budget"
                    )
                if isinstance(token, collection_starts):
                    depth += 1
                    if depth > self.max_depth:
                        raise ValueError(f"YAML exceeds max_depth={self.max_depth}")
                elif isinstance(token, collection_ends):
                    depth = max(0, depth - 1)
        except yaml.YAMLError:
            raise ValueError("YAML files must contain one valid YAML document") from None

    def _check_node_complexity(self, root: yaml.Node) -> None:
        count = 0

        def visit(node: yaml.Node, *, depth: int, ancestors: set[int]) -> None:
            nonlocal count
            count += 1
            if count > self.max_nodes or depth > self.max_depth:
                raise ValueError(
                    f"YAML exceeds max_nodes={self.max_nodes} or max_depth={self.max_depth}"
                )
            identity = id(node)
            if identity in ancestors:
                raise ValueError("YAML aliases must not contain cycles")
            children: Sequence[yaml.Node]
            if isinstance(node, yaml.MappingNode):
                children = [child for pair in node.value for child in pair]
            elif isinstance(node, yaml.SequenceNode):
                children = node.value
            else:
                children = ()
            if children:
                ancestors.add(identity)
                try:
                    for child in children:
                        visit(child, depth=depth + 1, ancestors=ancestors)
                finally:
                    ancestors.remove(identity)

        visit(root, depth=0, ancestors=set())


class TOMLTextParser:
    """Validate bounded TOML and render tables as searchable dotted-path text."""

    extensions = frozenset({".toml"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 10 * 1024 * 1024,
        max_nodes: int = 10_000,
        max_depth: int = 32,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_nodes", max_nodes),
            ("max_depth", max_depth),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Parse one TOML document, rejecting malformed or over-complex data."""
        if len(content) > self.max_input_bytes:
            raise ValueError(f"TOML exceeds max_input_bytes={self.max_input_bytes}")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TOML files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("TOML files must not contain NUL characters")
        self._check_token_complexity(text)
        try:
            loaded = tomllib.loads(text)
            normalized = _normalize_structured_value(
                loaded,
                ancestors=set(),
                count=[0],
                max_nodes=self.max_nodes,
                max_depth=self.max_depth,
                source_label="TOML",
            )
        except (ValueError, RecursionError):
            raise ValueError("TOML must contain one valid, bounded document") from None
        return (
            ParsedPage(
                _render_structured_value(
                    normalized,
                    max_output_bytes=self.max_output_bytes,
                    format_name="TOML",
                )
            ),
        )

    def _check_token_complexity(self, text: str) -> None:
        """Limit assignments, array/table separators, nesting, and scanner work pre-parse."""
        token_count = 0
        depth = 0
        in_string: str | None = None
        multiline_string = False
        comment = False
        index = 0
        while index < len(text):
            character = text[index]
            if character == "\n":
                comment = False
                index += 1
                continue
            if comment:
                index += 1
                continue
            if in_string is not None:
                if multiline_string:
                    if text.startswith(in_string * 3, index):
                        in_string = None
                        multiline_string = False
                        index += 3
                        continue
                elif in_string == '"' and character == "\\":
                    index += 2
                    continue
                elif character == in_string:
                    in_string = None
                index += 1
                continue
            if character == "#":
                comment = True
            elif character in ('"', "'"):
                in_string = character
                multiline_string = text.startswith(character * 3, index)
                index += 3 if multiline_string else 1
                continue
            elif character in "[{":
                depth += 1
                token_count += 1
                if depth > self.max_depth:
                    raise ValueError(f"TOML exceeds max_depth={self.max_depth}")
            elif character in "]}":
                depth = max(0, depth - 1)
                token_count += 1
            elif character in "=,":
                token_count += 1
            if token_count > self.max_nodes * 4 + 32:
                raise ValueError(f"TOML exceeds max_nodes={self.max_nodes} scanner token budget")
            index += 1


class JSONLinesTextParser:
    """Validate bounded JSON Lines records and render source line numbers for retrieval."""

    extensions = frozenset({".jsonl"})

    def __init__(
        self,
        *,
        max_records: int = 100_000,
        max_record_bytes: int = 1024 * 1024,
        max_output_bytes: int = 12 * 1024 * 1024,
        max_depth: int = 64,
        max_tokens: int = 200_000,
    ) -> None:
        for name, value in (
            ("max_records", max_records),
            ("max_record_bytes", max_record_bytes),
            ("max_output_bytes", max_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_records = max_records
        self.max_record_bytes = max_record_bytes
        self.max_output_bytes = max_output_bytes
        self.validator = JSONTextParser(max_depth=max_depth, max_tokens=max_tokens)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Validate one JSON value per non-blank line and bound rendered retrieval text."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("JSON Lines files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("JSON Lines files must not contain NUL characters")

        records: list[str] = []
        output_bytes = 0
        record_count = 0
        for line_number, line in enumerate(StringIO(text), start=1):
            if line.endswith("\n"):
                line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]
            record = line.strip(" \t")
            if not record:
                continue
            record_count += 1
            if record_count > self.max_records:
                raise ValueError(f"JSON Lines exceeds max_records={self.max_records}")
            record_bytes = record.encode("utf-8")
            if len(record_bytes) > self.max_record_bytes:
                raise ValueError(
                    f"JSON Lines record on line {line_number} exceeds "
                    f"max_record_bytes={self.max_record_bytes}"
                )
            try:
                self.validator.parse(record_bytes)
            except ValueError:
                raise ValueError(
                    f"JSON Lines record on line {line_number} must be valid JSON"
                ) from None

            rendered = f"Record {line_number}:\n{record}"
            output_bytes += len(rendered.encode("utf-8"))
            if records:
                output_bytes += 2
            if output_bytes > self.max_output_bytes:
                raise ValueError(f"JSON Lines exceeds max_output_bytes={self.max_output_bytes}")
            records.append(rendered)

        if not records:
            raise ValueError("JSON Lines file must contain at least one record")
        return (ParsedPage("\n\n".join(records)),)


class NotebookTextParser:
    """Extract source-ordered Jupyter notebook cells while excluding execution outputs."""

    extensions = frozenset({".ipynb"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_cells: int = 10_000,
        max_cell_source_bytes: int = 1024 * 1024,
        max_output_bytes: int = 10 * 1024 * 1024,
        max_depth: int = 64,
        max_tokens: int = 200_000,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_cells", max_cells),
            ("max_cell_source_bytes", max_cell_source_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_depth", max_depth),
            ("max_tokens", max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_cells = max_cells
        self.max_cell_source_bytes = max_cell_source_bytes
        self.max_output_bytes = max_output_bytes
        self.validator = JSONTextParser(max_depth=max_depth, max_tokens=max_tokens)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Validate a notebook v4 document and return bounded pages for non-empty cells."""
        if not isinstance(content, bytes):
            raise TypeError("notebook content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"notebook exceeds max_input_bytes={self.max_input_bytes}")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("notebook files must contain valid UTF-8") from None
        if "\x00" in text:
            raise ValueError("notebook files must not contain NUL characters")
        self.validator._check_complexity(text)

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("notebook JSON objects must not contain duplicate keys")
                result[key] = value
            return result

        try:
            document = json.loads(
                text,
                object_pairs_hook=unique_object,
                parse_constant=self.validator._reject_constant,
            )
        except (ValueError, RecursionError):
            raise ValueError("notebook must contain one valid JSON value") from None
        if not isinstance(document, dict):
            raise ValueError("notebook must be a JSON object")
        major_version = document.get("nbformat")
        if type(major_version) is not int or major_version != 4:
            raise ValueError("only Jupyter notebook format version 4 is supported")
        minor_version = document.get("nbformat_minor")
        if type(minor_version) is not int or minor_version < 0:
            raise ValueError("notebook nbformat_minor must be a non-negative integer")
        if not isinstance(document.get("metadata"), dict):
            raise ValueError("notebook metadata must be an object")
        cells = document.get("cells")
        if not isinstance(cells, list):
            raise ValueError("notebook cells must be an array")
        if len(cells) > self.max_cells:
            raise ValueError(f"notebook exceeds max_cells={self.max_cells}")

        pages: list[ParsedPage] = []
        output_bytes = 0
        cell_ids: set[str] = set()
        for cell_number, cell in enumerate(cells, start=1):
            if not isinstance(cell, dict):
                raise ValueError(f"notebook cell {cell_number} must be an object")
            if not isinstance(cell.get("metadata"), dict):
                raise ValueError(f"notebook cell {cell_number} metadata must be an object")
            if minor_version >= 5:
                cell_id = cell.get("id")
                if (
                    not isinstance(cell_id, str)
                    or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) is None
                ):
                    raise ValueError(f"notebook cell {cell_number} has an invalid id")
                if cell_id in cell_ids:
                    raise ValueError("notebook cell IDs must be unique")
                cell_ids.add(cell_id)
            cell_type = cell.get("cell_type")
            if not isinstance(cell_type, str) or cell_type not in {"markdown", "code", "raw"}:
                raise ValueError(f"notebook cell {cell_number} has an unsupported cell_type")
            if cell_type == "code":
                execution_count = cell.get("execution_count")
                if "execution_count" not in cell or (
                    execution_count is not None
                    and (isinstance(execution_count, bool) or not isinstance(execution_count, int))
                ):
                    raise ValueError(
                        f"notebook code cell {cell_number} has an invalid execution_count"
                    )
                if not isinstance(cell.get("outputs"), list):
                    raise ValueError(f"notebook code cell {cell_number} outputs must be an array")
            source = cell.get("source")
            if isinstance(source, str):
                source_text = source
            elif isinstance(source, list) and all(isinstance(part, str) for part in source):
                source_text = "".join(source)
            else:
                raise ValueError(f"notebook cell {cell_number} source must be text or text parts")
            if "\x00" in source_text:
                raise ValueError(f"notebook cell {cell_number} source must not contain NUL")
            try:
                source_bytes = source_text.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(
                    f"notebook cell {cell_number} source must contain valid Unicode"
                ) from None
            if len(source_bytes) > self.max_cell_source_bytes:
                raise ValueError(
                    f"notebook cell {cell_number} exceeds "
                    f"max_cell_source_bytes={self.max_cell_source_bytes}"
                )
            if not source_text.strip():
                continue
            rendered = f"Cell {cell_number} ({cell_type})\n{source_text}"
            rendered_bytes = len(rendered.encode("utf-8"))
            if output_bytes + rendered_bytes > self.max_output_bytes:
                raise ValueError(f"notebook exceeds max_output_bytes={self.max_output_bytes}")
            output_bytes += rendered_bytes
            pages.append(ParsedPage(rendered, page_number=cell_number))
        if not pages:
            raise ValueError("notebook must contain at least one non-empty cell")
        return tuple(pages)


class _XMLInputError(ValueError):
    """Validated XML input failure with a safe message."""


def _xml_local_name(element: ET.Element) -> str:
    """Return a namespace-independent, case-folded XML element name."""
    name = element.tag
    if not isinstance(name, str):
        return ""
    return name.rsplit("}", 1)[-1].casefold()


def _is_rss_atom_root(element: ET.Element) -> bool:
    """Recognize standard RSS, RSS/RDF, and Atom document roots."""
    name = _xml_local_name(element)
    tag = element.tag
    if name == "rss":
        return True
    if not isinstance(tag, str):
        return False
    if name == "rdf":
        return tag.startswith("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}")
    return tag == "{http://www.w3.org/2005/Atom}feed"


def _is_opml_root(element: ET.Element) -> bool:
    """Recognize an OPML document root without requiring a file extension."""
    return _xml_local_name(element) == "opml"


class XMLTextParser:
    """Extract bounded, path-labeled text and attributes from UTF-8 XML."""

    extensions = frozenset({".xml"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 12 * 1024 * 1024,
        max_elements: int = 250_000,
        max_depth: int = 128,
        max_attributes_per_element: int = 256,
        max_feed_items: int = 10_000,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_elements", max_elements),
            ("max_depth", max_depth),
            ("max_attributes_per_element", max_attributes_per_element),
            ("max_feed_items", max_feed_items),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.max_attributes_per_element = max_attributes_per_element
        self.max_feed_items = max_feed_items

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Stream bounded XML paths, dispatching recognized feed and OPML roots."""
        if len(content) > self.max_input_bytes:
            raise _XMLInputError(f"XML exceeds max_input_bytes={self.max_input_bytes}")
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise _XMLInputError("XML files must contain valid UTF-8") from exc
        if "\x00" in source:
            raise _XMLInputError("XML files must not contain NUL characters")
        declaration = re.match(r"\s*<\?xml\b(.*?)\?>", source, re.IGNORECASE | re.DOTALL)
        if declaration:
            encoding = re.search(
                r"\bencoding\s*=\s*(['\"])(.*?)\1",
                declaration.group(1),
                re.IGNORECASE,
            )
            if encoding and encoding.group(2).casefold().replace("_", "-") not in {
                "utf-8",
                "utf8",
            }:
                raise _XMLInputError("XML declaration must specify UTF-8 encoding")
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", source, re.IGNORECASE):
            raise _XMLInputError("XML DTD and entity declarations are not supported")

        records: list[str] = []
        output_bytes = 0
        path: list[str] = []
        elements: list[ET.Element] = []
        text_processed: list[bool] = []
        pending_tail: tuple[ET.Element, tuple[str, ...]] | None = None
        element_count = 0

        def append_record(record: str) -> None:
            nonlocal output_bytes
            output_bytes += len(record.encode("utf-8"))
            if records:
                output_bytes += 2
            if output_bytes > self.max_output_bytes:
                raise _XMLInputError(f"XML exceeds max_output_bytes={self.max_output_bytes}")
            records.append(record)

        def label(name: str) -> str:
            if name.startswith("{") and "}" in name:
                namespace, local_name = name[1:].split("}", 1)
                return f"{{{namespace}}}{local_name}"
            return name

        def append_text(element_path: Sequence[str], value: str | None) -> None:
            if not value:
                return
            normalized = re.sub(r"\s+", " ", value).strip()
            if normalized:
                append_record(f"/{'/'.join(element_path)}: {normalized}")

        try:
            for event, element in ET.iterparse(BytesIO(content), events=("start", "end")):
                if pending_tail is not None:
                    previous, parent_path = pending_tail
                    append_text(parent_path, previous.tail)
                    previous.clear()
                    pending_tail = None
                if event == "start":
                    if element_count == 0 and _is_rss_atom_root(element):
                        return RSSAtomTextParser(
                            max_input_bytes=self.max_input_bytes,
                            max_output_bytes=self.max_output_bytes,
                            max_items=self.max_feed_items,
                            max_elements=self.max_elements,
                            max_depth=self.max_depth,
                            max_attributes_per_element=self.max_attributes_per_element,
                        ).parse(content)
                    if element_count == 0 and _is_opml_root(element):
                        return OPMLTextParser(
                            max_input_bytes=self.max_input_bytes,
                            max_output_bytes=self.max_output_bytes,
                            max_outlines=self.max_feed_items,
                            max_elements=self.max_elements,
                            max_depth=self.max_depth,
                            max_attributes_per_element=self.max_attributes_per_element,
                        ).parse(content)
                    if elements and not text_processed[-1]:
                        append_text(path, elements[-1].text)
                        text_processed[-1] = True
                    element_count += 1
                    if element_count > self.max_elements:
                        raise _XMLInputError(f"XML exceeds max_elements={self.max_elements}")
                    if len(path) + 1 > self.max_depth:
                        raise _XMLInputError(f"XML exceeds max_depth={self.max_depth}")
                    if len(element.attrib) > self.max_attributes_per_element:
                        raise _XMLInputError(
                            "XML element exceeds "
                            f"max_attributes_per_element={self.max_attributes_per_element}"
                        )
                    path.append(label(element.tag))
                    elements.append(element)
                    text_processed.append(False)
                    if element.attrib:
                        attributes = ", ".join(
                            f"@{label(name)}={json.dumps(value, ensure_ascii=False)}"
                            for name, value in element.attrib.items()
                        )
                        append_record(f"/{'/'.join(path)} [{attributes}]")
                    continue

                if not text_processed[-1]:
                    append_text(path, element.text)
                elements.pop()
                path.pop()
                text_processed.pop()
                if elements:
                    elements[-1].remove(element)
                    pending_tail = (element, tuple(path))
                else:
                    element.clear()
            if pending_tail is not None:
                previous, parent_path = pending_tail
                append_text(parent_path, previous.tail)
                previous.clear()
        except _XMLInputError:
            raise
        except ValueError:
            raise
        except Exception:
            raise ValueError("XML files must contain one well-formed document") from None

        return (ParsedPage("\n\n".join(records)),)


class RSSAtomTextParser:
    """Extract bounded RSS 2.0, RSS 1.0/RDF, and Atom feed entries as cited pages."""

    extensions = frozenset({".rss", ".atom"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 12 * 1024 * 1024,
        max_items: int = 10_000,
        max_elements: int = 250_000,
        max_depth: int = 128,
        max_attributes_per_element: int = 256,
        max_categories_per_item: int = 100,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_items", max_items),
            ("max_elements", max_elements),
            ("max_depth", max_depth),
            ("max_attributes_per_element", max_attributes_per_element),
            ("max_categories_per_item", max_categories_per_item),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_items = max_items
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.max_attributes_per_element = max_attributes_per_element
        self.max_categories_per_item = max_categories_per_item

    @staticmethod
    def _local_name(element: ET.Element) -> str:
        return _xml_local_name(element)

    @staticmethod
    def _plain_text(value: str) -> str:
        if not value.strip():
            return ""
        visible = HTMLTextParser().parse(value.encode("utf-8"))[0].text
        return re.sub(r"\s+", " ", visible).strip()

    @staticmethod
    def _safe_link(value: str) -> str | None:
        """Keep only absolute HTTP(S) links suitable for source citations."""
        if not value or any(ord(character) <= 0x20 for character in value):
            return None
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme.casefold() not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
            ):
                return None
        except ValueError:
            return None
        return value

    def _entry_page(self, element: ET.Element, feed_title: str | None, page: int) -> ParsedPage:
        values: dict[str, str] = {}
        categories: list[str] = []
        link: str | None = None
        alternate_link: str | None = None
        for child in element:
            name = self._local_name(child)
            if name == "link":
                href = child.attrib.get("href", "").strip()
                relation = child.attrib.get("rel", "alternate").casefold()
                candidate = self._safe_link(href or "".join(child.itertext()).strip())
                if candidate and relation == "alternate" and alternate_link is None:
                    alternate_link = candidate
                elif candidate and link is None:
                    link = candidate
                continue
            if name == "category":
                category = (child.attrib.get("term") or child.attrib.get("label") or "").strip()
                category = category or "".join(child.itertext()).strip()
                if category:
                    categories.append(category)
                    if len(categories) > self.max_categories_per_item:
                        raise ValueError(
                            "feed item exceeds "
                            f"max_categories_per_item={self.max_categories_per_item}"
                        )
                continue
            if (
                name
                in {
                    "title",
                    "id",
                    "guid",
                    "description",
                    "summary",
                    "content",
                    "encoded",
                    "published",
                    "pubdate",
                    "updated",
                    "date",
                    "author",
                    "creator",
                }
                and name not in values
            ):
                values[name] = " ".join("".join(child.itertext()).split())
                if name == "author":
                    author_name = next(
                        (
                            " ".join("".join(part.itertext()).split())
                            for part in child
                            if self._local_name(part) == "name"
                        ),
                        "",
                    )
                    if author_name:
                        values[name] = author_name

        link = alternate_link or link
        title = self._plain_text(values.get("title", ""))
        identifier = values.get("id") or values.get("guid")
        published = (
            values.get("published")
            or values.get("pubdate")
            or values.get("updated")
            or values.get("date")
        )
        author = values.get("author") or values.get("creator")
        content = values.get("encoded") or values.get("content") or values.get("description")
        content = content or values.get("summary", "")
        body = self._plain_text(content)
        display_fields = [
            f"Title: {title}" if title else "",
            f"Feed: {feed_title}" if feed_title else "",
            f"Published: {self._plain_text(published)}" if published else "",
            f"Author: {self._plain_text(author)}" if author else "",
            f"Categories: {', '.join(self._plain_text(item) for item in categories)}"
            if categories
            else "",
            f"Link: {link}" if link else "",
        ]
        text = "\n".join(field for field in display_fields if field)
        if body:
            text = f"{text}\n\n{body}" if text else body
        metadata: dict[str, Any] = {"feed_item_number": page}
        if feed_title:
            metadata["feed_title"] = feed_title
        if identifier:
            metadata["feed_item_id"] = identifier
        if link:
            metadata["url"] = link
        if published:
            metadata["published_at"] = self._plain_text(published)
        if author:
            metadata["author"] = self._plain_text(author)
        if categories:
            metadata["categories"] = [self._plain_text(item) for item in categories]
        return ParsedPage(text=text, page_number=page, metadata=metadata)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Stream feed items into bounded pages; reject DTDs, entities, and malformed XML."""
        if not isinstance(content, bytes):
            raise TypeError("feed content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"feed exceeds max_input_bytes={self.max_input_bytes}")
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("RSS and Atom feeds must contain valid UTF-8") from None
        if "\x00" in source:
            raise ValueError("RSS and Atom feeds must not contain NUL characters")
        declaration = re.match(r"\s*<\?xml\b(.*?)\?>", source, re.IGNORECASE | re.DOTALL)
        if declaration:
            encoding = re.search(
                r"\bencoding\s*=\s*(['\"])(.*?)\1",
                declaration.group(1),
                re.IGNORECASE,
            )
            if encoding and encoding.group(2).casefold().replace("_", "-") not in {
                "utf-8",
                "utf8",
            }:
                raise ValueError("RSS and Atom feed declarations must specify UTF-8")
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", source, re.IGNORECASE):
            raise ValueError("RSS and Atom feed DTD and entity declarations are not supported")

        pages: list[ParsedPage] = []
        stack: list[ET.Element] = []
        root_name: str | None = None
        item_depth: int | None = None
        feed_title: str | None = None
        element_count = 0
        output_bytes = 0
        try:
            for event, element in ET.iterparse(BytesIO(content), events=("start", "end")):
                name = self._local_name(element)
                if event == "start":
                    if not stack:
                        root_name = name
                        if root_name not in {"rss", "rdf", "feed"}:
                            raise ValueError("feed root must be RSS, RDF, or Atom")
                    element_count += 1
                    if element_count > self.max_elements:
                        raise ValueError(f"feed exceeds max_elements={self.max_elements}")
                    if len(stack) + 1 > self.max_depth:
                        raise ValueError(f"feed exceeds max_depth={self.max_depth}")
                    if len(element.attrib) > self.max_attributes_per_element:
                        raise ValueError(
                            "feed element exceeds "
                            "max_attributes_per_element="
                            f"{self.max_attributes_per_element}"
                        )
                    if len(stack) == 1 and root_name == "feed" and name == "entry":
                        item_depth = 2
                    elif (
                        root_name in {"rss", "rdf"}
                        and name == "item"
                        and stack
                        and self._local_name(stack[-1]) in {"channel", "rdf"}
                    ):
                        item_depth = len(stack) + 1
                    stack.append(element)
                    continue

                if item_depth == len(stack):
                    if len(pages) >= self.max_items:
                        raise ValueError(f"feed exceeds max_items={self.max_items}")
                    page = self._entry_page(element, feed_title, len(pages) + 1)
                    try:
                        page_bytes = len(page.text.encode("utf-8")) + len(
                            json.dumps(
                                page.metadata,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        )
                    except (TypeError, ValueError, UnicodeEncodeError):
                        raise ValueError("feed item metadata is not bounded JSON text") from None
                    output_bytes += page_bytes
                    if output_bytes > self.max_output_bytes:
                        raise ValueError(f"feed exceeds max_output_bytes={self.max_output_bytes}")
                    pages.append(page)
                    item_depth = None
                elif item_depth is None and name == "title" and len(stack) >= 2:
                    parent_name = self._local_name(stack[-2])
                    if (root_name == "feed" and parent_name == "feed") or (
                        root_name in {"rss", "rdf"} and parent_name == "channel"
                    ):
                        title = self._plain_text("".join(element.itertext()))
                        if title and feed_title is None:
                            feed_title = title

                stack.pop()
                if stack and item_depth is None:
                    stack[-1].remove(element)
                    element.clear()
                elif not stack:
                    element.clear()
        except ValueError:
            raise
        except Exception:
            raise ValueError(
                "RSS and Atom feeds must contain one well-formed XML document"
            ) from None
        return tuple(pages)


class OPMLTextParser:
    """Index bounded OPML outlines as cited pages without following their URLs."""

    extensions = frozenset({".opml"})
    _SUPPORTED_VERSIONS = frozenset({"1.0", "1.1", "2.0"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_output_bytes: int = 12 * 1024 * 1024,
        max_outlines: int = 10_000,
        max_elements: int = 250_000,
        max_depth: int = 128,
        max_attributes_per_element: int = 256,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_output_bytes", max_output_bytes),
            ("max_outlines", max_outlines),
            ("max_elements", max_elements),
            ("max_depth", max_depth),
            ("max_attributes_per_element", max_attributes_per_element),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_output_bytes = max_output_bytes
        self.max_outlines = max_outlines
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.max_attributes_per_element = max_attributes_per_element

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Validate one OPML document and render source-ordered outline pages."""
        if not isinstance(content, bytes):
            raise TypeError("OPML content must be bytes")
        if len(content) > self.max_input_bytes:
            raise ValueError(f"OPML exceeds max_input_bytes={self.max_input_bytes}")
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError("OPML files must contain valid UTF-8") from None
        if "\x00" in source:
            raise ValueError("OPML files must not contain NUL characters")
        declaration = re.match(r"\s*<\?xml\b(.*?)\?>", source, re.IGNORECASE | re.DOTALL)
        if declaration:
            encoding = re.search(
                r"\bencoding\s*=\s*(['\"])(.*?)\1",
                declaration.group(1),
                re.IGNORECASE,
            )
            if encoding and encoding.group(2).casefold().replace("_", "-") not in {
                "utf-8",
                "utf8",
            }:
                raise ValueError("OPML declaration must specify UTF-8 encoding")
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", source, re.IGNORECASE):
            raise ValueError("OPML DTD and entity declarations are not supported")
        root_seen = False
        version = ""
        head_count = 0
        body_count = 0
        title_count = 0
        document_title = ""
        element_count = 0
        stack: list[ET.Element] = []
        try:
            for event, element in ET.iterparse(BytesIO(content), events=("start", "end")):
                name = _xml_local_name(element)
                if event == "start":
                    element_count += 1
                    if element_count > self.max_elements:
                        raise ValueError(f"OPML exceeds max_elements={self.max_elements}")
                    if len(stack) + 1 > self.max_depth:
                        raise ValueError(f"OPML exceeds max_depth={self.max_depth}")
                    if len(element.attrib) > self.max_attributes_per_element:
                        raise ValueError(
                            "OPML element exceeds "
                            f"max_attributes_per_element={self.max_attributes_per_element}"
                        )
                    if not stack:
                        root_seen = True
                        if name != "opml":
                            raise ValueError("OPML root element must be opml")
                        version = element.attrib.get("version", "")
                        if version not in self._SUPPORTED_VERSIONS:
                            raise ValueError("OPML version must be 1.0, 1.1, or 2.0")
                    elif len(stack) == 1:
                        head_count += name == "head"
                        body_count += name == "body"
                    stack.append(element)
                    continue

                if len(stack) == 3 and _xml_local_name(stack[1]) == "head" and name == "title":
                    title_count += 1
                    if title_count == 1:
                        document_title = " ".join("".join(element.itertext()).split())
                if stack:
                    stack.pop()
                if stack:
                    stack[-1].remove(element)
                element.clear()
        except ValueError:
            raise
        except Exception:
            raise ValueError("OPML must contain one well-formed XML document") from None
        if not root_seen:
            raise ValueError("OPML must contain one well-formed XML document")
        if head_count != 1 or body_count != 1:
            raise ValueError("OPML must contain exactly one head and one body")
        if title_count > 1:
            raise ValueError("OPML head must not contain duplicate titles")

        pages: list[ParsedPage] = []
        output_bytes = 0
        body_depth: int | None = None
        element_names: list[str] = []
        element_stack: list[ET.Element] = []
        outline_titles: list[str] = []
        pushed_outline: list[bool] = []
        try:
            for event, element in ET.iterparse(BytesIO(content), events=("start", "end")):
                name = _xml_local_name(element)
                if event == "start":
                    depth = len(element_names) + 1
                    if depth == 2 and name == "body":
                        body_depth = depth
                    is_outline = body_depth is not None and name == "outline"
                    if is_outline:
                        if len(pages) >= self.max_outlines:
                            raise ValueError(f"OPML exceeds max_outlines={self.max_outlines}")
                        attributes = {
                            key.casefold(): value for key, value in element.attrib.items()
                        }
                        outline_text = attributes.get("text", "").strip()
                        outline_title = attributes.get("title", "").strip() or outline_text
                        if not outline_title:
                            outline_title = attributes.get("url", "").strip() or "Untitled outline"
                        description = attributes.get("description", "").strip()
                        outline_type = attributes.get("type", "").strip()
                        xml_url = attributes.get("xmlurl", "").strip()
                        html_url = attributes.get("htmlurl", "").strip()
                        outline_url = attributes.get("url", "").strip()
                        safe_xml_url = RSSAtomTextParser._safe_link(xml_url) if xml_url else None
                        safe_html_url = RSSAtomTextParser._safe_link(html_url) if html_url else None
                        safe_outline_url = (
                            RSSAtomTextParser._safe_link(outline_url) if outline_url else None
                        )
                        citation = safe_xml_url or safe_outline_url or safe_html_url
                        categories = list(outline_titles)
                        lines = [
                            f"Title: {outline_title}",
                            f"Outline: {document_title}" if document_title else "",
                            f"Type: {outline_type}" if outline_type else "",
                            f"Description: {description}" if description else "",
                            f"Categories: {' / '.join(categories)}" if categories else "",
                            f"Feed URL: {safe_xml_url}" if safe_xml_url else "",
                            f"Website URL: {safe_html_url}" if safe_html_url else "",
                            f"URL: {safe_outline_url}"
                            if safe_outline_url and safe_outline_url != citation
                            else "",
                        ]
                        page_text = "\n".join(line for line in lines if line)
                        metadata: dict[str, Any] = {"opml_outline_number": len(pages) + 1}
                        if document_title:
                            metadata["opml_title"] = document_title
                        metadata["outline_title"] = outline_title
                        if outline_type:
                            metadata["outline_type"] = outline_type
                        if categories:
                            metadata["categories"] = categories
                        if safe_xml_url:
                            metadata["xml_url"] = safe_xml_url
                        if safe_html_url:
                            metadata["html_url"] = safe_html_url
                        if citation:
                            metadata["url"] = citation
                        try:
                            output_bytes += len(page_text.encode("utf-8")) + len(
                                json.dumps(
                                    metadata,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            )
                        except (TypeError, ValueError, UnicodeEncodeError):
                            raise ValueError("OPML outline contains invalid Unicode text") from None
                        if output_bytes > self.max_output_bytes:
                            raise ValueError(
                                f"OPML exceeds max_output_bytes={self.max_output_bytes}"
                            )
                        pages.append(
                            ParsedPage(page_text, page_number=len(pages) + 1, metadata=metadata)
                        )
                        outline_titles.append(outline_title)
                    element_names.append(name)
                    element_stack.append(element)
                    pushed_outline.append(is_outline)
                    continue

                if pushed_outline.pop():
                    outline_titles.pop()
                if body_depth == len(element_names):
                    body_depth = None
                if element_names:
                    element_names.pop()
                if element_stack:
                    element_stack.pop()
                if element_stack:
                    element_stack[-1].remove(element)
                element.clear()
        except ValueError:
            raise
        except Exception:
            raise ValueError("OPML must contain one well-formed XML document") from None
        return tuple(pages)


class CSVTextParser:
    """Render bounded CSV records as labeled text for retrieval."""

    extensions = frozenset({".csv"})

    def __init__(
        self,
        *,
        delimiter: str = ",",
        max_rows: int = 100_000,
        max_columns: int = 1_000,
        max_output_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        if not isinstance(delimiter, str) or len(delimiter) != 1 or delimiter in {'"', "\r", "\n"}:
            raise ValueError("delimiter must be one character other than a quote or newline")
        for name, value in (
            ("max_rows", max_rows),
            ("max_columns", max_columns),
            ("max_output_bytes", max_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.delimiter = delimiter
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_output_bytes = max_output_bytes

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Parse a header-based CSV file into searchable, field-labeled records."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV files must contain valid UTF-8") from exc
        if "\x00" in text:
            raise ValueError("CSV files must not contain NUL characters")

        reader = csv.reader(StringIO(text, newline=""), delimiter=self.delimiter, strict=True)
        try:
            headers = next(reader, None)
            if headers is None or not headers:
                raise ValueError("CSV files must include a header row")
            if len(headers) > self.max_columns:
                raise ValueError(f"CSV exceeds max_columns={self.max_columns}")
            headers = [header.strip() for header in headers]
            if any(not header for header in headers):
                raise ValueError("CSV header names must not be empty")
            normalized_headers = [header.casefold() for header in headers]
            if len(set(normalized_headers)) != len(normalized_headers):
                raise ValueError("CSV header names must be unique")

            fragments = [f"Columns: {json.dumps(headers, ensure_ascii=False)}"]
            output_bytes = len(fragments[0].encode("utf-8"))
            if output_bytes > self.max_output_bytes:
                raise ValueError(f"CSV exceeds max_output_bytes={self.max_output_bytes}")
            row_count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(headers):
                    raise ValueError(
                        "CSV records must have the same number of fields as the header"
                    )
                row_count += 1
                if row_count > self.max_rows:
                    raise ValueError(f"CSV exceeds max_rows={self.max_rows}")
                fields = "\n".join(
                    f"{json.dumps(header, ensure_ascii=False)}: "
                    f"{json.dumps(value, ensure_ascii=False)}"
                    for header, value in zip(headers, row, strict=True)
                )
                record = f"Record {row_count}:\n{fields}"
                output_bytes += 2 + len(record.encode("utf-8"))
                if output_bytes > self.max_output_bytes:
                    raise ValueError(f"CSV exceeds max_output_bytes={self.max_output_bytes}")
                fragments.append(record)
        except csv.Error:
            raise ValueError("could not parse CSV file") from None
        return (ParsedPage("\n\n".join(fragments)),)


class _ParquetInputError(ValueError):
    """A bounded Parquet file violates the supported ingestion contract."""


class ParquetTextParser:
    """Render bounded, flat Parquet tables as searchable field-labeled row pages."""

    extensions = frozenset({".parquet"})

    def __init__(
        self,
        *,
        max_input_bytes: int = 10 * 1024 * 1024,
        max_rows: int = 100_000,
        max_columns: int = 256,
        max_row_groups: int = 10_000,
        max_pages: int = 1_000,
        max_rows_per_page: int = 100,
        max_cell_bytes: int = 64 * 1024,
        max_row_bytes: int = 1024 * 1024,
        max_uncompressed_bytes: int = 64 * 1024 * 1024,
        max_output_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        for name, value in (
            ("max_input_bytes", max_input_bytes),
            ("max_rows", max_rows),
            ("max_columns", max_columns),
            ("max_row_groups", max_row_groups),
            ("max_pages", max_pages),
            ("max_rows_per_page", max_rows_per_page),
            ("max_cell_bytes", max_cell_bytes),
            ("max_row_bytes", max_row_bytes),
            ("max_uncompressed_bytes", max_uncompressed_bytes),
            ("max_output_bytes", max_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_input_bytes = max_input_bytes
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_row_groups = max_row_groups
        self.max_pages = max_pages
        self.max_rows_per_page = max_rows_per_page
        self.max_cell_bytes = max_cell_bytes
        self.max_row_bytes = max_row_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_output_bytes = max_output_bytes

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Decode a flat Parquet table in bounded batches without loading it as one table."""
        if not isinstance(content, bytes):
            raise TypeError("Parquet content must be bytes")
        if len(content) > self.max_input_bytes:
            raise _ParquetInputError(f"Parquet exceeds max_input_bytes={self.max_input_bytes}")
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet ingestion requires the optional dependency; "
                "install gabby-agent-runtime[parquet]"
            ) from exc

        reader = None
        try:
            try:
                reader = parquet.ParquetFile(
                    arrow.BufferReader(content),
                    memory_map=False,
                    pre_buffer=False,
                    thrift_string_size_limit=min(self.max_input_bytes, 1024 * 1024),
                    thrift_container_size_limit=100_000,
                )
            except Exception:
                raise _ParquetInputError("could not read Parquet file") from None
            metadata = reader.metadata
            if metadata.num_rows > self.max_rows:
                raise _ParquetInputError(f"Parquet exceeds max_rows={self.max_rows}")
            if metadata.num_columns > self.max_columns:
                raise _ParquetInputError(f"Parquet exceeds max_columns={self.max_columns}")
            if metadata.num_row_groups > self.max_row_groups:
                raise _ParquetInputError(f"Parquet exceeds max_row_groups={self.max_row_groups}")

            decoded_bytes = sum(
                metadata.row_group(group_index).column(column_index).total_uncompressed_size
                for group_index in range(metadata.num_row_groups)
                for column_index in range(metadata.num_columns)
            )
            if decoded_bytes > self.max_uncompressed_bytes:
                raise _ParquetInputError(
                    f"Parquet exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                )

            fields = list(reader.schema_arrow)
            names = [field.name for field in fields]
            if not names or any(not name or len(name) > 1024 for name in names):
                raise _ParquetInputError(
                    "Parquet fields must have non-empty names up to 1024 characters"
                )
            if len({name.casefold() for name in names}) != len(names):
                raise _ParquetInputError("Parquet field names must be unique ignoring case")
            supported = (
                arrow.types.is_boolean,
                arrow.types.is_integer,
                arrow.types.is_floating,
                arrow.types.is_decimal,
                arrow.types.is_string,
                arrow.types.is_large_string,
                arrow.types.is_date,
                arrow.types.is_time,
                arrow.types.is_timestamp,
                arrow.types.is_duration,
                arrow.types.is_null,
            )
            for field in fields:
                if not any(check(field.type) for check in supported):
                    raise _ParquetInputError(
                        f"Parquet field {field.name!r} has an unsupported nested or binary type"
                    )

            pages: list[ParsedPage] = []
            output_bytes = 0
            row_number = 0
            for batch in reader.iter_batches(
                batch_size=min(self.max_rows_per_page, self.max_rows),
                use_threads=False,
                use_pandas_metadata=False,
            ):
                fragments = [f"Columns: {json.dumps(names, ensure_ascii=False)}"]
                page_bytes = len(fragments[0].encode("utf-8"))
                if output_bytes + page_bytes > self.max_output_bytes:
                    raise _ParquetInputError(
                        f"Parquet exceeds max_output_bytes={self.max_output_bytes}"
                    )
                page_start = row_number + 1
                for batch_row in range(batch.num_rows):
                    row_number += 1
                    rendered_values: list[str] = []
                    row_bytes = 0
                    for column_index, name in enumerate(names):
                        scalar = batch.column(column_index)[batch_row]
                        field_type = fields[column_index].type
                        if (
                            scalar.is_valid
                            and (
                                arrow.types.is_string(field_type)
                                or arrow.types.is_large_string(field_type)
                            )
                            and scalar.as_buffer().size > self.max_cell_bytes
                        ):
                            raise _ParquetInputError(
                                f"Parquet cell exceeds max_cell_bytes={self.max_cell_bytes}"
                            )
                        value = scalar.as_py()
                        if isinstance(value, str):
                            try:
                                value_bytes = len(value.encode("utf-8"))
                            except UnicodeEncodeError:
                                raise _ParquetInputError(
                                    "Parquet text values must contain valid Unicode"
                                ) from None
                            if value_bytes > self.max_cell_bytes:
                                raise _ParquetInputError(
                                    f"Parquet cell exceeds max_cell_bytes={self.max_cell_bytes}"
                                )
                        elif isinstance(value, (datetime.date, datetime.time)):
                            value = value.isoformat()
                        elif isinstance(value, (datetime.timedelta, decimal.Decimal)):
                            value = str(value)
                        rendered = json.dumps(
                            {name: value},
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ": "),
                        )
                        row_bytes += len(rendered.encode("utf-8"))
                        if row_bytes > self.max_row_bytes:
                            raise _ParquetInputError(
                                f"Parquet exceeds max_row_bytes={self.max_row_bytes}"
                            )
                        rendered_values.append(rendered[1:-1])
                    record = f"Row {row_number}:\n{{" + ", ".join(rendered_values) + "}"
                    page_bytes += len(record.encode("utf-8")) + 2
                    if output_bytes + page_bytes > self.max_output_bytes:
                        raise _ParquetInputError(
                            f"Parquet exceeds max_output_bytes={self.max_output_bytes}"
                        )
                    fragments.append(record)

                output_bytes += page_bytes
                text = "\n\n".join(fragments)
                if len(pages) >= self.max_pages:
                    raise _ParquetInputError(f"Parquet exceeds max_pages={self.max_pages}")
                pages.append(
                    ParsedPage(
                        text,
                        page_number=len(pages) + 1,
                        metadata={"row_start": page_start, "row_end": row_number},
                    )
                )
            if not pages:
                raise _ParquetInputError("Parquet file must contain at least one row")
            return tuple(pages)
        except _ParquetInputError:
            raise
        except Exception:
            raise ValueError("could not read Parquet file") from None
        finally:
            if reader is not None:
                reader.close()


class HTMLTextParser:
    """Extract deterministic visible text from bounded UTF-8 HTML sources."""

    extensions = frozenset({".html", ".htm"})
    _SUPPRESSED_TAGS = frozenset(
        {"head", "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object"}
    )
    _BLOCK_TAGS = frozenset(
        {
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "td",
            "th",
            "tr",
            "ul",
        }
    )
    _VOID_TAGS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Decode UTF-8 HTML, omit executable and non-visible subtrees, and return one page."""
        try:
            source = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("HTML files must contain valid UTF-8") from exc
        if "\x00" in source:
            raise ValueError("HTML files must not contain NUL characters")

        parser = _VisibleHTMLTextParser()
        try:
            parser.feed(source)
            parser.close()
        except Exception:
            raise ValueError("could not parse HTML text") from None
        return (ParsedPage(parser.text()),)


class _VisibleHTMLTextParser(HTMLParser):
    """HTML tokenizer that extracts text without executing or retaining markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.casefold(): value for name, value in attrs}
        hidden = (
            "hidden" in attributes or (attributes.get("aria-hidden") or "").casefold() == "true"
        )
        if tag == "title":
            self._in_title = True
            self._skip_stack.append(tag)
            return
        if self._skip_stack:
            if tag not in HTMLTextParser._VOID_TAGS:
                self._skip_stack.append(tag)
            return
        if tag in HTMLTextParser._SUPPRESSED_TAGS or hidden:
            if tag not in HTMLTextParser._VOID_TAGS:
                self._skip_stack.append(tag)
            return
        if tag in HTMLTextParser._BLOCK_TAGS:
            self._boundary()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self._skip_stack:
            if tag in self._skip_stack:
                reverse_index = self._skip_stack[::-1].index(tag)
                del self._skip_stack[len(self._skip_stack) - reverse_index - 1 :]
            return
        if tag in HTMLTextParser._BLOCK_TAGS:
            self._boundary()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif not self._skip_stack:
            self._parts.append(data)

    def _boundary(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n\n")

    @staticmethod
    def _normalize(parts: Sequence[str]) -> str:
        text = "".join(parts).replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def text(self) -> str:
        title = self._normalize(self._title_parts)
        body = self._normalize(self._parts)
        return "\n\n".join(part for part in (title, body) if part)


class _PDFInputError(ValueError):
    """Validated input failure with a safe message for callers."""


class _MissingPDFOCRDependencyError(RuntimeError):
    """Missing optional image extraction support for PDF OCR."""


class PDFTextParser:
    """Extract bounded, page-attributed text from PDFs using the optional ``pdf`` extra."""

    extensions = frozenset({".pdf"})

    def __init__(
        self,
        *,
        max_pages: int = 1000,
        max_extracted_chars: int = 10_000_000,
        max_content_stream_bytes: int = 4 * 1024 * 1024,
        max_total_content_stream_bytes: int = 32 * 1024 * 1024,
        ocr_backend: OCRBackend | None = None,
        page_renderer: PDFPageRenderer | None = None,
        max_ocr_pages: int = 100,
        max_image_pixels: int = 40_000_000,
        max_ocr_image_bytes: int = 16 * 1024 * 1024,
        ocr_timeout_seconds: float = 30.0,
    ) -> None:
        for name, value in (
            ("max_pages", max_pages),
            ("max_extracted_chars", max_extracted_chars),
            ("max_content_stream_bytes", max_content_stream_bytes),
            ("max_total_content_stream_bytes", max_total_content_stream_bytes),
            ("max_ocr_pages", max_ocr_pages),
            ("max_image_pixels", max_image_pixels),
            ("max_ocr_image_bytes", max_ocr_image_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(ocr_timeout_seconds, bool) or not isinstance(
            ocr_timeout_seconds, (int, float)
        ):
            raise TypeError("ocr_timeout_seconds must be a positive number")
        if not math.isfinite(ocr_timeout_seconds) or ocr_timeout_seconds <= 0:
            raise ValueError("ocr_timeout_seconds must be a positive number")
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars
        self.max_content_stream_bytes = max_content_stream_bytes
        self.max_total_content_stream_bytes = max_total_content_stream_bytes
        self.ocr_backend = ocr_backend
        self.page_renderer = (
            page_renderer
            if page_renderer is not None
            else PDFiumPageRenderer()
            if ocr_backend is not None
            else None
        )
        self.max_ocr_pages = max_ocr_pages
        self.max_image_pixels = max_image_pixels
        self.max_ocr_image_bytes = max_ocr_image_bytes
        self.ocr_timeout_seconds = float(ocr_timeout_seconds)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Extract text from a PDF while preserving one-based page attribution."""
        try:
            from pypdf import PdfReader, apply_configuration
            from pypdf.errors import LimitReachedError
            from pypdf.generic import ArrayObject, StreamObject
        except ImportError as exc:
            raise RuntimeError(
                "PDF ingestion requires the optional dependency; install gabby-agent-runtime[pdf]"
            ) from exc

        renderer_session: PDFPageRenderSession | None = None
        try:
            if not content.startswith(b"%PDF-"):
                raise _PDFInputError("PDF header is missing")
            with apply_configuration(
                maximum_declared_stream_length=len(content),
                array_based_stream_maximum_output_length=self.max_content_stream_bytes,
                jbig2_maximum_output_length=self.max_content_stream_bytes,
                lzw_maximum_output_length=self.max_content_stream_bytes,
                run_length_maximum_output_length=self.max_content_stream_bytes,
                zlib_maximum_output_length=self.max_content_stream_bytes,
                zlib_maximum_recovery_input_length=min(5_000_000, len(content)),
                image_maximum_buffer_size=self.max_content_stream_bytes,
            ):
                reader = PdfReader(BytesIO(content), strict=False)
                if reader.is_encrypted:
                    raise _PDFInputError("encrypted PDFs are not supported")
                page_count = len(reader.pages)
                if page_count > self.max_pages:
                    raise _PDFInputError(f"PDF exceeds max_pages={self.max_pages}")
                pages: list[ParsedPage] = []
                extracted_chars = 0
                decoded_content_bytes = 0
                ocr_page_count = 0
                decoded_image_pixels = 0

                def count_decoded_streams(value: Any) -> None:
                    """Decode streams incrementally under per-stream and aggregate limits."""
                    nonlocal decoded_content_bytes
                    resolved = value.get_object()
                    if isinstance(resolved, ArrayObject):
                        for item in resolved:
                            count_decoded_streams(item)
                    elif isinstance(resolved, StreamObject):
                        decoded_content_bytes += len(resolved.get_data())
                        if decoded_content_bytes > self.max_total_content_stream_bytes:
                            raise _PDFInputError(
                                "PDF exceeds "
                                f"max_total_content_stream_bytes={self.max_total_content_stream_bytes}"
                            )

                for index, page in enumerate(reader.pages, start=1):
                    if "/Contents" in page:
                        count_decoded_streams(page.raw_get("/Contents"))
                    text = page.extract_text() or ""
                    if (
                        not text.strip()
                        and self.ocr_backend is not None
                        and ocr_page_count < self.max_ocr_pages
                    ):
                        try:
                            image = next(iter(page.images), None)
                        except ImportError as exc:
                            raise _MissingPDFOCRDependencyError(
                                "PDF image OCR requires the optional dependency; install "
                                "gabby-agent-runtime[pdf-ocr]"
                            ) from exc
                        image_data = getattr(image, "data", None) if image is not None else None
                        rendered_pixel_count: int | None = None
                        if image is None and self.page_renderer is not None:
                            remaining_pixels = self.max_image_pixels - decoded_image_pixels
                            if remaining_pixels < 1:
                                raise _PDFInputError(
                                    f"PDF exceeds max_image_pixels={self.max_image_pixels}"
                                )
                            if renderer_session is None:
                                renderer_session = self.page_renderer.open_document(content)
                            rendered = renderer_session.render_page(
                                index,
                                max_image_pixels=remaining_pixels,
                                max_image_bytes=self.max_ocr_image_bytes,
                            )
                            if not isinstance(rendered, PDFRenderedPage):
                                raise _PDFInputError(
                                    "PDF page renderer must return a PDFRenderedPage"
                                )
                            image_data = rendered.image
                            rendered_pixel_count = rendered.pixel_count
                            if (
                                not isinstance(image_data, bytes)
                                or not image_data
                                or len(image_data) > self.max_ocr_image_bytes
                            ):
                                raise _PDFInputError(
                                    "PDF page renderer returned an invalid or oversized image"
                                )
                            if (
                                isinstance(rendered_pixel_count, bool)
                                or not isinstance(rendered_pixel_count, int)
                                or rendered_pixel_count < 1
                                or rendered_pixel_count > remaining_pixels
                            ):
                                raise _PDFInputError(
                                    f"PDF exceeds max_image_pixels={self.max_image_pixels}"
                                )
                        if image_data is not None:
                            if not isinstance(image_data, bytes):
                                raise _PDFInputError("PDF image could not be decoded for OCR")
                            if len(image_data) > self.max_ocr_image_bytes:
                                raise _PDFInputError(
                                    "PDF OCR image exceeds "
                                    f"max_ocr_image_bytes={self.max_ocr_image_bytes}"
                                )
                            remaining_pixels = self.max_image_pixels - decoded_image_pixels
                            if remaining_pixels < 1:
                                raise _PDFInputError(
                                    f"PDF exceeds max_image_pixels={self.max_image_pixels}"
                                )
                            recognized = self.ocr_backend.recognize(
                                image_data,
                                max_pages=1,
                                max_image_pixels=remaining_pixels,
                                timeout_seconds=self.ocr_timeout_seconds,
                            )
                            if (
                                isinstance(recognized, (str, bytes))
                                or not isinstance(recognized, Sequence)
                                or len(recognized) != 1
                                or not isinstance(recognized[0], OCRPageResult)
                            ):
                                raise _PDFInputError(
                                    "PDF OCR backend must return one bounded image result"
                                )
                            result = recognized[0]
                            if (
                                isinstance(result.pixel_count, bool)
                                or not isinstance(result.pixel_count, int)
                                or result.pixel_count < 1
                            ):
                                raise _PDFInputError(
                                    "PDF OCR backend must report a bounded positive pixel_count"
                                )
                            if result.pixel_count > remaining_pixels:
                                raise _PDFInputError(
                                    f"PDF exceeds max_image_pixels={self.max_image_pixels}"
                                )
                            if (
                                rendered_pixel_count is not None
                                and result.pixel_count != rendered_pixel_count
                            ):
                                raise _PDFInputError(
                                    "PDF OCR pixel count does not match the rendered page"
                                )
                            decoded_image_pixels += result.pixel_count
                            if not isinstance(result.text, str) or "\x00" in result.text:
                                raise _PDFInputError("PDF OCR returned invalid page text")
                            text = result.text
                            ocr_page_count += 1
                    if "\x00" in text:
                        raise _PDFInputError("PDF text must not contain NUL characters")
                    extracted_chars += len(text)
                    if extracted_chars > self.max_extracted_chars:
                        raise _PDFInputError(
                            f"PDF exceeds max_extracted_chars={self.max_extracted_chars}"
                        )
                    pages.append(ParsedPage(text, page_number=index))
                return pages
        except _MissingPDFOCRDependencyError:
            raise
        except LimitReachedError:
            raise _PDFInputError(
                f"PDF exceeds max_content_stream_bytes={self.max_content_stream_bytes}"
            ) from None
        except _PDFInputError:
            raise
        except Exception:
            raise ValueError("could not parse or extract text from PDF") from None
        finally:
            if renderer_session is not None:
                with suppress(Exception):
                    renderer_session.close()


class DOCXTextParser:
    """Extract ordered paragraph text from bounded, unencrypted DOCX archives."""

    extensions = frozenset({".docx"})
    _document_part = "word/document.xml"
    _content_types_part = "[Content_Types].xml"
    _package_relationships_part = "_rels/.rels"
    _word_namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    _content_types_namespace = "http://schemas.openxmlformats.org/package/2006/content-types"
    _relationships_namespace = "http://schemas.openxmlformats.org/package/2006/relationships"
    _document_tag = f"{{{_word_namespace}}}document"
    _body_tag = f"{{{_word_namespace}}}body"
    _content_types_tag = f"{{{_content_types_namespace}}}Types"
    _relationship_tag = f"{{{_relationships_namespace}}}Relationship"
    _main_document_content_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
    )
    _max_package_metadata_xml_bytes = 1024 * 1024
    _office_document_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    )
    _paragraph_tag = f"{{{_word_namespace}}}p"
    _text_tag = f"{{{_word_namespace}}}t"
    _tab_tag = f"{{{_word_namespace}}}tab"
    _break_tags = {f"{{{_word_namespace}}}br", f"{{{_word_namespace}}}cr"}

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_document_xml_bytes: int = 16 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_paragraphs: int = 100_000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_document_xml_bytes", max_document_xml_bytes, 512 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_paragraphs", max_paragraphs, 1_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_document_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_document_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_document_xml_bytes = max_document_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_paragraphs = max_paragraphs
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Extract text without unpacking files to disk or expanding unrelated archive parts."""
        if not isinstance(content, bytes):
            raise TypeError("DOCX content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"DOCX exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > self.max_xml_elements:
                    raise _DOCXInputError(f"DOCX exceeds max_xml_elements={self.max_xml_elements}")
                total_uncompressed = 0
                names: set[str] = set()
                by_name: dict[str, zipfile.ZipInfo] = {}
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _DOCXInputError("DOCX contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in names:
                        raise _DOCXInputError("DOCX contains duplicate archive paths")
                    names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _DOCXInputError("encrypted DOCX archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _DOCXInputError(
                            f"DOCX exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )

                document_info = by_name.get(self._document_part)
                content_types_info = by_name.get(self._content_types_part)
                relationships_info = by_name.get(self._package_relationships_part)
                if (
                    document_info is None
                    or content_types_info is None
                    or relationships_info is None
                ):
                    raise _DOCXInputError("DOCX package is missing required document parts")
                if document_info.file_size > self.max_document_xml_bytes:
                    raise _DOCXInputError(
                        "DOCX document XML exceeds "
                        f"max_document_xml_bytes={self.max_document_xml_bytes}"
                    )
                for metadata_info in (content_types_info, relationships_info):
                    if metadata_info.file_size > self._max_package_metadata_xml_bytes:
                        raise _DOCXInputError(
                            "DOCX package metadata XML exceeds "
                            f"max_metadata_xml_bytes={self._max_package_metadata_xml_bytes}"
                        )
                document_xml = self._read_member(archive, document_info)
                content_types_xml = self._read_member(archive, content_types_info)
                relationships_xml = self._read_member(archive, relationships_info)
            self._validate_xml_part(document_xml)
            content_types_root = self._parse_xml_part(content_types_xml)
            relationships_root = self._parse_xml_part(relationships_xml)
            if content_types_root.tag != self._content_types_tag:
                raise _DOCXInputError("DOCX content types part is malformed")
            main_type_found = any(
                entry.attrib.get("PartName") == f"/{self._document_part}"
                and entry.attrib.get("ContentType") == self._main_document_content_type
                for entry in content_types_root
            )
            if not main_type_found:
                raise _DOCXInputError("DOCX content types do not declare the main document part")
            if relationships_root.tag != f"{{{self._relationships_namespace}}}Relationships":
                raise _DOCXInputError("DOCX package relationships part is malformed")
            document_relationship_found = any(
                relationship.tag == self._relationship_tag
                and relationship.attrib.get("Type") == self._office_document_relationship
                and relationship.attrib.get("Target", "").lstrip("/") == self._document_part
                and relationship.attrib.get("TargetMode") != "External"
                for relationship in relationships_root
            )
            if not document_relationship_found:
                raise _DOCXInputError("DOCX package does not reference its main document part")
            return (ParsedPage(self._extract_paragraphs(document_xml)),)
        except _DOCXInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
        ):
            raise ValueError("DOCX package is malformed or could not be read") from None

    def _read_member(self, archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> bytes:
        """Read the selected XML part incrementally and verify its declared size and CRC."""
        content = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(content) + len(chunk) > self.max_document_xml_bytes:
                    raise _DOCXInputError(
                        "DOCX document XML exceeds "
                        f"max_document_xml_bytes={self.max_document_xml_bytes}"
                    )
                content.extend(chunk)
        if len(content) != member.file_size:
            raise _DOCXInputError("DOCX document XML size does not match its archive entry")
        return bytes(content)

    def _parse_xml_part(self, content: bytes) -> ET.Element:
        """Decode and parse one bounded XML part after rejecting entity declarations."""
        self._validate_xml_part(content)
        return ET.fromstring(content)

    @staticmethod
    def _validate_xml_part(content: bytes) -> None:
        """Require bounded UTF-8/UTF-16 XML and reject DTD/entity declarations."""
        try:
            xml_text = (
                content.decode("utf-16")
                if content.startswith((b"\xff\xfe", b"\xfe\xff"))
                else content.decode("utf-8-sig")
            )
        except UnicodeDecodeError:
            raise _DOCXInputError("DOCX XML parts must use UTF-8 or UTF-16 encoding") from None
        if re.search(r"<!\s*(?:doctype|entity)\b", xml_text, flags=re.IGNORECASE):
            raise _DOCXInputError("DOCX XML parts must not declare entities or a doctype")

    def _extract_paragraphs(self, document_xml: bytes) -> str:
        """Stream XML elements, retaining only the current paragraph while enforcing bounds."""
        paragraphs: list[str] = []
        stack: list[ET.Element] = []
        active_paragraphs = 0
        body_depth = 0
        element_count = 0
        extracted_chars = 0
        root_seen = False
        body_seen = False
        for event, element in ET.iterparse(BytesIO(document_xml), events=("start", "end")):
            if event == "start":
                element_count += 1
                if element_count > self.max_xml_elements:
                    raise _DOCXInputError(f"DOCX exceeds max_xml_elements={self.max_xml_elements}")
                if not root_seen:
                    root_seen = True
                    if element.tag != self._document_tag:
                        raise _DOCXInputError("DOCX document XML has an unexpected root element")
                stack.append(element)
                if element.tag == self._body_tag:
                    if body_seen:
                        raise _DOCXInputError("DOCX document XML contains multiple body elements")
                    body_seen = True
                    body_depth += 1
                elif element.tag == self._paragraph_tag and body_depth:
                    active_paragraphs += 1
                continue

            if element.tag == self._paragraph_tag and body_depth:
                active_paragraphs -= 1
                if active_paragraphs == 0:
                    parts: list[str] = []
                    for child in element.iter():
                        if child.tag == self._text_tag and child.text:
                            parts.append(child.text)
                        elif child.tag == self._tab_tag:
                            parts.append("\t")
                        elif child.tag in self._break_tags:
                            parts.append("\n")
                    paragraph = "".join(parts)
                    if paragraph:
                        if len(paragraphs) >= self.max_paragraphs:
                            raise _DOCXInputError(
                                f"DOCX exceeds max_paragraphs={self.max_paragraphs}"
                            )
                        extracted_chars += len(paragraph)
                        if extracted_chars > self.max_extracted_chars:
                            raise _DOCXInputError(
                                f"DOCX exceeds max_extracted_chars={self.max_extracted_chars}"
                            )
                        paragraphs.append(paragraph)
                element.clear()
            elif element.tag == self._body_tag:
                body_depth -= 1
                element.clear()
            elif active_paragraphs == 0:
                element.clear()
            stack.pop()
        if not body_seen:
            raise _DOCXInputError("DOCX document XML is missing its document body")
        return "\n\n".join(paragraphs)


class _DOCXInputError(ValueError):
    """Expected DOCX validation failure with a safe, useful message."""


class PPTXTextParser:
    """Extract bounded, slide-attributed text from unencrypted PowerPoint archives."""

    extensions = frozenset({".pptx"})
    _presentation_part = "ppt/presentation.xml"
    _presentation_relationships_part = "ppt/_rels/presentation.xml.rels"
    _package_relationships_part = "_rels/.rels"
    _content_types_part = "[Content_Types].xml"
    _presentation_namespace = "http://schemas.openxmlformats.org/presentationml/2006/main"
    _drawing_namespace = "http://schemas.openxmlformats.org/drawingml/2006/main"
    _relationships_namespace = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    _package_relationships_namespace = (
        "http://schemas.openxmlformats.org/package/2006/relationships"
    )
    _content_types_namespace = "http://schemas.openxmlformats.org/package/2006/content-types"
    _office_document_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    )
    _slide_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"
    )
    _presentation_content_type = (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
    )
    _slide_content_type = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
    _max_package_metadata_xml_bytes = 1024 * 1024
    _presentation_tag = f"{{{_presentation_namespace}}}presentation"
    _slide_list_tag = f"{{{_presentation_namespace}}}sldIdLst"
    _slide_id_tag = f"{{{_presentation_namespace}}}sldId"
    _text_paragraph_tag = f"{{{_drawing_namespace}}}p"
    _text_tag = f"{{{_drawing_namespace}}}t"
    _break_tag = f"{{{_drawing_namespace}}}br"

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_slide_xml_bytes: int = 4 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_slides: int = 100,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_slide_xml_bytes", max_slide_xml_bytes, 512 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_slides", max_slides, 5_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_slide_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_slide_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_slide_xml_bytes = max_slide_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_slides = max_slides
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Read slides in presentation order without extracting archive files to disk."""
        if not isinstance(content, bytes):
            raise TypeError("PPTX content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"PPTX exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > self.max_xml_elements:
                    raise _PPTXInputError(f"PPTX exceeds max_xml_elements={self.max_xml_elements}")
                by_name: dict[str, zipfile.ZipInfo] = {}
                normalized_names: set[str] = set()
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _PPTXInputError("PPTX contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _PPTXInputError("PPTX contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _PPTXInputError("encrypted PPTX archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _PPTXInputError(
                            f"PPTX exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )

                required_parts = (
                    self._presentation_part,
                    self._presentation_relationships_part,
                    self._package_relationships_part,
                    self._content_types_part,
                )
                if any(part not in by_name for part in required_parts):
                    raise _PPTXInputError("PPTX package is missing required presentation parts")
                presentation_info = by_name[self._presentation_part]
                metadata_infos = [
                    by_name[part] for part in required_parts if part != self._presentation_part
                ]
                for metadata_info in metadata_infos:
                    if metadata_info.file_size > self._max_package_metadata_xml_bytes:
                        raise _PPTXInputError(
                            "PPTX package metadata XML exceeds "
                            f"max_metadata_xml_bytes={self._max_package_metadata_xml_bytes}"
                        )
                if presentation_info.file_size > self._max_package_metadata_xml_bytes:
                    raise _PPTXInputError(
                        "PPTX presentation XML exceeds max_metadata_xml_bytes=1048576"
                    )

                presentation_xml = self._read_member(archive, presentation_info, 1024 * 1024)
                presentation_relationships_xml = self._read_member(
                    archive, by_name[self._presentation_relationships_part], 1024 * 1024
                )
                package_relationships_xml = self._read_member(
                    archive, by_name[self._package_relationships_part], 1024 * 1024
                )
                content_types_xml = self._read_member(
                    archive, by_name[self._content_types_part], 1024 * 1024
                )

                presentation_root = self._parse_xml_part(presentation_xml)
                presentation_relationships = self._parse_xml_part(presentation_relationships_xml)
                package_relationships = self._parse_xml_part(package_relationships_xml)
                content_types_root = self._parse_xml_part(content_types_xml)
                self._validate_package_parts(
                    presentation_root,
                    presentation_relationships,
                    package_relationships,
                    content_types_root,
                )
                slide_targets = self._slide_targets(
                    presentation_root, presentation_relationships, content_types_root
                )
                if not slide_targets:
                    raise _PPTXInputError("PPTX presentation contains no slides")
                if len(slide_targets) > self.max_slides:
                    raise _PPTXInputError(f"PPTX exceeds max_slides={self.max_slides}")
                pages: list[ParsedPage] = []
                total_chars = 0
                for slide_number, target in enumerate(slide_targets, start=1):
                    slide_info = by_name.get(target)
                    if slide_info is None:
                        raise _PPTXInputError("PPTX references a missing slide part")
                    if slide_info.file_size > self.max_slide_xml_bytes:
                        raise _PPTXInputError(
                            f"PPTX slide XML exceeds max_slide_xml_bytes={self.max_slide_xml_bytes}"
                        )
                    slide_xml = self._read_member(archive, slide_info, self.max_slide_xml_bytes)
                    slide_root = self._parse_xml_part(slide_xml)
                    text = self._extract_slide_text(slide_root)
                    total_chars += len(text)
                    if total_chars > self.max_extracted_chars:
                        raise _PPTXInputError(
                            f"PPTX exceeds max_extracted_chars={self.max_extracted_chars}"
                        )
                    pages.append(ParsedPage(text, page_number=slide_number))
                return pages
        except _PPTXInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
        ):
            raise ValueError("PPTX package is malformed or could not be read") from None

    def _read_member(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, byte_limit: int
    ) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _PPTXInputError(f"PPTX XML part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _PPTXInputError("PPTX XML part size does not match its archive entry")
        return bytes(data)

    def _parse_xml_part(self, content: bytes) -> ET.Element:
        DOCXTextParser._validate_xml_part(content)
        return ET.fromstring(content)

    def _validate_package_parts(
        self,
        presentation: ET.Element,
        presentation_relationships: ET.Element,
        package_relationships: ET.Element,
        content_types: ET.Element,
    ) -> None:
        rels_tag = f"{{{self._package_relationships_namespace}}}Relationships"
        rel_tag = f"{{{self._package_relationships_namespace}}}Relationship"
        if presentation.tag != self._presentation_tag:
            raise _PPTXInputError("PPTX presentation XML has an unexpected root element")
        if presentation_relationships.tag != rels_tag or package_relationships.tag != rels_tag:
            raise _PPTXInputError("PPTX relationships part is malformed")
        if content_types.tag != f"{{{self._content_types_namespace}}}Types":
            raise _PPTXInputError("PPTX content types part is malformed")
        for root in (
            presentation,
            presentation_relationships,
            package_relationships,
            content_types,
        ):
            if sum(1 for _ in root.iter()) > self.max_xml_elements:
                raise _PPTXInputError(f"PPTX exceeds max_xml_elements={self.max_xml_elements}")
        has_presentation_type = any(
            item.attrib.get("PartName") == f"/{self._presentation_part}"
            and item.attrib.get("ContentType") == self._presentation_content_type
            for item in content_types
        )
        if not has_presentation_type:
            raise _PPTXInputError("PPTX content types do not declare the presentation part")
        package_reference = any(
            relation.tag == rel_tag
            and relation.attrib.get("Type") == self._office_document_relationship
            and relation.attrib.get("Target") == self._presentation_part
            and relation.attrib.get("TargetMode") != "External"
            for relation in package_relationships
        )
        if not package_reference:
            raise _PPTXInputError("PPTX package does not reference its presentation part")

    def _slide_targets(
        self,
        presentation: ET.Element,
        relationships: ET.Element,
        content_types: ET.Element,
    ) -> tuple[str, ...]:
        rel_tag = f"{{{self._package_relationships_namespace}}}Relationship"
        relationships_by_id: dict[str, str] = {}
        seen_relationship_ids: set[str] = set()
        content_type_by_part = {
            entry.attrib.get("PartName"): entry.attrib.get("ContentType") for entry in content_types
        }
        for relation in relationships:
            if relation.tag != rel_tag:
                continue
            relation_id = relation.attrib.get("Id")
            if not relation_id or relation_id in seen_relationship_ids:
                raise _PPTXInputError("PPTX contains duplicate or empty relationship IDs")
            seen_relationship_ids.add(relation_id)
            if relation.attrib.get("Type") != self._slide_relationship:
                continue
            relationship_target = relation.attrib.get("Target", "")
            target_path = PurePosixPath(relationship_target)
            if (
                relation.attrib.get("TargetMode") == "External"
                or not relationship_target.startswith("slides/")
                or len(target_path.parts) != 2
                or target_path.suffix.casefold() != ".xml"
                or "\\" in relationship_target
                or target_path.is_absolute()
                or ".." in target_path.parts
                or (target_path.parts and ":" in target_path.parts[0])
            ):
                raise _PPTXInputError("PPTX contains an unsafe slide relationship")
            relationships_by_id[relation_id] = f"ppt/{target_path.as_posix()}"

        slide_list = presentation.find(self._slide_list_tag)
        if slide_list is None:
            raise _PPTXInputError("PPTX presentation is missing its slide list")
        relationship_id_tag = f"{{{self._relationships_namespace}}}id"
        targets: list[str] = []
        for slide_id in slide_list:
            if slide_id.tag != self._slide_id_tag:
                continue
            relation_id = slide_id.attrib.get(relationship_id_tag)
            slide_target = relationships_by_id.get(relation_id or "")
            if slide_target is None:
                raise _PPTXInputError("PPTX slide list references an invalid relationship")
            if slide_target in targets:
                raise _PPTXInputError("PPTX presentation repeats a slide relationship")
            if content_type_by_part.get(f"/{slide_target}") != self._slide_content_type:
                raise _PPTXInputError("PPTX content types do not declare a referenced slide part")
            targets.append(slide_target)
        return tuple(targets)

    def _extract_slide_text(self, root: ET.Element) -> str:
        if root.tag != f"{{{self._presentation_namespace}}}sld":
            raise _PPTXInputError("PPTX slide XML has an unexpected root element")
        if sum(1 for _ in root.iter()) > self.max_xml_elements:
            raise _PPTXInputError(f"PPTX exceeds max_xml_elements={self.max_xml_elements}")
        paragraphs: list[str] = []
        for paragraph in root.iter(self._text_paragraph_tag):
            parts: list[str] = []
            for element in paragraph.iter():
                if element.tag == self._text_tag and element.text:
                    parts.append(element.text)
                elif element.tag == self._break_tag:
                    parts.append("\n")
            text = "".join(parts)
            if text:
                paragraphs.append(text)
        return "\n\n".join(paragraphs)


class _PPTXInputError(ValueError):
    """Expected PPTX validation failure with a safe, useful message."""


class EPUBTextParser:
    """Extract visible chapter text in EPUB reading order from bounded ZIP packages."""

    extensions = frozenset({".epub"})
    _container_part = "META-INF/container.xml"
    _mimetype = b"application/epub+zip"
    _container_namespace = "urn:oasis:names:tc:opendocument:xmlns:container"
    _opf_namespace = "http://www.idpf.org/2007/opf"
    _xhtml_media_types = frozenset({"application/xhtml+xml", "text/html"})

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_package_xml_bytes: int = 1024 * 1024,
        max_chapter_bytes: int = 4 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_chapters: int = 1000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_package_xml_bytes", max_package_xml_bytes, 16 * 1024 * 1024),
            ("max_chapter_bytes", max_chapter_bytes, 64 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_chapters", max_chapters, 10_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_package_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_package_xml_bytes must not exceed max_uncompressed_bytes")
        if max_chapter_bytes > max_uncompressed_bytes:
            raise ValueError("max_chapter_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_package_xml_bytes = max_package_xml_bytes
        self.max_chapter_bytes = max_chapter_bytes
        self.max_xml_elements = max_xml_elements
        self.max_chapters = max_chapters
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Return one source-attributed page per spine item without extracting files to disk."""
        if not isinstance(content, bytes):
            raise TypeError("EPUB content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"EPUB exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if not members or members[0].filename != "mimetype":
                    raise _EPUBInputError("EPUB must begin with its mimetype entry")
                if members[0].compress_type != zipfile.ZIP_STORED:
                    raise _EPUBInputError("EPUB mimetype entry must be uncompressed")
                if len(members) > self.max_xml_elements:
                    raise _EPUBInputError(f"EPUB exceeds max_xml_elements={self.max_xml_elements}")
                by_name: dict[str, zipfile.ZipInfo] = {}
                normalized_names: set[str] = set()
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _EPUBInputError("EPUB contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _EPUBInputError("EPUB contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _EPUBInputError("encrypted EPUB archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _EPUBInputError(
                            f"EPUB exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )
                mimetype = by_name.get("mimetype")
                container = by_name.get(self._container_part)
                if mimetype is None or container is None:
                    raise _EPUBInputError("EPUB package is missing required metadata parts")
                if mimetype.file_size != len(self._mimetype):
                    raise _EPUBInputError("EPUB mimetype entry is invalid")
                if self._read_member(archive, mimetype, len(self._mimetype)) != self._mimetype:
                    raise _EPUBInputError("EPUB mimetype entry is invalid")
                if container.file_size > self.max_package_xml_bytes:
                    raise _EPUBInputError("EPUB package metadata XML exceeds configured limit")
                container_root = self._parse_xml(
                    self._read_member(archive, container, self.max_package_xml_bytes)
                )
                container_tag = f"{{{self._container_namespace}}}container"
                rootfile_tag = f"{{{self._container_namespace}}}rootfile"
                if container_root.tag != container_tag:
                    raise _EPUBInputError("EPUB container metadata is malformed")
                rootfiles = [
                    item
                    for item in container_root.iter(rootfile_tag)
                    if item.attrib.get("media-type") == "application/oebps-package+xml"
                ]
                if not rootfiles:
                    raise _EPUBInputError("EPUB package has no OPF rootfile")
                opf_path = self._resolve_member_path("", rootfiles[0].attrib.get("full-path", ""))
                opf_info = by_name.get(opf_path)
                if opf_info is None:
                    raise _EPUBInputError("EPUB references a missing OPF package")
                if opf_info.file_size > self.max_package_xml_bytes:
                    raise _EPUBInputError("EPUB package metadata XML exceeds configured limit")
                opf_root = self._parse_xml(
                    self._read_member(archive, opf_info, self.max_package_xml_bytes)
                )
                if opf_root.tag != f"{{{self._opf_namespace}}}package":
                    raise _EPUBInputError("EPUB OPF package has an unexpected root element")
                if sum(1 for _ in container_root.iter()) + sum(1 for _ in opf_root.iter()) > (
                    self.max_xml_elements
                ):
                    raise _EPUBInputError(f"EPUB exceeds max_xml_elements={self.max_xml_elements}")
                manifest = opf_root.find(f"{{{self._opf_namespace}}}manifest")
                spine = opf_root.find(f"{{{self._opf_namespace}}}spine")
                if manifest is None or spine is None:
                    raise _EPUBInputError("EPUB OPF package is missing its manifest or spine")
                items: dict[str, ET.Element] = {}
                for manifest_item in manifest.findall(f"{{{self._opf_namespace}}}item"):
                    item_id = manifest_item.attrib.get("id", "")
                    if not item_id or item_id in items:
                        raise _EPUBInputError(
                            "EPUB manifest contains a missing or duplicate item id"
                        )
                    items[item_id] = manifest_item
                itemrefs = spine.findall(f"{{{self._opf_namespace}}}itemref")
                if not itemrefs:
                    raise _EPUBInputError("EPUB reading order is empty")
                if len(itemrefs) > self.max_chapters:
                    raise _EPUBInputError(f"EPUB exceeds max_chapters={self.max_chapters}")
                pages: list[ParsedPage] = []
                total_chars = 0
                for chapter_number, itemref in enumerate(itemrefs, start=1):
                    item = items.get(itemref.attrib.get("idref", ""))
                    if item is None or item.attrib.get("media-type") not in self._xhtml_media_types:
                        raise _EPUBInputError(
                            "EPUB spine references an unsupported or missing item"
                        )
                    chapter_path = self._resolve_member_path(
                        posixpath.dirname(opf_path), item.attrib.get("href", "")
                    )
                    chapter_info = by_name.get(chapter_path)
                    if chapter_info is None:
                        raise _EPUBInputError("EPUB spine references a missing chapter")
                    chapter = self._read_member(archive, chapter_info, self.max_chapter_bytes)
                    self._validate_chapter(chapter)
                    text_parser = _VisibleHTMLTextParser()
                    chapter_text = (
                        chapter.decode("utf-16")
                        if chapter.startswith((b"\xff\xfe", b"\xfe\xff"))
                        else chapter.decode("utf-8-sig")
                    )
                    text_parser.feed(chapter_text)
                    text_parser.close()
                    text = text_parser.text()
                    total_chars += len(text)
                    if total_chars > self.max_extracted_chars:
                        raise _EPUBInputError(
                            f"EPUB exceeds max_extracted_chars={self.max_extracted_chars}"
                        )
                    pages.append(ParsedPage(text, page_number=chapter_number))
                return pages
        except _EPUBInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
            UnicodeDecodeError,
        ):
            raise ValueError("EPUB package is malformed or could not be read") from None

    def _read_member(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, byte_limit: int
    ) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _EPUBInputError(f"EPUB part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _EPUBInputError("EPUB part size does not match its archive entry")
        return bytes(data)

    def _parse_xml(self, content: bytes) -> ET.Element:
        try:
            DOCXTextParser._validate_xml_part(content)
            root = ET.fromstring(content)
        except (_DOCXInputError, ET.ParseError):
            raise _EPUBInputError("EPUB XML metadata is malformed or unsafe") from None
        return root

    @staticmethod
    def _resolve_member_path(base: str, href: str) -> str:
        parsed = urlsplit(href)
        decoded = unquote(parsed.path)
        if (
            not decoded
            or parsed.scheme
            or parsed.netloc
            or decoded.startswith("/")
            or "\\" in decoded
            or (PurePosixPath(decoded).parts and ":" in PurePosixPath(decoded).parts[0])
        ):
            raise _EPUBInputError("EPUB contains an unsafe package reference")
        parts = list(PurePosixPath(base).parts)
        for part in PurePosixPath(decoded).parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    raise _EPUBInputError("EPUB contains an unsafe package reference")
                parts.pop()
            else:
                parts.append(part)
        if not parts:
            raise _EPUBInputError("EPUB contains an unsafe package reference")
        return "/".join(parts)

    @staticmethod
    def _validate_chapter(content: bytes) -> None:
        try:
            source = (
                content.decode("utf-16")
                if content.startswith((b"\xff\xfe", b"\xfe\xff"))
                else content.decode("utf-8-sig")
            )
        except UnicodeDecodeError:
            raise _EPUBInputError("EPUB chapter files must contain valid UTF-8") from None
        if "\x00" in source or re.search(r"<!\s*(?:doctype|entity)\b", source, flags=re.IGNORECASE):
            raise _EPUBInputError("EPUB chapters must not contain NUL, entities, or a doctype")


class _EPUBInputError(ValueError):
    """Expected EPUB validation failure with a safe, useful message."""


class ODTTextParser:
    """Extract ordered paragraph and heading text from bounded ODT packages."""

    extensions = frozenset({".odt"})
    _mimetype = b"application/vnd.oasis.opendocument.text"
    _content_part = "content.xml"
    _office_namespace = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    _text_namespace = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    _document_tag = f"{{{_office_namespace}}}document-content"
    _body_tag = f"{{{_office_namespace}}}body"
    _text_body_tag = f"{{{_office_namespace}}}text"
    _paragraph_tag = f"{{{_text_namespace}}}p"
    _heading_tag = f"{{{_text_namespace}}}h"
    _space_tag = f"{{{_text_namespace}}}s"
    _tab_tag = f"{{{_text_namespace}}}tab"
    _line_break_tag = f"{{{_text_namespace}}}line-break"

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_content_xml_bytes: int = 16 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_paragraphs: int = 100_000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_content_xml_bytes", max_content_xml_bytes, 512 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_paragraphs", max_paragraphs, 1_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_content_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_content_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_content_xml_bytes = max_content_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_paragraphs = max_paragraphs
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Read the bounded content XML without extracting archive members to disk."""
        if not isinstance(content, bytes):
            raise TypeError("ODT content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"ODT exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if not members or members[0].filename != "mimetype":
                    raise _ODTInputError("ODT must begin with its mimetype entry")
                if members[0].compress_type != zipfile.ZIP_STORED:
                    raise _ODTInputError("ODT mimetype entry must be uncompressed")
                if len(members) > self.max_xml_elements:
                    raise _ODTInputError(f"ODT exceeds max_xml_elements={self.max_xml_elements}")
                normalized_names: set[str] = set()
                by_name: dict[str, zipfile.ZipInfo] = {}
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _ODTInputError("ODT contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _ODTInputError("ODT contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _ODTInputError("encrypted ODT archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _ODTInputError(
                            f"ODT exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )
                mimetype = by_name.get("mimetype")
                content_info = by_name.get(self._content_part)
                if mimetype is None or content_info is None:
                    raise _ODTInputError("ODT package is missing required content parts")
                if mimetype.file_size != len(self._mimetype):
                    raise _ODTInputError("ODT mimetype entry is invalid")
                if self._read_member(archive, mimetype, len(self._mimetype)) != self._mimetype:
                    raise _ODTInputError("ODT mimetype entry is invalid")
                if content_info.file_size > self.max_content_xml_bytes:
                    raise _ODTInputError(
                        "ODT content XML exceeds "
                        f"max_content_xml_bytes={self.max_content_xml_bytes}"
                    )
                content_xml = self._read_member(archive, content_info, self.max_content_xml_bytes)
                root = self._parse_xml(content_xml)
                if root.tag != self._document_tag:
                    raise _ODTInputError("ODT content XML has an unexpected root element")
                element_count = sum(1 for _ in root.iter())
                if element_count > self.max_xml_elements:
                    raise _ODTInputError(f"ODT exceeds max_xml_elements={self.max_xml_elements}")
                body = root.find(self._body_tag)
                text_body = body.find(self._text_body_tag) if body is not None else None
                if text_body is None:
                    raise _ODTInputError("ODT content XML is missing its text body")
                block_elements: list[ET.Element] = []
                hidden_tags = {
                    f"{{{self._office_namespace}}}annotation",
                    f"{{{self._text_namespace}}}tracked-changes",
                    f"{{{self._text_namespace}}}deletion",
                }
                pending_nodes = list(reversed(list(text_body)))
                while pending_nodes:
                    node = pending_nodes.pop()
                    if node.tag in hidden_tags:
                        continue
                    if node.tag in {self._paragraph_tag, self._heading_tag}:
                        block_elements.append(node)
                        continue
                    pending_nodes.extend(reversed(list(node)))
                paragraphs: list[str] = []
                extracted_chars = 0
                for element in block_elements:
                    paragraph = self._extract_block_text(element)
                    if paragraph:
                        if len(paragraphs) >= self.max_paragraphs:
                            raise _ODTInputError(
                                f"ODT exceeds max_paragraphs={self.max_paragraphs}"
                            )
                        extracted_chars += len(paragraph)
                        if extracted_chars > self.max_extracted_chars:
                            raise _ODTInputError(
                                f"ODT exceeds max_extracted_chars={self.max_extracted_chars}"
                            )
                        paragraphs.append(paragraph)
                return (ParsedPage("\n\n".join(paragraphs)),)
        except _ODTInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
            UnicodeDecodeError,
        ):
            raise ValueError("ODT package is malformed or could not be read") from None

    def _read_member(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, byte_limit: int
    ) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _ODTInputError(f"ODT part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _ODTInputError("ODT part size does not match its archive entry")
        return bytes(data)

    @staticmethod
    def _parse_xml(content: bytes) -> ET.Element:
        try:
            DOCXTextParser._validate_xml_part(content)
            return ET.fromstring(content)
        except (_DOCXInputError, ET.ParseError):
            raise _ODTInputError("ODT XML is malformed or unsafe") from None

    def _extract_block_text(self, element: ET.Element) -> str:
        fragments: list[str] = []
        extracted_chars = 0
        stack: list[tuple[str, ET.Element | str]] = [("node", element)]
        hidden_tags = {
            f"{{{self._office_namespace}}}annotation",
            f"{{{self._text_namespace}}}tracked-changes",
            f"{{{self._text_namespace}}}deletion",
        }
        while stack:
            action, value = stack.pop()
            if action == "text":
                fragment = value
                assert isinstance(fragment, str)
                fragments.append(fragment)
                extracted_chars += len(fragment)
                if extracted_chars > self.max_extracted_chars:
                    raise _ODTInputError(
                        f"ODT exceeds max_extracted_chars={self.max_extracted_chars}"
                    )
                continue
            node = value
            assert isinstance(node, ET.Element)
            if node.tag in hidden_tags:
                continue
            if node.tag == self._space_tag:
                count_text = node.attrib.get(f"{{{self._text_namespace}}}c", "1")
                if (
                    not count_text.isdecimal()
                    or not 1 <= int(count_text) <= self.max_extracted_chars
                ):
                    raise _ODTInputError("ODT contains an invalid repeated-space count")
                if extracted_chars + int(count_text) > self.max_extracted_chars:
                    raise _ODTInputError(
                        f"ODT exceeds max_extracted_chars={self.max_extracted_chars}"
                    )
                fragment = " " * int(count_text)
            elif node.tag == self._tab_tag:
                fragment = "\t"
            elif node.tag == self._line_break_tag:
                fragment = "\n"
            elif node.text:
                fragment = node.text
            else:
                fragment = ""
            if fragment:
                fragments.append(fragment)
                extracted_chars += len(fragment)
                if extracted_chars > self.max_extracted_chars:
                    raise _ODTInputError(
                        f"ODT exceeds max_extracted_chars={self.max_extracted_chars}"
                    )
            children = list(node)
            for child in reversed(children):
                if child.tail:
                    stack.append(("text", child.tail))
                stack.append(("node", child))
        return "".join(fragments).strip()


class ODPTextParser:
    """Extract bounded, slide-attributed text from OpenDocument presentations."""

    extensions = frozenset({".odp"})
    _mimetype = b"application/vnd.oasis.opendocument.presentation"
    _content_part = "content.xml"
    _office_namespace = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    _draw_namespace = "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
    _text_namespace = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    _document_tag = f"{{{_office_namespace}}}document-content"
    _body_tag = f"{{{_office_namespace}}}body"
    _presentation_tag = f"{{{_office_namespace}}}presentation"
    _page_tag = f"{{{_draw_namespace}}}page"
    _paragraph_tags = frozenset({f"{{{_text_namespace}}}p", f"{{{_text_namespace}}}h"})
    _page_name_attribute = f"{{{_draw_namespace}}}name"

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_content_xml_bytes: int = 16 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_slides: int = 100,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_content_xml_bytes", max_content_xml_bytes, 512 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_slides", max_slides, 5_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_content_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_content_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_content_xml_bytes = max_content_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_slides = max_slides
        self.max_extracted_chars = max_extracted_chars
        self._text_extractor = ODTTextParser(max_extracted_chars=max_extracted_chars)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Read slide text in source order without extracting archive files to disk."""
        if not isinstance(content, bytes):
            raise TypeError("ODP content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"ODP exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if not members or members[0].filename != "mimetype":
                    raise _ODPInputError("ODP must begin with its mimetype entry")
                if members[0].compress_type != zipfile.ZIP_STORED:
                    raise _ODPInputError("ODP mimetype entry must be uncompressed")
                if len(members) > self.max_xml_elements:
                    raise _ODPInputError(f"ODP exceeds max_xml_elements={self.max_xml_elements}")
                by_name: dict[str, zipfile.ZipInfo] = {}
                normalized_names: set[str] = set()
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _ODPInputError("ODP contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _ODPInputError("ODP contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _ODPInputError("encrypted ODP archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _ODPInputError(
                            f"ODP exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )
                mimetype = by_name.get("mimetype")
                content_info = by_name.get(self._content_part)
                if mimetype is None or content_info is None:
                    raise _ODPInputError("ODP package is missing required content parts")
                if mimetype.file_size != len(self._mimetype):
                    raise _ODPInputError("ODP mimetype entry is invalid")
                if self._read_member(archive, mimetype, len(self._mimetype)) != self._mimetype:
                    raise _ODPInputError("ODP mimetype entry is invalid")
                if content_info.file_size > self.max_content_xml_bytes:
                    raise _ODPInputError(
                        "ODP content XML exceeds "
                        f"max_content_xml_bytes={self.max_content_xml_bytes}"
                    )
                try:
                    root = ODTTextParser._parse_xml(
                        self._read_member(archive, content_info, self.max_content_xml_bytes)
                    )
                except _ODTInputError:
                    raise _ODPInputError("ODP XML is malformed or unsafe") from None
                if root.tag != self._document_tag:
                    raise _ODPInputError("ODP content XML has an unexpected root element")
                if sum(1 for _ in root.iter()) > self.max_xml_elements:
                    raise _ODPInputError(f"ODP exceeds max_xml_elements={self.max_xml_elements}")
                body = root.find(self._body_tag)
                presentation = body.find(self._presentation_tag) if body is not None else None
                slides = presentation.findall(self._page_tag) if presentation is not None else []
                if not slides:
                    raise _ODPInputError("ODP presentation contains no slides")
                if len(slides) > self.max_slides:
                    raise _ODPInputError(f"ODP exceeds max_slides={self.max_slides}")
                pages: list[ParsedPage] = []
                total_chars = 0
                for slide_number, slide in enumerate(slides, start=1):
                    paragraphs: list[str] = []
                    pending = list(reversed(list(slide)))
                    while pending:
                        node = pending.pop()
                        if node.tag in self._paragraph_tags:
                            text = self._text_extractor._extract_block_text(node)
                            if text:
                                paragraphs.append(text)
                            continue
                        pending.extend(reversed(list(node)))
                    slide_text = "\n\n".join(paragraphs)
                    total_chars += len(slide_text)
                    if total_chars > self.max_extracted_chars:
                        raise _ODPInputError(
                            f"ODP exceeds max_extracted_chars={self.max_extracted_chars}"
                        )
                    page_name = slide.attrib.get(self._page_name_attribute, "").strip()
                    metadata = {"slide_name": page_name} if page_name else {}
                    pages.append(ParsedPage(slide_text, slide_number, metadata))
                return pages
        except _ODPInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
            UnicodeDecodeError,
        ):
            raise ValueError("ODP package is malformed or could not be read") from None

    @staticmethod
    def _read_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, byte_limit: int) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _ODPInputError(f"ODP XML part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _ODPInputError("ODP part size does not match its archive entry")
        return bytes(data)


class _ODPInputError(ValueError):
    """Expected ODP validation failure with a safe, useful message."""


class _ODTInputError(ValueError):
    """Expected ODT validation failure with a safe, useful message."""


class ODSTextParser:
    """Extract bounded, sheet- and row-labeled values from OpenDocument spreadsheets."""

    extensions = frozenset({".ods"})
    _mimetype = b"application/vnd.oasis.opendocument.spreadsheet"
    _content_part = "content.xml"
    _office_namespace = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    _table_namespace = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    _text_namespace = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    _document_tag = f"{{{_office_namespace}}}document-content"
    _body_tag = f"{{{_office_namespace}}}body"
    _spreadsheet_tag = f"{{{_office_namespace}}}spreadsheet"
    _table_tag = f"{{{_table_namespace}}}table"
    _row_tag = f"{{{_table_namespace}}}table-row"
    _cell_tags = frozenset(
        {f"{{{_table_namespace}}}table-cell", f"{{{_table_namespace}}}covered-table-cell"}
    )

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_content_xml_bytes: int = 16 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_rows: int = 100_000,
        max_columns: int = 1000,
        max_cells: int = 1_000_000,
        max_tables: int = 1000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_content_xml_bytes", max_content_xml_bytes, 512 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_rows", max_rows, 1_000_000),
            ("max_columns", max_columns, 10_000),
            ("max_cells", max_cells, 10_000_000),
            ("max_tables", max_tables, 10_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_content_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_content_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_content_xml_bytes = max_content_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_cells = max_cells
        self.max_tables = max_tables
        self.max_extracted_chars = max_extracted_chars
        self._odt_text = ODTTextParser(max_extracted_chars=max_extracted_chars)

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Read a bounded ODS package and render non-empty spreadsheet rows deterministically."""
        if not isinstance(content, bytes):
            raise TypeError("ODS content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"ODS exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if not members or members[0].filename != "mimetype":
                    raise _ODSInputError("ODS must begin with its mimetype entry")
                if members[0].compress_type != zipfile.ZIP_STORED:
                    raise _ODSInputError("ODS mimetype entry must be uncompressed")
                if len(members) > self.max_xml_elements:
                    raise _ODSInputError(f"ODS exceeds max_xml_elements={self.max_xml_elements}")
                by_name: dict[str, zipfile.ZipInfo] = {}
                normalized_names: set[str] = set()
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _ODSInputError("ODS contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _ODSInputError("ODS contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _ODSInputError("encrypted ODS archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _ODSInputError(
                            f"ODS exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )
                mimetype = by_name.get("mimetype")
                content_info = by_name.get(self._content_part)
                if mimetype is None or content_info is None:
                    raise _ODSInputError("ODS package is missing required content parts")
                if mimetype.file_size != len(self._mimetype):
                    raise _ODSInputError("ODS mimetype entry is invalid")
                if self._read_member(archive, mimetype, len(self._mimetype)) != self._mimetype:
                    raise _ODSInputError("ODS mimetype entry is invalid")
                if content_info.file_size > self.max_content_xml_bytes:
                    raise _ODSInputError(
                        "ODS content XML exceeds "
                        f"max_content_xml_bytes={self.max_content_xml_bytes}"
                    )
                content_xml = self._read_member(archive, content_info, self.max_content_xml_bytes)
                try:
                    root = ODTTextParser._parse_xml(content_xml)
                except _ODTInputError:
                    raise _ODSInputError("ODS XML is malformed or unsafe") from None
                if root.tag != self._document_tag:
                    raise _ODSInputError("ODS content XML has an unexpected root element")
                if sum(1 for _ in root.iter()) > self.max_xml_elements:
                    raise _ODSInputError(f"ODS exceeds max_xml_elements={self.max_xml_elements}")
                body = root.find(self._body_tag)
                spreadsheet = body.find(self._spreadsheet_tag) if body is not None else None
                if spreadsheet is None:
                    raise _ODSInputError("ODS content XML is missing its spreadsheet body")
                tables = spreadsheet.findall(self._table_tag)
                if not tables:
                    raise _ODSInputError("ODS workbook contains no sheets")
                if len(tables) > self.max_tables:
                    raise _ODSInputError(f"ODS exceeds max_tables={self.max_tables}")
                rendered_rows: list[str] = []
                total_rows = 0
                total_cells = 0
                extracted_chars = 0
                for table_index, table in enumerate(tables, start=1):
                    sheet_name = table.attrib.get(f"{{{self._table_namespace}}}name")
                    label = sheet_name or f"Sheet {table_index}"
                    row_number = 0
                    for row in table.iter(self._row_tag):
                        repeat_count = self._positive_repeat(
                            row,
                            f"{{{self._table_namespace}}}number-rows-repeated",
                            1_000_000,
                            "rows",
                        )
                        cells = [child for child in row if child.tag in self._cell_tags]
                        parsed_cells: list[tuple[int, str]] = []
                        column_number = 0
                        for cell in cells:
                            column_repeat = self._positive_repeat(
                                cell,
                                f"{{{self._table_namespace}}}number-columns-repeated",
                                self.max_columns,
                                "columns",
                            )
                            if column_number + column_repeat > self.max_columns:
                                raise _ODSInputError(f"ODS exceeds max_columns={self.max_columns}")
                            value = self._cell_text(cell)
                            total_cells += column_repeat * repeat_count
                            if total_cells > self.max_cells:
                                raise _ODSInputError(f"ODS exceeds max_cells={self.max_cells}")
                            if value:
                                parsed_cells.extend(
                                    (column_number + offset, value)
                                    for offset in range(column_repeat)
                                )
                            column_number += column_repeat
                        if row_number + repeat_count > self.max_rows:
                            raise _ODSInputError(f"ODS exceeds max_rows={self.max_rows}")
                        for _ in range(repeat_count):
                            row_number += 1
                            total_rows += 1
                            if total_rows > self.max_rows:
                                raise _ODSInputError(f"ODS exceeds max_rows={self.max_rows}")
                            if not parsed_cells:
                                continue
                            fields = ", ".join(
                                f"{self._column_label(column)}="
                                f"{json.dumps(value, ensure_ascii=False)}"
                                for column, value in parsed_cells
                            )
                            rendered = (
                                f"Sheet {json.dumps(label, ensure_ascii=False)}, "
                                f"row {row_number}: {fields}"
                            )
                            extracted_chars += len(rendered) + (1 if rendered_rows else 0)
                            if extracted_chars > self.max_extracted_chars:
                                raise _ODSInputError(
                                    f"ODS exceeds max_extracted_chars={self.max_extracted_chars}"
                                )
                            rendered_rows.append(rendered)
                return (ParsedPage("\n".join(rendered_rows)),)
        except _ODSInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
            UnicodeDecodeError,
        ):
            raise ValueError("ODS package is malformed or could not be read") from None

    def _cell_text(self, cell: ET.Element) -> str:
        paragraphs = [
            self._odt_text._extract_block_text(element)
            for element in cell.findall(f".//{{{self._text_namespace}}}p")
        ]
        value = "\n".join(part for part in paragraphs if part)
        if value:
            return value
        for attribute in (
            f"{{{self._office_namespace}}}string-value",
            f"{{{self._office_namespace}}}boolean-value",
            f"{{{self._office_namespace}}}date-value",
            f"{{{self._office_namespace}}}time-value",
            f"{{{self._office_namespace}}}value",
        ):
            if attribute in cell.attrib:
                return cell.attrib[attribute]
        return ""

    @staticmethod
    def _positive_repeat(element: ET.Element, attribute: str, maximum: int, label: str) -> int:
        value = element.attrib.get(attribute, "1")
        if not value.isdecimal() or not 1 <= int(value) <= maximum:
            raise _ODSInputError(f"ODS contains an invalid repeated {label} count")
        return int(value)

    @staticmethod
    def _column_label(index: int) -> str:
        label = ""
        while index >= 0:
            index, remainder = divmod(index, 26)
            label = chr(65 + remainder) + label
            index -= 1
        return label

    def _read_member(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, byte_limit: int
    ) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _ODSInputError(f"ODS part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _ODSInputError("ODS part size does not match its archive entry")
        return bytes(data)


class _ODSInputError(ValueError):
    """Expected ODS validation failure with a safe, useful message."""


class XLSXTextParser:
    """Extract bounded, sheet- and row-labeled values from OOXML workbooks."""

    extensions = frozenset({".xlsx"})
    _main_namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    _document_relationships_namespace = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    _package_relationships_namespace = (
        "http://schemas.openxmlformats.org/package/2006/relationships"
    )
    _workbook_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    )
    _worksheet_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    )
    _shared_strings_relationship = (
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings"
    )
    _max_metadata_xml_bytes = 1024 * 1024

    def __init__(
        self,
        *,
        max_archive_bytes: int = 10 * 1024 * 1024,
        max_uncompressed_bytes: int = 32 * 1024 * 1024,
        max_worksheet_xml_bytes: int = 4 * 1024 * 1024,
        max_shared_strings_xml_bytes: int = 16 * 1024 * 1024,
        max_xml_elements: int = 250_000,
        max_sheets: int = 1000,
        max_rows: int = 100_000,
        max_columns: int = 1000,
        max_cells: int = 1_000_000,
        max_shared_strings: int = 1_000_000,
        max_extracted_chars: int = 10_000_000,
    ) -> None:
        for name, value, maximum in (
            ("max_archive_bytes", max_archive_bytes, 256 * 1024 * 1024),
            ("max_uncompressed_bytes", max_uncompressed_bytes, 1024 * 1024 * 1024),
            ("max_worksheet_xml_bytes", max_worksheet_xml_bytes, 64 * 1024 * 1024),
            ("max_shared_strings_xml_bytes", max_shared_strings_xml_bytes, 128 * 1024 * 1024),
            ("max_xml_elements", max_xml_elements, 1_000_000),
            ("max_sheets", max_sheets, 10_000),
            ("max_rows", max_rows, 1_000_000),
            ("max_columns", max_columns, 16_384),
            ("max_cells", max_cells, 10_000_000),
            ("max_shared_strings", max_shared_strings, 10_000_000),
            ("max_extracted_chars", max_extracted_chars, 100_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be from 1 through {maximum}")
        if max_worksheet_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_worksheet_xml_bytes must not exceed max_uncompressed_bytes")
        if max_shared_strings_xml_bytes > max_uncompressed_bytes:
            raise ValueError("max_shared_strings_xml_bytes must not exceed max_uncompressed_bytes")
        self.max_archive_bytes = max_archive_bytes
        self.max_uncompressed_bytes = max_uncompressed_bytes
        self.max_worksheet_xml_bytes = max_worksheet_xml_bytes
        self.max_shared_strings_xml_bytes = max_shared_strings_xml_bytes
        self.max_xml_elements = max_xml_elements
        self.max_sheets = max_sheets
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_cells = max_cells
        self.max_shared_strings = max_shared_strings
        self.max_extracted_chars = max_extracted_chars

    def parse(self, content: bytes) -> Sequence[ParsedPage]:
        """Read workbook and worksheet parts without extracting archive members to disk."""
        if not isinstance(content, bytes):
            raise TypeError("XLSX content must be bytes")
        if len(content) > self.max_archive_bytes:
            raise ValueError(f"XLSX exceeds max_archive_bytes={self.max_archive_bytes}")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.infolist()
                if len(members) > self.max_xml_elements:
                    raise _XLSXInputError(f"XLSX exceeds max_xml_elements={self.max_xml_elements}")
                by_name: dict[str, zipfile.ZipInfo] = {}
                normalized_names: set[str] = set()
                total_uncompressed = 0
                for member in members:
                    name = member.filename
                    path = PurePosixPath(name)
                    if (
                        not name
                        or "\\" in name
                        or path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and ":" in path.parts[0])
                    ):
                        raise _XLSXInputError("XLSX contains an unsafe archive path")
                    normalized = path.as_posix().casefold()
                    if normalized in normalized_names:
                        raise _XLSXInputError("XLSX contains duplicate archive paths")
                    normalized_names.add(normalized)
                    by_name[name] = member
                    if member.flag_bits & 0x1:
                        raise _XLSXInputError("encrypted XLSX archives are not supported")
                    total_uncompressed += member.file_size
                    if total_uncompressed > self.max_uncompressed_bytes:
                        raise _XLSXInputError(
                            f"XLSX exceeds max_uncompressed_bytes={self.max_uncompressed_bytes}"
                        )
                package_rels = by_name.get("_rels/.rels")
                if package_rels is None:
                    raise _XLSXInputError("XLSX package is missing its root relationships")
                package_rels_root = self._read_xml_root(
                    archive, package_rels, self._max_metadata_xml_bytes, "package relationships"
                )
                package_relations = self._relationships(package_rels_root)
                workbook_targets = [
                    relation.attrib.get("Target", "")
                    for relation in package_relations
                    if relation.attrib.get("Type") == self._workbook_relationship
                    and relation.attrib.get("TargetMode") != "External"
                ]
                if len(workbook_targets) != 1:
                    raise _XLSXInputError("XLSX package does not identify one workbook part")
                workbook_path = self._resolve_target("", workbook_targets[0])
                workbook_info = by_name.get(workbook_path)
                if workbook_info is None:
                    raise _XLSXInputError("XLSX package references a missing workbook")
                workbook_root = self._read_xml_root(
                    archive, workbook_info, self._max_metadata_xml_bytes, "workbook"
                )
                if workbook_root.tag != f"{{{self._main_namespace}}}workbook":
                    raise _XLSXInputError("XLSX workbook has an unexpected root element")
                rels_path = posixpath.join(
                    posixpath.dirname(workbook_path),
                    "_rels",
                    posixpath.basename(workbook_path) + ".rels",
                )
                workbook_rels_info = by_name.get(rels_path)
                if workbook_rels_info is None:
                    raise _XLSXInputError("XLSX package is missing workbook relationships")
                workbook_rels_root = self._read_xml_root(
                    archive,
                    workbook_rels_info,
                    self._max_metadata_xml_bytes,
                    "workbook relationships",
                )
                workbook_relations = self._relationships(workbook_rels_root)
                relationships_by_id: dict[str, ET.Element] = {}
                for relation in workbook_relations:
                    relationship_id = relation.attrib.get("Id", "")
                    if not relationship_id or relationship_id in relationships_by_id:
                        raise _XLSXInputError("XLSX workbook has an invalid relationship ID")
                    relationships_by_id[relationship_id] = relation
                sheets_element = workbook_root.find(f"{{{self._main_namespace}}}sheets")
                sheet_elements = (
                    sheets_element.findall(f"{{{self._main_namespace}}}sheet")
                    if sheets_element is not None
                    else []
                )
                if not sheet_elements:
                    raise _XLSXInputError("XLSX workbook contains no worksheets")
                if len(sheet_elements) > self.max_sheets:
                    raise _XLSXInputError(f"XLSX exceeds max_sheets={self.max_sheets}")
                worksheet_targets: list[tuple[str, str]] = []
                for sheet in sheet_elements:
                    sheet_name = sheet.attrib.get("name", "")
                    relationship_id = sheet.attrib.get(
                        f"{{{self._document_relationships_namespace}}}id", ""
                    )
                    sheet_relation = relationships_by_id.get(relationship_id)
                    if not sheet_name or sheet_relation is None:
                        raise _XLSXInputError(
                            "XLSX workbook contains an invalid worksheet reference"
                        )
                    if (
                        sheet_relation.attrib.get("Type") != self._worksheet_relationship
                        or sheet_relation.attrib.get("TargetMode") == "External"
                    ):
                        raise _XLSXInputError("XLSX workbook references an unsupported worksheet")
                    target = self._resolve_target(
                        posixpath.dirname(workbook_path), sheet_relation.attrib.get("Target", "")
                    )
                    if target not in by_name:
                        raise _XLSXInputError("XLSX workbook references a missing worksheet")
                    worksheet_targets.append((sheet_name, target))
                shared_strings: list[str] = []
                shared_relations = [
                    relation
                    for relation in workbook_relations
                    if relation.attrib.get("Type") == self._shared_strings_relationship
                ]
                if len(shared_relations) > 1:
                    raise _XLSXInputError("XLSX workbook has multiple shared string tables")
                if shared_relations:
                    relation = shared_relations[0]
                    if relation.attrib.get("TargetMode") == "External":
                        raise _XLSXInputError("XLSX shared strings cannot be external")
                    shared_path = self._resolve_target(
                        posixpath.dirname(workbook_path), relation.attrib.get("Target", "")
                    )
                    shared_info = by_name.get(shared_path)
                    if shared_info is None:
                        raise _XLSXInputError(
                            "XLSX workbook references a missing shared string table"
                        )
                    shared_root = self._read_xml_root(
                        archive,
                        shared_info,
                        self.max_shared_strings_xml_bytes,
                        "shared strings",
                    )
                    shared_strings = self._shared_string_values(shared_root)
                rendered_rows: list[str] = []
                total_rows = 0
                total_cells = 0
                extracted_chars = 0
                for sheet_name, target in worksheet_targets:
                    worksheet_info = by_name[target]
                    worksheet_root = self._read_xml_root(
                        archive,
                        worksheet_info,
                        self.max_worksheet_xml_bytes,
                        "worksheet",
                    )
                    if worksheet_root.tag != f"{{{self._main_namespace}}}worksheet":
                        raise _XLSXInputError("XLSX worksheet has an unexpected root element")
                    sheet_data = worksheet_root.find(f"{{{self._main_namespace}}}sheetData")
                    if sheet_data is None:
                        continue
                    previous_row = 0
                    for row in sheet_data.findall(f"{{{self._main_namespace}}}row"):
                        row_text = row.attrib.get("r", "")
                        if not row_text.isdecimal() or int(row_text) <= previous_row:
                            raise _XLSXInputError("XLSX worksheet has an invalid row index")
                        row_number = int(row_text)
                        if row_number > self.max_rows:
                            raise _XLSXInputError(f"XLSX exceeds max_rows={self.max_rows}")
                        previous_row = row_number
                        total_rows += 1
                        if total_rows > self.max_rows:
                            raise _XLSXInputError(f"XLSX exceeds max_rows={self.max_rows}")
                        parsed_cells: list[tuple[int, str]] = []
                        seen_columns: set[int] = set()
                        for cell in row.findall(f"{{{self._main_namespace}}}c"):
                            total_cells += 1
                            if total_cells > self.max_cells:
                                raise _XLSXInputError(f"XLSX exceeds max_cells={self.max_cells}")
                            coordinate = cell.attrib.get("r", "")
                            match = re.fullmatch(r"([A-Za-z]+)([1-9][0-9]*)", coordinate)
                            if match is None or int(match.group(2)) != row_number:
                                raise _XLSXInputError(
                                    "XLSX worksheet has an invalid cell reference"
                                )
                            column_number = self._column_number(match.group(1))
                            if column_number >= self.max_columns:
                                raise _XLSXInputError(
                                    f"XLSX exceeds max_columns={self.max_columns}"
                                )
                            if column_number in seen_columns:
                                raise _XLSXInputError(
                                    "XLSX worksheet contains a duplicate cell reference"
                                )
                            seen_columns.add(column_number)
                            value = self._cell_value(cell, shared_strings)
                            if value:
                                extracted_chars += len(value)
                                if extracted_chars > self.max_extracted_chars:
                                    raise _XLSXInputError(
                                        "XLSX exceeds "
                                        f"max_extracted_chars={self.max_extracted_chars}"
                                    )
                                parsed_cells.append((column_number, value))
                        if not parsed_cells:
                            continue
                        fields = ", ".join(
                            f"{self._column_label(column)}={json.dumps(value, ensure_ascii=False)}"
                            for column, value in parsed_cells
                        )
                        rendered = (
                            f"Sheet {json.dumps(sheet_name, ensure_ascii=False)}, "
                            f"row {row_number}: {fields}"
                        )
                        extracted_chars += len(rendered) + (1 if rendered_rows else 0)
                        if extracted_chars > self.max_extracted_chars:
                            raise _XLSXInputError(
                                f"XLSX exceeds max_extracted_chars={self.max_extracted_chars}"
                            )
                        rendered_rows.append(rendered)
                return (ParsedPage("\n".join(rendered_rows)),)
        except _XLSXInputError as exc:
            raise ValueError(str(exc)) from None
        except (
            zipfile.BadZipFile,
            zlib.error,
            ET.ParseError,
            RuntimeError,
            OSError,
            EOFError,
            NotImplementedError,
            UnicodeDecodeError,
        ):
            raise ValueError("XLSX package is malformed or could not be read") from None

    def _relationships(self, root: ET.Element) -> list[ET.Element]:
        if root.tag != f"{{{self._package_relationships_namespace}}}Relationships":
            raise _XLSXInputError("XLSX relationships part is malformed")
        relations = root.findall(f"{{{self._package_relationships_namespace}}}Relationship")
        if len(relations) > self.max_xml_elements:
            raise _XLSXInputError(f"XLSX exceeds max_xml_elements={self.max_xml_elements}")
        return relations

    def _read_xml_root(
        self,
        archive: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        byte_limit: int,
        label: str,
    ) -> ET.Element:
        if member.file_size > byte_limit:
            raise _XLSXInputError(f"XLSX {label} part exceeds max_bytes={byte_limit}")
        content = self._read_member(archive, member, byte_limit, label)
        try:
            DOCXTextParser._validate_xml_part(content)
            root = ET.fromstring(content)
        except (_DOCXInputError, ET.ParseError):
            raise _XLSXInputError(f"XLSX {label} XML is malformed or unsafe") from None
        if sum(1 for _ in root.iter()) > self.max_xml_elements:
            raise _XLSXInputError(f"XLSX exceeds max_xml_elements={self.max_xml_elements}")
        return root

    def _shared_string_values(self, root: ET.Element) -> list[str]:
        if root.tag != f"{{{self._main_namespace}}}sst":
            raise _XLSXInputError("XLSX shared string table has an unexpected root element")
        values: list[str] = []
        total_chars = 0
        for item in root.findall(f"{{{self._main_namespace}}}si"):
            if len(values) >= self.max_shared_strings:
                raise _XLSXInputError(f"XLSX exceeds max_shared_strings={self.max_shared_strings}")
            value = self._rich_text(item)
            total_chars += len(value)
            if total_chars > self.max_extracted_chars:
                raise _XLSXInputError(
                    f"XLSX exceeds max_extracted_chars={self.max_extracted_chars}"
                )
            values.append(value)
        return values

    def _cell_value(self, cell: ET.Element, shared_strings: Sequence[str]) -> str:
        cell_type = cell.attrib.get("t", "n")
        if cell_type == "inlineStr":
            inline = cell.find(f"{{{self._main_namespace}}}is")
            return self._rich_text(inline) if inline is not None else ""
        value_element = cell.find(f"{{{self._main_namespace}}}v")
        if value_element is None or value_element.text is None:
            return ""
        value = value_element.text
        if cell_type == "s":
            if not value.isdecimal():
                raise _XLSXInputError("XLSX cell has an invalid shared string index")
            index = int(value)
            if index >= len(shared_strings):
                raise _XLSXInputError("XLSX cell references a missing shared string")
            return shared_strings[index]
        if cell_type == "b":
            if value not in {"0", "1"}:
                raise _XLSXInputError("XLSX cell has an invalid boolean value")
            return "TRUE" if value == "1" else "FALSE"
        return value

    def _rich_text(self, element: ET.Element) -> str:
        text_tag = f"{{{self._main_namespace}}}t"
        run_tag = f"{{{self._main_namespace}}}r"
        fragments: list[str] = []
        for child in element:
            if child.tag == text_tag:
                fragments.append(child.text or "")
            elif child.tag == run_tag:
                text = child.find(text_tag)
                if text is not None and text.text:
                    fragments.append(text.text)
        return "".join(fragments)

    @staticmethod
    def _resolve_target(base: str, target: str) -> str:
        parsed = urlsplit(target)
        path = unquote(parsed.path)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or not path:
            raise _XLSXInputError("XLSX contains an unsafe package relationship")
        if "\\" in path or "\x00" in path:
            raise _XLSXInputError("XLSX contains an unsafe package relationship")
        parts = [] if path.startswith("/") else list(PurePosixPath(base).parts)
        for part in PurePosixPath(path.lstrip("/")).parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    raise _XLSXInputError("XLSX contains an unsafe package relationship")
                parts.pop()
            else:
                if ":" in part and not parts:
                    raise _XLSXInputError("XLSX contains an unsafe package relationship")
                parts.append(part)
        if not parts:
            raise _XLSXInputError("XLSX contains an unsafe package relationship")
        return "/".join(parts)

    @staticmethod
    def _column_number(label: str) -> int:
        value = 0
        for character in label.upper():
            value = value * 26 + ord(character) - ord("A") + 1
        return value - 1

    @staticmethod
    def _column_label(index: int) -> str:
        label = ""
        while index >= 0:
            index, remainder = divmod(index, 26)
            label = chr(65 + remainder) + label
            index -= 1
        return label

    def _read_member(
        self,
        archive: zipfile.ZipFile,
        member: zipfile.ZipInfo,
        byte_limit: int,
        label: str,
    ) -> bytes:
        data = bytearray()
        with archive.open(member) as source:
            while chunk := source.read(64 * 1024):
                if len(data) + len(chunk) > byte_limit:
                    raise _XLSXInputError(f"XLSX {label} part exceeds max_bytes={byte_limit}")
                data.extend(chunk)
        if len(data) != member.file_size:
            raise _XLSXInputError(f"XLSX {label} size does not match its archive entry")
        return bytes(data)


class _XLSXInputError(ValueError):
    """Expected XLSX validation failure with a safe, useful message."""


class ParagraphChunker:
    """Group paragraphs into bounded chunks and repeat a small tail across boundaries."""

    def __init__(self, *, max_chars: int = 2000, overlap_chars: int = 160) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise ValueError("max_chars must be a positive integer")
        if (
            isinstance(overlap_chars, bool)
            or not isinstance(overlap_chars, int)
            or overlap_chars < 0
            or overlap_chars >= max_chars
        ):
            raise ValueError("overlap_chars must be a non-negative integer smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> list[str]:
        """Split on paragraph boundaries, with deterministic splits for oversized paragraphs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
        chunks: list[str] = []
        pending = ""

        def flush() -> None:
            nonlocal pending
            if pending:
                chunks.append(pending)
                pending = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                flush()
                chunks.extend(self._split_long_paragraph(paragraph))
                continue
            candidate = f"{pending}\n\n{paragraph}" if pending else paragraph
            if pending and len(candidate) > self.max_chars:
                flush()
                pending = paragraph
            else:
                pending = candidate
        flush()

        if self.overlap_chars == 0 or len(chunks) < 2:
            return chunks
        overlapped = [chunks[0]]
        for previous, current in zip(chunks, chunks[1:], strict=False):
            tail = previous[-self.overlap_chars :].lstrip()
            overlapped.append(f"{tail}\n{current}" if tail else current)
        return overlapped

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        while start < len(paragraph):
            end = min(start + self.max_chars, len(paragraph))
            if end < len(paragraph):
                whitespace = paragraph.rfind(" ", start, end)
                if whitespace > start + self.max_chars // 2:
                    end = whitespace
            piece = paragraph[start:end].strip()
            if piece:
                pieces.append(piece)
            start = end
            while start < len(paragraph) and paragraph[start].isspace():
                start += 1
        return pieces


@dataclass(frozen=True)
class IngestionReport:
    """Summary from one file or directory ingestion operation."""

    source_count: int
    document_count: int
    sources: tuple[str, ...]


class FileIngestor:
    """Ingest supported files below one explicit root directory.

    A file is one replaceable source. Reingesting it atomically replaces all its chunks in
    the backing ``KnowledgeStore``. The caller owns the store, root path, and file retention.
    """

    supported_extensions = frozenset(
        {
            ".md",
            ".markdown",
            ".txt",
            ".adoc",
            ".asciidoc",
            ".rst",
            ".rtf",
            ".eml",
            ".mbox",
            ".ics",
            ".vcf",
            ".rss",
            ".atom",
            ".opml",
            ".html",
            ".htm",
            ".json",
            ".jsonl",
            ".toml",
            ".yaml",
            ".yml",
            ".ipynb",
            ".xml",
            ".csv",
            ".pdf",
            ".docx",
            ".odt",
            ".ods",
            ".xlsx",
            ".epub",
            *ImageOCRParser.extensions,
        }
    )

    def __init__(
        self,
        store: SourceKnowledgeWriter,
        root: str | Path,
        *,
        chunker: TextChunker | None = None,
        parsers: Sequence[FileParser] | None = None,
        max_file_bytes: int = 10 * 1024 * 1024,
        max_files: int = 10_000,
        max_total_bytes: int = 1024 * 1024 * 1024,
        max_pages: int = 1000,
        max_extracted_chars: int = 10_000_000,
        max_ocr_pages: int = 100,
        max_image_pixels: int = 40_000_000,
        ocr_timeout_seconds: float = 30.0,
        pdf_ocr_backend: OCRBackend | None = None,
        pdf_page_renderer: PDFPageRenderer | None = None,
        max_ocr_image_bytes: int = 16 * 1024 * 1024,
        max_content_stream_bytes: int = 4 * 1024 * 1024,
        max_total_content_stream_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        for name, value in (
            ("max_file_bytes", max_file_bytes),
            ("max_files", max_files),
            ("max_total_bytes", max_total_bytes),
            ("max_pages", max_pages),
            ("max_extracted_chars", max_extracted_chars),
            ("max_ocr_pages", max_ocr_pages),
            ("max_image_pixels", max_image_pixels),
            ("max_content_stream_bytes", max_content_stream_bytes),
            ("max_total_content_stream_bytes", max_total_content_stream_bytes),
            ("max_ocr_image_bytes", max_ocr_image_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a positive integer")
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(ocr_timeout_seconds, bool) or not isinstance(
            ocr_timeout_seconds, (int, float)
        ):
            raise TypeError("ocr_timeout_seconds must be a positive number")
        if not math.isfinite(ocr_timeout_seconds) or ocr_timeout_seconds <= 0:
            raise ValueError("ocr_timeout_seconds must be a positive number")
        resolved_root = Path(root).expanduser().resolve(strict=True)
        if not resolved_root.is_dir():
            raise ValueError("root must be an existing directory")
        self.store = store
        self.root = resolved_root
        self.chunker = chunker or ParagraphChunker()
        configured_parsers = (
            tuple(parsers)
            if parsers is not None
            else (
                Utf8TextParser(),
                MarkupTextParser(
                    max_input_bytes=max_file_bytes,
                    max_extracted_chars=max_extracted_chars,
                ),
                RTFTextParser(
                    max_input_bytes=max_file_bytes,
                    max_extracted_chars=max_extracted_chars,
                ),
                EmailTextParser(
                    max_input_bytes=max_file_bytes, max_extracted_chars=max_extracted_chars
                ),
                MboxTextParser(
                    max_input_bytes=max_file_bytes,
                    max_messages=min(max_pages, 100_000),
                    max_extracted_chars=max_extracted_chars,
                ),
                ICalendarTextParser(
                    max_input_bytes=max_file_bytes,
                    max_events=min(max_pages, 100_000),
                    max_extracted_chars=max_extracted_chars,
                ),
                VCardTextParser(
                    max_input_bytes=max_file_bytes,
                    max_cards=min(max_pages, 100_000),
                    max_extracted_chars=max_extracted_chars,
                ),
                HTMLTextParser(),
                JSONTextParser(
                    max_feed_items=min(max_pages, 10_000),
                    max_output_bytes=max_extracted_chars * 4,
                ),
                TOMLTextParser(
                    max_input_bytes=max_file_bytes,
                    max_output_bytes=max_extracted_chars,
                ),
                YAMLTextParser(
                    max_input_bytes=max_file_bytes,
                    max_output_bytes=max_extracted_chars,
                ),
                JSONLinesTextParser(),
                NotebookTextParser(
                    max_input_bytes=max_file_bytes,
                    max_cells=min(max_pages, 10_000),
                    max_output_bytes=max_extracted_chars,
                ),
                XMLTextParser(max_feed_items=min(max_pages, 10_000)),
                RSSAtomTextParser(
                    max_input_bytes=max_file_bytes,
                    max_output_bytes=max_extracted_chars * 4,
                    max_items=min(max_pages, 10_000),
                ),
                OPMLTextParser(
                    max_input_bytes=max_file_bytes,
                    max_output_bytes=max_extracted_chars * 4,
                    max_outlines=min(max_pages, 10_000),
                ),
                CSVTextParser(),
                ParquetTextParser(
                    max_input_bytes=max_file_bytes,
                    max_pages=max_pages,
                    max_output_bytes=max_extracted_chars * 4,
                ),
                PDFTextParser(
                    max_pages=max_pages,
                    max_extracted_chars=max_extracted_chars,
                    max_content_stream_bytes=max_content_stream_bytes,
                    max_total_content_stream_bytes=max_total_content_stream_bytes,
                    ocr_backend=pdf_ocr_backend,
                    page_renderer=pdf_page_renderer,
                    max_ocr_pages=max_ocr_pages,
                    max_image_pixels=max_image_pixels,
                    max_ocr_image_bytes=max_ocr_image_bytes,
                    ocr_timeout_seconds=ocr_timeout_seconds,
                ),
                DOCXTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                PPTXTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                ODPTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                EPUBTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                ODTTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                ODSTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                XLSXTextParser(
                    max_archive_bytes=min(max_file_bytes, 256 * 1024 * 1024),
                    max_extracted_chars=min(max_extracted_chars, 100_000_000),
                ),
                ImageOCRParser(
                    max_pages=max_ocr_pages,
                    max_image_pixels=max_image_pixels,
                    max_extracted_chars=max_extracted_chars,
                    timeout_seconds=ocr_timeout_seconds,
                ),
            )
        )
        self.parsers: dict[str, FileParser] = {}
        for parser in configured_parsers:
            extensions = getattr(parser, "extensions", None)
            parse = getattr(parser, "parse", None)
            if (
                not isinstance(extensions, frozenset)
                or not extensions
                or not callable(parse)
                or any(
                    not isinstance(extension, str)
                    or not extension.startswith(".")
                    or extension != extension.casefold()
                    for extension in extensions
                )
            ):
                raise TypeError("parsers must expose lowercase dotted extensions and parse(bytes)")
            for extension in extensions:
                if extension in self.parsers:
                    raise ValueError(f"multiple parsers claim extension {extension!r}")
                self.parsers[extension] = parser
        self.supported_extensions = frozenset(self.parsers)
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes
        self.max_pages = max_pages
        self.max_extracted_chars = max_extracted_chars
        self.max_ocr_pages = max_ocr_pages
        self.max_image_pixels = max_image_pixels
        self.ocr_timeout_seconds = float(ocr_timeout_seconds)
        self.max_content_stream_bytes = max_content_stream_bytes
        self.max_total_content_stream_bytes = max_total_content_stream_bytes

    async def ingest_file(
        self,
        relative_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Read, chunk, and atomically reindex one supported file under ``root``."""
        byte_limit = min(self.max_file_bytes, self.max_total_bytes)
        if byte_limit < self.max_file_bytes:
            byte_limit_error = f"ingestion exceeds max_total_bytes={self.max_total_bytes}"
        else:
            byte_limit_error = f"file exceeds max_file_bytes={self.max_file_bytes}"
        count, _ = await self._ingest_file(
            relative_path,
            metadata=metadata,
            byte_limit=byte_limit,
            byte_limit_error=byte_limit_error,
        )
        return count

    async def _ingest_file(
        self,
        relative_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None,
        byte_limit: int,
        byte_limit_error: str,
    ) -> tuple[int, int]:
        """Load one source while enforcing a caller-selected byte budget."""

        def resolve_supported_file() -> tuple[str, Path]:
            source, candidate = self._resolve_file(relative_path, must_exist=False)
            if candidate.suffix.casefold() not in self.parsers:
                raise ValueError(f"unsupported file extension: {candidate.suffix or '<none>'}")
            if not candidate.exists():
                raise ValueError("file path must name an existing regular file")
            return self._resolve_file(relative_path)

        source, path = await run_sync_callback(resolve_supported_file)
        file_metadata = self._validate_metadata(metadata)

        def load_documents() -> tuple[list[Document], int]:
            with path.open("rb") as stream:
                content = stream.read(byte_limit + 1)
            if len(content) > byte_limit:
                raise ValueError(byte_limit_error)
            parser = self.parsers[path.suffix.casefold()]
            pages = parser.parse(content)
            if len(pages) > self.max_pages:
                raise ValueError(f"parser returned too many pages (maximum {self.max_pages})")
            indexed_chunks: list[tuple[str, int | None, dict[str, Any]]] = []
            extracted_chars = 0
            for page in pages:
                if not isinstance(page, ParsedPage) or not isinstance(page.text, str):
                    raise ValueError("parser must return ParsedPage values with string text")
                extracted_chars += len(page.text)
                if extracted_chars > self.max_extracted_chars:
                    raise ValueError(
                        f"parser output exceeds max_extracted_chars={self.max_extracted_chars}"
                    )
                if page.page_number is not None and (
                    isinstance(page.page_number, bool)
                    or not isinstance(page.page_number, int)
                    or page.page_number < 1
                ):
                    raise ValueError("parser page numbers must be positive integers")
                page_metadata = self._validate_metadata(page.metadata)
                chunks = self.chunker.chunk(page.text)
                if any(not isinstance(chunk, str) or not chunk.strip() for chunk in chunks):
                    raise ValueError("chunker must return only non-empty strings")
                indexed_chunks.extend((chunk, page.page_number, page_metadata) for chunk in chunks)
            count = len(indexed_chunks)
            documents = []
            for index, (chunk, page_number, page_metadata) in enumerate(indexed_chunks):
                document_metadata = dict(page_metadata)
                document_metadata.update(file_metadata)
                document_metadata.update(chunk_index=index, chunk_count=count)
                if page_number is not None:
                    document_metadata["page_number"] = page_number
                document_id = hashlib.sha256(f"{source}\0{index}\0{chunk}".encode()).hexdigest()
                documents.append(
                    Document(
                        text=chunk,
                        source=source,
                        metadata=document_metadata,
                        id=document_id,
                    )
                )
            return documents, len(content)

        documents, byte_count = await run_sync_callback(load_documents)
        count = await self.store.replace_source(source, documents)
        return count, byte_count

    async def ingest_directory(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> IngestionReport:
        """Ingest supported files in lexical path order, one atomic source at a time."""
        self._validate_metadata(metadata)

        def discover_files() -> list[tuple[Path, int]]:
            files: list[tuple[Path, int]] = []
            total_bytes = 0
            for path in self.root.rglob("*"):
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.suffix.casefold() not in self.parsers
                ):
                    continue
                size = path.stat().st_size
                if size > self.max_file_bytes:
                    source = path.relative_to(self.root).as_posix()
                    raise ValueError(
                        f"file {source!r} exceeds max_file_bytes={self.max_file_bytes}"
                    )
                files.append((path, size))
                if len(files) > self.max_files:
                    raise ValueError(f"directory exceeds max_files={self.max_files}")
                total_bytes += size
                if total_bytes > self.max_total_bytes:
                    raise ValueError(f"directory exceeds max_total_bytes={self.max_total_bytes}")
            return sorted(
                files,
                key=lambda item: item[0].relative_to(self.root).as_posix(),
            )

        files = await run_sync_callback(discover_files)
        sources: list[str] = []
        document_count = 0
        actual_bytes = 0
        for path, _ in files:
            source = path.relative_to(self.root).as_posix()
            remaining_bytes = self.max_total_bytes - actual_bytes
            byte_limit = min(self.max_file_bytes, remaining_bytes)
            if byte_limit < self.max_file_bytes:
                byte_limit_error = (
                    f"directory ingestion exceeds max_total_bytes={self.max_total_bytes}"
                )
            else:
                byte_limit_error = f"file exceeds max_file_bytes={self.max_file_bytes}"
            count, actual_size = await self._ingest_file(
                source,
                metadata=metadata,
                byte_limit=byte_limit,
                byte_limit_error=byte_limit_error,
            )
            actual_bytes += actual_size
            sources.append(source)
            document_count += count
        return IngestionReport(len(sources), document_count, tuple(sources))

    async def delete_file(self, relative_path: str | Path) -> int:
        """Remove a file's indexed source, including chunks left after the file was deleted."""
        source, _ = await run_sync_callback(
            self._resolve_file,
            relative_path,
            must_exist=False,
        )
        return await self.store.delete_source(source)

    def _resolve_file(
        self, relative_path: str | Path, *, must_exist: bool = True
    ) -> tuple[str, Path]:
        raw = str(relative_path)
        windows_path = PureWindowsPath(raw)
        path = Path(relative_path)
        if (
            not raw
            or "\\" in raw
            or path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(part in ("", ".", "..") for part in path.parts)
            or not path.parts
        ):
            raise ValueError("file path must be a safe relative path under the configured root")
        candidate = self.root.joinpath(path)
        current = self.root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("file path must not traverse symbolic links")
        if not must_exist and not candidate.exists():
            return path.as_posix(), candidate
        try:
            resolved = candidate.resolve(strict=must_exist)
        except FileNotFoundError as exc:
            raise ValueError("file path must name an existing regular file") from exc
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("file path resolves outside the configured root") from exc
        if must_exist and not resolved.is_file():
            raise ValueError("file path must name a regular file")
        return path.as_posix(), resolved

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if metadata is None:
            return {}
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        if any(not isinstance(key, str) for key in metadata):
            raise TypeError("metadata keys must be strings")
        reserved = {"chunk_index", "chunk_count", "page_number"}
        if reserved.intersection(metadata):
            raise ValueError("chunk_index, chunk_count, and page_number are reserved metadata keys")
        return dict(metadata)


# Backward-compatible public name retained for existing text-only callers.
TextFileIngestor = FileIngestor

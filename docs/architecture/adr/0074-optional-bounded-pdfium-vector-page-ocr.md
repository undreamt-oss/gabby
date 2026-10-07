# ADR 0074: Optional bounded PDFium rendering for vector-only PDF OCR

- Status: Accepted under the architect's delegated implementation authority
- Date: 2026-10-02

## Context

The page-aware PDF parser can extract embedded raster images from scanned pages and pass them to a
host-selected OCR backend. Some PDFs contain text drawn as vector paths or other graphics and have
no embedded raster image, so the existing image-extraction path cannot provide searchable text.
Rasterizing every PDF page would add cost and alter ordinary text extraction behavior.

## Decision

- Keep OCR opt-in. Only blank-text pages are considered, and text-bearing pages continue to use the
  existing pypdf extraction path.
- When OCR is configured, prefer the first embedded image on a blank page. If there is no embedded
  image, rasterize that page with the optional PDFium adapter and pass the resulting JPEG bytes to
  the same injected `OCRBackend`.
- Define a replaceable `PDFPageRenderer` and run-scoped `PDFPageRenderSession` contract. Ship
  `PDFiumPageRenderer` through the existing `pdf-ocr` extra; do not add PDFium to core dependencies.
- Bound rendered and OCR image input to 16 MiB by default and share the existing per-run image-pixel,
  OCR-page, extracted-text, and OCR-engine timeout limits. Preserve the original one-based PDF page
  number as citation metadata.
- Rendering is synchronous native code and is not preemptible by the OCR timeout. Hosts processing
  hostile PDFs should isolate ingestion in a separate process or service.
- Install the optional renderer dependency in the Python/platform CI and release test environments.

## Consequences

Text-only PDF use remains unchanged and does not load PDFium. Configured OCR can now handle vector-
only pages without changing the OCR backend contract. The renderer is injectable for hosts that need
another library or execution boundary. The PDFium adapter bounds output dimensions and encoded image
bytes but cannot guarantee CPU or memory bounds for every malformed PDF; it runs in process and
must be isolated for stronger containment.

## Alternatives considered

- **Render every page:** rejected because it duplicates pypdf text extraction work and adds cost to
  documents whose text is already searchable.
- **Require a custom parser for vector-only pages:** rejected because this is a common PDF layout and
  a small optional renderer can reuse the existing parser and OCR contracts.
- **Make PDFium a core dependency:** rejected because OCR is optional and native PDFium binaries
  would unnecessarily enlarge the base installation.
- **Run rendering in a subprocess by default:** deferred; a cross-platform worker lifecycle and
  cancellation contract require separate design. The current synchronous renderer explicitly
  documents this limit.

## Compatibility and evidence

This is an additive pre-1.0 API feature. Tests cover injected renderer bounds and cleanup, a real
PDFium render of vector graphics, page citation preservation, and end-to-end SQLite ingestion. CI
and tagged-release quality jobs install the optional `pdf-ocr` extra. See [ADR 0032](0032-pluggable-page-aware-file-ingestion.md),
the [file-ingestion guide](../../../README.md#knowledge-retrieval), and the
[extension contract](../../EXTENSIONS.md).

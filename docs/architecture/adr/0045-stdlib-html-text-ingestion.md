# ADR 0045: Dependency-free bounded HTML text ingestion

**Status:** Accepted.  
**Date:** 2026-10-01.

## Context

The built-in `FileIngestor` handled Markdown, plain text, and optional PDF, leaving web exports and
saved help pages to custom parser implementations. HTML parsing is useful across research,
customer-support, and documentation agents. It should not add a third-party parser dependency to the
core package, and it must fit the existing file and extracted-text bounds.

## Decision

- Include `HTMLTextParser`, implemented with Python's standard-library `HTMLParser`, for UTF-8
  `.html` and `.htm` files.
- Return one source-ordered page and preserve the document title, headings, paragraphs, and common
  block boundaries for deterministic paragraph chunking.
- Omit content inside `head` except the title, plus script, style, noscript, template, SVG, canvas,
  iframe, object, HTML `hidden`, and `aria-hidden="true"` subtrees. Do not evaluate scripts, CSS, or
  remote resources.
- Keep existing root confinement, file-byte, total-import, page-count, and extracted-text limits.
  Reject invalid UTF-8 and NUL input using bounded, sanitized errors.
- Treat parsed content as untrusted retrieved text; HTML extraction is not a sanitization or
  instruction-security boundary.

## Consequences

Saved HTML help pages can be indexed with the default `FileIngestor` and no extra dependency.
Unsupported encodings, image-only content, CSS-based visibility, DOM rendering, and script-generated
text are not interpreted. Hosts can inject a specialized parser when those formats are required.

## Alternatives considered

- **Require a third-party DOM parser:** rejected for this initial parser because the common static
  text extraction path can use the standard library without widening the install surface.
- **Accept arbitrary HTML as plain text:** rejected because markup and executable source would pollute
  retrieval context.
- **Render pages with a browser:** deferred because it adds a browser runtime, network policy,
  resource bounds, and a separate sandboxing requirement.

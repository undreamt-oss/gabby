# ADR 0119: Bounded OPML ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

Research agents can ingest RSS, Atom, and JSON Feed publications. OPML is a common XML outline
format for exchanging subscription lists and nested directories. Gabby should index those source
lists without performing network requests as a side effect of ingestion.

## Decision

Add `OPMLTextParser` for `.opml` files and dispatch OPML roots from the existing `XMLTextParser`
for `.xml` files. Support OPML versions 1.0, 1.1, and 2.0; require one `head`, one `body`, and a
valid supported version. Emit one `ParsedPage` per outline in document order, retain ancestor
labels as category metadata, and expose safe feed, website, and link URLs as citations. Reject
DTD/entity declarations and enforce input, aggregate output, outline, element, depth, and attribute
bounds. URLs are never fetched or followed.

The parser follows the OPML 2.0 structure where an outline is a node in a tree and subscription
entries use attributes such as `type="rss"`, `text`, and `xmlUrl`. Unknown outline attributes are
ignored to preserve compatibility with extensions.

## Consequences

- Existing research ingestion can index portable subscription lists with filterable hierarchy.
- `.xml` parser selection remains unambiguous; root dispatch handles OPML.
- No network client or optional parser dependency is added.
- Every outline node is indexed, including folders, so folder labels remain retrievable.

## Alternatives considered

- Fetch each `xmlUrl` during ingestion: rejected because it would add network side effects and SSRF
  exposure to a file parser.
- Index only RSS leaves: rejected because OPML is a general outline format and folders carry useful
  classification labels.
- Register a second parser for `.xml`: rejected because extension-based parser selection must stay
  deterministic.

## Reference

- [OPML 2.0 specification](https://2005.opml.org/spec2.html)

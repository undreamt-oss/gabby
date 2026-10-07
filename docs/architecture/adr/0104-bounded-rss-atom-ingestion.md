# ADR 0104: Bounded RSS and Atom feed ingestion

## Status

Accepted

## Context

Research agents can ingest local documents and calendars, but external feed content required each
host to write its own RSS or Atom parser. Generic XML indexing preserves markup paths but does not
give each feed entry a useful citation or searchable metadata. Feed ingestion must not imply network
fetching, since Gabby's file ingestor is root-confined and does not own network policy.

## Decision

Add `RSSAtomTextParser` for RSS 1.0/2.0, RSS RDF, and Atom feeds. It emits one `ParsedPage` per item
or entry, with title, body, source link, published time, author, categories, feed title, and entry ID
where available. It supports `.rss` and `.atom`; `XMLTextParser` dispatches `.xml` documents with
RSS, RDF, or Atom root names to the same bounded parser. HTML summaries use the existing visible-text
parser. The implementation uses standard-library XML parsing and performs no network requests.

Input, output, XML element/depth/attribute, item, and category counts are bounded. UTF-8 is required;
DTD and entity declarations, malformed XML, and over-limit feeds are rejected. The host may tune
`RSSAtomTextParser` directly or set `XMLTextParser.max_feed_items`, and `FileIngestor` also applies
its existing page and extracted-character limits.

## Consequences

Saved feeds can be indexed locally as individually cited and filterable records without an optional
parser dependency. Feed URLs and item links remain data; ingestion does not fetch or follow them.
Hosts that need scheduled feed retrieval must perform it separately under their own network and
credential policies.

## Alternatives considered

- Index the feed as generic XML only. This loses the entry-level citations and feed metadata useful
  to research retrieval.
- Fetch URLs inside `FileIngestor`. This would expand the root-confined file ingestion feature into
  network access and mix parser behavior with the host's network policy.
- Add a third-party feed library. The supported RSS/Atom fields can be extracted with the existing
  bounded XML and HTML facilities, so a new runtime dependency is unnecessary.

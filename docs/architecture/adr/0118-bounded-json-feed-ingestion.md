# ADR 0118: Bounded JSON Feed ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

Research and news agents can already ingest RSS and Atom entries as separately cited pages.
JSON Feed is another syndication format and is commonly published with the `.json` extension,
which Gabby already uses for general JSON documents. Assigning a second parser to that extension
would make parser selection ambiguous.

## Decision

Keep `JSONTextParser` as the sole `.json` parser. After strict parsing, detect only the recognized
JSON Feed 1.0 and 1.1 version URLs. For those documents, validate the required item collection and
render each item as a `ParsedPage` with its one-based item number and safe citation metadata.
Unrecognized JSON continues to be returned as the original source text. Recognized but malformed
feeds fail validation rather than silently becoming raw JSON.

The parser bounds feed item count and aggregate rendered output, relies on the existing strict JSON
depth/token/duplicate-key limits, strips HTML item bodies to visible text, and retains only absolute
HTTP(S) item links without embedded credentials. It never fetches feed or item URLs.

## Consequences

- `.json` feed files are searchable per item and can be filtered using feed metadata.
- Ordinary JSON source behavior and parser registration remain unchanged.
- No network or optional parser dependency is introduced.
- JSON Feed extensions are ignored unless represented by standard fields used for indexing.
- Feed schemas can add optional fields; malformed required item fields are rejected.

## Alternatives considered

- Register another `.json` parser: rejected because extension dispatch would be ambiguous.
- Treat every JSON object with an `items` array as a feed: rejected because this would change
  unrelated application JSON ingestion.
- Fetch item links or feed pagination URLs: rejected because ingestion must not gain network
  access as a side effect of parsing.

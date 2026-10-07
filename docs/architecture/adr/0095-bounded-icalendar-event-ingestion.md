# ADR 0095: Bounded iCalendar event ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

Gabby's knowledge pipeline already ingests common text, office, tabular, and email documents, but
cannot index calendar schedules. Calendar files commonly contain many events, folded content lines,
escaped text, and nested alarm components. Indexing the source syntax as plain text would make event
retrieval noisy and would not provide stable metadata filters.

## Decision

Add a dependency-free `ICalendarTextParser` for UTF-8 `.ics` sources. Each `VEVENT` becomes a
source-ordered page with a one-based citation. The parser unfolds continuation lines, decodes the
RFC 5545 text escapes it indexes, and renders selected event fields. Start/end values remain in the
source's RFC 5545 representation to avoid timezone interpretation errors. UID, start/end, location,
organizer, attendees, and event ordinal are exposed as page metadata; file-level metadata retains its
existing precedence and Gabby's reserved page/chunk metadata remains authoritative.

The parser excludes nested `VALARM` and other subcomponent fields. It retains only indexed event
properties and limits attendees per event. It requires a single balanced `VCALENDAR` root and
direct-child `VEVENT` components. Input bytes, physical lines, event count, attendees per event, and
rendered characters are bounded. Defaults are 10 MiB, 100,000 lines, 1,000 events, 1,000 attendees
per event, and 10 million characters. Invalid UTF-8, malformed content lines, component mismatches,
and exceeded limits fail
the source ingestion instead of indexing partial calendar data.

## Consequences

- Calendar events can be searched independently and filtered by event metadata in supported stores.
- The implementation has no added dependency and uses the existing `FileParser` and `ParsedPage`
  contracts.
- Timezone conversion, recurrence expansion, attendee parameter interpretation, and VTODO/VJOURNAL
  indexing remain outside this parser. Repeating events are indexed as stored VEVENT components.
- A parser error aborts source replacement, preserving the previous atomic source index.

## Alternatives considered

- Treat `.ics` as ordinary text. Rejected because folded lines and component syntax make retrieval
  noisy and event fields are not available for metadata filters.
- Add a third-party calendar parser. Deferred because the bounded VEVENT subset fits the current
  dependency-light parser contract; broader recurrence and timezone semantics can be implemented as
  an optional parser later.

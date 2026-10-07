# ADR 0116: Bounded iCalendar journal ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

`ICalendarTextParser` supports events and task records. iCalendar `VJOURNAL` components hold
time-stamped journal text, which can provide useful source material for research and knowledge
agents but is currently ignored.

## Decision

Extend the parser to index direct-child `VJOURNAL` components as cited pages in source order. Render
summary, start time, description, status, organizer, and attendee values when present; expose journal
UID, start, status, attendees, and per-file journal ordinal through `calendar_journal_*` metadata.
Use an independent `max_journals` limit. Preserve the existing event and task behavior. Dates remain
in their original RFC 5545 representation, and nested subcomponents remain excluded.

## Consequences

- Journal entries become independently searchable and filterable in supported knowledge stores.
- Event and task metadata and page behavior remain unchanged.
- `VFREEBUSY`, recurrence expansion, timezone conversion, and attendee parameter interpretation
  remain outside the parser's contract.

## Alternatives considered

- Treat journal components as unknown data. Rejected because their bounded text fields fit the
  existing cited-page and metadata model.
- Add a third-party calendar library. Deferred because extracting these fields does not require
  recurrence or timezone interpretation.

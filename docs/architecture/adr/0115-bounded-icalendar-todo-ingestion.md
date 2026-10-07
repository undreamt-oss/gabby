# ADR 0115: Bounded iCalendar task ingestion

- Status: Accepted
- Date: 2026-10-03

## Context

`ICalendarTextParser` indexes calendar events as cited, filterable pages. iCalendar calendars also
commonly contain `VTODO` components for tasks, due dates, completion state, and priority. Treating
those components as unsupported leaves task information out of research and planning knowledge
sources.

## Decision

Extend the existing parser to recognize direct-child `VTODO` components alongside `VEVENT`. Preserve
the event rendering and event metadata contract. Render task summary, start, due, completion,
description, status, priority, and selected common fields; expose task identifiers and selected
properties under `calendar_task_*` metadata keys. Both record kinds receive one-based citations in
their source order, while each kind retains an independent bounded count (`max_events` and
`max_tasks`). Nested alarms and other subcomponents remain excluded, and dates remain in their
original RFC 5545 representation.

## Consequences

- Calendar tasks can be retrieved and filtered without requiring a separate parser or dependency.
- Event-only files preserve their page citations and metadata as before.
- Event and task pages in mixed files use source-order citations; their type-specific ordinals remain
  available in metadata.
- Recurrence expansion, timezone conversion, VJOURNAL, and attendee parameter interpretation remain
  outside the supported subset.

## Alternatives considered

- Keep `VEVENT` as the only supported component. Rejected because it omits a common first-class
  calendar record even though it uses the existing bounded parsing and page contracts.
- Add a third-party calendar library. Deferred because bounded `VTODO` extraction needs no
  recurrence or timezone interpretation and can reuse the existing parser's strict line handling.

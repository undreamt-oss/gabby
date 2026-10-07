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
"""Root-confined, deterministic text and PDF ingestion."""

from __future__ import annotations

import json
import sys
import zipfile
import zlib
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from gabby import (
    CSVTextParser,
    DOCXTextParser,
    EmailTextParser,
    EPUBTextParser,
    FileIngestor,
    HTMLTextParser,
    ICalendarTextParser,
    ImageOCRParser,
    JSONLinesTextParser,
    JSONTextParser,
    MarkupTextParser,
    MboxTextParser,
    NotebookTextParser,
    OCRPageResult,
    ODPTextParser,
    ODSTextParser,
    ODTTextParser,
    OPMLTextParser,
    ParagraphChunker,
    ParsedPage,
    PDFiumPageRenderer,
    PDFPageRenderSession,
    PDFRenderedPage,
    PDFTextParser,
    PPTXTextParser,
    RSSAtomTextParser,
    RTFTextParser,
    SQLiteFTS5Store,
    TesseractOCRBackend,
    TextFileIngestor,
    TOMLTextParser,
    Utf8TextParser,
    VCardTextParser,
    XLSXTextParser,
    XMLTextParser,
    YAMLTextParser,
)


def test_markup_parser_preserves_source_and_enforces_utf8_and_bounds() -> None:
    parser = MarkupTextParser(max_input_bytes=128, max_extracted_chars=100)
    source = "= Service Guide\n\nRotate signing keys every month.\ninclude::private.adoc[]\n"
    assert parser.parse(source.encode())[0].text == source
    assert parser.parse(b"\xef\xbb\xbfTitle")[0].text == "Title"
    with pytest.raises(ValueError, match="valid UTF-8"):
        parser.parse(b"\xff")
    with pytest.raises(ValueError, match="NUL"):
        parser.parse(b"a\x00b")
    with pytest.raises(ValueError, match="max_input_bytes"):
        parser.parse(b"x" * 129)
    with pytest.raises(ValueError, match="max_extracted_chars"):
        MarkupTextParser(max_extracted_chars=2).parse(b"long")


@pytest.mark.asyncio
async def test_file_ingestor_indexes_asciidoc_and_restructuredtext_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "markup"
    root.mkdir()
    (root / "operations.adoc").write_text(
        "= Operations Guide\n\nRotate signing keys monthly.\ninclude::private.adoc[]\n",
        encoding="utf-8",
    )
    (root / "recovery.rst").write_text(
        "Recovery Process\n=================\n\n"
        "Restore encrypted backups before reconnecting traffic.\n",
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("operations.adoc", "recovery.rst")
    assert {".adoc", ".asciidoc", ".rst"} <= ingestor.supported_extensions
    key_results = await store.retrieve("signing keys")
    assert key_results and "Rotate signing keys monthly." in key_results[0].text
    restore_results = await store.retrieve("encrypted backups")
    assert restore_results and restore_results[0].source == "recovery.rst"
    assert "include::private.adoc[]" in key_results[0].text


@pytest.mark.asyncio
async def test_file_ingestor_indexes_log_files_as_utf8_text(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    root.mkdir()
    (root / "service.log").write_text(
        "2026-10-03T10:15:00Z ERROR payment authorization failed\n",
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("service.log",)
    assert ".log" in ingestor.supported_extensions
    results = await store.retrieve("authorization failed")
    assert results and "ERROR payment authorization failed" in results[0].text


def test_icalendar_parser_unfolds_and_extracts_bounded_event_pages() -> None:
    content = (
        b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n"
        b"BEGIN:VEVENT\r\nUID:event-1\r\n"
        b"DTSTART;TZID=America/New_York:20261003T090000\r\n"
        b"DTEND:20261003T100000Z\r\n"
        b"SUMMARY:Architecture\\, \r\n review\r\n"
        b"DESCRIPTION:First line\\nSecond line\r\n"
        b'LOCATION;ALTREP="CID:room@example.test":Room 3\r\n'
        b'ORGANIZER;CN="Example, One":mailto:organizer@example.test\r\n'
        b'ATTENDEE;CN="Participant":mailto:person@example.test\r\n'
        b"BEGIN:VALARM\r\nDESCRIPTION:alarm text is excluded\r\nEND:VALARM\r\n"
        b"END:VEVENT\r\n"
        b"BEGIN:VEVENT\r\nUID:event-2\r\nSUMMARY:Second event\r\nEND:VEVENT\r\n"
        b"END:VCALENDAR\r\n"
    )

    pages = ICalendarTextParser().parse(content)

    assert len(pages) == 2
    assert pages[0].text == (
        "Summary: Architecture, review\n"
        "Start: 20261003T090000\n"
        "End: 20261003T100000Z\n"
        "Location: Room 3\n"
        "Description: First line\nSecond line\n"
        "Organizer: mailto:organizer@example.test\n"
        "Attendee: mailto:person@example.test"
    )
    assert "alarm text" not in pages[0].text
    assert pages[0].page_number == 1
    assert pages[0].metadata == {
        "calendar_event_index": 1,
        "calendar_uid": "event-1",
        "calendar_start": "20261003T090000",
        "calendar_end": "20261003T100000Z",
        "calendar_location": "Room 3",
        "calendar_organizer": "mailto:organizer@example.test",
        "calendar_attendees": ["mailto:person@example.test"],
    }
    assert pages[1].page_number == 2
    assert pages[1].metadata["calendar_uid"] == "event-2"
    assert ICalendarTextParser().parse(b"BEGIN:VCALENDAR\nEND:VCALENDAR\n") == ()


def test_icalendar_parser_indexes_vtodo_tasks_in_order_with_filterable_fields() -> None:
    content = (
        b"BEGIN:VCALENDAR\nVERSION:2.0\n"
        b"BEGIN:VEVENT\nUID:event-1\nSUMMARY:Planning\nEND:VEVENT\n"
        b"BEGIN:VTODO\nUID:task-1\nSUMMARY:Review\\, architecture\n"
        b"DUE;TZID=Europe/Paris:20261005T120000\nCOMPLETED:20261004T090000Z\n"
        b"STATUS:COMPLETED\nPRIORITY:2\nDESCRIPTION:Check\\naccepted design\n"
        b"BEGIN:VALARM\nDESCRIPTION:reminder is excluded\nEND:VALARM\nEND:VTODO\n"
        b"BEGIN:VTODO\nUID:task-2\nSUMMARY:Publish notes\nSTATUS:NEEDS-ACTION\nEND:VTODO\n"
        b"END:VCALENDAR\n"
    )

    pages = ICalendarTextParser().parse(content)

    assert len(pages) == 3
    assert pages[0].metadata == {"calendar_event_index": 1, "calendar_uid": "event-1"}
    assert pages[1].page_number == 2
    assert pages[1].text == (
        "Summary: Review, architecture\n"
        "Due: 20261005T120000\n"
        "Completed: 20261004T090000Z\n"
        "Description: Check\naccepted design\n"
        "Status: COMPLETED\n"
        "Priority: 2"
    )
    assert "reminder is excluded" not in pages[1].text
    assert pages[1].metadata == {
        "calendar_task_index": 1,
        "calendar_task_uid": "task-1",
        "calendar_task_due": "20261005T120000",
        "calendar_task_completed": "20261004T090000Z",
        "calendar_task_status": "COMPLETED",
        "calendar_task_priority": "2",
    }
    assert pages[2].page_number == 3
    assert pages[2].metadata["calendar_task_index"] == 2


def test_icalendar_parser_indexes_vjournal_entries_with_citations() -> None:
    pages = ICalendarTextParser().parse(
        b"BEGIN:VCALENDAR\nVERSION:2.0\n"
        b"BEGIN:VJOURNAL\nUID:journal-1\nDTSTART:20261003\n"
        b"SUMMARY:Field notes\nDESCRIPTION:Collected\\, observations\n"
        b"END:VJOURNAL\n"
        b"BEGIN:VEVENT\nUID:event-1\nSUMMARY:Review\nEND:VEVENT\n"
        b"END:VCALENDAR\n"
    )

    assert pages[0].page_number == 1
    assert (
        pages[0].text
        == "Summary: Field notes\nStart: 20261003\nDescription: Collected, observations"
    )
    assert pages[0].metadata == {
        "calendar_journal_index": 1,
        "calendar_journal_uid": "journal-1",
        "calendar_journal_start": "20261003",
    }
    assert pages[1].page_number == 2
    assert pages[1].metadata["calendar_event_index"] == 1


def test_vcard_parser_extracts_bounded_cited_contacts_and_filter_metadata() -> None:
    content = (
        b"BEGIN:VCARD\r\nVERSION:4.0\r\nUID:contact-1\r\n"
        b"FN:Jane Doe\r\nN:Doe;Jane;;;\r\n"
        b"ORG:Example\\, Inc.\r\nTITLE:Research Director\r\n"
        b"EMAIL;TYPE=work:jane@example.test\r\n"
        b"TEL;TYPE=work,voice:tel:+1-555-0100\r\n"
        b"NOTE:Coordinates regional\r\n research projects.\r\n"
        b"END:VCARD\r\n"
        b"BEGIN:VCARD\r\nVERSION:3.0\r\nFN:Alex Smith\r\n"
        b"EMAIL:alex@example.test\r\nEND:VCARD\r\n"
    )

    pages = VCardTextParser().parse(content)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Name: Jane Doe" in pages[0].text
    assert "Organization: Example, Inc." in pages[0].text
    assert "Note: Coordinates regionalresearch projects." in pages[0].text
    assert pages[0].metadata["vcard_uid"] == "contact-1"
    assert pages[0].metadata["vcard_emails"] == ["jane@example.test"]
    assert pages[0].metadata["vcard_telephones"] == ["tel:+1-555-0100"]
    assert pages[1].metadata["vcard_formatted_name"] == "Alex Smith"


def test_vcard_parser_rejects_unbounded_or_unsupported_contact_data() -> None:
    valid = b"BEGIN:VCARD\nVERSION:4.0\nFN:Example Person\nEND:VCARD\n"
    with pytest.raises(ValueError, match="max_input_bytes"):
        VCardTextParser(max_input_bytes=16).parse(valid)
    with pytest.raises(ValueError, match="max_cards"):
        VCardTextParser(max_cards=1).parse(valid + valid)
    with pytest.raises(ValueError, match="max_lines"):
        VCardTextParser(max_lines=3).parse(valid)
    with pytest.raises(ValueError, match="max_properties_per_card"):
        VCardTextParser(max_properties_per_card=1).parse(valid)
    with pytest.raises(ValueError, match="max_extracted_chars"):
        VCardTextParser(max_extracted_chars=3).parse(valid)
    with pytest.raises(ValueError, match="quoted-printable"):
        VCardTextParser().parse(
            b"BEGIN:VCARD\nVERSION:2.1\nFN;ENCODING=QUOTED-PRINTABLE:Jane=20Doe\nEND:VCARD\n"
        )
    with pytest.raises(ValueError, match="boundaries"):
        VCardTextParser().parse(b"BEGIN:VCARD\nVERSION:4.0\nFN:Incomplete\n")
    with pytest.raises(ValueError, match="valid UTF-8"):
        VCardTextParser().parse(b"BEGIN:VCARD\nVERSION:4.0\nFN:\xff\nEND:VCARD\n")


def test_icalendar_parser_rejects_malformed_or_over_limit_input() -> None:
    parser = ICalendarTextParser(max_events=1, max_lines=20, max_extracted_chars=12)
    invalid = (
        b"VERSION:2.0\nBEGIN:VCALENDAR\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\nEND:VCALENDAR\nVERSION:2.0\n",
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\nBEGIN:VTODO\nBEGIN:VEVENT\nEND:VEVENT\nEND:VTODO\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:x\nEND:VEVENT\n"
        b"BEGIN:VEVENT\nSUMMARY:y\nEND:VEVENT\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\rBROKEN\rEND:VCALENDAR",
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:long summary\nEND:VEVENT\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\n" + b"\n" * 20 + b"END:VCALENDAR\n",
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:\xff\nEND:VEVENT\nEND:VCALENDAR\n",
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:x\nEND:VCALENDAR\n",
    )
    for content in invalid:
        with pytest.raises(ValueError, match="iCalendar"):
            parser.parse(content)
    too_many_attendees = (
        b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nATTENDEE:mailto:a@example.test\n"
        b"ATTENDEE:mailto:b@example.test\nEND:VEVENT\nEND:VCALENDAR\n"
    )
    with pytest.raises(ValueError, match="max_attendees"):
        ICalendarTextParser(max_attendees=1).parse(too_many_attendees)
    with pytest.raises(ValueError, match="max_tasks"):
        ICalendarTextParser(max_tasks=1).parse(
            b"BEGIN:VCALENDAR\nBEGIN:VTODO\nSUMMARY:one\nEND:VTODO\n"
            b"BEGIN:VTODO\nSUMMARY:two\nEND:VTODO\nEND:VCALENDAR\n"
        )
    with pytest.raises(ValueError, match="max_journals"):
        ICalendarTextParser(max_journals=1).parse(
            b"BEGIN:VCALENDAR\nBEGIN:VJOURNAL\nSUMMARY:one\nEND:VJOURNAL\n"
            b"BEGIN:VJOURNAL\nSUMMARY:two\nEND:VJOURNAL\nEND:VCALENDAR\n"
        )


def test_icalendar_parser_rejects_uncovered_encoding_and_component_edges() -> None:
    parser = ICalendarTextParser()
    invalid_inputs: tuple[tuple[object, type[Exception], str], ...] = (
        ("not bytes", TypeError, "must be bytes"),
        (b"x\x00y", ValueError, "NUL"),
        (b"\xff", ValueError, "UTF-8"),
        (b" folded\nBEGIN:VCALENDAR\nEND:VCALENDAR\n", ValueError, "folded line"),
        (b"SUMMARY:outside\nBEGIN:VCALENDAR\nEND:VCALENDAR\n", ValueError, "outside"),
        (
            b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nBEGIN:VALARM\nBEGIN:VCALENDAR\n",
            ValueError,
            "nested VCALENDAR",
        ),
        (
            b"BEGIN:VCALENDAR\nBEGIN:VALARM\nBEGIN:VEVENT\n",
            ValueError,
            "direct child",
        ),
        (
            b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nBEGIN:VEVENT\n",
            ValueError,
            "direct child",
        ),
        (b"BEGIN:VCALENDAR\nEND:VEVENT\n", ValueError, "boundaries"),
        (b"BEGIN:VCALENDAR\nBEGIN:VUNKNOWN\nEND:VUNKNOWN\n", ValueError, "incomplete"),
    )
    for content, error, message in invalid_inputs:
        with pytest.raises(error, match=message):
            parser.parse(content)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_input_bytes"):
        ICalendarTextParser(max_input_bytes=8).parse(b"BEGIN:VCALENDAR")
    with pytest.raises(ValueError, match="max_lines"):
        ICalendarTextParser(max_lines=2).parse(b"BEGIN:VCALENDAR\nEND:VCALENDAR\n")
    with pytest.raises(ValueError, match="max_extracted_chars"):
        ICalendarTextParser(max_extracted_chars=1).parse(
            b"BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:long\nEND:VEVENT\nEND:VCALENDAR\n"
        )


@pytest.mark.asyncio
async def test_file_ingestor_indexes_icalendar_events_with_filterable_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "calendars"
    root.mkdir()
    (root / "events.ics").write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\nUID:planning-1\n"
        "DTSTART:20261003T090000Z\nSUMMARY:Planning review\nLOCATION:Room Alpha\n"
        "DESCRIPTION:Agenda includes deployment review\nEND:VEVENT\nEND:VCALENDAR\n",
        encoding="utf-8",
    )
    ingestor = FileIngestor(store, root)

    assert await ingestor.ingest_file("events.ics") == 1
    results = await store.retrieve("deployment review", filters={"calendar_uid": "planning-1"})
    assert len(results) == 1
    assert "Summary: Planning review" in results[0].text
    assert results[0].metadata["calendar_start"] == "20261003T090000Z"
    assert results[0].metadata["page_number"] == 1


@pytest.mark.asyncio
async def test_file_ingestor_indexes_icalendar_tasks_with_filterable_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "calendars"
    root.mkdir()
    (root / "tasks.ics").write_text(
        "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VTODO\nUID:task-deploy\n"
        "DUE:20261005T120000Z\nSUMMARY:Review deployment controls\n"
        "END:VTODO\nEND:VCALENDAR\n",
        encoding="utf-8",
    )

    assert await FileIngestor(store, root).ingest_file("tasks.ics") == 1
    results = await store.retrieve(
        "deployment controls", filters={"calendar_task_uid": "task-deploy"}
    )
    assert len(results) == 1
    assert "Summary: Review deployment controls" in results[0].text
    assert results[0].metadata["calendar_task_due"] == "20261005T120000Z"


@pytest.mark.asyncio
async def test_file_ingestor_indexes_vcard_contacts_with_filterable_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "contacts"
    root.mkdir()
    (root / "team.vcf").write_text(
        "BEGIN:VCARD\nVERSION:4.0\nFN:Jordan Lee\nORG:Field Research\n"
        "EMAIL;TYPE=work:jordan@example.test\nEND:VCARD\n",
        encoding="utf-8",
    )
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("team.vcf",)
    assert ".vcf" in ingestor.supported_extensions
    results = await store.retrieve(
        "Field Research", filters={"vcard_emails": ["jordan@example.test"]}
    )
    assert len(results) == 1
    assert results[0].metadata["page_number"] == 1
    assert results[0].metadata["vcard_formatted_name"] == "Jordan Lee"


def test_utf8_parser_extracts_bounded_yaml_front_matter() -> None:
    parsed = Utf8TextParser().parse(
        b"---\ntitle: Quarterly results\nlabels: [finance, internal]\nissued: 2026-10-03\n---\n"
        b"The report body is indexed.\n"
    )

    assert len(parsed) == 1
    assert parsed[0].text == "The report body is indexed.\n"
    assert parsed[0].metadata == {
        "title": "Quarterly results",
        "labels": ["finance", "internal"],
        "issued": "2026-10-03",
    }


def test_utf8_parser_keeps_plain_text_and_rejects_invalid_front_matter() -> None:
    parser = Utf8TextParser()
    assert parser.parse(b"---") == (ParsedPage("---"),)
    assert parser.parse(b"plain text\n---\nnot a front matter opener") == (
        ParsedPage("plain text\n---\nnot a front matter opener"),
    )

    invalid_documents = (
        b"---\ntitle: first\ntitle: second\n---\nbody",
        b"---\n- list\n---\nbody",
        b"---\nvalue: !!python/object/apply:os.system ['echo unsafe']\n---\nbody",
        b"---\ntitle: missing closer\nbody",
        b"---\nself: &cycle\n  child: *cycle\n---\nbody",
        b"---\ntitle: " + b"x" * (16 * 1024) + b"\n---\nbody",
    )
    for content in invalid_documents:
        with pytest.raises(ValueError, match="front matter"):
            parser.parse(content)


@pytest.mark.asyncio
async def test_file_ingestor_indexes_front_matter_as_filterable_metadata(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "documents"
    root.mkdir()
    (root / "report.md").write_text(
        "---\ntitle: Quarterly report\ndepartment: finance\nreviewed: true\n---\n"
        "Searchable body marker appears here.\n",
        encoding="utf-8",
    )
    ingestor = FileIngestor(store, root)

    assert await ingestor.ingest_file("report.md", metadata={"department": "legal"}) == 1
    results = await store.retrieve("Searchable body marker", filters={"department": "legal"})
    assert len(results) == 1
    assert "Quarterly report" not in results[0].text
    assert results[0].metadata["title"] == "Quarterly report"
    assert results[0].metadata["department"] == "legal"
    assert results[0].metadata["reviewed"] is True


def test_email_parser_extracts_selected_headers_and_plain_body_without_attachment_text() -> None:
    message = b"""From: Support <support@example.test>
To: Customer <customer@example.test>
Subject: =?utf-8?b?QWNjb3VudCByZWNvdmVyeQ==?=
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="outer"

--outer
Content-Type: multipart/alternative; boundary="alternative"

--alternative
Content-Type: text/plain; charset=utf-8
Content-Transfer-Encoding: quoted-printable

The recovery code is 123456.
--alternative
Content-Type: text/html; charset=utf-8

<p>The HTML copy must not be duplicated.</p>
--alternative--
--outer
Content-Type: text/plain
Content-Disposition: attachment; filename="private.txt"

Do not index this attachment.
--outer
Content-Type: message/rfc822
Content-Disposition: attachment; filename="forwarded.eml"

From: private@example.test
Subject: forwarded secret

The attached forwarded message must not be indexed.
--outer--
"""

    parsed = EmailTextParser().parse(message)

    assert len(parsed) == 1
    assert "From: Support <support@example.test>" in parsed[0].text
    assert "Subject: Account recovery" in parsed[0].text
    assert "The recovery code is 123456." in parsed[0].text
    assert "HTML copy" not in parsed[0].text
    assert "Do not index" not in parsed[0].text
    assert "forwarded secret" not in parsed[0].text
    assert "attached forwarded message" not in parsed[0].text


def test_email_parser_uses_html_title_and_visible_text_when_no_plain_text_exists() -> None:
    message = b"""Subject: HTML only
Content-Type: text/html; charset=utf-8

<html><head><title>hidden</title></head><body><p>Visible <b>email</b>.</p>
<script>secret()</script></body></html>
"""

    parsed = EmailTextParser().parse(message)

    assert "Visible email." in parsed[0].text
    assert "hidden" in parsed[0].text
    assert "secret" not in parsed[0].text


def test_mbox_parser_extracts_bounded_messages_and_preserves_envelope_framing() -> None:
    content = (
        b"From sender-one@example.test Sat Oct  3 10:00:00 2026\n"
        b"From: sender-one@example.test\nSubject: First support case\n"
        b"Content-Type: text/plain; charset=utf-8\n\n"
        b"Payment authorization failed.\n>From a quoted body line\n"
        b"From sender-two@example.test Sat Oct  3 10:05:00 2026\n"
        b"From: sender-two@example.test\nSubject: Second support case\n"
        b"MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n"
        b"--x\nContent-Type: text/plain\n\n"
        b"The replacement card fixed the issue.\n"
        b"--x\nContent-Type: text/plain\n"
        b"Content-Disposition: attachment; filename=private.txt\n\n"
        b"mbox attachment secret phrase\n--x--\n"
    )

    pages = MboxTextParser().parse(content)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].metadata["mailbox_message_index"] == 1
    assert "Payment authorization failed." in pages[0].text
    assert ">From a quoted body line" in pages[0].text
    assert "Subject: Second support case" in pages[1].text
    assert "mbox attachment secret" not in pages[1].text
    assert pages[1].metadata["mailbox_message_index"] == 2


def test_mbox_parser_rejects_invalid_or_over_limit_archives() -> None:
    valid = b"From sender@example.test Sat Oct  3 10:00:00 2026\nSubject: A valid message\n\nBody\n"
    for parser, content, message in (
        (MboxTextParser(max_input_bytes=16), valid, "max_input_bytes"),
        (MboxTextParser(max_lines=2), valid, "max_lines"),
        (MboxTextParser(max_messages=1), valid + valid, "max_messages"),
        (MboxTextParser(), b"not an mbox message\n", "before the first From"),
        (MboxTextParser(), b"From sender@example.test\n\n", "empty message"),
        (
            MboxTextParser(),
            b"From sender@example.test\nContent-Length: 12\n\nbody\n",
            "Content-Length framing",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            parser.parse(content)


@pytest.mark.parametrize(
    ("options", "message", "content"),
    [
        ({"max_input_bytes": 2}, "max_input_bytes=2", b"Subject: x\n\nbody"),
        (
            {"max_parts": 1},
            "max_parts=1",
            b"MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n--x\n\nbody\n--x--",
        ),
        (
            {"max_depth": 1},
            "max_depth=1",
            b"MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n"
            b"--x\nContent-Type: multipart/mixed; boundary=y\n\n--y\n\nbody\n--y--\n--x--",
        ),
        ({"max_extracted_chars": 5}, "max_extracted_chars=5", b"Subject: too long\n\nbody"),
    ],
)
def test_email_parser_enforces_input_structure_and_output_limits(
    options: dict[str, int], message: str, content: bytes
) -> None:
    with pytest.raises(ValueError, match=message):
        EmailTextParser(**options).parse(content)


@pytest.mark.asyncio
async def test_file_ingestor_indexes_eml_and_excludes_attachment_text(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "mail"
    root.mkdir()
    (root / "case.eml").write_bytes(
        b"Subject: Travel refund\nMIME-Version: 1.0\n"
        b"Content-Type: multipart/mixed; boundary=x\n\n"
        b"--x\nContent-Type: text/plain; charset=utf-8\n\n"
        b"The airline refund is pending.\n"
        b"--x\nContent-Type: text/plain\nContent-Disposition: attachment; filename=secret.txt\n\n"
        b"attachment-only-private phrase\n--x--\n"
    )
    ingestor = FileIngestor(store, root)

    assert ".eml" in ingestor.supported_extensions
    assert await ingestor.ingest_file("case.eml") >= 1
    found = await store.retrieve("airline refund", limit=5)
    assert found
    assert all("attachment-only-private" not in document.text for document in found)


@pytest.mark.asyncio
async def test_file_ingestor_indexes_mbox_messages_with_exact_citations(tmp_path: Path) -> None:
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    root = tmp_path / "mailbox"
    root.mkdir()
    (root / "support.mbox").write_bytes(
        b"From agent@example.test Sat Oct  3 10:00:00 2026\n"
        b"From: agent@example.test\nSubject: Payment review\n\n"
        b"Authorization is still pending.\n"
        b"From agent@example.test Sat Oct  3 10:05:00 2026\n"
        b"From: agent@example.test\nSubject: Payment resolved\n\n"
        b"The issuing bank released the payment.\n"
    )
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("support.mbox",)
    assert ".mbox" in ingestor.supported_extensions
    results = await store.retrieve("issuing bank", filters={"mailbox_message_index": 2})
    assert len(results) == 1
    assert results[0].metadata["page_number"] == 2
    assert "Subject: Payment resolved" in results[0].text


def test_email_parser_rejects_invalid_types_and_nul_text() -> None:
    parser = EmailTextParser()
    with pytest.raises(TypeError, match="content must be bytes"):
        parser.parse("not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="NUL"):
        parser.parse(b"Subject: unsafe\n\nbody\x00text")


def _pdf_with_text_pages(*texts: str) -> bytes:
    """Build a small PDF fixture with extractable Helvetica text on each page."""
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(texts)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(texts)} >>".encode(),
    ]
    font_object = 3 + len(texts) * 2
    for index, text in enumerate(texts):
        page_object = 3 + index * 2
        content_object = page_object + 1
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
        objects.extend(
            [
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
                    f"/Contents {content_object} 0 R >>"
                ).encode(),
                f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
            ]
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref_offset}\n%%EOF\n"
    output.extend(trailer.encode())
    return bytes(output)


def _pdf_with_flate_content(*contents: bytes) -> bytes:
    """Build a one-page PDF with one or more Flate-compressed content streams."""
    first_stream_object = 4
    content_references = " ".join(
        f"{first_stream_object + index} 0 R" for index in range(len(contents))
    )
    font_object = first_stream_object + len(contents)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
            f"/Contents [{content_references}] >>"
        ).encode(),
    ]
    for content in contents:
        compressed = zlib.compress(content)
        objects.append(
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode()
            + compressed
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Root 1 0 R /Size {len(offsets)} >>\nstartxref\n{xref_offset}\n%%EOF\n"
    output.extend(trailer.encode())
    return bytes(output)


class _FakeOCRBackend:
    def __init__(self, pages: tuple[str, ...]) -> None:
        self.pages = pages
        self.options: dict[str, object] = {}

    def recognize(
        self,
        content: bytes,
        *,
        max_pages: int,
        max_image_pixels: int,
        timeout_seconds: float,
    ) -> tuple[OCRPageResult, ...]:
        self.options = {
            "content": content,
            "max_pages": max_pages,
            "max_image_pixels": max_image_pixels,
            "timeout_seconds": timeout_seconds,
        }
        return tuple(OCRPageResult(text, 100) for text in self.pages)


class _FakePDFPageRenderSession:
    def __init__(self, output: PDFRenderedPage | None = None) -> None:
        self.output = output or PDFRenderedPage(b"rendered", 100)
        self.closed = False
        self.page_numbers: list[int] = []
        self.options: dict[str, int] = {}

    def render_page(
        self, page_number: int, *, max_image_pixels: int, max_image_bytes: int
    ) -> PDFRenderedPage:
        self.page_numbers.append(page_number)
        self.options = {
            "max_image_pixels": max_image_pixels,
            "max_image_bytes": max_image_bytes,
        }
        return self.output

    def close(self) -> None:
        self.closed = True


class _FakePDFPageRenderer:
    def __init__(self, session: _FakePDFPageRenderSession | None = None) -> None:
        self.session = session or _FakePDFPageRenderSession()
        self.content: bytes | None = None

    def open_document(self, content: bytes) -> PDFPageRenderSession:
        self.content = content
        return self.session


def test_image_ocr_parser_returns_bounded_page_attributed_text() -> None:
    backend = _FakeOCRBackend(("Scanned title", "Page two text"))
    parser = ImageOCRParser(
        backend=backend,
        max_pages=3,
        max_image_pixels=1234,
        max_extracted_chars=32,
        timeout_seconds=4.5,
    )

    pages = parser.parse(b"opaque-image-data")

    assert pages == [ParsedPage("Scanned title", 1), ParsedPage("Page two text", 2)]
    assert backend.options == {
        "content": b"opaque-image-data",
        "max_pages": 3,
        "max_image_pixels": 1234,
        "timeout_seconds": 4.5,
    }


@pytest.mark.parametrize("timeout_seconds", [float("nan"), float("inf")])
def test_image_ocr_parser_rejects_non_finite_timeout(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        ImageOCRParser(backend=_FakeOCRBackend(("text",)), timeout_seconds=timeout_seconds)


@pytest.mark.parametrize(
    ("pages", "max_chars", "message"),
    [
        (("page one", "page two"), 10, "max_extracted_chars=10"),
        (("page\x00text",), 20, "without NUL"),
        (("valid", "excess"), 1, "max_extracted_chars=1"),
    ],
)
def test_image_ocr_parser_validates_backend_output(
    pages: tuple[str, ...], max_chars: int, message: str
) -> None:
    parser = ImageOCRParser(backend=_FakeOCRBackend(pages), max_extracted_chars=max_chars)

    with pytest.raises(ValueError, match=message):
        parser.parse(b"opaque-image-data")


def test_tesseract_backend_bounds_pixels_and_closes_converted_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RGBFrame:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Image:
        n_frames = 2
        size = (10, 10)
        frame_index = 0

        def __enter__(self) -> Image:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def seek(self, index: int) -> None:
            self.frame_index = index

        def load(self) -> None:
            return None

        def convert(self, mode: str) -> RGBFrame:
            assert mode == "RGB"
            frame = RGBFrame()
            rgb_frames.append(frame)
            return frame

    image = Image()
    rgb_frames: list[RGBFrame] = []
    calls: list[tuple[RGBFrame, str, float]] = []
    pil = ModuleType("PIL")
    pil_image = ModuleType("PIL.Image")
    pil_image.__dict__["open"] = lambda _content: image
    pil.__dict__.update(
        {
            "Image": pil_image,
            "UnidentifiedImageError": type("UnidentifiedImageError", (Exception,), {}),
        }
    )
    tesseract = ModuleType("pytesseract")

    def image_to_string(frame: RGBFrame, lang: str, timeout: float) -> str:
        calls.append((frame, lang, timeout))
        return "text"

    tesseract.__dict__["image_to_string"] = image_to_string
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "pytesseract", tesseract)
    backend = TesseractOCRBackend(language="afr")

    assert backend.recognize(
        b"image",
        max_pages=2,
        max_image_pixels=200,
        timeout_seconds=3.0,
    ) == [OCRPageResult("text", 100), OCRPageResult("text", 100)]
    assert calls == [(rgb_frames[0], "afr", 3.0), (rgb_frames[1], "afr", 3.0)]
    assert all(frame.closed for frame in rgb_frames)

    with pytest.raises(ValueError, match="Tesseract language code list"):
        TesseractOCRBackend(language="../../private")

    image.frame_index = 0
    with pytest.raises(ValueError, match="max_image_pixels=199"):
        backend.recognize(
            b"image",
            max_pages=2,
            max_image_pixels=199,
            timeout_seconds=3.0,
        )


@pytest.mark.asyncio
async def test_image_ocr_ingestion_preserves_page_citations(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "scan.png").write_bytes(b"opaque-image-data")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    parser = ImageOCRParser(backend=_FakeOCRBackend(("First page invoice", "Second page total")))
    ingestor = FileIngestor(store, root, parsers=(parser,))

    assert await ingestor.ingest_file("scan.png") == 2
    first = await store.retrieve("invoice")
    second = await store.retrieve("total")
    assert first[0].metadata["page_number"] == 1
    assert second[0].metadata["page_number"] == 2


def _docx_package(
    document_xml: bytes, *, extra_entries: tuple[tuple[str, bytes], ...] = ()
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            b'<Override PartName="/word/document.xml" '
            b'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            b"</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rId1" '
            b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            b'officeDocument" '
            b'Target="word/document.xml"/>'
            b"</Relationships>",
        )
        archive.writestr("word/document.xml", document_xml)
        for name, content in extra_entries:
            archive.writestr(name, content)
    return output.getvalue()


def _docx_xml(*paragraphs: str) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    return (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    ).encode()


def _pptx_package(
    slides: tuple[bytes, ...], *, extra_entries: tuple[tuple[str, bytes], ...] = ()
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        slide_overrides = b"".join(
            (
                f'<Override PartName="/ppt/slides/slide{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            ).encode()
            for index in range(1, len(slides) + 1)
        )
        archive.writestr(
            "[Content_Types].xml",
            b'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            b'<Override PartName="/ppt/presentation.xml" '
            b'ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
            + slide_overrides
            + b"</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rIdOffice" '
            b'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
            b'relationships/officeDocument" '
            b'Target="ppt/presentation.xml"/></Relationships>',
        )
        relationships = b"".join(
            (
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" '
                f'Target="slides/slide{index}.xml"/>'
            ).encode()
            for index in range(1, len(slides) + 1)
        )
        archive.writestr(
            "ppt/_rels/presentation.xml.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + relationships
            + b"</Relationships>",
        )
        slide_ids = b"".join(
            f'<p:sldId id="{255 + index}" r:id="rId{index}"/>'.encode()
            for index in range(1, len(slides) + 1)
        )
        archive.writestr(
            "ppt/presentation.xml",
            b'<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
            b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            b"<p:sldIdLst>" + slide_ids + b"</p:sldIdLst></p:presentation>",
        )
        for index, slide in enumerate(slides, start=1):
            archive.writestr(f"ppt/slides/slide{index}.xml", slide)
        for name, content in extra_entries:
            archive.writestr(name, content)
    return output.getvalue()


def _pptx_slide_xml(*paragraphs: str) -> bytes:
    text_paragraphs = "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)
    return (
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        f"<p:cSld><p:spTree>{text_paragraphs}</p:spTree></p:cSld></p:sld>"
    ).encode()


def _odp_package(
    slides: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    extra_entries: tuple[tuple[str, bytes], ...] = (),
    mimetype: bytes = b"application/vnd.oasis.opendocument.presentation",
) -> bytes:
    output = BytesIO()
    paragraphs = "".join(
        f'<draw:page draw:name="{name}">'
        + "".join(f"<text:p>{text}</text:p>" for text in slide_paragraphs)
        + "</draw:page>"
        for name, slide_paragraphs in slides
    )
    content_xml = (
        "<office:document-content "
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        f"<office:body><office:presentation>{paragraphs}</office:presentation>"
        "</office:body></office:document-content>"
    ).encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        mime_info = zipfile.ZipInfo("mimetype")
        mime_info.compress_type = zipfile.ZIP_STORED
        archive.writestr(mime_info, mimetype)
        archive.writestr("content.xml", content_xml)
        for name, content in extra_entries:
            archive.writestr(name, content)
    return output.getvalue()


def _epub_package(
    chapters: tuple[tuple[str, bytes], ...],
    *,
    hrefs: tuple[str, ...] | None = None,
    extra_entries: tuple[tuple[str, bytes], ...] = (),
) -> bytes:
    output = BytesIO()
    chapter_hrefs = hrefs or tuple(
        f"text/chapter{index}.xhtml" for index in range(1, len(chapters) + 1)
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        mime_info = zipfile.ZipInfo("mimetype")
        mime_info.compress_type = zipfile.ZIP_STORED
        archive.writestr(mime_info, b"application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            b'<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            b'<rootfiles><rootfile full-path="OEBPS/content.opf" '
            b'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        manifest_items = b"".join(
            (
                f'<item id="chapter{index}" href="{href}" media-type="application/xhtml+xml"/>'
            ).encode()
            for index, href in enumerate(chapter_hrefs, start=1)
        )
        spine_items = b"".join(
            f'<itemref idref="chapter{index}"/>'.encode() for index in range(1, len(chapters) + 1)
        )
        archive.writestr(
            "OEBPS/content.opf",
            b'<package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            b"<manifest>"
            + manifest_items
            + b"</manifest><spine>"
            + spine_items
            + b"</spine></package>",
        )
        for index, (_name, content) in enumerate(chapters, start=1):
            archive.writestr(f"OEBPS/text/chapter{index}.xhtml", content)
        for name, content in extra_entries:
            archive.writestr(name, content)
    return output.getvalue()


def _odt_package(content_xml: bytes, *, extra_entries: tuple[tuple[str, bytes], ...] = ()) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        mimetype_info = zipfile.ZipInfo("mimetype")
        mimetype_info.compress_type = zipfile.ZIP_STORED
        archive.writestr(mimetype_info, b"application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", content_xml)
        for name, value in extra_entries:
            archive.writestr(name, value)
    return output.getvalue()


def _odt_xml(body: str) -> bytes:
    return (
        "<office:document-content "
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">'
        f"<office:body><office:text>{body}</office:text></office:body></office:document-content>"
    ).encode()


def _ods_package(content_xml: bytes, *, extra_entries: tuple[tuple[str, bytes], ...] = ()) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        mimetype_info = zipfile.ZipInfo("mimetype")
        mimetype_info.compress_type = zipfile.ZIP_STORED
        archive.writestr(mimetype_info, b"application/vnd.oasis.opendocument.spreadsheet")
        archive.writestr("content.xml", content_xml)
        for name, value in extra_entries:
            archive.writestr(name, value)
    return output.getvalue()


def _ods_xml(body: str) -> bytes:
    return (
        "<office:document-content "
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">'
        f"<office:body><office:spreadsheet>{body}</office:spreadsheet>"
        "</office:body></office:document-content>"
    ).encode()


def _xlsx_package(
    worksheets: tuple[tuple[str, bytes], ...],
    *,
    shared_strings: tuple[str, ...] = (),
    first_target: str = "worksheets/sheet1.xml",
    extra_entries: tuple[tuple[str, bytes], ...] = (),
) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "_rels/.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            b'<Relationship Id="rIdWorkbook" '
            b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            b'officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        sheet_tags = b"".join(
            f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'.encode()
            for index, (name, _xml) in enumerate(worksheets, start=1)
        )
        archive.writestr(
            "xl/workbook.xml",
            b'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            b'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            b"<sheets>" + sheet_tags + b"</sheets></workbook>",
        )
        relationships = b"".join(
            (
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/worksheet" '
                f'Target="{first_target if index == 1 else f"worksheets/sheet{index}.xml"}"/>'
            ).encode()
            for index in range(1, len(worksheets) + 1)
        )
        if shared_strings:
            relationships += (
                b'<Relationship Id="rIdShared" '
                b'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
                b'sharedStrings" Target="sharedStrings.xml"/>'
            )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            b'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + relationships
            + b"</Relationships>",
        )
        if shared_strings:
            strings = b"".join(f"<si>{value}</si>".encode() for value in shared_strings)
            archive.writestr(
                "xl/sharedStrings.xml",
                b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                + strings
                + b"</sst>",
            )
        for index, (_name, worksheet_xml) in enumerate(worksheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet_xml)
        for name, value in extra_entries:
            archive.writestr(name, value)
    return output.getvalue()


def _xlsx_worksheet(sheet_data: str) -> bytes:
    return (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{sheet_data}</sheetData></worksheet>"
    ).encode()


def test_paragraph_chunker_groups_paragraphs_with_deterministic_overlap() -> None:
    chunker = ParagraphChunker(max_chars=10, overlap_chars=3)

    assert chunker.chunk("  alpha\r\n\r\nbeta\n\ncharlie  ") == [
        "alpha",
        "pha\nbeta",
        "eta\ncharlie",
    ]


def test_paragraph_chunker_splits_long_paragraph_and_handles_empty_text() -> None:
    chunker = ParagraphChunker(max_chars=8, overlap_chars=0)

    assert chunker.chunk("abcdefghijklmno") == ["abcdefgh", "ijklmno"]
    assert chunker.chunk(" \n\n ") == []


def test_pdf_parser_loads_optional_dependency_only_when_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(RuntimeError, match=r"install gabby-agent-runtime\[pdf\]"):
        PDFTextParser().parse(b"%PDF-1.4")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_pages": 0}, ValueError),
        ({"max_pages": True}, TypeError),
        ({"max_extracted_chars": 1.5}, TypeError),
        ({"max_content_stream_bytes": 0}, ValueError),
        ({"max_content_stream_bytes": True}, TypeError),
        ({"max_total_content_stream_bytes": 0}, ValueError),
        ({"max_total_content_stream_bytes": True}, TypeError),
        ({"max_ocr_pages": 0}, ValueError),
        ({"max_image_pixels": True}, TypeError),
        ({"ocr_timeout_seconds": float("nan")}, ValueError),
    ],
)
def test_pdf_parser_validates_limits(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        PDFTextParser(**kwargs)  # type: ignore[arg-type]


def test_pdf_parser_sanitizes_malformed_pdf_errors() -> None:
    with pytest.raises(ValueError, match="could not parse or extract text from PDF"):
        PDFTextParser().parse(b"%PDF-malformed")


def test_pdf_parser_ocr_fills_only_blank_pages_and_preserves_page_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    class Page:
        def __init__(self, text: str, image: bytes | None = None) -> None:
            self.text = text
            self.images = [SimpleNamespace(data=image)] if image is not None else []

        def __contains__(self, key: str) -> bool:
            return False

        def extract_text(self) -> str:
            return self.text

    class Reader:
        is_encrypted = False
        pages = [
            Page("", b"scanned-page-image"),
            Page("Existing searchable text", b"decorative-image"),
            Page("", b"second-scanned-page-image"),
        ]

        def __init__(self, *_: object, **__: object) -> None:
            return None

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    backend = _FakeOCRBackend(("Recognized invoice",))
    parser = PDFTextParser(
        ocr_backend=backend,
        max_image_pixels=200,
        max_ocr_pages=1,
    )

    pages = parser.parse(b"%PDF-1.4")

    assert pages == [
        ParsedPage("Recognized invoice", 1),
        ParsedPage("Existing searchable text", 2),
        ParsedPage("", 3),
    ]
    assert backend.options == {
        "content": b"scanned-page-image",
        "max_pages": 1,
        "max_image_pixels": 200,
        "timeout_seconds": 30.0,
    }


def test_pdf_parser_enforces_aggregate_ocr_pixel_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    class Page:
        images = [SimpleNamespace(data=b"image")]

        def __contains__(self, key: str) -> bool:
            return False

        def extract_text(self) -> str:
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page(), Page()]

        def __init__(self, *_: object, **__: object) -> None:
            return None

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    parser = PDFTextParser(
        ocr_backend=_FakeOCRBackend(("recognized",)),
        max_image_pixels=150,
    )

    with pytest.raises(ValueError, match="max_image_pixels=150"):
        parser.parse(b"%PDF-1.4")


def test_pdf_parser_rasterizes_vector_only_pages_with_bounded_page_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    class Page:
        images: list[object] = []

        def __contains__(self, key: str) -> bool:
            return False

        def extract_text(self) -> str:
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page(), Page()]

        def __init__(self, *_: object, **__: object) -> None:
            return None

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    backend = _FakeOCRBackend(("vector label",))
    session = _FakePDFPageRenderSession()
    renderer = _FakePDFPageRenderer(session)
    parser = PDFTextParser(
        ocr_backend=backend,
        page_renderer=renderer,
        max_ocr_pages=1,
        max_image_pixels=200,
    )

    pages = parser.parse(b"%PDF-1.4")

    assert pages == [ParsedPage("vector label", 1), ParsedPage("", 2)]
    assert renderer.content == b"%PDF-1.4"
    assert session.page_numbers == [1]
    assert session.options == {
        "max_image_pixels": 200,
        "max_image_bytes": 16 * 1024 * 1024,
    }
    assert session.closed
    assert backend.options["content"] == b"rendered"


def test_pdf_parser_rejects_oversized_renderer_results_and_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    class Page:
        images: list[object] = []

        def __contains__(self, key: str) -> bool:
            return False

        def extract_text(self) -> str:
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, *_: object, **__: object) -> None:
            return None

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    session = _FakePDFPageRenderSession(PDFRenderedPage(b"oversized", 100))
    parser = PDFTextParser(
        ocr_backend=_FakeOCRBackend(("text",)),
        page_renderer=_FakePDFPageRenderer(session),
        max_ocr_image_bytes=4,
    )

    with pytest.raises(ValueError, match="invalid or oversized image"):
        parser.parse(b"%PDF-1.4")
    assert session.closed


def test_pdfium_renderer_bounds_pixel_count_and_closes_native_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RenderedImage:
        def convert(self, mode: str) -> RenderedImage:
            assert mode == "RGB"
            return self

        def save(self, output: BytesIO, **_: object) -> None:
            output.write(b"jpeg")

        def close(self) -> None:
            return None

    class Bitmap:
        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height
            self.closed = False

        def to_pil(self) -> RenderedImage:
            return RenderedImage()

        def close(self) -> None:
            self.closed = True

    class Page:
        closed = False

        def get_size(self) -> tuple[int, int]:
            return (1000, 1000)

        def render(self, *, scale: float, fill_color: tuple[int, ...]) -> Bitmap:
            assert fill_color == (255, 255, 255, 255)
            self.bitmap = Bitmap(int(1000 * scale), int(1000 * scale))
            return self.bitmap

        def close(self) -> None:
            self.closed = True

    class Document:
        def __init__(self, _: bytes) -> None:
            self.page = Page()
            self.closed = False

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> Page:
            assert index == 0
            return self.page

        def close(self) -> None:
            self.closed = True

    document = Document(b"%PDF-1.4")
    pdfium = ModuleType("pypdfium2")
    pdfium.PdfDocument = lambda _: document  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdfium2", pdfium)
    renderer = PDFiumPageRenderer()
    session = renderer.open_document(b"%PDF-1.4")
    result = session.render_page(1, max_image_pixels=10_000, max_image_bytes=8)

    assert result == PDFRenderedPage(b"jpeg", 9_801)
    assert document.page.bitmap.width * document.page.bitmap.height <= 10_000
    with pytest.raises(ValueError, match="max_ocr_image_bytes=3"):
        session.render_page(1, max_image_pixels=10_000, max_image_bytes=3)
    session.close()
    assert document.page.closed
    assert document.page.bitmap.closed
    assert document.closed


def test_pdf_ocr_rasterizes_real_vector_graphics_when_pdfium_is_installed() -> None:
    pytest.importorskip("pypdfium2")
    from PIL import Image

    class RasterAwareOCR:
        def recognize(
            self,
            content: bytes,
            *,
            max_pages: int,
            max_image_pixels: int,
            timeout_seconds: float,
        ) -> tuple[OCRPageResult, ...]:
            assert max_pages == 1
            assert timeout_seconds > 0
            with Image.open(BytesIO(content)) as image:
                pixels = image.width * image.height
            assert pixels <= max_image_pixels
            return (OCRPageResult("vector drawing recognized", pixels),)

    vector_pdf = _pdf_with_flate_content(b"1 0 0 rg 72 700 400 40 re f")
    pages = PDFTextParser(
        ocr_backend=RasterAwareOCR(),
        max_image_pixels=1_000_000,
    ).parse(vector_pdf)

    assert pages == [ParsedPage("vector drawing recognized", 1)]


def test_pdf_parser_bounds_decoded_content_stream_and_restores_configuration() -> None:
    parser = PDFTextParser(max_content_stream_bytes=128)
    oversized = _pdf_with_flate_content(b"BT /F1 12 Tf 72 720 Td (" + b"A" * 2048 + b") Tj ET")

    with pytest.raises(ValueError, match="max_content_stream_bytes=128"):
        parser.parse(oversized)

    aggregate_parser = PDFTextParser(
        max_content_stream_bytes=2048,
        max_total_content_stream_bytes=2048,
    )
    per_stream_bounded = _pdf_with_flate_content(
        b"BT /F1 12 Tf 72 720 Td (" + b"A" * 1100 + b") Tj ET",
        b"BT /F1 12 Tf 72 700 Td (" + b"B" * 1100 + b") Tj ET",
    )
    with pytest.raises(ValueError, match="max_total_content_stream_bytes=2048"):
        aggregate_parser.parse(per_stream_bounded)

    assert PDFTextParser().parse(_pdf_with_text_pages("normal document"))[0].text == (
        "normal document"
    )


def test_docx_parser_extracts_ordered_paragraphs_tables_tabs_and_breaks() -> None:
    xml = (
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>Heading</w:t><w:tab/><w:t>One</w:t>"
        b"<w:br/><w:t>Two</w:t></w:r></w:p>"
        b"<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell A</w:t></w:r></w:p></w:tc>"
        b"<w:tc><w:p><w:r><w:t>Cell B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
        b"<w:sectPr/></w:body></w:document>"
    )

    parsed = DOCXTextParser().parse(_docx_package(xml))

    assert parsed == (ParsedPage("Heading\tOne\nTwo\n\nCell A\n\nCell B"),)


@pytest.mark.parametrize(
    ("parser_kwargs", "xml", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {},
            b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            b"<w:sectPr/></w:document>",
            "missing its document body",
        ),
        ({"max_archive_bytes": 8}, _docx_xml("hello"), "max_archive_bytes=8"),
        ({"max_document_xml_bytes": 10}, _docx_xml("hello"), "max_document_xml_bytes=10"),
        ({"max_paragraphs": 1}, _docx_xml("first", "second"), "max_paragraphs=1"),
        ({"max_extracted_chars": 4}, _docx_xml("hello"), "max_extracted_chars=4"),
        ({"max_xml_elements": 4}, _docx_xml("hello"), "max_xml_elements=4"),
    ],
)
def test_docx_parser_rejects_malformed_or_over_limit_inputs(
    parser_kwargs: dict[str, int], xml: bytes, message: str
) -> None:
    parser = DOCXTextParser(**parser_kwargs)
    content = xml if xml == b"not a zip archive" else _docx_package(xml)

    with pytest.raises(ValueError, match=message):
        parser.parse(content)


def test_docx_parser_rejects_entities_and_unsafe_archive_paths() -> None:
    dtd = (
        b'<!DOCTYPE doc [<!ENTITY value "expanded">]>'
        b'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        b"<w:body><w:p><w:r><w:t>&value;</w:t></w:r></w:p></w:body></w:document>"
    )
    with pytest.raises(ValueError, match="must not declare entities or a doctype"):
        DOCXTextParser().parse(_docx_package(dtd))
    with pytest.raises(ValueError, match="must not declare entities or a doctype"):
        DOCXTextParser().parse(_docx_package(dtd.decode().encode("utf-16")))

    unsafe = _docx_package(_docx_xml("hello"), extra_entries=(("../outside.txt", b"data"),))
    with pytest.raises(ValueError, match="unsafe archive path"):
        DOCXTextParser().parse(unsafe)


def test_docx_parser_rejects_duplicate_paths_and_uncompressed_size_overflow() -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = _docx_package(
            _docx_xml("hello"), extra_entries=(("word/document.xml", _docx_xml("shadow")),)
        )
    with pytest.raises(ValueError, match="duplicate archive paths"):
        DOCXTextParser().parse(duplicate)

    xml = _docx_xml("hello")
    content = _docx_package(xml)
    with zipfile.ZipFile(BytesIO(content)) as archive:
        uncompressed_bytes = sum(member.file_size for member in archive.infolist())
    parser = DOCXTextParser(
        max_uncompressed_bytes=uncompressed_bytes - 1,
        max_document_xml_bytes=len(xml),
    )
    with pytest.raises(ValueError, match="max_uncompressed_bytes"):
        parser.parse(content)


def test_docx_parser_requires_document_parts() -> None:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr("_rels/.rels", b"<Relationships/>")

    with pytest.raises(ValueError, match="missing required document parts"):
        DOCXTextParser().parse(output.getvalue())


@pytest.mark.asyncio
async def test_file_ingestor_indexes_docx_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "guide.docx").write_bytes(_docx_package(_docx_xml("Rotate the access key quarterly.")))
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("guide.docx",)
    results = await store.retrieve("access key")
    assert len(results) == 1
    assert results[0].source == "guide.docx"
    assert "Rotate the access key quarterly." in results[0].text


def test_pptx_parser_extracts_text_in_slide_and_paragraph_order_with_citations() -> None:
    content = _pptx_package(
        (
            _pptx_slide_xml("Quarterly review", "Revenue increased"),
            _pptx_slide_xml("Next steps"),
        )
    )

    parsed = PPTXTextParser().parse(content)

    assert parsed == [
        ParsedPage("Quarterly review\n\nRevenue increased", page_number=1),
        ParsedPage("Next steps", page_number=2),
    ]


@pytest.mark.parametrize(
    ("parser_kwargs", "content", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {"max_archive_bytes": 8},
            _pptx_package((_pptx_slide_xml("text"),)),
            "max_archive_bytes=8",
        ),
        (
            {"max_slide_xml_bytes": 10},
            _pptx_package((_pptx_slide_xml("text"),)),
            "max_slide_xml_bytes=10",
        ),
        (
            {"max_slides": 1},
            _pptx_package((_pptx_slide_xml("one"), _pptx_slide_xml("two"))),
            "max_slides=1",
        ),
        (
            {"max_extracted_chars": 4},
            _pptx_package((_pptx_slide_xml("first"),)),
            "max_extracted_chars=4",
        ),
    ],
)
def test_pptx_parser_rejects_malformed_or_over_limit_inputs(
    parser_kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PPTXTextParser(**parser_kwargs).parse(content)


def test_pptx_parser_rejects_dtd_unsafe_and_duplicate_archive_entries() -> None:
    dtd = (
        b'<!DOCTYPE doc [<!ENTITY value "expanded">]>'
        b'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        b"<p:cSld><p:spTree><a:p><a:r><a:t>&value;</a:t></a:r></a:p></p:spTree></p:cSld></p:sld>"
    )
    with pytest.raises(ValueError, match="must not declare entities or a doctype"):
        PPTXTextParser().parse(_pptx_package((dtd,)))
    unsafe = _pptx_package((_pptx_slide_xml("safe"),), extra_entries=(("../outside.txt", b"data"),))
    with pytest.raises(ValueError, match="unsafe archive path"):
        PPTXTextParser().parse(unsafe)
    with pytest.warns(UserWarning, match="Duplicate name"):
        duplicate = _pptx_package(
            (_pptx_slide_xml("safe"),),
            extra_entries=(("ppt/slides/slide1.xml", _pptx_slide_xml("shadow")),),
        )
    with pytest.raises(ValueError, match="duplicate archive paths"):
        PPTXTextParser().parse(duplicate)


def test_odp_parser_extracts_slide_text_and_names_with_citations() -> None:
    parsed = ODPTextParser().parse(
        _odp_package(
            (
                ("Quarterly review", ("Revenue increased", "Costs stayed flat")),
                ("Next steps", ("Expand the pilot",)),
            )
        )
    )

    assert parsed == [
        ParsedPage(
            "Revenue increased\n\nCosts stayed flat",
            page_number=1,
            metadata={"slide_name": "Quarterly review"},
        ),
        ParsedPage("Expand the pilot", page_number=2, metadata={"slide_name": "Next steps"}),
    ]


@pytest.mark.parametrize(
    ("parser_kwargs", "content", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        ({"max_archive_bytes": 8}, _odp_package((("slide", ("text",)),)), "max_archive_bytes=8"),
        (
            {"max_content_xml_bytes": 8},
            _odp_package((("slide", ("text",)),)),
            "max_content_xml_bytes=8",
        ),
        (
            {"max_slides": 1},
            _odp_package((("one", ("text",)), ("two", ("text",)))),
            "max_slides=1",
        ),
        (
            {"max_extracted_chars": 4},
            _odp_package((("slide", ("first",)),)),
            "max_extracted_chars=4",
        ),
    ],
)
def test_odp_parser_rejects_malformed_or_over_limit_inputs(
    parser_kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ODPTextParser(**parser_kwargs).parse(content)


def test_odp_parser_rejects_unsafe_or_invalid_packages() -> None:
    unsafe = _odp_package((("slide", ("safe",)),), extra_entries=(("../outside", b"data"),))
    with pytest.raises(ValueError, match="unsafe archive path"):
        ODPTextParser().parse(unsafe)
    wrong_mimetype = _odp_package(
        (("slide", ("safe",)),), mimetype=b"application/vnd.oasis.opendocument.text"
    )
    with pytest.raises(ValueError, match="mimetype entry is invalid"):
        ODPTextParser().parse(wrong_mimetype)


@pytest.mark.asyncio
async def test_file_ingestor_indexes_pptx_slides_with_page_citations(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "review.pptx").write_bytes(
        _pptx_package((_pptx_slide_xml("Revenue plan"), _pptx_slide_xml("Hiring roadmap")))
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("review.pptx",)
    revenue = await store.retrieve("revenue")
    hiring = await store.retrieve("hiring")
    assert revenue[0].metadata["page_number"] == 1
    assert hiring[0].metadata["page_number"] == 2


@pytest.mark.asyncio
async def test_file_ingestor_indexes_odp_slides_with_page_citations(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "review.odp").write_bytes(
        _odp_package((("Revenue", ("Revenue plan",)), ("Hiring", ("Hiring roadmap",))))
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("review.odp",)
    revenue = await store.retrieve("revenue")
    hiring = await store.retrieve("hiring")
    assert revenue[0].metadata["page_number"] == 1
    assert revenue[0].metadata["slide_name"] == "Revenue"
    assert hiring[0].metadata["page_number"] == 2


def test_epub_parser_extracts_visible_text_in_spine_order_with_chapter_citations() -> None:
    content = _epub_package(
        (
            (
                "first",
                b"<html><head><title>Hidden title</title></head>"
                b"<body><h1>Welcome</h1></body></html>",
            ),
            ("second", b"<html><body><p>Second chapter</p><script>secret()</script></body></html>"),
            ("third", "<html><body><p>UTF-16 chapter</p></body></html>".encode("utf-16")),
        )
    )

    parsed = EPUBTextParser().parse(content)

    assert parsed == [
        ParsedPage("Hidden title\n\nWelcome", page_number=1),
        ParsedPage("Second chapter", page_number=2),
        ParsedPage("UTF-16 chapter", page_number=3),
    ]


@pytest.mark.parametrize(
    ("kwargs", "content", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {"max_archive_bytes": 8},
            _epub_package((("chapter", b"<p>Text</p>"),)),
            "max_archive_bytes=8",
        ),
        (
            {"max_chapters": 1},
            _epub_package((("one", b"<p>One</p>"), ("two", b"<p>Two</p>"))),
            "max_chapters=1",
        ),
        (
            {"max_chapter_bytes": 8},
            _epub_package((("chapter", b"<p>Too long</p>"),)),
            "max_bytes=8",
        ),
        (
            {"max_extracted_chars": 4},
            _epub_package((("chapter", b"<p>Long text</p>"),)),
            "max_extracted_chars=4",
        ),
    ],
)
def test_epub_parser_rejects_malformed_or_over_limit_inputs(
    kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EPUBTextParser(**kwargs).parse(content)


def test_epub_parser_rejects_entities_unsafe_references_and_invalid_package() -> None:
    dtd_chapter = (
        b'<!DOCTYPE html [<!ENTITY value "expanded">]><html><body><p>&value;</p></body></html>'
    )
    with pytest.raises(ValueError, match="must not contain NUL, entities, or a doctype"):
        EPUBTextParser().parse(_epub_package((("chapter", dtd_chapter),)))
    with pytest.raises(ValueError, match="unsafe package reference"):
        EPUBTextParser().parse(
            _epub_package((("chapter", b"<p>text</p>"),), hrefs=("../../outside.xhtml",))
        )
    with pytest.raises(ValueError, match="unsafe package reference"):
        EPUBTextParser().parse(
            _epub_package((("chapter", b"<p>text</p>"),), hrefs=("https://example.test/ch.xhtml",))
        )


@pytest.mark.asyncio
async def test_file_ingestor_indexes_epub_chapters_with_chapter_citations(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "guide.epub").write_bytes(
        _epub_package(
            (
                ("intro", b"<html><body><p>Set up the research notebook.</p></body></html>"),
                ("results", b"<html><body><p>Findings support the new policy.</p></body></html>"),
            )
        )
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("guide.epub",)
    assert ".epub" in FileIngestor.supported_extensions
    found = await store.retrieve("new policy")
    assert found[0].metadata["page_number"] == 2


def test_odt_parser_preserves_heading_paragraph_table_and_inline_order() -> None:
    content_xml = _odt_xml(
        "<text:h>Overview <text:span>of policy</text:span></text:h>"
        '<text:p>Keep<text:s text:c="2"/>two<text:tab/>columns'
        "<text:line-break/>next<office:annotation><text:p>private comment</text:p>"
        "</office:annotation></text:p>"
        "<table:table><table:table-row><table:table-cell><text:p>North</text:p></table:table-cell>"
        "<table:table-cell><text:p>South</text:p></table:table-cell></table:table-row></table:table>"
    )

    parsed = ODTTextParser().parse(_odt_package(content_xml))

    assert parsed == (
        ParsedPage("Overview of policy\n\nKeep  two\tcolumns\nnext\n\nNorth\n\nSouth"),
    )


@pytest.mark.parametrize(
    ("kwargs", "content", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {"max_archive_bytes": 8},
            _odt_package(_odt_xml("<text:p>Text</text:p>")),
            "max_archive_bytes=8",
        ),
        (
            {"max_content_xml_bytes": 8},
            _odt_package(_odt_xml("<text:p>Text</text:p>")),
            "max_content_xml_bytes=8",
        ),
        (
            {"max_paragraphs": 1},
            _odt_package(_odt_xml("<text:p>One</text:p><text:p>Two</text:p>")),
            "max_paragraphs=1",
        ),
        (
            {"max_extracted_chars": 4},
            _odt_package(_odt_xml("<text:p>Long text</text:p>")),
            "max_extracted_chars=4",
        ),
    ],
)
def test_odt_parser_rejects_malformed_or_over_limit_inputs(
    kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ODTTextParser(**kwargs).parse(content)


def test_odt_parser_rejects_entities_and_unsafe_archive_paths() -> None:
    dtd = b'<!DOCTYPE doc [<!ENTITY value "expanded">]>' + _odt_xml("<text:p>&value;</text:p>")
    with pytest.raises(ValueError, match="ODT XML is malformed or unsafe"):
        ODTTextParser().parse(_odt_package(dtd))
    with pytest.raises(ValueError, match="unsafe archive path"):
        ODTTextParser().parse(
            _odt_package(_odt_xml("<text:p>Text</text:p>"), extra_entries=(("../escape", b"x"),))
        )


@pytest.mark.asyncio
async def test_file_ingestor_indexes_odt_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "minutes.odt").write_bytes(
        _odt_package(_odt_xml("<text:p>The budget review is scheduled for Thursday.</text:p>"))
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("minutes.odt",)
    assert ".odt" in FileIngestor.supported_extensions
    found = await store.retrieve("budget review")
    assert found and "scheduled for Thursday" in found[0].text


def test_ods_parser_renders_sheet_and_row_labeled_values() -> None:
    content_xml = _ods_xml(
        '<table:table table:name="Budget">'
        "<table:table-row>"
        '<table:table-cell office:value-type="string"><text:p>Quarter</text:p></table:table-cell>'
        '<table:table-cell office:value-type="string"><text:p>Amount</text:p></table:table-cell>'
        "</table:table-row><table:table-row>"
        '<table:table-cell office:value-type="string"><text:p>Q1</text:p></table:table-cell>'
        '<table:table-cell office:value-type="float" office:value="1250.50"/>'
        "</table:table-row></table:table>"
        '<table:table table:name="Notes"><table:table-row>'
        '<table:table-cell office:value-type="string"><text:p>Reviewed</text:p></table:table-cell>'
        "</table:table-row></table:table>"
    )

    parsed = ODSTextParser().parse(_ods_package(content_xml))

    assert parsed == (
        ParsedPage(
            'Sheet "Budget", row 1: A="Quarter", B="Amount"\n'
            'Sheet "Budget", row 2: A="Q1", B="1250.50"\n'
            'Sheet "Notes", row 1: A="Reviewed"'
        ),
    )


@pytest.mark.parametrize(
    ("kwargs", "content", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {"max_archive_bytes": 8},
            _ods_package(_ods_xml('<table:table table:name="S"/>')),
            "max_archive_bytes=8",
        ),
        (
            {"max_rows": 2},
            _ods_package(
                _ods_xml(
                    '<table:table table:name="S"><table:table-row table:number-rows-repeated="3">'
                    '<table:table-cell office:value-type="string"><text:p>x</text:p>'
                    "</table:table-cell>"
                    "</table:table-row></table:table>"
                )
            ),
            "max_rows=2",
        ),
        (
            {"max_columns": 2},
            _ods_package(
                _ods_xml(
                    '<table:table table:name="S"><table:table-row>'
                    '<table:table-cell table:number-columns-repeated="3" '
                    'office:value-type="string"><text:p>x</text:p></table:table-cell>'
                    "</table:table-row></table:table>"
                )
            ),
            "invalid repeated columns count",
        ),
        (
            {"max_cells": 1},
            _ods_package(
                _ods_xml(
                    '<table:table table:name="S"><table:table-row>'
                    '<table:table-cell office:value-type="string"><text:p>a</text:p>'
                    "</table:table-cell>"
                    '<table:table-cell office:value-type="string"><text:p>b</text:p>'
                    "</table:table-cell>"
                    "</table:table-row></table:table>"
                )
            ),
            "max_cells=1",
        ),
    ],
)
def test_ods_parser_rejects_malformed_or_over_limit_inputs(
    kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ODSTextParser(**kwargs).parse(content)


def test_ods_parser_rejects_entities_and_unsafe_archive_paths() -> None:
    dtd = b'<!DOCTYPE doc [<!ENTITY value "expanded">]>' + _ods_xml(
        '<table:table table:name="S"><table:table-row><table:table-cell>'
        "<text:p>&value;</text:p></table:table-cell></table:table-row></table:table>"
    )
    with pytest.raises(ValueError, match="ODS XML is malformed or unsafe"):
        ODSTextParser().parse(_ods_package(dtd))
    with pytest.raises(ValueError, match="unsafe archive path"):
        ODSTextParser().parse(
            _ods_package(
                _ods_xml('<table:table table:name="S"/>'),
                extra_entries=(("../escape", b"x"),),
            )
        )


@pytest.mark.asyncio
async def test_file_ingestor_indexes_ods_sheet_values_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "budget.ods").write_bytes(
        _ods_package(
            _ods_xml(
                '<table:table table:name="Forecast"><table:table-row>'
                '<table:table-cell office:value-type="string"><text:p>Operating cost</text:p>'
                "</table:table-cell></table:table-row></table:table>"
            )
        )
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("budget.ods",)
    assert ".ods" in FileIngestor.supported_extensions
    found = await store.retrieve("operating cost")
    assert found and 'Sheet "Forecast"' in found[0].text


def test_xlsx_parser_reads_workbook_order_shared_strings_and_typed_values() -> None:
    first_sheet = _xlsx_worksheet(
        '<row r="1"><c r="A1" t="s"><v>0</v></c>'
        '<c r="B1" t="inlineStr"><is><t>Q1</t></is></c></row>'
        '<row r="3"><c r="A3" t="s"><v>1</v></c>'
        '<c r="B3"><v>1250.50</v></c><c r="C3" t="b"><v>1</v></c></row>'
    )
    second_sheet = _xlsx_worksheet(
        '<row r="1"><c r="A1" t="d"><v>2025-01-01</v></c>'
        '<c r="B1" t="str"><v>Reviewed</v></c></row>'
    )
    package = _xlsx_package(
        (("Budget", first_sheet), ("Notes", second_sheet)),
        shared_strings=("<t>Quarter</t>", "<r><t>Total</t></r><r><t> revenue</t></r>"),
    )

    parsed = XLSXTextParser().parse(package)

    assert parsed == (
        ParsedPage(
            'Sheet "Budget", row 1: A="Quarter", B="Q1"\n'
            'Sheet "Budget", row 3: A="Total revenue", B="1250.50", C="TRUE"\n'
            'Sheet "Notes", row 1: A="2025-01-01", B="Reviewed"'
        ),
    )


@pytest.mark.parametrize(
    ("kwargs", "package", "message"),
    [
        ({}, b"not a zip archive", "malformed or could not be read"),
        (
            {"max_archive_bytes": 8},
            _xlsx_package((("Data", _xlsx_worksheet('<row r="1"/>')),)),
            "max_archive_bytes=8",
        ),
        (
            {"max_worksheet_xml_bytes": 8},
            _xlsx_package((("Data", _xlsx_worksheet('<row r="1"/>')),)),
            "worksheet part exceeds max_bytes=8",
        ),
        (
            {"max_rows": 2},
            _xlsx_package((("Data", _xlsx_worksheet('<row r="3"/>')),)),
            "max_rows=2",
        ),
        (
            {"max_columns": 2},
            _xlsx_package((("Data", _xlsx_worksheet('<row r="1"><c r="C1"><v>1</v></c></row>')),)),
            "max_columns=2",
        ),
        (
            {"max_cells": 1},
            _xlsx_package(
                (
                    (
                        "Data",
                        _xlsx_worksheet(
                            '<row r="1"><c r="A1"><v>1</v></c><c r="B1"><v>2</v></c></row>'
                        ),
                    ),
                )
            ),
            "max_cells=1",
        ),
        (
            {"max_shared_strings": 1},
            _xlsx_package(
                (("Data", _xlsx_worksheet('<row r="1"><c r="A1" t="s"><v>0</v></c></row>')),),
                shared_strings=("<t>first</t>", "<t>second</t>"),
            ),
            "max_shared_strings=1",
        ),
        (
            {"max_extracted_chars": 5},
            _xlsx_package(
                (
                    (
                        "Data",
                        _xlsx_worksheet(
                            '<row r="1"><c r="A1" t="inlineStr"><is><t>long text</t></is></c></row>'
                        ),
                    ),
                )
            ),
            "max_extracted_chars=5",
        ),
    ],
)
def test_xlsx_parser_rejects_malformed_or_over_limit_inputs(
    kwargs: dict[str, int], package: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        XLSXTextParser(**kwargs).parse(package)


def test_xlsx_parser_rejects_unsafe_relationships_and_dtds() -> None:
    normal_sheet = _xlsx_worksheet('<row r="1"><c r="A1"><v>1</v></c></row>')
    with pytest.raises(ValueError, match="unsafe package relationship"):
        XLSXTextParser().parse(
            _xlsx_package((("Data", normal_sheet),), first_target="https://example.test/sheet.xml")
        )
    dtd_sheet = (
        b'<!DOCTYPE worksheet [<!ENTITY value "unsafe">]>'
        + b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData><row r='1'><c r='A1' t='inlineStr'><is><t>&value;</t></is></c>"
        b"</row></sheetData></worksheet>"
    )
    with pytest.raises(ValueError, match="worksheet XML is malformed or unsafe"):
        XLSXTextParser().parse(_xlsx_package((("Data", dtd_sheet),)))


@pytest.mark.asyncio
async def test_file_ingestor_indexes_xlsx_sheet_values_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    (root / "forecast.xlsx").write_bytes(
        _xlsx_package(
            (
                (
                    "Forecast",
                    _xlsx_worksheet(
                        '<row r="1"><c r="A1" t="inlineStr">'
                        "<is><t>Quarterly revenue forecast</t></is></c></row>"
                    ),
                ),
            )
        )
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("forecast.xlsx",)
    assert ".xlsx" in FileIngestor.supported_extensions
    found = await store.retrieve("revenue forecast")
    assert found and 'Sheet "Forecast"' in found[0].text


def test_rtf_parser_extracts_visible_text_unicode_and_escaped_characters() -> None:
    content = (
        b"{\\rtf1\\ansi\\ansicpg1252 Caf\\'e9 \\b formatted\\b0 \\par "
        b"Unicode: \\uc1\\u233? \\u-10179?\\u-9047?\\par "
        b"Escapes: \\{brace\\} \\\\ slash\\tab next}"
    )

    parsed = RTFTextParser().parse(content)

    assert parsed == (ParsedPage("Café formatted\nUnicode: é 💩\nEscapes: {brace} \\ slash\tnext"),)


def test_rtf_parser_skips_destinations_hidden_text_and_binary_payloads() -> None:
    content = (
        b"{\\rtf1 visible{\\fonttbl{\\f0 Hidden Font;}}"
        b"{\\*\\unknown Hidden Destination}{\\v Hidden Text\\v0 Visible Text}"
        b"{\\pict\\bin4 }\x00{\x10}\\par after}"
    )

    parsed = RTFTextParser().parse(content)

    assert parsed == (ParsedPage("visibleVisible Text\nafter"),)


@pytest.mark.parametrize(
    ("kwargs", "content", "message"),
    [
        ({}, b"not rtf", "valid header"),
        ({}, b"{\\rtf1 unclosed", "unbalanced groups"),
        ({}, b"{\\rtf1 bad \\'x0G}", "invalid hexadecimal"),
        ({}, b"{\\rtf1\\bin100 short}", "invalid binary"),
        ({"max_input_bytes": 4}, b"{\\rtf1 text}", "max_input_bytes=4"),
        ({"max_group_depth": 1}, b"{\\rtf1 {nested}}", "max_group_depth=1"),
        ({"max_control_words": 1}, b"{\\rtf1\\ansi text}", "max_control_words=1"),
        ({"max_extracted_chars": 3}, b"{\\rtf1 text}", "max_extracted_chars=3"),
    ],
)
def test_rtf_parser_rejects_malformed_or_over_limit_inputs(
    kwargs: dict[str, int], content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RTFTextParser(**kwargs).parse(content)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_input_bytes": 0},
        {"max_group_depth": True},
        {"max_control_words": 10_000_001},
        {"max_extracted_chars": -1},
    ],
)
def test_rtf_parser_validates_limits(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        RTFTextParser(**kwargs)  # type: ignore[arg-type]


def test_html_parser_extracts_text_structure_and_omits_nonvisible_subtrees() -> None:
    content = b"""<!doctype html>
<html><head><title>Account Help</title><script>secretHead()</script></head>
<body><main><h1>Password reset</h1>
<p>Go to <strong>Account Settings</strong> &amp; choose reset.</p>
<ul><li>Verify account ownership</li><li>Set a new password</li></ul>
<div hidden><div>nested hidden data</div></div>
<div aria-hidden="true">inaccessible text</div><img hidden>
<p>Done.</p><script>exfiltrate()</script><style>.hidden { color:red }</style>
<p>After the hidden image.</p>
<template>template-only text</template><svg><text>vector label</text></svg>
</main></body></html>"""

    page = HTMLTextParser().parse(content)[0]

    assert page.page_number is None
    assert page.text == (
        "Account Help\n\nPassword reset\n\nGo to Account Settings & choose reset.\n\n"
        "Verify account ownership\n\nSet a new password\n\nDone.\n\nAfter the hidden image."
    )
    for omitted in (
        "secretHead",
        "hidden account data",
        "nested hidden data",
        "inaccessible text",
        "exfiltrate",
        "template-only",
        "vector label",
        ".hidden",
    ):
        assert omitted not in page.text


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"<p>\xff</p>", "valid UTF-8"), (b"<p>a\x00b</p>", "NUL")],
)
def test_html_parser_rejects_invalid_text_inputs(content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HTMLTextParser().parse(content)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_chars": 0}, ValueError),
        ({"max_chars": True}, ValueError),
        ({"max_chars": 8, "overlap_chars": 8}, ValueError),
        ({"max_chars": 8, "overlap_chars": -1}, ValueError),
    ],
)
def test_paragraph_chunker_validates_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ParagraphChunker(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_file_ingestor_replaces_each_source_and_stores_chunk_metadata(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    source = root / "guide.md"
    source.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = TextFileIngestor(
        store,
        root,
        chunker=ParagraphChunker(max_chars=20, overlap_chars=0),
    )

    assert await ingestor.ingest_file("guide.md", metadata={"area": "support"}) == 2
    first = await store.retrieve("First")
    assert len(first) == 1
    assert first[0].metadata == {"area": "support", "chunk_count": 2, "chunk_index": 0}
    original_ids = {document.id for document in await store.retrieve("paragraph")}

    assert await ingestor.ingest_file("guide.md", metadata={"area": "support"}) == 2
    assert {document.id for document in await store.retrieve("paragraph")} == original_ids

    source.write_text("Replacement content only.", encoding="utf-8")
    assert await ingestor.ingest_file("guide.md") == 2
    assert await store.retrieve("First paragraph") == []
    assert len(await store.retrieve("Replacement content")) == 1


@pytest.mark.asyncio
async def test_file_ingestor_indexes_default_html_parser_output(tmp_path: Path) -> None:
    root = tmp_path / "html"
    root.mkdir()
    (root / "help.html").write_text(
        "<html><head><title>Account help</title></head><body>"
        "<h1>Reset a password</h1><p>Open account security settings.</p>"
        "<script>ignore this text</script></body></html>",
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root, chunker=ParagraphChunker(overlap_chars=0))

    report = await ingestor.ingest_directory()

    assert report.sources == ("help.html",)
    assert report.document_count == 1
    results = await store.retrieve("account security")
    assert any("Open account security settings." in document.text for document in results)
    assert all("ignore this text" not in document.text for document in await store.retrieve("text"))


@pytest.mark.asyncio
async def test_file_ingestor_indexes_default_rtf_parser_output(tmp_path: Path) -> None:
    root = tmp_path / "rtf"
    root.mkdir()
    (root / "balance.rtf").write_bytes(
        b"{\\rtf1\\ansi\\b Quarterly balance\\b0\\par Expenses were 42 dollars.}"
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("balance.rtf",)
    assert ".rtf" in FileIngestor.supported_extensions
    results = await store.retrieve("Expenses dollars")
    assert results and "Expenses were 42 dollars." in results[0].text


def test_json_parser_preserves_valid_source_and_rejects_ambiguous_documents() -> None:
    parser = JSONTextParser()
    source = '{"city":"Pune","population":3100000,"active":true}'

    assert parser.parse(source.encode())[0].text == source
    for invalid in (
        '{"role":"analyst","role":"admin"}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"trailing":true} false',
    ):
        with pytest.raises(ValueError, match="valid JSON"):
            parser.parse(invalid.encode())


def test_json_parser_handles_escapes_and_rejects_invalid_utf8_or_nul() -> None:
    parser = JSONTextParser()
    escaped = b'{"quote":"a \\"quoted\\" value"}'

    assert parser.parse(escaped)[0].text == escaped.decode()
    with pytest.raises(ValueError, match="valid UTF-8"):
        parser.parse(b"\xff")
    with pytest.raises(ValueError, match="NUL"):
        parser.parse(b'{"value":"\x00"}')


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"max_depth": 0}, ValueError), ({"max_tokens": True}, TypeError)],
)
def test_json_parser_validates_resource_limits(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        JSONTextParser(**kwargs)  # type: ignore[arg-type]


def test_json_parser_rejects_documents_over_complexity_limits() -> None:
    with pytest.raises(ValueError, match="max_depth=2"):
        JSONTextParser(max_depth=2).parse(b'{"a":{"b":{"c":1}}}')
    with pytest.raises(ValueError, match="max_tokens=2"):
        JSONTextParser(max_tokens=2).parse(b'{"a":1}')


def test_json_parser_extracts_json_feed_items_with_citations_and_metadata() -> None:
    source = {
        "version": "https://jsonfeed.org/version/1.1",
        "title": "Research notes",
        "author": {"name": "Gabby Research"},
        "items": [
            {
                "id": "entry-1",
                "title": "A useful finding",
                "content_html": "<p>Evidence from <strong>the report</strong>.</p>",
                "url": "https://example.org/research/1",
                "date_published": "2026-10-03T09:00:00Z",
                "tags": ["research", "evidence"],
            },
            {"id": 2, "content_text": "A short update."},
        ],
    }

    pages = JSONTextParser().parse(json.dumps(source).encode())

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Evidence from the report." in pages[0].text
    assert pages[0].metadata == {
        "feed_item_number": 1,
        "feed_item_id": "entry-1",
        "feed_title": "Research notes",
        "url": "https://example.org/research/1",
        "published_at": "2026-10-03T09:00:00Z",
        "author": "Gabby Research",
        "categories": ["research", "evidence"],
    }
    assert pages[1].page_number == 2
    assert pages[1].metadata["feed_item_id"] == "2"
    assert "A short update." in pages[1].text


def test_json_feed_detection_preserves_other_json_and_rejects_malformed_feeds() -> None:
    parser = JSONTextParser()
    ordinary = '{"version":"custom","items":[{"id":"still ordinary JSON"}]}'

    assert parser.parse(ordinary.encode())[0].text == ordinary
    with pytest.raises(ValueError, match="items must be an array"):
        parser.parse(b'{"version":"https://jsonfeed.org/version/1","items":{}}')
    with pytest.raises(ValueError, match="must contain text, HTML, or summary"):
        parser.parse(b'{"version":"https://jsonfeed.org/version/1.1","items":[{"id":"x"}]}')
    with pytest.raises(ValueError, match="max_feed_items=1"):
        JSONTextParser(max_feed_items=1).parse(
            b'{"version":"https://jsonfeed.org/version/1","items":['
            b'{"id":"1","content_text":"a"},'
            b'{"id":"2","content_text":"b"}]}'
        )


def test_opml_parser_indexes_nested_outlines_and_safe_links() -> None:
    source = b"""<?xml version="1.0" encoding="utf-8"?>
    <opml version="2.0"><head><title>Global News</title></head><body>
      <outline text="Research" type="folder">
        <outline text="Climate report" title="Climate report" type="rss"
          xmlUrl="https://feeds.example.org/climate.xml"
          htmlUrl="https://example.org/climate" description="Climate observations" />
        <outline text="Unsafe link" type="link" url="https://user:secret@example.org/private" />
      </outline>
    </body></opml>"""

    pages = OPMLTextParser().parse(source)

    assert len(pages) == 3
    assert pages[1].page_number == 2
    assert "Outline: Global News" in pages[1].text
    assert "Categories: Research" in pages[1].text
    assert pages[1].metadata["url"] == "https://feeds.example.org/climate.xml"
    assert pages[1].metadata["xml_url"] == "https://feeds.example.org/climate.xml"
    assert pages[1].metadata["categories"] == ["Research"]
    assert "secret" not in pages[2].text
    assert "url" not in pages[2].metadata
    assert XMLTextParser().parse(source) == pages


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"max_outlines": 0}, ValueError), ({"max_depth": True}, TypeError)],
)
def test_opml_parser_validates_limits(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        OPMLTextParser(**kwargs)  # type: ignore[arg-type]


def test_opml_parser_rejects_invalid_structure_and_bounds() -> None:
    with pytest.raises(ValueError, match="exactly one head and one body"):
        OPMLTextParser().parse(b'<opml version="2.0"><head/><body/><body/></opml>')
    with pytest.raises(ValueError, match="version must be"):
        OPMLTextParser().parse(b'<opml version="3.0"><head/><body/></opml>')
    with pytest.raises(ValueError, match="DTD and entity"):
        OPMLTextParser().parse(
            b'<!DOCTYPE opml [<!ENTITY x "unsafe">]><opml version="2.0"><head/><body/></opml>'
        )
    with pytest.raises(ValueError, match="max_outlines=1"):
        OPMLTextParser(max_outlines=1).parse(
            b'<opml version="2.0"><head/><body><outline text="one"/>'
            b'<outline text="two"/></body></opml>'
        )
    with pytest.raises(ValueError, match="max_elements=2"):
        OPMLTextParser(max_elements=2).parse(b'<opml version="2.0"><head/><body/></opml>')


def test_yaml_parser_renders_nested_values_as_searchable_paths() -> None:
    source = (
        "title: Quarterly Results\n"
        "finance:\n"
        "  revenue: 4.5\n"
        "  approved: true\n"
        "regions:\n"
        "  - APAC\n"
        "published: 2026-10-03\n"
    )

    parsed = YAMLTextParser().parse(source.encode())

    assert parsed == (
        ParsedPage(
            "title: Quarterly Results\n"
            "finance.revenue: 4.5\n"
            "finance.approved: true\n"
            "regions[0]: APAC\n"
            "published: 2026-10-03"
        ),
    )


def test_yaml_parser_rejects_duplicates_unsafe_values_and_multiple_documents() -> None:
    parser = YAMLTextParser()
    for invalid in (
        "role: analyst\nrole: admin\n",
        "first: &cycle [*cycle]\n",
        "---\nvalue: 1\n---\nvalue: 2\n",
        "base: &base {role: researcher}\nitem: {<<: *base}\n",
        "? [unhashable, key]\n: value\n",
        "value: !!python/object/apply:os.system ['echo unsafe']\n",
        "value: .nan\n",
    ):
        with pytest.raises(ValueError, match="YAML"):
            parser.parse(invalid.encode())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"max_input_bytes": 0}, ValueError), ({"max_nodes": True}, TypeError)],
)
def test_yaml_parser_validates_limits(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        YAMLTextParser(**kwargs)  # type: ignore[arg-type]


def test_yaml_parser_enforces_input_depth_node_and_rendered_output_limits() -> None:
    with pytest.raises(ValueError, match="max_input_bytes=4"):
        YAMLTextParser(max_input_bytes=4).parse(b"value: 1")
    with pytest.raises(ValueError, match="max_nodes=2"):
        YAMLTextParser(max_nodes=2).parse(b"a: 1")
    with pytest.raises(ValueError, match="max_depth=2"):
        YAMLTextParser(max_depth=2).parse(b"a:\n  b:\n    c: 1")
    with pytest.raises(ValueError, match="max_output_bytes=5"):
        YAMLTextParser(max_output_bytes=5).parse(b"value: 1")


@pytest.mark.asyncio
async def test_file_ingestor_indexes_yaml_as_structured_content(tmp_path: Path) -> None:
    root = tmp_path / "yaml"
    root.mkdir()
    (root / "itinerary.yaml").write_text(
        "trip:\n  destination: Kyoto\n  month: April\n", encoding="utf-8"
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("itinerary.yaml",)
    assert ".yaml" in FileIngestor.supported_extensions
    results = await store.retrieve("Kyoto April")
    assert results and "trip.destination: Kyoto" in results[0].text


def test_toml_parser_renders_tables_arrays_and_dates_as_searchable_paths() -> None:
    source = (
        'title = "Quarterly Results"\n'
        'regions = ["APAC", "EMEA"]\n'
        "published = 2026-10-03\n"
        "[finance]\n"
        "revenue = 4.5\n"
        "approved = true\n"
        '[[milestones]]\nname = "launch"\ndate = 2026-04-05\n'
    )

    parsed = TOMLTextParser().parse(source.encode())

    assert parsed == (
        ParsedPage(
            "title: Quarterly Results\n"
            "regions[0]: APAC\n"
            "regions[1]: EMEA\n"
            "published: 2026-10-03\n"
            "finance.revenue: 4.5\n"
            "finance.approved: true\n"
            "milestones[0].name: launch\n"
            "milestones[0].date: 2026-04-05"
        ),
    )


def test_toml_parser_rejects_duplicates_non_finite_and_invalid_documents() -> None:
    parser = TOMLTextParser()
    for invalid in (b'role = "analyst"\nrole = "admin"', b"value = nan", b"[invalid"):
        with pytest.raises(ValueError, match="TOML"):
            parser.parse(invalid)
    with pytest.raises(ValueError, match="UTF-8"):
        parser.parse(b"value = \xff")
    with pytest.raises(ValueError, match="NUL"):
        parser.parse(b"value = \x00")


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"max_input_bytes": 0}, ValueError), ({"max_nodes": True}, TypeError)],
)
def test_toml_parser_validates_limits(kwargs: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        TOMLTextParser(**kwargs)  # type: ignore[arg-type]


def test_toml_parser_enforces_input_depth_node_and_rendered_output_limits() -> None:
    with pytest.raises(ValueError, match="max_input_bytes=4"):
        TOMLTextParser(max_input_bytes=4).parse(b"value = 1")
    with pytest.raises(ValueError, match="valid, bounded"):
        TOMLTextParser(max_nodes=1).parse(b"a = 1")
    with pytest.raises(ValueError, match="max_depth=2"):
        TOMLTextParser(max_depth=2).parse(b"a = [[[1]]]")
    with pytest.raises(ValueError, match="max_output_bytes=5"):
        TOMLTextParser(max_output_bytes=5).parse(b"value = 1")


@pytest.mark.asyncio
async def test_file_ingestor_indexes_toml_project_metadata(tmp_path: Path) -> None:
    root = tmp_path / "toml"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "field-notes"\nlicense = "Apache-2.0"\n', encoding="utf-8"
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")

    report = await FileIngestor(store, root).ingest_directory()

    assert report.sources == ("pyproject.toml",)
    assert ".toml" in FileIngestor.supported_extensions
    results = await store.retrieve("field-notes Apache")
    assert results and "project.name: field-notes" in results[0].text


def test_json_lines_parser_validates_each_record_and_preserves_physical_line_numbers() -> None:
    source = '\ufeff{"event":"launch","status":"scheduled"}\r\n\r\n["countdown", 3]\n'

    page = JSONLinesTextParser().parse(source.encode())[0]

    assert page.text == (
        'Record 1:\n{"event":"launch","status":"scheduled"}\n\nRecord 3:\n["countdown", 3]'
    )
    for invalid in (
        b'{"role":"analyst","role":"admin"}',
        b'{"value":NaN}',
        b'{"trailing":true} false',
        '\u00a0{"not_json":true}'.encode(),
    ):
        with pytest.raises(ValueError, match="line 1 must be valid JSON"):
            JSONLinesTextParser().parse(invalid)


@pytest.mark.parametrize(
    ("source", "kwargs", "message"),
    [
        (b"", {}, "at least one record"),
        (b'{"a":1}\n{"b":2}', {"max_records": 1}, "max_records=1"),
        (b'{"value":"long"}', {"max_record_bytes": 8}, "max_record_bytes=8"),
        (b'{"a":1}\n{"b":2}', {"max_output_bytes": 24}, "max_output_bytes=24"),
        (b"\xff", {}, "valid UTF-8"),
        (b'{"value":"\x00"}', {}, "NUL"),
    ],
)
def test_json_lines_parser_enforces_limits_and_input_validity(
    source: bytes,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        JSONLinesTextParser(**kwargs).parse(source)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_records": 0}, ValueError),
        ({"max_record_bytes": True}, TypeError),
        ({"max_output_bytes": -1}, ValueError),
        ({"max_depth": 0}, ValueError),
        ({"max_tokens": True}, TypeError),
    ],
)
def test_json_lines_parser_validates_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        JSONLinesTextParser(**kwargs)  # type: ignore[arg-type]


def test_notebook_parser_indexes_cells_but_ignores_execution_outputs() -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "cell_1",
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Results\n", "Carbon rose 2%."],
            },
            {
                "id": "cell_2",
                "cell_type": "code",
                "metadata": {},
                "source": ["measurements = [1, 2, 3]\n", "print('analysis')"],
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": ["sensitive output"]}],
            },
            {
                "id": "cell_3",
                "cell_type": "raw",
                "metadata": {},
                "source": "Appendix note",
            },
            {
                "id": "cell_4",
                "cell_type": "code",
                "metadata": {},
                "source": "   ",
                "execution_count": None,
                "outputs": [],
            },
        ],
    }

    pages = NotebookTextParser().parse(json.dumps(notebook).encode())

    assert pages == (
        ParsedPage("Cell 1 (markdown)\n# Results\nCarbon rose 2%.", page_number=1),
        ParsedPage("Cell 2 (code)\nmeasurements = [1, 2, 3]\nprint('analysis')", page_number=2),
        ParsedPage("Cell 3 (raw)\nAppendix note", page_number=3),
    )
    assert "sensitive output" not in "\n".join(page.text for page in pages)


def test_notebook_parser_checks_required_metadata_and_unique_cell_ids() -> None:
    cell = {
        "id": "same",
        "cell_type": "markdown",
        "metadata": {},
        "source": "content",
    }
    missing_metadata = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": [cell],
    }
    duplicate_ids = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [cell, dict(cell)],
    }

    with pytest.raises(ValueError, match="notebook metadata must be an object"):
        NotebookTextParser().parse(json.dumps(missing_metadata).encode())
    with pytest.raises(ValueError, match="cell IDs must be unique"):
        NotebookTextParser().parse(json.dumps(duplicate_ids).encode())


@pytest.mark.parametrize(
    ("content", "kwargs", "message"),
    [
        (
            b'{"nbformat":4,"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[]}',
            {},
            "one valid JSON value",
        ),
        (b'{"nbformat":3,"nbformat_minor":5,"metadata":{},"cells":[]}', {}, "version 4"),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":[],"source":"x"}]}',
            {},
            "cell_type",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":1}]}',
            {},
            "source",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":"\\u0000"}]}',
            {},
            "must not contain NUL",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":"\\ud800"}]}',
            {},
            "valid Unicode",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":"xx"}]}',
            {"max_input_bytes": 10},
            "max_input_bytes=10",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":"xx"}]}',
            {"max_cell_source_bytes": 1},
            "max_cell_source_bytes=1",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"raw","source":"x"},{"id":"y",'
            b'"metadata":{},"cell_type":"raw","source":"y"}]}',
            {"max_cells": 1},
            "max_cells=1",
        ),
        (
            b'{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[{"id":"x",'
            b'"metadata":{},"cell_type":"code","execution_count":null,"outputs":[],'
            b'"source":"x"}]}',
            {"max_output_bytes": 1},
            "max_output_bytes=1",
        ),
    ],
)
def test_notebook_parser_rejects_malformed_or_oversized_input(
    content: bytes, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        NotebookTextParser(**kwargs).parse(content)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"max_cells": 0}, ValueError), ({"max_tokens": True}, TypeError)],
)
def test_notebook_parser_validates_limits(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        NotebookTextParser(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_file_ingestor_indexes_notebook_cells_as_cited_documents(tmp_path: Path) -> None:
    root = tmp_path / "notebooks"
    root.mkdir()
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "id": "cell_1",
                "cell_type": "markdown",
                "metadata": {},
                "source": "Atmospheric study",
            },
            {
                "id": "cell_2",
                "cell_type": "code",
                "metadata": {},
                "source": "print('carbon dioxide trend')",
                "execution_count": 1,
                "outputs": [{"output_type": "stream", "text": "private run output"}],
            },
        ],
    }
    (root / "climate.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()
    results = await store.retrieve("carbon dioxide trend")

    assert report.sources == ("climate.ipynb",)
    assert report.document_count == 2
    assert len(results) == 1
    assert "Cell 2 (code)" in results[0].text
    assert results[0].metadata["page_number"] == 2
    assert "private run output" not in results[0].text
    assert ".ipynb" in FileIngestor.supported_extensions


def test_xml_parser_preserves_paths_attributes_and_mixed_content_order() -> None:
    source = (
        '<root id="mission">before<item code="A">one</item> after &amp; <item>two</item>end</root>'
    )

    page = XMLTextParser().parse(source.encode())[0]

    assert page.text == (
        '/root [@id="mission"]\n\n'
        "/root: before\n\n"
        '/root/item [@code="A"]\n\n'
        "/root/item: one\n\n"
        "/root: after &\n\n"
        "/root/item: two\n\n"
        "/root: end"
    )


def test_xml_parser_keeps_generic_feed_root_as_path_labeled_xml() -> None:
    page = XMLTextParser().parse(b"<feed><entry>not Atom</entry></feed>")[0]

    assert page.text == "/feed/entry: not Atom"


def test_rss_parser_emits_cited_entry_pages_and_filterable_metadata() -> None:
    source = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Research Briefing</title>
      <item><title>New telescope finds a planet</title><link>https://example.test/planet</link>
        <guid>story-1</guid><pubDate>2026-10-03T09:00:00Z</pubDate>
        <category>astronomy</category><description><![CDATA[
          <p>Visible discovery details.</p><script>ignored active content</script>
        ]]></description></item>
      <item><title>Second report</title><description>Another finding.</description></item>
    </channel></rss>"""

    pages = RSSAtomTextParser().parse(source)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].metadata == {
        "feed_item_number": 1,
        "feed_title": "Research Briefing",
        "feed_item_id": "story-1",
        "url": "https://example.test/planet",
        "published_at": "2026-10-03T09:00:00Z",
        "categories": ["astronomy"],
    }
    assert "Title: New telescope finds a planet" in pages[0].text
    assert "Visible discovery details." in pages[0].text
    assert "ignored active content" not in pages[0].text
    assert "Another finding." in pages[1].text


def test_atom_parser_extracts_namespaced_entries_and_links() -> None:
    source = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"><title>Lab updates</title>
      <entry><title>Instrument calibration</title><id>tag:example.test,2026:1</id>
        <link rel="alternate" href="https://example.test/calibration"/>
        <updated>2026-10-02T15:30:00Z</updated><author><name>Research Team</name></author>
        <category term="instrumentation"/><summary type="html">&lt;p&gt;Calibration
          complete.&lt;/p&gt;</summary>
      </entry></feed>"""

    page = RSSAtomTextParser().parse(source)[0]

    assert page.metadata["feed_title"] == "Lab updates"
    assert page.metadata["feed_item_id"] == "tag:example.test,2026:1"
    assert page.metadata["url"] == "https://example.test/calibration"
    assert page.metadata["author"] == "Research Team"
    assert page.metadata["categories"] == ["instrumentation"]
    assert "Calibration complete." in page.text


def test_feed_parser_prefers_first_alternate_http_link_and_drops_unsafe_links() -> None:
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Safe</title>
      <link rel="self" href="https://example.test/self"/>
      <link rel="alternate" href="https://example.test/first"/>
      <link rel="alternate" href="https://example.test/second"/>
    </entry></feed>"""
    unsafe_rss = b"""<rss><channel><item><title>Unsafe</title>
      <link>javascript:alert(1)</link><description>No citation link.</description>
    </item></channel></rss>"""

    atom_page = RSSAtomTextParser().parse(atom)[0]
    unsafe_page = RSSAtomTextParser().parse(unsafe_rss)[0]

    assert atom_page.metadata["url"] == "https://example.test/first"
    assert "url" not in unsafe_page.metadata
    assert "Link:" not in unsafe_page.text


def test_feed_parser_rejects_unsafe_link_forms_and_enforces_category_limit() -> None:
    parser = RSSAtomTextParser(max_categories_per_item=1)
    assert parser._safe_link("https://example.test/path") == "https://example.test/path"
    for link in (
        "",
        "https://example.test/with space",
        "ftp://example.test/file",
        "https://user:secret@example.test/private",
        "https://[invalid/host",
    ):
        assert parser._safe_link(link) is None

    with pytest.raises(ValueError, match="max_categories_per_item=1"):
        parser.parse(
            b"<rss><channel><item><title>Tagged</title>"
            b"<category>one</category><category>two</category></item></channel></rss>"
        )


def test_rss_10_rdf_items_are_extracted() -> None:
    source = b"""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns="http://purl.org/rss/1.0/"><channel><title>Archive</title></channel>
      <item><title>Historical measurement</title><link>https://example.test/archive</link>
      <description>Recorded in 1984.</description></item></rdf:RDF>"""

    page = RSSAtomTextParser().parse(source)[0]

    assert page.metadata["feed_title"] == "Archive"
    assert "Historical measurement" in page.text
    assert page.metadata["url"] == "https://example.test/archive"


@pytest.mark.parametrize(
    ("source", "kwargs", "message"),
    [
        (b"<!DOCTYPE rss [<!ENTITY x 'expanded'>]><rss>&x;</rss>", {}, "DTD and entity"),
        (b"<?xml version='1.0' encoding='UTF-16'?><rss/>", {}, "must specify UTF-8"),
        (b"<rss><channel></rss>", {}, "well-formed"),
        (b"<html/>", {}, "root must be RSS"),
        (b"\xff", {}, "valid UTF-8"),
        (b"<rss>payload</rss>", {"max_input_bytes": 8}, "max_input_bytes=8"),
        (
            b"<rss><channel><item><title>one</title></item>"
            b"<item><title>two</title></item></channel></rss>",
            {"max_items": 1},
            "max_items=1",
        ),
        (b"<rss><channel><item/></channel></rss>", {"max_depth": 2}, "max_depth=2"),
        (
            b"<rss><channel><item><title>oversized body</title></item></channel></rss>",
            {"max_output_bytes": 10},
            "max_output_bytes=10",
        ),
    ],
)
def test_rss_atom_parser_rejects_unsafe_or_oversized_feeds(
    source: bytes, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RSSAtomTextParser(**kwargs).parse(source)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_input_bytes": 0}, ValueError),
        ({"max_items": True}, TypeError),
        ({"max_elements": 0}, ValueError),
        ({"max_categories_per_item": False}, TypeError),
    ],
)
def test_rss_atom_parser_validates_resource_limits(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        RSSAtomTextParser(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "kwargs", "message"),
    [
        (b"<!DOCTYPE root [<!ENTITY x 'expanded'>]><root>&x;</root>", {}, "DTD and entity"),
        (b"<?xml version='1.0' encoding='UTF-16'?><root/>", {}, "must specify UTF-8"),
        (b"<root><broken></root>", {}, "well-formed"),
        (b"\xff", {}, "valid UTF-8"),
        (b"<root>\x00</root>", {}, "NUL"),
        (b"<root>payload</root>", {"max_input_bytes": 8}, "max_input_bytes=8"),
        (b"<root><one/><two/></root>", {"max_elements": 2}, "max_elements=2"),
        (
            b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>one</title></entry>'
            b"<entry><title>two</title></entry></feed>",
            {"max_feed_items": 1},
            "max_items=1",
        ),
        (b"<a><b><c/></b></a>", {"max_depth": 2}, "max_depth=2"),
        (
            b'<root first="1" second="2"/>',
            {"max_attributes_per_element": 1},
            "max_attributes_per_element=1",
        ),
        (b"<root>some text</root>", {"max_output_bytes": 5}, "max_output_bytes=5"),
    ],
)
def test_xml_parser_rejects_unsafe_or_oversized_documents(
    source: bytes,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        XMLTextParser(**kwargs).parse(source)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"max_input_bytes": 0}, ValueError),
        ({"max_output_bytes": True}, TypeError),
        ({"max_elements": 0}, ValueError),
        ({"max_depth": False}, TypeError),
        ({"max_attributes_per_element": 0}, ValueError),
        ({"max_feed_items": False}, TypeError),
    ],
)
def test_xml_parser_validates_resource_limits(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        XMLTextParser(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_file_ingestor_indexes_json_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "records"
    root.mkdir()
    (root / "observations.json").write_text(
        '{"site":"Atacama","instrument":"spectrometer","finding":"methane detected"}',
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("observations.json",)
    assert report.document_count == 1
    results = await store.retrieve("methane detected")
    assert len(results) == 1
    assert '"site":"Atacama"' in results[0].text


@pytest.mark.asyncio
async def test_file_ingestor_indexes_json_feed_items_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "json-feeds"
    root.mkdir()
    (root / "updates.json").write_text(
        json.dumps(
            {
                "version": "https://jsonfeed.org/version/1",
                "title": "Lab updates",
                "items": [
                    {
                        "id": "run-42",
                        "title": "New spectral result",
                        "content_text": "Methane absorption increased near the target star.",
                        "url": "https://example.org/lab/run-42",
                        "tags": ["spectroscopy", "observations"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()
    results = await store.retrieve(
        "methane absorption",
        filters={"categories": ["spectroscopy", "observations"]},
    )

    assert report.sources == ("updates.json",)
    assert report.document_count == 1
    assert results and "New spectral result" in results[0].text
    assert results[0].metadata["url"] == "https://example.org/lab/run-42"
    assert results[0].metadata["feed_item_id"] == "run-42"


@pytest.mark.asyncio
async def test_file_ingestor_indexes_rss_entries_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "feeds"
    root.mkdir()
    feed = (
        "<rss version='2.0'><channel><title>Observatory</title><item>"
        "<title>Water vapor signature</title><link>https://example.test/water</link>"
        "<description>Atmospheric observation reported.</description>"
        "<category>spectroscopy</category></item></channel></rss>"
    )
    (root / "observatory.rss").write_text(feed, encoding="utf-8")
    (root / "observatory.xml").write_text(feed, encoding="utf-8")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()
    results = await store.retrieve(
        "Atmospheric observation", filters={"categories": ["spectroscopy"]}
    )

    assert report.sources == ("observatory.rss", "observatory.xml")
    assert report.document_count == 2
    assert len(results) == 2
    assert results[0].metadata["url"] == "https://example.test/water"
    assert ".rss" in FileIngestor.supported_extensions
    assert ".atom" in FileIngestor.supported_extensions


@pytest.mark.asyncio
async def test_file_ingestor_indexes_opml_outlines_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "subscriptions"
    root.mkdir()
    (root / "research.opml").write_text(
        '<opml version="2.0"><head><title>Research feeds</title></head><body>'
        '<outline text="Environment"><outline text="Climate lab" type="rss" '
        'xmlUrl="https://feeds.example.org/climate.xml" '
        'description="Climate change observations"/></outline></body></opml>',
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()
    results = await store.retrieve(
        "climate change observations",
        filters={"categories": ["Environment"]},
    )

    assert report.sources == ("research.opml",)
    assert report.document_count == 2
    assert results and results[0].metadata["url"] == "https://feeds.example.org/climate.xml"
    assert results[0].metadata["outline_title"] == "Climate lab"
    assert ".opml" in FileIngestor.supported_extensions


@pytest.mark.asyncio
async def test_file_ingestor_indexes_json_lines_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "events"
    root.mkdir()
    (root / "measurements.jsonl").write_text(
        '{"station":"Mauna Loa","finding":"carbon dioxide rising"}\n'
        '{"station":"Cape Grim","finding":"methane stable"}\n',
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("measurements.jsonl",)
    assert report.document_count == 1
    results = await store.retrieve("carbon dioxide rising")
    assert len(results) == 1
    assert "Record 1:" in results[0].text
    assert '"station":"Mauna Loa"' in results[0].text


@pytest.mark.asyncio
async def test_file_ingestor_indexes_xml_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    (root / "telemetry.xml").write_text(
        '<telemetry><station name="Vostok">temperature minus 80</station></telemetry>',
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("telemetry.xml",)
    assert report.document_count == 1
    result = await store.retrieve("Vostok temperature")
    assert len(result) == 1
    assert '[@name="Vostok"]' in result[0].text
    assert "temperature minus 80" in result[0].text


def test_csv_parser_renders_quoted_records_with_their_headers() -> None:
    source = 'region,note,value\n"South, Zone","line one\nline two",café\n\n'

    page = CSVTextParser().parse(source.encode())[0]

    assert '"region": "South, Zone"' in page.text
    assert '"note": "line one\\nline two"' in page.text
    assert '"value": "café"' in page.text


def test_csv_parser_supports_a_custom_delimiter_and_validates_limits() -> None:
    assert (
        CSVTextParser(delimiter=";").parse(b"city;country\nPune;India")[0].text.count("Record 1:")
        == 1
    )
    with pytest.raises(ValueError, match="delimiter"):
        CSVTextParser(delimiter="||")
    with pytest.raises(TypeError, match="max_rows"):
        CSVTextParser(max_rows=True)
    with pytest.raises(ValueError, match="max_columns"):
        CSVTextParser(max_columns=0)


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b"name,,value\none,two,three",
        b"name,NAME\none,two",
        b"name,value\none",
        b'name,value\n"unterminated,field',
    ],
)
def test_csv_parser_rejects_invalid_or_ambiguous_records(source: bytes) -> None:
    with pytest.raises(ValueError):
        CSVTextParser().parse(source)


def test_csv_parser_enforces_row_column_and_rendered_output_limits() -> None:
    with pytest.raises(ValueError, match="max_rows=1"):
        CSVTextParser(max_rows=1).parse(b"name\none\ntwo")
    with pytest.raises(ValueError, match="max_columns=1"):
        CSVTextParser(max_columns=1).parse(b"name,value\none,two")
    with pytest.raises(ValueError, match="max_output_bytes=10"):
        CSVTextParser(max_output_bytes=10).parse(b"name\none")
    with pytest.raises(ValueError, match="max_output_bytes=20"):
        CSVTextParser(max_output_bytes=20).parse(b"name\na")


@pytest.mark.parametrize(
    ("source", "message"),
    [(b"\xff", "valid UTF-8"), (b"name\nva\x00lue", "NUL")],
)
def test_csv_parser_rejects_invalid_encoding_and_nul(source: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CSVTextParser().parse(source)


@pytest.mark.asyncio
async def test_file_ingestor_indexes_csv_with_default_parser(tmp_path: Path) -> None:
    root = tmp_path / "records"
    root.mkdir()
    (root / "weather.csv").write_text(
        "station,observation\nKangerlussuaq,freezing rain reported\n",
        encoding="utf-8",
    )
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root)

    report = await ingestor.ingest_directory()

    assert report.sources == ("weather.csv",)
    assert report.document_count == 1
    results = await store.retrieve("freezing rain")
    assert len(results) == 1
    assert '"station": "Kangerlussuaq"' in results[0].text


@pytest.mark.asyncio
async def test_pdf_ingestion_keeps_page_citations_and_atomically_reindexes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "brief.pdf"
    source.write_bytes(_pdf_with_text_pages("Apollo mission overview", "Launch date confirmed"))
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = TextFileIngestor(store, root)

    assert await ingestor.ingest_file("brief.pdf", metadata={"collection": "research"}) == 2
    first = await store.retrieve("Apollo")
    second = await store.retrieve("confirmed")
    assert len(first) == len(second) == 1
    assert first[0].metadata == {
        "collection": "research",
        "chunk_count": 2,
        "chunk_index": 0,
        "page_number": 1,
    }
    assert second[0].metadata["page_number"] == 2
    assert first[0].source == second[0].source == "brief.pdf"

    source.write_bytes(_pdf_with_text_pages("Replacement research findings"))
    assert await ingestor.ingest_file("brief.pdf") == 1
    assert await store.retrieve("Apollo") == []
    assert [item.metadata["page_number"] for item in await store.retrieve("Replacement")] == [1]


@pytest.mark.asyncio
async def test_file_ingestor_can_enable_scanned_pdf_ocr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pypdf

    class Page:
        images = [SimpleNamespace(data=b"scanned-page-image")]

        def __contains__(self, key: str) -> bool:
            return False

        def extract_text(self) -> str:
            return ""

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, *_: object, **__: object) -> None:
            return None

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    root = tmp_path / "documents"
    root.mkdir()
    (root / "scan.pdf").write_bytes(b"%PDF-1.4")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(
        store,
        root,
        pdf_ocr_backend=_FakeOCRBackend(("Invoice total 420",)),
    )

    assert await ingestor.ingest_file("scan.pdf") == 1
    result = await store.retrieve("Invoice")
    assert len(result) == 1
    assert result[0].text == "Invoice total 420"
    assert result[0].source == "scan.pdf"
    assert result[0].metadata["page_number"] == 1


@pytest.mark.asyncio
async def test_file_ingestor_indexes_vector_only_pdf_ocr_with_real_pdfium(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pypdfium2")
    from PIL import Image

    class RasterAwareOCR:
        def recognize(
            self,
            content: bytes,
            *,
            max_pages: int,
            max_image_pixels: int,
            timeout_seconds: float,
        ) -> tuple[OCRPageResult, ...]:
            assert max_pages == 1
            assert timeout_seconds > 0
            with Image.open(BytesIO(content)) as image:
                pixels = image.width * image.height
            assert pixels <= max_image_pixels
            return (OCRPageResult("Vector chart label", pixels),)

    root = tmp_path / "documents"
    root.mkdir()
    (root / "chart.pdf").write_bytes(_pdf_with_flate_content(b"1 0 0 rg 72 700 400 40 re f"))
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root, pdf_ocr_backend=RasterAwareOCR())

    assert await ingestor.ingest_file("chart.pdf") == 1
    results = await store.retrieve("chart")
    assert len(results) == 1
    assert results[0].text == "Vector chart label"
    assert results[0].source == "chart.pdf"
    assert results[0].metadata["page_number"] == 1


@pytest.mark.asyncio
async def test_pdf_ingestion_rejects_malformed_and_over_limit_documents_without_replacing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "brief.pdf"
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = TextFileIngestor(store, root)
    source.write_bytes(_pdf_with_text_pages("Existing searchable evidence"))
    assert await ingestor.ingest_file("brief.pdf") == 1

    source.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="PDF header is missing"):
        await ingestor.ingest_file("brief.pdf")
    assert len(await store.retrieve("Existing")) == 1

    source.write_bytes(_pdf_with_text_pages("Page one", "Page two"))
    limited = TextFileIngestor(store, root, max_pages=1)
    with pytest.raises(ValueError, match="max_pages=1"):
        await limited.ingest_file("brief.pdf")
    assert len(await store.retrieve("Existing")) == 1

    short_output = TextFileIngestor(store, root, max_extracted_chars=4)
    with pytest.raises(ValueError, match="max_extracted_chars=4"):
        await short_output.ingest_file("brief.pdf")
    assert len(await store.retrieve("Existing")) == 1

    source.write_bytes(
        _pdf_with_flate_content(
            b"BT /F1 12 Tf 72 720 Td (" + b"A" * 1100 + b") Tj ET",
            b"BT /F1 12 Tf 72 700 Td (" + b"B" * 1100 + b") Tj ET",
        )
    )
    aggregate_limited = FileIngestor(
        store,
        root,
        max_content_stream_bytes=2048,
        max_total_content_stream_bytes=2048,
    )
    with pytest.raises(ValueError, match="max_total_content_stream_bytes=2048"):
        await aggregate_limited.ingest_file("brief.pdf")
    assert len(await store.retrieve("Existing")) == 1


@pytest.mark.asyncio
async def test_directory_ingestion_is_sorted_and_delete_cleans_removed_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    (root / "z.txt").write_text("Zebra policy", encoding="utf-8")
    (root / "nested" / "a.markdown").write_text("Alpha policy", encoding="utf-8")
    (root / "brief.pdf").write_bytes(_pdf_with_text_pages("Page one", "Page two"))
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = TextFileIngestor(store, root)

    report = await ingestor.ingest_directory()
    assert report.sources == ("brief.pdf", "nested/a.markdown", "z.txt")
    assert report.source_count == 3
    assert report.document_count == 4
    assert [item.metadata["page_number"] for item in await store.retrieve("one")] == [1]
    (root / "z.txt").unlink()
    assert await ingestor.delete_file("z.txt") == 1
    assert await store.retrieve("Zebra") == []


@pytest.mark.asyncio
async def test_custom_parser_extensions_are_used_for_directory_discovery(tmp_path: Path) -> None:
    class JsonParser:
        extensions = frozenset({".json"})

        def parse(self, content: bytes) -> tuple[ParsedPage, ...]:
            return (ParsedPage(content.decode("utf-8"), page_number=1),)

    root = tmp_path / "docs"
    root.mkdir()
    (root / "record.json").write_text('{"topic": "volcanology"}', encoding="utf-8")
    (root / "ignored.txt").write_text("not claimed", encoding="utf-8")
    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    ingestor = FileIngestor(store, root, parsers=(JsonParser(),))

    report = await ingestor.ingest_directory()

    assert report.sources == ("record.json",)
    assert report.document_count == 1
    result = await store.retrieve("volcanology")
    assert len(result) == 1
    assert result[0].metadata["page_number"] == 1


def test_file_ingestor_rejects_ambiguous_or_invalid_parser_extensions(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()

    class Parser:
        def __init__(self, extensions: frozenset[str]) -> None:
            self.extensions = extensions

        def parse(self, content: bytes) -> tuple[ParsedPage, ...]:
            return (ParsedPage(content.decode("utf-8")),)

    store = SQLiteFTS5Store(tmp_path / "knowledge.db")
    with pytest.raises(TypeError, match="lowercase dotted extensions"):
        FileIngestor(store, root, parsers=(Parser(frozenset({"JSON"})),))
    with pytest.raises(ValueError, match="multiple parsers claim"):
        FileIngestor(
            store,
            root,
            parsers=(Parser(frozenset({".json"})), Parser(frozenset({".json"}))),
        )


@pytest.mark.asyncio
async def test_file_ingestor_rejects_unsafe_or_invalid_inputs(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "ok.md").write_text("valid", encoding="utf-8")
    (root / "bad.txt").write_bytes(b"\xff")
    (root / "binary.md").write_bytes(b"a\x00b")
    (root / "large.txt").write_text("too large", encoding="utf-8")
    ingestor = TextFileIngestor(SQLiteFTS5Store(tmp_path / "knowledge.db"), root, max_file_bytes=4)

    for unsafe in ("../outside.md", "/tmp/file.md", "C:\\outside.md", "a\\b.md"):
        with pytest.raises(ValueError, match="safe relative path"):
            await ingestor.ingest_file(unsafe)
    with pytest.raises(ValueError, match="existing regular file"):
        await ingestor.ingest_file("missing.pdf")
    with pytest.raises(ValueError, match="valid UTF-8"):
        await ingestor.ingest_file("bad.txt")
    with pytest.raises(ValueError, match="NUL"):
        await ingestor.ingest_file("binary.md")
    with pytest.raises(ValueError, match="max_file_bytes"):
        await ingestor.ingest_file("large.txt")
    with pytest.raises(ValueError, match="reserved"):
        await ingestor.ingest_file("ok.md", metadata={"chunk_index": 4})
    with pytest.raises(ValueError, match="reserved"):
        await ingestor.ingest_file("ok.md", metadata={"page_number": 4})

    outside = tmp_path / "outside.md"
    outside.write_text("not in root", encoding="utf-8")
    try:
        (root / "linked.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic links"):
        await ingestor.ingest_file("linked.md")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pages", "options", "error", "message"),
    [
        (["raw text"], {}, ValueError, "ParsedPage values"),
        ([ParsedPage(text=1)], {}, ValueError, "string text"),  # type: ignore[arg-type]
        ([ParsedPage("ok", page_number=True)], {}, ValueError, "positive integers"),
        ([ParsedPage("ok", page_number=0)], {}, ValueError, "positive integers"),
        ([ParsedPage("ok", metadata=["bad"])], {}, TypeError, "metadata must be a mapping"),  # type: ignore[arg-type]
        ([ParsedPage("ok", metadata={1: "bad"})], {}, TypeError, "metadata keys must be strings"),  # type: ignore[dict-item]
        ([ParsedPage("ok", metadata={"page_number": 7})], {}, ValueError, "reserved metadata"),
        (
            [ParsedPage("first"), ParsedPage("second")],
            {"max_pages": 1},
            ValueError,
            "too many pages",
        ),
        ([ParsedPage("long")], {"max_extracted_chars": 2}, ValueError, "max_extracted_chars"),
    ],
)
async def test_file_ingestor_enforces_custom_parser_output_contracts(
    tmp_path: Path,
    pages: object,
    options: dict[str, int],
    error: type[Exception],
    message: str,
) -> None:
    class Parser:
        extensions = frozenset({".custom"})

        def parse(self, _content: bytes) -> object:
            return pages

    root = tmp_path / "docs"
    root.mkdir()
    (root / "input.custom").write_text("source", encoding="utf-8")
    ingestor = FileIngestor(
        SQLiteFTS5Store(tmp_path / "knowledge.db"),
        root,
        parsers=(Parser(),),  # type: ignore[arg-type]
        **options,
    )

    with pytest.raises(error, match=message):
        await ingestor.ingest_file("input.custom")


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", [("",), ("   ",), (None,)])
async def test_file_ingestor_rejects_invalid_chunks_from_custom_chunker(
    tmp_path: Path, chunks: tuple[object, ...]
) -> None:
    class Parser:
        extensions = frozenset({".custom"})

        def parse(self, _content: bytes) -> tuple[ParsedPage, ...]:
            return (ParsedPage("content"),)

    class EmptyChunker:
        def chunk(self, _text: str) -> tuple[object, ...]:
            return chunks

    root = tmp_path / "docs"
    root.mkdir()
    (root / "input.custom").write_text("source", encoding="utf-8")
    ingestor = FileIngestor(
        SQLiteFTS5Store(tmp_path / "knowledge.db"),
        root,
        parsers=(Parser(),),
        chunker=EmptyChunker(),
    )

    with pytest.raises(ValueError, match="chunker must return only non-empty strings"):
        await ingestor.ingest_file("input.custom")


@pytest.mark.asyncio
async def test_directory_ingestion_rechecks_total_bytes_while_reading_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.custom").write_text("a", encoding="utf-8")
    b_path = root / "b.custom"
    b_path.write_text("b", encoding="utf-8")

    class Parser:
        extensions = frozenset({".custom"})

        def parse(self, content: bytes) -> tuple[ParsedPage, ...]:
            if content == b"a":
                b_path.write_text("expanded", encoding="utf-8")
            return (ParsedPage(content.decode("utf-8")),)

    ingestor = FileIngestor(
        SQLiteFTS5Store(tmp_path / "knowledge.db"),
        root,
        parsers=(Parser(),),
        max_file_bytes=10,
        max_total_bytes=5,
    )

    with pytest.raises(ValueError, match="directory ingestion exceeds max_total_bytes=5"):
        await ingestor.ingest_directory()


@pytest.mark.asyncio
async def test_directory_ingestion_ignores_symlinked_files(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "real.txt").write_text("index this", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("do not index", encoding="utf-8")
    try:
        (root / "linked.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    ingestor = TextFileIngestor(SQLiteFTS5Store(tmp_path / "knowledge.db"), root)
    report = await ingestor.ingest_directory()

    assert report.sources == ("real.txt",)


@pytest.mark.asyncio
async def test_directory_ingestion_enforces_file_count_and_total_byte_caps(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("abc", encoding="utf-8")
    (root / "b.txt").write_text("def", encoding="utf-8")

    by_count = FileIngestor(SQLiteFTS5Store(tmp_path / "count.db"), root, max_files=1)
    with pytest.raises(ValueError, match="directory exceeds max_files=1"):
        await by_count.ingest_directory()

    by_bytes = FileIngestor(SQLiteFTS5Store(tmp_path / "bytes.db"), root, max_total_bytes=5)
    with pytest.raises(ValueError, match="directory exceeds max_total_bytes=5"):
        await by_bytes.ingest_directory()

    by_single_file_total = FileIngestor(
        SQLiteFTS5Store(tmp_path / "single.db"), root, max_file_bytes=20, max_total_bytes=2
    )
    with pytest.raises(ValueError, match="ingestion exceeds max_total_bytes=2"):
        await by_single_file_total.ingest_file("a.txt")

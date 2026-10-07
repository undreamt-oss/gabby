# ADR 0098: Bounded YAML knowledge ingestion

## Status

Accepted

## Context

YAML is a common format for operational policies, event schedules, product catalogs, research
metadata, and application data. Gabby's ingestion registry supports JSON, JSON Lines, CSV, and XML,
but treats standalone YAML files as ordinary text. Indexing source YAML verbatim makes nested fields
and their labels less useful for lexical retrieval. Reusing the Markdown front matter parser would
also apply metadata-specific semantics to complete data files.

## Decision

Add a separate `YAMLTextParser` for `.yaml` and `.yml` files. Use the existing PyYAML safe loader,
reject duplicate keys, unsupported object tags, non-JSON values, invalid mapping keys, and cyclic
aliases. Reject merge keys until their override and duplicate-key semantics can be preserved without
ambiguity. Convert dates and times to ISO-8601 strings using the existing bounded YAML value
normalizer. Preserve mapping order and render nested fields as dotted paths and list positions as
indexes, returning one searchable page for the source.

Bound input and rendered output bytes, scanner token count, node count, and collection depth. Scan
tokens before composing the node graph, then check the graph again so aliases cannot bypass the
structural limits. Continue to rely on `FileIngestor` for root confinement, per-source replacement,
and aggregate directory limits. Keep the parser replaceable through `FileIngestor(parsers=...)`.

## Consequences

- Developers can retrieve nested structured values from YAML without adding a dependency; PyYAML is
  already a core dependency for agent and skill definitions.
- Duplicate keys and unsafe tags fail ingestion instead of choosing an ambiguous or executable value.
- One source maps to one page, so citations resolve to the file; indexed text includes key paths and
  array positions for useful lexical matching.
- Anchors and aliases are supported only when acyclic and within configured structure limits.
- This adapter does not evaluate YAML templates, merge external references, or preserve comments.

## References

- [`YAMLTextParser` extension guide](../../EXTENSIONS.md)
- [PyYAML documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

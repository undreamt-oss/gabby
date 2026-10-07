# Public API stability and compatibility

Gabby treats its documented Python and HTTP interfaces as contracts for consuming applications,
model adapters, tools, skills, and deployment systems. The project is pre-1.0, so deliberate
breaking changes are possible, but they must be documented, tested, and visible to adopters.

## Public surface

The supported Python API consists of names exported from `gabby`, documented extension protocols,
and documented configuration fields. The versioned HTTP API consists of the documented `/v1`
routes, request and response schemas, streaming event types, and error behavior. Type signatures,
validation rules, lifecycle ownership, timeout and cancellation behavior, redaction guarantees, and
policy enforcement are part of these contracts.

Underscore-prefixed modules and members, private runtime details, and undocumented behavior are
implementation details. Consumers should not depend on them. A missing capability should be raised
as a public contract proposal instead.

## Versioning and extension contracts

Before 1.0, a breaking API change may ship in a minor release when the migration is documented.
Patch releases should remain compatible whenever practical. After 1.0, breaking public behavior
requires a major release and migration notes. A security fix may require an exceptional compatibility
change when retaining the old behavior would leave users unsafe; explain that exception in the
changelog and release notes.

The package version and extension contract versions are separate concerns. If a stable extension
contract is introduced, version it independently from the package and document its compatibility
rules. Until then, extension interfaces remain pre-1.0 and may evolve with documented migration
guidance.

## Deprecation process

When replacing a public name or behavior:

1. Document the replacement and reason in the owning guide.
2. Add a changelog entry and migration instructions.
3. Retain the old API when safe and emit a targeted `DeprecationWarning` from its deprecated path.
4. Add regression coverage for the replacement and warning.
5. Remove the deprecated API only at a release boundary allowed by the versioning policy.

Deprecation warnings must not expose credentials, prompts, provider payloads, tool arguments, or
unbounded values. If a security issue makes a deprecation period unsafe, explain the exception.

## Review requirements for contract changes

Changes to public Python APIs, configuration, HTTP behavior, or extension points require:

- focused regression coverage through the public interface;
- updated type annotations, docstrings, API reference, and the closest concept guide;
- migration notes when callers or operators can observe the change;
- an ADR when ownership, lifecycle, concurrency, hosting, dependency direction, or security
  assumptions change;
- validation across supported Python versions and relevant platform or hosting paths.

Keep provider-specific SDKs and credentials in adapters. A new provider must not silently change
provider-neutral runtime contracts.

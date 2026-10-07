# Security policy

Please report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/orbit-projects/gabby/security/advisories/new) rather
than opening a public issue. If private vulnerability reporting is unavailable, contact the project
maintainers privately before sharing exploit details.

Include the affected version or commit, component, a minimal technical description, security impact,
and any mitigation identified. Redact credentials, personal data, customer content, and production
endpoints from reports.

Gabby is pre-release software. Maintainers will acknowledge reports as soon as practical and
coordinate a fix and disclosure with the reporter. Support for a version is stated in its release
notes. Security behavior in provider adapters, tool handlers, operating-system sandboxes, and hosting
environments remains the responsibility of those implementations unless Gabby explicitly documents
and verifies a guarantee.

The HTTP run endpoint requires an authenticator unless an embedded caller explicitly opts into
unauthenticated use. The built-in bearer-token authenticator is single-tenant and is not a substitute
for tenant authorization or TLS. Do not expose an unauthenticated app publicly; provide a trusted
authenticator and terminate TLS at the service boundary.

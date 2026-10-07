# Gabby governance

Gabby is developed in the open as part of the Orbit parent organization. It follows the same
baseline contribution, review, security, and release practices as Orbit Core while retaining
project-specific architecture and release gates. Anyone may propose an issue, design, documentation
change, or pull request. Contributions follow the [Code of Conduct](CODE_OF_CONDUCT.md) and
[contribution guide](CONTRIBUTING.md); vulnerabilities follow [SECURITY.md](SECURITY.md) and must
not be disclosed in public issues.

## Participation and review

Changes are proposed through public pull requests and reviewed against Gabby's documented
architecture, API contracts, and security boundaries. Maintainers should seek evidence-backed
consensus in the review. Pull requests should include relevant tests and documentation and identify
any security, compatibility, or deployment assumptions that remain unverified. Branch protection
and required checks are described in the [repository governance guide](docs/REPOSITORY_GOVERNANCE.md);
repository settings enforce those rules and cannot be replaced by workflow files alone.

## Architecture decisions

The project architect sets product direction and makes final decisions on major architecture and
public-contract changes. Contributors should raise those changes before implementation when
practical. A new or revised architectural convention requires an ADR describing the context,
decision, alternatives, consequences, and compatibility implications. Routine implementation
choices can be resolved in review when they do not change an accepted contract.

Backward-incompatible public API changes must identify affected callers, a migration path, and the
deprecation and release impact in line with the [API stability policy](docs/API_STABILITY.md).
Decisions may be revised through a new ADR when implementation or operational evidence changes.

## Maintainers and releases

The maintainer directory is in [MAINTAINERS.md](MAINTAINERS.md). Maintainers review contributions,
triage issues and security reports, protect package and repository settings, and keep CI, release
automation, documentation, and the changelog accurate. Releases follow [RELEASING.md](docs/RELEASING.md)
and must be reproducible from the committed lockfile and pass the documented gates. An open release
gate is not waived by this policy. Publishing a release does not certify a particular provider,
sandbox host, or production deployment.

The public [roadmap](ROADMAP.md) records planned work and the
[project scorecard](docs/PROJECT_SCORECARD.md) records evidence-backed readiness. Changes to this
policy should be proposed publicly and reviewed by maintainers; changes to repository protections,
security boundaries, or release responsibilities should also update the relevant guide and include
an ADR when they establish or change an architectural convention.

# ADR 0020: Tag-based PyPI releases with Trusted Publishing

- Status: Accepted
- Date: 2026-09-29

## Context

Gabby is intended to be an open-source Python package. Releases need a repeatable verification and
publishing path, with narrow credentials and a clear maintainer approval point.

## Decision

Version tags matching the version in `pyproject.toml` trigger the release workflow. The workflow
runs the supported Python and host-platform CI gates, confirms the tag commit is reachable from
`main`, and builds the wheel and source distribution only after those gates pass. A separate
publishing job uploads those artifacts through GitHub OIDC Trusted Publishing to PyPI. The publisher
job uses the protected GitHub environment named `pypi`, and has `id-token: write` only within that
job. GitHub environment approval is required before publication.

## Consequences

- No long-lived PyPI API token is stored in repository secrets.
- Release artifacts are built and checked before the job that can request a publishing identity.
- Tagging a commit does not publish until every verification job succeeds and a maintainer approves
  the protected environment deployment.
- Repository administrators must configure the GitHub environment and the matching PyPI Trusted
  Publisher; workflow files cannot create either external setting.
- The workflow does not create GitHub Releases or move tags. Maintainers publish release notes
  after PyPI publication.

## Alternatives considered

- Publish from every merge to `main`: rejected because it removes the explicit version and review
  point for user-visible releases.
- Store a PyPI API token in GitHub secrets: rejected because OIDC Trusted Publishing avoids a
  long-lived publishing credential.
- Build artifacts inside the publishing job: rejected because the job with publishing identity
  should only upload already-built artifacts.

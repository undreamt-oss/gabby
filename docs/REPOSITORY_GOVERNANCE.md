# Repository governance and branch protection

Gabby uses reviewed pull requests, automated checks, and protected branches to maintain the
quality and security of its open-source releases. GitHub rulesets live in repository settings;
this document records the intended policy so maintainers can review and reproduce it.

Gabby follows the parent organization's shared OSS baseline used by Orbit Core: Apache-2.0
licensing, public contribution and conduct policies, private vulnerability reporting, reviewed pull
requests, issue and pull-request templates, `CODEOWNERS`, Dependabot, pinned GitHub Actions,
CodeQL, OpenSSF Scorecard, a supported-Python CI matrix, lockfile-based development, and
OIDC-backed PyPI Trusted Publishing. Project-specific workflows can add checks required by
Gabby's agent, sandbox, and provider boundaries; they should preserve the common permissions,
review, supply-chain, and release controls.

## Recommended default-branch rules

Protect the default branch with an active ruleset and keep its bypass list empty during normal
operation. Limit emergency bypass access to maintainers and document each use in the related
incident or release record.

| Rule | Policy | Reason |
| --- | --- | --- |
| Require a pull request | Enabled | Changes receive review and automated evidence before merging. |
| Require status checks | Enabled after CI reports stable check names | Each change must pass the Python/platform matrix and quality gates. |
| Block force pushes and deletion | Enabled | Reviewed history and release references remain auditable. |
| Require linear history | Enable if squash or rebase merging is the project norm | Keeps the public history straightforward to review. |
| Require code-owner review | Require review from `@orbit-projects/maintainers`, as declared in `.github/CODEOWNERS` | Keeps review ownership consistent across projects in the parent organization. |
| Require signed commits | Adopt progressively | Signing adds provenance but needs a documented contributor setup. |
| Require deployments or merge queue | Disabled until a real environment or queue is maintained | Do not require checks that have no reproducible service behind them. |

After CI is installed, open a successful pull request and select the exact status-check names shown
by GitHub. Do not guess check names. Require the Python 3.11, 3.12, 3.13, and 3.14 quality jobs;
enable CodeQL and Scorecard as merge checks only after their pull-request results are reliable.
Repository settings cannot be enforced by workflow files alone. Confirm the shared maintainers team
exists and has write access to this repository before making code-owner approval a required rule.
Track this and other repository-owned controls in the [maintainer handoff](../HANDOFF.md).

## Workflow and supply-chain policy

Pull-request and release workflows should retain the following controls:

- lockfile validation and supported-Python coverage;
- lint, formatting, strict type checking, tests, and the configured coverage threshold;
- documentation, license, package-integrity, and dependency-audit checks;
- CodeQL and OpenSSF Scorecard on pull requests, default-branch updates, and scheduled runs;
- explicit least-privilege workflow and job permissions, pinned third-party actions, job timeouts,
  and checkout steps with persisted Git credentials disabled;
- isolated release builds and a separate PyPI Trusted Publishing job using GitHub OIDC.

A successful workflow proves only that its checks passed for the tested source. It does not certify
an untested model provider, sandbox host, or production deployment. Record external setup such as
branch rules, protected release environments, and PyPI Trusted Publisher configuration in the
[release guide](RELEASING.md) and [maintainer handoff](../HANDOFF.md).

## Contributor expectations

Pull requests should identify the Gabby boundary that owns the change and include focused regression
coverage, relevant documentation, and public API docstrings. Architectural changes need an ADR.
Changes to branch protection or release evidence should update this governance guide. Keep
credentials, generated environments, coverage output, and unrelated formatting changes out of
commits.

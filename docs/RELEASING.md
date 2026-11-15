<!--
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2026-present Gabby Contributors.
-->

# Releasing Gabby

Gabby publishes the `gabby-ai` distribution to PyPI through GitHub Actions Trusted
Publishing. Release builds run the platform and quality checks before building the wheel and source
distribution. The isolated build job creates a signed SLSA provenance attestation for both files and
generates a CycloneDX SBOM from the locked runtime dependency graph. GitHub's OIDC-backed Sigstore
integration attests that SBOM against both distributions. The SBOM is retained as a separate
workflow artifact so the PyPI publisher receives only installable distributions. The publishing job
receives a separate OIDC token only after those checks pass and the `pypi` GitHub environment
approves the deployment.

The first alpha was published as PEP 440 version `0.1.0a1`, corresponding to Git tag
`v0.1.0a1`. The next planned beta uses version `0.1.0b5` and matching tag `v0.1.0b5`.
Both are Python pre-releases; package installers may require consumers to opt in to pre-release
versions. The final `0.1.0` version remains reserved for a later stable release after beta feedback
and compatibility decisions are addressed.

## One-time repository setup

1. Create a GitHub environment named `pypi` and require maintainer approval before deployment.
   Restrict deployments to version tags matching `v*`.
2. Configure PyPI Trusted Publishing for the `gabby-ai` project with this GitHub owner,
   repository, workflow filename `.github/workflows/release.yml`, and environment `pypi`.
3. Protect the `main` branch and version tags so only maintainers can change release inputs.
4. Do not add a PyPI API token to GitHub secrets. The publishing job uses GitHub OIDC.

See PyPI's [Trusted Publishers guide](https://docs.pypi.org/trusted-publishers/) and the
[GitHub Actions OIDC guide for PyPI](https://docs.github.com/en/actions/how-tos/secure-your-work/security-harden-deployments/oidc-in-pypi)
for configuring the external trust relationship.

## Release procedure

1. Update `project.version` in `pyproject.toml` and add the release heading and notes to
   `CHANGELOG.md`.
2. Merge the release commit into `main` and wait for normal CI to pass.
3. Create and push the matching annotated tag, such as `v0.1.0b5`, on that commit. Python
   pre-releases use PEP 440 versions so installers can distinguish them from final releases and
   normally require an explicit pre-release opt-in.
4. The Release workflow repeats platform tests and quality gates for the tagged source, confirms the
   tag is on `main` and matches `project.version`, builds wheel and source distribution files,
   creates signed provenance attestations for both distributions, and generates and attests a
   CycloneDX SBOM from the production dependency graph in `uv.lock`.
5. Review and approve the deployment in the protected `pypi` environment. The publisher uploads
   only the prebuilt distributions and does not check out or execute project code.
6. Download each PyPI distribution and verify its GitHub attestation, for example:

   ```sh
   gh attestation verify dist/gabby_ai-<version>-py3-none-any.whl --repo OWNER/REPO
   gh attestation verify dist/gabby_ai-<version>.tar.gz --repo OWNER/REPO
   ```

   Download the `gabby-runtime-sbom` artifact from the same Release workflow run and verify that its
   SBOM attestation is associated with each distribution:

   ```sh
   gh attestation verify dist/gabby_ai-<version>-py3-none-any.whl \
     --repo OWNER/REPO --predicate-type https://cyclonedx.org/bom
   gh attestation verify dist/gabby_ai-<version>.tar.gz \
     --repo OWNER/REPO --predicate-type https://cyclonedx.org/bom
   ```

   Confirm the PyPI release and its provenance, then create the corresponding GitHub release with
   the changelog notes.

The SBOM describes the resolved core and `server` extra dependency graph selected by the lockfile
and includes Gabby's package as the root component. It does not describe optional development tools
or prove that dependencies are safe. `uv export` emits CycloneDX 1.5 through its SBOM export
feature; CI checks that this export remains part of the release workflow.

The workflow does not create or move Git tags, edit the changelog, or create a GitHub release. A
failed gate blocks publishing; correct the source and issue a new version tag rather than reusing a
published version.

See GitHub's [`actions/attest` documentation](https://github.com/actions/attest),
[`uv export` documentation](https://docs.astral.sh/uv/concepts/projects/export/), and
[`gh attestation verify` manual](https://cli.github.com/manual/gh_attestation_verify) for
attestation generation and verification behavior.

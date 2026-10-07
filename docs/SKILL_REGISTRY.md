# Gabby skill registries

Gabby supports static HTTPS skill registries. A registry can be hosted from object storage, a static
web server, or a Git repository site; it does not need a Gabby server process. Discovery metadata is
public and untrusted. Every remotely fetched package must have a detached Ed25519 signature from a
public key the consuming host already trusts.

## Registry layout

The registry root contains a versioned catalog and immutable package artifacts:

```text
registry-root/
└── v1/
    ├── catalog.json
    └── skills/
        └── support/
            └── triage/
                └── versions/
                    └── 1.2.0/
                        ├── package.gabskill
                        └── package.gabskill.sig
```

`catalog.json` has this exact JSON shape:

```json
{
  "format": "gabby-skill-catalog",
  "format_version": 1,
  "generated_at": "2026-10-03T12:00:00Z",
  "skills": [
    {
      "name": "support/triage",
      "description": "Sort incoming support cases",
      "versions": ["1.2.0", "1.1.0"]
    }
  ]
}
```

`generated_at` is an optional RFC 3339 timestamp emitted by `gabby skill catalog build`; older
catalogs without it remain readable unless the client requires a maximum age. IDs and versions are
validated against Gabby's skill definition rules. IDs are case-sensitive. The catalog is limited to
4 MiB, 10,000 skills, 1,000 versions per skill, and 16 KiB per description.
Each package and its adjacent signature are served from the fixed paths above. Catalog values never
provide arbitrary package URLs.

## Inspect packages before installation

Review package metadata and file digests without installing it into a skill registry:

```sh
gabby skill inspect ./dist/triage-1.2.0.gabskill
gabby skill inspect ./dist/triage-1.2.0.gabskill \
  --trusted-key-dir ./trusted-publishers --require-signature
```

The command emits bounded JSON with the validated skill ID, version, description, declared tools,
knowledge references, dependencies, verification labels, archive digest, and package file paths,
sizes, and digests. It validates and extracts only in private temporary storage; instructions and
examples are not returned. Without trusted keys, a present signature is reported but not
authenticated. Use `--require-signature` with the publisher trust map when the package must be
authenticated. A trusted signature proves which configured key signed the archive; it does not
make the skill's instructions safe or authorize its declared tools.

Publishers create and sign immutable artifacts locally:

```sh
gabby skill pack ./skills/support/triage --output ./dist/package.gabskill
gabby skill sign ./dist/package.gabskill --key-id support-publisher \
  --private-key-env GABBY_SKILL_SIGNING_KEY
```

Provision `GABBY_SKILL_SIGNING_KEY` from the publisher's secret manager before running the signing
command; do not put private key values in scripts, shell history, or registry files.

Build the static registry files from one or more signed package archives:

```sh
uv sync --frozen --extra skill-signing
gabby skill catalog build \
  --package ./dist/triage-1.2.0.gabskill \
  --package ./dist/triage-1.1.0.gabskill \
  --trusted-key-dir ./trusted-publishers \
  --output ./dist/registry

# For the next release, preserve cataloged versions from the prior generated tree.
gabby skill catalog build \
  --existing ./dist/registry \
  --package ./dist/triage-1.3.0.gabskill \
  --trusted-key-dir ./trusted-publishers \
  --output ./dist/registry-next
```

The command verifies each archive and adjacent `.sig` file against the supplied public keys, checks
the skill identity and version from the signed package, and writes the catalog and exact artifact
paths shown above. Every output version must be unique; the output directory must not already exist.
The command caps the input set at 10,000 packages, the combined archives at 1 GiB, and the catalog
at 4 MiB. Serve the generated directory as the registry root so `v1/catalog.json` is available at
`/v1/catalog.json`. Gabby generates files only; static hosting, access control, and atomic deployment
remain with the publisher.

When `--existing` is supplied, Gabby reads the prior bounded catalog, rejects symlinked or special
files, verifies every referenced archive against the same trusted publisher keys, and includes only
the cataloged exact versions in the new tree. The destination is still a new directory; publishers
can validate it and switch their static-host release pointer after the build succeeds. Existing
artifacts with mismatched paths, signatures, or package identities abort the build without
publishing the destination.

The private signing key stays with the publisher and must not be placed in the registry. Deploy
generated version artifacts and catalog as one release where the hosting system supports atomic
directory or object-set updates. A registry may retain older signed versions; clients select exact
versions and do not resolve mutable `latest` aliases.

## Discover and install

Install the optional signing support and configure the publisher public key through the host:

```sh
uv sync --frozen --extra skill-signing
gabby skill search support --registry-url https://skills.example.org
gabby skill versions support/triage --registry-url https://skills.example.org
gabby skill fetch support/triage 1.2.0 \
  --registry-url https://skills.example.org \
  --local-registry ./skills \
  --trusted-key-dir ./trusted-publishers
gabby skill fetch research/weekly-report 2.0.0 --with-dependencies \
  --registry-url https://skills.example.org \
  --local-registry ./skills \
  --trusted-key-dir ./trusted-publishers
gabby skill audit --registry ./skills --trusted-key-dir ./trusted-publishers \
  --revoked-key old-support-publisher
gabby skill uninstall support/triage 1.2.0 --registry ./skills --yes
```

The trust directory is host-managed and contains raw 32-byte Ed25519 public keys named
`KEY_ID.pub`. Gabby rejects symlinks, subdirectories, unexpected files, malformed IDs, empty
directories, and more than 256 keys. Embedded hosts can load the same directory with
`load_skill_trust_keys(path)` or construct `SkillTrustPolicy.from_directory(path)`; the policy
copies the key mapping at construction. For rotation, add the new public key and reconstruct agents
before publishing skills signed by it, then remove the old key and reconstruct after migration.
Emergency revocation uses `SQLiteSkillRevocationStore` or a host-provided
`SkillRevocationChecker`, which can cancel active runs without waiting for agent reconstruction.
SQLite is a single-host option and requires a local filesystem with SQLite locking semantics. For
multiple Gabby hosts, inject a shared checker (or implement `SkillRevocationStore` for a backend
that also needs management operations). That backend must define durable writes and a bounded
read-after-write propagation guarantee; every instance polls it during active runs, so detection
also includes each agent's configured poll interval and checker latency. Trusted public keys remain
deployment configuration and must be rotated consistently across instances before using a new signer.
`RedisSkillRevocationStore` is a concrete Redis 6.2+ shared-set adapter for a host-owned async Redis
client; it does not distribute publisher public keys. Configure persistence and route checks to an
authority that meets the deployment's read-after-write consistency requirement. See the
[Redis revocation example](../examples/redis_skill_revocation.py).

Private registries may receive a bearer token from an environment variable with `--token-env`; the
token is never read from the catalog or written to agent configuration. Python hosts can use the
async `SkillRegistryClient` and provide headers through a secret manager. Both interfaces require
an HTTPS URL for remote registries. HTTP is permitted for loopback development only.
The HTTPX client honors system proxy settings by default; hosts can pass `trust_env=False` when
network routing must be explicit. To reject catalogs older than seven days, configure
`max_catalog_age_seconds=604800` on `SkillRegistryClient` or pass
`--max-catalog-age-seconds 604800` to `gabby skill search`, `versions`, or `fetch`. Requiring an
age rejects legacy catalogs without `generated_at`; timestamps more than five minutes in the future
are always rejected. The timestamp is unsigned metadata, so age checking limits accidental stale
publication or caching and does not authenticate the catalog or prevent a malicious publisher from
advertising older signed versions. Pin exact versions when a specific release is required.

The client rejects redirects, credentials and query parameters in the registry URL, malformed
catalogs, duplicate JSON keys, mismatched package identities, oversized responses, and requests that
exceed their configured deadline. It streams package downloads to private temporary files. Remote
installation always verifies the detached signature before writing to the caller's skill registry.
The client verifies and validates the package identity in a private scratch registry before making
the final install, then installs the same signed archive through Gabby's normal atomic installer.
Each installation records its archive digest and detached signature evidence in a reserved
`.gabby-install.json` file. New v2 signatures bind both the archive digest and the canonical package
manifest. Give `gabby skill audit` the same host-owned public keys used for installation; it hashes
the installed files, checks them against the signed manifest, and verifies the Ed25519 signature.
The command emits one JSON record per installed skill, and exits with status 1 when a record is
revoked, unknown, or invalid. Without `--trusted-key`, signed records are labeled
`recorded-signed`: this means installation-time verification was recorded, but current files were
not verified. V1 signatures remain install-verifiable but cannot prove the current installed tree.

`skill fetch --with-dependencies` resolves an exact signed dependency closure from one catalog,
validates every package before changing the local registry, then installs dependencies before the
requested skill. Remote dependency fetching requires each dependency to use the exact
`skill-id@version` form; unpinned references fail instead of selecting a moving version. A closure
is limited to 256 skills, 1 GiB of archives, and five minutes. Each package install is atomic and
already-installed versions are reused only when the trusted signature, installed content, and
archive digest match. A process or filesystem failure during the final install sequence can leave a
valid prefix installed; rerunning the same request resumes that exact closure safely. Conflicting,
tampered, or differently signed exact versions are rejected.

The trust map is host-owned. A valid v2 signature proves that one trusted key signed the exact
archive bytes and its package manifest; the audit checks the current installed tree against that
manifest. The install provenance and local registry remain writable by the registry owner, so audit
is an integrity check against trusted key material, not a tamper-proof log or runtime authorization
decision. Catalog metadata may be replayed or altered to hide skills or advertise older signed
versions; hosts that require a specific release must pin and install its exact version. Review skill
instructions and assign only the tools and knowledge sources the installed agent should use. Trust
key provisioning, rotation, revocation, registry publication authorization, and server-side
availability remain deployment responsibilities.

For service deployments that must reject untrusted or changed skills during agent construction,
inject a host-owned policy:

```python
from gabby import Agent, SkillTrustPolicy
from gabby.config import load_agent

policy = SkillTrustPolicy(
    trusted_keys={"support-publisher": public_key_bytes},
    revoked_key_ids=frozenset({"old-support-publisher"}),
)
agent = Agent(load_agent("agent.yaml"), skill_trust_policy=policy)
```

With this policy, every resolved filesystem skill must have valid v2 install provenance from a
trusted publisher. Unsigned, legacy v1, in-memory, or modified skills fail construction, and
installed content is revalidated before each run and stream. Integrity failures stop execution
before sandbox, model, or tool work and are returned as sanitized 403 errors. Keys and the local
revoked-key set are snapshotted on the policy; inject a `SkillRevocationChecker` alongside it when
already-running services must observe new revocations during each run and stream. A revoked key
denies newly admitted executions, while host code decides how to cancel active runs. Reconstruct
agents after changing trusted key material. Keep skill registry files stable while constructing an
agent and during its runs; per-run checks do not provide an atomic snapshot against concurrent host
mutations. Local development agents can omit the policy.

## Rotate or revoke a publisher key

Keep the trust map in the consuming deployment's secret/configuration system, separate from the
registry. A key ID identifies a public key in that map; changing the key bytes under an existing ID
is not a rotation because signatures bind the ID and key material. Assign a new, never-reused ID to
each replacement key.

For a planned rotation:

1. Generate the replacement key in the publisher's secret manager and distribute its public key to
   consumers under a new key ID. Keep the old public key trusted during the transition.
2. Sign new package versions with the new key. For older versions that must remain installable,
   re-sign the unchanged package archive with the new key and replace its `.sig` sidecar. Publish
   those sidecars and the matching catalog as one atomic registry deployment where the hosting
   system supports atomic releases.
3. Verify representative exact-version installs using the new key from every consuming deployment.
   Then remove the old private key from signing systems and remove its public key from consumer trust
   maps after all retained artifacts have a new signature.

For a compromised key, remove its key ID from consumer trust maps immediately and stop publishing
its signatures. This blocks future installs that rely on the removed key; it does not disable skill
directories already installed in a local registry. Run the following for each revoked ID, then
disable or remove the reported versions before redeploying affected agents:

```sh
gabby skill audit --registry ./skills --revoked-key KEY_ID
```

Installations created before provenance metadata was introduced are reported as `unknown`; review
or reinstall those packages rather than assuming they are safe.

Remove an affected exact version with `gabby skill uninstall NAME VERSION --registry PATH --yes`.
The command validates the requested skill identity from `skill.yaml` before deletion and never
removes sibling versions. Update agent definitions and restart or reconstruct agents that referenced
the removed version; existing `Agent` instances retain their already-resolved immutable skill plan.

Re-sign only package archives that have been independently reviewed and approved, publish their
replacement sidecars, and distribute the replacement public key through the host's trusted
configuration channel. Do not accept a replacement key merely because the registry advertises it.

The current signature format has no trusted signing timestamp and the client accepts one adjacent
signature sidecar per package. Consequently, trust removal rejects all packages signed by that key,
including older signatures; the format cannot distinguish pre-compromise signatures from later
ones. The local provenance file is an inventory aid, not tamper-proof evidence: an actor who can
change the skill registry can also change the metadata. Treat publisher key backups, key ID
ownership, registry deployment access, and consumer trust updates as one operational control, and
rehearse the rotation before relying on it for a production registry.

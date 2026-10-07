# Container deployment reference

This reference builds and serves the repository's hosted research agent with Docker Compose. It
provides a concrete starting point for a single-tenant API deployment. The example agent accepts
caller-supplied context and has no tools, knowledge database, or writable workspace. It does not
mount the host container-engine socket.

## Run locally

Provide credentials through the host secret manager or shell environment:

```sh
export HF_TOKEN='...'
export GABBY_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml up -d
```

The image installs the project from `uv.lock`, copies the example agent and skill, runs as UID
10001, and uses a read-only root filesystem with a bounded temporary filesystem. Compose publishes
port 8787 on loopback only; place a TLS-terminating reverse proxy or deployment ingress in front
of it. Set `GABBY_UV_IMAGE` and `GABBY_RUNTIME_IMAGE` to reviewed image references ending in
`@sha256:<digest>` before building a release deployment; Compose forwards these values to the
Dockerfile's `UV_IMAGE` and `RUNTIME_IMAGE` build arguments. If unset, the development defaults use
version-tagged uv and Python 3.14 Trixie slim images.

Check service readiness and authentication without making a model request:

```sh
curl --fail http://127.0.0.1:8787/health
curl --include http://127.0.0.1:8787/v1/agents/research-synthesizer/run \
  -H 'Content-Type: application/json' \
  --data '{"input":"Summarize this","context":{}}'
```

The second request should receive HTTP 401 because it omits the bearer token. Add
`-H "Authorization: Bearer $GABBY_API_TOKEN"` to make a real provider request. A healthy `/health`
response confirms only that the HTTP application is responding; it does not check provider access.

Stop and remove the example service with:

```sh
docker compose -f deploy/compose.yaml down
```

## Production use

Replace the example agent, skill, provider settings, image references, and resource bounds with
deployment-owned values. Inject provider credentials and the API token through a secret manager.
Keep TLS termination, request-rate and connection limits, tenant routing, process supervision,
log and trace retention, and release image scanning at the deployment boundary. The included
Compose file is a reference configuration, not evidence that a deployment has been certified.

This image is intentionally for the hosted research example. Coding agents that use Gabby's
container-backed shell and filesystem tools need a separately designed engine-control boundary.
Do not add a host Docker or Podman socket mount to this Compose file: access to that socket grants
the service broad control over the host container engine. Validate the chosen engine service,
identity, image policy, network isolation, and cleanup controls for the actual deployment.

See the [operations guide](OPERATIONS.md) and [threat model](THREAT_MODEL.md) for service limits,
ingress controls, and residual risks.

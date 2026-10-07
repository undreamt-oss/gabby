# ADR 0107: Kubernetes hosted-service reference

## Status

Accepted

## Context

Gabby provides a local Compose reference for a single-tenant hosted research agent. Kubernetes users
need a deployable baseline that keeps secrets and TLS deployment-owned, limits pod privileges and
resources, and exposes the HTTP API through an ingress without granting the agent container-engine
or Kubernetes API access.

## Decision

Provide `deploy/kubernetes/research-agent.yaml` as a single-replica reference for the bundled
research agent. It includes a digest-pinned image placeholder, non-root UID/GID 10001, read-only root
filesystem, dropped Linux capabilities, seccomp `RuntimeDefault`, no service-account token, bounded
CPU/memory and temporary storage, HTTP startup/readiness/liveness probes, ClusterIP service, TLS
NGINX Ingress, and a NetworkPolicy that permits ingress only from the configured controller and
egress to cluster DNS and TCP 443.

Credentials and TLS private keys are supplied as pre-created Kubernetes Secrets and are not
committed in manifests. The ingress uses standard ingress-nginx annotations. The NetworkPolicy
requires an enforcing CNI and cluster-specific labels. Kubernetes NetworkPolicy cannot restrict
HTTPS destinations by hostname, so FQDN provider egress policy belongs in a CNI feature or egress
gateway. The example is single-replica because SSE replay and request admission are process-local.

## Consequences

- The repository contains a concrete Kubernetes starting point in addition to its local Compose
  deployment.
- Operators must supply immutable image digests, credentials, certificates, ingress controller,
  cluster selectors, provider egress policy, and target-cluster validation.
- The manifest does not claim cluster acceptance or production certification.
- Horizontal scaling requires deliberate SSE affinity, aggregate capacity, and quota configuration.

## Validation

Repository contract tests parse all five resources and verify the security context, secret refs,
resource bounds, TLS ingress, service exposure, and network-policy ports. Live Kubernetes admission,
ingress behavior, and CNI enforcement remain target-environment acceptance work.

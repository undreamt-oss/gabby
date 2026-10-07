# Kubernetes deployment reference

This reference deploys the bundled single-tenant research agent behind an NGINX Ingress Controller.
It keeps one replica, does not mount a container-engine socket, runs as the image's non-root UID,
and uses a read-only root filesystem, dropped capabilities, bounded CPU/memory/temp storage, health
probes, and a pod network policy. The service is an example deployment template; Kubernetes and
cluster-specific acceptance has not been run by this repository.

## Prepare the cluster

Install a NetworkPolicy-enforcing CNI and an NGINX Ingress Controller. The policy currently expects
the controller in the `ingress-nginx` namespace with label
`app.kubernetes.io/component=controller` and cluster DNS pods in `kube-system` labeled
`k8s-app=kube-dns`; adjust these selectors for your cluster. Kubernetes NetworkPolicy cannot restrict
HTTPS by hostname. This template permits TCP 443 egress for the hosted model provider. Use a CNI or
egress gateway with FQDN policies when the service must contact only specific provider domains.

Build and push the hosted research image from the repository root, then obtain the registry's
immutable digest:

```sh
docker build -t registry.example.org/gabby/research-agent:VERSION .
docker push registry.example.org/gabby/research-agent:VERSION
docker inspect --format='{{index .RepoDigests 0}}' \
  registry.example.org/gabby/research-agent:VERSION
```

Replace the image reference in `deploy/kubernetes/research-agent.yaml` with that full digest-pinned
reference. For a private registry, create an image-pull secret and add it to the pod spec.

Create the secret outside the manifest through your secret manager or deployment tooling. The
example expects two protected files created by that mechanism, readable only to the deploying
identity:

```sh
kubectl create namespace gabby-research
kubectl -n gabby-research create secret generic gabby-research-secrets \
  --from-file=GABBY_API_TOKEN=/run/secrets/gabby-api-token \
  --from-file=HF_TOKEN=/run/secrets/hf-token
kubectl -n gabby-research create secret tls gabby-research-tls \
  --cert=fullchain.pem --key=privkey.pem
```

Change `research.example.org` in the Ingress to a domain covered by the TLS secret. Apply the
reference and inspect readiness:

```sh
kubectl -n gabby-research apply -f deploy/kubernetes/research-agent.yaml
kubectl -n gabby-research rollout status deployment/gabby-research
kubectl -n gabby-research get pods,service,ingress,networkpolicy
```

The Ingress uses NGINX annotations for HTTPS redirect, request-size, streaming, timeouts, and
per-IP connection/request limits. The body limit is slightly above Gabby's 1,000,000-byte API limit;
the application enforces the exact bound. These controls require the NGINX Ingress Controller and
its standard annotation prefix. The default ingress-nginx controller returns 503 for its rate and
connection limits; operators can set `limit-req-status-code: "429"` and
`limit-conn-status-code: "429"` in the controller ConfigMap if they want edge limits to match
Gabby's 429 overload response. Those ConfigMap values affect the controller's other Ingress rules
too. The `/health` probes check the HTTP process only, not provider reachability. A failed dependency
should be diagnosed through service telemetry and a real authenticated model request.

## Isolation and scaling

The NetworkPolicy allows inbound traffic only from the selected ingress controller and egress only
to cluster DNS and TCP 443. It does not implement FQDN allowlisting; cluster operators must provide
that boundary if required. It also depends on a CNI that enforces Kubernetes NetworkPolicy. Review
pod selectors, namespace labels, IPv6 behavior, DNS labels, ingress controller settings, and cloud
egress controls for the target cluster.

The manifest uses one replica. Gabby remains stateless per request, but SSE replay journals and
concurrency limits are process-local. If you scale horizontally, configure ingress affinity for
reconnecting streams and size total concurrency, provider quotas, and resource limits across all
replicas. The readiness and liveness probes do not test the model provider. Set rollout and
termination timing against the agent deadline and your own shutdown target.

Do not grant this service Kubernetes API permissions or mount a Docker/Podman socket for the
research example. It does not need them. The manifest supplies platform defaults, not cluster
certification; rehearse secret rotation, provider outage, ingress 429 handling, resource pressure,
and termination in the target environment. For local single-host deployment, see the
[Docker Compose reference](DEPLOYMENT_CONTAINER.md).

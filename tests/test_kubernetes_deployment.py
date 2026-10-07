# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contract checks for the single-tenant Kubernetes deployment reference."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def _manifests() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parents[1] / "deploy" / "kubernetes" / "research-agent.yaml"
    documents = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
    assert len({doc["kind"] for doc in documents}) == len(documents)
    return {doc["kind"]: doc for doc in documents}


def test_kubernetes_example_keeps_single_tenant_service_isolated_and_bounded() -> None:
    manifests = _manifests()
    assert set(manifests) == {"ServiceAccount", "Deployment", "Service", "Ingress", "NetworkPolicy"}

    account = manifests["ServiceAccount"]
    assert account["automountServiceAccountToken"] is False

    deployment = manifests["Deployment"]
    assert deployment["spec"]["replicas"] == 1
    pod = deployment["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "gabby-research"
    assert pod["automountServiceAccountToken"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001
    assert pod["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"

    container = pod["containers"][0]
    image = container["image"]
    assert "@sha256:" in image
    digest = image.rsplit("@sha256:", maxsplit=1)[1]
    assert digest == "REPLACE_WITH_64_HEX_DIGEST" or re.fullmatch(r"[a-f0-9]{64}", digest)
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert container["resources"]["requests"] == {"cpu": "250m", "memory": "512Mi"}
    assert container["resources"]["limits"] == {"cpu": "1", "memory": "1Gi"}
    assert container["volumeMounts"] == [{"name": "tmp", "mountPath": "/tmp"}]
    assert pod["volumes"][0]["emptyDir"]["sizeLimit"] == "16Mi"
    assert {probe for probe in ("startupProbe", "readinessProbe", "livenessProbe")} <= set(
        container
    )
    env = {item["name"]: item for item in container["env"]}
    assert set(env) == {"GABBY_API_TOKEN", "HF_TOKEN"}
    assert all("secretKeyRef" in item["valueFrom"] for item in env.values())

    service = manifests["Service"]
    assert service["spec"]["type"] == "ClusterIP"
    ingress = manifests["Ingress"]
    assert ingress["spec"]["tls"]
    assert (
        ingress["metadata"]["annotations"]["nginx.ingress.kubernetes.io/proxy-buffering"] == "off"
    )

    policy = manifests["NetworkPolicy"]["spec"]
    assert policy["policyTypes"] == ["Ingress", "Egress"]
    assert len(policy["ingress"]) == 1
    assert len(policy["egress"]) == 3
    assert {port["port"] for rule in policy["egress"] for port in rule["ports"]} == {
        53,
        443,
    }


def test_kubernetes_reference_documents_cluster_specific_limits() -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "DEPLOYMENT_KUBERNETES.md"
    documentation = path.read_text(encoding="utf-8")
    for requirement in (
        "NetworkPolicy-enforcing CNI",
        "NetworkPolicy cannot restrict",
        "immutable digest",
        "secret manager",
        "not been run by this repository",
        "process-local",
        "affinity",
    ):
        assert requirement in documentation

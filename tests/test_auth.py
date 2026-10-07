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
"""Issuer-bound JWT authenticator contract coverage."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request

from gabby import (
    Agent,
    AgentDefinition,
    AuthenticationUnavailable,
    JWTBearerAuthenticator,
    ModelResponse,
    ModelStreamDelta,
    Principal,
    SQLiteTokenRevocationStore,
    TokenRevocationChecker,
)
from gabby.server import create_app

ISSUER = "https://identity.example.test/"
AUDIENCE = "gabby-api"


def _base64url(value: int) -> str:
    encoded = base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big"))
    return encoded.rstrip(b"=").decode("ascii")


@pytest.fixture
def signing_material() -> Iterator[tuple[Any, bytes]]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    document = json.dumps(
        {
            "keys": [
                {
                    "kty": "RSA",
                    "kid": "test-key",
                    "use": "sig",
                    "alg": "RS256",
                    "n": _base64url(public_numbers.n),
                    "e": _base64url(public_numbers.e),
                }
            ]
        }
    ).encode()
    yield private_key, document


class StaticJWKSAuthenticator(JWTBearerAuthenticator):
    def __init__(
        self,
        document: bytes,
        *,
        scope_mapping: dict[str, str | tuple[str, ...]] | None = None,
        revocation_checker: TokenRevocationChecker | None = None,
        revocation_timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_url="https://identity.example.test/keys",
            scope_mapping=scope_mapping,
            revocation_checker=revocation_checker,
            revocation_timeout_seconds=revocation_timeout_seconds,
        )
        self.document = document
        self.fetch_count = 0

    async def _fetch_jwks_document(self) -> bytes:
        self.fetch_count += 1
        return self.document


class RecordingRevocationChecker:
    def __init__(self, revoked_ids: frozenset[str] = frozenset()) -> None:
        self.revoked_ids = revoked_ids
        self.lookups: list[tuple[str, str, float]] = []

    async def is_revoked(self, *, issuer: str, token_id: str, expires_at: float) -> bool:
        self.lookups.append((issuer, token_id, expires_at))
        return token_id in self.revoked_ids


def _request(token: str | None) -> Request:
    headers = [] if token is None else [(b"authorization", f"Bearer {token}".encode())]
    return Request({"type": "http", "headers": headers})


def _token(private_key: Any, *, key_id: str = "test-key", **overrides: Any) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-123",
        "iat": now,
        "exp": now + 300,
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": key_id})


@pytest.mark.parametrize(
    "overrides",
    [
        {"audience": " "},
        {"algorithms": []},
        {"algorithms": ("HS256",)},
        {"algorithms": ("RS256", "RS256")},
        {"algorithms": (1,)},
        {"jwks_cache_seconds": True},
        {"jwks_cache_seconds": float("nan")},
        {"unknown_key_refresh_seconds": 0},
        {"request_timeout_seconds": "fast"},
        {"revocation_timeout_seconds": 0},
        {"revocation_timeout_seconds": float("nan")},
        {"revocation_checker": object()},
        {"scope_mapping": {"bad scope": "agent:run"}},
        {"scope_mapping": {"tasks:run": "bad scope"}},
        {"scope_mapping": {"tasks:run": ()}},
        {"scope_mapping": {"tasks:run": ("agent:run", "agent:run")}},
        {"scope_mapping": []},
    ],
)
def test_jwt_authenticator_rejects_invalid_configuration(overrides: dict[str, Any]) -> None:
    configuration: dict[str, Any] = {
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "jwks_url": "https://identity.example.test/keys",
    }
    configuration.update(overrides)

    with pytest.raises(ValueError):
        JWTBearerAuthenticator(**configuration)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document", "error"),
    [
        (b"not-json", "document is invalid"),
        (b"{}", "no key list"),
        (json.dumps({"keys": [None]}).encode(), "no supported signing keys"),
        (
            json.dumps({"keys": [{"kid": "signing", "alg": "RS256", "use": "sig"}]}).encode(),
            "invalid signing key",
        ),
        (
            json.dumps(
                {"keys": [{"kid": "signing", "alg": "RS256", "use": "sig", "key_ops": ["sign"]}]}
            ).encode(),
            "no supported signing keys",
        ),
        (json.dumps({"keys": [{}] * 101}).encode(), "too many keys"),
    ],
)
async def test_jwt_authenticator_rejects_invalid_jwks_documents(
    document: bytes, error: str
) -> None:
    authenticator = StaticJWKSAuthenticator(document)

    with pytest.raises(AuthenticationUnavailable, match=error):
        await authenticator._refresh_jwks()


@pytest.mark.asyncio
async def test_jwt_authenticator_verifies_identity_and_caches_signing_key(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    authenticator = StaticJWKSAuthenticator(document)
    token = _token(private_key)

    assert await authenticator.authenticate(_request(token)) == Principal("user-123")
    assert await authenticator.authenticate(_request(token)) == Principal("user-123")
    assert authenticator.fetch_count == 1


@pytest.mark.asyncio
async def test_jwt_revocation_checker_requires_jti_and_checks_each_request(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    checker = RecordingRevocationChecker(frozenset({"revoked-token"}))
    authenticator = StaticJWKSAuthenticator(document, revocation_checker=checker)
    live_token = _token(private_key, jti="live-token")
    revoked_token = _token(private_key, jti="revoked-token")
    missing_id_token = _token(private_key)

    assert await authenticator.authenticate(_request(live_token)) == Principal("user-123")
    assert await authenticator.authenticate(_request(revoked_token)) is None
    assert await authenticator.authenticate(_request(live_token)) == Principal("user-123")
    assert await authenticator.authenticate(_request(missing_id_token)) is None
    assert [lookup[1] for lookup in checker.lookups] == [
        "live-token",
        "revoked-token",
        "live-token",
    ]
    assert all(lookup[0] == ISSUER for lookup in checker.lookups)
    assert all(lookup[2] > time.time() for lookup in checker.lookups)


@pytest.mark.asyncio
async def test_sqlite_revocation_store_rejects_token_after_authenticator_restart(
    signing_material: tuple[Any, bytes], tmp_path: Path
) -> None:
    private_key, document = signing_material
    token = _token(private_key, jti="persisted-revocation")
    claims = jwt.decode(token, options={"verify_signature": False})
    database = tmp_path / "auth-revocations.sqlite3"
    writer = SQLiteTokenRevocationStore(database)
    await writer.revoke(
        issuer=ISSUER,
        token_id=claims["jti"],
        expires_at=claims["exp"],
    )

    restarted_store = SQLiteTokenRevocationStore(database)
    authenticator = StaticJWKSAuthenticator(document, revocation_checker=restarted_store)
    assert await authenticator.authenticate(_request(token)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["raise", "timeout", "invalid-result"])
async def test_jwt_revocation_checker_failures_are_unavailable(
    signing_material: tuple[Any, bytes], failure: str
) -> None:
    private_key, document = signing_material

    class BrokenChecker:
        async def is_revoked(self, **_: Any) -> Any:
            if failure == "raise":
                raise RuntimeError("private store connection detail")
            if failure == "timeout":
                await asyncio.sleep(1)
            return "unknown"

    authenticator = StaticJWKSAuthenticator(
        document,
        revocation_checker=BrokenChecker(),
        revocation_timeout_seconds=0.01,
    )

    with pytest.raises(
        AuthenticationUnavailable, match="configured token revocation service"
    ) as exc:
        await authenticator.authenticate(_request(_token(private_key, jti="any-token")))
    assert "private store connection detail" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claim", "value", "expected"),
    [
        ("scope", "agent:run knowledge:read", frozenset({"agent:run", "knowledge:read"})),
        ("scp", ["agent:run", "knowledge:read"], frozenset({"agent:run", "knowledge:read"})),
    ],
)
async def test_jwt_authenticator_returns_supported_scope_claims(
    signing_material: tuple[Any, bytes], claim: str, value: Any, expected: frozenset[str]
) -> None:
    private_key, document = signing_material
    authenticator = StaticJWKSAuthenticator(document)

    principal = await authenticator.authenticate(_request(_token(private_key, **{claim: value})))

    assert principal == Principal("user-123", expected)


@pytest.mark.asyncio
async def test_jwt_authenticator_maps_issuer_scopes_and_drops_unmapped_scopes(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    authenticator = StaticJWKSAuthenticator(
        document,
        scope_mapping={
            "tasks:execute": "agent:run",
            "tasks:observe": ("agent:stream", "trace:read"),
        },
    )
    token = _token(
        private_key,
        scope="tasks:execute tasks:observe identity:profile",
    )

    principal = await authenticator.authenticate(_request(token))

    assert principal == Principal(
        "user-123", frozenset({"agent:run", "agent:stream", "trace:read"})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        {"scope": 42},
        {"scope": "read café"},
        {"scope": "read\twrite"},
        {"scope": "read  write"},
        {"scp": "read write"},
        {"scp": ["read", 42]},
        {"scope": "read", "scp": ["write"]},
    ],
)
async def test_jwt_authenticator_rejects_malformed_scope_claims(
    signing_material: tuple[Any, bytes], claims: dict[str, Any]
) -> None:
    private_key, document = signing_material

    principal = await StaticJWKSAuthenticator(document).authenticate(
        _request(_token(private_key, **claims))
    )

    assert principal is None


@pytest.mark.asyncio
async def test_jwt_authenticator_ignores_encryption_keys_from_jwks(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    jwks = json.loads(document)
    jwks["keys"].append(
        {
            **jwks["keys"][0],
            "kid": "encryption-key",
            "use": "enc",
            "alg": "RSA-OAEP-256",
        }
    )
    authenticator = StaticJWKSAuthenticator(json.dumps(jwks).encode())

    assert await authenticator.authenticate(_request(_token(private_key))) == Principal("user-123")


@pytest.mark.asyncio
async def test_jwt_authenticator_fails_closed_on_duplicate_signing_key_ids(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    jwks = json.loads(document)
    jwks["keys"].append(jwks["keys"][0])
    authenticator = StaticJWKSAuthenticator(json.dumps(jwks).encode())

    with pytest.raises(AuthenticationUnavailable, match="duplicate signing keys"):
        await authenticator.authenticate(_request(_token(private_key)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://attacker.example.test/"},
        {"aud": "another-api"},
        {"aud": None},
        {"exp": 1_700_000_000},
        {"sub": ""},
        {"exp": None},
    ],
)
async def test_jwt_authenticator_rejects_invalid_required_claims(
    signing_material: tuple[Any, bytes], claims: dict[str, Any]
) -> None:
    private_key, document = signing_material
    authenticator = StaticJWKSAuthenticator(document)

    assert await authenticator.authenticate(_request(_token(private_key, **claims))) is None


@pytest.mark.asyncio
async def test_jwt_authenticator_rejects_a_valid_token_signed_by_another_key(
    signing_material: tuple[Any, bytes],
) -> None:
    _, document = signing_material
    unrelated_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    authenticator = StaticJWKSAuthenticator(document)

    assert await authenticator.authenticate(_request(_token(unrelated_key))) is None


@pytest.mark.asyncio
async def test_jwt_authenticator_rejects_missing_and_unsupported_bearer_tokens(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material
    authenticator = StaticJWKSAuthenticator(document)
    unsupported = jwt.encode(
        {"sub": "user-123"},
        "caller-controlled-secret-that-is-long-enough",
        algorithm="HS256",
        headers={"kid": "test-key"},
    )

    assert await authenticator.authenticate(_request(None)) is None
    oversized = "x" * (authenticator._MAX_TOKEN_BYTES + 1)
    assert await authenticator.authenticate(_request(oversized)) is None
    assert await authenticator.authenticate(_request(unsupported)) is None
    assert authenticator.fetch_count == 0
    assert await authenticator.authenticate(_request(_token(private_key, key_id="unknown"))) is None
    assert authenticator.fetch_count == 1


@pytest.mark.asyncio
async def test_jwks_fetch_rejects_redirects_and_oversized_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    response_content = b"x" * (JWTBearerAuthenticator._MAX_JWKS_BYTES + 1)
    authenticator = JWTBearerAuthenticator(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url="https://identity.example.test/keys",
    )

    def redirect_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(302, headers={"Location": "https://other.test/keys"})
            ),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", redirect_client)
    with pytest.raises(AuthenticationUnavailable, match="JWKS endpoint is unavailable"):
        await authenticator._fetch_jwks_document()

    def oversized_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=response_content)),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", oversized_client)
    with pytest.raises(AuthenticationUnavailable, match="JWKS response is too large"):
        await authenticator._fetch_jwks_document()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issuer,metadata_url",
    [
        (
            "https://identity.example.test/tenant/",
            "https://identity.example.test/tenant/.well-known/openid-configuration",
        ),
        (
            "https://identity.example.test/tenant//",
            "https://identity.example.test/tenant//.well-known/openid-configuration",
        ),
        (
            "https://identity.example.test/",
            "https://identity.example.test/.well-known/openid-configuration",
        ),
    ],
)
async def test_oidc_discovery_uses_issuer_path_and_validates_metadata(
    monkeypatch: pytest.MonkeyPatch,
    issuer: str,
    metadata_url: str,
) -> None:
    real_async_client = httpx.AsyncClient
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "application/json; charset=utf-8"},
            json={"issuer": issuer, "jwks_uri": "https://keys.example.test/jwks"},
        )

    def mock_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    authenticator = await JWTBearerAuthenticator.from_oidc_issuer(
        issuer=issuer,
        audience=AUDIENCE,
    )

    assert isinstance(authenticator, JWTBearerAuthenticator)
    assert authenticator._jwks_url == "https://keys.example.test/jwks"
    assert seen == [metadata_url]


@pytest.mark.asyncio
async def test_oidc_discovery_validates_verifier_options_before_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    requests: list[httpx.Request] = []

    def mock_client(**kwargs: Any) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            raise AssertionError("discovery must not run for invalid verifier configuration")

        return real_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    with pytest.raises(ValueError, match="supported asymmetric algorithms"):
        await JWTBearerAuthenticator.from_oidc_issuer(
            issuer=ISSUER,
            audience=AUDIENCE,
            algorithms=("HS256",),
        )
    with pytest.raises(ValueError, match="discovers jwks_url"):
        await JWTBearerAuthenticator.from_oidc_issuer(
            issuer=ISSUER,
            audience=AUDIENCE,
            jwks_url="https://identity.example.test/keys",
        )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,headers,document,error",
    [
        (302, {"location": "https://attacker.example/metadata"}, b"{}", "metadata is unavailable"),
        (200, {"content-type": "text/plain"}, b"{}", "invalid content type"),
        (
            200,
            {"content-type": "application/json"},
            b'{"issuer":"https://attacker.example/","jwks_uri":"https://keys.example.test/jwks"}',
            "issuer does not match",
        ),
        (
            200,
            {"content-type": "application/json"},
            b'{"issuer":"https://identity.example.test/","jwks_uri":"http://keys.example.test/jwks"}',
            "unsafe JWKS URL",
        ),
        (
            200,
            {"content-type": "application/json"},
            b'{"issuer":"https://attacker.example/","issuer":"https://identity.example.test/",'
            b'"jwks_uri":"https://keys.example.test/jwks"}',
            "metadata is unavailable",
        ),
    ],
)
async def test_oidc_discovery_rejects_untrusted_metadata(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    headers: dict[str, str],
    document: bytes,
    error: str,
) -> None:
    real_async_client = httpx.AsyncClient

    def mock_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(status, headers=headers, content=document)
            ),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    with pytest.raises(AuthenticationUnavailable, match=error):
        await JWTBearerAuthenticator.from_oidc_issuer(issuer=ISSUER, audience=AUDIENCE)


@pytest.mark.asyncio
async def test_oidc_discovery_rejects_oversized_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_async_client = httpx.AsyncClient
    content = b" " * (JWTBearerAuthenticator._MAX_OIDC_METADATA_BYTES + 1)

    def mock_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, headers={"content-type": "application/json"}, content=content
                )
            ),
            **kwargs,
        )

    monkeypatch.setattr(httpx, "AsyncClient", mock_client)
    with pytest.raises(AuthenticationUnavailable, match="metadata response is too large"):
        await JWTBearerAuthenticator.from_oidc_issuer(issuer=ISSUER, audience=AUDIENCE)


@pytest.mark.asyncio
async def test_oidc_discovery_has_an_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    real_async_client = httpx.AsyncClient

    async def slow_response(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"issuer": ISSUER, "jwks_uri": "https://identity.example.test/keys"},
        )

    def slow_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(slow_response), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", slow_client)
    with pytest.raises(AuthenticationUnavailable, match="metadata is unavailable"):
        await JWTBearerAuthenticator.from_oidc_issuer(
            issuer=ISSUER,
            audience=AUDIENCE,
            request_timeout_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_jwks_fetch_has_an_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    real_async_client = httpx.AsyncClient
    authenticator = JWTBearerAuthenticator(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url="https://identity.example.test/keys",
        request_timeout_seconds=0.01,
    )

    async def slow_response(_: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"keys": []})

    def slow_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(transport=httpx.MockTransport(slow_response), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", slow_client)
    with pytest.raises(AuthenticationUnavailable, match="JWKS endpoint is unavailable"):
        await authenticator._fetch_jwks_document()


@pytest.mark.asyncio
async def test_jwt_authenticator_protects_fastapi_execution_routes(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material

    class FakeModel:
        name = "jwt-auth-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="accepted")

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="accepted")

    agent = Agent(
        AgentDefinition(
            name="jwt-agent",
            model={"provider": "test", "model": "jwt"},
            policies={"max_steps": 1, "timeout_seconds": 2},
        ),
        model=FakeModel(),
    )
    app = create_app(
        agent,
        authenticator=StaticJWKSAuthenticator(document),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        accepted = await client.post(
            "/v1/agents/jwt-agent/run",
            json={"input": "hello"},
            headers={"Authorization": f"Bearer {_token(private_key)}"},
        )
        rejected = await client.post(
            "/v1/agents/jwt-agent/run",
            json={"input": "hello"},
            headers={"Authorization": "Bearer malformed"},
        )

    assert accepted.status_code == 200
    assert accepted.json()["output"] == "accepted"
    assert rejected.status_code == 401


@pytest.mark.asyncio
async def test_jwt_capability_mapping_controls_route_authorization(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material

    class FakeModel:
        name = "jwt-scope-map-test"

        async def complete(self, **_: Any) -> ModelResponse:
            return ModelResponse(content="accepted")

        async def stream(self, **_: Any) -> Any:
            yield ModelStreamDelta(content_delta="accepted")

    app = create_app(
        Agent(
            AgentDefinition(
                name="jwt-agent",
                model={"provider": "test", "model": "jwt"},
                policies={"max_steps": 1, "timeout_seconds": 2},
            ),
            model=FakeModel(),
        ),
        authenticator=StaticJWKSAuthenticator(
            document, scope_mapping={"tasks:execute": "agent:run"}
        ),
        run_scopes=("agent:run",),
    )
    allowed_token = _token(private_key, scope="tasks:execute profile:read")
    denied_token = _token(private_key, scope="profile:read")

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        allowed = await client.post(
            "/v1/agents/jwt-agent/run",
            json={"input": "hello"},
            headers={"Authorization": f"Bearer {allowed_token}"},
        )
        denied = await client.post(
            "/v1/agents/jwt-agent/run",
            json={"input": "hello"},
            headers={"Authorization": f"Bearer {denied_token}"},
        )

    assert allowed.status_code == 200
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_jwt_revocation_backend_failure_fails_closed_over_http(
    signing_material: tuple[Any, bytes],
) -> None:
    private_key, document = signing_material

    class BrokenChecker:
        async def is_revoked(self, **_: Any) -> bool:
            raise RuntimeError("private revocation storage details")

    agent = Agent(
        AgentDefinition(
            name="jwt-agent",
            model={"provider": "test", "model": "jwt"},
            policies={"max_steps": 1, "timeout_seconds": 2},
        ),
        model=object(),  # type: ignore[arg-type]
    )
    app = create_app(
        agent,
        authenticator=StaticJWKSAuthenticator(
            document,
            revocation_checker=BrokenChecker(),
        ),
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/v1/agents/jwt-agent/run",
            json={"input": "hello"},
            headers={"Authorization": f"Bearer {_token(private_key, jti='revocation-test')}"},
        )

    assert response.status_code == 503
    assert "private revocation storage details" not in response.text


@pytest.mark.parametrize(
    "options",
    [
        {"issuer": "http://identity.example.test/"},
        {"jwks_url": "http://identity.example.test/keys"},
        {"jwks_url": "https://user:password@identity.example.test/keys"},
        {"jwks_url": "https://identity.example.test/keys?token=secret"},
        {"algorithms": ("HS256",)},
    ],
)
def test_jwt_authenticator_rejects_unsafe_configuration(options: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "issuer": ISSUER,
        "audience": AUDIENCE,
        "jwks_url": "https://identity.example.test/keys",
    }
    values.update(options)

    with pytest.raises(ValueError):
        JWTBearerAuthenticator(**values)

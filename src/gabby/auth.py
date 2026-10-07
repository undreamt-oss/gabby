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
"""Authentication contracts for Gabby's HTTP serving boundary."""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from starlette.requests import Request


@dataclass(frozen=True)
class Principal:
    """Authenticated identity and optional OAuth-style scopes."""

    subject: str
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.subject, str) or not self.subject.strip():
            raise ValueError("principal subject must be a non-empty string")
        if not isinstance(self.scopes, frozenset) or any(
            not _is_scope_token(scope) for scope in self.scopes
        ):
            raise ValueError("principal scopes must be a frozenset of valid scope tokens")


def _is_scope_token(value: object) -> bool:
    """Return whether a value is one printable ASCII OAuth scope token."""
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and all(0x21 <= ord(char) <= 0x7E and char not in {'"', "\\"} for char in value)
    )


def _json_object_without_duplicate_members(
    members: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build a JSON object while rejecting ambiguous duplicate member names."""
    result: dict[str, Any] = {}
    for name, value in members:
        if name in result:
            raise ValueError("JSON object contains a duplicate member")
        result[name] = value
    return result


def _jwt_scopes(claims: dict[str, Any]) -> frozenset[str] | None:
    """Read standard ``scope`` or list-valued ``scp`` claims; reject malformed claims."""
    if "scope" in claims and "scp" in claims:
        return None
    if "scope" in claims:
        raw_scope = claims["scope"]
        if not isinstance(raw_scope, str) or any(
            char != " " and not _is_scope_token(char) for char in raw_scope
        ):
            return None
        scopes = [] if raw_scope == "" else raw_scope.split(" ")
        if any(not _is_scope_token(scope) for scope in scopes):
            return None
        return frozenset(scopes)
    if "scp" in claims:
        raw_scopes = claims["scp"]
        if not isinstance(raw_scopes, list) or any(
            not _is_scope_token(scope) for scope in raw_scopes
        ):
            return None
        return frozenset(raw_scopes)
    return frozenset()


def _validate_scope_mapping(
    value: Mapping[str, str | tuple[str, ...]] | None,
) -> dict[str, frozenset[str]] | None:
    """Copy and validate external identity-provider scopes to Gabby capabilities."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("scope_mapping must be a mapping of scope tokens")
    mapped: dict[str, frozenset[str]] = {}
    for issuer_scope, capabilities in value.items():
        if not _is_scope_token(issuer_scope):
            raise ValueError("scope_mapping keys must be valid scope tokens")
        if isinstance(capabilities, str):
            targets: tuple[str, ...] = (capabilities,)
        elif isinstance(capabilities, tuple) and capabilities:
            targets = capabilities
        else:
            raise ValueError("scope_mapping values must be a scope token or non-empty tuple")
        if any(not _is_scope_token(target) for target in targets):
            raise ValueError("scope_mapping values must contain valid capability tokens")
        if len(set(targets)) != len(targets):
            raise ValueError("scope_mapping values cannot contain duplicate capabilities")
        mapped[issuer_scope] = frozenset(targets)
    return mapped


class Authenticator(Protocol):
    """Pluggable asynchronous request authenticator."""

    async def authenticate(self, request: Request) -> Principal | None:
        """Return a principal for valid credentials, or ``None`` to reject them."""
        ...


class AuthenticationUnavailable(RuntimeError):
    """The configured identity provider could not authenticate requests."""


class TokenRevocationChecker(Protocol):
    """Host-owned asynchronous lookup for JWT revocation by issuer and token ID."""

    async def is_revoked(self, *, issuer: str, token_id: str, expires_at: float) -> bool:
        """Return whether this JWT ID is revoked; raise on backend failure."""
        ...


class BearerTokenAuthenticator:
    """Single-token authenticator for trusted, single-tenant deployments."""

    def __init__(
        self,
        token: str,
        *,
        subject: str = "bearer-token",
        scopes: frozenset[str] = frozenset(),
    ) -> None:
        if not isinstance(token, str) or not token or not token.isascii():
            raise ValueError("bearer token must be a non-empty ASCII string")
        if any(ord(char) < 33 or ord(char) > 126 for char in token):
            raise ValueError("bearer token must contain visible ASCII characters only")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("bearer token subject must be a non-empty string")
        self._token = token
        self._principal = Principal(subject, scopes)

    @classmethod
    def from_env(
        cls,
        variable: str = "GABBY_API_TOKEN",
        *,
        subject: str = "bearer-token",
        scopes: frozenset[str] = frozenset(),
    ) -> BearerTokenAuthenticator:
        """Load the token from a host-managed environment variable."""
        if not variable or not variable.isidentifier():
            raise ValueError("token environment variable name must be a valid identifier")
        token = os.environ.get(variable)
        if token is None:
            raise ValueError(f"required bearer token environment variable is not set: {variable}")
        return cls(token, subject=subject, scopes=scopes)

    async def authenticate(self, request: Request) -> Principal | None:
        """Compare the request's bearer credential and return the configured principal."""
        authorization = request.headers.get("authorization")
        if authorization is None:
            return None
        scheme, separator, credential = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not credential
            or credential != credential.strip()
            or any(char.isspace() for char in credential)
            or not credential.isascii()
        ):
            return None
        if hmac.compare_digest(credential.encode("ascii"), self._token.encode("ascii")):
            return self._principal
        return None


class JWTBearerAuthenticator:
    """Verify issuer-bound JWT bearer tokens from a configured JWKS endpoint.

    This validates identity and standard ``scope`` or list-valued ``scp`` claims. Route
    Authorization is opt-in through ``create_app`` scope requirements. The async
    ``from_oidc_issuer`` factory supports fixed-issuer metadata discovery; token tenant routing is
    not implemented. Optional scope mapping translates issuer scopes into Gabby capabilities and
    drops unmapped scopes. A host-owned revocation checker can reject token IDs on every request.
    The issuer and JWKS endpoint are fixed by the host, never read from a token.
    Install the optional ``auth`` extra for cryptographic checks.
    """

    _MAX_TOKEN_BYTES = 16 * 1024
    _MAX_JWKS_BYTES = 1024 * 1024
    _MAX_OIDC_METADATA_BYTES = 1024 * 1024
    _MAX_JWKS_KEYS = 100
    _ALLOWED_PUBLIC_KEY_ALGORITHMS = frozenset(
        {
            "RS256",
            "RS384",
            "RS512",
            "PS256",
            "PS384",
            "PS512",
            "ES256",
            "ES384",
            "ES512",
            "EdDSA",
        }
    )

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str,
        algorithms: tuple[str, ...] = ("RS256", "ES256"),
        jwks_cache_seconds: float = 300.0,
        unknown_key_refresh_seconds: float = 30.0,
        request_timeout_seconds: float = 5.0,
        scope_mapping: Mapping[str, str | tuple[str, ...]] | None = None,
        revocation_checker: TokenRevocationChecker | None = None,
        revocation_timeout_seconds: float = 1.0,
    ) -> None:
        self._validate_endpoint(issuer, label="issuer")
        self._validate_endpoint(jwks_url, label="jwks_url")
        if not isinstance(audience, str) or not audience.strip():
            raise ValueError("JWT audience must be a non-empty string")
        if (
            not isinstance(algorithms, tuple)
            or not algorithms
            or any(
                not isinstance(algorithm, str)
                or algorithm not in self._ALLOWED_PUBLIC_KEY_ALGORITHMS
                for algorithm in algorithms
            )
            or len(set(algorithms)) != len(algorithms)
        ):
            raise ValueError("JWT algorithms must be unique supported asymmetric algorithms")
        for name, value in (
            ("jwks_cache_seconds", jwks_cache_seconds),
            ("unknown_key_refresh_seconds", unknown_key_refresh_seconds),
            ("request_timeout_seconds", request_timeout_seconds),
            ("revocation_timeout_seconds", revocation_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive number")

        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - exercised without the optional extra
            raise ImportError(
                "JWTBearerAuthenticator requires the optional dependency; "
                "install gabby-agent-runtime[auth]"
            ) from exc

        self._jwt = jwt
        self._issuer = issuer
        self._audience = audience
        self._jwks_url = jwks_url
        self._algorithms = algorithms
        self._jwks_cache_seconds = float(jwks_cache_seconds)
        self._unknown_key_refresh_seconds = float(unknown_key_refresh_seconds)
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._scope_mapping = _validate_scope_mapping(scope_mapping)
        if revocation_checker is not None and not callable(
            getattr(revocation_checker, "is_revoked", None)
        ):
            raise ValueError("revocation_checker must provide async is_revoked()")
        self._revocation_checker = revocation_checker
        self._revocation_timeout_seconds = float(revocation_timeout_seconds)
        self._jwks: dict[tuple[str, str], Any] = {}
        self._jwks_expires_at = 0.0
        self._last_jwks_refresh_at = 0.0
        self._jwks_lock = asyncio.Lock()

    @classmethod
    async def from_oidc_issuer(
        cls,
        *,
        issuer: str,
        audience: str,
        **options: Any,
    ) -> JWTBearerAuthenticator:
        """Discover a fixed issuer's JWKS URL from its bounded OIDC configuration.

        The issuer remains host-configured and is matched exactly against the discovery document.
        Discovery is performed once by this factory; key rotation continues through the normal
        bounded JWKS refresh path. The returned JWKS URL is validated before use.
        """
        cls._validate_endpoint(issuer, label="issuer")
        if "jwks_url" in options:
            raise ValueError("from_oidc_issuer discovers jwks_url; do not pass it directly")
        timeout = options.get("request_timeout_seconds", 5.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("request_timeout_seconds must be a positive number")
        issuer_parts = urlsplit(issuer)
        issuer_path = issuer_parts.path
        if issuer_path.endswith("/"):
            issuer_path = issuer_path[:-1]
        metadata_path = issuer_path + "/.well-known/openid-configuration"
        metadata_url = urlunsplit((issuer_parts.scheme, issuer_parts.netloc, metadata_path, "", ""))
        # Construct first so the normal verifier validates all arguments before any network call.
        authenticator = cls(
            issuer=issuer,
            audience=audience,
            jwks_url=metadata_url,
            **options,
        )
        try:
            async with asyncio.timeout(float(timeout)):
                async with (
                    httpx.AsyncClient(
                        timeout=float(timeout),
                        follow_redirects=False,
                        trust_env=False,
                    ) as client,
                    client.stream(
                        "GET", metadata_url, headers={"Accept": "application/json"}
                    ) as response,
                ):
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if content_type.split(";", 1)[0].strip().casefold() != "application/json":
                        raise AuthenticationUnavailable(
                            "configured OIDC metadata has an invalid content type"
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > cls._MAX_OIDC_METADATA_BYTES:
                            raise AuthenticationUnavailable(
                                "configured OIDC metadata response is too large"
                            )
                        content.extend(chunk)
            document = json.loads(content, object_pairs_hook=_json_object_without_duplicate_members)
        except AuthenticationUnavailable:
            raise
        except Exception as exc:
            raise AuthenticationUnavailable("configured OIDC metadata is unavailable") from exc

        if not isinstance(document, dict) or document.get("issuer") != issuer:
            raise AuthenticationUnavailable("configured OIDC metadata issuer does not match")
        jwks_url = document.get("jwks_uri")
        if not isinstance(jwks_url, str):
            raise AuthenticationUnavailable("configured OIDC metadata has an unsafe JWKS URL")
        try:
            cls._validate_endpoint(jwks_url, label="discovered jwks_uri")
        except ValueError as exc:
            raise AuthenticationUnavailable(
                "configured OIDC metadata has an unsafe JWKS URL"
            ) from exc
        authenticator._jwks_url = jwks_url
        return authenticator

    @staticmethod
    def _validate_endpoint(value: str, *, label: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be an absolute HTTPS URL")
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f"{label} must be an absolute HTTPS URL without user info")
        if parsed.query or parsed.fragment:
            raise ValueError(f"{label} cannot contain a query or fragment")

    @property
    def _jwks_cache_is_fresh(self) -> bool:
        return time.monotonic() < self._jwks_expires_at

    async def _fetch_jwks_document(self) -> bytes:
        """Fetch the configured JWK document within a strict byte and time bound."""
        try:
            async with asyncio.timeout(self._request_timeout_seconds):
                async with (
                    httpx.AsyncClient(
                        timeout=self._request_timeout_seconds,
                        follow_redirects=False,
                        trust_env=False,
                    ) as client,
                    client.stream(
                        "GET", self._jwks_url, headers={"Accept": "application/json"}
                    ) as response,
                ):
                    response.raise_for_status()
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(content) + len(chunk) > self._MAX_JWKS_BYTES:
                            raise AuthenticationUnavailable("configured JWKS response is too large")
                        content.extend(chunk)
            return bytes(content)
        except AuthenticationUnavailable:
            raise
        except Exception as exc:
            raise AuthenticationUnavailable("configured JWKS endpoint is unavailable") from exc

    async def _refresh_jwks(self) -> None:
        """Fetch and parse the configured JWK set within a strict byte and time bound."""
        document = await self._fetch_jwks_document()
        try:
            payload = json.loads(document, object_pairs_hook=_json_object_without_duplicate_members)
        except Exception as exc:
            raise AuthenticationUnavailable("configured JWKS document is invalid") from exc

        raw_keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(raw_keys, list):
            raise AuthenticationUnavailable("configured JWKS document has no key list")
        if len(raw_keys) > self._MAX_JWKS_KEYS:
            raise AuthenticationUnavailable("configured JWKS contains too many keys")
        keys: dict[tuple[str, str], Any] = {}
        for jwk_data in raw_keys:
            if not isinstance(jwk_data, dict):
                continue
            kid = jwk_data.get("kid")
            algorithm = jwk_data.get("alg")
            key_use = jwk_data.get("use")
            key_ops = jwk_data.get("key_ops")
            if (
                isinstance(kid, str)
                and kid
                and isinstance(algorithm, str)
                and algorithm in self._algorithms
                and algorithm in self._ALLOWED_PUBLIC_KEY_ALGORITHMS
                and key_use in (None, "sig")
            ):
                if key_ops is not None and (
                    not isinstance(key_ops, list) or "verify" not in key_ops
                ):
                    continue
                key_reference = (kid, algorithm)
                if key_reference in keys:
                    raise AuthenticationUnavailable(
                        "configured JWKS contains duplicate signing keys"
                    )
                try:
                    keys[key_reference] = self._jwt.PyJWK.from_dict(jwk_data)
                except Exception as exc:
                    raise AuthenticationUnavailable(
                        "configured JWKS contains an invalid signing key"
                    ) from exc
        if not keys:
            raise AuthenticationUnavailable("configured JWKS has no supported signing keys")

        refreshed_at = time.monotonic()
        self._jwks = keys
        self._jwks_expires_at = refreshed_at + self._jwks_cache_seconds
        self._last_jwks_refresh_at = refreshed_at

    async def _key_for(self, kid: str, algorithm: str) -> Any | None:
        async with self._jwks_lock:
            now = time.monotonic()
            refresh_due_to_unknown_key = (
                (kid, algorithm) not in self._jwks
                and now - self._last_jwks_refresh_at >= self._unknown_key_refresh_seconds
            )
            if not self._jwks_cache_is_fresh or refresh_due_to_unknown_key:
                await self._refresh_jwks()
            return self._jwks.get((kid, algorithm))

    async def authenticate(self, request: Request) -> Principal | None:
        """Verify JWT signature and required identity and validity claims."""
        authorization = request.headers.get("authorization")
        if authorization is None:
            return None
        scheme, separator, token = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not token
            or token != token.strip()
            or any(char.isspace() for char in token)
            or not token.isascii()
            or len(token) > self._MAX_TOKEN_BYTES
        ):
            return None

        try:
            header = self._jwt.get_unverified_header(token)
            kid = header.get("kid")
            algorithm = header.get("alg")
            if (
                not isinstance(kid, str)
                or not kid
                or algorithm not in self._algorithms
                or algorithm not in self._ALLOWED_PUBLIC_KEY_ALGORITHMS
            ):
                return None
            key = await self._key_for(kid, algorithm)
            if key is None:
                return None
            required_claims = ["exp", "iss", "aud", "sub"]
            if self._revocation_checker is not None:
                required_claims.append("jti")
            claims = self._jwt.decode(
                token,
                key=key,
                algorithms=list(self._algorithms),
                issuer=self._issuer,
                audience=self._audience,
                options={"require": required_claims},
            )
        except self._jwt.InvalidTokenError:
            return None

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            return None
        scopes = _jwt_scopes(claims)
        if scopes is None:
            return None
        if self._revocation_checker is not None:
            token_id = claims.get("jti")
            expires_at = claims.get("exp")
            if (
                not isinstance(token_id, str)
                or not token_id
                or not token_id.isascii()
                or any(ord(char) < 33 or ord(char) > 126 for char in token_id)
                or len(token_id) > 512
                or isinstance(expires_at, bool)
                or not isinstance(expires_at, (int, float))
                or not math.isfinite(expires_at)
            ):
                return None
            try:
                async with asyncio.timeout(self._revocation_timeout_seconds):
                    revoked = await self._revocation_checker.is_revoked(
                        issuer=self._issuer,
                        token_id=token_id,
                        expires_at=float(expires_at),
                    )
            except TimeoutError as exc:
                raise AuthenticationUnavailable(
                    "configured token revocation service timed out"
                ) from exc
            except Exception as exc:
                raise AuthenticationUnavailable(
                    "configured token revocation service is unavailable"
                ) from exc
            if not isinstance(revoked, bool):
                raise AuthenticationUnavailable(
                    "configured token revocation service returned an invalid result"
                )
            if revoked:
                return None
        if self._scope_mapping is not None:
            scopes = frozenset(
                capability
                for issuer_scope in scopes
                for capability in self._scope_mapping.get(issuer_scope, frozenset())
            )
        return Principal(subject, scopes)

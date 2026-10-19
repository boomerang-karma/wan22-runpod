"""Auth0 login for the portal: OpenID Connect authorization-code flow, server side.

  /login          -> redirect to Auth0 (state + nonce kept in the signed session cookie)
  /auth/callback  -> exchange the code for tokens, verify the ID token (RS256, issuer, audience, nonce, expiry)
  /logout         -> clear the session and end the Auth0 session

Only the verified ID token's claims are kept (sub, email, name, picture) in the session cookie; no access token is
stored. PORTAL_ALLOWED_EMAILS optionally restricts who may use the portal ("a@x.com,@company.com"); listed users
must have a verified email.
"""
from __future__ import annotations

import secrets
import time
import urllib.parse
from typing import Optional

import jwt
import requests

LEEWAY_S = 60


class AuthError(Exception):
    pass


def _get_json(url: str) -> dict:                          # patched in tests
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def _post_form(url: str, data: dict) -> dict:             # patched in tests
    r = requests.post(url, data=data, timeout=10)
    if not r.ok:
        raise AuthError(f"token exchange failed ({r.status_code}): {r.text[:200]}")
    return r.json()


class Auth0:
    def __init__(self, domain: str, client_id: str, client_secret: str, base_url: str, allowed: str = ""):
        self.domain = domain.strip().removeprefix("https://").rstrip("/")
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.base_url = base_url.rstrip("/")
        self.allowed = [a.strip().lower() for a in allowed.split(",") if a.strip()]
        self._jwks: tuple[float, Optional[jwt.PyJWKSet]] = (0.0, None)

    @property
    def issuer(self) -> str:
        return f"https://{self.domain}/"

    @property
    def redirect_uri(self) -> str:
        return f"{self.base_url}/auth/callback"

    def check_config(self) -> None:
        missing = [n for n, v in (("AUTH0_DOMAIN", self.domain), ("AUTH0_CLIENT_ID", self.client_id),
                                  ("AUTH0_CLIENT_SECRET", self.client_secret), ("PORTAL_BASE_URL", self.base_url)) if not v]
        if missing:
            raise RuntimeError("Auth0 login is half configured; missing " + ", ".join(missing))

    # ------------------------------------------------------------------ flow
    def authorize_url(self, session: dict) -> str:
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        session["oidc"] = {"state": state, "nonce": nonce, "t": int(time.time())}
        q = {"response_type": "code", "client_id": self.client_id, "redirect_uri": self.redirect_uri,
             "scope": "openid profile email", "state": state, "nonce": nonce}
        return f"https://{self.domain}/authorize?" + urllib.parse.urlencode(q)

    def callback(self, session: dict, params: dict) -> dict:
        pending = session.pop("oidc", None)
        if params.get("error"):
            raise AuthError(params.get("error_description") or params["error"])
        if not pending or not params.get("state") or not secrets.compare_digest(params["state"], pending["state"]):
            raise AuthError("login expired or was started in another tab; try again")
        if time.time() - pending.get("t", 0) > 900:
            raise AuthError("login took too long; try again")
        if not params.get("code"):
            raise AuthError("Auth0 returned no authorization code")
        tokens = _post_form(f"https://{self.domain}/oauth/token", {
            "grant_type": "authorization_code", "code": params["code"], "redirect_uri": self.redirect_uri,
            "client_id": self.client_id, "client_secret": self.client_secret})
        claims = self.verify_id_token(tokens.get("id_token", ""), pending["nonce"])
        user = {"sub": claims["sub"], "email": claims.get("email", ""), "name": claims.get("name") or
                claims.get("nickname") or claims.get("email", ""), "picture": claims.get("picture", ""),
                "email_verified": bool(claims.get("email_verified"))}
        if not self.is_allowed(user):
            raise AuthError(f"{user['email'] or user['sub']} isn't on this portal's guest list")
        return user

    def verify_id_token(self, token: str, nonce: str) -> dict:
        if not token:
            raise AuthError("Auth0 returned no ID token")
        try:
            kid = jwt.get_unverified_header(token).get("kid")
            key = self._signing_key(kid)
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self.client_id, issuer=self.issuer,
                                leeway=LEEWAY_S, options={"require": ["exp", "iat", "sub"]})
        except jwt.PyJWTError as e:
            raise AuthError(f"invalid ID token: {e}")
        if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise AuthError("invalid ID token: nonce mismatch")
        return claims

    def _signing_key(self, kid: Optional[str]):
        for refresh in (False, True):                       # refetch once when Auth0 rotated its keys
            t, keyset = self._jwks
            if refresh or keyset is None or time.time() - t > 3600:
                keyset = jwt.PyJWKSet.from_dict(_get_json(f"https://{self.domain}/.well-known/jwks.json"))
                self._jwks = (time.time(), keyset)
            for k in keyset.keys:
                if k.key_id == kid:
                    return k.key
        raise AuthError("ID token signed with an unknown key")

    def is_allowed(self, user: dict) -> bool:
        if not self.allowed:
            return True
        email = (user.get("email") or "").lower()
        if not email or not user.get("email_verified"):
            return False
        return any(email == a or (a.startswith("@") and email.endswith(a)) for a in self.allowed)

    def logout_url(self) -> str:
        q = {"client_id": self.client_id, "returnTo": self.base_url + "/"}
        return f"https://{self.domain}/v2/logout?" + urllib.parse.urlencode(q)

from datetime import datetime, timedelta, timezone
import time
from collections.abc import Mapping
from typing import Any, Optional, cast
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, ValidationError
from app.config import settings
import app.database as database

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


class TokenData(BaseModel):
    user_id: str
    tenant_id: str
    role: str = Field(pattern=r"^(admin|analyst|viewer)$")


_oidc_jwks: dict[str, object] = {}
_oidc_jwks_expires_at = 0.0


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def register_tenant_user(
    username: str,
    password: str,
    tenant_id: str,
    role: str,
) -> bool:
    """Hash and persist a new user within the supplied tenant."""
    return await database.create_tenant_user(
        tenant_id=tenant_id,
        username=username,
        password_hash=get_password_hash(password),
        role=role,
    )


async def authenticate_user(username: str, password: str, tenant_id: str) -> Optional[TokenData]:
    """Verify a tenant-scoped user against the PostgreSQL credential store."""
    if database.pool is None:
        raise RuntimeError("Database pool is not initialized.")

    async with database.pool.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT username, tenant_id, role, password_hash
                FROM tenant_users
                WHERE username = %s AND tenant_id = %s
                """,
                (username, tenant_id),
            )
            user = cast(Mapping[str, object] | None, await cursor.fetchone())

            if not user:
                return None
            password_hash = user.get("password_hash")
            username_value = user.get("username")
            tenant_value = user.get("tenant_id")
            role_value = user.get("role")
            if not isinstance(password_hash, str):
                return None
            if not isinstance(username_value, str):
                return None
            if not isinstance(tenant_value, str):
                return None
            if not isinstance(role_value, str):
                return None
            if not verify_password(password, password_hash):
                return None
            return TokenData(user_id=username_value, tenant_id=tenant_value, role=role_value)


async def get_current_tenant_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate multi-tenant authorization credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if settings.AUTH_MODE == "oidc":
        try:
            user = await verify_oidc_token(token)
            return user
        except (JWTError, KeyError, ValueError, ValidationError, httpx.HTTPError):
            raise credentials_exception
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub", "")
        tenant_id: str = payload.get("tenant_id", "")
        role: str = payload.get("role", "")
        if not user_id or not tenant_id or not role:
            raise credentials_exception
        user = TokenData(user_id=user_id, tenant_id=tenant_id, role=role)
        if not await database.is_active_tenant_user(user.tenant_id, user.user_id, user.role):
            raise credentials_exception
        return user
    except (JWTError, ValidationError, ValueError):
        raise credentials_exception


async def verify_oidc_token(token: str) -> TokenData:
    """Validate an OIDC JWT using the configured issuer, audience, and JWKS."""
    global _oidc_jwks, _oidc_jwks_expires_at
    if time.monotonic() >= _oidc_jwks_expires_at:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(settings.OIDC_JWKS_URL)
            response.raise_for_status()
            _oidc_jwks = response.json()
        _oidc_jwks_expires_at = time.monotonic() + 900

    header = jwt.get_unverified_header(token)
    key_id = header.get("kid")
    keys = _oidc_jwks.get("keys")
    if not isinstance(keys, list):
        raise ValueError("OIDC signing keys were not found.")
    key = next(
        (
            candidate
            for candidate in keys
            if isinstance(candidate, dict) and candidate.get("kid") == key_id
        ),
        None,
    )
    if key is None:
        raise ValueError("OIDC signing key was not found.")

    payload = jwt.decode(
        token,
        key,
        algorithms=settings.oidc_algorithms,
        audience=settings.OIDC_AUDIENCE,
        issuer=settings.OIDC_ISSUER,
    )
    role_claim: Any = payload.get(settings.OIDC_ROLE_CLAIM)
    role = role_claim[0] if isinstance(role_claim, list) and role_claim else role_claim
    tenant_id = payload.get(settings.OIDC_TENANT_CLAIM)
    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("OIDC token is missing a valid subject claim.")
    if not isinstance(tenant_id, str) or not tenant_id:
        raise ValueError("OIDC token is missing a valid tenant claim.")
    if not isinstance(role, str) or not role:
        raise ValueError("OIDC token is missing required subject, tenant, or role claims.")
    return TokenData(user_id=user_id, tenant_id=tenant_id, role=role)

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.refresh_token import RefreshToken
from app.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

# --- Password Hashing ---

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plain-text password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


# --- JWT Token ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decode and validate a JWT access token.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str | None = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# --- Refresh Token (opaque, DB-tracked) ---


def _refresh_token_hash(raw: str) -> str:
    """One-way hash of the refresh token, stored in the DB row.

    The raw value is never persisted: only this hash is used for lookup,
    revocation and reuse detection.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_refresh_token(user_id: int) -> tuple[str, RefreshToken]:
    """Create a new refresh token for ``user_id``.

    Returns the raw token value (to be set in an HttpOnly cookie) and the
    corresponding ORM row.  The raw token must be treated as secret.
    """
    raw = secrets.token_urlsafe(48)
    token_hash = _refresh_token_hash(raw)

    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    row = RefreshToken(
        token_hash=token_hash,
        user_id=user_id,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    return raw, row


def lookup_refresh_token(raw: str, db: Session) -> RefreshToken | None:
    """Return the refresh-token row for ``raw`` if it exists and is valid.

    A token is valid when it exists, has not expired and has not been revoked.
    """
    token_hash = _refresh_token_hash(raw)
    stmt = (
        select(RefreshToken)
        .where(RefreshToken.token_hash == token_hash)
        .where(RefreshToken.user_id.isnot(None))
    )
    token = db.execute(stmt).scalars().first()
    if token is None:
        return None
    return token if token.is_valid() else None


def rotate_refresh_token(token: RefreshToken, db: Session) -> tuple[str, RefreshToken, RefreshToken]:
    """Rotate ``token``: revoke it, issue a replacement, and return both.

    This is the core rotation/revocation mechanism.

    Returns:
        (new_raw, new_row, old_revoked_row)
    """
    now = datetime.now(timezone.utc)

    # Revoke the current (about-to-be-used) token so it cannot be reused.
    token.revoked = True
    token.issued_at = now
    old = token

    new_raw, new_row = create_refresh_token(user_id=token.user_id)
    new_row.parent_id = old.id

    db.add(new_row)
    db.flush()
    return new_raw, new_row, old


def revoke_refresh_token(raw: str, db: Session) -> bool:
    """Revoke a refresh token (logout). Returns True if a token was revoked."""
    token = lookup_refresh_token(raw, db)
    if token is None:
        return False
    token.revoked = True
    db.flush()
    return True


def revoke_all_user_refresh_tokens(user_id: int, db: Session) -> int:
    """Revoke every active refresh token for ``user_id"."""
    count = (
        db.query(RefreshToken)
        .filter(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked == False,
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
        .update({"revoked": True})
    )
    db.flush()
    return count


# --- Authentication Dependency ---


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency that extracts and validates the current user from a JWT token.

    Usage:
        @router.get("/protected")
        def protected_route(current_user: User = Depends(get_current_user)):
            ...
    """
    payload = decode_access_token(token)
    username: str = payload.get("sub")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return user

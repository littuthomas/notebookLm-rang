import logging

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import (
    RefreshTokenResponse,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)
from app.services.auth import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    hash_password,
    lookup_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
    verify_password,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["auth"])

# OpenAPI request-body declaration for /login so the docs advertise both the
# OAuth2 form body (used by Swagger UI's Authorize dialog) and the JSON body
# (used by programmatic clients).
_login_request_body = {
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "Username or email"},
                    "password": {"type": "string", "description": "Plain-text password"},
                },
                "required": ["username", "password"],
            }
        },
        "application/x-www-form-urlencoded": {
            "schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "Username or email"},
                    "password": {"type": "string", "description": "Plain-text password"},
                },
                "required": ["username", "password"],
            }
        },
    },
    "required": True,
}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: UserRegister, db: Session = Depends(get_db)) -> User:
    """Register a new user account."""
    # Check for duplicate username
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    # Check for duplicate email
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Create user
    user = User(
        username=request.username,
        email=request.email,
        hashed_password=hash_password(request.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"User registered: {user.username}")
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    openapi_extra={"requestBody": _login_request_body},
)
async def login(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    username: str | None = Form(default=None),
    password: str | None = Form(default=None),
) -> TokenResponse:
    """Authenticate a user and return a JWT access token with a secure refresh cookie.

    Accepts credentials either as a JSON body (``{"username", "password"}``)
    or as ``application/x-www-form-urlencoded`` fields, so Swagger UI's
    OAuth2 (password flow) Authorize dialog works out of the box.

    The access token is returned in the JSON body.  The refresh token is issued
    as an HttpOnly cookie so JavaScript can never read it.
    """
    # Optional JSON body (application/json)
    json_username: str | None = None
    json_password: str | None = None
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            json_username = body.get("username")
            json_password = body.get("password")

    # Form fields take precedence, JSON body as fallback
    final_username = username if username is not None else json_username
    final_password = password if password is not None else json_password

    if not final_username or not final_password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="username and password are required (JSON body or form fields)",
        )

    login_data = UserLogin(username=final_username, password=final_password)

    # Find user by username or email
    user = db.query(User).filter(
        (User.username == login_data.username) | (User.email == login_data.username)
    ).first()

    if user is None or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    # Create tokens
    access_token = create_access_token(data={"sub": user.username})
    raw_refresh, _refresh_row = create_refresh_token(user_id=user.id)
    set_refresh_cookie(response, raw_refresh)

    logger.info(f"User logged in: {user.username}")

    return TokenResponse(access_token=access_token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    """Return the currently authenticated user's profile."""
    return current_user


@router.post("/logout", response_model=None)
async def logout(response: Response, db: Session = Depends(get_db)):
    """Revoke the current refresh token and clear the refresh cookie.

    The access token (in localStorage) is cleared on the frontend; this
    endpoint ensures the server-side refresh token is also revoked so a
    stolen/expired access token cannot be refreshed again.
    """
    raw_refresh = request_refresh_token_from_cookie(request)
    if raw_refresh:
        revoke_refresh_token(raw_refresh, db)

    clear_refresh_cookie(response)
    return None


@router.post("/refresh", response_model=RefreshTokenResponse)
async def refresh(response: Response, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a new access token.

    Implements refresh-token rotation: the presented refresh token is revoked
    immediately and a fresh one is issued in its place so a stolen token cannot
    be reused indefinitely.
    """
    raw_refresh = request_refresh_token_from_cookie(request)
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = lookup_refresh_token(raw_refresh, db)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_raw, _new_row, _old = rotate_refresh_token(token, db)

    user = db.query(User).filter(User.id == token.user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token owner not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_access = create_access_token(data={"sub": user.username})
    set_refresh_cookie(response, new_raw)

    logger.info("Refresh token rotated for user id=%s", token.user_id)

    return RefreshTokenResponse(access_token=new_access)


# ------------------------------------------------------------------ cookie helpers

def _get_request_from_fastapi_context() -> Request | None:
    """Return the active FastAPI/Starlette request if one is in context.

    This is used by the refresh/logout cookie helpers so they can work with
    TestClient cookies and real browser cookies without each endpoint having to
    thread the request object through helper functions.
    """
    from starlette.requests import Request as StarletteRequest

    try:
        candidate = StarletteRequest._current_request  # type: ignore[attr-defined]
    except Exception:
        candidate = None

    if candidate is None:
        return None

    if not isinstance(candidate, StarletteRequest):
        return None

    return candidate


# Optional router exposed for deployments that want refresh token lifecycle
# events emitted through a separate router path instead of being co-located on
# the main auth router. Currently unused in this project; kept so the import in
# app.main stays meaningful if that feature is enabled later.
refresh_cookie_router = APIRouter(prefix="/api/auth/cookies", tags=["auth-cookies"])


def request_refresh_token_from_cookie(request: Request | None = None) -> str | None:
    """Return the raw refresh token cookie from the current request, if present.

    Accepts an optional explicit request so callers that already have one (for
    example /login) can pass it directly; otherwise it falls back to the
    context-aware helper above.
    """
    if request is not None:
        return request.cookies.get(settings.REFRESH_COOKIE_NAME)

    active = _get_request_from_fastapi_context()
    if active is None:
        return None

    return active.cookies.get(settings.REFRESH_COOKIE_NAME)


def set_refresh_cookie(response: Response, raw_token: str) -> None:
    """Set the HttpOnly refresh cookie with environment-configured security attrs."""
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=raw_token,
        max_age=int(settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400),
        httponly=True,
        secure=settings.REFRESH_COOKIE_SECURE,
        samesite=settings.REFRESH_COOKIE_SAME_SITE,
        domain=settings.REFRESH_COOKIE_DOMAIN or None,
        path="/",
    )


def clear_refresh_cookie(response: Response) -> None:
    """Remove the refresh cookie from the browser."""
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path="/",
    )

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """Request schema for user registration."""

    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, max_length=128, description="Plain-text password")


class UserLogin(BaseModel):
    """Request schema for user login."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Plain-text password")


class TokenResponse(BaseModel):
    """Response schema for successful authentication."""

    access_token: str
    token_type: str = "bearer"


class RefreshTokenResponse(BaseModel):
    """Minimal response returned after login/refresh to keep the cookie set."""

    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    """Response schema for user data (never includes password)."""

    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

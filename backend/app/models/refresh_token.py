from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class RefreshToken(Base):
    """Server-tracked refresh tokens for rotation + revocation.

    Refresh tokens are opaque random values stored in the database and
    delivered to the browser as an HttpOnly cookie.  Rotation is implemented
    by issuing a new refresh token on each successful refresh and revoking the
    previous one immediately.  Revocation (logout / detected reuse) prevents
    stale or stolen tokens from being used again.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    token_hash: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The issued-at timestamp drives expiry calculation.
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # When the refresh token itself expires.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    # Revoked tokens are rejected even if not technically expired yet.
    revoked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    # During rotation, the parent of the newly issued token.  Used to detect
    # reuse of a previously rotated (and therefore revoked) token.
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (
        Index("ix_refresh_tokens_user_id_valid", "user_id", "revoked", "expires_at"),
    )

    def is_valid(self, now: datetime | None = None) -> bool:
        """Return True only if the token is unexpired and not revoked."""
        now = now or datetime.now(timezone.utc)
        if self.revoked:
            return False
        if now >= self.expires_at:
            return False
        return True


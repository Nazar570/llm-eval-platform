from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from training.config import settings

logger = logging.getLogger(__name__)


def _build_engine() -> AsyncEngine:
    return create_async_engine(
        settings.async_uri,
        echo=settings.eval_database_echo,
        pool_pre_ping=True,
    )


engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine, expire_on_commit=False, autoflush=False
)


async def dispose_engine() -> None:
    await engine.dispose()
    logger.info("training engine disposed")

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_sync_url() -> str:
    user = os.getenv("EVAL_DATABASE_USER", "eval")
    password = os.getenv("EVAL_DATABASE_PASSWORD", "eval")
    host = os.getenv("EVAL_DATABASE_HOST", "localhost")
    port = os.getenv("EVAL_DATABASE_PORT", "5433")
    name = os.getenv("EVAL_DATABASE_NAME", "evaldb")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = _resolve_sync_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

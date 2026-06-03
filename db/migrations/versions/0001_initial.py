"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-02 00:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "traces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("contexts", sa.JSON(), nullable=False),
        sa.Column("ground_truth", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_traces_trace_id", "traces", ["trace_id"], unique=True)
    op.create_index("ix_traces_created_at", "traces", ["created_at"], unique=False)
    op.create_index("ix_traces_model_created", "traces", ["model_name", "created_at"], unique=False)

    op.create_table(
        "evaluations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("trace_pk", sa.Integer(), nullable=False),
        sa.Column("scorer", sa.String(length=64), nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["trace_pk"], ["traces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_evaluations_trace_pk", "evaluations", ["trace_pk"], unique=False)
    op.create_index("ix_evaluations_scorer", "evaluations", ["scorer"], unique=False)
    op.create_index("ix_evaluations_metric", "evaluations", ["metric"], unique=False)
    op.create_index("ix_evaluations_scored_at", "evaluations", ["scored_at"], unique=False)
    op.create_index("ix_evaluations_metric_scored", "evaluations", ["metric", "scored_at"], unique=False)
    op.create_index("ix_evaluations_scorer_metric", "evaluations", ["scorer", "metric"], unique=False)

    op.create_table(
        "drift_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("metric", sa.String(length=64), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("stattest", sa.String(length=32), nullable=False),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("reference_window_size", sa.Integer(), nullable=False),
        sa.Column("current_window_size", sa.Integer(), nullable=False),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_events_metric", "drift_events", ["metric"], unique=False)
    op.create_index("ix_drift_events_detected_at", "drift_events", ["detected_at"], unique=False)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("registered_model_name", sa.String(length=128), nullable=False),
        sa.Column("mlflow_version", sa.String(length=32), nullable=False),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=False),
        sa.Column("base_model", sa.String(length=128), nullable=False),
        sa.Column("eval_accuracy", sa.Float(), nullable=True),
        sa.Column("eval_f1", sa.Float(), nullable=True),
        sa.Column("training_rows", sa.Integer(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_versions_registered_model_name",
        "model_versions",
        ["registered_model_name"],
        unique=False,
    )
    op.create_index(
        "ix_model_versions_registered_at",
        "model_versions",
        ["registered_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_model_versions_registered_at", table_name="model_versions")
    op.drop_index("ix_model_versions_registered_model_name", table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_drift_events_detected_at", table_name="drift_events")
    op.drop_index("ix_drift_events_metric", table_name="drift_events")
    op.drop_table("drift_events")

    op.drop_index("ix_evaluations_scorer_metric", table_name="evaluations")
    op.drop_index("ix_evaluations_metric_scored", table_name="evaluations")
    op.drop_index("ix_evaluations_scored_at", table_name="evaluations")
    op.drop_index("ix_evaluations_metric", table_name="evaluations")
    op.drop_index("ix_evaluations_scorer", table_name="evaluations")
    op.drop_index("ix_evaluations_trace_pk", table_name="evaluations")
    op.drop_table("evaluations")

    op.drop_index("ix_traces_model_created", table_name="traces")
    op.drop_index("ix_traces_created_at", table_name="traces")
    op.drop_index("ix_traces_trace_id", table_name="traces")
    op.drop_table("traces")

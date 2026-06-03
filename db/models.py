from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    contexts: Mapped[list[str]] = mapped_column(JSON, default=list)
    ground_truth: Mapped[str | None] = mapped_column(Text, nullable=True)
    model_name: Mapped[str] = mapped_column(String(128))
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="trace", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_traces_model_created", "model_name", "created_at"),)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trace_pk: Mapped[int] = mapped_column(Integer, ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    scorer: Mapped[str] = mapped_column(String(64), index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

    trace: Mapped[Trace] = relationship(back_populates="evaluations")

    __table_args__ = (
        Index("ix_evaluations_metric_scored", "metric", "scored_at"),
        Index("ix_evaluations_scorer_metric", "scorer", "metric"),
    )


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    drift_score: Mapped[float] = mapped_column(Float)
    stattest: Mapped[str] = mapped_column(String(32))
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    reference_window_size: Mapped[int] = mapped_column(Integer)
    current_window_size: Mapped[int] = mapped_column(Integer)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    registered_model_name: Mapped[str] = mapped_column(String(128), index=True)
    mlflow_version: Mapped[str] = mapped_column(String(32))
    mlflow_run_id: Mapped[str] = mapped_column(String(64))
    base_model: Mapped[str] = mapped_column(String(128))
    eval_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    eval_f1: Mapped[float | None] = mapped_column(Float, nullable=True)
    training_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    stage: Mapped[str] = mapped_column(String(32), default="None")
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

"""
SQLAlchemy metadata models.

This is the durable audit trail: rooms, membership transitions, round
outcomes, and checkpoint records. The live/hot round state machine lives in
RoomManager (in-memory, per the assignment's emphasis on correctness and
testability); this layer is what survives a coordinator restart and what an
operator queries for "what happened".
"""
from __future__ import annotations

import time

from sqlalchemy import Column, Float, Integer, JSON, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RoomORM(Base):
    __tablename__ = "rooms"

    room_id = Column(String, primary_key=True)
    model_contract = Column(JSON, nullable=False)
    preprocessing_contract = Column(String, nullable=False)
    agg_config = Column(JSON, nullable=False)
    target_rounds = Column(Integer, nullable=False, default=1)
    state = Column(String, nullable=False, default="created")
    created_at = Column(Float, default=time.time)


class ClientEventORM(Base):
    __tablename__ = "client_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(String, index=True, nullable=False)
    client_id = Column(String, index=True, nullable=False)
    event = Column(String, nullable=False)  # joined / left / selected / completed / failed / dropped
    detail = Column(JSON, nullable=True)
    ts = Column(Float, default=time.time)


class RoundEventORM(Base):
    __tablename__ = "round_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(String, index=True, nullable=False)
    round_number = Column(Integer, nullable=False)
    outcome = Column(String, nullable=False)  # aggregated / failed_quorum
    summary = Column(JSON, nullable=True)
    ts = Column(Float, default=time.time)


class CheckpointORM(Base):
    __tablename__ = "checkpoints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(String, index=True, nullable=False)
    version = Column(Integer, nullable=False)
    uri = Column(String, nullable=False)
    n_clients = Column(Integer, nullable=False)
    created_at = Column(Float, default=time.time)


class AuditLogORM(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(String, index=True, nullable=False)
    event = Column(String, nullable=False)
    detail = Column(JSON, nullable=True)
    ts = Column(Float, default=time.time)
